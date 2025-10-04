"""
Terminal-based evaluation for real Franka robot that uses policy inference and avoids OpenCV display issues.

Usage:
python eval_real_franka_terminal.py -i <ckpt_path> -o <save_dir> --robot_ip <ip_of_franka>

================ Human in control ==============
Commands (type and press Enter):
- c: Start evaluation (hand control over to policy)
- s: Stop evaluation and gain control back
- q: Exit program
- status: Show current status
- help: Show this help

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
- s: Stop evaluation and gain control back
- q: Exit program
"""

import os
import time
import threading
import queue
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
import skvideo.io
import transforms3d
import open3d as o3d
from omegaconf import OmegaConf
from omegaconf import open_dict
import scipy.spatial.transform as st
import diffusers
from d3fields.utils.draw_utils import np2o3d
# from gendp.real_world.real_env_franka_gripper import RealEnvFranka, CAMERA_NAMES
from gendp.real_world.real_env_franka_gripper_gelsight import RealEnvFranka, CAMERA_NAMES, GELSIGHT_NAMES
from gendp.common.precise_sleep import precise_wait
from gendp.real_world.real_inference_util import (
    get_real_obs_resolution, 
    get_real_obs_dict)
from gendp.common.pytorch_util import dict_apply
from gendp.common.kinematics_utils import KinHelper
from gendp.workspace.base_workspace import BaseWorkspace
from gendp.policy.base_image_policy import BaseImagePolicy
from gendp.common.cv2_util import get_image_transform
from gendp.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from d3fields.fusion import Fusion

# hacks to be compatiable with old models
import gendp
import sys
sys.modules["diffusion_policy"] = gendp

# Global variables for communication between threads
command_queue = queue.Queue()
robot_state = {
    'running': False,
    'policy_active': False,
    'episode_id': 0,
    'stage': 0,
    'gripper_pos': 0.08,
    'stop': False,
    'iteration': 0
}

def obs_dict_np_to_o3d(obs_dict_np):
    from matplotlib import cm
    cmap = cm.get_cmap('viridis')
    if 'd3fields' in obs_dict_np:
        d3fields = obs_dict_np['d3fields'][0, :3].transpose()
    else:
        d3fields = np.zeros((1,3))

    if ('d3fields' in obs_dict_np) and (obs_dict_np['d3fields'].shape[1] > 3):
        d3fields_heatmap = obs_dict_np['d3fields'][0, 3].transpose() # (N, 1)
        d3fields_heatmap_color = cmap(d3fields_heatmap)[...,:3] # (N, 3)
        d3fields_o3d = np2o3d(d3fields, d3fields_heatmap_color)
    else:
        d3fields_o3d = np2o3d(d3fields)
    return d3fields_o3d

def policy_action_to_env_action(policy_action, action_mode, num_bots):
    # policy_action: (T, Da), Da=10 * num_bots (3 dof translation, 6 dof rotation, 1 gripper)
    if action_mode == 'eef':
        T = policy_action.shape[0]
        action_reshape = policy_action.reshape((T * num_bots, 10))
        env_actions = np.zeros((T * num_bots, 7), dtype=np.float64)
        env_actions[:,:3] = action_reshape[:,:3]
        from pytorch3d.transforms import rotation_conversions as pt
        action_rot_mat = pt.rotation_6d_to_matrix(torch.from_numpy(action_reshape[:,3:9])).numpy()
        env_actions[:, 3:6] = st.Rotation.from_matrix(action_rot_mat).as_euler('xyz')
        # env_actions[:, 6:] = action_reshape[:,9:]
        env_actions[:, 6:] = 0
        env_actions = env_actions.reshape((T, num_bots * 7))
    elif action_mode == 'joint':
        env_actions = policy_action
    return env_actions

def terminal_input_thread():
    """Handle terminal input in separate thread"""
    print("\n" + "="*60)
    print("FRANKA ROBOT EVALUATION - TERMINAL CONTROL")
    print("="*60)
    print("Commands:")
    print("  c       - Start evaluation (policy takes control)")
    print("  s       - Stop evaluation (human takes control)")
    print("  q       - Exit program")
    print("  g       - Close gripper")
    print("  o       - Open gripper")
    print("  status  - Show current status")
    print("  help    - Show commands")
    print("="*60)
    print("Type commands and press Enter...")
    
    while not robot_state['stop']:
        try:
            cmd = input().strip().lower()
            if cmd:
                command_queue.put(cmd)
                if cmd == 'q':
                    break
        except (EOFError, KeyboardInterrupt):
            command_queue.put('q')
            break

def process_commands(key_counter):
    """Process commands from terminal input and spacemouse"""
    global robot_state
    
    # Process terminal commands
    while not command_queue.empty():
        try:
            command = command_queue.get_nowait()
            
            if command == 'q':
                robot_state['stop'] = True
                print('🔴 Quitting...')
            elif command == 'c':
                if not robot_state['policy_active']:
                    robot_state['policy_active'] = True
                    print('🤖 Policy taking control! (Type "s" to stop)')
                else:
                    print('⚠️  Policy already active')
            elif command == 's':
                if robot_state['policy_active']:
                    robot_state['policy_active'] = False
                    print('👤 Human taking control back')
                else:
                    print('⚠️  Policy not active')
            elif command == 'g':
                robot_state['gripper_pos'] = 0.0
                print('✊ Closing gripper...')
            elif command == 'o':
                robot_state['gripper_pos'] = 0.08
                print('✋ Opening gripper...')
            elif command == 'status':
                status = f"Episode: {robot_state['episode_id']}, Stage: {robot_state['stage']}, Iter: {robot_state['iteration']}"
                status += f", Policy Active: {'YES' if robot_state['policy_active'] else 'NO'}"
                status += f", Gripper: {'CLOSED' if robot_state['gripper_pos'] < 0.05 else 'OPEN'}"
                print(f"📊 Status: {status}")
            elif command == 'help':
                print("\nCommands: c(start policy) s(stop policy) q(quit) g(grip) o(open) status help")
            else:
                print(f"❓ Unknown command: {command}. Type 'help' for commands.")
                
        except queue.Empty:
            break
    
    # Process SpaceMouse/KeystrokeCounter commands
    press_events = key_counter.get_press_events()
    for key_stroke in press_events:
        if key_stroke == KeyCode(char='q'):
            robot_state['stop'] = True
            print('🔴 Quitting...')
        elif key_stroke == KeyCode(char='c'):
            if not robot_state['policy_active']:
                robot_state['policy_active'] = True
                print('🤖 Policy taking control! (Type "s" to stop)')
            else:
                print('⚠️  Policy already active')
        elif key_stroke == KeyCode(char='s'):
            if robot_state['policy_active']:
                robot_state['policy_active'] = False
                print('👤 Human taking control back')
            else:
                print('⚠️  Policy not active')
        elif key_stroke == KeyCode(char='g'):
            robot_state['gripper_pos'] = 0.0
            print('✊ Closing gripper...')
        elif key_stroke == KeyCode(char='o'):
            robot_state['gripper_pos'] = 0.08
            print('✋ Opening gripper...')

def save_visualization_images(vis_img, output_dir, iter_idx, save_interval=30):
    """Save visualization images periodically"""
    if iter_idx % save_interval == 0:
        viz_dir = os.path.join(output_dir, 'visualization')
        os.makedirs(viz_dir, exist_ok=True)
        latest_file_name = os.path.join(viz_dir, 'latest.jpg')
        cv2.imwrite(latest_file_name, vis_img)

OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('--input_dir', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_ip', '-ri ', default="192.168.1.143", help="Franka's IP address ")
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--vis_d3fields', default=False, type=bool, help="Visualize d3fields.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=60, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving command to executing on Robot in Sec.")
@click.option('--n_action_steps', '-n', default=-1, type=int, help="Number of action steps to execute. -1 means invalid.")
@click.option('--init_joints', '-j', is_flag=True, default=True, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--save_viz_interval', default=30, type=int, help="Save visualization every N frames (0 to disable)")
def main(input_dir, output, robot_ip, match_dataset, match_episode,
    vis_camera_idx, vis_d3fields,
    steps_per_inference, max_duration,
    frequency, command_latency, n_action_steps, init_joints, save_viz_interval):
    
    # load match_dataset
    match_camera_idx = 0
    episode_first_frame_map = dict()
    os.system(f'mkdir -p {output}/d3fields_vis')
    
    if match_dataset is not None:
        match_dir = pathlib.Path(match_dataset)
        match_video_dir = match_dir.joinpath('videos')
        for vid_dir in match_video_dir.glob("*/"):
            episode_idx = int(vid_dir.stem)
            match_video_path = vid_dir.joinpath(f'{match_camera_idx}.mp4')
            if match_video_path.exists():
                frames = skvideo.io.vread(
                    str(match_video_path), num_frames=1)
                episode_first_frame_map[episode_idx] = frames[0]
    print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")
    
    # load checkpoint
    ckpt_path = input_dir
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    num_bots = 1

    # hacks for method-specific setup.
    action_offset = 0
    delta_action = False
    if 'diffusion' in cfg.name:
        # diffusion model
        policy: BaseImagePolicy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device('cuda')
        policy.eval().to(device)

        # set inference params
        if n_action_steps > 0:
            policy.n_action_steps = n_action_steps
        policy.num_inference_steps = 16 # DDIM inference iterations
        noise_scheduler = diffusers.schedulers.scheduling_ddim.DDIMScheduler(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='epsilon'
        )
        policy.noise_scheduler = noise_scheduler
        if 'd3fields' in cfg.task.shape_meta['obs']:
            with open_dict(cfg.task.shape_meta['obs']):
                cfg.task.shape_meta['obs']['d3fields']['info']['exclude_colors'] = ['yellow']
        if 'key' not in cfg.task.shape_meta['action'] or cfg.task.shape_meta['action']['key'] == 'eef_action':
            action_mode = 'eef'
        elif cfg.task.shape_meta['action']['key'] == 'joint_action':
            action_mode = 'joint'
        else:
            raise RuntimeError("Unsupported action mode: ", cfg.task.shape_meta['action']['key'])
    else:
        raise RuntimeError("Unsupported policy type: ", cfg.name)

    # setup experiment
    dt = 1/frequency
    os.system(f'mkdir -p {output}')
    kin_helper = KinHelper(robot_name=cfg.task.dataset.robot_name)
    fusion = None
    expected_labels = None
    for obs_key in cfg.task.shape_meta['obs'].keys():
        if 'd3fields' in obs_key:
            num_cam = len(cfg.task.shape_meta['obs'][obs_key]['info']['view_keys'])
            fusion = Fusion(num_cam=num_cam, dtype=torch.float16)
            expected_labels = cfg.task.expected_labels if 'expected_labels' in cfg.task else None
            break

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)
    print("action_offset:", action_offset)

    # Start terminal input thread
    input_thread = threading.Thread(target=terminal_input_thread, daemon=True)
    input_thread.start()

    try:
        with SharedMemoryManager() as shm_manager:
            with KeystrokeCounter() as key_counter, \
                RealEnvFranka(
                output_dir=output, 
                robot_ip=robot_ip, 
                frequency=frequency,
                n_obs_steps=n_obs_steps,
                obs_float32=False,
                init_joints=init_joints,
                ctrl_mode=action_mode,
                enable_multi_cam_vis=True,
                record_raw_video=True,
                video_capture_fps=30,
                thread_per_video=3,
                video_crf=21,
                shm_manager=shm_manager) as env:

                print("Waiting for realsense")
                time.sleep(1.0)

                print("Warming up policy inference")
                obs = env.get_obs()
                with torch.no_grad():
                    policy.reset()
                    exclude_colors = cfg.task.dataset.exclude_colors if 'exclude_colors' in cfg.task.dataset else []
                    obs_dict_np = get_real_obs_dict(
                        env_obs=obs, shape_meta=cfg.task.shape_meta, 
                        fusion=fusion, expected_labels=expected_labels, teleop=kin_helper, exclude_colors=exclude_colors)

                    obs_dict = dict_apply(obs_dict_np, 
                        lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                    result = policy.predict_action(obs_dict)
                    action = result['action'][0].detach().to('cpu').numpy()
                    del result

                print('🤖 Robot ready! Type commands in terminal...')
                robot_state['running'] = True
                
                # Main control loop
                while not robot_state['stop']:
                    # ========= human control loop ==========
                    if not robot_state['policy_active']:
                        print("👤 Human in control!")
                        state = env.get_robot_state()
                        t_start = time.monotonic()
                        iter_idx = 0
                        last_status_time = time.time()
                        
                        while not robot_state['stop'] and not robot_state['policy_active']:
                            # calculate timing
                            t_cycle_end = t_start + (iter_idx + 1) * dt
                            t_sample = t_cycle_end - command_latency
                            t_command_target = t_cycle_end + dt

                            # pump obs
                            obs = env.get_obs()
                            
                            # Process commands
                            process_commands(key_counter)
                            
                            # Update state
                            robot_state['stage'] = key_counter[Key.space]
                            robot_state['episode_id'] = env.episode_id

                            # visualize (save only, no display)
                            episode_id = robot_state['episode_id']
                            stage = robot_state['stage']
                            robot_state['iteration'] = iter_idx
                            
                            # Create visualization similar to demo_real_franka_terminal
                            if vis_camera_idx == 0:
                                vis_img = obs[f'camera_{CAMERA_NAMES[vis_camera_idx]}_color'][-1]
                            else:
                                # Use front and right cameras like in demo
                                rs_front = obs['camera_front_color'][-1,:,:,::-1].copy() if 'camera_front_color' in obs else obs[f'camera_{CAMERA_NAMES[0]}_color'][-1]
                                rs_right = obs['camera_right_color'][-1,:,:,::-1].copy() if 'camera_right_color' in obs else obs[f'camera_{CAMERA_NAMES[1]}_color'][-1]
                                
                                # Concatenate images
                                vis_img = np.concatenate([rs_front, rs_right], axis=1)
                                vis_img = cv2.resize(vis_img, (960, 360))
                            
                            # Match dataset overlay if provided
                            match_episode_id = episode_id
                            if match_episode is not None:
                                match_episode_id = match_episode
                            if match_episode_id in episode_first_frame_map:
                                match_img = episode_first_frame_map[match_episode_id]
                                ih, iw, _ = match_img.shape
                                oh, ow, _ = vis_img.shape
                                tf = get_image_transform(
                                    input_res=(iw, ih), 
                                    output_res=(ow, oh), 
                                    bgr_to_rgb=False)
                                match_img = tf(match_img).astype(np.float32) / 255
                                vis_img = np.minimum(vis_img, match_img)

                            text = f'Episode: {episode_id}, Stage: {stage} [HUMAN]'
                            cv2.putText(
                                vis_img,
                                text,
                                (10, 30),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.8,
                                thickness=2,
                                color=(255, 255, 255)
                            )
                            
                            # Save visualization images periodically
                            if save_viz_interval > 0:
                                save_visualization_images(vis_img, output, iter_idx, save_viz_interval)
                            # Execute robot actions (maintain current position + gripper control)
                            if action_mode == 'joint':
                                joint_pos = obs['full_joint_pos']
                                actions = joint_pos[-1, :8].copy()
                                actions[-1] = robot_state['gripper_pos']
                                env.exec_actions(
                                    actions=[actions],
                                    timestamps=[t_command_target-time.monotonic()+time.time()],
                                    mode='joint')
                            elif action_mode == 'eef':
                                # For EEF mode, maintain current EEF position and gripper
                                curr_ee_pose = obs['ee_pose'][-1].copy()  # [x, y, z, rx, ry, rz, gripper]
                                curr_ee_pose[-1] = robot_state['gripper_pos']  # Update gripper position
                                env.exec_actions(
                                    actions=[curr_ee_pose],
                                    timestamps=[t_command_target-time.monotonic()+time.time()],
                                    mode='eef')

                            precise_wait(t_cycle_end)
                            iter_idx += 1
                            
                            # Print status periodically
                            current_time = time.time()
                            if current_time - last_status_time > 5.0:
                                status = f"📊 Human Control - Iter: {iter_idx}, Ep: {episode_id}, Stage: {stage}"
                                status += f", Gripper: {'CLOSED' if robot_state['gripper_pos'] < 0.05 else 'OPEN'}"
                                print(status)
                                last_status_time = current_time

                    # ========== policy control loop ==============
                    elif robot_state['policy_active']:
                        try:
                            # start episode
                            policy.reset()
                            start_delay = 1.0
                            eval_t_start = time.time() + start_delay
                            t_start = time.monotonic() + start_delay
                            env.start_episode(eval_t_start)
                            # wait for 1/30 sec to get the closest frame actually
                            frame_latency = 1/30
                            precise_wait(eval_t_start - frame_latency, time_func=time.time)
                            print("🤖 Policy started!")
                            iter_idx = 0
                            last_status_time = time.time()
                            
                            while robot_state['policy_active'] and not robot_state['stop']:
                                # calculate timing
                                t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                                # get obs
                                obs = env.get_obs()
                                obs_timestamps = obs['timestamp']

                                # Process commands (check for stop)
                                process_commands(key_counter)

                                # run inference
                                with torch.no_grad():
                                    s = time.time()
                                    obs_dict_np = get_real_obs_dict(
                                        env_obs=obs, shape_meta=cfg.task.shape_meta, 
                                        fusion=fusion, expected_labels=expected_labels, teleop=kin_helper, exclude_colors=exclude_colors)
                                    obs_dict = dict_apply(obs_dict_np, 
                                        lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                                    result = policy.predict_action(obs_dict)
                                    action = result['action'][0].detach().to('cpu').numpy()

                                env_actions = policy_action_to_env_action(action, action_mode, num_bots)

                                # deal with timing
                                action_timestamps = (np.arange(len(action), dtype=np.float64) + action_offset
                                    ) * dt + obs_timestamps[-1]
                                action_exec_latency = 0.2
                                curr_time = time.time()
                                is_new = action_timestamps > (curr_time + action_exec_latency)
                                if np.sum(is_new) == 0:
                                    # exceeded time budget, still do something
                                    env_actions = env_actions[[-1]]
                                    # schedule on next available step
                                    next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                                    action_timestamp = eval_t_start + (next_step_idx) * dt
                                    action_timestamps = np.array([action_timestamp])
                                else:
                                    env_actions = env_actions[is_new]
                                    action_timestamps = action_timestamps[is_new]
                                
                                # execute actions
                                if action_mode == 'eef':
                                    env.exec_actions(
                                        actions=env_actions,
                                        timestamps=action_timestamps,
                                        mode=action_mode)
                                elif action_mode == 'joint':
                                    env.exec_actions(
                                        actions=env_actions,
                                        timestamps=action_timestamps,
                                        mode=action_mode)

                                # visualize (save only)
                                episode_id = env.episode_id
                                robot_state['episode_id'] = episode_id
                                robot_state['iteration'] = iter_idx
                                
                                vis_camera_name = CAMERA_NAMES[vis_camera_idx]
                                vis_img = obs[f'camera_{vis_camera_name}_color'][-1]
                                text = 'Episode: {}, Time: {:.1f} [POLICY]'.format(
                                    episode_id, time.monotonic() - t_start
                                )
                                cv2.putText(
                                    vis_img,
                                    text,
                                    (10, 30),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.8,
                                    thickness=2,
                                    color=(0,255,0)
                                )
                                
                                # Save visualization images periodically
                                if save_viz_interval > 0:
                                    save_visualization_images(vis_img, output, iter_idx, save_viz_interval)

                                # wait for execution
                                precise_wait(t_cycle_end - frame_latency)
                                iter_idx += steps_per_inference

                                # Print status periodically
                                current_time = time.time()
                                if current_time - last_status_time > 5.0:
                                    print(f"📊 Policy Control - Iter: {iter_idx}, Episode: {episode_id}, Freq: {1/(time.perf_counter() - (current_time - 5.0)):.1f}Hz")
                                    last_status_time = current_time

                        except KeyboardInterrupt:
                            print("Policy interrupted!")
                            env.end_episode()
                            robot_state['policy_active'] = False
                        
                        if robot_state['policy_active']:
                            env.end_episode()
                            robot_state['policy_active'] = False
                            print("Policy episode ended")

    except KeyboardInterrupt:
        print("\n🔴 Interrupted by user")
        robot_state['stop'] = True
    except Exception as e:
        print(f"❌ Error: {e}")
        robot_state['stop'] = True
        raise
    finally:
        robot_state['running'] = False
        print("🏁 Robot evaluation ended")

if __name__ == '__main__':
    main()
