"""
Record robot end-effector keypoints interactively for trajectory generation.

This script allows you to record a series of EE poses for a 2-stage trajectory:
- Stage 0: Approach (multiple keypoints) - Move to grasp pose with gripper open
- Stage 1: Move (multiple keypoints) - Move item to target with gripper closed

Note: Grasping happens automatically between Stage 0 and Stage 1 (gripper closes at end of Stage 0)

Usage:
python record_ee_keypoints.py -o <keypoints_save_dir> --robot_ip <ip_of_franka>

Commands (type and press Enter):
- r: Record current EE pose as keypoint
- n: Move to next stage
- p: Preview all recorded keypoints
- s: Save all keypoints to file
- d: Delete last recorded keypoint
- q: Exit program (with save prompt)
- status: Show current status
- help: Show this help
"""

import os
import time
import threading
import queue
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import json
from pathlib import Path

from gendp.real_world.real_env_franka_gripper_gelsight import RealEnvFranka, CAMERA_NAMES
from gendp.real_world.keystroke_counter import KeystrokeCounter, Key, KeyCode

# Global variables for communication between threads
command_queue = queue.Queue()
keypoint_state = {
    'stage': 0,
    'stage_names': ['Approach', 'Move'],
    'keypoints': {0: [], 1: []},  # Stage -> list of (ee_pose, gripper_pos, timestamp)
    'gripper_pos': 0.08,
    'stop': False
}

def terminal_input_thread():
    """Handle terminal input in separate thread"""
    print("\n" + "="*60)
    print("ROBOT EE KEYPOINT RECORDER")
    print("="*60)
    print("Record keypoints for 2-stage trajectory:")
    print("  Stage 0: Approach (gripper open)")
    print("  Stage 1: Move (gripper closed)")
    print("Note: Gripper closes automatically at end of Stage 0")
    print("="*60)
    print("Commands:")
    print("  r       - Record current EE pose")
    print("  n       - Next stage")
    print("  p       - Preview keypoints")
    print("  s       - Save keypoints")
    print("  d       - Delete last keypoint")
    print("  g       - Close gripper")
    print("  o       - Open gripper")
    print("  q       - Exit")
    print("  status  - Show status")
    print("  help    - Show commands")
    print("="*60)
    print("Type commands and press Enter...")
    
    while not keypoint_state['stop']:
        try:
            cmd = input().strip().lower()
            if cmd:
                command_queue.put(cmd)
                if cmd == 'q':
                    break
        except (EOFError, KeyboardInterrupt):
            command_queue.put('q')
            break

def format_pose(pose):
    """Format pose array for display"""
    if len(pose) == 6:
        return f"pos:[{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}] rot:[{pose[3]:.3f}, {pose[4]:.3f}, {pose[5]:.3f}]"
    elif len(pose) == 7:
        return f"pos:[{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}] rot:[{pose[3]:.3f}, {pose[4]:.3f}, {pose[5]:.3f}] grip:{pose[6]:.3f}"
    else:
        return str(pose)

def preview_keypoints():
    """Display all recorded keypoints"""
    print("\n" + "="*60)
    print("RECORDED KEYPOINTS:")
    print("="*60)
    total_keypoints = 0
    for stage_id in [0, 1]:
        stage_name = keypoint_state['stage_names'][stage_id]
        keypoints = keypoint_state['keypoints'][stage_id]
        print(f"\nStage {stage_id} ({stage_name}): {len(keypoints)} keypoints")
        for i, (ee_pose, gripper_pos, timestamp) in enumerate(keypoints):
            print(f"  [{i}] {format_pose(ee_pose)} gripper:{gripper_pos:.3f}")
            total_keypoints += 1
    print("="*60)
    print(f"Total keypoints: {total_keypoints}")
    print(f"Note: Gripper will close automatically after Stage 0")
    print("="*60 + "\n")

def save_keypoints(output_dir):
    """Save keypoints to JSON file"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"keypoints_{timestamp}.json"
    
    # Prepare data for saving
    save_data = {
        'metadata': {
            'created_at': timestamp,
            'total_stages': 2,
            'stage_names': keypoint_state['stage_names']
        },
        'keypoints': {}
    }
    
    for stage_id in [0, 1]:
        stage_keypoints = []
        for ee_pose, gripper_pos, kp_timestamp in keypoint_state['keypoints'][stage_id]:
            stage_keypoints.append({
                'ee_pose': ee_pose.tolist(),
                'gripper_pos': float(gripper_pos),
                'timestamp': float(kp_timestamp)
            })
        save_data['keypoints'][f'stage_{stage_id}'] = stage_keypoints
    
    # Save to JSON
    with open(filename, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\n✅ Keypoints saved to: {filename}")
    return filename

def process_commands(env):
    """Process commands from terminal input"""
    global keypoint_state
    
    while not command_queue.empty():
        try:
            command = command_queue.get_nowait()
            
            if command == 'q':
                # Prompt to save before exit
                total_kp = sum(len(keypoint_state['keypoints'][i]) for i in [0, 1])
                if total_kp > 0:
                    print(f"\n⚠️  You have {total_kp} recorded keypoints. Save before exit? (y/n): ", end='', flush=True)
                    # Note: This will be handled in the main loop
                keypoint_state['stop'] = True
                print('🔴 Exiting...')
                
            elif command == 'r':
                # Record current EE pose
                obs = env.get_obs()
                ee_pose = obs['ee_pose'][-1]  # Get latest EE pose [x, y, z, rx, ry, rz]
                gripper_pos = obs['full_joint_pos'][-1, -1]  # Get gripper position
                timestamp = time.time()
                
                current_stage = keypoint_state['stage']
                keypoint_state['keypoints'][current_stage].append((ee_pose.copy(), gripper_pos, timestamp))
                
                stage_name = keypoint_state['stage_names'][current_stage]
                num_keypoints = len(keypoint_state['keypoints'][current_stage])
                print(f"📍 Recorded keypoint #{num_keypoints} for Stage {current_stage} ({stage_name})")
                print(f"   {format_pose(ee_pose)} gripper:{gripper_pos:.3f}")
                
            elif command == 'n':
                # Move to next stage
                current_stage = keypoint_state['stage']
                num_keypoints = len(keypoint_state['keypoints'][current_stage])
                
                if num_keypoints == 0:
                    print(f"⚠️  No keypoints recorded for Stage {current_stage}. Record at least one before moving to next stage.")
                elif current_stage < 1:
                    keypoint_state['stage'] += 1
                    new_stage = keypoint_state['stage']
                    new_stage_name = keypoint_state['stage_names'][new_stage]
                    print(f"⏭️  Moved to Stage {new_stage}: {new_stage_name}")
                    print(f"💡 Remember: Gripper will close automatically after Stage 0")
                else:
                    print("ℹ️  Already at the last stage (Stage 1: Move)")
                    
            elif command == 'p':
                # Preview all keypoints
                preview_keypoints()
                
            elif command == 's':
                # Save keypoints
                total_kp = sum(len(keypoint_state['keypoints'][i]) for i in [0, 1])
                if total_kp == 0:
                    print("⚠️  No keypoints to save!")
                else:
                    save_keypoints(env.output_dir)
                    preview_keypoints()
                    
            elif command == 'd':
                # Delete last keypoint
                current_stage = keypoint_state['stage']
                if len(keypoint_state['keypoints'][current_stage]) > 0:
                    removed = keypoint_state['keypoints'][current_stage].pop()
                    stage_name = keypoint_state['stage_names'][current_stage]
                    print(f"🗑️  Deleted last keypoint from Stage {current_stage} ({stage_name})")
                else:
                    print(f"ℹ️  No keypoints to delete in current stage")
                    
            elif command == 'g':
                keypoint_state['gripper_pos'] = 0.0
                print('✊ Closing gripper...')
                
            elif command == 'o':
                keypoint_state['gripper_pos'] = 0.08
                print('✋ Opening gripper...')
                    
            elif command == 'status':
                current_stage = keypoint_state['stage']
                stage_name = keypoint_state['stage_names'][current_stage]
                num_keypoints = len(keypoint_state['keypoints'][current_stage])
                total_kp = sum(len(keypoint_state['keypoints'][i]) for i in [0, 1])
                print(f"📊 Status: Stage {current_stage} ({stage_name}), {num_keypoints} keypoints in current stage, {total_kp} total")
                
            elif command == 'help':
                print("\nCommands: r(record) n(next stage) p(preview) s(save) d(delete) g(grip) o(open) q(quit) status help")
                
            else:
                print(f"❓ Unknown command: {command}. Type 'help' for commands.")
                
        except queue.Empty:
            break

def save_visualization_images(vis_img, output_dir, iter_idx, save_interval=30):
    """Save visualization images periodically"""
    if iter_idx % save_interval == 0:
        viz_dir = os.path.join(output_dir, 'visualization')
        os.makedirs(viz_dir, exist_ok=True)
        latest_file_name = os.path.join(viz_dir, 'latest.jpg')
        cv2.imwrite(latest_file_name, vis_img)

@click.command()
@click.option('--output_dir', '-o', required=True, help='Directory to save keypoints')
@click.option('--robot_ip', '-ri', default="192.168.1.143", help="Franka's IP address")
@click.option('--init_joints', '-j', is_flag=True, default=True, help="Whether to initialize robot joint configuration")
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving command to executing on Robot in Sec")
@click.option('--save_viz_interval', default=30, type=int, help="Save visualization every N frames")
def main(output_dir, robot_ip, init_joints, vis_camera_idx, frequency, command_latency, save_viz_interval):
    dt = 1/frequency
    os.makedirs(output_dir, exist_ok=True)
    
    # Start terminal input thread
    input_thread = threading.Thread(target=terminal_input_thread, daemon=True)
    input_thread.start()
    
    try:
        with SharedMemoryManager() as shm_manager:
            with KeystrokeCounter() as key_counter, \
                RealEnvFranka(
                    output_dir=output_dir,
                    robot_ip=robot_ip,
                    frequency=frequency,
                    n_obs_steps=2,
                    obs_float32=False,
                    init_joints=init_joints,
                    enable_multi_cam_vis=True,
                    record_raw_video=False,
                    thread_per_video=3,
                    video_crf=21,
                    shm_manager=shm_manager) as env:
                
                print('🤖 Robot ready! Manually move the robot and record keypoints...')
                print(f"📁 Keypoints will be saved to: {output_dir}")
                
                time.sleep(1.0)
                t_start = time.monotonic()
                iter_idx = 0
                last_status_time = time.time()
                
                while not keypoint_state['stop']:
                    # Calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    
                    # Get observations
                    obs = env.get_obs()
                    
                    # Process commands
                    process_commands(env)
                    
                    # Create visualization
                    rs_front = obs['camera_front_color'][-1,:,:,::-1].copy()
                    rs_right = obs['camera_right_color'][-1,:,:,::-1].copy()
                    
                    # Concatenate images
                    vis_img = np.concatenate([rs_front, rs_right], axis=1)
                    vis_img = cv2.resize(vis_img, (960, 360))
                    
                    # Add status text
                    current_stage = keypoint_state['stage']
                    stage_name = keypoint_state['stage_names'][current_stage]
                    num_keypoints = len(keypoint_state['keypoints'][current_stage])
                    text = f'Stage {current_stage} ({stage_name}): {num_keypoints} keypoints'
                    
                    cv2.putText(
                        vis_img,
                        text,
                        (10, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8,
                        thickness=2,
                        color=(0, 255, 0)
                    )
                    
                    # Add EE pose info
                    ee_pose = obs['ee_pose'][-1]
                    gripper_pos = obs['full_joint_pos'][-1, -1]
                    pose_text = f'EE: [{ee_pose[0]:.3f}, {ee_pose[1]:.3f}, {ee_pose[2]:.3f}] G:{gripper_pos:.3f}'
                    cv2.putText(
                        vis_img,
                        pose_text,
                        (10, 60),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.6,
                        thickness=1,
                        color=(255, 255, 255)
                    )
                    
                    # Save visualization images periodically
                    # if save_viz_interval > 0:
                    #     save_visualization_images(vis_img, output_dir, iter_idx, save_viz_interval)
                    
                    # Execute gripper control
                    joint_pos = obs['full_joint_pos']
                    actions = joint_pos[-1, :8].copy()
                    actions[-1] = keypoint_state['gripper_pos']
                    env.exec_actions(
                        actions=[actions],
                        timestamps=[t_cycle_end-time.monotonic()+time.time()])
                    
                    # Wait for next cycle
                    time.sleep(dt)
                    iter_idx += 1
                    
                    # Print status periodically (every 5 seconds)
                    current_time = time.time()
                    if current_time - last_status_time > 5.0:
                        total_kp = sum(len(keypoint_state['keypoints'][i]) for i in [0, 1])
                        status = f"📊 Stage: {current_stage} ({stage_name}), Current: {num_keypoints}, Total: {total_kp}"
                        print(status)
                        last_status_time = current_time
                
                # Final save prompt
                total_kp = sum(len(keypoint_state['keypoints'][i]) for i in [0, 1])
                if total_kp > 0:
                    save_keypoints(output_dir)
                    
    except KeyboardInterrupt:
        print("\n🔴 Interrupted by user")
        keypoint_state['stop'] = True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        keypoint_state['stop'] = True
    finally:
        print("🏁 Keypoint recording ended")

if __name__ == '__main__':
    main()
