import os
from typing import Optional
import pathlib
import glob
import cv2
import numpy as np
import time
import shutil
import math
from multiprocessing.managers import SharedMemoryManager
# from diffusion_policy.real_world.rtde_interpolation_controller import RTDEInterpolationController
from gendp.real_world.franka_interpolation_controller import FrankaInterpolationController
from gendp.real_world.multi_realsense import MultiRealsense, SingleRealsense
from gendp.real_world.video_recorder import VideoRecorder
from gendp.common.timestamp_accumulator import (
    TimestampObsAccumulator,
    TimestampActionAccumulator,
    align_timestamps
)
from gendp.common.precise_sleep import precise_wait
from gendp.common.data_utils import load_dict_from_hdf5
from gendp.real_world.multi_camera_visualizer import MultiCameraVisualizer
from gendp.common.replay_buffer import ReplayBuffer
from gendp.common.cv2_util import (get_extrinsic,
    get_image_transform, optimal_row_cols)
from gendp.common.data_utils import save_dict_to_hdf5


DEFAULT_OBS_KEY_MAP = {
    # robot
    'ActualTCPPoseWGripper': 'ee_pos',
    'ActualQWGripper': 'joint_pos',
    'FullActualQWGripper': 'full_joint_pos',
    'ActualQdWGripper': 'joint_vel',
    'WristCamExtrinsics': 'wrist_cam_extrinsics',
    # 'ForceTorque': 'force_torque',
    #gripper
    # 'gripper_position': 'gripper_position',
    # timestamps
    'step_idx': 'step_idx',
    'timestamp': 'timestamp'
}

CAMERA_NAMES = {
    0: 'wrist',
    1: 'left',
    2: 'right'
}

class RealEnvFranka:
    def __init__(self,
            # required params
            output_dir,
            robot_ip='192.168.1.143',
            # env params
            frequency=10,
            n_obs_steps=2,
            # obs
            obs_image_resolution=(640, 480),
            max_obs_buffer_size=30,
            camera_serial_numbers=None,
            obs_key_map=DEFAULT_OBS_KEY_MAP,
            obs_float32=False,
            # this latency compensates receive_timestamp
            # all in seconds
            robot_obs_latency=0.0001,
            # action
            max_pos_speed=0.25,
            max_rot_speed=0.6,
            # robot
            tcp_offset=0.13,
            init_joints=False,
            ctrl_mode='joint',
            # video capture params
            video_capture_fps=30,
            video_capture_resolution=(640, 480),
            # saving params
            record_raw_video=True,
            thread_per_video=2,
            video_crf=21,
            # vis params
            enable_multi_cam_vis=True,
            multi_cam_vis_resolution=(640, 480),
            # shared memory
            shm_manager=None
            ):
        assert frequency <= video_capture_fps
        output_dir = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir()
        video_dir = output_dir.joinpath('videos')
        video_dir.mkdir(parents=True, exist_ok=True)
        self.episode_id = len(glob.glob(os.path.join(output_dir.absolute().as_posix(), '*.hdf5')))
        # zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())
        # replay_buffer = ReplayBuffer.create_from_path(
        #     zarr_path=zarr_path, mode='a')

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
        if camera_serial_numbers is None:
            camera_serial_numbers = SingleRealsense.get_connected_devices_serial()

        print(camera_serial_numbers)

        color_tf = get_image_transform(
            input_res=video_capture_resolution,
            output_res=obs_image_resolution,
            # obs output rgb
            bgr_to_rgb=True)
        color_transform = color_tf
        if obs_float32:
            color_transform = lambda x: color_tf(x).astype(np.float32) / 255

        def transform(data):
            data['color'] = color_transform(data['color'])
            data['depth'] = cv2.resize(data['depth'], obs_image_resolution, interpolation=cv2.INTER_NEAREST)
            return data

        rw, rh, col, row = optimal_row_cols(
            n_cameras=len(camera_serial_numbers),
            in_wh_ratio=obs_image_resolution[0] / obs_image_resolution[1],
            max_resolution=multi_cam_vis_resolution
        )
        vis_color_transform = get_image_transform(
            input_res=video_capture_resolution,
            output_res=(rw, rh),
            bgr_to_rgb=False
        )

        def vis_transform(data):
            data['color'] = vis_color_transform(data['color'])
            return data

        recording_transfrom = None
        recording_fps = video_capture_fps
        recording_pix_fmt = 'bgr24'
        if not record_raw_video:
            recording_transfrom = transform
            recording_fps = frequency
            recording_pix_fmt = 'rgb24'

        video_recorder = VideoRecorder.create_h264(
            fps=recording_fps,
            codec='h264',
            input_pix_fmt=recording_pix_fmt,
            crf=video_crf,
            thread_type='FRAME',
            thread_count=thread_per_video)

        realsense = MultiRealsense(
            serial_numbers=camera_serial_numbers,
            shm_manager=shm_manager,
            resolution=video_capture_resolution,
            capture_fps=video_capture_fps,
            put_fps=video_capture_fps,
            # send every frame immediately after arrival
            # ignores put_fps
            put_downsample=False,
            record_fps=recording_fps,
            enable_color=True,
            enable_depth=True,
            enable_infrared=False,
            get_max_k=max_obs_buffer_size,
            transform=transform,
            vis_transform=vis_transform,
            recording_transform=recording_transfrom,
            video_recorder=video_recorder,
            verbose=False
        )

        multi_cam_vis = None
        if enable_multi_cam_vis:
            multi_cam_vis = MultiCameraVisualizer(
                realsense=realsense,
                row=row,
                col=col,
                rgb_to_bgr=False
            )

        # cube_diag = np.linalg.norm([1, 1, 1])
        j_init = np.array([0.0702805, -0.90773028, -0.09513126, -2.67802477, -0.0919309, 1.82060218, 0.16051947])
        # j_init = np.array([-0.03173639, -0.24618988, -0.2356476 , -2.3970356 , -0.07736383, 2.19674683, -0.01091733])
        if not init_joints:
            j_init = None

        robot = FrankaInterpolationController(
            shm_manager=shm_manager,
            robot_ip=robot_ip,
            frequency=200,
            Kx_scale=1.0,
            Kxd_scale=np.array([2.0, 1.5, 2.0, 1.0, 1.0, 1.0]),
            joints_init=j_init,
            joints_init_duration=3.0,
            verbose=False,
            receive_latency=robot_obs_latency,
            ctrl_mode=ctrl_mode,
        )

        self.realsense = realsense
        self.robot = robot
        # self.gripper = gripper
        self.multi_cam_vis = multi_cam_vis
        self.video_capture_fps = video_capture_fps
        self.frequency = frequency
        self.n_obs_steps = n_obs_steps
        self.max_obs_buffer_size = max_obs_buffer_size
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.obs_key_map = obs_key_map
        self.ctrl_mode = ctrl_mode
        # recording
        self.output_dir = output_dir
        self.video_dir = video_dir
        # self.replay_buffer = replay_buffer
        # temp memory buffers
        self.last_realsense_data = None
        # recording buffers
        self.obs_accumulator = None
        self.action_accumulator = None
        self.stage_accumulator = None

        self.start_time = None

    # ======== start-stop API =============
    @property
    def is_ready(self):
        return self.realsense.is_ready and self.robot.is_ready

    def start(self, wait=True):
        self.realsense.start(wait=False)
        self.robot.start(wait=False)
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start(wait=False)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.end_episode()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop(wait=False)
        self.robot.stop(wait=False)
        self.realsense.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.realsense.start_wait()
        self.robot.start_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start_wait()

    def stop_wait(self):
        self.robot.stop_wait()
        self.realsense.stop_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop_wait()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= async env API ===========
    def get_obs(self) -> dict:
        "observation dict"
        assert self.is_ready

        # get data
        # 30 Hz, camera_receive_timestamp
        k = math.ceil(self.n_obs_steps * (self.video_capture_fps / self.frequency))
        self.last_realsense_data = self.realsense.get(
            k=k,
            out=self.last_realsense_data)

        # 125 hz, robot_receive_timestamp
        last_robot_data = self.robot.get_all_state()
        # both have more than n_obs_steps data

        # align camera obs timestamps
        dt = 1 / self.frequency
        last_timestamp = np.max([x['timestamp'][-1] for x in self.last_realsense_data.values()])
        obs_align_timestamps = last_timestamp - (np.arange(self.n_obs_steps)[::-1] * dt)

        camera_obs = dict()
        for camera_idx, value in self.last_realsense_data.items():
            this_timestamps = value['timestamp']
            this_idxs = list()
            for t in obs_align_timestamps:
                is_before_idxs = np.nonzero(this_timestamps < t)[0]
                this_idx = 0
                if len(is_before_idxs) > 0:
                    this_idx = is_before_idxs[-1]
                this_idxs.append(this_idx)
            # remap key
            camera_name = CAMERA_NAMES[camera_idx]
            camera_obs[f'camera_{camera_name}_color'] = value['color'][this_idxs]
            camera_obs[f'camera_{camera_name}_depth'] = value['depth'][this_idxs]
            camera_obs[f'camera_{camera_name}_intrinsics'] = value['intrinsics'][this_idxs]
            camera_obs[f'camera_{camera_name}_extrinsics'] = value['extrinsics'][this_idxs]

        # align robot obs
        robot_timestamps = last_robot_data['robot_receive_timestamp']
        this_timestamps = robot_timestamps
        this_idxs = list()
        for t in obs_align_timestamps:
            is_before_idxs = np.nonzero(this_timestamps < t)[0]
            this_idx = 0
            if len(is_before_idxs) > 0:
                this_idx = is_before_idxs[-1]
            this_idxs.append(this_idx)

        robot_obs_raw = dict()
        for k, v in last_robot_data.items():
            if k in self.obs_key_map:
                robot_obs_raw[self.obs_key_map[k]] = v


        robot_obs = dict()
        for k, v in robot_obs_raw.items():
            if k == 'wrist_cam_extrinsics':
                camera_obs[f'camera_wrist_extrinsics'] = v[this_idxs]
            else:
                robot_obs[k] = v[this_idxs]
        # fixed_extri = get_extrinsic([0.924, -0.046, 0.256], [0.596, 0.584, -0.398, -0.380])
        cam_1_extri = get_extrinsic([0.33140649116301046, 0.5088971764480946, 0.11701259737976832],
                                    [-0.14112166822506242, 0.7072779422922251, -0.6794010610611806, 0.1351176721720494])
        cam_2_extri = get_extrinsic([0.6093451170018969, -0.21758103645789892, 0.14001694566120815],
                                    [0.06233815389071674, -0.6854456458464545, -0.717860365560187, 0.10466478260345545])
        # camera_obs[f'camera_fixed_extrinsics'] = np.tile(fixed_extri, (camera_obs[f'camera_wrist_extrinsics'].shape[0], 1, 1))
        camera_obs[f'camera_1_extrinsics'] = np.tile(cam_1_extri, (camera_obs[f'camera_wrist_extrinsics'].shape[0], 1, 1))
        camera_obs[f'camera_2_extrinsics'] = np.tile(cam_2_extri, (camera_obs[f'camera_wrist_extrinsics'].shape[0], 1, 1))

        # return obs
        obs_data = dict(camera_obs)
        obs_data.update(robot_obs)

        # accumulate obs
        if self.obs_accumulator is not None:
            self.obs_accumulator.put(
                obs_data,
                obs_align_timestamps
            )
        obs_data['timestamp'] = obs_align_timestamps
        return obs_data

    def exec_actions(self,
                     actions: np.ndarray,
                     timestamps: np.ndarray,
                     mode: str = 'joint',
                     stages: Optional[np.ndarray] = None):
        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)
        if stages is None:
            stages = np.zeros_like(timestamps, dtype=np.int64)
        elif not isinstance(stages, np.ndarray):
            stages = np.array(stages, dtype=np.int64)

        # convert action to pose
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]
        new_stages = stages[is_new]


        # schedule waypoints
        if mode == 'joint':
            for i in range(len(new_actions)):
                self.robot.schedule_joint_waypoint(
                    pos=new_actions[i],
                    target_time=new_timestamps[i]
                )
        elif mode == 'eef':
            for i in range(len(new_actions)):
                self.robot.schedule_ee_waypoint(
                    pose=new_actions[i],
                    target_time=new_timestamps[i]
                )

        # record actions
        if self.action_accumulator is not None:
            self.action_accumulator.put(
                new_actions,
                new_timestamps
            )
        if self.stage_accumulator is not None:
            self.stage_accumulator.put(
                new_stages,
                new_timestamps
            )

    def get_robot_state(self):
        return self.robot.get_state()

    # recording API
    def start_episode(self, start_time=None, curr_outdir=None):
        "Start recording and return first obs"
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        assert self.is_ready

        # prepare recording stuff
        if curr_outdir is None:
            this_video_dir = self.video_dir.joinpath(str(self.episode_id))
        else:
            curr_outdir = pathlib.Path(curr_outdir)
            video_dir = curr_outdir.joinpath('videos')
            video_dir.mkdir(parents=True, exist_ok=True)
            this_video_dir = video_dir.joinpath(str(self.episode_id))
        this_video_dir.mkdir(parents=True, exist_ok=True)
        n_cameras = self.realsense.n_cameras
        video_paths = list()
        for i in range(n_cameras):
            video_paths.append(
                str(this_video_dir.joinpath(f'{i}.mp4').absolute()))

        # start recording on realsense
        self.realsense.restart_put(start_time=start_time)
        self.realsense.start_recording(video_path=video_paths, start_time=start_time)

        # create accumulators
        self.obs_accumulator = TimestampObsAccumulator(
            start_time=start_time,
            dt=1 / self.frequency
        )
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1 / self.frequency
        )
        self.stage_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1 / self.frequency
        )
        print(f'Episode {self.episode_id} started!')

    def end_episode(self, curr_outdir=None, incr_epi=False):
        "Stop recording"
        assert self.is_ready

        # stop video recorder
        self.realsense.stop_recording()

        if self.obs_accumulator is not None:
            # recording
            assert self.action_accumulator is not None
            assert self.stage_accumulator is not None

            # Since the only way to accumulate obs and action is by calling
            # get_obs and exec_actions, which will be in the same thread.
            # We don't need to worry new data come in here.
            obs_data = self.obs_accumulator.data
            obs_timestamps = self.obs_accumulator.timestamps

            num_cam = 0
            cam_width = -1
            cam_height = -1
            for key in obs_data.keys():
                if 'camera' in key and 'color' in key:
                    num_cam += 1
                    cam_height, cam_width = obs_data[key].shape[1:3]

            actions = self.action_accumulator.actions
            action_timestamps = self.action_accumulator.timestamps
            stages = self.stage_accumulator.actions
            n_steps = min(len(obs_timestamps), len(action_timestamps))
            if n_steps > 0:
                ### init episode data
                episode = {
                    'timestamp': None,
                    'stage': None,
                    'observations': 
                        {'joint_pos': [],
                         'joint_vel': [],
                         'full_joint_pos': [], # this is to compute FK
                         'robot_base_pose_in_world': np.asarray([np.eye(4)] * n_steps),
                        #  'joint_vel': [],
                         'ee_pos': [],
                        #  'ee_vel': [],
                        #  'finger_pos': {},
                        #  'force_torque': [],
                         'images': {},
                        },
                    # 'joint_action': [],
                    # 'cartesian_action': [],
                }
                # finger_names = ['left_finger_link','right_finger_link']
                # for finger in finger_names:
                #     episode['observations']['finger_pos'][finger] = []
                for cam in range(num_cam):
                    cam_name = CAMERA_NAMES[cam]
                    episode['observations']['images'][f'camera_{cam_name}_color'] = []
                    episode['observations']['images'][f'camera_{cam_name}_depth'] = []
                    episode['observations']['images'][f'camera_{cam_name}_intrinsics'] = []
                    episode['observations']['images'][f'camera_{cam_name}_extrinsics'] = []

                ### create attr dict
                attr_dict = {
                    'sim': False,
                }

                ### create config dict
                config_dict = {
                    'observations': {
                        'images': {},
                    },
                    'timestamp': {
                        'dtype': 'float64'
                    },
                }
                for cam in range(num_cam):
                    cam_name = CAMERA_NAMES[cam]
                    color_save_kwargs = {
                        'chunks': (1, cam_height, cam_width, 3), # (1, 480, 640, 3)
                        'compression': 'gzip',
                        'compression_opts': 9,
                        'dtype': 'uint8',
                    }
                    depth_save_kwargs = {
                        'chunks': (1, cam_height, cam_width), # (1, 480, 640)
                        'compression': 'gzip',
                        'compression_opts': 9,
                        'dtype': 'uint16',
                    }
                    config_dict['observations']['images'][f'camera_{cam_name}_color'] = color_save_kwargs
                    config_dict['observations']['images'][f'camera_{cam_name}_depth'] = depth_save_kwargs

                episode['timestamp'] = obs_timestamps[:n_steps]
                if self.ctrl_mode == 'joint':
                    episode['joint_action'] = actions[:n_steps]
                else:
                    episode['cartesian_action'] = actions[:n_steps]
                episode['stage'] = stages[:n_steps]
                for key, value in obs_data.items():
                    if 'camera' in key:
                        episode['observations']['images'][key] = value[:n_steps]
                    # elif 'finger' in key:
                    #     episode['observations']['finger_pos'][key] = value[:n_steps]
                    else:
                        episode['observations'][key] = value[:n_steps]

                ### save episode data
                if curr_outdir is None:
                    episode_path = self.output_dir.joinpath(f'episode_{self.episode_id}.hdf5')
                else:
                    self.curr_outdir = pathlib.Path(curr_outdir)
                    episode_path = self.curr_outdir.joinpath(f'episode_{self.episode_id}.hdf5')
                save_dict_to_hdf5(episode, config_dict, str(episode_path), attr_dict=attr_dict)

                print(f'Episode {self.episode_id} saved!')
                if incr_epi:
                    self.episode_id += 1

            self.obs_accumulator = None
            self.action_accumulator = None
            self.stage_accumulator = None

    def drop_episode(self):
        self.end_episode()
        self.replay_buffer.drop_episode()
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        if this_video_dir.exists():
            shutil.rmtree(str(this_video_dir))
        print(f'Episode {episode_id} dropped!')

def test_episode_start():
    # create env
    os.system('mkdir -p tmp')
    with RealEnvFranka(
            output_dir='tmp',
            robot_ip=''
        ) as env:
        print('Created env!')
        
        env.start_episode()
        print('Started episode!')

def test_env_obs_latency():
    os.system('mkdir -p tmp')
    with RealEnvFranka(
            output_dir='tmp',
        ) as env:
        print('Created env!')

        for i in range(100):
            start_time = time.time()
            obs = env.get_obs()
            end_time = time.time()
            print(f'obs latency: {end_time - start_time}')
            time.sleep(0.1)

def test_env_demo_replay():
    os.system('mkdir -p tmp')
    demo_path = '/home/manipulation-lab/gendp/data/rosbag/episode_1.hdf5'
    demo_dict, _ = load_dict_from_hdf5(demo_path)
    # actions = demo_dict['cartesian_action']
    actions = demo_dict['joint_action']
    with RealEnvFranka(
            output_dir='tmp',
            init_joints=True,
            ctrl_mode='joint'
        ) as env:
        print('Created env!')

        timestamps = time.time() + np.arange(len(actions)) / 10 + 1.0
        ik_init = [demo_dict['observations']['full_joint_pos'][0]] * len(actions)
        # print(demo_dict['observations']['full_joint_pos'][()])
        print(demo_dict['observations']['images']['fixed_extrinsic'][0])
        print(demo_dict['observations']['images']['wrist_extrinsic'][0])
        obs_dict = env.get_obs()
        # print(obs_dict['camera_fixed_extrinsics'])
        print(obs_dict['camera_left_extrinsics'])
        print(obs_dict['camera_right_extrinsics'])
        print(obs_dict['camera_wrist_extrinsics'])
        start_step = 0
        while True:
            curr_time = time.monotonic()
            loop_end_time = curr_time + 1.0
            end_step = min(start_step+10, len(actions))
            action_batch = actions[start_step:end_step]
            timestamp_batch = timestamps[start_step:end_step]
            ik_init_batch = ik_init[start_step:end_step]
            # print(action_batch)
            env.exec_actions(
                actions=action_batch,
                timestamps=timestamp_batch,
                mode='joint',
                # ik_init=ik_init_batch
            )
            print(f'executed {end_step - start_step} actions')
            start_step = end_step
            precise_wait(loop_end_time)
            if start_step >= len(actions):
                break

def test_cache_replay():
    import zarr
    import pytorch3d.transforms
    import torch
    import scipy.spatial.transform as st
    os.system('mkdir -p tmp')
    
    # only to get initial joint pos
    demo_path = '/home/yixuan/general_dp/data/real_aloha_demo/open_bag_demo_1/episode_0.hdf5'
    demo_dict, _ = load_dict_from_hdf5(demo_path)

    cache_path = '/home/yixuan/general_dp/data/real_aloha_demo/open_bag_demo_1/cache_no_seg_dino.zarr.zip'
    robot_sides = ['right', 'left']
    with zarr.ZipStore(cache_path, mode='r') as zip_store:
        replay_buffer = ReplayBuffer.copy_from_store(
            src_store=zip_store, store=zarr.MemoryStore())
    actions = replay_buffer['action'][()]
    with RealEnvFranka(
            output_dir='tmp',
            robot_sides=robot_sides,
        ) as env:
        print('Created env!')

        timestamps = time.time() + np.arange(len(actions)) / 10 + 1.0
        ik_init = [demo_dict['observations']['full_joint_pos'][0]] * len(actions)
        print(demo_dict['observations']['full_joint_pos'][()])

        # convert action from rotation 6d to euler
        actions_reshape = actions.reshape(actions.shape[0] * len(robot_sides), 10)
        action_pos = actions_reshape[:,:3]
        action_rot_6d = actions_reshape[:,3:9]
        action_rot_mat = pytorch3d.transforms.rotation_6d_to_matrix(torch.from_numpy(action_rot_6d)).numpy()
        action_rot_euler = st.Rotation.from_matrix(action_rot_mat).as_euler('xyz')
        actions_reshape = np.concatenate([action_pos, action_rot_euler, actions_reshape[:,-1:]], axis=-1)
        actions = actions_reshape.reshape(actions.shape[0], len(robot_sides) * 7)

        start_step = 0
        while True:
            curr_time = time.monotonic()
            loop_end_time = curr_time + 1.0
            end_step = min(start_step+10, len(actions))
            action_batch = actions[start_step:end_step]
            timestamp_batch = timestamps[start_step:end_step]
            ik_init_batch = ik_init[start_step:end_step]
            env.exec_actions(
                joint_actions=np.zeros((action_batch.shape[0], 7)),
                eef_actions=action_batch,
                timestamps=timestamp_batch,
                mode='eef',
                ik_init=ik_init_batch
            )
            print(f'executed {end_step - start_step} actions')
            start_step = end_step
            precise_wait(loop_end_time)
            if start_step >= len(actions):
                break


if __name__ == '__main__':
    # test_episode_start()
    # test_env_obs_latency()
    test_env_demo_replay()
    # test_cache_replay()
