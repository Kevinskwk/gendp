import glob
import os
import shutil
import time
import pickle
from typing import Dict, Optional
from pathlib import Path
import torch
import numpy as np
import copy
import multiprocessing
import zarr
from tqdm import tqdm
import concurrent.futures
import h5py
import cv2
import open3d as o3d
import scipy.spatial.transform as st
from filelock import FileLock
from threadpoolctl import threadpool_limits
from omegaconf import OmegaConf
import transforms3d
import scipy.spatial.transform as st

from gendp.common.pytorch_util import dict_apply
from gendp.common.replay_buffer import ReplayBuffer
from gendp.model.common.rotation_transformer import RotationTransformer
from gendp.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from gendp.common.kinematics_utils import KinHelper
from gendp.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from gendp.common.rob_mesh_utils import load_mesh, mesh_poses_to_pc
from gendp.common.data_utils import d3fields_proc, _convert_actions, load_dict_from_hdf5, modify_hdf5_from_dict
from gendp.common.tactile_utils import TactileProcessor
from gendp.dataset.base_dataset import BaseImageDataset
from gendp.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from gendp.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats,
)

from d3fields.fusion import Fusion
from d3fields.utils.my_utils import get_current_YYYY_MM_DD_hh_mm_ss_ms

register_codecs()

def transform_ee_pose_for_contact_field(ee_pos, ee_rpy):
    """
    Transform real-world end-effector pose to contact field model format.
    Applies z-translation of -0.14 and z-rotation of 45 degrees.
    
    Args:
        ee_pos: End-effector position (3D array)
        ee_rpy: End-effector roll-pitch-yaw (3D array)
    
    Returns:
        Transformed pose as 7D array [x, y, z, qx, qy, qz, qw]
    """
    # Apply z-translation of -0.14
    transformed_pos = ee_pos.copy()
    transformed_pos[2] -= 0.14
    
    # Convert RPY to rotation
    current_rot = st.Rotation.from_euler('xyz', ee_rpy)
    
    # Apply z-rotation of 45 degrees
    z_rotation = st.Rotation.from_euler('z', np.pi/4)  # 45 degrees in radians
    transformed_rot = z_rotation * current_rot
    
    # Convert back to quaternion in [x, y, z, w] format
    transformed_quat = transformed_rot.as_quat()
    
    # Return as 7D pose [x, y, z, qx, qy, qz, qw]
    return np.concatenate([transformed_pos, transformed_quat])


def get_tactile_marker_coordinates(ee_pose_7d, gripper_pos):
    """
    Generate tactile marker coordinates based on end-effector pose.
    
    Args:
        ee_pose_7d: 7D end-effector pose [x, y, z, qx, qy, qz, qw]
        gripper_pos: Gripper position (scalar, represents gripper opening)
    
    Returns:
        tuple: (tactile_coord_left, tactile_coord_right)
            Each is a numpy array of shape (7, 9, 3) representing marker positions
    """
    # Extract position and rotation from ee_pose
    ee_pos = ee_pose_7d[:3]
    ee_quat = ee_pose_7d[3:7]  # [qx, qy, qz, qw]
    
    # Convert quaternion to rotation matrix
    rotation = st.Rotation.from_quat(ee_quat)
    
    # Tactile sensor dimensions - NOTE: Model expects (7, 9, 3) format
    rows = 9  # Along z-axis of end-effector frame  
    cols = 7  # Along x-axis of end-effector frame
    marker_spacing = 0.002  # 2mm between markers
    
    # Calculate gripper offset (left: negative y, right: positive y)
    gripper_offset = gripper_pos / 2.0
    
    # Generate base marker grid in end-effector frame
    # X-axis: cols markers centered around 0
    x_positions = np.linspace(-(cols-1)*marker_spacing/2, (cols-1)*marker_spacing/2, cols)
    # Z-axis: rows markers centered around 0  
    z_positions = np.linspace(-(rows-1)*marker_spacing/2, (rows-1)*marker_spacing/2, rows)
    
    # Create meshgrid for marker positions - Note: X should be first dimension for (7, 9) format
    X, Z = np.meshgrid(x_positions, z_positions, indexing='ij')  # Use 'ij' indexing for (7, 9) format
    
    # Left tactile sensor (negative y offset) - Shape: (7, 9, 3)
    left_markers_local = np.zeros((cols, rows, 3))
    left_markers_local[:, :, 0] = X  # x positions
    left_markers_local[:, :, 1] = -gripper_offset  # y offset (negative for left)
    left_markers_local[:, :, 2] = Z  # z positions
    
    # Right tactile sensor (positive y offset) - Shape: (7, 9, 3)
    right_markers_local = np.zeros((cols, rows, 3))
    right_markers_local[:, :, 0] = X  # x positions
    right_markers_local[:, :, 1] = gripper_offset  # y offset (positive for right)
    right_markers_local[:, :, 2] = Z  # z positions
    
    # Transform marker positions to world frame
    left_markers_world = np.zeros_like(left_markers_local)
    right_markers_world = np.zeros_like(right_markers_local)
    
    for i in range(cols):  # Now iterating over cols (7)
        for j in range(rows):  # Now iterating over rows (9)
            # Left sensor
            local_pos_left = left_markers_local[i, j, :]
            world_pos_left = rotation.apply(local_pos_left) + ee_pos
            left_markers_world[i, j, :] = world_pos_left
            
            # Right sensor
            local_pos_right = right_markers_local[i, j, :]
            world_pos_right = rotation.apply(local_pos_right) + ee_pos
            right_markers_world[i, j, :] = world_pos_right
    
    return left_markers_world, right_markers_world


def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

# convert raw hdf5 data to replay buffer, which is used for diffusion policy training
def _convert_real_to_dp_replay(store, shape_meta, dataset_dir, rotation_transformer, 
        n_workers=None, max_inflight_tasks=None, fusion : Optional[Fusion]=None, robot_name='panda', expected_labels=None,
        exclude_colors=[]):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    depth_keys = list()
    lowdim_keys = list()
    spatial_keys = list()
    tactile_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta['obs']
    trim_tail = shape_meta['trim_tail']
    for key, attr in obs_shape_meta.items():
        shape = attr['shape']
        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            rgb_keys.append(key)
        if type == 'depth':
            depth_keys.append(key)
        elif type == 'low_dim':
            lowdim_keys.append(key)
        elif type == 'spatial':
            spatial_keys.append(key)
            max_pts_num = obs_shape_meta[key]['shape'][1]
        elif type == 'tactile':
            tactile_keys.append(key)
    
    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)

    episodes_paths = glob.glob(os.path.join(dataset_dir, 'episode_*.hdf5'))
    episodes_stem_name = [Path(path).stem for path in episodes_paths]
    episodes_idx = [int(stem_name.split('_')[-1]) for stem_name in episodes_stem_name]
    episodes_idx = sorted(episodes_idx)
    kin_helper = KinHelper(robot_name=robot_name)
        
    episode_ends = list()
    prev_end = 0
    lowdim_data_dict = dict()
    rgb_data_dict = dict()
    depth_data_dict = dict()
    spatial_data_dict = dict()
    tactile_data_dict = dict()
    tactile_coord_dict = dict()  # Store 3D marker coordinates for tactile sensors
    tactile_processors = dict()  # Store TactileProcessor instances for each tactile key
    for epi_idx in tqdm(episodes_idx, desc=f"Loading episodes"):
        dataset_path = os.path.join(dataset_dir, f'episode_{epi_idx}.hdf5')
        feats_per_epi = list() # save it separately to avoid OOM
        with h5py.File(dataset_path) as file:
            # count total steps
            # episode_length = file['cartesian_action'].shape[0]
            episode_length = file['joint_action'].shape[0] - trim_tail
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
            
            # save lowdim data to lowedim_data_dict
            for key in lowdim_keys + ['action']:
                data_key = 'observations/' + key
                if key == 'action':
                    data_key = 'cartesian_action' if 'key' not in shape_meta['action'] else shape_meta['action']['key']
                if key not in lowdim_data_dict:
                    lowdim_data_dict[key] = list()
                if data_key == 'cartesian_action':
                    this_data = file['observations']['ee_pose'][:episode_length]
                else:
                    this_data = file[data_key][:episode_length]
                if key == 'action':
                    this_data = _convert_actions(
                        raw_actions=this_data,
                        rotation_transformer=rotation_transformer,
                        action_key=data_key,
                    )
                    assert this_data.shape == (episode_length,) + tuple(shape_meta['action']['shape'])
                else:
                    assert this_data.shape == (episode_length,) + tuple(shape_meta['obs'][key]['shape'])
                lowdim_data_dict[key].append(this_data)
            
            for key in rgb_keys:
                if key not in rgb_data_dict:
                    rgb_data_dict[key] = list()
                if 'tactile' in key:
                    frames = file['observations']['tactile'][key][:episode_length]
                else:
                    frames = file['observations']['images'][key][:episode_length]
                shape = tuple(shape_meta['obs'][key]['shape'])
                c,h,w = shape
                resize_imgs = [cv2.resize(img, (w,h), interpolation=cv2.INTER_AREA) for img in frames]
                frames = np.stack(resize_imgs, axis=0)
                assert frames[0].shape == (h,w,c)
                rgb_data_dict[key].append(frames)
            
            for key in depth_keys:
                if key not in depth_data_dict:
                    depth_data_dict[key] = list()
                frames = file['observations']['images'][key][:episode_length]
                shape = tuple(shape_meta['obs'][key]['shape'])
                c,h,w = shape
                resize_imgs = [cv2.resize(img, (w,h), interpolation=cv2.INTER_AREA) for img in frames]
                frames = np.stack(resize_imgs, axis=0)[..., None]
                frames = np.clip(frames, 0, 1000).astype(np.uint16)
                assert frames[0].shape == (h,w,c)
                depth_data_dict[key].append(frames)
            
            for key in spatial_keys:
                assert key == 'd3fields' # only support d3fields for now
                if fusion is None:
                    raise RuntimeError('fusion must be specified')
                
                # construct inputs for d3fields processing
                view_keys = shape_meta['obs'][key]['info']['view_keys']
                use_dino = False
                distill_dino = shape_meta['obs'][key]['info']['distill_dino'] if 'distill_dino' in shape_meta['obs'][key]['info'] else False
                use_seg = False
                tool_names = [None, None]
                if 'right_tool' in shape_meta['obs'][key]['info']:
                    tool_names[0] = shape_meta['obs'][key]['info']['right_tool']
                if 'left_tool' in shape_meta['obs'][key]['info']:
                    tool_names[1] = shape_meta['obs'][key]['info']['left_tool']
                is_joint = ('key' in shape_meta['action'].keys()) and (shape_meta['action']['key'] == 'joint_action')
                color_seq = np.stack([file['observations']['images'][f'{k}_color'][:episode_length] for k in view_keys], axis=1) # (T, V, H ,W, C)
                depth_seq = np.stack([file['observations']['images'][f'{k}_depth'][:episode_length] for k in view_keys], axis=1) / 1000. # (T, V, H ,W)
                extri_seq = np.stack([file['observations']['images'][f'{k}_extrinsics'][:episode_length] for k in view_keys], axis=1) # (T, V, 4, 4)
                intri_seq = np.stack([file['observations']['images'][f'{k}_intrinsics'][:episode_length] for k in view_keys], axis=1) # (T, V, 3, 3)
                qpos_seq = file['observations']['full_joint_pos'][:episode_length] if 'full_joint_pos' in file['observations'] else file['observations']['joint_pos'][:-trim_tail] # (T, -1)
                if 'robot_base_pose_in_world' in file['observations']:
                    robot_base_pose_in_world_seq = file['observations']['robot_base_pose_in_world'][:episode_length] # (T, 4, 4)
                else:
                    print('using default robot base pose!')
                    robot_base_pose_in_world = np.array([[ 1.  ,  0.  ,  0.  , -0.52],
                                                         [ 0.  ,  1.  ,  0.  , -0.06],
                                                         [ 0.  ,  0.  ,  1.  ,  0.03],
                                                         [ 0.  ,  0.  ,  0.  ,  1.  ]])
                    robot_base_pose_in_world_seq = np.stack([robot_base_pose_in_world] * qpos_seq.shape[0], axis=0)
                
                aggr_src_pts_ls, aggr_feats_ls = d3fields_proc(
                    fusion=fusion,
                    shape_meta=shape_meta['obs'][key],
                    color_seq=color_seq,
                    depth_seq=depth_seq,
                    extri_seq=extri_seq,
                    intri_seq=intri_seq,
                    robot_base_pose_in_world_seq=robot_base_pose_in_world_seq,
                    teleop_robot=kin_helper,
                    qpos_seq=qpos_seq,
                    expected_labels=expected_labels,
                    tool_names=tool_names,
                    exclude_colors=exclude_colors,
                )
                
                if distill_dino:
                    for pts_idx, aggr_src_pts in enumerate(aggr_src_pts_ls):
                        aggr_src_pts_ls[pts_idx] = np.concatenate([aggr_src_pts, aggr_feats_ls[pts_idx]], axis=-1)
                
                if key not in spatial_data_dict:
                    spatial_data_dict[key] = list()
                
                spatial_data_dict[key] = spatial_data_dict[key] + aggr_src_pts_ls
                feats_per_epi = feats_per_epi + aggr_feats_ls
                
            # save feats for every episode as hdf5
            if 'd3fields' in shape_meta['obs'].keys() and use_dino:
                feats_per_epi = np.stack(feats_per_epi, axis=0) # (T, N, 1024)
                if use_seg:
                    feats_prefix = ''
                else:
                    feats_prefix = '_no_seg'
                if is_joint:
                    feats_prefix += '_joint'
                os.system(f'mkdir -p {os.path.join(dataset_dir, f"feats{feats_prefix}")}')
                with h5py.File(os.path.join(dataset_dir, f'feats{feats_prefix}', f'episode_{epi_idx}.hdf5'), 'w') as file:
                    file.create_dataset('feats', data=feats_per_epi, dtype=np.float32)

            for key in tactile_keys:
                if key not in tactile_data_dict:
                    tactile_data_dict[key] = list()
                
                frames = file['observations']['tactile'][key][:episode_length]
                setting = shape_meta['obs'][key]['setting']
                
                # Get robot pose data for 3D marker coordinate computation
                ee_poses = file['observations']['ee_pose'][:episode_length]  # (T, 8) [x,y,z,qx,qy,qz,qw,gripper]
                
                # Initialize TactileProcessor for this key if not already done
                if key not in tactile_processors:
                    tactile_processors[key] = TactileProcessor(
                        width=320,
                        height=240,
                        marker_config=setting,
                        use_gpu=True
                    )
                
                # Process frames using TactileProcessor to get force field data
                force_fields = []
                tactile_coords = []
                
                for frame_idx, frame in enumerate(frames):
                    # Get force field using TactileProcessor.process_frame
                    # This returns force_field (N*M, 3) with [current_x, current_y, depth]
                    # and initial_positions (N*M, 2) with [initial_x, initial_y]
                    force_field = tactile_processors[key].process_frame(frame)  # (M, N, 3)
                    force_fields.append(force_field)
                    
                    # Get 3D marker coordinates using robot pose
                    ee_pose_8d = ee_poses[frame_idx]  # [x,y,z,qx,qy,qz,qw,gripper]
                    ee_pos = ee_pose_8d[:3]  # [x, y, z]
                    ee_quat = ee_pose_8d[3:7]  # [qx, qy, qz, qw]
                    gripper_pos = ee_pose_8d[7] if len(ee_pose_8d) > 7 else 0.05  # Default gripper width
                    
                    # Transform the pose for contact field model
                    # Apply z-translation of -0.14
                    transformed_pos = ee_pos.copy()
                    transformed_pos[2] -= 0.14
                    
                    # Apply z-rotation of 45 degrees
                    current_rot = st.Rotation.from_quat(ee_quat)
                    z_rotation = st.Rotation.from_euler('z', np.pi/4)  # 45 degrees in radians
                    transformed_rot = z_rotation * current_rot
                    transformed_quat = transformed_rot.as_quat()
                    
                    # Create 7D pose for marker coordinate calculation
                    ee_pose_7d = np.concatenate([transformed_pos, transformed_quat])
                    
                    # Get 3D marker coordinates for left and right sensors
                    if 'left' in key:
                        tactile_coord_left, _ = get_tactile_marker_coordinates(ee_pose_7d, gripper_pos)
                        tactile_coords.append(tactile_coord_left)  # (7, 9, 3)
                    elif 'right' in key:
                        _, tactile_coord_right = get_tactile_marker_coordinates(ee_pose_7d, gripper_pos)
                        tactile_coords.append(tactile_coord_right)  # (7, 9, 3)
                    else:
                        # If not specified as left/right, assume it's left sensor
                        tactile_coord_left, _ = get_tactile_marker_coordinates(ee_pose_7d, gripper_pos)
                        tactile_coords.append(tactile_coord_left)  # (7, 9, 3)
                
                # Stack the processed data
                force_field_data = np.stack(force_fields, axis=0)  # (T, M, N, 3)
                tactile_coord_data = np.stack(tactile_coords, axis=0)  # (T, 7, 9, 3)
                
                print(f'{key} force field shape:', force_field_data.shape)
                print(f'{key} tactile coord shape:', tactile_coord_data.shape)
                
                # Store force field data with "_force_field" suffix
                force_field_key = f"{key}_force_field"
                if force_field_key not in tactile_data_dict:
                    tactile_data_dict[force_field_key] = list()
                tactile_data_dict[force_field_key].append(force_field_data)
                
                # Store 3D marker coordinates with "_coord" suffix
                coord_key = f"{key}_coord"
                if coord_key not in tactile_coord_dict:
                    tactile_coord_dict[coord_key] = list()
                tactile_coord_dict[coord_key].append(tactile_coord_data)


        if fusion is not None:
            fusion.clear_xmem_memory()
    
    def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
        try:
            zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
            # make sure we can successfully decode
            _ = zarr_arr[zarr_idx]
            return True
        except Exception as e:
            return False
    
    # dump data_dict
    print('Dumping meta data')
    n_steps = episode_ends[-1]
    _ = meta_group.array('episode_ends', episode_ends, 
        dtype=np.int64, compressor=None, overwrite=True)

    print('Dumping lowdim data')
    for key, data in lowdim_data_dict.items():
        data = np.concatenate(data, axis=0)
        _ = data_group.array(
            name=key,
            data=data,
            shape=data.shape,
            chunks=data.shape,
            compressor=None,
            dtype=data.dtype
        )
    
    print('Dumping rgb data')
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = set()
        for key, data in rgb_data_dict.items():
            hdf5_arr = np.concatenate(data, axis=0)
            shape = tuple(shape_meta['obs'][key]['shape'])
            c,h,w = shape
            this_compressor = Jpeg2k(level=50)
            img_arr = data_group.require_dataset(
                name=key,
                shape=(n_steps,h,w,c),
                chunks=(1,h,w,c),
                compressor=this_compressor,
                dtype=np.uint8
            )
            for hdf5_idx in tqdm(range(hdf5_arr.shape[0])):
                if len(futures) >= max_inflight_tasks:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    for f in completed:
                        if not f.result():
                            raise RuntimeError('Failed to encode image!')
                zarr_idx = hdf5_idx
                futures.add(
                    executor.submit(img_copy, 
                        img_arr, zarr_idx, hdf5_arr, hdf5_idx))
        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError('Failed to encode image!')
    
    print('Dumping depth data')
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = set()
        for key, data in depth_data_dict.items():
            hdf5_arr = np.concatenate(data, axis=0)
            shape = tuple(shape_meta['obs'][key]['shape'])
            c,h,w = shape
            this_compressor = Jpeg2k(level=50)
            img_arr = data_group.require_dataset(
                name=key,
                shape=(n_steps,h,w,c),
                chunks=(1,h,w,c),
                compressor=this_compressor,
                dtype=np.uint16
            )
            for hdf5_idx in tqdm(range(hdf5_arr.shape[0])):
                if len(futures) >= max_inflight_tasks:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    for f in completed:
                        if not f.result():
                            raise RuntimeError('Failed to encode image!')
                zarr_idx = hdf5_idx
                futures.add(
                    executor.submit(img_copy, 
                        img_arr, zarr_idx, hdf5_arr, hdf5_idx))
        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError('Failed to encode image!')
            
    # dump spatial data
    print('Dumping spatial data')
    for key, data in spatial_data_dict.items():
        # pad to max_pts_num
        for d_i, d in enumerate(data):
            if d.shape[0] > max_pts_num:
                data[d_i] = d[:max_pts_num]
            else:
                data[d_i] = np.pad(d, ((0,max_pts_num-d.shape[0]),(0,0)), mode='constant')
        data = np.stack(data, axis=0) # (T, N, 1027)
        _ = data_group.array(
            name=key,
            data=data,
            shape=data.shape,
            chunks=(1,) + data.shape[1:],
            compressor=None,
            dtype=data.dtype
        )

    # dump tactile data
    print('Dumping tactile data')
    for key, data in tactile_data_dict.items():
        # pad to max_pts_num
        # for d_i, d in enumerate(data):
        #     if d.shape[0] > max_pts_num:
        #         data[d_i] = d[:max_pts_num]
        #     else:
        #         data[d_i] = np.pad(d, ((0,max_pts_num-d.shape[0]),(0,0)), mode='constant')
        data = np.concatenate(data, axis=0)
        print('data shape', data.shape)
        _ = data_group.array(
            name=key,
            data=data,
            shape=data.shape,
            chunks=(1,) + data.shape[1:],
            compressor=None,
            dtype=data.dtype
        )

    # dump tactile coordinate data
    print('Dumping tactile coordinate data')
    for key, data in tactile_coord_dict.items():
        data = np.concatenate(data, axis=0)
        print(f'{key} coordinate data shape', data.shape)
        _ = data_group.array(
            name=key,
            data=data,
            shape=data.shape,
            chunks=(1,) + data.shape[1:],
            compressor=None,
            dtype=data.dtype
        )
    
    replay_buffer = ReplayBuffer(root)
    return replay_buffer

class RealDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_dir: str,
            vis_input: False,
            horizon=1,
            pad_before=0,
            pad_after=0,
            rotation_rep='rotation_6d',
            use_legacy_normalizer=True,
            use_cache=True,
            seed=42,
            val_ratio=0.0,
            manual_val_mask=False,
            manual_val_start=-1,
            n_obs_steps=None,
            robot_name='panda',
            expected_labels=None,
            exclude_colors=[]
            ):
        
        super().__init__()
        
        rotation_transformer = RotationTransformer(
            from_rep='euler_angles', to_rep=rotation_rep, from_convention='xyz')
        
        replay_buffer = None
        fusion = None
        cache_info_str = ''
        for key, attr in shape_meta['obs'].items():
            if ('type' in attr) and (attr['type'] == 'depth'):
                cache_info_str += '_rgbd'
                break
        if 'force_torque' in shape_meta['obs']:
            cache_info_str += '_ft'
        for key, attr in shape_meta['obs'].items():
            if ('tactile' in key) and ('type' in attr) and (attr['type'] == 'rgb'):
                cache_info_str += '_tactile_rgb'
                break
        for key, attr in shape_meta['obs'].items():
            if ('tactile' in key) and ('type' in attr) and (attr['type'] == 'tactile'):
                cache_info_str += '_tactile_ff'
                break
        if 'd3fields' in shape_meta['obs']:
            use_seg = False
            use_dino = False
            distill_dino = shape_meta['obs']['d3fields']['info']['distill_dino'] if 'distill_dino' in shape_meta['obs']['d3fields']['info'] else False
            if use_seg:
                cache_info_str += '_seg'
            else:
                cache_info_str += '_no_seg'
            if not use_dino and not distill_dino:
                cache_info_str += '_no_dino'
            elif not use_dino and distill_dino:
                cache_info_str += '_distill_dino'
            else:
                cache_info_str += '_dino'
            if 'key' in shape_meta['action'] and shape_meta['action']['key'] == 'joint_action':
                cache_info_str += '_joint'
            else:
                cache_info_str += '_eef'
            if 'trim_tail' in shape_meta and shape_meta['trim_tail'] > 0:
                cache_info_str += '_trim'
        if use_cache:
            cache_zarr_path = os.path.join(dataset_dir, f'cache{cache_info_str}.zarr.zip')
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # create fusion if necessary
                    self.fusion_dtype = torch.float16
                    # If get stuck here, try to `export OMP_NUM_THREADS=1`
                    # refer: https://github.com/pytorch/pytorch/issues/21956
                    for key, attr in shape_meta['obs'].items():
                        if ('type' in attr) and (attr['type'] == 'spatial'):
                            num_cam = len(attr['info']['view_keys'])
                            fusion = Fusion(num_cam=num_cam, dtype=self.fusion_dtype)
                            break
                    # cache does not exists
                    try:
                        print('Cache does not exist. Creating!')
                        # store = zarr.DirectoryStore(cache_zarr_path)
                        replay_buffer = _convert_real_to_dp_replay(
                            store=zarr.MemoryStore(), 
                            shape_meta=shape_meta, 
                            dataset_dir=dataset_dir, 
                            rotation_transformer=rotation_transformer,
                            fusion=fusion,
                            robot_name=robot_name,
                            expected_labels=expected_labels,
                            exclude_colors=exclude_colors,
                            )
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            # create fusion if necessary
            self.fusion_dtype = torch.float16
            # If get stuck here, try to `export OMP_NUM_THREADS=1`
            # refer: https://github.com/pytorch/pytorch/issues/21956
            for key, attr in shape_meta['obs'].items():
                if ('type' in attr) and (attr['type'] == 'spatial'):
                    num_cam = len(attr['info']['view_keys'])
                    fusion = Fusion(num_cam=num_cam, dtype=self.fusion_dtype)
                    break
            replay_buffer = _convert_real_to_dp_replay(
                store=zarr.MemoryStore(), 
                shape_meta=shape_meta, 
                dataset_dir=dataset_dir,
                rotation_transformer=rotation_transformer,
                fusion=fusion,
                robot_name=robot_name,
                expected_labels=expected_labels,
            )
        self.replay_buffer = replay_buffer
        if fusion is not None:
            fusion.close()
        
        if vis_input:
            if 'd3fields' in shape_meta['obs'] and shape_meta['obs']['d3fields']['info']['distill_dino']:
                vis_distill_feats = True
            else:
                vis_distill_feats = False
            self.replay_buffer.visualize_data(output_dir=os.path.join(dataset_dir, f'replay_buffer_vis_{get_current_YYYY_MM_DD_hh_mm_ss_ms()}'), vis_distill_feats=vis_distill_feats)
        
        rgb_keys = list()
        depth_keys = list()
        lowdim_keys = list()
        spatial_keys = list()
        tactile_keys = list()
        tactile_coord_keys = list()  # For 3D tactile marker coordinates
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'depth':
                depth_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
            elif type == 'spatial':
                spatial_keys.append(key)
            elif type == 'tactile':
                tactile_keys.append(key)
        
        # Also detect dynamically created tactile coordinate keys
        for key in self.replay_buffer.keys():
            if key.endswith('_coord') and any(tkey in key for tkey in tactile_keys):
                tactile_coord_keys.append(key)
            # Also add force field keys to tactile_keys if they're not already there
            elif key.endswith('_force_field') and any(tkey in key for tkey in tactile_keys):
                if key not in tactile_keys:
                    tactile_keys.append(key)
        
        # for key in rgb_keys:
        #     replay_buffer[key].compressor.numthreads=1

        if not manual_val_mask:
            val_mask = get_val_mask(
                n_episodes=replay_buffer.n_episodes, 
                val_ratio=val_ratio,
                seed=seed)
        else:
            try:
                assert manual_val_start >= 0
                assert manual_val_start < replay_buffer.n_episodes
            except:
                raise RuntimeError('invalid manual_val_start')
            val_mask = np.zeros((replay_buffer.n_episodes,), dtype=np.bool)
            val_mask[manual_val_start:] = True
        train_mask = ~val_mask
        
        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + depth_keys + lowdim_keys + spatial_keys + tactile_keys:
                key_first_k[key] = n_obs_steps

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            dataset_dir=dataset_dir,
            key_first_k=key_first_k,
            shape_meta=shape_meta,)
        
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.depth_keys = depth_keys
        self.lowdim_keys = lowdim_keys
        self.spatial_keys = spatial_keys
        self.tactile_keys = tactile_keys
        self.tactile_coord_keys = tactile_coord_keys
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.use_legacy_normalizer = use_legacy_normalizer
        self.dataset_dir = dataset_dir
        self.key_first_k = key_first_k

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
            dataset_dir=self.dataset_dir,
            key_first_k=self.key_first_k,
            shape_meta=self.shape_meta,
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer['action'])
        if self.use_legacy_normalizer:
            this_normalizer = normalizer_from_stat(stat)
        else:
            raise RuntimeError('unsupported')
        normalizer['action'] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if key.endswith('pos'):
                # this_normalizer = get_range_normalizer_from_stat(stat)
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('pose'):
                # this_normalizer = get_range_normalizer_from_stat(stat)
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('vel'):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('force_torque'):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            else:
                raise RuntimeError(f'{key} unsupported')
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        
        for key in self.depth_keys:
            normalizer[key] = get_image_range_normalizer()
        
        # spatial
        is_joint = ('key' in self.shape_meta['action'].keys()) and (self.shape_meta['action']['key'] == 'joint_action')
        for key in self.spatial_keys:
            B, N, C = self.replay_buffer[key].shape
            stat = array_to_stats(self.replay_buffer[key][()].reshape(B * N, C))
            if self.shape_meta['obs'][key]['shape'][0] == 1027:
                # compute normalizer for feats of top 10 demos
                feats = []
                for demo_i in range(1):
                    if is_joint:
                        feats_prefix += '_joint'
                    if os.path.exists(os.path.join(self.dataset_dir, f'feats{feats_prefix}', f'episode_{demo_i}.hdf5')):
                        with h5py.File(os.path.join(self.dataset_dir, f'feats{feats_prefix}', f'episode_{demo_i}.hdf5')) as file:
                            feats.append(file['feats'][()])
                feats = np.concatenate(feats, axis=0) # (10, T, N, 1024)
                feats = feats.reshape(-1, 1024) # (10 * T * N, 1024)
                feat_stat = array_to_stats(feats)
                for stat_key in stat:
                    stat[stat_key] = np.concatenate([stat[stat_key], feat_stat[stat_key]], axis=0)
                normalizer[key] = get_range_normalizer_from_stat(stat, ignore_dim=[0,1,2])
            else:
                normalizer[key] = get_identity_normalizer_from_stat(stat)

        # tactile
        for key in self.tactile_keys:
            B, N, C = self.replay_buffer[key].shape
            stat = array_to_stats(self.replay_buffer[key][()].reshape(B * N, C))
            normalizer[key] = get_identity_normalizer_from_stat(stat)

        # tactile coordinates (3D marker positions)
        for key in self.tactile_coord_keys:
            B, H, W, C = self.replay_buffer[key].shape  # (T, 7, 9, 3)
            stat = array_to_stats(self.replay_buffer[key][()].reshape(B * H * W, C))
            normalizer[key] = get_identity_normalizer_from_stat(stat)

        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(sample[key][T_slice],-1,1
                ).astype(np.float32) / 255.
            # T,C,H,W
            del sample[key]
        for key in self.depth_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint16 image to float32
            obs_dict[key] = np.moveaxis(sample[key][T_slice],-1,1
                ).astype(np.float32) / 1000.
            # T,C,H,W
            del sample[key]
        for key in self.lowdim_keys:
            obs_dict[key] = sample[key][T_slice].astype(np.float32)
            del sample[key]
        for key in self.spatial_keys:
            obs_dict[key] = np.moveaxis(sample[key][T_slice],1,2).astype(np.float32)
            del sample[key]
        for key in self.tactile_keys:
            obs_dict[key] = np.moveaxis(sample[key][T_slice],1,2).astype(np.float32)
            del sample[key]
        for key in self.tactile_coord_keys:
            # Tactile coordinates have shape (T, 7, 9, 3), reshape to (T, 7*9, 3) for consistency
            coord_data = sample[key][T_slice].astype(np.float32)  # (T, 7, 9, 3)
            T, H, W, C = coord_data.shape
            obs_dict[key] = coord_data.reshape(T, H*W, C)  # (T, 63, 3)
            del sample[key]

        data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(sample['action'].astype(np.float32))
        }
        return data
    

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return data

def update_ee_pose():
    # create on 12/17/2023, only for one-time use
    
    ### ALOHA fixed constants
    DT = 0.02
    JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
    START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]
    START_EE_POSE = [2.56418115e-01, -5.50126845e-04,  2.95703636e-01,  6.88682872e-04, -3.83402967e-02, -1.18223866e-03,  9.99263804e-01, -0.3]

    # Left finger position limits (qpos[7]), right_finger = -1 * left_finger
    MASTER_GRIPPER_POSITION_OPEN = 0.02417
    MASTER_GRIPPER_POSITION_CLOSE = 0.01244
    PUPPET_GRIPPER_POSITION_OPEN = 0.05800
    PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

    # Gripper joint limits (qpos[6])
    MASTER_GRIPPER_JOINT_OPEN = 0.3083
    MASTER_GRIPPER_JOINT_CLOSE = -0.6842
    PUPPET_GRIPPER_JOINT_OPEN = 1.4910
    PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

    ############################ Helper functions ############################

    MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
    PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
    MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
    PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
    MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

    MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
    PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
    MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
    PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
    MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

    MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
    PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

    MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
    MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
    PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
    PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

    MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2
    
    
    DEBUG = False
    
    # update ee_pose in hdf5
    data_dir = '/home/yixuan/general_dp/data/real_aloha_demo/open_bag_v2_demo_1'
    epi_s = 0
    epi_e = 1
    kin_helper = KinHelper(robot_name='trossen_vx300s_v3')
    for epi_i in tqdm(range(epi_s, epi_e)):
        epi_fn = os.path.join(data_dir, f'episode_{epi_i}.hdf5')
        epi_data, fn = load_dict_from_hdf5(epi_fn)
        # joint_action = epi_data['joint_action']
        old_cartesian_action = epi_data['cartesian_action']
        robot_base_in_world_seq = epi_data['observations']['robot_base_pose_in_world']
        sec_base_in_world = np.array([[0,1,0,-0.13],
                                    [-1,0,0,0.27],
                                    [0,0,1,0.02],
                                    [0,0,0,1]])
        new_epi_data = {'cartesian_action': np.array(epi_data['cartesian_action']).copy(),
                        # 'observations': {
                        #     'ee_pose': np.array(epi_data['observations']['ee_pose']).copy(),
                        #     }
                        }
        for i in range(old_cartesian_action.shape[0]):
            ### update cartesian_action
            # puppet_gripper_pos = PUPPET_JOINT2POS(joint_action[i,-1])
            # puppet_action_qpos = np.concatenate([joint_action[i,:-1], np.array([puppet_gripper_pos, -puppet_gripper_pos])])
            # puppet_action_eef_mat = kin_helper.compute_fk_links(qpos=puppet_action_qpos, link_idx=[kin_helper.eef_link_idx])[0]
            puppet_action_old_eef = old_cartesian_action[i] # (14,)
            puppet_action_old_sec_eef = puppet_action_old_eef[7:] # (7,)
            puppet_action_old_sec_mat = np.eye(4)
            puppet_action_old_sec_mat[:3, 3] = puppet_action_old_sec_eef[:3]
            puppet_action_old_sec_mat[:3, :3] = transforms3d.euler.euler2mat(*puppet_action_old_sec_eef[3:6])
            puppet_action_new_sec_mat = np.linalg.inv(robot_base_in_world_seq[i, 0]) @ sec_base_in_world @ puppet_action_old_sec_mat
            puppet_action_eef = np.concatenate([puppet_action_old_eef[:7],
                                                puppet_action_new_sec_mat[:3,3],
                                                transforms3d.euler.mat2euler(puppet_action_new_sec_mat[:3,:3]),
                                                puppet_action_old_sec_eef[-1:]])
            if DEBUG:
                print('original cartesian_action: ', epi_data['cartesian_action'][i])
                print('new cartesian_action: ', puppet_action_eef)
            new_epi_data['cartesian_action'][i] = puppet_action_eef

            # ### update ee_pose
            # qpos = epi_data['observations']['joint_pos'][i]
            # full_qpos = epi_data['observations']['full_joint_pos'][i]
            # puppet_eef_mat = kin_helper.compute_fk_links(qpos=full_qpos, link_idx=[kin_helper.eef_link_idx])[0]
            # puppet_eef = np.concatenate([puppet_eef_mat[:3,3],
            #                              transforms3d.euler.mat2euler(puppet_eef_mat[:3,:3]),
            #                              qpos[-1:]])
            # if DEBUG:
            #     print('original ee_pose: ', epi_data['observations']['ee_pose'][i])
            #     print('new ee_pose: ', puppet_eef)
            # new_epi_data['observations']['ee_pose'][i] = puppet_eef
        fn.close()
        modify_hdf5_from_dict(epi_fn, new_epi_data)

if __name__ == '__main__':
    update_ee_pose()
