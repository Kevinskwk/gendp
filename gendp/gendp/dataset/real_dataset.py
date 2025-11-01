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
from omegaconf import OmegaConf, DictConfig
import transforms3d
import scipy.spatial.transform as st
import yaml

from gendp.common.pytorch_util import dict_apply
from gendp.common.replay_buffer import ReplayBuffer
from gendp.model.common.rotation_transformer import RotationTransformer
from gendp.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from gendp.common.kinematics_utils import KinHelper
from gendp.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from gendp.common.rob_mesh_utils import load_mesh, mesh_poses_to_pc
from gendp.common.data_utils import d3fields_proc, _convert_actions, _convert_ee_pose_obs, load_dict_from_hdf5, modify_hdf5_from_dict
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
        exclude_colors=[], contact_field_model=None, contact_field_device='cuda'):
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
    
    # Check if contact field is enabled
    use_contact_field = contact_field_model is not None
    if use_contact_field:
        print("✅ Contact field model enabled - will add 4 contact channels to d3fields")
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
                    delta_action = shape_meta['action'].get('delta', False)
                    this_data = _convert_actions(
                        raw_actions=this_data,
                        rotation_transformer=rotation_transformer,
                        action_key=data_key,
                        delta_action=delta_action,
                    )
                    assert this_data.shape == (episode_length,) + tuple(shape_meta['action']['shape']), \
                        f"Action shape mismatch: {this_data.shape} vs expected {(episode_length,) + tuple(shape_meta['action']['shape'])}"
                elif key == 'ee_pose':
                    # Convert ee_pose from [pos(3), euler(3), gripper(1)] to [pos(3), rot6d(6)]
                    # print(f"Converting ee_pose: input shape {this_data.shape}, expected output shape {(episode_length,) + tuple(shape_meta['obs'][key]['shape'])}")
                    this_data = _convert_ee_pose_obs(this_data, rotation_transformer)
                    # print(f"After conversion: {this_data.shape}")
                    assert this_data.shape == (episode_length,) + tuple(shape_meta['obs'][key]['shape']), \
                        f"EE pose shape mismatch: {this_data.shape} vs expected {(episode_length,) + tuple(shape_meta['obs'][key]['shape'])}"
                else:
                    assert this_data.shape == (episode_length,) + tuple(shape_meta['obs'][key]['shape']), \
                        f"Obs {key} shape mismatch: {this_data.shape} vs expected {(episode_length,) + tuple(shape_meta['obs'][key]['shape'])}"
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
                
                # Add contact field prediction if enabled
                if use_contact_field:
                    print(f"Processing contact field for episode {epi_idx}...")
                    
                    # Run d3fields_proc once with object/background segmentation enabled
                    obj_bg_result = d3fields_proc(
                        fusion=fusion,
                        shape_meta=shape_meta['obs'][key],
                        color_seq=color_seq,
                        depth_seq=depth_seq,
                        extri_seq=extri_seq,
                        intri_seq=intri_seq,
                        robot_base_pose_in_world_seq=robot_base_pose_in_world_seq,
                        teleop_robot=kin_helper,
                        qpos_seq=qpos_seq,
                        exclude_threshold=0.01,
                        use_obj_bg_seg=True,
                        gripper_pose_seq=file['observations']['ee_pose'][:episode_length],
                        use_gripper_crop=True,
                    )
                    
                    # Unpack object and background point clouds
                    aggr_src_pts_ls, aggr_feats_ls, obj_pts_ls, obj_feats_ls, bg_pts_ls, bg_feats_ls, aggr_colors_ls = obj_bg_result
                    print(f"✅ Successfully processed episode {epi_idx} with object/background segmentation")
                    # else:
                    #     print("Warning: Could not segment object/background, using standard d3fields_proc")
                    #     aggr_src_pts_ls, aggr_feats_ls = d3fields_proc(
                    #         fusion=fusion,
                    #         shape_meta=shape_meta['obs'][key],
                    #         color_seq=color_seq,
                    #         depth_seq=depth_seq,
                    #         extri_seq=extri_seq,
                    #         intri_seq=intri_seq,
                    #         robot_base_pose_in_world_seq=robot_base_pose_in_world_seq,
                    #         teleop_robot=kin_helper,
                    #         qpos_seq=qpos_seq,
                    #         expected_labels=expected_labels,
                    #         tool_names=tool_names,
                    #         exclude_colors=exclude_colors,
                    #     )
                    #     obj_pts_ls = aggr_src_pts_ls
                    #     obj_feats_ls = aggr_feats_ls
                else:
                    # Standard d3fields_proc without object/background segmentation
                    aggr_src_pts_ls, aggr_feats_ls, aggr_colors_ls = d3fields_proc(
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
                
                # Process contact field if enabled
                if use_contact_field:
                    # Compute reference tactile from first 5 frames (like in viz script)
                    reference_tactile_left = None
                    reference_tactile_right = None
                    reference_tactile_steps = 5
                    
                    # Check if tactile settings are available
                    has_tactile_left = 'tactile_left' in shape_meta['obs'] or ('tactile_settings' in shape_meta and 'tactile_left' in shape_meta['tactile_settings'])
                    has_tactile_right = 'tactile_right' in shape_meta['obs'] or ('tactile_settings' in shape_meta and 'tactile_right' in shape_meta['tactile_settings'])
                    
                    if has_tactile_left and has_tactile_right and 'tactile' in file['observations']:
                        if 'tactile_img_left' in file['observations']['tactile'] and 'tactile_img_right' in file['observations']['tactile']:
                            # Initialize tactile processors if needed (with scaling enabled for contact field)
                            if 'tactile_left' not in tactile_processors:
                                if 'tactile_left' in shape_meta['obs']:
                                    setting_left = shape_meta['obs']['tactile_left']['setting']
                                else:
                                    setting_left = shape_meta['tactile_settings']['tactile_left']
                                tactile_processors['tactile_left'] = TactileProcessor(
                                    width=320, height=240, marker_config=setting_left, use_gpu=True,
                                    apply_scaling=True,   # Enable scaling for contact field inference
                                    scale_factor=0.15,    # Scale DOWN real-world data to match pre-training
                                    clip_range=(-10.0, 10.0)
                                )
                            if 'tactile_right' not in tactile_processors:
                                if 'tactile_right' in shape_meta['obs']:
                                    setting_right = shape_meta['obs']['tactile_right']['setting']
                                else:
                                    setting_right = shape_meta['tactile_settings']['tactile_right']
                                tactile_processors['tactile_right'] = TactileProcessor(
                                    width=320, height=240, marker_config=setting_right, use_gpu=True,
                                    apply_scaling=True,   # Enable scaling for contact field inference
                                    scale_factor=0.15,    # Scale DOWN real-world data to match pre-training
                                    clip_range=(-10.0, 10.0)
                                )
                            
                            # Process first N frames to compute reference
                            print(f"Computing reference tactile from first {reference_tactile_steps} frames...")
                            left_ref_frames = []
                            right_ref_frames = []
                            n_ref_steps = min(reference_tactile_steps, episode_length)
                            
                            for ref_idx in range(n_ref_steps):
                                tactile_img_left = file['observations']['tactile']['tactile_img_left'][ref_idx]
                                tactile_img_right = file['observations']['tactile']['tactile_img_right'][ref_idx]
                                
                                tactile_ff_left = tactile_processors['tactile_left'].process_frame(tactile_img_left)
                                tactile_ff_right = tactile_processors['tactile_right'].process_frame(tactile_img_right)
                                
                                left_ref_frames.append(tactile_ff_left)
                                right_ref_frames.append(tactile_ff_right)
                            
                            # Compute median as reference
                            if left_ref_frames and right_ref_frames:
                                reference_tactile_left = np.median(np.stack(left_ref_frames, axis=0), axis=0)  # (7, 9, 3)
                                reference_tactile_right = np.median(np.stack(right_ref_frames, axis=0), axis=0)  # (7, 9, 3)
                                print(f"✅ Reference tactile computed from {n_ref_steps} frames")
                    
                    # Process each timestep to add contact field data
                    contact_field_pts_ls = []
                    for t_idx in range(len(aggr_src_pts_ls)):
                        # Get object point cloud for this timestep
                        obj_pcd = obj_pts_ls[t_idx] if t_idx < len(obj_pts_ls) else np.zeros((0, 3))
                        full_pcd = aggr_src_pts_ls[t_idx]  # Full point cloud (N_total, 3 or 3+C)
                        
                        if obj_pcd.shape[0] > 0 and has_tactile_left and has_tactile_right:
                            
                            if has_tactile_left and has_tactile_right:
                                # Get tactile image frames (raw images, not force fields)
                                tactile_img_left = None
                                tactile_img_right = None
                                
                                # Try to get tactile images from observations
                                if 'tactile' in file['observations']:
                                    if 'tactile_img_left' in file['observations']['tactile'] and t_idx < episode_length:
                                        tactile_img_left = file['observations']['tactile']['tactile_img_left'][t_idx]
                                    if 'tactile_img_right' in file['observations']['tactile'] and t_idx < episode_length:
                                        tactile_img_right = file['observations']['tactile']['tactile_img_right'][t_idx]
                                
                                if tactile_img_left is not None and tactile_img_right is not None:
                                    # Initialize tactile processors if not already done (with scaling for contact field)
                                    if 'tactile_left' not in tactile_processors:
                                        # Get settings from either obs or tactile_settings
                                        if 'tactile_left' in shape_meta['obs']:
                                            setting_left = shape_meta['obs']['tactile_left']['setting']
                                        else:
                                            setting_left = shape_meta['tactile_settings']['tactile_left']
                                        tactile_processors['tactile_left'] = TactileProcessor(
                                            width=320, height=240, marker_config=setting_left, use_gpu=True,
                                            apply_scaling=True,   # Enable scaling for contact field inference
                                            scale_factor=0.15,    # Scale DOWN real-world data to match pre-training
                                            clip_range=(-10.0, 10.0)
                                        )
                                    if 'tactile_right' not in tactile_processors:
                                        # Get settings from either obs or tactile_settings
                                        if 'tactile_right' in shape_meta['obs']:
                                            setting_right = shape_meta['obs']['tactile_right']['setting']
                                        else:
                                            setting_right = shape_meta['tactile_settings']['tactile_right']
                                        tactile_processors['tactile_right'] = TactileProcessor(
                                            width=320, height=240, marker_config=setting_right, use_gpu=True,
                                            apply_scaling=True,   # Enable scaling for contact field inference
                                            scale_factor=0.15,    # Scale DOWN real-world data to match pre-training
                                            clip_range=(-10.0, 10.0)
                                        )
                                    
                                    # Process raw tactile images to get force field data
                                    tactile_ff_left = tactile_processors['tactile_left'].process_frame(tactile_img_left)
                                    tactile_ff_right = tactile_processors['tactile_right'].process_frame(tactile_img_right)
                                    
                                    # Contact field model expects 6 channels (current + reference)
                                    # Concatenate current with pre-computed reference
                                    if reference_tactile_left is not None and reference_tactile_right is not None:
                                        tactile_ff_left = np.concatenate([tactile_ff_left, reference_tactile_left], axis=-1)  # (7, 9, 6)
                                        tactile_ff_right = np.concatenate([tactile_ff_right, reference_tactile_right], axis=-1)  # (7, 9, 6)
                                    else:
                                        # Fallback: use current tactile as reference if reference not available
                                        print("⚠️ Warning: Reference tactile not computed, using current twice")
                                        tactile_ff_left = np.concatenate([tactile_ff_left, tactile_ff_left], axis=-1)  # (7, 9, 6)
                                        tactile_ff_right = np.concatenate([tactile_ff_right, tactile_ff_right], axis=-1)  # (7, 9, 6)
                                    
                                    # Get ee_pose and transform for contact field model
                                    ee_pose_8d = file['observations']['ee_pose'][t_idx]
                                    ee_pos = ee_pose_8d[:3]
                                    ee_quat = ee_pose_8d[3:7]
                                    gripper_pos = ee_pose_8d[7] if len(ee_pose_8d) > 7 else 0.05
                                    
                                    # Transform the pose for contact field model (z-translation + z-rotation)
                                    transformed_pos = ee_pos.copy()
                                    transformed_pos[2] -= 0.14
                                    current_rot = st.Rotation.from_quat(ee_quat)
                                    z_rotation = st.Rotation.from_euler('z', np.pi/4)
                                    transformed_rot = z_rotation * current_rot
                                    transformed_quat = transformed_rot.as_quat()
                                    ee_pose_7d = np.concatenate([transformed_pos, transformed_quat])
                                    
                                    # Get 3D marker coordinates
                                    tactile_coord_left, tactile_coord_right = get_tactile_marker_coordinates_for_contact_field(
                                        ee_pose_7d, gripper_pos
                                    )

                                    # Predict contact field on object point cloud
                                    contact_prob, contact_force = predict_contact_field(
                                        model=contact_field_model,
                                        obj_pointcloud=obj_pcd[:, :3],  # Only xyz coordinates
                                        tactile_data_left=tactile_ff_left,
                                        tactile_data_right=tactile_ff_right,
                                        tactile_coord_left=tactile_coord_left,
                                        tactile_coord_right=tactile_coord_right,
                                        ee_pose=ee_pose_7d,
                                        device=contact_field_device
                                    )
                                    
                                    # Create contact field data (N_obj, 4): [contact_prob, fx, fy, fz]
                                    contact_field_data = np.concatenate([contact_prob, contact_force], axis=-1).astype(np.float32)  # (N_obj, 4)
                                    
                                    # Since full_pcd = [obj_pcd, bg_pcd], we can directly assign contact field
                                    # Object points are first N_obj points, background points are the rest
                                    N_obj = obj_pcd.shape[0]
                                    N_bg = full_pcd.shape[0] - N_obj
                                    
                                    # Create contact field for object points (N_obj, 4)
                                    obj_contact_field = contact_field_data
                                    
                                    # Create zeros for background points (N_bg, 4) - explicitly float32
                                    bg_contact_field = np.zeros((N_bg, 4), dtype=np.float32)
                                    
                                    # Concatenate: [obj_contact_field, bg_contact_field]
                                    full_contact_field = np.concatenate([obj_contact_field, bg_contact_field], axis=0)
                                    
                                    # Concatenate contact field to point cloud features
                                    pcd_with_contact = np.concatenate([full_pcd, full_contact_field], axis=-1).astype(np.float32)
                                    contact_field_pts_ls.append(pcd_with_contact)
                                else:
                                    # No tactile data, pad with zeros (explicitly float32)
                                    zeros_contact = np.zeros((full_pcd.shape[0], 4), dtype=np.float32)
                                    pcd_with_contact = np.concatenate([full_pcd, zeros_contact], axis=-1).astype(np.float32)
                                    contact_field_pts_ls.append(pcd_with_contact)
                            else:
                                # No tactile keys defined, pad with zeros (explicitly float32)
                                zeros_contact = np.zeros((full_pcd.shape[0], 4), dtype=np.float32)
                                pcd_with_contact = np.concatenate([full_pcd, zeros_contact], axis=-1).astype(np.float32)
                                contact_field_pts_ls.append(pcd_with_contact)
                        else:
                            # No object point cloud, pad with zeros (explicitly float32)
                            zeros_contact = np.zeros((full_pcd.shape[0], 4), dtype=np.float32)
                            pcd_with_contact = np.concatenate([full_pcd, zeros_contact], axis=-1).astype(np.float32)
                            contact_field_pts_ls.append(pcd_with_contact)
                    
                    # Replace aggr_src_pts_ls with contact field enhanced version
                    aggr_src_pts_ls = contact_field_pts_ls

                if distill_dino:
                    for pts_idx, aggr_src_pts in enumerate(aggr_src_pts_ls):
                        if use_contact_field:
                            # Extract contact field channels (last 4 channels)
                            contact_channels = aggr_src_pts[:, -4:]
                            # Concatenate: [xyz, dino_feats, rgb (if enabled), contact_field]
                            parts_to_concat = [
                                aggr_src_pts[:, :3],  # xyz
                                aggr_feats_ls[pts_idx],  # dino features
                            ]
                            # Add RGB channels if enabled
                            if aggr_colors_ls[pts_idx] is not None:
                                parts_to_concat.append(aggr_colors_ls[pts_idx])  # RGB channels
                            parts_to_concat.append(contact_channels)  # contact field
                            aggr_src_pts_ls[pts_idx] = np.concatenate(parts_to_concat, axis=-1)
                        else:
                            # Concatenate: [xyz, dino_feats, rgb (if enabled)]
                            parts_to_concat = [aggr_src_pts, aggr_feats_ls[pts_idx]]
                            if aggr_colors_ls[pts_idx] is not None:
                                parts_to_concat.append(aggr_colors_ls[pts_idx])  # RGB channels
                            aggr_src_pts_ls[pts_idx] = np.concatenate(parts_to_concat, axis=-1)
                elif use_contact_field:
                    # distill_dino=False but contact_field=True
                    # aggr_src_pts_ls already contains [xyz, contact_field] from contact field processing
                    pass  # Keep as is

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
                # key is like "tactile_left_force_field" or "tactile_right_force_field"
                # Extract base name: "tactile_left_force_field" -> "tactile_left"
                base_key = key.replace('_force_field', '')
                # Map to image key: "tactile_left" -> "tactile_img_left"
                tactile_img_key = base_key.replace('tactile_', 'tactile_img_')
                
                # Check if the key exists in observations
                if 'tactile' not in file['observations'] or tactile_img_key not in file['observations']['tactile']:
                    print(f"Warning: {tactile_img_key} not found in observations, skipping...")
                    continue
                
                frames = file['observations']['tactile'][tactile_img_key][:episode_length]
                
                # Get settings from either obs or tactile_settings
                if base_key in shape_meta.get('tactile_settings', {}):
                    setting = shape_meta['tactile_settings'][base_key]
                elif key in shape_meta['obs']:
                    setting = shape_meta['obs'][key].get('setting', None)
                else:
                    print(f"Warning: Settings for {base_key} not found in shape_meta, skipping...")
                    continue
                
                # Get robot pose data for 3D marker coordinate computation
                ee_poses = file['observations']['ee_pose'][:episode_length]  # (T, 8) [x,y,z,qx,qy,qz,qw,gripper]
                
                # Initialize TactileProcessor for this key if not already done
                # IMPORTANT: Use same preprocessing as contact field prediction for consistency
                if key not in tactile_processors:
                    tactile_processors[key] = TactileProcessor(
                        width=320,
                        height=240,
                        marker_config=setting,
                        use_gpu=True,
                        apply_scaling=True,       # Match contact field preprocessing
                        scale_factor=0.15,        # Scale DOWN real-world data to match pre-training
                        clip_range=(-10.0, 10.0)  # Final clip range after scaling
                    )
                
                # Compute reference tactile from first 5 frames
                reference_tactile = None
                reference_tactile_steps = 5
                n_ref_steps = min(reference_tactile_steps, len(frames))
                
                if n_ref_steps > 0:
                    print(f"Computing reference tactile for {key} from first {n_ref_steps} frames...")
                    ref_frames = []
                    for ref_idx in range(n_ref_steps):
                        ref_ff = tactile_processors[key].process_frame(frames[ref_idx])
                        ref_frames.append(ref_ff)
                    reference_tactile = np.median(np.stack(ref_frames, axis=0), axis=0)  # (7, 9, 3)
                    print(f"✅ Reference tactile computed with shape {reference_tactile.shape}")
                
                # Process frames using TactileProcessor to get force field data
                force_fields = []
                tactile_coords = []
                
                for frame_idx, frame in enumerate(frames):
                    # Get force field using TactileProcessor.process_frame
                    # This returns force_field (7, 9, 3) with [depth, dy, dx]
                    force_field = tactile_processors[key].process_frame(frame)  # (7, 9, 3)
                    
                    # Concatenate with reference to get 6D data
                    if reference_tactile is not None:
                        force_field = np.concatenate([force_field, reference_tactile], axis=-1)  # (7, 9, 6)
                    else:
                        # Fallback: duplicate current if no reference available
                        force_field = np.concatenate([force_field, force_field], axis=-1)  # (7, 9, 6)
                    
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
                force_field_data = np.stack(force_fields, axis=0)  # (T, 7, 9, 6) - 6D with reference
                tactile_coord_data = np.stack(tactile_coords, axis=0)  # (T, 7, 9, 3)
                
                print(f'{key} force field shape:', force_field_data.shape)
                print(f'{key} tactile coord shape:', tactile_coord_data.shape)
                
                # Store force field data - key already has "_force_field" suffix
                if key not in tactile_data_dict:
                    tactile_data_dict[key] = list()
                tactile_data_dict[key].append(force_field_data)
                
                # Store 3D marker coordinates with "_coord" suffix (use base_key)
                coord_key = f"{base_key}_coord"
                if coord_key not in tactile_coord_dict:
                    tactile_coord_dict[coord_key] = list()
                tactile_coord_dict[coord_key].append(tactile_coord_data)


        if fusion is not None:
            fusion.clear_xmem_memory()
    
    # Update tactile_keys to only contain the actual force_field keys that were created
    # This removes confusion with empty base keys (tactile_left/tactile_right)
    if len(tactile_keys) > 0:
        actual_tactile_keys = [key for key in tactile_data_dict.keys() if key.endswith('_force_field')]
        if len(actual_tactile_keys) > 0:
            print(f"Replacing tactile_keys {tactile_keys} with actual force_field keys {actual_tactile_keys}")
            tactile_keys = actual_tactile_keys
        else:
            print(f"Warning: No tactile force field data was processed, clearing tactile_keys")
            tactile_keys = []
    
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
        if len(data) == 0:
            print(f'Warning: No data for tactile key {key}, skipping...')
            continue
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
        if len(data) == 0:
            print(f'Warning: No data for tactile coordinate key {key}, skipping...')
            continue
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
            exclude_colors=[],
            contact_field_checkpoint_path=None,
            contact_field_device='cuda'
            ):
        
        super().__init__()
        
        rotation_transformer = RotationTransformer(
            from_rep='euler_angles', to_rep=rotation_rep, from_convention='xyz')
        
        # Load contact field model if checkpoint path is provided
        contact_field_model = None
        if contact_field_checkpoint_path is not None:
            print(f"Loading contact field model from {contact_field_checkpoint_path}")
            contact_field_model, contact_field_config = load_contact_field_model_and_config(
                contact_field_checkpoint_path, device=contact_field_device
            )
            print("✅ Contact field model loaded successfully")
        
        replay_buffer = None
        fusion = None
        cache_info_str = ''
        
        # Add contact field to cache string if enabled
        if contact_field_model is not None:
            cache_info_str += '_contact_field'
        
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
            distill_dino = shape_meta['obs']['d3fields']['info'].get('distill_dino', False)
            include_rgb = shape_meta['obs']['d3fields']['info'].get('add_rgb_channels', False)
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
            if include_rgb:
                cache_info_str += '_w_rgb'
            if 'key' in shape_meta['action'] and shape_meta['action']['key'] == 'joint_action':
                cache_info_str += '_joint'
            else:
                cache_info_str += '_eef'
            if 'trim_tail' in shape_meta and shape_meta['trim_tail'] > 0:
                cache_info_str += '_trim'
            # Add delta action info to cache string
            if 'delta' in shape_meta['action'] and shape_meta['action']['delta']:
                cache_info_str += '_delta'
                cache_info_str += f"_act{shape_meta['action']['shape'][0]}"
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
                            contact_field_model=contact_field_model,
                            contact_field_device=contact_field_device,
                            )
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                    except Exception as e:
                        if os.path.exists(cache_zarr_path):
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
                exclude_colors=exclude_colors,
                contact_field_model=contact_field_model,
                contact_field_device=contact_field_device,
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
        
        # First pass: collect base keys from shape_meta
        base_tactile_keys = list()
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
                base_tactile_keys.append(key)
        
        # Second pass: detect actual tactile keys in replay buffer
        # We use _force_field suffixed keys as the actual tactile data
        for key in self.replay_buffer.keys():
            # Add force field keys - these are the actual tactile observations
            if key.endswith('_force_field') and any(tkey in key for tkey in base_tactile_keys):
                tactile_keys.append(key)
            # Add coordinate keys separately
            elif key.endswith('_coord') and any(tkey in key for tkey in base_tactile_keys):
                tactile_coord_keys.append(key)
        
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

        # tactile (force field data has shape (T, 7, 9, 6) - 6D with reference)
        for key in self.tactile_keys:
            B, H, W, C = self.replay_buffer[key].shape  # (T, 7, 9, 6)
            stat = array_to_stats(self.replay_buffer[key][()].reshape(B * H * W, C))
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
        
        # Process tactile data: combine force_field and coord into single tensor
        for key in self.tactile_keys:
            # key is like "tactile_left_force_field" or "tactile_right_force_field"
            # Get corresponding coord key
            coord_key = key.replace('_force_field', '_coord')
            
            # Check if we should use 2D format (based on shape_meta)
            expected_shape = self.shape_meta['obs'][key]['shape']
            use_2d_format = len(expected_shape) == 3  # (C, H, W) for 2D, (C, N) for 1D
            
            if coord_key in sample:
                # Force field: (T, 7, 9, 6) with 6D = [current(3), reference(3)]
                force_field = sample[key][T_slice].astype(np.float32)  # (T, 7, 9, 6)
                T, H, W, C_ff = force_field.shape
                
                # Coordinates: (T, 7, 9, 3)
                coord_data = sample[coord_key][T_slice].astype(np.float32)  # (T, 7, 9, 3)
                
                if use_2d_format:
                    # Keep 2D spatial structure: (T, 7, 9, 6) + (T, 7, 9, 3) -> (T, 9, 7, 9)
                    # Move channel dimension to front: (T, H, W, C) -> (T, C, H, W)
                    force_field = np.moveaxis(force_field, -1, 1)  # (T, 6, 7, 9)
                    coord_data = np.moveaxis(coord_data, -1, 1)  # (T, 3, 7, 9)
                    # Combine: [force_field(6), coordinates(3)] -> (T, 9, 7, 9)
                    obs_dict[key] = np.concatenate([force_field, coord_data], axis=1)  # (T, 9, 7, 9)
                else:
                    # Flatten spatial dimensions: (T, 7, 9, 6) -> (T, 6, 63)
                    force_field = force_field.reshape(T, H*W, C_ff)  # (T, 63, 6)
                    force_field = np.moveaxis(force_field, 1, 2)  # (T, 6, 63)
                    
                    # Coordinates: (T, 7, 9, 3) -> (T, 3, 63)
                    coord_data = coord_data.reshape(T, H*W, 3)  # (T, 63, 3)
                    coord_data = np.moveaxis(coord_data, 1, 2)  # (T, 3, 63)
                    
                    # Combine: [force_field(6), coordinates(3)] -> (T, 9, 63)
                    obs_dict[key] = np.concatenate([force_field, coord_data], axis=1)  # (T, 9, 63)
                
                del sample[key]
                del sample[coord_key]
            else:
                # Fallback: just use force field if coord not available
                force_field = sample[key][T_slice].astype(np.float32)  # (T, 7, 9, 6)
                T, H, W, C_ff = force_field.shape
                
                if use_2d_format:
                    # Keep 2D format: (T, 7, 9, 6) -> (T, 6, 7, 9)
                    obs_dict[key] = np.moveaxis(force_field, -1, 1)  # (T, 6, 7, 9)
                else:
                    # Flatten: (T, 7, 9, 6) -> (T, 6, 63)
                    obs_dict[key] = force_field.reshape(T, C_ff, H*W)  # (T, 6, 63)
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

def get_tactile_marker_coordinates_for_contact_field(ee_pose_7d, gripper_pos):
    """
    Generate tactile marker coordinates based on end-effector pose for contact field model.
    
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
    rotation_matrix = rotation.as_matrix()
    
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
    
    # Transform to world frame
    left_markers_world = np.zeros_like(left_markers_local)
    right_markers_world = np.zeros_like(right_markers_local)
    
    for i in range(cols):
        for j in range(rows):
            # Transform left marker
            left_local = left_markers_local[i, j, :]
            left_world = rotation_matrix @ left_local + ee_pos
            left_markers_world[i, j, :] = left_world
            
            # Transform right marker
            right_local = right_markers_local[i, j, :]
            right_world = rotation_matrix @ right_local + ee_pos
            right_markers_world[i, j, :] = right_world
    
    return left_markers_world, right_markers_world


def load_contact_field_model_and_config(contact_field_checkpoint_path, device='cuda'):
    """
    Load contact field model and configuration from checkpoint.
    
    Args:
        contact_field_checkpoint_path: Path to contact field model checkpoint
        device: Device to load model on
        
    Returns:
        tuple: (model, config) - loaded model and its configuration
    """
    import sys
    from omegaconf import DictConfig, OmegaConf
    from omegaconf.listconfig import ListConfig
    from omegaconf.base import ContainerMetadata
    
    # Add contact_field to Python path for importing models
    contact_field_dir = Path(__file__).parent.parent.parent.parent / 'contact_field'
    if contact_field_dir.exists():
        sys.path.insert(0, str(contact_field_dir))
    
    # Add safe globals for torch.load - include all omegaconf types
    try:
        torch.serialization.add_safe_globals([
            DictConfig, 
            OmegaConf, 
            ListConfig,
            ContainerMetadata
        ])
    except (AttributeError, ImportError):
        pass
    
    checkpoint_path = Path(contact_field_checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Contact field checkpoint not found: {contact_field_checkpoint_path}")
    
    # Load checkpoint with weights_only=False to handle omegaconf objects
    try:
        checkpoint = torch.load(contact_field_checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")
    
    # Load config from checkpoint or find it in checkpoint directory
    config = None
    if 'hyper_parameters' in checkpoint and 'cfg' in checkpoint['hyper_parameters']:
        config = checkpoint['hyper_parameters']['cfg']
        if isinstance(config, DictConfig):
            config = OmegaConf.to_object(config)
    elif 'config' in checkpoint:
        config = checkpoint['config']
        if isinstance(config, DictConfig):
            config = OmegaConf.to_object(config)
    else:
        # Try to find config in checkpoint directory
        for config_name in ['config.yaml', 'config.yml']:
            config_path_candidate = checkpoint_path.parent / config_name
            if config_path_candidate.exists():
                with open(config_path_candidate, 'r') as f:
                    config = yaml.safe_load(f)
                break
    
    if config is None:
        raise FileNotFoundError(f"Could not find configuration for contact field model at {contact_field_checkpoint_path}")
    
    # Import contact field model creation function
    try:
        from models import create_model
    except ImportError:
        # If direct import fails, the path should already be in sys.path from above
        raise ImportError(
            "Failed to import 'models' module. Ensure contact_field directory is accessible. "
            f"Attempted to add {contact_field_dir} to sys.path."
        )
    
    # Create model
    model = create_model(config)
    
    # Load state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Remove module prefixes if present
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('network.'):
            new_key = key[8:]
        elif key.startswith('model.'):
            new_key = key[6:]
        elif key.startswith('module.'):
            new_key = key[7:]
        else:
            new_key = key
        new_state_dict[new_key] = value
    
    # Load weights
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()
    
    print(f"✅ Contact field model loaded from {contact_field_checkpoint_path}")
    
    return model, config


def predict_contact_field(model, obj_pointcloud, tactile_data_left, tactile_data_right, 
                          tactile_coord_left, tactile_coord_right, ee_pose, device='cuda'):
    """
    Predict contact field (contact probability + contact force vector) for object point cloud.
    
    Args:
        model: Contact field model
        obj_pointcloud: Object point cloud (N, 3) numpy array
        tactile_data_left: Left tactile force field data (7, 9, 3) numpy array
        tactile_data_right: Right tactile force field data (7, 9, 3) numpy array
        tactile_coord_left: Left tactile marker coordinates (7, 9, 3) numpy array
        tactile_coord_right: Right tactile marker coordinates (7, 9, 3) numpy array
        ee_pose: End-effector pose (7,) numpy array [x, y, z, qx, qy, qz, qw]
        device: Device for inference
        
    Returns:
        tuple: (contact_prob, contact_force)
            contact_prob: (N, 1) contact probabilities for each point
            contact_force: (N, 3) contact force vectors for each point
    """
    if obj_pointcloud.shape[0] == 0:
        return np.zeros((0, 1)), np.zeros((0, 3))
    
    # Prepare batch data for model
    # Note: Model expects 'point_cloud' not 'object_point_cloud'
    # Tactile data format: Model stacks left/right to get (B, 2, H, W, C) where H=7, W=9, C=6
    # So we need to keep channels LAST, not channels first!
    # Input: (7, 9, 6) -> keep as (7, 9, 6) for (H, W, C) format
    
    batch_data = {
        'point_cloud': torch.from_numpy(obj_pointcloud).float().unsqueeze(0).to(device),  # (1, N, 3)
        'env_point_cloud': None,  # No environment points for contact prediction
        'tactile_data_left': torch.from_numpy(tactile_data_left).float().unsqueeze(0).to(device),  # (1, 7, 9, 6)
        'tactile_data_right': torch.from_numpy(tactile_data_right).float().unsqueeze(0).to(device),  # (1, 7, 9, 6)
        'tactile_coord_left': torch.from_numpy(tactile_coord_left).float().unsqueeze(0).to(device),  # (1, 7, 9, 3)
        'tactile_coord_right': torch.from_numpy(tactile_coord_right).float().unsqueeze(0).to(device),  # (1, 7, 9, 3)
        'ee_pose': torch.from_numpy(ee_pose).float().unsqueeze(0).unsqueeze(0).to(device),  # (1, 1, 7) - add time dimension
        'ee_vel': torch.zeros(1, 1, 6, device=device),  # (1, 1, 6) - zero velocity as placeholder
    }
    
    # Run inference
    with torch.no_grad():
        output = model(batch_data)
        
        if isinstance(output, dict):
            pred_prob = output.get('contact_prob', output.get('prob', None))
            pred_force = output.get('contact_force', output.get('force', None))
        elif isinstance(output, (tuple, list)):
            pred_prob = output[0]
            pred_force = output[1] if len(output) > 1 else None
        else:
            pred_prob = output
            pred_force = None
        
        # Convert to numpy
        if pred_prob is not None:
            contact_prob = pred_prob.squeeze(0).cpu().numpy()  # (N, 1) or (N,)
            if contact_prob.ndim == 1:
                contact_prob = contact_prob[:, None]  # (N, 1)
        else:
            contact_prob = np.zeros((obj_pointcloud.shape[0], 1))
        
        if pred_force is not None:
            contact_force = pred_force.squeeze(0).cpu().numpy()  # (N, 3)
        else:
            contact_force = np.zeros((obj_pointcloud.shape[0], 3))
    
    return contact_prob, contact_force
