#!/usr/bin/env python3
"""
Convert real-world data from HDF5 format to contact field dataset format.

This script converts HDF5 files with tactile images to the format expected by
the contact field training dataset. The output consists of two pickle files per episode:
1. Main file (episode_name.pkl) containing observations with robot states and tactile data
2. Contact file (episode_name_contact.pkl) containing object/env point clouds and contact vectors
3. Wrench plot (episode_name_wrench.png) showing net wrench over time from force estimation

USAGE:
    # Basic conversion
    python convert_real_data.py --hdf5_paths episode_1.hdf5 episode_2.hdf5 --output_dir ./data/real_episodes
    python convert_real_data.py --data_dir /path/to/hdf5_files --output_dir ./data/real_episodes
    
    # With verification (tests loading through ContactFieldDataset)
    python convert_real_data.py --data_dir /path/to/hdf5_files --output_dir ./data/real_episodes --verify
    
    # With semantic segmentation (more robust than color-based)
    python convert_real_data.py --data_dir /path/to/hdf5_files --output_dir ./data/real_episodes --use_semantic_segmentation
    
    # The script will automatically generate wrench plots showing force and moment components over time
"""

import argparse
import os
import sys
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import yaml

import numpy as np
import torch
import cv2
from scipy.spatial.transform import Rotation as R

# Add the gendp path for importing
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gendp.gendp.common.data_utils import load_dict_from_hdf5, d3fields_proc
from gendp.gendp.common.tactile_utils import TactileProcessor
from gendp.gendp.common.kinematics_utils import KinHelper
from gendp.gendp.real_world.real_inference_utils import (
    transform_ee_pose_for_contact_field,
    get_tactile_marker_coordinates,
)
from d3fields.utils.draw_utils import aggr_point_cloud_from_data

# Add the vis_utils for segmentation
# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'gendp', 'tests'))
from gendp.tests.vis_utils import segment_pointcloud_by_color
from d3fields.fusion import Fusion

# Import force estimator
from force_estimator import (
    SimpleForceEstimator, 
    AnalyticalForceEstimator, 
    create_force_estimator,
    estimate_tool_normals,
    smooth_normals,
    plot_wrench_history
)


def transform_to_gripper_frame(points: np.ndarray, ee_pose_7d: np.ndarray) -> np.ndarray:
    """
    Transform points from world frame to gripper frame.
    
    Args:
        points: (N, 3) or (H, W, 3) points in world frame
        ee_pose_7d: (7,) end-effector pose [x, y, z, qx, qy, qz, qw]
        
    Returns:
        Points in gripper frame with same shape as input
    """
    original_shape = points.shape
    points_flat = points.reshape(-1, 3)
    
    # Extract position and quaternion
    pos = ee_pose_7d[:3]
    quat = ee_pose_7d[3:]  # [qx, qy, qz, qw]
    
    # Create rotation matrix from quaternion
    rot = R.from_quat(quat).as_matrix()
    
    # Create world-to-gripper transform (inverse of gripper-to-world)
    # For rotation: R_inv = R^T
    # For translation: t_inv = -R^T @ t
    rot_inv = rot.T
    trans_inv = -rot_inv @ pos
    
    # Transform points: p_gripper = R_inv @ (p_world - t_world)
    # Which is equivalent to: p_gripper = R_inv @ p_world + t_inv
    points_transformed = (rot_inv @ points_flat.T).T + trans_inv
    
    return points_transformed.reshape(original_shape)


# Segmentation parameters (same as in viz_contact_field_with_tactile_images.py)
PURPLE_HUE_RANGE = (230, 240)
PURPLE_SATURATION_RANGE = (50, 200)
PURPLE_VALUE_RANGE = (100, 200)

CUCUMBER_HUE_RANGE = (90, 110)
CUCUMBER_SATURATION_RANGE = (50, 255)
CUCUMBER_VALUE_RANGE = (60, 255)

OBJECT_BOUNDARIES = {
    'x_lower': 0.3,
    'x_upper': 0.6,
    'y_lower': -0.15,
    'y_upper': 0.15,
    'z_lower': 0.01,
    'z_upper': 0.25,
}

ENV_BOUNDARIES = {
    'x_lower': 0.3,
    'x_upper': 0.7,
    'y_lower': -0.2,
    'y_upper': 0.2,
    'z_lower': -0.03,
    'z_upper': 0.1,
}

# Contact detection parameters
CONTACT_Z_THRESHOLD = 0.015  # Contact when object is within this distance to z=0.01 plane
TACTILE_CHANGE_THRESHOLD = 0.001  # Threshold for detecting tactile change (normal force)
CONTACT_FORCE_SCALE = 100  # Scale factor for converting tactile to contact force magnitude
CONTACT_MODE = 'top_k'  # 'single', 'top_k', or 'threshold'
CONTACT_TOP_K = 5  # For 'top_k' mode
CONTACT_DEPTH_THRESHOLD = 0.015  # For 'threshold' mode

# Force estimator parameters
FORCE_ESTIMATOR_METHOD = 'analytical'  # 'simple' or 'analytical'
FORCE_ESTIMATOR_LAMBDA = 0.01  # Regularization weight for analytical estimator
FORCE_ESTIMATOR_EPSILON = 1e-6  # Epsilon for analytical estimator

# Default shape_meta for d3fields processing
shape_meta = {
    'shape': [6, 868],
    'type': 'spatial',
    'info': {
        'reference_frame': 'robot',
        'distill_dino': True,
        'distill_obj': 'scraper',
        'view_keys': ['camera_front', 'camera_left', 'camera_right'],
        'N_gripper': 100,
        'N_obj': 256,
        'N_env': 512,
        'boundaries': {
            'x_lower': 0.3,
            'x_upper': 0.7,
            'y_lower': -0.2,
            'y_upper': 0.2,
            'z_lower': -0.03,
            'z_upper': 0.5,
        },
        'resize_ratio': 0.5
    }
}


def sample_or_pad_pointcloud(pointcloud: np.ndarray, target_size: int = 256) -> np.ndarray:
    """Sample or pad a pointcloud to reach the target size."""
    if len(pointcloud) == 0:
        return np.zeros((target_size, pointcloud.shape[1] if pointcloud.ndim > 1 else 3))
    
    if len(pointcloud) >= target_size:
        indices = np.random.choice(len(pointcloud), target_size, replace=False)
        return pointcloud[indices]
    else:
        # Pad by repeating points
        repeats = (target_size // len(pointcloud)) + 1
        padded = np.tile(pointcloud, (repeats, 1))
        return padded[:target_size]


def compute_contact_depth(obj_points: np.ndarray, z_plane: float = 0.01) -> np.ndarray:
    """
    Compute contact depth as distance to the z=z_plane plane.
    Negative depth means penetration/contact.
    
    Args:
        obj_points: (N, 3) array of object point cloud
        z_plane: Z-coordinate of the contact plane
        
    Returns:
        (N,) array of contact depths
    """
    return z_plane - obj_points[:, 2]


def compute_tactile_torque(tactile_ff: np.ndarray) -> float:
    """
    Compute mean torque w.r.t center of tactile sensor.
    
    Args:
        tactile_ff: (7, 9, 3) tactile force field where channels are [depth, shear_x, shear_y]
    
    Returns:
        Mean torque value
    """
    # Extract force components
    depth = tactile_ff[:, :, 0]
    shear_x = tactile_ff[:, :, 1]
    shear_y = tactile_ff[:, :, 2]
    
    # Sensor dimensions: 7x9 grid
    N, M = tactile_ff.shape[:2]
    center_y, center_x = N / 2.0, M / 2.0
    
    # Create coordinate grids (relative to center)
    y_coords, x_coords = np.meshgrid(np.arange(N) - center_y, 
                                     np.arange(M) - center_x, 
                                     indexing='ij')
    
    # Torque = r x F, where r is position vector from center
    # For 2D case: torque_z = x * F_y - y * F_x
    torque = x_coords * shear_y - y_coords * shear_x
    return np.mean(torque)


def detect_contact_with_tactile(obj_points: np.ndarray, 
                                tactile_force_left: np.ndarray,
                                tactile_force_right: np.ndarray,
                                prev_tactile_left: Optional[np.ndarray] = None,
                                prev_tactile_right: Optional[np.ndarray] = None,
                                z_threshold: float = CONTACT_Z_THRESHOLD,
                                tactile_threshold: float = TACTILE_CHANGE_THRESHOLD,
                                contact_mode: str = 'single',
                                top_k: int = 3,
                                depth_threshold: float = 0.005) -> Tuple[bool, np.ndarray, np.ndarray]:
    """
    Detect contact based on object proximity to ground plane and tactile response.
    
    Args:
        obj_points: (N, 3) object point cloud
        tactile_force_left: (7, 9, 3) left tactile force field (depth, shear_x, shear_y)
        tactile_force_right: (7, 9, 3) right tactile force field
        prev_tactile_left: Reference left tactile force field (reference frame)
        prev_tactile_right: Reference right tactile force field (reference frame)
        z_threshold: Distance threshold to ground plane for contact
        tactile_threshold: Threshold for tactile force/torque change (default: 1.0)
        contact_mode: Detection mode - 'single', 'top_k', or 'threshold'
        top_k: Number of contact points to return (for 'top_k' mode)
        depth_threshold: Depth threshold for 'threshold' mode (distance below z_plane)
        
    Returns:
        Tuple of (has_contact, contact_points, contact_indices)
        - has_contact: bool indicating if any contact detected
        - contact_points: (M, 3) array of contact point coordinates
        - contact_indices: (M,) array of indices in obj_points
    """
    # Handle empty point cloud
    if len(obj_points) == 0:
        return False, np.zeros(3), 0.0
    
    # Check if object is close to the ground plane
    min_z = np.min(obj_points[:, 2])
    is_close = min_z < z_threshold
    
    # Extract shear force components (channels: [depth, shear_x, shear_y])
    shear_x_left = tactile_force_left[:, :, 1]
    shear_y_left = tactile_force_left[:, :, 2]
    shear_x_right = tactile_force_right[:, :, 1]
    shear_y_right = tactile_force_right[:, :, 2]
    
    # Compute current statistics
    mean_shear_x_left = np.mean(shear_x_left)
    mean_shear_y_left = np.mean(shear_y_left)
    mean_shear_x_right = np.mean(shear_x_right)
    mean_shear_y_right = np.mean(shear_y_right)
    
    # Compute torques
    torque_left = compute_tactile_torque(tactile_force_left)
    torque_right = compute_tactile_torque(tactile_force_right)
    
    # Sum of shear forces and torques (current frame)
    current_sum = (np.abs(mean_shear_x_left) + np.abs(mean_shear_y_left) + 
                   np.abs(mean_shear_x_right) + np.abs(mean_shear_y_right) +
                   np.abs(torque_left) + np.abs(torque_right))
    
    # Check for tactile change if reference frame is available
    has_tactile_change = False
    if prev_tactile_left is not None and prev_tactile_right is not None:
        # Compute reference statistics
        ref_shear_x_left = np.mean(prev_tactile_left[:, :, 1])
        ref_shear_y_left = np.mean(prev_tactile_left[:, :, 2])
        ref_shear_x_right = np.mean(prev_tactile_right[:, :, 1])
        ref_shear_y_right = np.mean(prev_tactile_right[:, :, 2])
        
        ref_torque_left = compute_tactile_torque(prev_tactile_left)
        ref_torque_right = compute_tactile_torque(prev_tactile_right)
        
        # Sum of reference shear forces and torques
        ref_sum = (np.abs(ref_shear_x_left) + np.abs(ref_shear_y_left) + 
                   np.abs(ref_shear_x_right) + np.abs(ref_shear_y_right) +
                   np.abs(ref_torque_left) + np.abs(ref_torque_right))
        
        # Check if difference exceeds threshold
        tactile_change = np.abs(current_sum - ref_sum)
        has_tactile_change = tactile_change > tactile_threshold
    else:
        # No reference frame - just check if current sum exceeds threshold
        has_tactile_change = current_sum > tactile_threshold
    
    # Contact occurs when object is close AND tactile shows response
    has_contact = is_close and has_tactile_change
    
    if not has_contact:
        return False, np.zeros((0, 3)), np.array([], dtype=int)
    
    # Find contact points based on mode
    if contact_mode == 'single':
        # Single contact: lowest point
        lowest_idx = np.argmin(obj_points[:, 2])
        contact_points = obj_points[lowest_idx:lowest_idx+1]
        contact_indices = np.array([lowest_idx])
        
    elif contact_mode == 'top_k':
        # Top k closest points to ground plane
        z_coords = obj_points[:, 2]
        sorted_indices = np.argsort(z_coords)
        contact_indices = sorted_indices[:top_k]
        contact_points = obj_points[contact_indices]
        
    elif contact_mode == 'threshold':
        # All points within depth threshold
        z_coords = obj_points[:, 2]
        # z_plane = 0.01  # Ground plane height
        # depths = z_plane - z_coords
        # contact_mask = depths > depth_threshold
        contact_mask = z_coords < depth_threshold
        contact_indices = np.where(contact_mask)[0]
        
        if len(contact_indices) == 0:
            # No points meet threshold, fallback to single lowest point
            lowest_idx = np.argmin(z_coords)
            contact_indices = np.array([lowest_idx])
        
        contact_points = obj_points[contact_indices]
    else:
        raise ValueError(f"Unknown contact_mode: {contact_mode}")
    
    return has_contact, contact_points, contact_indices


def compute_contact_vectors(obj_points: np.ndarray,
                           tactile_force_left: np.ndarray,
                           tactile_force_right: np.ndarray,
                           tactile_coord_left: np.ndarray,
                           tactile_coord_right: np.ndarray,
                           ee_pose_7d: np.ndarray,
                           prev_tactile_left: Optional[np.ndarray] = None,
                           prev_tactile_right: Optional[np.ndarray] = None,
                           z_threshold: float = CONTACT_Z_THRESHOLD,
                           force_estimator: Optional[object] = None,
                           normals: Optional[np.ndarray] = None,
                           prob_weights: Optional[np.ndarray] = None,
                           contact_mode: str = 'single',
                           top_k: int = 3,
                           depth_threshold: float = 0.005) -> np.ndarray:
    """
    Compute contact vectors for points in contact using force estimation.
    
    Contact vector format: [pos_x, pos_y, pos_z, norm_x, norm_y, norm_z, force_magnitude, distance]
    
    Args:
        obj_points: (N, 3) object point cloud in world frame
        tactile_force_left: (7, 9, 3) left tactile force field
        tactile_force_right: (7, 9, 3) right tactile force field
        tactile_coord_left: (7, 9, 3) left tactile marker coordinates in world frame
        tactile_coord_right: (7, 9, 3) right tactile marker coordinates in world frame
        ee_pose_7d: (7,) end-effector pose [x, y, z, qx, qy, qz, qw]
        prev_tactile_left: Previous left tactile for change detection
        prev_tactile_right: Previous right tactile for change detection
        z_threshold: Distance threshold for contact
        force_estimator: Force estimator instance (SimpleForceEstimator or AnalyticalForceEstimator)
        normals: (N, 3) array of surface normals (required for AnalyticalForceEstimator)
        prob_weights: (N,) array of contact probabilities (optional for AnalyticalForceEstimator)
        contact_mode: Detection mode - 'single', 'top_k', or 'threshold'
        top_k: Number of contact points to return (for 'top_k' mode)
        depth_threshold: Depth threshold for 'threshold' mode
        
    Returns:
        (M, 8) array of contact vectors where M is number of contact points
    """
    has_contact, contact_points, contact_indices = detect_contact_with_tactile(
        obj_points, tactile_force_left, tactile_force_right,
        prev_tactile_left, prev_tactile_right, z_threshold,
        contact_mode=contact_mode, top_k=top_k, depth_threshold=depth_threshold
    )
    
    if not has_contact:
        return np.array([]).reshape(0, 8), None
    
    # Transform coordinates from world frame to gripper frame
    tactile_coord_left_gripper = transform_to_gripper_frame(tactile_coord_left, ee_pose_7d)
    tactile_coord_right_gripper = transform_to_gripper_frame(tactile_coord_right, ee_pose_7d)
    contact_points_gripper = transform_to_gripper_frame(contact_points, ee_pose_7d)
    
    # Use force estimator to compute contact force vectors
    if force_estimator is not None:
        
        # Check if using AnalyticalForceEstimator
        if isinstance(force_estimator, AnalyticalForceEstimator):
            # For analytical estimator, we need normals for each contact point
            if normals is None:
                contact_normals_input = np.tile([0.0, 0.0, 1.0], (len(contact_indices), 1))
            else:
                # Get normals for contact points using their indices
                if len(normals) > 0:
                    contact_normals_input = normals[contact_indices]  # (M, 3) in gripper frame
                else:
                    contact_normals_input = np.tile([0.0, 0.0, 1.0], (len(contact_indices), 1))
            
            # Get prob_weights for contact points if available
            contact_prob_weights = None
            if prob_weights is not None and len(prob_weights) > 0:
                contact_prob_weights = prob_weights[contact_indices]

            # Scale tactile forces into calibrated unit (N)
            DEPTH_SCALE = 100
            SHEAR_SCALE = 20
            tactile_force_left_scaled = (tactile_force_left - prev_tactile_left) * np.array([DEPTH_SCALE, SHEAR_SCALE, SHEAR_SCALE])
            tactile_force_right_scaled = (tactile_force_right - prev_tactile_right) * np.array([DEPTH_SCALE, SHEAR_SCALE, SHEAR_SCALE])

            # Compute with analytical estimator
            wrench, contact_forces = force_estimator.compute_contact_forces_from_tactile(
                tactile_force_left_scaled,
                tactile_force_right_scaled,
                tactile_coord_left_gripper,
                tactile_coord_right_gripper,
                contact_points_gripper,  # (M, 3)
                normals=contact_normals_input,
                prob_weights=contact_prob_weights
            )
        else:
            # Scale tactile forces into calibrated unit (N)
            DEPTH_SCALE = 100
            SHEAR_SCALE = 20
            tactile_force_left_scaled = (tactile_force_left - prev_tactile_left) * np.array([DEPTH_SCALE, SHEAR_SCALE, SHEAR_SCALE])
            tactile_force_right_scaled = (tactile_force_right - prev_tactile_right) * np.array([DEPTH_SCALE, SHEAR_SCALE, SHEAR_SCALE])

            # Simple estimator
            wrench, contact_forces = force_estimator.compute_contact_forces_from_tactile(
                tactile_force_left_scaled,
                tactile_force_right_scaled,
                tactile_coord_left_gripper,
                tactile_coord_right_gripper,
                contact_points_gripper  # (M, 3)
            )
        
        # Process each contact point
        contact_vectors = []
        for i, (contact_pt, force_vec) in enumerate(zip(contact_points, contact_forces)):
            force_magnitude = np.linalg.norm(force_vec)
            
            # Normalize to get force direction (contact normal)
            if force_magnitude > 1e-6:
                contact_normal = force_vec / force_magnitude
            else:
                # Fallback to upward normal if force is too small
                contact_normal = np.array([0.0, 0.0, 1.0])
            
            # Distance to ground plane
            distance = 0.01 - contact_pt[2]
            
            # Build contact vector: [pos_x, pos_y, pos_z, norm_x, norm_y, norm_z, force, distance]
            contact_vector = np.concatenate([
                contact_pt,  # Position (3)
                contact_normal,  # Normal direction (3)
                [force_magnitude],  # Force magnitude (1)
                [distance]  # Distance/penetration depth (1)
            ])
            contact_vectors.append(contact_vector)
        
        contact_vectors = np.array(contact_vectors)  # (M, 8)
    else:
        # Fallback: assume upward force for all contact points
        contact_vectors = []
        for contact_pt in contact_points:
            contact_normal = np.array([0.0, 0.0, 1.0])
            force_magnitude = 0.0
            distance = 0.01 - contact_pt[2]
            
            contact_vector = np.concatenate([
                contact_pt,
                contact_normal,
                [force_magnitude],
                [distance]
            ])
            contact_vectors.append(contact_vector)
        
        contact_vectors = np.array(contact_vectors)  # (M, 8)
    
    # Return both contact vectors and wrench (if force_estimator was used)
    if force_estimator is not None:
        return contact_vectors, wrench
    else:
        return contact_vectors, None


def convert_hdf5_to_dataset(hdf5_path: str,
                            output_dir: str,
                            tactile_processor_left: TactileProcessor,
                            tactile_processor_right: TactileProcessor,
                            use_semantic_segmentation: bool = False,
                            num_object_points: int = 256,
                            num_env_points: int = 256,
                            max_steps: Optional[int] = None,
                            fusion: Optional[Fusion] = None,
                            kin_helper: Optional[KinHelper] = None,
                            include_front_rgb: bool = True,
                            force_estimator_config: Optional[Dict] = None) -> Tuple[str, str]:
    """
    Convert a single HDF5 file to dataset format.
    
    Args:
        hdf5_path: Path to input HDF5 file
        output_dir: Output directory for converted files
        tactile_processor_left: TactileProcessor for left gripper
        tactile_processor_right: TactileProcessor for right gripper
        use_semantic_segmentation: Whether to use semantic segmentation
        num_object_points: Number of points for object point cloud
        num_env_points: Number of points for environment point cloud
        max_steps: Maximum number of steps to process
        fusion: D3Fields Fusion object for semantic segmentation
        kin_helper: Kinematics helper for robot state
        include_front_rgb: Whether to include front RGB camera image in the output (default: True)
        force_estimator_config: Dictionary with force estimator settings:
            - 'method': 'simple' or 'analytical'
            - 'lambda_reg': regularization weight for analytical
            - 'epsilon': epsilon for analytical
        
    Returns:
        Tuple of (main_file_path, contact_file_path)
    """
    print(f"\nProcessing {hdf5_path}...")
    
    # Load HDF5 data
    data_dict, _ = load_dict_from_hdf5(hdf5_path)
    observations = data_dict['observations']
    
    # Get episode name from filename
    episode_name = Path(hdf5_path).stem
    
    # Detect available cameras
    available_cameras = []
    for cam in ['camera_front', 'camera_left', 'camera_right']:
        if f'{cam}_color' in observations['images']:
            available_cameras.append(cam)
    
    if not available_cameras:
        raise ValueError("No valid cameras found in the data")
    
    print(f"Available cameras: {available_cameras}")
    
    # Get robot poses
    robot_base_in_world_seq = observations.get('robot_base_pose_in_world', None)
    
    # Get tactile images
    tactile_images_left = observations['tactile']['tactile_img_left']
    tactile_images_right = observations['tactile']['tactile_img_right']
    
    # Determine number of steps
    num_steps = len(tactile_images_left)
    if max_steps is not None:
        num_steps = min(num_steps, max_steps)
    
    print(f"Processing {num_steps} steps...")
    
    # Initialize storage
    observations_list = []
    object_point_clouds = []
    env_point_clouds = []
    contact_vectors_list = []
    wrench_history = []  # Track wrench over time
    
    # Point cloud boundaries for filtering
    boundaries = {
        'x_lower': 0.2,
        'x_upper': 0.7,
        'y_lower': -0.2,
        'y_upper': 0.2,
        'z_lower': -0.1,
        'z_upper': 0.5,
    }
    
    # Previous tactile for change detection
    prev_tactile_left = None
    prev_tactile_right = None
    
    # NOTE: Reference tactile computation is handled by dataset.py, not here
    # We only store raw tactile data in pickle files
    
    # Process each timestep
    for step_idx in tqdm(range(num_steps), desc=f"Converting {episode_name}"):
        # === Process Tactile Data ===
        left_img = tactile_images_left[step_idx]
        right_img = tactile_images_right[step_idx]
        
        # Resize tactile images to (60, 80, 3) for storage
        # Note: Original images are kept at full resolution for TactileProcessor
        left_img_resized = cv2.resize(left_img, (80, 60))  # (width, height) -> (60, 80, 3)
        right_img_resized = cv2.resize(right_img, (80, 60))
        
        # Normalize to 0-1 range if they're uint8 (0-255)
        if left_img_resized.dtype == np.uint8:
            left_img_resized = left_img_resized.astype(np.float32) / 255.0
        if right_img_resized.dtype == np.uint8:
            right_img_resized = right_img_resized.astype(np.float32) / 255.0
        
        # Process tactile force fields
        force_field_left = tactile_processor_left.process_frame(left_img)  # (7, 9, 3)
        force_field_right = tactile_processor_right.process_frame(right_img)  # (7, 9, 3)
        
        # Scale to match simulation data format (raw ~0-1000 -> sim scale ~0-1)
        # Dataset loader will later apply scale_factor: 1000.0 to normalize
        force_field_left = force_field_left * 0.001
        force_field_right = force_field_right * 0.001
        
        # === Get RGB Image (front camera) ===
        # Extract front camera RGB image for visualization (only if requested)
        front_rgb = None
        if include_front_rgb and 'camera_front_color' in observations['images']:
            front_rgb = observations['images']['camera_front_color'][step_idx]
            # Normalize to 0-1 range if it's uint8 (0-255)
            if front_rgb.dtype == np.uint8:
                front_rgb = front_rgb.astype(np.float32) / 255.0
        
        # === Process Point Cloud ===
        # Collect data from all cameras
        all_colors = []
        all_depths = []
        all_intrinsics = []
        all_extrinsics = []
        
        for cam in available_cameras:
            all_colors.append(observations['images'][f'{cam}_color'][step_idx])
            all_depths.append(observations['images'][f'{cam}_depth'][step_idx] / 1000.0)
            all_intrinsics.append(observations['images'][f'{cam}_intrinsics'][step_idx])
            all_extrinsics.append(observations['images'][f'{cam}_extrinsics'][step_idx])
        
        colors = np.stack(all_colors)
        depths = np.stack(all_depths)
        intrinsics = np.stack(all_intrinsics)
        extrinsics = np.stack(all_extrinsics)
        
        # Generate point cloud
        pcd, pcd_colors = aggr_point_cloud_from_data(
            colors, depths, intrinsics, extrinsics,
            downsample=False,
            out_o3d=False,
            boundaries=boundaries
        )
        
        # Transform to robot base frame
        if robot_base_in_world_seq is not None:
            robot_base_in_world = robot_base_in_world_seq[step_idx]
            pcd = np.linalg.inv(robot_base_in_world) @ np.concatenate([pcd, np.ones((pcd.shape[0], 1))], axis=-1).T
            pcd = pcd.T[:, :3]
        
        # Segment point cloud
        if use_semantic_segmentation and fusion is not None and kin_helper is not None:
            # Use semantic segmentation
            gripper_crop_params = {
                'tool_length': 0.15,
                'tool_width': 0.15,
                'gripper_finger_length': 0.1,
                'safety_margin': 0.0,
                'global_z_threshold': 0.01
            }
            
            result = d3fields_proc(
                fusion=fusion,
                shape_meta=shape_meta,
                color_seq=colors[None],
                depth_seq=depths[None],
                extri_seq=extrinsics[None],
                intri_seq=intrinsics[None],
                robot_base_pose_in_world_seq=robot_base_in_world_seq[()],
                teleop_robot=kin_helper,
                qpos_seq=observations['full_joint_pos'][step_idx:step_idx+1],
                exclude_threshold=0.01,
                use_obj_bg_seg=True,
                gripper_pose_seq=observations['ee_pose'][step_idx:step_idx+1],
                use_gripper_crop=True,
                gripper_crop_params=gripper_crop_params
            )
            
            if len(result) == 7:
                _, _, obj_pts_ls, obj_feats_ls, bg_pts_ls, bg_feats_ls, rgb = result
                object_pointcloud = obj_pts_ls[0] if len(obj_pts_ls) > 0 else np.zeros((0, 3))
                bg_points = bg_pts_ls[0] if len(bg_pts_ls) > 0 else np.zeros((0, 3))
                
                object_pointcloud = object_pointcloud[object_pointcloud[:, 2] > 0.01]
                env_pointcloud = bg_points[bg_points[:, 2] < 0.01]
            else:
                # Fallback to color segmentation
                object_pointcloud, env_pointcloud, _, _ = segment_pointcloud_by_color(
                    pcd, pcd_colors,
                    obj_hue_range=PURPLE_HUE_RANGE,
                    obj_saturation_range=PURPLE_SATURATION_RANGE,
                    obj_value_range=PURPLE_VALUE_RANGE,
                    env_hue_range=CUCUMBER_HUE_RANGE,
                    env_saturation_range=CUCUMBER_SATURATION_RANGE,
                    env_value_range=CUCUMBER_VALUE_RANGE,
                    object_boundaries=OBJECT_BOUNDARIES,
                    env_boundaries=ENV_BOUNDARIES
                )
        else:
            # Use color-based segmentation
            object_pointcloud, env_pointcloud, _, _ = segment_pointcloud_by_color(
                pcd, pcd_colors,
                obj_hue_range=PURPLE_HUE_RANGE,
                obj_saturation_range=PURPLE_SATURATION_RANGE,
                obj_value_range=PURPLE_VALUE_RANGE,
                env_hue_range=CUCUMBER_HUE_RANGE,
                env_saturation_range=CUCUMBER_SATURATION_RANGE,
                env_value_range=CUCUMBER_VALUE_RANGE,
                object_boundaries=OBJECT_BOUNDARIES,
                env_boundaries=ENV_BOUNDARIES
            )
        
        # === Compute Contact Information ===
        # Compute contact depth for each object point
        contact_depths = compute_contact_depth(object_pointcloud, z_plane=0.01)
        
        # Add contact depth as 4th dimension to object points
        obj_pcd_with_depth = np.concatenate([
            object_pointcloud,
            contact_depths.reshape(-1, 1)
        ], axis=1)  # Shape: (N, 4)
        
        # Sample/pad to target size
        obj_pcd_sampled = sample_or_pad_pointcloud(obj_pcd_with_depth, num_object_points)
        env_pcd_sampled = sample_or_pad_pointcloud(env_pointcloud, num_env_points)
        
        # Add dummy 4th dimension (contact depth = 0) to env points for consistency
        if env_pcd_sampled.shape[1] == 3:
            env_pcd_sampled = np.concatenate([
                env_pcd_sampled,
                np.zeros((env_pcd_sampled.shape[0], 1))
            ], axis=1)
        
        # === Process Robot State (must happen before contact vectors) ===
        # Get end-effector pose
        if 'ee_pose' in observations and step_idx < len(observations['ee_pose']):
            ee_pose_raw = observations['ee_pose'][step_idx][:-1]  # Exclude gripper pos
            
            if len(ee_pose_raw) == 6:
                # Convert from xyzrpy to position + quaternion
                ee_pos = ee_pose_raw[:3]
                ee_rpy = ee_pose_raw[3:6]
                transformed_pose = transform_ee_pose_for_contact_field(ee_pos, ee_rpy)
                ee_pose_tensor = torch.from_numpy(transformed_pose).float()
            else:
                print(f"Warning: Unexpected ee_pose format at step {step_idx}")
                ee_pose_tensor = torch.zeros(7).float()
        else:
            ee_pose_tensor = torch.zeros(7).float()
        
        # Get end-effector velocity
        if 'ee_vel' in observations and step_idx < len(observations['ee_vel']):
            ee_vel = observations['ee_vel'][step_idx]
            if len(ee_vel) == 6:
                ee_vel_tensor = torch.from_numpy(ee_vel).float()
            else:
                ee_vel_tensor = torch.zeros(6).float()
        else:
            # Compute from pose differences
            if step_idx > 0 and 'ee_pose' in observations:
                curr_pose = observations['ee_pose'][step_idx]
                prev_pose = observations['ee_pose'][step_idx - 1]
                
                if len(curr_pose) >= 6 and len(prev_pose) >= 6:
                    lin_vel = (curr_pose[:3] - prev_pose[:3]) * 10.0
                    ang_vel = (curr_pose[3:6] - prev_pose[3:6]) * 10.0
                    ee_vel_tensor = torch.from_numpy(np.concatenate([lin_vel, ang_vel])).float()
                else:
                    ee_vel_tensor = torch.zeros(6).float()
            else:
                ee_vel_tensor = torch.zeros(6).float()
        
        # Get gripper position for tactile coordinates
        if 'ee_pose' in observations and step_idx < len(observations['ee_pose']):
            ee_pose_data = observations['ee_pose'][step_idx]
            if len(ee_pose_data) >= 7:
                gripper_pos = ee_pose_data[-1]
            else:
                gripper_pos = 0.08
        else:
            gripper_pos = 0.08
        
        # Generate tactile marker coordinates (needed for contact vector computation)
        ee_pose_7d = ee_pose_tensor.numpy()
        tactile_coord_left, tactile_coord_right = get_tactile_marker_coordinates(ee_pose_7d, gripper_pos)
        
        # === Compute Contact Information ===
        # Check if object is close to ground plane for reference frame management
        min_z = np.min(object_pointcloud[:, 2]) if len(object_pointcloud) > 0 else float('inf')
        is_close = min_z < CONTACT_Z_THRESHOLD
        
        # Update reference frame when object is not close to ground plane
        if not is_close:
            prev_tactile_left = force_field_left.copy()
            prev_tactile_right = force_field_right.copy()
        
        # Extract normals from object point cloud if using analytical estimator
        obj_normals = None
        
        # Get force estimator configuration
        if force_estimator_config is None:
            force_estimator_config = {
                'method': FORCE_ESTIMATOR_METHOD,
                'lambda_reg': FORCE_ESTIMATOR_LAMBDA,
                'epsilon': FORCE_ESTIMATOR_EPSILON
            }
        
        force_estimator_method = force_estimator_config.get('method', FORCE_ESTIMATOR_METHOD)
        force_estimator_lambda = force_estimator_config.get('lambda_reg', FORCE_ESTIMATOR_LAMBDA)
        force_estimator_epsilon = force_estimator_config.get('epsilon', FORCE_ESTIMATOR_EPSILON)
        
        if force_estimator_method == 'analytical' and len(object_pointcloud) > 0:
            # Transform object points to gripper frame for normal estimation
            obj_points_gripper = transform_to_gripper_frame(object_pointcloud, ee_pose_7d)
            
            # Estimate normals (inward-pointing for force solver)
            obj_normals = estimate_tool_normals(
                obj_points_gripper, 
                camera_location=np.array([0.0, 0.0, 0.15]),  # Camera roughly 5cm above gripper center
                inward=True
            )
            
            # Apply smoothing to reduce RealSense noise
            obj_normals = smooth_normals(obj_points_gripper, obj_normals, iterations=2)
        
        # Compute contact vectors with force estimation
        force_estimator = create_force_estimator(
            method=force_estimator_method,
            lambda_reg=force_estimator_lambda,
            epsilon=force_estimator_epsilon
        )
        
        contact_vecs, wrench = compute_contact_vectors(
            object_pointcloud,
            force_field_left,
            force_field_right,
            tactile_coord_left,
            tactile_coord_right,
            ee_pose_7d,
            prev_tactile_left,
            prev_tactile_right,
            z_threshold=CONTACT_Z_THRESHOLD,
            force_estimator=force_estimator,
            contact_mode=CONTACT_MODE,
            top_k=CONTACT_TOP_K,
            depth_threshold=CONTACT_DEPTH_THRESHOLD,
            normals=obj_normals,  # Pass extracted normals for analytical estimator
            prob_weights=None  # TODO: Extract contact probabilities if available
        )
        
        # Store wrench for plotting
        if wrench is not None:
            wrench_history.append(wrench)
        
        # Convert force fields to tensors (store RAW data - dataset.py will handle reference processing)
        force_field_left_tensor = torch.from_numpy(force_field_left).float()
        force_field_right_tensor = torch.from_numpy(force_field_right).float()
        
        # NOTE: We do NOT apply reference tactile processing here to avoid double-processing
        # The dataset.py loader will handle reference tactile computation and application
        # We only store the raw tactile force fields in the pickle files
        
        # === Create Observation ===
        obs = {
            'ee_pos': ee_pose_tensor[:3].unsqueeze(0),  # (1, 3)
            'ee_quat': ee_pose_tensor[3:].unsqueeze(0),  # (1, 4)
            'ee_lin_vel': ee_vel_tensor[:3].unsqueeze(0),  # (1, 3)
            'ee_ang_vel': ee_vel_tensor[3:].unsqueeze(0),  # (1, 3)
            'left_tactile_camera_taxim': torch.from_numpy(left_img_resized).unsqueeze(0).float(),  # (1, 60, 80, 3)
            'right_tactile_camera_taxim': torch.from_numpy(right_img_resized).unsqueeze(0).float(),  # (1, 60, 80, 3)
            'tactile_force_field_left': force_field_left_tensor.unsqueeze(0),  # (1, 7, 9, 3) - RAW data, no reference processing
            'tactile_force_field_right': force_field_right_tensor.unsqueeze(0),  # (1, 7, 9, 3) - RAW data, no reference processing
            'tactile_coord_left': torch.from_numpy(tactile_coord_left).unsqueeze(0).float(),  # (1, 7, 9, 3)
            'tactile_coord_right': torch.from_numpy(tactile_coord_right).unsqueeze(0).float(),  # (1, 7, 9, 3)
        }
        
        # Add front RGB image if available
        if front_rgb is not None:
            obs['front'] = torch.from_numpy(front_rgb).unsqueeze(0).float()  # (1, H, W, 3)
        
        observations_list.append({'obs': obs})
        object_point_clouds.append(obj_pcd_sampled)
        env_point_clouds.append(env_pcd_sampled)
        contact_vectors_list.append(contact_vecs)
    
    # === Save Files ===
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Main file
    main_file_path = output_dir / f"{episode_name}.pkl"
    main_data = {
        'observations': observations_list
    }
    
    with open(main_file_path, 'wb') as f:
        pickle.dump(main_data, f)
    
    print(f"Saved main file: {main_file_path}")
    
    # Contact file - single environment format
    contact_file_path = output_dir / f"{episode_name}_contact.pkl"
    contact_data = {
        'object_point_clouds': [object_point_clouds],  # List of episodes, each episode is list of timesteps
        'env_point_clouds': [env_point_clouds],
        'contact_vectors': [contact_vectors_list]
    }
    
    with open(contact_file_path, 'wb') as f:
        pickle.dump(contact_data, f)
    
    print(f"Saved contact file: {contact_file_path}")
    print(f"  - Object point clouds: {len(object_point_clouds)} timesteps, shape {obj_pcd_sampled.shape}")
    print(f"  - Environment point clouds: {len(env_point_clouds)} timesteps, shape {env_pcd_sampled.shape}")
    print(f"  - Contact vectors: {len(contact_vectors_list)} timesteps")
    
    # Plot wrench history if available
    if len(wrench_history) > 0:
        wrench_plot_path = output_dir / f"{episode_name}_wrench.png"
        plot_wrench_history(
            wrench_history,
            str(wrench_plot_path),
            title=f"Net Wrench Over Time - {episode_name}"
        )
    
    return str(main_file_path), str(contact_file_path)


def main():
    parser = argparse.ArgumentParser(description="Convert real-world HDF5 data to contact field dataset format")
    parser.add_argument('--hdf5_paths', nargs='+', help='Paths to HDF5 files to convert')
    parser.add_argument('--data_dir', type=str, help='Directory containing HDF5 files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for converted files')
    parser.add_argument('--max_steps', type=int, default=None, help='Maximum number of steps to process per episode')
    parser.add_argument('--num_object_points', type=int, default=256, help='Number of points for object point cloud')
    parser.add_argument('--num_env_points', type=int, default=512, help='Number of points for environment point cloud')
    parser.add_argument('--use_semantic_segmentation', action='store_true', help='Use semantic segmentation instead of color')
    parser.add_argument('--tactile_nn_model', type=str, 
                       default=os.path.expanduser('~/gendp/gsrobotics/models/nnmini.pt'),
                       help='Path to tactile neural network model')
    parser.add_argument('--tactile_width', type=int, default=320, help='Tactile image width')
    parser.add_argument('--tactile_height', type=int, default=240, help='Tactile image height')
    parser.add_argument('--disable_tactile_gpu', action='store_true', help='Disable GPU for tactile processing')
    parser.add_argument('--verify', action='store_true', help='Verify converted files with ContactFieldDataset')
    parser.add_argument('--tactile_ref_img_left', type=str, default='./data/ref_imgs/tactile_left_rgb.png', help='Path to reference tactile image for left gripper')
    parser.add_argument('--tactile_ref_img_right', type=str, default='./data/ref_imgs/tactile_right_rgb_old.png', help='Path to reference tactile image for right gripper')
    parser.add_argument('--force_estimator', type=str, default='simple', choices=['simple', 'analytical'], 
                       help='Force estimation method: simple (pseudo-inverse) or analytical (cvxpy optimization)')
    parser.add_argument('--force_lambda', type=float, default=0.01, help='Regularization weight for analytical force estimator')
    parser.add_argument('--force_epsilon', type=float, default=1e-6, help='Epsilon for analytical force estimator weight division')

    args = parser.parse_args()
    
    # Collect HDF5 files
    hdf5_files = []
    if args.hdf5_paths:
        hdf5_files.extend([Path(p) for p in args.hdf5_paths])
    
    if args.data_dir:
        data_dir = Path(args.data_dir)
        hdf5_files.extend(list(data_dir.glob('*.hdf5')))
        hdf5_files.extend(list(data_dir.glob('*.h5')))
    
    if not hdf5_files:
        print("Error: No HDF5 files found. Please specify --hdf5_paths or --data_dir")
        return
    
    # Remove duplicates
    hdf5_files = sorted(list(set(hdf5_files)))
    print(f"Found {len(hdf5_files)} HDF5 files to convert")
    
    # Separate files into train and test sets
    # Test set: files ending with "0.hdf5" or "0.h5" (every 10th file)
    # Train set: all other files
    test_files = []
    train_files = []
    
    for hdf5_file in hdf5_files:
        filename = hdf5_file.stem  # Get filename without extension
        if filename.endswith('0'):
            test_files.append(hdf5_file)
        else:
            train_files.append(hdf5_file)
    
    print(f"\n📊 Dataset Split:")
    print(f"  Train set: {len(train_files)} files (no front RGB)")
    print(f"  Test set:  {len(test_files)} files (with front RGB)")
    
    # Create train and test subdirectories
    train_dir = Path(args.output_dir) / 'train'
    test_dir = Path(args.output_dir) / 'test'
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output directories:")
    print(f"  Train: {train_dir}")
    print(f"  Test:  {test_dir}")
    
    print(f"\n⚙️  Force Estimator Configuration:")
    print(f"  Method: {args.force_estimator}")
    if args.force_estimator == 'analytical':
        print(f"  lambda_reg: {args.force_lambda}")
        print(f"  epsilon: {args.force_epsilon}")
    
    # Create force estimator config dict
    force_estimator_config = {
        'method': args.force_estimator,
        'lambda_reg': args.force_lambda,
        'epsilon': args.force_epsilon
    }
    
    # Initialize tactile processors
    print("Initializing tactile processors...")
    
    # Default marker configurations
    DEFAULT_MARKER_CONFIG_LEFT = {
        'N': 7,
        'M': 9,
        'fps': 10,
        'x0': 34.5,
        'y0': 37.5,
        'dx': 28.7,
        'dy': 29.6
    }
    DEFAULT_MARKER_CONFIG_RIGHT = {
        'N': 7,
        'M': 9,
        'fps': 10,
        'x0': 40.6,
        'y0': 46,
        'dx': 28.6,
        'dy': 29.1,
    }
    
    nn_model_path = os.path.expanduser(args.tactile_nn_model)
    
    tactile_processor_left = TactileProcessor(
        width=args.tactile_width,
        height=args.tactile_height,
        nn_model_path=nn_model_path,
        marker_config=DEFAULT_MARKER_CONFIG_LEFT,
        use_gpu=not args.disable_tactile_gpu,
        ref_img=args.tactile_ref_img_left if hasattr(args, 'tactile_ref_img_left') and args.tactile_ref_img_left else None
    )
    
    tactile_processor_right = TactileProcessor(
        width=args.tactile_width,
        height=args.tactile_height,
        nn_model_path=nn_model_path,
        marker_config=DEFAULT_MARKER_CONFIG_RIGHT,
        use_gpu=not args.disable_tactile_gpu,
        ref_img=args.tactile_ref_img_right if hasattr(args, 'tactile_ref_img_right') and args.tactile_ref_img_right else None
    )
    
    # Initialize D3Fields components if using semantic segmentation
    fusion = None
    kin_helper = None
    if args.use_semantic_segmentation:
        print("Initializing D3Fields components for semantic segmentation...")
        # from gendp.gendp.common.robot_utils import FrankaRobot
        
        # fusion = Fusion(
        #     fusion_rgb_weighting=0.1,
        #     fusion_depth_weighting=0.0
        # )
        fusion = Fusion(num_cam=3, dtype=torch.float16)
        kin_helper = KinHelper(robot_name='panda')
    
    # Convert each file
    converted_files = []

    # Process test files (with front RGB)
    if test_files:
        print(f"\n{'='*80}")
        print(f"CONVERTING TEST SET ({len(test_files)} files)")
        print(f"{'='*80}")
        for hdf5_file in test_files:
            try:
                main_path, contact_path = convert_hdf5_to_dataset(
                    str(hdf5_file),
                    str(test_dir),
                    tactile_processor_left,
                    tactile_processor_right,
                    use_semantic_segmentation=args.use_semantic_segmentation,
                    num_object_points=args.num_object_points,
                    num_env_points=args.num_env_points,
                    max_steps=args.max_steps,
                    fusion=fusion,
                    kin_helper=kin_helper,
                    include_front_rgb=True,  # Include front RGB for testing/visualization
                    force_estimator_config=force_estimator_config
                )
                converted_files.append((main_path, contact_path))
            except Exception as e:
                print(f"❌ Error converting {hdf5_file}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Process train files (without front RGB)
    if train_files:
        print(f"\n{'='*80}")
        print(f"CONVERTING TRAIN SET ({len(train_files)} files)")
        print(f"{'='*80}")
        for hdf5_file in train_files:
            try:
                main_path, contact_path = convert_hdf5_to_dataset(
                    str(hdf5_file),
                    str(train_dir),
                    tactile_processor_left,
                    tactile_processor_right,
                    use_semantic_segmentation=args.use_semantic_segmentation,
                    num_object_points=args.num_object_points,
                    num_env_points=args.num_env_points,
                    max_steps=args.max_steps,
                    fusion=fusion,
                    kin_helper=kin_helper,
                    include_front_rgb=False,  # No front RGB for training
                    force_estimator_config=force_estimator_config
                )
                converted_files.append((main_path, contact_path))
            except Exception as e:
                print(f"❌ Error converting {hdf5_file}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n{'='*80}")
    print(f"CONVERSION COMPLETE!")
    print(f"{'='*80}")
    print(f"✅ Successfully converted {len(converted_files)} / {len(hdf5_files)} files")
    print(f"   - Train set: {len(train_files)} files → {train_dir}")
    print(f"   - Test set:  {len(test_files)} files → {test_dir}")
    print(f"{'='*80}")
    
    # Verify converted files if requested
    if args.verify and converted_files:
        print(f"\n{'='*80}")
        print(f"VERIFYING CONVERTED FILES")
        print(f"{'='*80}")
        
        from verify_converted_data import verify_pair, test_dataset_loading, DATASET_AVAILABLE
        
        # Verify each file pair
        all_verified = True
        for main_path, contact_path in converted_files:
            result = verify_pair(main_path, contact_path, test_dataset=False)
            if not result:
                all_verified = False
        
        # Test dataset loading for both train and test sets
        if all_verified and DATASET_AVAILABLE:
            if train_files:
                print(f"\n📊 Testing train dataset loading...")
                train_ok = test_dataset_loading(str(train_dir))
                if not train_ok:
                    all_verified = False
            
            if test_files:
                print(f"\n📊 Testing test dataset loading...")
                test_ok = test_dataset_loading(str(test_dir))
                if not test_ok:
                    all_verified = False
        
        print(f"\n{'='*80}")
        if all_verified:
            print(f"✅ All files verified successfully!")
        else:
            print(f"❌ Some files failed verification")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
