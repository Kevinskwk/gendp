#!/usr/bin/env python
"""
Visualization script for semantic fields and action trajectories.
Renders frame-by-frame visualizations with consistent viewing angles and saves separate images.

USAGE:
    python vis_combined_fields.py --hdf5_path episode_0.hdf5 \
                                   --output_dir outputs/combined_viz
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from matplotlib import colormaps
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial.transform as st
from typing import Dict, Optional, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gendp.common.data_utils import load_dict_from_hdf5, d3fields_proc
from gendp.common.kinematics_utils import KinHelper
from d3fields.utils.draw_utils import aggr_point_cloud_from_data, np2o3d, o3dVisualizer, ImgEncoding
from d3fields.fusion import Fusion


# Default camera keys for real-world data
DEFAULT_CAM_KEYS = ['camera_front', 'camera_left', 'camera_right']

# Shape metadata for d3fields processing
SHAPE_META = {
    'shape': [6, 2000],
    'type': 'spatial',
    'info': {
        'reference_frame': 'robot',
        'distill_dino': True,
        'distill_obj': 'scraper',
        'query_text': 'dark green plastic scraper tool',
        'view_keys': ['front', 'left', 'right'],
        'N_gripper': 100,
        'boundaries': {
            'x_lower': 0.3,
            'x_upper': 0.7,
            'y_lower': -0.2,
            'y_upper': 0.2,
            'z_lower': -0.03,
            'z_upper': 0.3,
        },
        'resize_ratio': 0.5
    }
}

# Viewing angle configuration (elevation, azimuth)
DEFAULT_VIEW_ANGLE = (10, -45)


def process_episode_data(hdf5_path: str, fusion: Fusion, kin_helper: KinHelper,
                         cam_keys: List[str] = DEFAULT_CAM_KEYS, max_steps: int = 200):
    """
    Process episode data to extract point clouds, semantic features, and observations.
    
    Returns:
        tuple: (obj_pcd_list, obj_feats_list, bg_pcd_list, bg_feats_list, rgb_pcd_list, rgb_colors_list, observations_list, data_dict)
    """
    # Load HDF5 data
    data_dict, _ = load_dict_from_hdf5(hdf5_path)
    observations = data_dict['observations']
    
    # Get sequence length
    T = min(observations['images'][f'{cam_keys[0]}_color'].shape[0], max_steps)
    robot_base_in_world_seq = observations.get('robot_base_pose_in_world', None)
    
    # Convert HDF5 Dataset to numpy array if needed
    if robot_base_in_world_seq is not None and hasattr(robot_base_in_world_seq, 'shape'):
        robot_base_in_world_seq = np.array(robot_base_in_world_seq)
    
    # Initialize lists
    obj_pcd_list = []
    obj_feats_list = []
    bg_pcd_list = []
    bg_feats_list = []
    rgb_pcd_list = []
    rgb_colors_list = []
    observations_list = []
    
    print(f"Processing {T} frames from episode...")
    
    # Define boundaries for point cloud filtering
    boundaries = {
        'x_lower': 0.2,
        'x_upper': 0.7,
        'y_lower': -0.2,
        'y_upper': 0.2,
        'z_lower': -0.03,
        'z_upper': 0.3,
    }
    
    for t in tqdm(range(T), desc="Processing frames"):
        # Get current robot base pose
        if robot_base_in_world_seq is not None:
            robot_base_in_world = robot_base_in_world_seq[t]
        else:
            robot_base_in_world = np.eye(4)
        
        # Stack camera data for d3fields processing
        colors_d3f = np.stack([observations['images'][f'{cam_key}_color'][t:t+1] for cam_key in cam_keys], axis=1)
        depths_d3f = np.stack([observations['images'][f'{cam_key}_depth'][t:t+1] for cam_key in cam_keys], axis=1) / 1000.
        intrinsics_d3f = np.stack([observations['images'][f'{cam_key}_intrinsics'][t:t+1] for cam_key in cam_keys], axis=1)
        extrinsics_d3f = np.stack([observations['images'][f'{cam_key}_extrinsics'][t:t+1] for cam_key in cam_keys], axis=1)
        ee_poses = observations['ee_pose'][t:t+1]
        
        # Stack camera data for RGB point cloud extraction (no batch dimension)
        colors_rgb = np.stack([observations['images'][f'{cam_key}_color'][t] for cam_key in cam_keys])
        depths_rgb = np.stack([observations['images'][f'{cam_key}_depth'][t] for cam_key in cam_keys]) / 1000.
        intrinsics_rgb = np.stack([observations['images'][f'{cam_key}_intrinsics'][t] for cam_key in cam_keys])
        extrinsics_rgb = np.stack([observations['images'][f'{cam_key}_extrinsics'][t] for cam_key in cam_keys])
        
        # Process with d3fields for semantic segmentation
        result = d3fields_proc(
            fusion=fusion,
            shape_meta=SHAPE_META,
            color_seq=colors_d3f,
            depth_seq=depths_d3f,
            extri_seq=extrinsics_d3f,
            intri_seq=intrinsics_d3f,
            robot_base_pose_in_world_seq=robot_base_in_world_seq[t:t+1] if robot_base_in_world_seq is not None else None,
            teleop_robot=kin_helper,
            qpos_seq=observations['full_joint_pos'][t:t+1],
            exclude_threshold=0.01,
            use_obj_bg_seg=True,
            gripper_pose_seq=ee_poses,
            use_gripper_crop=True,
        )
        
        # Extract RGB point cloud with colors using aggr_point_cloud_from_data
        rgb_pcd_world, rgb_colors = aggr_point_cloud_from_data(
            colors_rgb,
            depths_rgb,
            intrinsics_rgb,
            extrinsics_rgb,
            downsample=True,  # Enable downsampling for visualization efficiency
            downsample_r=0.005,
            out_o3d=False,
            boundaries=boundaries
        )
        
        # Transform RGB point cloud to robot frame
        rgb_pcd = np.linalg.inv(robot_base_in_world) @ np.concatenate([rgb_pcd_world, np.ones((rgb_pcd_world.shape[0], 1))], axis=-1).T
        rgb_pcd = rgb_pcd.T[:, :3]
        
        # Unpack d3fields results
        if len(result) == 6:
            pcd, pcd_feats, obj_pcd, obj_feats, bg_pcd, bg_feats = result
            obj_pcd = obj_pcd[0]
            obj_feats = obj_feats[0]
            bg_pcd = bg_pcd[0]
            bg_feats = bg_feats[0]
            
            # Transform semantic point clouds to robot frame
            obj_pcd = np.linalg.inv(robot_base_in_world) @ np.concatenate([obj_pcd, np.ones((obj_pcd.shape[0], 1))], axis=-1).T
            obj_pcd = obj_pcd.T[:, :3]
            
            bg_pcd = np.linalg.inv(robot_base_in_world) @ np.concatenate([bg_pcd, np.ones((bg_pcd.shape[0], 1))], axis=-1).T
            bg_pcd = bg_pcd.T[:, :3]
            
        else:
            # Fallback if no segmentation
            pcd, pcd_feats = result
            obj_pcd = pcd[0]
            obj_feats = pcd_feats[0]
            bg_pcd = np.zeros((0, 3))
            bg_feats = np.zeros((0, obj_feats.shape[1]))
            
            obj_pcd = np.linalg.inv(robot_base_in_world) @ np.concatenate([obj_pcd, np.ones((obj_pcd.shape[0], 1))], axis=-1).T
            obj_pcd = obj_pcd.T[:, :3]
        
        # Store processed data
        obj_pcd_list.append(obj_pcd)
        obj_feats_list.append(obj_feats)
        bg_pcd_list.append(bg_pcd)
        bg_feats_list.append(bg_feats)
        rgb_pcd_list.append(rgb_pcd)
        rgb_colors_list.append(rgb_colors)
        
        # Store observation data
        obs_data = {
            'ee_pose': observations['ee_pose'][t],
            'gripper_pos': observations.get('gripper_pos', np.zeros(1))[t] if 'gripper_pos' in observations else 0.0,
            'full_joint_pos': observations['full_joint_pos'][t],
        }
        observations_list.append(obs_data)
    
    return obj_pcd_list, obj_feats_list, bg_pcd_list, bg_feats_list, rgb_pcd_list, rgb_colors_list, observations_list, data_dict


def extract_action_trajectory(data_dict, frame_idx, horizon=10):
    """Extract action trajectory for visualization"""
    observations = data_dict['observations']
    T = observations['ee_pose'].shape[0]
    
    # Get future poses
    t_start = frame_idx
    t_end = min(t_start + horizon, T)
    
    # Extract end-effector trajectory
    ee_trajectory = observations['ee_pose'][t_start:t_end]
    
    # Transformation from EE to gripper tip: [0, 0, 0.14, 0, 0, 0]
    ee_to_gripper_offset = np.array([0.0, 0.0, 0.14])
    
    # Convert to transformation matrices
    trajectory_poses = []
    for ee_pose in ee_trajectory:
        # Create EE pose matrix
        ee_pose_mat = np.eye(4)
        ee_pose_mat[:3, 3] = ee_pose[:3]
        ee_pose_mat[:3, :3] = st.Rotation.from_euler('xyz', ee_pose[3:6]).as_matrix()
        
        # Apply gripper tip offset in EE frame
        gripper_tip_offset_mat = np.eye(4)
        gripper_tip_offset_mat[:3, 3] = ee_to_gripper_offset
        
        # Gripper tip pose = EE pose @ offset
        gripper_tip_pose = ee_pose_mat @ gripper_tip_offset_mat
        
        trajectory_poses.append(gripper_tip_pose)
    
    return trajectory_poses


def plot_semantic_field(ax, obj_pcd, obj_feats, bg_pcd, bg_feats, view_angle, point_size=20):
    """Plot semantic field visualization"""
    ax.clear()
    
    # Use different colormaps for object and background
    obj_cmap = colormaps.get_cmap('viridis')
    bg_cmap = colormaps.get_cmap('Reds')
    
    # Color by semantic features
    obj_colors = obj_cmap(obj_feats[:, 1])[:, :3] if obj_feats.shape[1] > 1 else obj_cmap(obj_feats[:, 0])[:, :3]
    bg_colors = bg_cmap(bg_feats[:, 1])[:, :3] if bg_feats.shape[1] > 1 else bg_cmap(bg_feats[:, 0])[:, :3]
    
    # Plot object points
    if len(obj_pcd) > 0:
        ax.scatter(obj_pcd[:, 0], obj_pcd[:, 1], obj_pcd[:, 2],
                  c=obj_colors, s=point_size, alpha=0.8)
    
    # Plot background points
    if len(bg_pcd) > 0:
        ax.scatter(bg_pcd[:, 0], bg_pcd[:, 1], bg_pcd[:, 2],
                  c=bg_colors, s=point_size, alpha=0.3)
    
    # Set view angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # Remove labels and ticks
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Set axis limits
    ax.set_xlim([0.4, 0.6])
    ax.set_ylim([-0.1, 0.1])
    ax.set_zlim([0.0, 0.2])
    ax.set_box_aspect([1, 1, 1])


def plot_rgb_with_trajectory(ax, rgb_pcd, rgb_colors, trajectory_poses, robot_base_in_world,
                              view_angle, point_size=20):
    """Plot RGB point cloud with action trajectory overlay"""
    ax.clear()
    
    # Plot RGB point cloud
    if len(rgb_pcd) > 0:
        # Use actual RGB colors if available, otherwise use gray
        if rgb_colors is not None and len(rgb_colors) == len(rgb_pcd):
            colors = rgb_colors / 255.0 if rgb_colors.max() > 1 else rgb_colors
        else:
            colors = np.ones((len(rgb_pcd), 3)) * 0.7
        
        # Plot with reduced alpha for background effect
        ax.scatter(rgb_pcd[:, 0], rgb_pcd[:, 1], rgb_pcd[:, 2],
                  c=colors, s=point_size, alpha=0.3, depthshade=False)
    
    # Plot action trajectory (rendered last to ensure it's in front)
    if trajectory_poses is not None and len(trajectory_poses) > 0:
        # Transform trajectory to robot frame
        trajectory_points = []
        for pose_mat in trajectory_poses:
            if robot_base_in_world is not None:
                pose_in_robot = np.linalg.inv(robot_base_in_world) @ pose_mat
            else:
                pose_in_robot = pose_mat
            trajectory_points.append(pose_in_robot[:3, 3])
        
        trajectory_points = np.array(trajectory_points)
        
        # Plot trajectory line with higher zorder to force it to front
        ax.plot(trajectory_points[:, 0], trajectory_points[:, 1], trajectory_points[:, 2],
               'r-', linewidth=4, alpha=0.5, label='Action Trajectory', zorder=500)
        
        # Plot trajectory points with higher zorder and depthshade=False
        action_cmap = colormaps.get_cmap('plasma')
        colors = action_cmap(np.linspace(0.3, 1.0, len(trajectory_points)))[:, :3]
        ax.scatter(trajectory_points[:, 0], trajectory_points[:, 1], trajectory_points[:, 2],
                  c=colors, s=100, alpha=1.0, edgecolors='black', linewidths=2,
                  zorder=1000, depthshade=False)
    
    # Set view angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # Remove labels and ticks
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Set axis limits
    ax.set_xlim([0.4, 0.6])
    ax.set_ylim([-0.1, 0.1])
    ax.set_zlim([0.0, 0.2])
    ax.set_box_aspect([1, 1, 1])


def save_frame_visualizations(frame_idx, obj_pcd, obj_feats, bg_pcd, bg_feats,
                               rgb_pcd, rgb_colors, trajectory_poses, robot_base_in_world,
                               output_dir, view_angle=DEFAULT_VIEW_ANGLE, dpi=100, point_size=20):
    """Save individual frame visualizations as separate images"""
    
    # Create output directory for this frame
    frame_dir = output_dir / f"frame_{frame_idx:04d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save semantic field
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    plot_semantic_field(ax, obj_pcd, obj_feats, bg_pcd, bg_feats, view_angle, point_size)
    plt.savefig(frame_dir / 'semantic_field.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    # 2. Save RGB with trajectory
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    plot_rgb_with_trajectory(ax, rgb_pcd, rgb_colors, trajectory_poses, robot_base_in_world, view_angle, point_size)
    plt.savefig(frame_dir / 'rgb_trajectory.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    if frame_idx % 10 == 0:  # Print progress every 10 frames
        print(f"Saved frame {frame_idx} visualizations to {frame_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize semantic fields and action trajectories')
    parser.add_argument('--hdf5_path', type=str, required=True,
                        help='Path to episode HDF5 file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for frame-by-frame visualizations')
    parser.add_argument('--max_steps', type=int, default=200,
                        help='Maximum number of steps to process')
    parser.add_argument('--action_horizon', type=int, default=10,
                        help='Number of future action steps to visualize')
    parser.add_argument('--point_size', type=int, default=20,
                        help='Point size in visualization')
    parser.add_argument('--dpi', type=int, default=100,
                        help='DPI for saved images')
    parser.add_argument('--view_elev', type=float, default=10,
                        help='Elevation angle for camera view')
    parser.add_argument('--view_azim', type=float, default=-45,
                        help='Azimuth angle for camera view')
    parser.add_argument('--fusion_device', type=str, default='cuda:0',
                        help='Device for fusion processing')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SEMANTIC FIELD AND ACTION TRAJECTORY VISUALIZATION")
    print("=" * 80)
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Set viewing angle
    view_angle = (args.view_elev, args.view_azim)
    print(f"View angle: elevation={view_angle[0]}, azimuth={view_angle[1]}")
    
    # Initialize fusion and kinematics
    print("\nInitializing fusion and kinematics...")
    fusion = Fusion(num_cam=len(DEFAULT_CAM_KEYS), dtype=torch.float16, device=args.fusion_device)
    kin_helper = KinHelper(robot_name='panda')
    print("✅ Fusion and kinematics initialized")
    
    # Process episode data
    print(f"\nProcessing episode: {args.hdf5_path}")
    (obj_pcd_list, obj_feats_list, bg_pcd_list, bg_feats_list, 
     rgb_pcd_list, rgb_colors_list, observations_list, data_dict) = process_episode_data(
        args.hdf5_path, fusion, kin_helper, DEFAULT_CAM_KEYS, args.max_steps
    )
    
    num_frames = len(obj_pcd_list)
    print(f"✅ Processed {num_frames} frames")
    
    # Get robot base poses for trajectory transformation
    robot_base_in_world_seq = data_dict['observations'].get('robot_base_pose_in_world', None)
    
    # Process each frame
    print(f"\nGenerating visualizations for {num_frames} frames...")
    for frame_idx in tqdm(range(num_frames), desc="Processing frames"):
        # Get data for current frame
        obj_pcd = obj_pcd_list[frame_idx]
        obj_feats = obj_feats_list[frame_idx]
        bg_pcd = bg_pcd_list[frame_idx]
        bg_feats = bg_feats_list[frame_idx]
        rgb_pcd = rgb_pcd_list[frame_idx]
        rgb_colors = rgb_colors_list[frame_idx]
        
        # Extract action trajectory
        trajectory_poses = extract_action_trajectory(data_dict, frame_idx, args.action_horizon)
        
        # Get robot base pose
        robot_base_in_world = robot_base_in_world_seq[frame_idx] if robot_base_in_world_seq is not None else None
        
        # Save visualizations
        save_frame_visualizations(
            frame_idx, obj_pcd, obj_feats, bg_pcd, bg_feats,
            rgb_pcd, rgb_colors, trajectory_poses, robot_base_in_world,
            output_dir, view_angle, args.dpi, args.point_size
        )
    
    print(f"\n✅ All visualizations saved to: {output_dir}")
    print(f"Total frames processed: {num_frames}")
    print(f"\nOutput structure:")
    print(f"  - frame_XXXX/semantic_field.png: Object/background semantic features")
    print(f"  - frame_XXXX/rgb_trajectory.png: RGB point cloud with action trajectory")


if __name__ == "__main__":
    main()
