#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import cv2
from tqdm import tqdm
from matplotlib import colormaps
import matplotlib.pyplot as plt
# add path for importing
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gendp.common.data_utils import load_dict_from_hdf5, extract_gripper_tool_pcd
from gendp.common.kinematics_utils import KinHelper
from d3fields.utils.draw_utils import aggr_point_cloud_from_data, np2o3d, o3dVisualizer, ImgEncoding, ExtriConvention
import scipy.spatial.transform as st

from vis_utils import segment_pointcloud_by_color


def sample_or_pad_pointcloud(pointcloud, colors, target_size):
    """
    Downsample or pad point cloud to target size.
    
    Args:
        pointcloud: (N, 3) array of point positions
        colors: (N, 3) array of point colors
        target_size: Target number of points
    
    Returns:
        Tuple of (sampled_pointcloud, sampled_colors) with shape (target_size, 3)
    """
    if len(pointcloud) == 0:
        return np.zeros((target_size, 3)), np.zeros((target_size, 3))
    
    if len(pointcloud) > target_size:
        # Random downsampling
        indices = np.random.choice(len(pointcloud), target_size, replace=False)
        return pointcloud[indices], colors[indices]
    elif len(pointcloud) < target_size:
        # Pad with zeros
        pad_size = target_size - len(pointcloud)
        padded_pointcloud = np.concatenate([
            pointcloud,
            np.zeros((pad_size, 3))
        ], axis=0)
        padded_colors = np.concatenate([
            colors,
            np.zeros((pad_size, 3))
        ], axis=0)
        return padded_pointcloud, padded_colors
    else:
        return pointcloud, colors


### hyper param
epi_range = [0]
vis_robot = True
vis_action = True
apply_color_segmentation = False  # Set to True to apply color filtering
use_gripper_segmentation = True   # Set to True to use gripper-based tool segmentation
vis_segmented_separately = True  # Set to True to show object and env separately
vis_only_object = True            # Set to True to visualize only object/tool points (when segmentation is enabled)

# Downsampling parameters (set to None to disable downsampling)
downsample_obj_points = 256   # Number of object points after downsampling
downsample_env_points = 512   # Number of environment points after downsampling

curr_dir = os.path.dirname(os.path.abspath(__file__))
# data_dir = f'{curr_dir}/../../data/sapien_demo/pencil_insertion_demo'
# data_dir = f'{curr_dir}/../../data/crayon_cross'
# data_dir = f'{curr_dir}/../../data/scraper_combined'
# data_dir = f'{curr_dir}/../../data/scrap_tool_test'
# data_dir = f'{curr_dir}/../../data/crayon_pickup_new'
data_dir = f'{curr_dir}/../../data/crayon_cross_z_58'
robot_name = 'panda'
# cam_keys = ['right_bottom_view', 'left_bottom_view', 'right_top_view', 'left_top_view']
# cam_keys = ['camera_wrist', 'camera_fixed']
cam_keys = ['camera_front', 'camera_left', 'camera_right']
# cam_keys = ['camera_front', 'camera_right']

# Color segmentation parameters
PINK_HUE_RANGE = (0, 100)   # Pink/Magenta range (wraps around)
PINK_SATURATION_RANGE = (50, 255)   # Pink saturation range (min, max)
PINK_VALUE_RANGE = (110, 255)        # Pink brightness range (min, max)

PURPLE_HUE_RANGE = (230, 240)  # Purple range
PURPLE_SATURATION_RANGE = (50, 200)  # Purple saturation range (min, max)
PURPLE_VALUE_RANGE = (100, 200)       # Purple brightness range (min, max)

GREEN_HUE_RANGE = (120, 180)  # Green range
GREEN_SATURATION_RANGE = (30, 255)  # Green saturation range (min, max)
GREEN_VALUE_RANGE = (100, 255)       # Green brightness range (min, max)

YELLOW_HUE_RANGE = (0, 100)  # Yellow range
YELLOW_SATURATION_RANGE = (0, 255)  # Yellow saturation range (min, max)
YELLOW_VALUE_RANGE = (0, 255)       # Yellow brightness range (min, max)

CUCUMBER_HUE_RANGE = (90, 110)  # Cucumber range
CUCUMBER_SATURATION_RANGE = (50, 255)  # Cucumber saturation range (min, max)
CUCUMBER_VALUE_RANGE = (60, 255)       # Cucumber brightness range (min, max)

if 'peg' in data_dir:
    obj_hue_range = PINK_HUE_RANGE
    obj_saturation_range = PINK_SATURATION_RANGE
    obj_value_range = PINK_VALUE_RANGE
    env_hue_range = GREEN_HUE_RANGE
    env_saturation_range = GREEN_SATURATION_RANGE
    env_value_range = GREEN_VALUE_RANGE
else:
    obj_hue_range = PURPLE_HUE_RANGE
    obj_saturation_range = PURPLE_SATURATION_RANGE
    obj_value_range = PURPLE_VALUE_RANGE
    env_hue_range = CUCUMBER_HUE_RANGE
    env_saturation_range = CUCUMBER_SATURATION_RANGE
    env_value_range = CUCUMBER_VALUE_RANGE

# Separate spatial boundaries for object and environment
OBJECT_BOUNDARIES = {
    'x_lower': 0.3,
    'x_upper': 0.7,
    'y_lower': -0.15,
    'y_upper': 0.15,
    'z_lower': 0.0,
    'z_upper': 0.4,
}

ENV_BOUNDARIES = {
    'x_lower': 0.36,
    'x_upper': 0.53,
    'y_lower': -0.1,
    'y_upper': 0.05,
    'z_lower': -0.03,
    'z_upper': 0.1
}


# Set initial camera view to Z-up
view_ctrl_info = {
    "front": [-1, 1, 0.5],      # X axis right
    "lookat": [0.5, 0, 0.1],     # Look at origin
    "up": [0, 0, 1],         # Z axis up
    "zoom": 1.0              # Adjust as needed
}
# visualizer = o3dVisualizer(view_ctrl_info=view_ctrl_info)
visualizer = o3dVisualizer()
visualizer.start()

### create kinematics helper
kin_helper = KinHelper(robot_name='panda')

# Initialize list to collect tool point counts
tool_point_counts = []
frame_numbers = []

for i in tqdm(epi_range):
    data_path = f'{data_dir}/episode_{i}.hdf5'

    data_dict, _ = load_dict_from_hdf5(data_path)

    # add meshes to visualize actions
    if vis_action:
        # init_cart = data_dict['cartesian_action'][0] # (horizon, 7)
        init_cart = data_dict['observations']['ee_pose'][0]
        action_horizon = init_cart.shape[0]
        action_cm = colormaps.get_cmap('plasma')
        action_colors = action_cm(np.linspace(0, 1, init_cart.shape[0], endpoint=True))[:, :3] # (horizon, 3)
        for a_i in range(action_horizon):
            visualizer.add_triangle_mesh('sphere', f'action_{a_i}', action_colors[a_i], radius=0.01)
    
    visualizer.add_triangle_mesh('origin', 'base', size=0.2)
    visualizer.add_triangle_mesh('origin', 'front')
    visualizer.add_triangle_mesh('origin', 'left')
    visualizer.add_triangle_mesh('origin', 'right')
    visualizer.add_triangle_mesh('origin', 'ee_pose', size=0.05)  # End-effector pose coordinate frame
    # visualizer.add_triangle_mesh('origin', 'left_finger', size=0.05)
    # visualizer.add_triangle_mesh('origin', 'right_finger', size=0.05)
    visualizer.update_triangle_mesh('base', tf=np.eye(4))

    T = data_dict['observations']['images'][f'{cam_keys[0]}_color'].shape[0]
    robot_base_in_world_seq = data_dict['observations']['robot_base_pose_in_world'][()]
    
    for t in tqdm(range(T)):
        # visualize point cloud
        robot_base_in_world = robot_base_in_world_seq[t]
        colors = np.stack([data_dict['observations']['images'][f'{cam_key}_color'][t] for cam_key in cam_keys]) # (N, H, W, 3)
        depths = np.stack([data_dict['observations']['images'][f'{cam_key}_depth'][t] for cam_key in cam_keys]) / 1000. # (N, H, W)
        intrinsics = np.stack([data_dict['observations']['images'][f'{cam_key}_intrinsics'][t] for cam_key in cam_keys])
        extrinsics = np.stack([data_dict['observations']['images'][f'{cam_key}_extrinsics'][t] for cam_key in cam_keys])

        # Tune the extrinsics
        # pose_0 = np.linalg.inv(extrinsics[2])
        # pose_0[0:3, 3] += np.array([-0.025, -0.01, -0.005])  # Adjust position
        # extrinsics[2] = np.linalg.inv(pose_0)

        boundaries = {
            'x_lower': 0.3,
            'x_upper': 0.7,
            'y_lower': -0.2,
            'y_upper': 0.2,
            'z_lower': -0.1,
            'z_upper': 0.5,
        }

        pcd, pcd_colors = aggr_point_cloud_from_data(colors[:],
                                                     depths[:],
                                                     intrinsics[:],
                                                     extrinsics[:],
                                                     downsample=False,
                                                     out_o3d=False,
                                                     boundaries=boundaries,
                                                    #  color_fmt=ImgEncoding.BGR_UINT8,
                                                    #  pose_fmt=ExtriConvention.CAM_IN_WORLD
                                                     )
        pcd = np.linalg.inv(robot_base_in_world) @ np.concatenate([pcd, np.ones((pcd.shape[0], 1))], axis=-1).T
        pcd = pcd.T[:, :3]
        
        # Extract end-effector pose for visualization and segmentation
        ee_pose = data_dict['observations']['ee_pose'][t]  # [x, y, z, rx, ry, rz, gripper_width]
        gripper_pose = ee_pose[:6]  # [x, y, z, rx, ry, rz]
        gripper_width = ee_pose[6]   # gripper width
        
        # Create transformation matrix for end-effector pose visualization
        ee_pose_mat = np.eye(4)
        ee_pose_mat[:3, 3] = gripper_pose[:3]  # position
        ee_pose_mat[:3, :3] = st.Rotation.from_euler('xyz', gripper_pose[3:6]).as_matrix()  # orientation

        # Transform to robot base frame (since visualization is in robot base frame)
        ee_pose_robot_base = np.linalg.inv(robot_base_in_world) @ ee_pose_mat
        
        # Update end-effector pose coordinate frame visualizations
        visualizer.update_triangle_mesh('ee_pose', tf=ee_pose_robot_base)
        
        # Extract gripper tool point cloud using original gripper pose (rotation is applied internally)
        if use_gripper_segmentation:
            # Transform gripper pose to robot base frame for segmentation
            gripper_pose_robot_base_translation = ee_pose_robot_base[:3, 3]
            gripper_pose_robot_base_rotation = st.Rotation.from_matrix(ee_pose_robot_base[:3, :3]).as_euler('xyz')
            gripper_pose_robot_base_6d = np.concatenate([gripper_pose_robot_base_translation, gripper_pose_robot_base_rotation])
            
            # Extract tool point cloud using original gripper pose (45-degree rotation applied internally)
            tool_pcd, tool_mask = extract_gripper_tool_pcd(
                pcd, gripper_pose_robot_base_6d, gripper_width,
                tool_length=0.15,  # Expected tool length (adjust as needed)
                tool_width=0.02,   # Expected tool width  
                # tool_width=0.15,   # Expected tool width  
                gripper_finger_length=0.1,  # Gripper finger length (adjust for your robot)
                safety_margin=0.00,  # Safety margin
                global_z_threshold=0.01  # Global Z threshold (adjust as needed)
            )
            
            # Get tool colors
            tool_colors = pcd_colors[tool_mask] if len(tool_pcd) > 0 else np.zeros((0, 3))
            
            # Get environment points (everything not in tool)
            env_mask = ~tool_mask
            env_pcd = pcd[env_mask]
            env_colors = pcd_colors[env_mask]

            # print floor env pcd z stats (min, max, mean, 95 percentile, 99 percentile)
            if len(env_pcd) > 0:
                floor_pcd = env_pcd[env_pcd[:, 2] < 0.03]
                print(f"Floor Env PCD Z stats: min={np.min(floor_pcd[:, 2]):.4f}, max={np.max(floor_pcd[:, 2]):.4f}, mean={np.mean(floor_pcd[:, 2]):.4f}, 95th={np.percentile(floor_pcd[:, 2], 95):.4f}, 99th={np.percentile(floor_pcd[:, 2], 99):.4f}")

            # Apply ENV_BOUNDARIES to environment points
            env_in_bounds = (
                (env_pcd[:, 0] >= ENV_BOUNDARIES['x_lower']) & (env_pcd[:, 0] <= ENV_BOUNDARIES['x_upper']) &
                (env_pcd[:, 1] >= ENV_BOUNDARIES['y_lower']) & (env_pcd[:, 1] <= ENV_BOUNDARIES['y_upper']) &
                (env_pcd[:, 2] >= ENV_BOUNDARIES['z_lower']) & (env_pcd[:, 2] <= ENV_BOUNDARIES['z_upper'])
            )
            env_pcd = env_pcd[env_in_bounds]
            env_colors = env_colors[env_in_bounds]
            
            # Apply downsampling if specified
            if downsample_obj_points is not None:
                raw_tool_pcd_size = len(tool_pcd)
                print(f"Raw tool point cloud size: {raw_tool_pcd_size}")
                # Record tool point count before downsampling
                tool_point_counts.append(raw_tool_pcd_size)
                frame_numbers.append(t)
                tool_pcd, tool_colors = sample_or_pad_pointcloud(tool_pcd, tool_colors, downsample_obj_points)
            if downsample_env_points is not None:
                env_pcd, env_colors = sample_or_pad_pointcloud(env_pcd, env_colors, downsample_env_points)
        
        # Apply color segmentation if enabled
        if apply_color_segmentation:
            object_pointcloud, env_pointcloud, object_colors, env_colors = segment_pointcloud_by_color(
                pcd, pcd_colors,
                obj_hue_range=obj_hue_range,
                obj_saturation_range=obj_saturation_range,
                obj_value_range=obj_value_range,
                env_hue_range=env_hue_range,
                env_saturation_range=env_saturation_range,
                env_value_range=env_value_range,
                object_boundaries=OBJECT_BOUNDARIES,
                env_boundaries=ENV_BOUNDARIES
            )
            
            # Apply downsampling if specified
            if downsample_obj_points is not None:
                object_pointcloud, object_colors = sample_or_pad_pointcloud(object_pointcloud, object_colors, downsample_obj_points)
            if downsample_env_points is not None:
                env_pointcloud, env_colors = sample_or_pad_pointcloud(env_pointcloud, env_colors, downsample_env_points)
            
            if vis_only_object:
                # Show only object points
                if len(object_pointcloud) > 0:
                    object_pcd_o3d = np2o3d(object_pointcloud, object_colors)
                    visualizer.update_pcd(object_pcd_o3d, 'pcd')
            elif vis_segmented_separately:
                # Show object and environment points separately
                if len(object_pointcloud) > 0:
                    object_pcd_o3d = np2o3d(object_pointcloud, object_colors)
                    visualizer.update_pcd(object_pcd_o3d, 'object_pcd')
                
                if len(env_pointcloud) > 0:
                    env_pcd_o3d = np2o3d(env_pointcloud, env_colors)
                    visualizer.update_pcd(env_pcd_o3d, 'env_pcd')
            else:
                # Show all segmented points in one view
                all_segmented_points = np.concatenate([object_pointcloud, env_pointcloud], axis=0) if len(object_pointcloud) > 0 and len(env_pointcloud) > 0 else \
                                     object_pointcloud if len(object_pointcloud) > 0 else env_pointcloud
                all_segmented_colors = np.concatenate([object_colors, env_colors], axis=0) if len(object_pointcloud) > 0 and len(env_pointcloud) > 0 else \
                                     object_colors if len(object_pointcloud) > 0 else env_colors
                if len(all_segmented_points) > 0:
                    combined_pcd_o3d = np2o3d(all_segmented_points, all_segmented_colors)
                    visualizer.update_pcd(combined_pcd_o3d, 'pcd')
            
            # Print statistics
            total_points = len(pcd)
            object_ratio = len(object_pointcloud) / total_points if total_points > 0 else 0
            env_ratio = len(env_pointcloud) / total_points if total_points > 0 else 0
            downsample_info = ""
            if downsample_obj_points is not None or downsample_env_points is not None:
                downsample_info = f" (downsampled to obj={downsample_obj_points}, env={downsample_env_points})"
            print(f"Frame {t}: Total: {total_points}, Object: {len(object_pointcloud)} ({object_ratio:.1%}), Environment: {len(env_pointcloud)} ({env_ratio:.1%}){downsample_info}")
        elif use_gripper_segmentation:
            # Use gripper-based segmentation to visualize tool vs environment
            if vis_only_object:
                # Show only tool points
                if len(tool_pcd) > 0:
                    tool_pcd_o3d = np2o3d(tool_pcd, tool_colors)
                    visualizer.update_pcd(tool_pcd_o3d, 'pcd')
            elif vis_segmented_separately:
                # Show tool and environment points separately
                if len(tool_pcd) > 0:
                    tool_pcd_o3d = np2o3d(tool_pcd, tool_colors)
                    visualizer.update_pcd(tool_pcd_o3d, 'tool_pcd')
                
                if len(env_pcd) > 0:
                    env_pcd_o3d = np2o3d(env_pcd, env_colors)
                    visualizer.update_pcd(env_pcd_o3d, 'env_pcd')
            else:
                # Show all points with different colors for tool vs environment
                if len(tool_pcd) > 0 and len(env_pcd) > 0:
                    # Color tool points in red, environment in original colors
                    tool_colors_highlight = np.ones((len(tool_pcd), 3)) * [1.0, 0.0, 0.0]  # Red for tool
                    all_points = np.concatenate([tool_pcd, env_pcd], axis=0)
                    all_colors = np.concatenate([tool_colors_highlight, env_colors], axis=0)
                    combined_pcd_o3d = np2o3d(all_points, all_colors)
                    visualizer.update_pcd(combined_pcd_o3d, 'pcd')
                else:
                    pcd_o3d = np2o3d(pcd, pcd_colors)
                    visualizer.update_pcd(pcd_o3d, 'pcd')
            
            # Print gripper-based segmentation statistics
            total_points = len(pcd)
            tool_ratio = len(tool_pcd) / total_points if total_points > 0 else 0
            env_ratio = len(env_pcd) / total_points if total_points > 0 else 0
            ee_pose = data_dict['observations']['ee_pose'][t]
            gripper_width = ee_pose[6]
            downsample_info = ""
            if downsample_obj_points is not None or downsample_env_points is not None:
                downsample_info = f" (downsampled to obj={downsample_obj_points}, env={downsample_env_points})"
            print(f"Frame {t}: Total: {total_points}, Tool: {len(tool_pcd)} ({tool_ratio:.1%}), Environment: {len(env_pcd)} ({env_ratio:.1%}), Gripper width: {gripper_width:.3f}{downsample_info}")
        else:
            # Original visualization without segmentation
            pcd_o3d = np2o3d(pcd, pcd_colors)
            visualizer.update_pcd(pcd_o3d, 'pcd')
        visualizer.update_triangle_mesh('front', tf=np.linalg.inv(extrinsics[0]))
        visualizer.update_triangle_mesh('left', tf=np.linalg.inv(extrinsics[1]))
        visualizer.update_triangle_mesh('right', tf=np.linalg.inv(extrinsics[2]))

        # left_finger_pose, right_finger_pose = get_finger_poses(
        #     data_dict['observations']['left_finger_pos'][t],
        #     data_dict['observations']['right_finger_pos'][t],
        #     data_dict['observations']['joint_pos'][t][-1])
        # visualizer.update_triangle_mesh('left_finger', tf=left_finger_pose)
        # visualizer.update_triangle_mesh('right_finger', tf=right_finger_pose)
        
        # visualize robot
        if vis_robot:
            robot_meshes = kin_helper.gen_robot_meshes(qpos = data_dict['observations']['full_joint_pos'][t])
            for m_i, mesh in enumerate(robot_meshes):
                visualizer.update_custom_mesh(mesh, f'mesh_{m_i}')
            
        if vis_action:
            # update action box
            t_start = t
            t_end = min(t_start + action_horizon, T)
            # ee_target_pose = data_dict['cartesian_action'][t_start:t_end] # (horizon, 7)
            ee_target_pose = data_dict['observations']['ee_pose'][t_start:t_end] # (horizon, 7)
            ee_target_pose_mat = np.tile(np.eye(4)[None], (t_end - t_start, 1, 1))
            ee_target_pose_mat[:, :3, 3] = ee_target_pose[:, :3]
            ee_target_pose_mat[:, :3, :3] = st.Rotation.from_euler('xyz', ee_target_pose[:, 3:6]).as_matrix()
            ee_target_pose_mat = np.linalg.inv(robot_base_in_world) @ ee_target_pose_mat
            for a_i in range(t_end - t_start):
                visualizer.update_triangle_mesh(f'action_{a_i}', tf=ee_target_pose_mat[a_i])
        
        visualizer.render()

# Generate and save plot of tool point counts vs frame
if len(tool_point_counts) > 0:
    plt.figure(figsize=(12, 6))
    plt.plot(frame_numbers, tool_point_counts, marker='o', linestyle='-', linewidth=2, markersize=4)
    plt.xlabel('Frame Number', fontsize=12)
    plt.ylabel('Number of Tool Points (Before Downsampling)', fontsize=12)
    plt.title('Tool Point Cloud Size vs Frame Number', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plot_filename = os.path.join(data_dir, 'tool_points_vs_frame.png')
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_filename}")
    
    # Print statistics
    print(f"\nTool Point Count Statistics:")
    print(f"  Mean: {np.mean(tool_point_counts):.2f}")
    print(f"  Median: {np.median(tool_point_counts):.2f}")
    print(f"  Min: {np.min(tool_point_counts)}")
    print(f"  Max: {np.max(tool_point_counts)}")
    print(f"  Std Dev: {np.std(tool_point_counts):.2f}")
    
    plt.show()
else:
    print("\nNo tool point data collected. Make sure use_gripper_segmentation=True and downsample_obj_points is set.")
