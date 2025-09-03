import numpy as np
import scipy.spatial.transform as st
import cv2

def t_quat_to_matrix(pose):
    t = pose[:3]
    quat = pose[3:]
    matrix = np.eye(4)
    rot = st.Rotation.from_quat(quat)
    matrix[:3, :3] = rot.as_matrix()
    matrix[:3, 3] = t

    return matrix

'''
gripper state - openning distance - z position:
3 - 140mm - 0.17969
220 - 4mm - 0.2
225 - 0mm
183 - 30mm - 0.2

gripper-base to left inner finger:
open:
pos: 0, -0.095238, 0.13121
quat: 0.70711, 0, 0, 0.70711
close:
pos: 0, -0.030882, 0.1549
quat: 0.70711, 0, 0, 0.70711

left inner finger to left inner finger pad:
0, 0.045755, -0.02722 (y up)

actual distance from inner finger to gelsight: 60mm (need to add 14.25mm)
'''

def get_finger_to_pad_offset(gripper_state):
    # opening: 225 - 175: 0.7mm, 175 - 0: 0.6mm 
    # y: 225 - 175: 83.69, 175 - 0: 83.69 - 60
    gripper_state = np.clip(gripper_state, 0, 225)
    if gripper_state >= 175:
        openning = (225 - gripper_state) * 0.7
        y = 83.69
    else:
        openning = 35 + (175 - gripper_state) * 0.6
        y = 83.69 - (175 - gripper_state) * 23.69 / 175

    z = 97.22 - openning / 2
    
    return z / 1000, y / 1000

def get_finger_poses(left_base_pose, right_base_pose, gripper_state):
    """
    Calculate finger poses based on base link pose and gripper state
    
    Parameters:
    left_base_pose: [x, y, z, qx, qy, qz, qw] or transformation matrix
    right_base_pose: [x, y, z, qx, qy, qz, qw] or transformation matrix
    gripper_state: normalized value between 0.0 (fully open) and 1.0 (fully closed)
    
    Returns:
    finger1_pose, finger2_pose: transformation matrices for both fingers
    """
    # Convert base_pose to transformation matrix if needed
    if len(left_base_pose) == 7:  # If in [x, y, z, qx, qy, qz, qw] format
        left_base_tf = t_quat_to_matrix(left_base_pose)
        right_base_tf = t_quat_to_matrix(right_base_pose)
    else:
        left_base_tf = left_base_pose
        right_base_tf = right_base_pose
    
    z, y = get_finger_to_pad_offset(gripper_state)
    
    # Create finger transformation matrices (relative to base_link)
    finger_local = np.identity(4)
    finger_local[1, 3] = y
    finger_local[2, 3] = -z
    
    # Apply base link transformation
    left_finger_global = np.dot(left_base_tf, finger_local)
    right_finger_global = np.dot(right_base_tf, finger_local)
    
    return left_finger_global, right_finger_global

def segment_pointcloud_by_color(pcd, pcd_colors, 
                               obj_hue_range=(300, 30), 
                               obj_saturation_range=(30, 255),
                               obj_value_range=(30, 255),
                               env_hue_range=(80, 140),
                               env_saturation_range=(30, 255),
                               env_value_range=(30, 255),
                               object_boundaries=None,
                               env_boundaries=None):
    """
    Segment point cloud by color using HSV color space with separate spatial boundaries.
    
    Args:
        pcd: (N, 3) point cloud coordinates
        pcd_colors: (N, 3) RGB colors in [0, 1] range
        obj_hue_range: (min_hue, max_hue) for pink in degrees
        obj_saturation_range: (min_sat, max_sat) for pink (0-255)
        obj_value_range: (min_val, max_val) for pink (0-255)
        env_hue_range: (min_hue, max_hue) for green in degrees
        env_saturation_range: (min_sat, max_sat) for green (0-255)
        env_value_range: (min_val, max_val) for green (0-255)
        object_boundaries: dict with x_lower, x_upper, y_lower, y_upper, z_lower, z_upper for objects
        env_boundaries: dict with x_lower, x_upper, y_lower, y_upper, z_lower, z_upper for environment
    
    Returns:
        object_pointcloud: (M, 3) pink colored points within object boundaries
        env_pointcloud: (K, 3) green colored points within env boundaries
        object_colors: (M, 3) corresponding colors
        env_colors: (K, 3) corresponding colors
    """
    # Convert RGB to HSV
    rgb_uint8 = (pcd_colors * 255).astype(np.uint8)
    # Reshape for cv2 processing
    rgb_image = rgb_uint8.reshape(1, -1, 3)
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    hsv_values = hsv_image.reshape(-1, 3)
    
    # Extract hue, saturation, value
    hue = hsv_values[:, 0] * 2  # Convert to 0-360 range
    saturation = hsv_values[:, 1]
    value = hsv_values[:, 2]
    
    # Create masks for pink and green colors
    # Pink mask - handle wrapping around 0/360 for pink/magenta colors
    if obj_hue_range[0] > obj_hue_range[1]:  # Wraps around (e.g., 300-30)
        pink_hue_mask = (hue >= obj_hue_range[0]) | (hue <= obj_hue_range[1])
    else:
        pink_hue_mask = (hue >= obj_hue_range[0]) & (hue <= obj_hue_range[1])
    
    pink_mask = pink_hue_mask & \
                (saturation >= obj_saturation_range[0]) & (saturation <= obj_saturation_range[1]) & \
                (value >= obj_value_range[0]) & (value <= obj_value_range[1])
    
    # Green mask
    green_mask = (hue >= env_hue_range[0]) & (hue <= env_hue_range[1]) & \
                 (saturation >= env_saturation_range[0]) & (saturation <= env_saturation_range[1]) & \
                 (value >= env_value_range[0]) & (value <= env_value_range[1])
    
    # Apply spatial boundaries for objects (pink)
    if object_boundaries is not None:
        object_spatial_mask = (pcd[:, 0] >= object_boundaries['x_lower']) & (pcd[:, 0] <= object_boundaries['x_upper']) & \
                              (pcd[:, 1] >= object_boundaries['y_lower']) & (pcd[:, 1] <= object_boundaries['y_upper']) & \
                              (pcd[:, 2] >= object_boundaries['z_lower']) & (pcd[:, 2] <= object_boundaries['z_upper'])
        pink_mask = pink_mask & object_spatial_mask
    
    # Apply spatial boundaries for environment (green)
    if env_boundaries is not None:
        env_spatial_mask = (pcd[:, 0] >= env_boundaries['x_lower']) & (pcd[:, 0] <= env_boundaries['x_upper']) & \
                           (pcd[:, 1] >= env_boundaries['y_lower']) & (pcd[:, 1] <= env_boundaries['y_upper']) & \
                           (pcd[:, 2] >= env_boundaries['z_lower']) & (pcd[:, 2] <= env_boundaries['z_upper'])
        green_mask = green_mask & env_spatial_mask
    
    # Filter point clouds
    object_pointcloud = pcd[pink_mask]
    object_colors = pcd_colors[pink_mask]
    
    env_pointcloud = pcd[green_mask]
    env_colors = pcd_colors[green_mask]
    
    return object_pointcloud, env_pointcloud, object_colors, env_colors 