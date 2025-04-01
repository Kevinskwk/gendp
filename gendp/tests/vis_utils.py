import numpy as np
import scipy.spatial.transform as st

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