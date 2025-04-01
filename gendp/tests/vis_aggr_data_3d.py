#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
from tqdm import tqdm
from matplotlib import colormaps
from gendp.common.data_utils import load_dict_from_hdf5
from gendp.common.kinematics_utils import KinHelper
from d3fields.utils.draw_utils import aggr_point_cloud_from_data, np2o3d, o3dVisualizer, ImgEncoding, ExtriConvention
import scipy.spatial.transform as st

from vis_utils import get_finger_poses 

### hyper param
epi_range = [2]
vis_robot = False
vis_action = True
curr_dir = os.path.dirname(os.path.abspath(__file__))
# data_dir = f'{curr_dir}/../../data/sapien_demo/pencil_insertion_demo'
data_dir = f'{curr_dir}/../../data/polymetis/screwdriver_short'
robot_name = 'panda'
# cam_keys = ['right_bottom_view', 'left_bottom_view', 'right_top_view', 'left_top_view']
cam_keys = ['camera_wrist', 'camera_fixed']

### create visualizer
visualizer = o3dVisualizer()
visualizer.start()

### create kinematics helper
kin_helper = KinHelper(robot_name='panda')

for i in tqdm(epi_range):
    data_path = f'{data_dir}/episode_{i}.hdf5'

    data_dict, _ = load_dict_from_hdf5(data_path)

    # add meshes to visualize actions
    if vis_action:
        # init_cart = data_dict['cartesian_action'][0] # (horizon, 7)
        init_cart = data_dict['observations']['ee_pos'][0]
        action_horizon = init_cart.shape[0]
        action_cm = colormaps.get_cmap('plasma')
        action_colors = action_cm(np.linspace(0, 1, init_cart.shape[0], endpoint=True))[:, :3] # (horizon, 3)
        for a_i in range(action_horizon):
            visualizer.add_triangle_mesh('sphere', f'action_{a_i}', action_colors[a_i], radius=0.01)
    
    visualizer.add_triangle_mesh('origin', 'base', size=0.2)
    visualizer.add_triangle_mesh('origin', 'wrist')
    visualizer.add_triangle_mesh('origin', 'fixed')
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
        # Manual offset for fixed camera extrinsics
        # TODO: Change the fixed camera tf directly when collecting data
        extrinsics[1, 1, 3] -= 0.02
        extrinsics[1, 0, 3] += 0.01
        # print(extrinsics)
        boundaries = {
            'x_lower': 0.2,
            'x_upper': 0.8,
            'y_lower': -0.4,
            'y_upper': 0.4,
            'z_lower': 0.03,
            'z_upper': 0.7,
        }
        pcd, pcd_colors = aggr_point_cloud_from_data(colors,
                                                     depths,
                                                     intrinsics,
                                                     extrinsics,
                                                     downsample=False,
                                                     out_o3d=False,
                                                     boundaries=boundaries,
                                                    #  color_fmt=ImgEncoding.BGR_UINT8,
                                                    #  pose_fmt=ExtriConvention.CAM_IN_WORLD
                                                     )
        pcd = np.linalg.inv(robot_base_in_world) @ np.concatenate([pcd, np.ones((pcd.shape[0], 1))], axis=-1).T
        pcd = pcd.T[:, :3]
        pcd = np2o3d(pcd, pcd_colors)
        visualizer.update_pcd(pcd, 'pcd')
        visualizer.update_triangle_mesh('wrist', tf=np.linalg.inv(extrinsics[0]))
        visualizer.update_triangle_mesh('fixed', tf=np.linalg.inv(extrinsics[1]))

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
            ee_target_pose = data_dict['observations']['ee_pos'][t_start:t_end] # (horizon, 7)
            ee_target_pose_mat = np.tile(np.eye(4)[None], (t_end - t_start, 1, 1))
            ee_target_pose_mat[:, :3, 3] = ee_target_pose[:, :3]
            ee_target_pose_mat[:, :3, :3] = st.Rotation.from_euler('xyz', ee_target_pose[:, 3:6]).as_matrix()
            ee_target_pose_mat = np.linalg.inv(robot_base_in_world) @ ee_target_pose_mat
            for a_i in range(t_end - t_start):
                visualizer.update_triangle_mesh(f'action_{a_i}', tf=ee_target_pose_mat[a_i])
        
        visualizer.render()
