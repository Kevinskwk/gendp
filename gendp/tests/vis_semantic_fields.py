#!/usr/bin/env python
# coding: utf-8
import os
import time
import numpy as np
import torch
from tqdm import tqdm
from matplotlib import colormaps
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gendp.common.data_utils import load_dict_from_hdf5, d3fields_proc
from gendp.common.kinematics_utils import KinHelper
from d3fields.utils.draw_utils import aggr_point_cloud_from_data, np2o3d, o3dVisualizer, ImgEncoding
from d3fields.fusion import Fusion
import scipy.spatial.transform as st

### hyper param
# epi_range = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# epi_range = [0, 1, 2, 3, 4, 5, 8, 9]
# epi_range = [0, 15, 30, 45, 60, 75]
epi_range = [23]
# epi_range = [20, 21, 22, 23]
vis_robot = False
vis_action = False
compute_feat_com = False
curr_dir = os.path.dirname(os.path.abspath(__file__))
# data_dir = f'{curr_dir}/../../data/sapien_demo/pencil_insertion_demo'
# data_dir = f'{curr_dir}/../../data/crayon_pickup_v2'
data_dir = f'{curr_dir}/../../data/pencil_pickup'
# data_dir = f'{curr_dir}/../../data/crayon_pickup_left'
# data_dir = f'{curr_dir}/../../data/peeler_carrot'
# data_dir = f'{curr_dir}/../../data/peeler_test'
# data_dir = f'{curr_dir}/../../data/scraper_z73'
# data_dir = f'{curr_dir}/../../data/scrap_tool_test'
# data_dir = f'{curr_dir}/../../data/outputs/2025.10.02/19.47.25_train_diffusion_unet_hybrid_scraper_real'
robot_name = 'panda'
# cam_keys = ['right_bottom_view', 'left_bottom_view', 'right_top_view', 'left_top_view']
cam_keys = ['camera_front', 'camera_left', 'camera_right']

### set up shape_meta
shape_meta = {
    'shape': [6, 868],
    'type': 'spatial',
    'info': {
        'reference_frame': 'world',
        'distill_dino': True,
        # 'distill_obj': 'peeler_v2',
        # 'distill_obj': 'pencil',
        'distill_obj': 'crayon_v4',
        # 'distill_obj': 'pencil_real',
        # 'distill_obj': 'scraper',
        # 'query_text': 'dark green scraper tool',
        # 'query_text': 'crayon',
        # 'query_text': 'peeler',
        'query_text': 'carrot',
        'sam_threshold': 0.1,
        # 'view_keys': ['left_bottom_view', 'right_bottom_view', 'left_top_view', 'right_top_view'],
        'view_keys': ['front', 'left', 'right'],
        # 'N_gripper': 100,
        'N_gripper': 10,
        'N_obj': 256,
        'N_env': 512,
        'boundaries': {
            'x_lower': 0.3,
            'x_upper': 0.7,
            'y_lower': -0.2,
            'y_upper': 0.15,
            'z_lower': -0.03,
            'z_upper': 0.4,
        },
        'obj_boundaries': {
            'x_lower': 0.3,
            'x_upper': 0.7,
            'y_lower': -0.2,
            'y_upper': 0.2,
            'z_lower': 0.0,
            'z_upper': 0.4
        },
        # crayon_pickup
        'env_boundaries': {
            'x_lower': 0.41,
            'x_upper': 0.7,
            # 'x_upper': 0.57,
            'y_lower': -0.14,
            'y_upper': 0.13,
            'z_lower': 0.1,
            # 'z_lower': -0.03,
            'z_upper': 0.17
        },
        # crayon draw
        # 'env_boundaries': {
        #     'x_lower': 0.36,
        #     'x_upper': 0.53,
        #     'y_lower': -0.1,
        #     'y_upper': 0.05,
        #     'z_lower': -0.03,
        #     'z_upper': 0.1
        # },
        # peeler
        # 'env_boundaries': {
        #     'x_lower': 0.3,
        #     'x_upper': 0.6,
        #     'y_lower': -0.2,
        #     'y_upper': 0.2,
        #     'z_lower': 0.0,
        #     'z_upper': 0.1
        # },
        'resize_ratio': 0.5
    }
}

### create kinematics helper
kin_helper = KinHelper(robot_name='panda')

### create fusion
fusion = Fusion(num_cam=len(cam_keys), dtype=torch.float16, device='cuda:0')

### create visualizer
visualizer = o3dVisualizer()
visualizer.start()

for i in tqdm(epi_range):
    data_path = f'{data_dir}/episode_{i}.hdf5'

    data_dict, _ = load_dict_from_hdf5(data_path)
    
    # Track center of mass for this episode
    y_com_list = []
    y_center_list = []
    y_delta_com_list = []

    # add meshes to visualize actions
    if vis_action:
        # init_cart = data_dict['cartesian_action'][0] # (horizon, 7)
        init_cart = data_dict['observations']['ee_pose'][0]
        action_horizon = init_cart.shape[0]
        action_cm = colormaps.get_cmap('plasma')
        action_colors = action_cm(np.linspace(0, 1, init_cart.shape[0], endpoint=True))[:, :3] # (horizon, 3)
        for a_i in range(action_horizon):
            visualizer.add_triangle_mesh('sphere', f'action_{a_i}', action_colors[a_i], radius=0.01)
    
    T = data_dict['observations']['images'][f'{cam_keys[0]}_color'].shape[0]
    # T = 30
    robot_base_in_world_seq = data_dict['observations']['robot_base_pose_in_world'][()]
    
    for t in tqdm(range(T)):
        # visualize point cloud
        robot_base_in_world = robot_base_in_world_seq[t]
        colors = np.stack([data_dict['observations']['images'][f'{cam_key}_color'][t:t+1] for cam_key in cam_keys], axis=1) # (N, H, W, 3)
        depths = np.stack([data_dict['observations']['images'][f'{cam_key}_depth'][t:t+1] for cam_key in cam_keys], axis=1) / 1000. # (N, H, W)
        intrinsics = np.stack([data_dict['observations']['images'][f'{cam_key}_intrinsics'][t:t+1] for cam_key in cam_keys], axis=1)
        extrinsics = np.stack([data_dict['observations']['images'][f'{cam_key}_extrinsics'][t:t+1] for cam_key in cam_keys], axis=1)
        
        # Tune the extrinsics
        # pose_0 = np.linalg.inv(extrinsics[0, 2])
        # pose_0[0:3, 3] += np.array([-0.025, -0.01, -0.005])  # Adjust position
        # extrinsics[0, 2] = np.linalg.inv(pose_0)

        ee_poses = data_dict['observations']['ee_pose'][t:t+1]
        t0 = time.time()
        result = d3fields_proc(
            fusion=fusion,
            shape_meta=shape_meta,
            color_seq=colors,
            depth_seq=depths,
            extri_seq=extrinsics,
            intri_seq=intrinsics,
            robot_base_pose_in_world_seq=robot_base_in_world_seq,
            teleop_robot=kin_helper,
            qpos_seq=data_dict['observations']['full_joint_pos'][t:t+1],
            exclude_threshold=0.01,
            use_obj_bg_seg=True,
            gripper_pose_seq=ee_poses,
            seg_method='gripper_crop',
            # seg_method='d3field_feat',
            # seg_method='sam',
            seg_params={
                'tool_length': 0.15,
                'tool_width': 0.04,  # crayon
                # 'tool_width': 0.1,  # peeler
                'gripper_finger_length': 0.1,
                'safety_margin': 0.0,
                # 'safety_margin': 0.003,
                'global_z_threshold' : 0.025,
                'auto_estimate_plane': False,
                'plane_margin': 0.012,
                'plane_percentile': 20,
                'ransac_iterations': 50,
                'ransac_distance_threshold': 0.01,
                'feat_threshold': 0.05,
                'use_any': True,
                'combine_with_gripper': True,
                'gripper_combine_mode': 'intersection',
                # 'reverse_selection': True,
            }
        )
        t1 = time.time()
        # print(f'Frame {t} processing time: {t1 - t0:.3f} seconds')
        
        # Unpack the returned values
        if len(result) == 7:
            pcd, pcd_feats, obj_pcd, obj_feats, bg_pcd, bg_feats, _ = result
            obj_pcd = obj_pcd[0]
            obj_feats = obj_feats[0]
            bg_pcd = bg_pcd[0]
            bg_feats = bg_feats[0]

            # print stats for obj_feats and bg_feats, including mean, min, max, percentiles
            # print(f'[Frame {t}] Obj feats: mean={obj_feats.mean(axis=0)}, min={obj_feats.min(axis=0)}, max={obj_feats.max(axis=0)}')
            # print(f'[Frame {t}] Obj feats: 25th={np.percentile(obj_feats, 25, axis=0)}, 50th={np.percentile(obj_feats, 50, axis=0)}, 75th={np.percentile(obj_feats, 75, axis=0)}')
            # print(f'[Frame {t}] Bg feats: mean={bg_feats.mean(axis=0)}, min={bg_feats.min(axis=0)}, max={bg_feats.max(axis=0)}')
            # print(f'[Frame {t}] Bg feats: 25th={np.percentile(bg_feats, 25, axis=0)}, 50th={np.percentile(bg_feats, 50, axis=0)}, 75th={np.percentile(bg_feats, 75, axis=0)}')
            # Transform object points to robot frame
            obj_pcd = np.linalg.inv(robot_base_in_world) @ np.concatenate([obj_pcd, np.ones((obj_pcd.shape[0], 1))], axis=-1).T
            obj_pcd = obj_pcd.T[:, :3]
            
            # Transform background points to robot frame
            bg_pcd = np.linalg.inv(robot_base_in_world) @ np.concatenate([bg_pcd, np.ones((bg_pcd.shape[0], 1))], axis=-1).T
            bg_pcd = bg_pcd.T[:, :3]
            
            # Use different colormaps for object and background
            obj_cmap = colormaps.get_cmap('viridis')
            # bg_cmap = colormaps.get_cmap('Reds')
            bg_cmap = colormaps.get_cmap('viridis')

            obj_colors = obj_cmap(obj_feats[:, 0])[:, :3]
            bg_colors = bg_cmap(bg_feats[:, 0])[:, :3]

            obj_pcd_o3d = np2o3d(obj_pcd, obj_colors)
            bg_pcd_o3d = np2o3d(bg_pcd, bg_colors)
            
            visualizer.update_pcd(obj_pcd_o3d, 'obj_pcd')
            visualizer.update_pcd(bg_pcd_o3d, 'bg_pcd')

            if compute_feat_com:
                # compute the distribution of bg feats along y axis
                y_coords = bg_pcd[:, 1]  # Extract y coordinates
                bg_feat_values = bg_feats[:, 0]  # Extract first feature dimension
                
                # Compute center of mass (weighted average) of feature intensity along y-axis
                total_intensity = bg_feat_values.sum()
                if total_intensity > 0:
                    y_center_of_mass = (y_coords * bg_feat_values).sum() / total_intensity
                    y_center = y_coords.mean()
                    y_com_list.append(y_center_of_mass)
                    y_center_list.append(y_center)
                    y_delta_com_list.append(y_center_of_mass - y_center)
                    # print(f'[Frame {t}] Background feature center of mass along Y axis: {y_center_of_mass:.4f}')
                else:
                    print(f'[Frame {t}] Warning: Total feature intensity is zero, cannot compute center of mass')
            
        else:
            # Fallback to original behavior if segmentation is not available
            pcd, pcd_feats, _ = result
            pcd = pcd[0]
            pcd_feats = pcd_feats[0]
            pcd = np.linalg.inv(robot_base_in_world) @ np.concatenate([pcd, np.ones((pcd.shape[0], 1))], axis=-1).T
            pcd = pcd.T[:, :3]
            feats_cmap = colormaps.get_cmap('viridis')
            pcd_colors = feats_cmap(pcd_feats[:, 0])[:, :3]
            pcd = np2o3d(pcd, pcd_colors)
            visualizer.update_pcd(pcd, 'pcd')
        
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
    
    # Print episode statistics
    if compute_feat_com:
        if y_com_list:
            avg_y_com = np.mean(y_com_list)
            std_y_com = np.std(y_com_list)
            avg_y_center = np.mean(y_center_list)
            std_y_center = np.std(y_center_list)
            avg_y_delta_com = np.mean(y_delta_com_list)
            std_y_delta_com = np.std(y_delta_com_list)
            print(f'\n[Episode {i}] Average Y-axis delta COM: {avg_y_delta_com:.4f} ± {std_y_delta_com:.4f} (over {len(y_delta_com_list)} frames)')
            # print(f'\n[Episode {i}] Average Y-axis point center: {avg_y_center:.4f} ± {std_y_center:.4f} (over {len(y_center_list)} frames)')
            # print(f'\n[Episode {i}] Average Y-axis feature COM: {avg_y_com:.4f} ± {std_y_com:.4f} (over {len(y_com_list)} frames)')
        else:
            print(f'\n[Episode {i}] No valid feature COM computed')
