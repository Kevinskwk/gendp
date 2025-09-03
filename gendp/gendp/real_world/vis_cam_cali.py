import time
import argparse
import scipy.spatial.transform as st

import numpy as np
import open3d as o3d

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from d3fields.utils.draw_utils import aggr_point_cloud_from_data
from gendp.real_world.multi_realsense import MultiRealsense
from gendp.common.cv2_util import get_extrinsic


boundaries = {
            'x_lower': 0.2,
            'x_upper': 0.8,
            'y_lower': -0.4,
            'y_upper': 0.4,
            'z_lower': 0.0,
            'z_upper': 0.5,
        }

def visualize_calibration_result(iterative=False):
    with MultiRealsense(
        resolution=(640, 480),
        put_downsample=False,
        enable_color=True,
        enable_depth=True,
        enable_infrared=False,
        verbose=False
        ) as realsense:
        realsense.set_exposure(200, 64)
        realsense.set_white_balance(2800)
        realsense.set_depth_preset('High Density')
        realsense.set_depth_exposure(7000, 16)
        for _ in range(30):
            out = realsense.get()
            time.sleep(0.1)

            # # visualize depth
            # import matplotlib.cm as cm
            # import cv2
            # depth_1 = out[1]['depth'] / 1000.
            # cmap = cm.get_cmap('jet')
            # depth_min = 0.2
            # depth_max = 0.8
            # depth_1 = (depth_1 - depth_min) / (depth_max - depth_min)
            # depth_1 = np.clip(depth_1, 0, 1)
            # depth_1_vis = cmap(depth_1.reshape(-1))[..., :3].reshape(depth_1.shape + (3,))
            # depth_1_vis = depth_1_vis[..., ::-1]
            # cv2.imshow('depth_1', depth_1_vis)
            # cv2.waitKey(1)

        colors = np.stack(value['color'] for value in out.values())[..., ::-1]
        depths = np.stack(value['depth'] for value in out.values()) / 1000.
        intrinsics = np.stack(value['intrinsics'] for value in out.values())
        extrinsics = np.stack(value['extrinsics'] for value in out.values())
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        # cam_front_extri = get_extrinsic([0.8408891228960659, -0.2306640654217388, 0.32780918960803124],
        #                             [-0.7493846612312357, -0.41228123056776256, 0.28491736181125427, 0.4327457837707866])
        # cam_left_extri = get_extrinsic([0.27804807679768973, -0.23545503302949033, 0.13971720258705824],
        #                             [-0.6954162447452916, 0.16984997468823826, -0.2334766373619014, 0.6580546272528907])
        cam_front_extri = get_extrinsic([0.8489497156928908, -0.22562991111452887, 0.3314941131288296],
                                    [-0.7504308292249032, -0.41235638789732665, 0.2876578890353489, 0.4290323050596778])
        cam_left_extri = get_extrinsic([0.29669299363755186, -0.26695822263290947, 0.13707501874264605],
                                    [-0.7464977308352696, 0.23098579079830828, -0.20289848726451698, 0.5901007593393219])
        cam_right_extri = get_extrinsic([0.3281305272599807, 0.5200284215384193, 0.12379313280393066],
                                    [-0.14095227077856376, 0.7139867190308669, -0.6725150321346475, 0.13445800073940598])

        extrinsics = np.stack([cam_front_extri, cam_left_extri, cam_right_extri])

        # visualize ee_pose
        ee_pose = ([0.421625, 0.012471, 0.287905-0.14], [3.14, 0.00, 0.0])
        ee_rot = st.Rotation.from_euler('xyz', ee_pose[1]).as_matrix()
        ee = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        ee.rotate(ee_rot, center=(0, 0, 0))
        ee.translate(ee_pose[0])
        
        if iterative:
            for i in range(colors.shape[0]):
                pcd = aggr_point_cloud_from_data(colors=colors[i:i+1], depths=depths[i:i+1], Ks=intrinsics[i:i+1], poses=extrinsics[i:i+1], downsample=False, boundaries=boundaries)
                o3d.visualization.draw_geometries([pcd, origin, ee])

        pcd = aggr_point_cloud_from_data(colors=colors, depths=depths, Ks=intrinsics, poses=extrinsics, downsample=False, boundaries=boundaries)
        o3d.visualization.draw_geometries([pcd, origin, ee])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterative', action='store_true')
    args = parser.parse_args()
    visualize_calibration_result(args.iterative)
