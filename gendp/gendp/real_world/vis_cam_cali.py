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


# boundaries = {
#             'x_lower': 0.42,
#             'x_upper': 0.58,
#             'y_lower': -0.08,
#             'y_upper': 0.08,
#             'z_lower': 0.0285,
#             'z_upper': 0.3,
#         }

boundaries = {
            'x_lower': 0.2,
            'x_upper': 0.8,
            'y_lower': -0.4,
            'y_upper': 0.4,
            'z_lower': -0.1,
            'z_upper': 0.5,
        }

def visualize_calibration_result(iterative=False, realtime=False):
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
        
        # Wait for camera to stabilize
        for _ in range(30):
            out = realsense.get()
            time.sleep(0.1)

        # Setup camera extrinsics
        cam_front_extri = get_extrinsic([0.8489497156928908, -0.22562991111452887, 0.3314941131288296],
                                    [-0.7504308292249032, -0.41235638789732665, 0.2876578890353489, 0.4290323050596778])
        cam_left_extri = get_extrinsic(
            [0.31669299363755186, -0.27395822263290947, 0.12707501874264605],
            [-0.7435734326325784, 0.20794098553588683, -0.17857879530445217, 0.6098923763132142])
        cam_right_extri = get_extrinsic([0.3281305272599807, 0.5200284215384193, 0.12379313280393066],
                                    [-0.14095227077856376, 0.7139867190308669, -0.6725150321346475, 0.13445800073940598])
        extrinsics = np.stack([cam_front_extri, cam_left_extri, cam_right_extri])

        # Setup static geometries
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        ee_pose = ([0.421625, 0.012471, 0.287905-0.14-0.14], [0.00, 0.00, 0.0])
        ee_rot = st.Rotation.from_euler('xyz', ee_pose[1]).as_matrix()
        ee = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        ee.rotate(ee_rot, center=(0, 0, 0))
        ee.translate(ee_pose[0])

        if realtime:
            # Setup visualizer for real-time display
            vis = o3d.visualization.Visualizer()
            vis.create_window("Real-time Point Cloud Calibration", width=1280, height=720)
            
            # Add static geometries
            vis.add_geometry(origin)
            vis.add_geometry(ee)
            
            # Initialize point cloud geometry
            pcd = o3d.geometry.PointCloud()
            vis.add_geometry(pcd)
            
            print("Starting real-time visualization. Press 'Q' or close window to exit.")
            
            try:
                while vis.poll_events():
                    # Capture new frame
                    out = realsense.get()
                    
                    colors = np.stack(value['color'] for value in out.values())[..., ::-1]
                    depths = np.stack(value['depth'] for value in out.values()) / 1000.
                    intrinsics = np.stack(value['intrinsics'] for value in out.values())
                    
                    # Generate new point cloud
                    new_pcd = aggr_point_cloud_from_data(colors=colors, depths=depths, Ks=intrinsics, poses=extrinsics, downsample=False, boundaries=boundaries)
                    
                    # Update point cloud geometry
                    pcd.points = new_pcd.points
                    pcd.colors = new_pcd.colors
                    
                    # Update visualization
                    vis.update_geometry(pcd)
                    vis.update_renderer()
                    
                    # Small delay to control frame rate
                    time.sleep(0.033)  # ~30 FPS
                    
            except KeyboardInterrupt:
                print("\nStopped by user")
            finally:
                vis.destroy_window()
        else:
            # Static mode - capture once and display
            out = realsense.get()
            colors = np.stack(value['color'] for value in out.values())[..., ::-1]
            depths = np.stack(value['depth'] for value in out.values()) / 1000.
            intrinsics = np.stack(value['intrinsics'] for value in out.values())
            
            if iterative:
                for i in range(colors.shape[0]):
                    pcd = aggr_point_cloud_from_data(colors=colors[i:i+1], depths=depths[i:i+1], Ks=intrinsics[i:i+1], poses=extrinsics[i:i+1], downsample=False, boundaries=boundaries)
                    o3d.visualization.draw_geometries([pcd, origin, ee])

            pcd = aggr_point_cloud_from_data(colors=colors, depths=depths, Ks=intrinsics, poses=extrinsics, downsample=False, boundaries=boundaries)
            o3d.visualization.draw_geometries([pcd, origin, ee])

            # print(np.asarray(pcd.points).shape)
            # np.save('obj_pcd/toilet_paper.npy', arr=np.asarray(pcd.points))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterative', action='store_true', help='Show point clouds from each camera individually')
    parser.add_argument('--realtime', action='store_true', help='Enable real-time point cloud visualization')
    args = parser.parse_args()
    
    visualize_calibration_result(iterative=args.iterative, realtime=args.realtime)
