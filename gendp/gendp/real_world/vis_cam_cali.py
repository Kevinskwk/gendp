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
        cam_front_extri = get_extrinsic([0.8425395551524414, -0.23980856223114248, 0.32430529343304803],
                                    # [-0.7436835728382364, -0.42678910445821283, 0.2849678185522488, 0.42846137071605955])
                                    [-0.7510188, -0.41374503, 0.27744673, 0.43336949])
        cam_left_extri = get_extrinsic([0.3015062294276259, -0.2731493583749173, 0.12464828611110033],
                                    # [-0.7461453297553833, 0.22962820091980163, -0.20344921784453623, 0.5908861582276331])
                                    [-0.75190061, 0.21001771, -0.1879119, 0.59600936])
        cam_right_extri = get_extrinsic([0.33561416964601004, 0.5235239893129717, 0.1137736727084544],
                                    [-0.14008225307645683, 0.7060390316876065, -0.6806702770786354, 0.13628581000362547])
        extrinsics = np.stack([cam_front_extri, cam_left_extri, cam_right_extri])

        # Setup static geometries
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        ee_pose = ([0.392823, 0.015103, 0.3365602-0.14], [3.14159, 0.00, 0.00])
        ee_rot = st.Rotation.from_euler('xyz', ee_pose[1]).as_matrix()
        ee = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        ee.rotate(ee_rot, center=(0, 0, 0))
        ee.translate(ee_pose[0])
        ee_1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        ee_1_rot = st.Rotation.from_euler('xyz', [0.00, 0.00, 1.5708]).as_matrix()
        ee_1.rotate(ee_1_rot, center=(0, 0, 0))
        ee_1.translate(ee_pose[0])
        floor = o3d.geometry.TriangleMesh.create_box(width=2.0, height=2.0, depth=0.01)
        floor.translate([-1.0, -1.0, -0.01])
        floor.paint_uniform_color([0.8, 0.8, 0.8])

        if realtime:
            # Setup visualizer for real-time display
            vis = o3d.visualization.Visualizer()
            vis.create_window("Real-time Point Cloud Calibration", width=1280, height=720)
            
            # Add static geometries
            vis.add_geometry(origin)
            vis.add_geometry(ee)
            vis.add_geometry(ee_1)

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
                    new_pcd = aggr_point_cloud_from_data(colors=colors, depths=depths, Ks=intrinsics, poses=extrinsics, downsample=True, boundaries=boundaries)
                    print(f"Current point cloud has {len(new_pcd.points)} points after downsampling.")
                    
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
                    o3d.visualization.draw_geometries([pcd, origin, ee, ee_1, floor])

            pcd = aggr_point_cloud_from_data(colors=colors, depths=depths, Ks=intrinsics, poses=extrinsics, downsample=True, boundaries=boundaries)
            print(f"Final point cloud has {len(pcd.points)} points after downsampling.")
            o3d.visualization.draw_geometries([pcd, origin, ee, ee_1, floor])

            # print(np.asarray(pcd.points).shape)
            # np.save('obj_pcd/toilet_paper.npy', arr=np.asarray(pcd.points))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterative', action='store_true', help='Show point clouds from each camera individually')
    parser.add_argument('--realtime', action='store_true', help='Enable real-time point cloud visualization')
    args = parser.parse_args()
    
    visualize_calibration_result(iterative=args.iterative, realtime=args.realtime)
