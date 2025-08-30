import os
import numpy as np
# import torch
# from tqdm import tqdm
# from matplotlib import colormaps
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gendp.common.data_utils import load_dict_from_hdf5, d3fields_proc
# from gendp.common.kinematics_utils import KinHelper
# from d3fields.utils.draw_utils import aggr_point_cloud_from_data, np2o3d, o3dVisualizer, ImgEncoding
# from d3fields.fusion import Fusion
# import scipy.spatial.transform as st
from gendp.common.cv2_util import combine_image_arrays_to_video_2x3

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from contact_field.utils import visualize_tactile_shear_image

epi_range = [0]

curr_dir = os.path.dirname(os.path.abspath(__file__))
# data_dir = f'{curr_dir}/../../data/sapien_demo/pencil_2_demo_100'
data_dir = f'{curr_dir}/../../data/sim2real'

for i in epi_range:
    print(f'visualizing episode {i}')
    data_path = f'{data_dir}/episode_{i}.hdf5'

    data_dict, _ = load_dict_from_hdf5(data_path)

    print(list(data_dict.keys()))

    # tactile_left = data_dict['observations']['tactile']['tactile_left']
    # tactile_right = data_dict['observations']['tactile']['tactile_right']
    marker_flow = data_dict['observations']['marker_flow']['marker_flow']
    depth_max = marker_flow[:, :, :, :, 2].max()
    print(marker_flow[:, :, :, :, 2].max(), marker_flow[:, :, :, :, 2].min(), marker_flow[:, :, :, :, 2].mean())
    print(marker_flow[:, :, :, :, :2].max())
    marker_flow_left_imgs = [visualize_tactile_shear_image(
        -(mf[0, :, :, 2] - depth_max),
        mf[0, :, :, :2],
        normal_force_threshold=40,
        shear_force_threshold=10,
        resolution=60,
        paddings=[60, 80]) for mf in marker_flow]
    marker_flow_right_imgs = [visualize_tactile_shear_image(
        -(mf[1, :, :, 2] - depth_max),
        mf[1, :, :, :2],
        normal_force_threshold=40,
        shear_force_threshold=10,
        resolution=60,
        paddings=[60, 80]) for mf in marker_flow]
    # import pdb; pdb.set_trace()

    combine_image_arrays_to_video_2x3(
        data_dict['observations']['images']['camera_wrist_depth'],
        data_dict['observations']['images']['camera_left_depth'],
        data_dict['observations']['images']['camera_wrist_color'],
        data_dict['observations']['images']['camera_left_color'],
        np.stack(marker_flow_left_imgs),
        np.stack(marker_flow_right_imgs)
        # data_dict['observations']['tactile']['tactile_left'],
        # data_dict['observations']['tactile']['tactile_right']
    )
