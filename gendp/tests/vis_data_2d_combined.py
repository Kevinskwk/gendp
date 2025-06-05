import os
# import numpy as np
# import torch
# from tqdm import tqdm
# from matplotlib import colormaps
from gendp.common.data_utils import load_dict_from_hdf5
# from gendp.common.kinematics_utils import KinHelper
# from d3fields.utils.draw_utils import aggr_point_cloud_from_data, np2o3d, o3dVisualizer, ImgEncoding
# from d3fields.fusion import Fusion
# import scipy.spatial.transform as st
from gendp.common.cv2_util import combine_image_arrays_to_video_2x3

epi_range = [0]

curr_dir = os.path.dirname(os.path.abspath(__file__))
# data_dir = f'{curr_dir}/../../data/sapien_demo/pencil_2_demo_100'
data_dir = f'{curr_dir}/../../data/polymetis/screwdriver_aligning'

for i in epi_range:
    print(f'visualizing episode {i}')
    data_path = f'{data_dir}/episode_{i}.hdf5'

    data_dict, _ = load_dict_from_hdf5(data_path)

    print(list(data_dict.keys()))
    print(list(data_dict['observations'].keys()))
    print(data_dict['observations']['ee_pos'][0])
    print(data_dict['observations']['joint_vel'][0])

    # tactile_left = data_dict['observations']['tactile']['tactile_left']
    # tactile_right = data_dict['observations']['tactile']['tactile_right']

    # combine_image_arrays_to_video_2x3(
    #     data_dict['observations']['images']['camera_wrist_depth'],
    #     data_dict['observations']['images']['camera_fixed_depth'],
    #     data_dict['observations']['images']['camera_wrist_color'],
    #     data_dict['observations']['images']['camera_fixed_color'],
    #     data_dict['observations']['tactile']['tactile_left'],
    #     data_dict['observations']['tactile']['tactile_right'])
