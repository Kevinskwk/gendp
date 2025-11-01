from typing import Dict, Callable, Tuple, Optional
import numpy as np
import cv2
import scipy.spatial.transform as st
from gendp.common.cv2_util import get_image_transform
from gendp.common.data_utils import d3fields_proc
from gendp.common.tactile_utils import TactileProcessor
from gendp.dataset.real_dataset import get_tactile_marker_coordinates_for_contact_field, predict_contact_field


def get_real_obs_dict(
        env_obs: Dict[str, np.ndarray], 
        shape_meta: dict,
        fusion = None,
        expected_labels = None,
        teleop = None,
        exclude_colors = [],
        contact_field_model = None,
        contact_field_device = 'cuda',
        tactile_processors: Optional[Dict] = None,
        ) -> Dict[str, np.ndarray]:
    obs_dict_np = dict()
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr['type']
        shape = attr['shape']
        if type == 'rgb':
            this_imgs_in = env_obs[key]
            t,hi,wi,ci = this_imgs_in.shape
            co,ho,wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(
                    input_res=(wi,hi), 
                    output_res=(wo,ho), 
                    bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            obs_dict_np[key] = np.moveaxis(out_imgs,-1,1)
        elif type == 'depth':
            this_depths_in = env_obs[key]
            t,hi,wi = this_depths_in.shape
            this_depths_in = this_depths_in.reshape(t,hi,wi,1)
            this_depths_in = np.clip(this_depths_in, 0, 1000).astype(np.uint16)
            co,ho,wo = shape
            assert co == 1
            out_depths = this_depths_in
            if (ho != hi) or (wo != wi):
                out_depths = np.stack([cv2.resize(x, (wo,ho)) for x in this_depths_in])
                if this_depths_in.dtype == np.uint16:
                    out_depths = out_depths.astype(np.float32) / 1000.
            # THWC to TCHW
            out_depths = out_depths[..., None]
            obs_dict_np[key] = np.moveaxis(out_depths,-1,1)
        elif type == 'low_dim':
            this_data_in = env_obs[key]
            if 'pose' in key and shape == (2,):
                # take X,Y coordinates
                this_data_in = this_data_in[...,[0,1]]
            elif key == 'ee_pose' and len(shape) == 1 and shape[0] == 9:
                # Convert ee_pose from [pos(3), euler(3), gripper(1)] to [pos(3), rot6d(6)]
                # this_data_in shape: (T, 7) or (T, 8) with gripper
                pos = this_data_in[..., :3]  # (T, 3)
                euler = this_data_in[..., 3:6]  # (T, 3)
                # Convert euler to rot6d
                from gendp.model.common.rotation_transformer import RotationTransformer
                rotation_transformer = RotationTransformer(from_rep='euler_angles', to_rep='rotation_6d', from_convention='xyz')
                rot6d = rotation_transformer.forward(euler)  # (T, 6)
                this_data_in = np.concatenate([pos, rot6d], axis=-1)
            obs_dict_np[key] = this_data_in
        elif type == 'tactile':
            # Handle tactile force field observations
            # Expected shape from config: [C, H, W] = [9, 7, 9]
            # where C=9 = force_current(3) + force_reference(3) + coordinates(3)
            # Key format: tactile_{left|right}_force_field
            
            # Determine which tactile sensor (left or right)
            if 'left' in key:
                tactile_key = 'tactile_img_left'  # Environment uses tactile_img_left/right
                sensor_name = 'tactile_left'
            elif 'right' in key:
                tactile_key = 'tactile_img_right'  # Environment uses tactile_img_left/right
                sensor_name = 'tactile_right'
            else:
                raise ValueError(f"Unknown tactile key format: {key}")
            
            # Get tactile frames from env_obs
            if tactile_key not in env_obs:
                raise ValueError(f"Tactile data '{tactile_key}' not found in env_obs. Available keys: {list(env_obs.keys())}")
            
            tactile_frames = env_obs[tactile_key]  # (T, H, W, C)
            
            # Initialize tactile processor if not already done
            if tactile_processors is None or sensor_name not in tactile_processors:
                raise ValueError(f"Tactile processor for '{sensor_name}' not initialized. Please pass tactile_processors dict.")
            
            # Need ee_pose for computing 3D marker coordinates
            if 'ee_pose' not in env_obs:
                raise ValueError("ee_pose required for tactile processing but not found in env_obs")
            
            # Process each frame to get force field + coordinates
            T = tactile_frames.shape[0]
            processed_frames = []
            
            # For real-time inference, use the first frame as reference
            reference_tactile = None
            if T > 0:
                reference_tactile = tactile_processors[sensor_name].process_frame(tactile_frames[0])  # (7, 9, 3)
            
            for t_idx in range(T):
                frame = tactile_frames[t_idx]
                # Process frame returns force field data (7, 9, 3) with [depth, dy, dx]
                force_field = tactile_processors[sensor_name].process_frame(frame)
                
                # Concatenate with reference to get 6D force data
                if reference_tactile is not None:
                    force_field_6d = np.concatenate([force_field, reference_tactile], axis=-1)  # (7, 9, 6)
                else:
                    force_field_6d = np.concatenate([force_field, force_field], axis=-1)  # (7, 9, 6)
                
                # Get 3D marker coordinates using robot pose
                ee_pose_8d = env_obs['ee_pose'][t_idx]  # [x,y,z,rx,ry,rz,gripper] or [x,y,z,qx,qy,qz,qw,gripper]
                ee_pos = ee_pose_8d[:3]
                
                # Check if quaternion or euler angles
                if len(ee_pose_8d) >= 7 and abs(np.linalg.norm(ee_pose_8d[3:7]) - 1.0) < 0.1:
                    # Likely quaternion (norm close to 1)
                    ee_quat = ee_pose_8d[3:7]
                    gripper_pos = ee_pose_8d[7] if len(ee_pose_8d) > 7 else 0.05
                else:
                    # Likely euler angles
                    ee_euler = ee_pose_8d[3:6]
                    ee_quat = st.Rotation.from_euler('xyz', ee_euler).as_quat()
                    gripper_pos = ee_pose_8d[6] if len(ee_pose_8d) > 6 else 0.05
                
                # Transform pose for coordinate calculation
                transformed_pos = ee_pos.copy()
                transformed_pos[2] -= 0.14
                current_rot = st.Rotation.from_quat(ee_quat)
                z_rotation = st.Rotation.from_euler('z', np.pi/4)
                transformed_rot = z_rotation * current_rot
                transformed_quat = transformed_rot.as_quat()
                ee_pose_7d = np.concatenate([transformed_pos, transformed_quat])
                
                # Get 3D marker coordinates
                from gendp.dataset.real_dataset import get_tactile_marker_coordinates_for_contact_field
                if 'left' in key:
                    tactile_coord, _ = get_tactile_marker_coordinates_for_contact_field(ee_pose_7d, gripper_pos)
                else:
                    _, tactile_coord = get_tactile_marker_coordinates_for_contact_field(ee_pose_7d, gripper_pos)
                
                # Combine force field (6D) + coordinates (3D) = 9D
                # force_field_6d: (7, 9, 6), tactile_coord: (7, 9, 3)
                combined = np.concatenate([force_field_6d, tactile_coord], axis=-1)  # (7, 9, 9)
                processed_frames.append(combined)
            
            # Stack into (T, H, W, C) where H=7, W=9, C=9
            tactile_data = np.stack(processed_frames, axis=0)  # (T, 7, 9, 9)
            
            # Convert to (T, C, H, W) format: (T, 9, 7, 9)
            obs_dict_np[key] = np.transpose(tactile_data, (0, 3, 1, 2))
        elif type == 'spatial':
            try:
                assert key == 'd3fields'
            except AssertionError:
                raise RuntimeError('Only support d3fields as spatial type.')
            try:
                assert fusion is not None
            except AssertionError:
                raise RuntimeError('fusion is None, but d3fields is requested.')

            # construct inputs for d3fields processing
            view_keys = attr['info']['view_keys']
            use_dino = False
            distill_dino = attr['info']['distill_dino'] if 'distill_dino' in attr['info'] else False
            tool_names = [None, None]
            if 'right_tool' in attr['info']:
                tool_names[0] = attr['info']['right_tool']
            if 'left_tool' in attr['info']:
                tool_names[1] = attr['info']['left_tool']
            color_seq = np.stack([env_obs[f'{k}_color'] for k in view_keys], axis=1) # (T, V, H ,W, C)
            depth_seq = np.stack([env_obs[f'{k}_depth'] for k in view_keys], axis=1) / 1000. # (T, V, H ,W)
            extri_seq = np.stack([env_obs[f'{k}_extrinsics'] for k in view_keys], axis=1) # (T, V, 4, 4)
            intri_seq = np.stack([env_obs[f'{k}_intrinsics'] for k in view_keys], axis=1) # (T, V, 3, 3)
            qpos_seq = env_obs['full_joint_pos'] if 'full_joint_pos' in env_obs else env_obs['joint_pos'] # (T, -1)
            if 'robot_base_pose_in_world' in env_obs:
                robot_base_pose_in_world_seq = env_obs['robot_base_pose_in_world'] # (T, 4, 4)
            else:
                robot_base_pose_in_world_seq = np.tile(np.eye(4), (intri_seq.shape[0], 1, 1))
            
            # Check if contact field is enabled
            use_contact_field = contact_field_model is not None
            
            if use_contact_field:
                # Run d3fields_proc with object/background segmentation
                obj_bg_result = d3fields_proc(
                    fusion=fusion,
                    shape_meta=attr,
                    color_seq=color_seq,
                    depth_seq=depth_seq,
                    extri_seq=extri_seq,
                    intri_seq=intri_seq,
                    robot_base_pose_in_world_seq=robot_base_pose_in_world_seq,
                    qpos_seq=qpos_seq,
                    teleop_robot=teleop,
                    expected_labels=expected_labels,
                    tool_names=tool_names,
                    exclude_colors=exclude_colors,
                    use_obj_bg_seg=True,
                    gripper_pose_seq=env_obs['ee_pose'] if 'ee_pose' in env_obs else None,
                    use_gripper_crop=True,
                )
                aggr_src_pts_ls, aggr_feats_ls, obj_pts_ls, obj_feats_ls, bg_pts_ls, bg_feats_ls = obj_bg_result
                
                # Process contact field for each timestep
                contact_field_pts_ls = []
                for t_idx in range(len(aggr_src_pts_ls)):
                    obj_pcd = obj_pts_ls[t_idx] if t_idx < len(obj_pts_ls) else np.zeros((0, 3))
                    full_pcd = aggr_src_pts_ls[t_idx]
                    
                    if obj_pcd.shape[0] > 0 and 'tactile_left' in env_obs and 'tactile_right' in env_obs:
                        # Get tactile frames
                        tactile_left_frame = env_obs['tactile_left'][t_idx] if t_idx < len(env_obs['tactile_left']) else None
                        tactile_right_frame = env_obs['tactile_right'][t_idx] if t_idx < len(env_obs['tactile_right']) else None
                        
                        if tactile_left_frame is not None and tactile_right_frame is not None:
                            # Initialize tactile processors if not provided
                            if tactile_processors is None:
                                tactile_processors = {}
                            
                            if 'tactile_left' not in tactile_processors:
                                # Get settings from either obs or tactile_settings
                                if 'tactile_left' in shape_meta['obs']:
                                    setting_left = shape_meta['obs']['tactile_left']['setting']
                                elif 'tactile_settings' in shape_meta and 'tactile_left' in shape_meta['tactile_settings']:
                                    setting_left = shape_meta['tactile_settings']['tactile_left']
                                else:
                                    setting_left = None
                                tactile_processors['tactile_left'] = TactileProcessor(
                                    width=320, height=240, marker_config=setting_left, use_gpu=True
                                )
                            if 'tactile_right' not in tactile_processors:
                                # Get settings from either obs or tactile_settings
                                if 'tactile_right' in shape_meta['obs']:
                                    setting_right = shape_meta['obs']['tactile_right']['setting']
                                elif 'tactile_settings' in shape_meta and 'tactile_right' in shape_meta['tactile_settings']:
                                    setting_right = shape_meta['tactile_settings']['tactile_right']
                                else:
                                    setting_right = None
                                tactile_processors['tactile_right'] = TactileProcessor(
                                    width=320, height=240, marker_config=setting_right, use_gpu=True
                                )
                            
                            # Process tactile frames
                            tactile_ff_left = tactile_processors['tactile_left'].process_frame(tactile_left_frame)
                            tactile_ff_right = tactile_processors['tactile_right'].process_frame(tactile_right_frame)
                            
                            # Get ee_pose and transform for contact field model
                            if 'ee_pose' in env_obs and t_idx < len(env_obs['ee_pose']):
                                ee_pose_8d = env_obs['ee_pose'][t_idx]
                                ee_pos = ee_pose_8d[:3]
                                ee_quat = ee_pose_8d[3:7]
                                gripper_pos = ee_pose_8d[7] if len(ee_pose_8d) > 7 else 0.05
                                
                                # Transform the pose for contact field model
                                transformed_pos = ee_pos.copy()
                                transformed_pos[2] -= 0.14
                                current_rot = st.Rotation.from_quat(ee_quat)
                                z_rotation = st.Rotation.from_euler('z', np.pi/4)
                                transformed_rot = z_rotation * current_rot
                                transformed_quat = transformed_rot.as_quat()
                                ee_pose_7d = np.concatenate([transformed_pos, transformed_quat])
                                
                                # Get 3D marker coordinates
                                tactile_coord_left, tactile_coord_right = get_tactile_marker_coordinates_for_contact_field(
                                    ee_pose_7d, gripper_pos
                                )
                                
                                # Predict contact field
                                contact_prob, contact_force = predict_contact_field(
                                    model=contact_field_model,
                                    obj_pointcloud=obj_pcd[:, :3],
                                    tactile_data_left=tactile_ff_left,
                                    tactile_data_right=tactile_ff_right,
                                    tactile_coord_left=tactile_coord_left,
                                    tactile_coord_right=tactile_coord_right,
                                    ee_pose=ee_pose_7d,
                                    device=contact_field_device
                                )
                                
                                # Create contact field data (N_obj, 4): [contact_prob, fx, fy, fz]
                                contact_field_data = np.concatenate([contact_prob, contact_force], axis=-1).astype(np.float32)
                                
                                # Create contact field for full point cloud
                                N_obj = obj_pcd.shape[0]
                                N_bg = full_pcd.shape[0] - N_obj
                                
                                obj_contact_field = contact_field_data
                                bg_contact_field = np.zeros((N_bg, 4), dtype=np.float32)
                                full_contact_field = np.concatenate([obj_contact_field, bg_contact_field], axis=0)
                                
                                pcd_with_contact = np.concatenate([full_pcd, full_contact_field], axis=-1).astype(np.float32)
                                contact_field_pts_ls.append(pcd_with_contact)
                            else:
                                # No ee_pose, pad with zeros
                                zeros_contact = np.zeros((full_pcd.shape[0], 4), dtype=np.float32)
                                pcd_with_contact = np.concatenate([full_pcd, zeros_contact], axis=-1).astype(np.float32)
                                contact_field_pts_ls.append(pcd_with_contact)
                        else:
                            # No tactile data, pad with zeros
                            zeros_contact = np.zeros((full_pcd.shape[0], 4), dtype=np.float32)
                            pcd_with_contact = np.concatenate([full_pcd, zeros_contact], axis=-1).astype(np.float32)
                            contact_field_pts_ls.append(pcd_with_contact)
                    else:
                        # No object point cloud or no tactile data, pad with zeros
                        zeros_contact = np.zeros((full_pcd.shape[0], 4), dtype=np.float32)
                        pcd_with_contact = np.concatenate([full_pcd, zeros_contact], axis=-1).astype(np.float32)
                        contact_field_pts_ls.append(pcd_with_contact)
                
                # Replace with contact field enhanced version
                aggr_src_pts_ls = contact_field_pts_ls
            else:
                # Standard d3fields_proc without contact field
                aggr_src_pts_ls, aggr_feats_ls, aggr_colors_ls = \
                    d3fields_proc(fusion=fusion,
                                shape_meta=attr,
                                color_seq=color_seq,
                                depth_seq=depth_seq,
                                extri_seq=extri_seq,
                                intri_seq=intri_seq,
                                robot_base_pose_in_world_seq=robot_base_pose_in_world_seq,
                                qpos_seq=qpos_seq,
                                teleop_robot=teleop,
                                expected_labels=expected_labels,
                                tool_names=tool_names,
                                exclude_colors=exclude_colors,)
            aggr_src_pts = np.stack(aggr_src_pts_ls)
            aggr_feats = np.stack(aggr_feats_ls) if aggr_feats_ls and aggr_feats_ls[0] is not None else None
            aggr_colors = np.stack(aggr_colors_ls) if aggr_colors_ls and aggr_colors_ls[0] is not None else None

            # Handle features based on distill_dino and contact field settings
            distill_dino = attr['info']['distill_dino'] if 'distill_dino' in attr['info'] else False
            
            if distill_dino and aggr_feats is not None:
                if use_contact_field:
                    # Extract contact field channels (last 4 channels) from aggr_src_pts
                    contact_channels = aggr_src_pts[:, :, -4:]  # (T, N, 4)
                    xyz = aggr_src_pts[:, :, :3]  # (T, N, 3)
                    # Concatenate: [xyz, dino_feats, contact_field]
                    parts_to_concat = [
                        xyz,
                        aggr_feats,  # dino features
                    ]
                    # Add RGB channels if enabled
                    if aggr_colors is not None:
                        parts_to_concat.append(aggr_colors)  # RGB channels
                    parts_to_concat.append(contact_channels)  # contact field
                    aggr_pts_feats = np.concatenate(parts_to_concat, axis=-1)

                else:
                    parts_to_concat = [aggr_src_pts, aggr_feats]
                    if aggr_colors is not None:
                        parts_to_concat.insert(1, aggr_colors)  # Insert RGB channels in the middle

                    aggr_pts_feats = np.concatenate(parts_to_concat, axis=-1)
            elif use_contact_field:
                # contact_field=True but distill_dino=False
                # aggr_src_pts already contains [xyz, contact_field] from contact field processing
                aggr_pts_feats = aggr_src_pts
            elif use_dino or distill_dino:
                if aggr_feats is not None:
                    aggr_pts_feats = np.concatenate([aggr_src_pts, aggr_feats], axis=-1)
                else:
                    aggr_pts_feats = aggr_src_pts
            else:
                aggr_pts_feats = aggr_src_pts
            obs_dict_np[key] = aggr_pts_feats.transpose(0,2,1)

    return obs_dict_np


def get_real_obs_resolution(
        shape_meta: dict
        ) -> Tuple[int, int]:
    out_res = None
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            co,ho,wo = shape
            if out_res is None:
                out_res = (wo, ho)
            assert out_res == (wo, ho)
    return out_res
