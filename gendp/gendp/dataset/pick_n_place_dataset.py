import glob
import os
import shutil
import time
import pickle
from typing import Dict, Optional
from pathlib import Path
import torch
import numpy as np
import copy
import multiprocessing
import zarr
from tqdm import tqdm
import concurrent.futures
import h5py
import cv2
import scipy.spatial.transform as st
from filelock import FileLock
# from threadpoolctl import threadpool_limits
# from omegaconf import OmegaConf, DictConfig
# import transforms3d
import scipy.spatial.transform as st
import yaml
from torchvision import transforms

from gendp.common.pytorch_util import dict_apply
from gendp.common.replay_buffer import ReplayBuffer
from gendp.model.common.rotation_transformer import RotationTransformer
from gendp.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
# from gendp.common.kinematics_utils import KinHelper
from gendp.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from gendp.common.data_utils import _convert_actions, _convert_ee_pose_obs, load_dict_from_hdf5, modify_hdf5_from_dict
from gendp.dataset.base_dataset import BaseImageDataset
from gendp.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from gendp.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats,
)

register_codecs()


def _filter_small_ee_changes(ee_poses, pos_threshold=0.001, rot_threshold=0.01, gripper_threshold=0.001):
    """
    Filter out steps with small changes in EE pose.
    
    Args:
        ee_poses: Array of EE poses with shape (T, 7) - [pos(3), euler(3), gripper(1)]
        pos_threshold: Position change threshold in meters
        rot_threshold: Rotation change threshold in radians
        
    Returns:
        keep_mask: Boolean mask indicating which steps to keep (True = keep, False = filter out)
    """
    T = ee_poses.shape[0]
    keep_mask = np.ones(T, dtype=bool)
    
    if T <= 1:
        return keep_mask
    
    # Always keep the first frame
    keep_mask[0] = True
    
    # Compute position changes
    pos = ee_poses[:, :3]
    pos_changes = np.linalg.norm(pos[1:] - pos[:-1], axis=-1)  # (T-1,)
    
    # Compute rotation changes using axis-angle representation
    rot_euler = ee_poses[:, 3:6]
    rot_mats = st.Rotation.from_euler('xyz', rot_euler).as_matrix()  # (T, 3, 3)

    # Compute gripper change
    gripper_changes = np.abs(ee_poses[1:, 6] - ee_poses[:-1, 6])  # (T-1,)

    rot_changes = np.zeros(T - 1)
    for t in range(T - 1):
        # R_delta = R_{t+1} * R_t^T
        delta_rot_mat = rot_mats[t + 1] @ rot_mats[t].T
        # Convert to axis-angle and get angle magnitude
        rotvec = st.Rotation.from_matrix(delta_rot_mat).as_rotvec()
        rot_changes[t] = np.linalg.norm(rotvec)
    
    # Filter: keep step if either position or rotation change exceeds threshold
    for t in range(1, T):
        if pos_changes[t - 1] > pos_threshold or rot_changes[t - 1] > rot_threshold or gripper_changes[t - 1] > gripper_threshold:
            keep_mask[t] = True
        else:
            keep_mask[t] = False
    
    return keep_mask


def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

# convert raw hdf5 data to replay buffer, which is used for diffusion policy training
def _convert_real_to_dp_replay(store, shape_meta, dataset_dir, rotation_transformer, 
        n_workers=None, max_inflight_tasks=None, robot_name='panda', 
        filter_small_changes=False, pos_threshold=0.001, rot_threshold=0.01, gripper_threshold=0.001):
    """
    Args:
        filter_small_changes: If True, filter out steps with small changes in EE pose
        pos_threshold: Position change threshold in meters (default: 0.001m = 1mm)
        rot_threshold: Rotation change threshold in radians (default: 0.01 rad ~= 0.57 degrees)
    """
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    depth_keys = list()
    lowdim_keys = list()

    # construct compressors and chunks
    obs_shape_meta = shape_meta['obs']
    trim_tail = shape_meta['trim_tail']
    for key, attr in obs_shape_meta.items():
        shape = attr['shape']
        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            rgb_keys.append(key)
        if type == 'depth':
            depth_keys.append(key)
        elif type == 'low_dim':
            lowdim_keys.append(key)
    
    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)

    episodes_paths = glob.glob(os.path.join(dataset_dir, 'episode_*.hdf5'))
    episodes_stem_name = [Path(path).stem for path in episodes_paths]
    episodes_idx = [int(stem_name.split('_')[-1]) for stem_name in episodes_stem_name]
    episodes_idx = sorted(episodes_idx)
        
    episode_ends = list()
    prev_end = 0
    lowdim_data_dict = dict()
    rgb_data_dict = dict()
    depth_data_dict = dict()
    total_filtered_steps = 0
    total_original_steps = 0
    
    for epi_idx in tqdm(episodes_idx, desc=f"Loading episodes"):
        dataset_path = os.path.join(dataset_dir, f'episode_{epi_idx}.hdf5')
        with h5py.File(dataset_path) as file:
            # count total steps
            # episode_length = file['cartesian_action'].shape[0]
            original_episode_length = file['joint_action'].shape[0] - trim_tail
            total_original_steps += original_episode_length
            
            # Apply filtering if enabled
            if filter_small_changes:
                raw_ee_pose = file['observations']['ee_pose'][:original_episode_length]
                keep_mask = _filter_small_ee_changes(
                    raw_ee_pose, 
                    pos_threshold=pos_threshold, 
                    rot_threshold=rot_threshold,
                    gripper_threshold=gripper_threshold
                )
                episode_length = np.sum(keep_mask)
                total_filtered_steps += (original_episode_length - episode_length)
            else:
                keep_mask = np.ones(original_episode_length, dtype=bool)
                episode_length = original_episode_length
            
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
            
            # save lowdim data to lowedim_data_dict
            for key in lowdim_keys + ['action']:
                data_key = 'observations/' + key
                if key == 'action':
                    data_key = 'cartesian_action' if 'key' not in shape_meta['action'] else shape_meta['action']['key']
                if key not in lowdim_data_dict:
                    lowdim_data_dict[key] = list()
                if data_key == 'cartesian_action':
                    this_data = file['observations']['ee_pose'][:original_episode_length]
                elif data_key == 'joint_action':
                    this_data = file['observations']['joint_pos'][:original_episode_length]
                else:
                    this_data = file[data_key][:original_episode_length]
                
                # Apply filtering mask before converting actions
                this_data = this_data[keep_mask]
                
                if key == 'action':
                    delta_action = shape_meta['action'].get('delta', False)
                    this_data = _convert_actions(
                        raw_actions=this_data,
                        rotation_transformer=rotation_transformer,
                        action_key=data_key,
                        delta_action=delta_action,
                        rot_format='rotvec',
                    )
                    assert this_data.shape == (episode_length,) + tuple(shape_meta['action']['shape']), \
                        f"Action shape mismatch: {this_data.shape} vs expected {(episode_length,) + tuple(shape_meta['action']['shape'])}"
                elif key == 'ee_pose':
                    # Convert ee_pose from [pos(3), rotvec(3), gripper(1)] to [pos(3), rot6d(6)]
                    # print(f"Converting ee_pose: input shape {this_data.shape}, expected output shape {(episode_length,) + tuple(shape_meta['obs'][key]['shape'])}")
                    with_gripper = shape_meta['obs'][key]['shape'][-1] == 10
                    this_data = _convert_ee_pose_obs(this_data, rotation_transformer, with_gripper=with_gripper)
                    # print(f"After conversion: {this_data.shape}")
                    assert this_data.shape == (episode_length,) + tuple(shape_meta['obs'][key]['shape']), \
                        f"EE pose shape mismatch: {this_data.shape} vs expected {(episode_length,) + tuple(shape_meta['obs'][key]['shape'])}"
                else:
                    assert this_data.shape == (episode_length,) + tuple(shape_meta['obs'][key]['shape']), \
                        f"Obs {key} shape mismatch: {this_data.shape} vs expected {(episode_length,) + tuple(shape_meta['obs'][key]['shape'])}"
                lowdim_data_dict[key].append(this_data)
            
            for key in rgb_keys:
                if key not in rgb_data_dict:
                    rgb_data_dict[key] = list()
                frames = file['observations']['images'][key][:original_episode_length]
                # Apply filtering mask
                frames = frames[keep_mask]
                shape = tuple(shape_meta['obs'][key]['shape'])
                c,h,w = shape
                resize_imgs = [cv2.resize(img, (w,h), interpolation=cv2.INTER_AREA) for img in frames]
                frames = np.stack(resize_imgs, axis=0)
                assert frames[0].shape == (h,w,c)
                rgb_data_dict[key].append(frames)
            
            for key in depth_keys:
                if key not in depth_data_dict:
                    depth_data_dict[key] = list()
                frames = file['observations']['images'][key][:original_episode_length]
                # Apply filtering mask
                frames = frames[keep_mask]
                shape = tuple(shape_meta['obs'][key]['shape'])
                c,h,w = shape
                resize_imgs = [cv2.resize(img, (w,h), interpolation=cv2.INTER_AREA) for img in frames]
                frames = np.stack(resize_imgs, axis=0)[..., None]
                frames = np.clip(frames, 0, 1000).astype(np.uint16)
                assert frames[0].shape == (h,w,c)
                depth_data_dict[key].append(frames)
    
    if filter_small_changes:
        print(f"Filtered {total_filtered_steps} steps out of {total_original_steps} "
              f"({100.0 * total_filtered_steps / total_original_steps:.2f}%) "
              f"with pos_threshold={pos_threshold}m, rot_threshold={rot_threshold}rad")
    

    def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
        try:
            zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
            # make sure we can successfully decode
            _ = zarr_arr[zarr_idx]
            return True
        except Exception as e:
            return False
    
    # dump data_dict
    print('Dumping meta data')
    n_steps = episode_ends[-1]
    _ = meta_group.array('episode_ends', episode_ends, 
        dtype=np.int64, compressor=None, overwrite=True)

    print('Dumping lowdim data')
    for key, data in lowdim_data_dict.items():
        data = np.concatenate(data, axis=0)
        _ = data_group.array(
            name=key,
            data=data,
            shape=data.shape,
            chunks=data.shape,
            compressor=None,
            dtype=data.dtype
        )
    
    print('Dumping rgb data')
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = set()
        for key, data in rgb_data_dict.items():
            hdf5_arr = np.concatenate(data, axis=0)
            shape = tuple(shape_meta['obs'][key]['shape'])
            c,h,w = shape
            this_compressor = Jpeg2k(level=50)
            img_arr = data_group.require_dataset(
                name=key,
                shape=(n_steps,h,w,c),
                chunks=(1,h,w,c),
                compressor=this_compressor,
                dtype=np.uint8
            )
            for hdf5_idx in tqdm(range(hdf5_arr.shape[0])):
                if len(futures) >= max_inflight_tasks:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    for f in completed:
                        if not f.result():
                            raise RuntimeError('Failed to encode image!')
                zarr_idx = hdf5_idx
                futures.add(
                    executor.submit(img_copy, 
                        img_arr, zarr_idx, hdf5_arr, hdf5_idx))
        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError('Failed to encode image!')
    
    print('Dumping depth data')
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = set()
        for key, data in depth_data_dict.items():
            hdf5_arr = np.concatenate(data, axis=0)
            shape = tuple(shape_meta['obs'][key]['shape'])
            c,h,w = shape
            this_compressor = Jpeg2k(level=50)
            img_arr = data_group.require_dataset(
                name=key,
                shape=(n_steps,h,w,c),
                chunks=(1,h,w,c),
                compressor=this_compressor,
                dtype=np.uint16
            )
            for hdf5_idx in tqdm(range(hdf5_arr.shape[0])):
                if len(futures) >= max_inflight_tasks:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    for f in completed:
                        if not f.result():
                            raise RuntimeError('Failed to encode image!')
                zarr_idx = hdf5_idx
                futures.add(
                    executor.submit(img_copy, 
                        img_arr, zarr_idx, hdf5_arr, hdf5_idx))
        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError('Failed to encode image!')
    
    replay_buffer = ReplayBuffer(root)
    return replay_buffer

class RealDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_dir: str,
            vis_input: False,
            horizon=1,
            pad_before=0,
            pad_after=0,
            rotation_rep='rotation_6d',
            use_legacy_normalizer=True,
            use_cache=True,
            seed=42,
            val_ratio=0.0,
            manual_val_mask=False,
            manual_val_start=-1,
            n_obs_steps=None,
            robot_name='panda',
            filter_small_changes=False,
            pos_threshold=0.001,
            rot_threshold=0.01,
            gripper_threshold=0.001
            ):
        
        super().__init__()
        
        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep=rotation_rep)

        replay_buffer = None
        cache_info_str = ''
        
        for key, attr in shape_meta['obs'].items():
            if ('type' in attr) and (attr['type'] == 'depth'):
                cache_info_str += '_rgbd'
                break
        if 'force_torque' in shape_meta['obs']:
            cache_info_str += '_ft'
        if 'key' in shape_meta['action'] and shape_meta['action']['key'] == 'joint_action':
            cache_info_str += '_joint'
        else:
            cache_info_str += '_eef'
        if 'trim_tail' in shape_meta and shape_meta['trim_tail'] > 0:
            cache_info_str += '_trim'
        # Add delta action info to cache string
        if 'delta' in shape_meta['action'] and shape_meta['action']['delta']:
            cache_info_str += '_delta'
            cache_info_str += f"_act{shape_meta['action']['shape'][0]}"
        # Add filtering info to cache string
        if filter_small_changes:
            cache_info_str += f'_filt_p{int(pos_threshold*1000)}mm_r{int(rot_threshold*1000)}mrad'
        if use_cache:
            cache_zarr_path = os.path.join(dataset_dir, f'cache{cache_info_str}.zarr.zip')
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):

                    # cache does not exists
                    try:
                        print('Cache does not exist. Creating!')
                        # store = zarr.DirectoryStore(cache_zarr_path)
                        replay_buffer = _convert_real_to_dp_replay(
                            store=zarr.MemoryStore(), 
                            shape_meta=shape_meta, 
                            dataset_dir=dataset_dir, 
                            rotation_transformer=rotation_transformer,
                            robot_name=robot_name,
                            filter_small_changes=filter_small_changes,
                            pos_threshold=pos_threshold,
                            rot_threshold=rot_threshold,
                            gripper_threshold=gripper_threshold
                            )
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                    except Exception as e:
                        if os.path.exists(cache_zarr_path):
                            shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:

            replay_buffer = _convert_real_to_dp_replay(
                store=zarr.MemoryStore(), 
                shape_meta=shape_meta, 
                dataset_dir=dataset_dir,
                rotation_transformer=rotation_transformer,
                robot_name=robot_name,
                filter_small_changes=filter_small_changes,
                pos_threshold=pos_threshold,
                rot_threshold=rot_threshold,
                gripper_threshold=gripper_threshold
            )
        self.replay_buffer = replay_buffer

        rgb_keys = list()
        depth_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        
        # First pass: collect base keys from shape_meta
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'depth':
                depth_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        
        # for key in rgb_keys:
        #     replay_buffer[key].compressor.numthreads=1

        if not manual_val_mask:
            val_mask = get_val_mask(
                n_episodes=replay_buffer.n_episodes, 
                val_ratio=val_ratio,
                seed=seed)
        else:
            try:
                assert manual_val_start >= 0
                assert manual_val_start < replay_buffer.n_episodes
            except:
                raise RuntimeError('invalid manual_val_start')
            val_mask = np.zeros((replay_buffer.n_episodes,), dtype=np.bool)
            val_mask[manual_val_start:] = True
        train_mask = ~val_mask
        
        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + depth_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            dataset_dir=dataset_dir,
            key_first_k=key_first_k,
            shape_meta=shape_meta,)
        
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.depth_keys = depth_keys
        self.lowdim_keys = lowdim_keys
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.use_legacy_normalizer = use_legacy_normalizer
        self.dataset_dir = dataset_dir
        self.key_first_k = key_first_k

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
            dataset_dir=self.dataset_dir,
            key_first_k=self.key_first_k,
            shape_meta=self.shape_meta,
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer['action'])
        if self.use_legacy_normalizer:
            this_normalizer = normalizer_from_stat(stat)
        else:
            raise RuntimeError('unsupported')
        normalizer['action'] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if key.endswith('pos'):
                # this_normalizer = get_range_normalizer_from_stat(stat)
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('pose'):
                # this_normalizer = get_range_normalizer_from_stat(stat)
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('vel'):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('force_torque'):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            else:
                raise RuntimeError(f'{key} unsupported')
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        
        for key in self.depth_keys:
            normalizer[key] = get_image_range_normalizer()
        
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    # Define augmentations
    image_augmentations = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # transforms.RandomCrop((224, 224)),  # Example crop size, adjust as needed
        transforms.Lambda(lambda img: img + torch.randn_like(img) * 0.05)  # Add Gaussian noise
    ])

    def _sample_to_data(self, sample):
        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(sample[key][T_slice], -1, 1
                ).astype(np.float32) / 255.
            # Apply augmentations
            obs_dict[key] = torch.stack([image_augmentations(torch.tensor(img)) for img in obs_dict[key]])
            # T,C,H,W
            del sample[key]
        for key in self.depth_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint16 image to float32
            obs_dict[key] = np.moveaxis(sample[key][T_slice], -1, 1
                ).astype(np.float32) / 1000.
            # T,C,H,W
            del sample[key]
        for key in self.lowdim_keys:
            obs_dict[key] = sample[key][T_slice].astype(np.float32)
            del sample[key]
        
        data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(sample['action'].astype(np.float32))
        }
        return data
    

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return data
