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
from threadpoolctl import threadpool_limits
from omegaconf import OmegaConf, DictConfig
import transforms3d
import scipy.spatial.transform as st
import yaml

from gendp.common.pytorch_util import dict_apply
from gendp.common.replay_buffer import ReplayBuffer
from gendp.model.common.rotation_transformer import RotationTransformer
from gendp.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from gendp.common.kinematics_utils import KinHelper
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
        n_workers=None, max_inflight_tasks=None, robot_name='panda'):
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
    for epi_idx in tqdm(episodes_idx, desc=f"Loading episodes"):
        dataset_path = os.path.join(dataset_dir, f'episode_{epi_idx}.hdf5')
        with h5py.File(dataset_path) as file:
            # count total steps
            # episode_length = file['cartesian_action'].shape[0]
            episode_length = file['joint_action'].shape[0] - trim_tail
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
                    this_data = file['observations']['ee_pose'][:episode_length]
                else:
                    this_data = file[data_key][:episode_length]
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
                    this_data = _convert_ee_pose_obs(this_data, rotation_transformer)
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
                frames = file['observations']['images'][key][:episode_length]
                shape = tuple(shape_meta['obs'][key]['shape'])
                c,h,w = shape
                resize_imgs = [cv2.resize(img, (w,h), interpolation=cv2.INTER_AREA) for img in frames]
                frames = np.stack(resize_imgs, axis=0)
                assert frames[0].shape == (h,w,c)
                rgb_data_dict[key].append(frames)
            
            for key in depth_keys:
                if key not in depth_data_dict:
                    depth_data_dict[key] = list()
                frames = file['observations']['images'][key][:episode_length]
                shape = tuple(shape_meta['obs'][key]['shape'])
                c,h,w = shape
                resize_imgs = [cv2.resize(img, (w,h), interpolation=cv2.INTER_AREA) for img in frames]
                frames = np.stack(resize_imgs, axis=0)[..., None]
                frames = np.clip(frames, 0, 1000).astype(np.uint16)
                assert frames[0].shape == (h,w,c)
                depth_data_dict[key].append(frames)
    

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
            robot_name='panda'
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
            obs_dict[key] = np.moveaxis(sample[key][T_slice],-1,1
                ).astype(np.float32) / 255.
            # T,C,H,W
            del sample[key]
        for key in self.depth_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint16 image to float32
            obs_dict[key] = np.moveaxis(sample[key][T_slice],-1,1
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

def update_ee_pose():
    # create on 12/17/2023, only for one-time use
    
    ### ALOHA fixed constants
    DT = 0.02
    JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
    START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]
    START_EE_POSE = [2.56418115e-01, -5.50126845e-04,  2.95703636e-01,  6.88682872e-04, -3.83402967e-02, -1.18223866e-03,  9.99263804e-01, -0.3]

    # Left finger position limits (qpos[7]), right_finger = -1 * left_finger
    MASTER_GRIPPER_POSITION_OPEN = 0.02417
    MASTER_GRIPPER_POSITION_CLOSE = 0.01244
    PUPPET_GRIPPER_POSITION_OPEN = 0.05800
    PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

    # Gripper joint limits (qpos[6])
    MASTER_GRIPPER_JOINT_OPEN = 0.3083
    MASTER_GRIPPER_JOINT_CLOSE = -0.6842
    PUPPET_GRIPPER_JOINT_OPEN = 1.4910
    PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

    ############################ Helper functions ############################

    MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
    PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
    MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
    PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
    MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

    MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
    PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
    MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
    PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
    MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

    MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
    PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

    MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
    MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
    PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
    PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

    MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2
    
    
    DEBUG = False
    
    # update ee_pose in hdf5
    data_dir = '/home/yixuan/general_dp/data/real_aloha_demo/open_bag_v2_demo_1'
    epi_s = 0
    epi_e = 1
    kin_helper = KinHelper(robot_name='trossen_vx300s_v3')
    for epi_i in tqdm(range(epi_s, epi_e)):
        epi_fn = os.path.join(data_dir, f'episode_{epi_i}.hdf5')
        epi_data, fn = load_dict_from_hdf5(epi_fn)
        # joint_action = epi_data['joint_action']
        old_cartesian_action = epi_data['cartesian_action']
        robot_base_in_world_seq = epi_data['observations']['robot_base_pose_in_world']
        sec_base_in_world = np.array([[0,1,0,-0.13],
                                    [-1,0,0,0.27],
                                    [0,0,1,0.02],
                                    [0,0,0,1]])
        new_epi_data = {'cartesian_action': np.array(epi_data['cartesian_action']).copy(),
                        # 'observations': {
                        #     'ee_pose': np.array(epi_data['observations']['ee_pose']).copy(),
                        #     }
                        }
        for i in range(old_cartesian_action.shape[0]):
            ### update cartesian_action
            # puppet_gripper_pos = PUPPET_JOINT2POS(joint_action[i,-1])
            # puppet_action_qpos = np.concatenate([joint_action[i,:-1], np.array([puppet_gripper_pos, -puppet_gripper_pos])])
            # puppet_action_eef_mat = kin_helper.compute_fk_links(qpos=puppet_action_qpos, link_idx=[kin_helper.eef_link_idx])[0]
            puppet_action_old_eef = old_cartesian_action[i] # (14,)
            puppet_action_old_sec_eef = puppet_action_old_eef[7:] # (7,)
            puppet_action_old_sec_mat = np.eye(4)
            puppet_action_old_sec_mat[:3, 3] = puppet_action_old_sec_eef[:3]
            puppet_action_old_sec_mat[:3, :3] = transforms3d.euler.euler2mat(*puppet_action_old_sec_eef[3:6])
            puppet_action_new_sec_mat = np.linalg.inv(robot_base_in_world_seq[i, 0]) @ sec_base_in_world @ puppet_action_old_sec_mat
            puppet_action_eef = np.concatenate([puppet_action_old_eef[:7],
                                                puppet_action_new_sec_mat[:3,3],
                                                transforms3d.euler.mat2euler(puppet_action_new_sec_mat[:3,:3]),
                                                puppet_action_old_sec_eef[-1:]])
            if DEBUG:
                print('original cartesian_action: ', epi_data['cartesian_action'][i])
                print('new cartesian_action: ', puppet_action_eef)
            new_epi_data['cartesian_action'][i] = puppet_action_eef

            # ### update ee_pose
            # qpos = epi_data['observations']['joint_pos'][i]
            # full_qpos = epi_data['observations']['full_joint_pos'][i]
            # puppet_eef_mat = kin_helper.compute_fk_links(qpos=full_qpos, link_idx=[kin_helper.eef_link_idx])[0]
            # puppet_eef = np.concatenate([puppet_eef_mat[:3,3],
            #                              transforms3d.euler.mat2euler(puppet_eef_mat[:3,:3]),
            #                              qpos[-1:]])
            # if DEBUG:
            #     print('original ee_pose: ', epi_data['observations']['ee_pose'][i])
            #     print('new ee_pose: ', puppet_eef)
            # new_epi_data['observations']['ee_pose'][i] = puppet_eef
        fn.close()
        modify_hdf5_from_dict(epi_fn, new_epi_data)
