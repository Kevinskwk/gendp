"""
Analyze tactile force field data from collected real-world episodes.
Compares distribution with contact field model pre-training statistics.
"""

import h5py
import numpy as np
import glob
import os
import json
from pathlib import Path
import sys

# Add gendp to path
sys.path.insert(0, '/home/kevin/gendp/gendp')

from gendp.common.tactile_utils import TactileProcessor
import yaml

def load_tactile_settings(config_path):
    """Load tactile settings from config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    tactile_settings = {}
    if 'obs' in config['shape_meta']:
        if 'tactile_left' in config['shape_meta']['obs']:
            tactile_settings['tactile_left'] = config['shape_meta']['obs']['tactile_left']['setting']
        if 'tactile_right' in config['shape_meta']['obs']:
            tactile_settings['tactile_right'] = config['shape_meta']['obs']['tactile_right']['setting']
    
    # Also check tactile_settings section
    if 'tactile_settings' in config['shape_meta']:
        if 'tactile_left' in config['shape_meta']['tactile_settings']:
            tactile_settings['tactile_left'] = config['shape_meta']['tactile_settings']['tactile_left']
        if 'tactile_right' in config['shape_meta']['tactile_settings']:
            tactile_settings['tactile_right'] = config['shape_meta']['tactile_settings']['tactile_right']
    
    return tactile_settings

def analyze_tactile_data(dataset_dir, config_path=None, max_episodes=None):
    """
    Analyze tactile force field data from episodes.
    
    Args:
        dataset_dir: Directory containing episode_*.hdf5 files
        config_path: Path to config yaml file with tactile settings
        max_episodes: Maximum number of episodes to analyze (None = all)
    """
    
    # Find all episode files
    episodes_paths = sorted(glob.glob(os.path.join(dataset_dir, 'episode_*.hdf5')))
    
    if not episodes_paths:
        print(f"No episode files found in {dataset_dir}")
        return
    
    print(f'Found {len(episodes_paths)} episode files')
    
    if max_episodes is not None and max_episodes < len(episodes_paths):
        # Evenly sample episodes across the dataset
        indices = np.linspace(0, len(episodes_paths) - 1, max_episodes, dtype=int)
        episodes_paths = [episodes_paths[i] for i in indices]
        print(f'Analyzing {len(episodes_paths)} evenly sampled episodes (indices: {list(indices)})')
    
    # Load tactile settings
    tactile_settings = None
    if config_path and os.path.exists(config_path):
        print(f'Loading tactile settings from {config_path}')
        tactile_settings = load_tactile_settings(config_path)
    else:
        print('No config provided, will try to use default settings')
        # Default settings for GelSight sensors
        default_setting = {
            'gel_width': 0.015,
            'gel_height': 0.020,
            'marker_width': 7,
            'marker_height': 9,
            'marker_spacing': 0.002
        }
        tactile_settings = {
            'tactile_left': default_setting,
            'tactile_right': default_setting
        }
    
    # Initialize tactile processors
    tactile_processors = {}
    for key, setting in tactile_settings.items():
        print(f'Initializing TactileProcessor for {key}...')
        tactile_processors[key] = TactileProcessor(
            width=320, height=240, marker_config=setting, use_gpu=True
        )
    
    # Collect tactile force field data
    all_tactile_ff_left = []
    all_tactile_ff_right = []
    all_reference_ff_left = []  # Store reference for each episode
    all_reference_ff_right = []  # Store reference for each episode
    episode_frame_counts = []  # Track how many frames per episode for difference calculation
    
    print('\nProcessing episodes...')
    for ep_idx, ep_path in enumerate(episodes_paths):
        print(f'[{ep_idx+1}/{len(episodes_paths)}] Processing {ep_path}...')
        
        try:
            with h5py.File(ep_path, 'r') as f:
                if 'observations' not in f or 'tactile' not in f['observations']:
                    print(f'  ⚠️  No tactile data found, skipping')
                    continue
                
                # Check for tactile images
                has_left = 'tactile_img_left' in f['observations']['tactile']
                has_right = 'tactile_img_right' in f['observations']['tactile']
                
                if not (has_left and has_right):
                    print(f'  ⚠️  Missing tactile images, skipping')
                    continue
                
                # Get tactile images
                tactile_imgs_left = f['observations']['tactile']['tactile_img_left'][:]
                tactile_imgs_right = f['observations']['tactile']['tactile_img_right'][:]
                
                print(f'  Found {len(tactile_imgs_left)} frames')
                
                # Compute reference tactile from FIRST FRAME (consistent with training/inference)
                if len(tactile_imgs_left) > 0:
                    reference_ff_left = tactile_processors['tactile_left'].process_frame(tactile_imgs_left[0])
                    reference_ff_right = tactile_processors['tactile_right'].process_frame(tactile_imgs_right[0])
                else:
                    print(f'  ⚠️  No frames found, skipping')
                    continue
                
                # Store reference for this episode (will be replicated for each frame)
                episode_frame_counts.append(len(tactile_imgs_left))
                for _ in range(len(tactile_imgs_left)):
                    all_reference_ff_left.append(reference_ff_left)
                    all_reference_ff_right.append(reference_ff_right)
                
                # Process each frame
                for frame_idx in range(len(tactile_imgs_left)):
                    # Process left sensor
                    img_left = tactile_imgs_left[frame_idx]
                    ff_left = tactile_processors['tactile_left'].process_frame(img_left)  # (7, 9, 3)
                    all_tactile_ff_left.append(ff_left)
                    
                    # Process right sensor
                    img_right = tactile_imgs_right[frame_idx]
                    ff_right = tactile_processors['tactile_right'].process_frame(img_right)  # (7, 9, 3)
                    all_tactile_ff_right.append(ff_right)
                
                print(f'  ✅ Processed {len(tactile_imgs_left)} frames')
        
        except Exception as e:
            print(f'  ❌ Error processing episode: {e}')
            continue
    
    if not all_tactile_ff_left or not all_tactile_ff_right:
        print('\n❌ No tactile data collected!')
        return
    
    # Stack all data
    print(f'\nStacking data...')
    all_tactile_ff_left = np.stack(all_tactile_ff_left, axis=0)  # (N_frames, 7, 9, 3)
    all_tactile_ff_right = np.stack(all_tactile_ff_right, axis=0)  # (N_frames, 7, 9, 3)
    all_reference_ff_left = np.stack(all_reference_ff_left, axis=0)  # (N_frames, 7, 9, 3)
    all_reference_ff_right = np.stack(all_reference_ff_right, axis=0)  # (N_frames, 7, 9, 3)
    
    print(f'Left sensor: {all_tactile_ff_left.shape}')
    print(f'Right sensor: {all_tactile_ff_right.shape}')
    print(f'Reference left: {all_reference_ff_left.shape}')
    print(f'Reference right: {all_reference_ff_right.shape}')
    
    # Compute difference with reference
    all_tactile_diff_left = all_tactile_ff_left - all_reference_ff_left
    all_tactile_diff_right = all_tactile_ff_right - all_reference_ff_right
    
    print(f'Difference left: {all_tactile_diff_left.shape}')
    print(f'Difference right: {all_tactile_diff_right.shape}')
    
    # Combine left and right for overall statistics
    all_tactile_ff = np.concatenate([all_tactile_ff_left, all_tactile_ff_right], axis=0)
    all_tactile_diff = np.concatenate([all_tactile_diff_left, all_tactile_diff_right], axis=0)
    
    # Flatten for analysis
    all_data_flat = all_tactile_ff.reshape(-1, 3)  # (N_total_markers, 3)
    all_diff_flat = all_tactile_diff.reshape(-1, 3)  # (N_total_markers, 3) - DIFFERENCE data
    
    # Extract channels: [depth, dx, dy]
    depth_data = all_data_flat[:, 0]
    dx_data = all_data_flat[:, 1]
    dy_data = all_data_flat[:, 2]
    
    # Extract difference channels
    depth_diff = all_diff_flat[:, 0]
    dx_diff = all_diff_flat[:, 1]
    dy_diff = all_diff_flat[:, 2]
    
    print(f'\n{"="*80}')
    print('TACTILE FORCE FIELD DATA ANALYSIS')
    print(f'{"="*80}')
    print(f'\n📝 Note: Analyzing both RAW data and DIFFERENCE (current - reference) data')
    print(f'   Reference = first frame of each episode')
    print(f'   Total episodes analyzed: {len(episode_frame_counts)}')
    
    # ============================================================================
    # RAW DATA ANALYSIS (STACKING MODE)
    # ============================================================================
    print(f'\n{"="*80}')
    print('PART 1: RAW TACTILE DATA ANALYSIS (for STACKING mode)')
    print(f'{"="*80}')
    
    # Overall statistics
    print('\n📊 OVERALL STATISTICS (all 3 channels combined):')
    print(f'  Total samples: {all_data_flat.size:,}')
    print(f'  Total markers: {all_data_flat.shape[0]:,}')
    print(f'  Min:        {all_data_flat.min():.10f}')
    print(f'  Max:        {all_data_flat.max():.10f}')
    print(f'  Mean:       {all_data_flat.mean():.10f}')
    print(f'  Std:        {all_data_flat.std():.10f}')
    print(f'  Median:     {np.median(all_data_flat):.10f}')
    print(f'  95th %ile:  {np.percentile(all_data_flat, 95):.10f}')
    print(f'  99th %ile:  {np.percentile(all_data_flat, 99):.10f}')
    
    nonzero = np.count_nonzero(all_data_flat)
    print(f'  Non-zero:   {nonzero:,} ({100*nonzero/all_data_flat.size:.2f}%)')
    
    # Channel-wise statistics
    print('\n📊 CHANNEL STATISTICS:')
    
    channels = {
        'penetration_depth (depth)': depth_data,
        'shear_force_x (dx)': dx_data,
        'shear_force_y (dy)': dy_data
    }
    
    for channel_name, data in channels.items():
        print(f'\n  {channel_name}:')
        print(f'    Min:        {data.min():.10f}')
        print(f'    Max:        {data.max():.10f}')
        print(f'    Mean:       {data.mean():.10f}')
        print(f'    Std:        {data.std():.10f}')
        print(f'    Median:     {np.median(data):.10f}')
        print(f'    95th %ile:  {np.percentile(data, 95):.10f}')
        print(f'    99th %ile:  {np.percentile(data, 99):.10f}')
        nonzero_ch = np.count_nonzero(data)
        print(f'    Non-zero:   {nonzero_ch:,} ({100*nonzero_ch/data.size:.2f}%)')
    
    # ============================================================================
    # DIFFERENCE DATA ANALYSIS (DIFFERENCE MODE)
    # ============================================================================
    print(f'\n{"="*80}')
    print('PART 2: DIFFERENCE TACTILE DATA ANALYSIS (for DIFFERENCE mode)')
    print(f'{"="*80}')
    print('\n📝 This shows statistics for (current - reference) where reference = first frame')
    
    # Overall difference statistics
    print('\n📊 OVERALL DIFFERENCE STATISTICS (all 3 channels combined):')
    print(f'  Total samples: {all_diff_flat.size:,}')
    print(f'  Total markers: {all_diff_flat.shape[0]:,}')
    print(f'  Min:        {all_diff_flat.min():.10f}')
    print(f'  Max:        {all_diff_flat.max():.10f}')
    print(f'  Mean:       {all_diff_flat.mean():.10f}')
    print(f'  Std:        {all_diff_flat.std():.10f}')
    print(f'  Median:     {np.median(all_diff_flat):.10f}')
    print(f'  95th %ile:  {np.percentile(all_diff_flat, 95):.10f}')
    print(f'  99th %ile:  {np.percentile(all_diff_flat, 99):.10f}')
    
    nonzero_diff = np.count_nonzero(all_diff_flat)
    print(f'  Non-zero:   {nonzero_diff:,} ({100*nonzero_diff/all_diff_flat.size:.2f}%)')
    
    # Channel-wise difference statistics
    print('\n📊 CHANNEL DIFFERENCE STATISTICS:')
    
    diff_channels = {
        'penetration_depth_diff (depth - depth_ref)': depth_diff,
        'shear_force_x_diff (dx - dx_ref)': dx_diff,
        'shear_force_y_diff (dy - dy_ref)': dy_diff
    }
    
    for channel_name, data in diff_channels.items():
        print(f'\n  {channel_name}:')
        print(f'    Min:        {data.min():.10f}')
        print(f'    Max:        {data.max():.10f}')
        print(f'    Mean:       {data.mean():.10f}')
        print(f'    Std:        {data.std():.10f}')
        print(f'    Median:     {np.median(data):.10f}')
        print(f'    Abs Mean:   {np.abs(data).mean():.10f}')
        print(f'    95th %ile:  {np.percentile(data, 95):.10f}')
        print(f'    99th %ile:  {np.percentile(data, 99):.10f}')
        nonzero_ch_diff = np.count_nonzero(data)
        print(f'    Non-zero:   {nonzero_ch_diff:,} ({100*nonzero_ch_diff/data.size:.2f}%)')
    
    # Compare with pre-training statistics
    print(f'\n{"="*80}')
    print('COMPARISON WITH PRE-TRAINING DATA')
    print(f'{"="*80}')
    
    pretraining_stats = {
        'overall': {
            'min': -0.002636931836605072,
            'max': 0.0026971683837473392,
            'mean': 0.00018681943765841424,
            'std': 0.0005340970819815993,
        },
        'shear_force_x': {
            'min': -0.002636931836605072,
            'max': 0.0026971683837473392,
            'mean': 4.721667210105807e-06,
            'std': 0.0007895460585132241,
        },
        'shear_force_y': {
            'min': -0.0015303436666727066,
            'max': 0.0019416429568082094,
            'mean': 0.00031987775582820177,
            'std': 0.00042217402369715273,
        },
        'penetration_depth': {
            'min': 0.0,
            'max': 0.0014810034772381186,
            'mean': 0.00040981388883665204,
            'std': 0.00032625554013065994,
        }
    }
    
    print('\n📈 RANGE COMPARISON:')
    print(f'  Overall:')
    print(f'    Pre-training range: [{pretraining_stats["overall"]["min"]:.6f}, {pretraining_stats["overall"]["max"]:.6f}]')
    print(f'    Real-world range:   [{all_data_flat.min():.6f}, {all_data_flat.max():.6f}]')
    print(f'    Ratio (max):        {all_data_flat.max() / pretraining_stats["overall"]["max"]:.2f}x')
    
    channel_mapping = {
        'penetration_depth (depth)': 'penetration_depth',
        'shear_force_x (dx)': 'shear_force_x',
        'shear_force_y (dy)': 'shear_force_y'
    }
    
    for channel_name, data in channels.items():
        pretrain_key = channel_mapping[channel_name]
        pretrain_stat = pretraining_stats[pretrain_key]
        print(f'\n  {channel_name}:')
        print(f'    Pre-training range: [{pretrain_stat["min"]:.6f}, {pretrain_stat["max"]:.6f}]')
        print(f'    Real-world range:   [{data.min():.6f}, {data.max():.6f}]')
        if pretrain_stat["max"] > 0:
            print(f'    Ratio (max):        {data.max() / pretrain_stat["max"]:.2f}x')
        if pretrain_stat["std"] > 0:
            print(f'    Std ratio:          {data.std() / pretrain_stat["std"]:.2f}x')
    
    # Scaling recommendations
    print(f'\n{"="*80}')
    print('SCALING RECOMMENDATIONS')
    print(f'{"="*80}')
    
    print('\n🔧 Model was trained with: scale * 1000, clip to [-10, 10]')
    print('   This means pre-training data in range ~[-0.0027, 0.0027] → [-2.7, 2.7] after scaling')
    
    # Calculate what scaling would map real-world max to similar range
    pretrain_max = pretraining_stats['overall']['max']
    realworld_max = all_data_flat.max()
    
    # Pre-training: max * 1000 ≈ 2.7
    # For real-world to have similar effective range
    suggested_scale = (pretrain_max / realworld_max) * 1000 if realworld_max > 0 else 1000
    
    print(f'\n💡 RECOMMENDED SCALING:')
    print(f'   Pre-training max value: {pretrain_max:.6f}')
    print(f'   Real-world max value:   {realworld_max:.6f}')
    print(f'   Ratio:                  {realworld_max / pretrain_max:.2f}x')
    print(f'\n   Suggested scale factor: {suggested_scale:.1f}')
    print(f'   (vs pre-training scale: 1000)')
    print(f'\n   With this scaling:')
    print(f'     Real-world [{all_data_flat.min():.6f}, {all_data_flat.max():.6f}]')
    print(f'     → [{all_data_flat.min()*suggested_scale:.2f}, {all_data_flat.max()*suggested_scale:.2f}]')
    print(f'     (Pre-training was scaled to ~[-2.7, 2.7])')
    
    # Alternative: keep 1000 scaling but different clipping
    print(f'\n   Alternative: Keep scale=1000, adjust clipping:')
    print(f'     With scale=1000:')
    print(f'     Real-world [{all_data_flat.min():.6f}, {all_data_flat.max():.6f}]')
    print(f'     → [{all_data_flat.min()*1000:.2f}, {all_data_flat.max()*1000:.2f}]')
    if all_data_flat.max() * 1000 > 10:
        clip_value = np.ceil(all_data_flat.max() * 1000)
        print(f'     Suggested clip range: [{-clip_value:.0f}, {clip_value:.0f}]')
    else:
        print(f'     Standard clip range [-10, 10] should work')
    
    # Difference mode recommendations
    print(f'\n📊 DIFFERENCE MODE STATISTICS:')
    print(f'   When using reference_tactile_use_difference=True:')
    print(f'   Difference range: [{all_diff_flat.min():.6f}, {all_diff_flat.max():.6f}]')
    print(f'   Difference mean:  {all_diff_flat.mean():.6f} (should be close to 0)')
    print(f'   Difference std:   {all_diff_flat.std():.6f}')
    print(f'\n   With scale=1000:')
    print(f'     Difference [{all_diff_flat.min():.6f}, {all_diff_flat.max():.6f}]')
    print(f'     → [{all_diff_flat.min()*1000:.2f}, {all_diff_flat.max()*1000:.2f}]')
    if max(abs(all_diff_flat.min()), abs(all_diff_flat.max())) * 1000 > 10:
        diff_clip_value = np.ceil(max(abs(all_diff_flat.min()), abs(all_diff_flat.max())) * 1000)
        print(f'     Suggested clip range: [{-diff_clip_value:.0f}, {diff_clip_value:.0f}]')
    else:
        print(f'     Standard clip range [-10, 10] should work')
    
    print(f'\n📝 NOTES ON TWO MODES:')
    print(f'   STACKING MODE (reference_tactile_use_difference=False):')
    print(f'     - Concatenates current + reference tactile → 6 channels')
    print(f'     - Use statistics from "RAW TACTILE DATA ANALYSIS" above')
    print(f'     - Final shape: (T, 9, 7, 9) = 6 force + 3 coords')
    print(f'\n   DIFFERENCE MODE (reference_tactile_use_difference=True):')
    print(f'     - Computes (current - reference) → 3 channels')
    print(f'     - Use statistics from "DIFFERENCE TACTILE DATA ANALYSIS" above')
    print(f'     - Final shape: (T, 6, 7, 9) = 3 force + 3 coords')
    print(f'     - Generally has smaller range and mean closer to 0')
    
    # Save detailed statistics to JSON
    output_file = os.path.join(dataset_dir, 'tactile_data_statistics.json')
    stats_dict = {
        'dataset_dir': dataset_dir,
        'num_episodes_analyzed': len(episodes_paths),
        'total_frames': len(all_tactile_ff_left) + len(all_tactile_ff_right),
        'overall_statistics_raw': {
            'min': float(all_data_flat.min()),
            'max': float(all_data_flat.max()),
            'mean': float(all_data_flat.mean()),
            'std': float(all_data_flat.std()),
            'median': float(np.median(all_data_flat)),
            'percentile_95': float(np.percentile(all_data_flat, 95)),
            'percentile_99': float(np.percentile(all_data_flat, 99)),
            'num_samples': int(all_data_flat.size),
            'num_nonzero': int(nonzero),
            'nonzero_percentage': float(100 * nonzero / all_data_flat.size)
        },
        'overall_statistics_difference': {
            'min': float(all_diff_flat.min()),
            'max': float(all_diff_flat.max()),
            'mean': float(all_diff_flat.mean()),
            'std': float(all_diff_flat.std()),
            'median': float(np.median(all_diff_flat)),
            'abs_mean': float(np.abs(all_diff_flat).mean()),
            'percentile_95': float(np.percentile(all_diff_flat, 95)),
            'percentile_99': float(np.percentile(all_diff_flat, 99)),
            'num_samples': int(all_diff_flat.size),
            'num_nonzero': int(nonzero_diff),
            'nonzero_percentage': float(100 * nonzero_diff / all_diff_flat.size)
        },
        'channel_statistics_raw': {
            'penetration_depth': {
                'min': float(depth_data.min()),
                'max': float(depth_data.max()),
                'mean': float(depth_data.mean()),
                'std': float(depth_data.std()),
                'median': float(np.median(depth_data)),
                'percentile_95': float(np.percentile(depth_data, 95)),
                'percentile_99': float(np.percentile(depth_data, 99)),
            },
            'shear_force_x': {
                'min': float(dx_data.min()),
                'max': float(dx_data.max()),
                'mean': float(dx_data.mean()),
                'std': float(dx_data.std()),
                'median': float(np.median(dx_data)),
                'percentile_95': float(np.percentile(dx_data, 95)),
                'percentile_99': float(np.percentile(dx_data, 99)),
            },
            'shear_force_y': {
                'min': float(dy_data.min()),
                'max': float(dy_data.max()),
                'mean': float(dy_data.mean()),
                'std': float(dy_data.std()),
                'median': float(np.median(dy_data)),
                'percentile_95': float(np.percentile(dy_data, 95)),
                'percentile_99': float(np.percentile(dy_data, 99)),
            }
        },
        'channel_statistics_difference': {
            'penetration_depth_diff': {
                'min': float(depth_diff.min()),
                'max': float(depth_diff.max()),
                'mean': float(depth_diff.mean()),
                'std': float(depth_diff.std()),
                'median': float(np.median(depth_diff)),
                'abs_mean': float(np.abs(depth_diff).mean()),
                'percentile_95': float(np.percentile(depth_diff, 95)),
                'percentile_99': float(np.percentile(depth_diff, 99)),
            },
            'shear_force_x_diff': {
                'min': float(dx_diff.min()),
                'max': float(dx_diff.max()),
                'mean': float(dx_diff.mean()),
                'std': float(dx_diff.std()),
                'median': float(np.median(dx_diff)),
                'abs_mean': float(np.abs(dx_diff).mean()),
                'percentile_95': float(np.percentile(dx_diff, 95)),
                'percentile_99': float(np.percentile(dx_diff, 99)),
            },
            'shear_force_y_diff': {
                'min': float(dy_diff.min()),
                'max': float(dy_diff.max()),
                'mean': float(dy_diff.mean()),
                'std': float(dy_diff.std()),
                'median': float(np.median(dy_diff)),
                'abs_mean': float(np.abs(dy_diff).mean()),
                'percentile_95': float(np.percentile(dy_diff, 95)),
                'percentile_99': float(np.percentile(dy_diff, 99)),
            }
        },
        'recommended_scaling': {
            'suggested_scale_factor': float(suggested_scale),
            'pretraining_scale_factor': 1000,
            'pretraining_clip_range': [-10, 10],
            'note': 'Scale factor normalizes real-world data to similar range as pre-training data'
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(stats_dict, f, indent=2)
    
    print(f'\n💾 Detailed statistics saved to: {output_file}')
    print(f'{"="*80}\n')

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze tactile force field data from episodes')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing episode_*.hdf5 files')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config yaml file with tactile settings')
    parser.add_argument('--max-episodes', type=int, default=None,
                        help='Maximum number of episodes to analyze (default: all)')
    
    args = parser.parse_args()
    
    analyze_tactile_data(args.data_dir, args.config, args.max_episodes)
