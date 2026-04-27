#!/usr/bin/env python
# coding: utf-8
"""
Script to plot end-effector pose and delta pose from a data episode.
Visualizes position (x, y, z), orientation (roll, pitch, yaw), gripper width,
and their frame-to-frame changes (deltas) with proper angle wrapping handling.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import mplcursors
from tqdm import tqdm
import sys

# Add path for importing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gendp.common.data_utils import load_dict_from_hdf5

# Enable interactive mode
plt.ion()


def wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def compute_angle_delta(angles):
    """
    Compute frame-to-frame angle differences with proper wrapping.
    
    Args:
        angles: (T, 3) array of angles in radians
    
    Returns:
        delta_angles: (T-1, 3) array of angle differences, wrapped to [-pi, pi]
    """
    delta = np.diff(angles, axis=0)
    # Wrap each delta to [-pi, pi]
    delta = wrap_angle(delta)
    return delta

### Hyper parameters
curr_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = f'{curr_dir}/../../data/crayon_cross'
# data_dir = f'{curr_dir}/../../data/scraper_combined'
# data_dir = f'{curr_dir}/../../data/sapien_demo/pencil_insertion_demo'

plot_std = True  # Set to True to show standard deviation as shaded area

# Find all episode files in the directory
import glob
episode_files = sorted(glob.glob(os.path.join(data_dir, 'episode_*.hdf5')))
print(f"Found {len(episode_files)} episode files in {data_dir}")

if len(episode_files) == 0:
    raise ValueError(f"No episode files found in {data_dir}")

# Load all episodes and collect data
all_positions_z = []
all_orientations_roll = []
all_orientations_pitch = []
all_delta_positions_z = []
all_delta_orientations_roll = []
all_delta_orientations_pitch = []

max_length = 0

print("Loading episodes...")
for episode_file in tqdm(episode_files):
    data_dict, _ = load_dict_from_hdf5(episode_file)
    
    # Extract ee_pose: [x, y, z, rx, ry, rz, gripper_width]
    ee_pose = data_dict['observations']['ee_pose']  # Shape: (T, 7)
    T = ee_pose.shape[0]
    max_length = max(max_length, T)
    
    # Split into components
    positions = ee_pose[:, :3]  # (T, 3) - x, y, z
    orientations = ee_pose[:, 3:6]  # (T, 3) - roll, pitch, yaw
    
    # Wrap orientations to [-pi, pi] for consistency
    orientations = wrap_angle(orientations)
    
    # Calculate deltas (frame-to-frame changes)
    delta_positions = np.diff(positions, axis=0)  # (T-1, 3)
    delta_orientations = compute_angle_delta(ee_pose[:, 3:6])  # (T-1, 3) with proper wrapping
    
    # Store data for averaging
    all_positions_z.append(positions[:, 2])
    all_orientations_roll.append(orientations[:, 0])
    all_orientations_pitch.append(orientations[:, 1])
    all_delta_positions_z.append(delta_positions[:, 2])
    all_delta_orientations_roll.append(delta_orientations[:, 0])
    all_delta_orientations_pitch.append(delta_orientations[:, 1])

print(f"Max trajectory length: {max_length} frames")

# Pad sequences to same length for averaging
def pad_sequences(sequences, max_len):
    """Pad sequences with NaN to max length for proper averaging."""
    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            padded_seq = np.concatenate([seq, np.full(max_len - len(seq), np.nan)])
        else:
            padded_seq = seq[:max_len]
        padded.append(padded_seq)
    return np.array(padded)

# Pad all sequences
positions_z_padded = pad_sequences(all_positions_z, max_length)
orientations_roll_padded = pad_sequences(all_orientations_roll, max_length)
orientations_pitch_padded = pad_sequences(all_orientations_pitch, max_length)
delta_positions_z_padded = pad_sequences(all_delta_positions_z, max_length - 1)
delta_orientations_roll_padded = pad_sequences(all_delta_orientations_roll, max_length - 1)
delta_orientations_pitch_padded = pad_sequences(all_delta_orientations_pitch, max_length - 1)

# Compute mean and std (ignoring NaN values)
positions_z_mean = np.nanmean(positions_z_padded, axis=0)
positions_z_std = np.nanstd(positions_z_padded, axis=0)
orientations_roll_mean = np.nanmean(orientations_roll_padded, axis=0)
orientations_roll_std = np.nanstd(orientations_roll_padded, axis=0)
orientations_pitch_mean = np.nanmean(orientations_pitch_padded, axis=0)
orientations_pitch_std = np.nanstd(orientations_pitch_padded, axis=0)

delta_positions_z_mean = np.nanmean(delta_positions_z_padded, axis=0)
delta_positions_z_std = np.nanstd(delta_positions_z_padded, axis=0)
delta_orientations_roll_mean = np.nanmean(delta_orientations_roll_padded, axis=0)
delta_orientations_roll_std = np.nanstd(delta_orientations_roll_padded, axis=0)
delta_orientations_pitch_mean = np.nanmean(delta_orientations_pitch_padded, axis=0)
delta_orientations_pitch_std = np.nanstd(delta_orientations_pitch_padded, axis=0)

T = max_length
print(f"Computing mean across {len(episode_files)} episodes")

# Split into components
positions = None  # Not needed anymore, using mean values
orientations = None  # Not needed anymore, using mean values
gripper_width = None  # Not used in plots

# These are now computed as means across all episodes
# positions_z_mean, orientations_roll_mean, orientations_pitch_mean
# delta_positions_z_mean, delta_orientations_roll_mean, delta_orientations_pitch_mean

# Wrap orientations to [-pi, pi] for consistency - already done during loading

# Calculate deltas (frame-to-frame changes) - already done during loading

# Calculate magnitudes - compute from mean deltas
delta_position_magnitude = np.abs(delta_positions_z_mean)  # (T-1,)
delta_orientation_magnitude = np.sqrt(delta_orientations_roll_mean**2 + delta_orientations_pitch_mean**2)  # (T-1,)

# Time arrays
time_full = np.arange(T)
time_delta = np.arange(T - 1)

# Create simplified plots
fig = plt.figure(figsize=(14, 8))

# ========== Plot 1: Absolute Position (Z only) ==========
ax1 = plt.subplot(2, 2, 1)
ax1.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.3, color='gray', zorder=0)
ax1.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.2, color='gray', zorder=0)
ax1.plot(time_full, positions_z_mean, 'b-', linewidth=2.5, label='z (mean)', zorder=2)
if plot_std:
    ax1.fill_between(time_full, positions_z_mean - positions_z_std, positions_z_mean + positions_z_std, 
                     color='b', alpha=0.2, label='±1 std', zorder=1)
ax1.set_xlabel('Frame', fontsize=12)
ax1.set_ylabel('Position Z (m)', fontsize=12)
ax1.set_title('Absolute Position (Z) - Mean Across Episodes', fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=11)
ax1.minorticks_on()

# ========== Plot 2: Delta Position (Z only) ==========
ax2 = plt.subplot(2, 2, 2)
ax2.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.3, color='gray', zorder=0)
ax2.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.2, color='gray', zorder=0)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1.5, zorder=1)
ax2.plot(time_delta, delta_positions_z_mean, 'b-', linewidth=2.5, label='Δz (mean)', zorder=2)
if plot_std:
    ax2.fill_between(time_delta, delta_positions_z_mean - delta_positions_z_std, 
                     delta_positions_z_mean + delta_positions_z_std, 
                     color='b', alpha=0.2, label='±1 std', zorder=1)
ax2.set_xlabel('Frame', fontsize=12)
ax2.set_ylabel('Delta Position Z (m)', fontsize=12)
ax2.set_title('Delta Position (Z) - Mean Across Episodes', fontsize=14, fontweight='bold')
ax2.legend(loc='best', fontsize=11)
ax2.minorticks_on()

# ========== Plot 3: Absolute Rotation (Roll and Pitch only) ==========
ax3 = plt.subplot(2, 2, 3)
ax3.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.3, color='gray', zorder=0)
ax3.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.2, color='gray', zorder=0)
ax3.plot(time_full, orientations_roll_mean, 'r-', linewidth=2.5, label='roll (mean)', zorder=2)
ax3.plot(time_full, orientations_pitch_mean, 'g-', linewidth=2.5, label='pitch (mean)', zorder=2)
if plot_std:
    ax3.fill_between(time_full, orientations_roll_mean - orientations_roll_std, 
                     orientations_roll_mean + orientations_roll_std, 
                     color='r', alpha=0.15, zorder=1)
    ax3.fill_between(time_full, orientations_pitch_mean - orientations_pitch_std, 
                     orientations_pitch_mean + orientations_pitch_std, 
                     color='g', alpha=0.15, zorder=1)
ax3.set_xlabel('Frame', fontsize=12)
ax3.set_ylabel('Orientation (rad)', fontsize=12)
ax3.set_title('Absolute Rotation (Roll, Pitch) - Mean Across Episodes', fontsize=14, fontweight='bold')
ax3.legend(loc='best', fontsize=11)
ax3.minorticks_on()

# ========== Plot 4: Delta Rotation (Roll and Pitch only) ==========
ax4 = plt.subplot(2, 2, 4)
ax4.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.3, color='gray', zorder=0)
ax4.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.2, color='gray', zorder=0)
ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1.5, zorder=1)
ax4.plot(time_delta, delta_orientations_roll_mean, 'r-', linewidth=2.5, label='Δroll (mean)', zorder=2)
ax4.plot(time_delta, delta_orientations_pitch_mean, 'g-', linewidth=2.5, label='Δpitch (mean)', zorder=2)
if plot_std:
    ax4.fill_between(time_delta, delta_orientations_roll_mean - delta_orientations_roll_std, 
                     delta_orientations_roll_mean + delta_orientations_roll_std, 
                     color='r', alpha=0.15, zorder=1)
    ax4.fill_between(time_delta, delta_orientations_pitch_mean - delta_orientations_pitch_std, 
                     delta_orientations_pitch_mean + delta_orientations_pitch_std, 
                     color='g', alpha=0.15, zorder=1)
ax4.set_xlabel('Frame', fontsize=12)
ax4.set_ylabel('Delta Orientation (rad)', fontsize=12)
ax4.set_title('Delta Rotation (Roll, Pitch) - Mean Across Episodes', fontsize=14, fontweight='bold')
ax4.legend(loc='best', fontsize=11)
ax4.minorticks_on()

plt.tight_layout()

# Add interactive cursors to all plots for precise value inspection
cursor1 = Cursor(ax1, useblit=True, color='red', linewidth=1, linestyle='--')
cursor2 = Cursor(ax2, useblit=True, color='red', linewidth=1, linestyle='--')
cursor3 = Cursor(ax3, useblit=True, color='red', linewidth=1, linestyle='--')
cursor4 = Cursor(ax4, useblit=True, color='red', linewidth=1, linestyle='--')

# Add mplcursors for showing exact values on hover
mplcursors.cursor([ax1.lines[0]], hover=True).connect(
    "add", lambda sel: sel.annotation.set_text(
        f'Frame: {int(sel.target[0])}\nZ (mean): {sel.target[1]:.6f} m'
    )
)

mplcursors.cursor([ax2.lines[1]], hover=True).connect(  # Skip the axhline at index 0
    "add", lambda sel: sel.annotation.set_text(
        f'Frame: {int(sel.target[0])}\nΔZ (mean): {sel.target[1]:.6f} m'
    )
)

mplcursors.cursor([ax3.lines[0], ax3.lines[1]], hover=True).connect(
    "add", lambda sel: sel.annotation.set_text(
        f'Frame: {int(sel.target[0])}\n{sel.artist.get_label()}: {sel.target[1]:.6f} rad'
    )
)

mplcursors.cursor([ax4.lines[1], ax4.lines[2]], hover=True).connect(  # Skip axhline
    "add", lambda sel: sel.annotation.set_text(
        f'Frame: {int(sel.target[0])}\n{sel.artist.get_label()}: {sel.target[1]:.6f} rad'
    )
)

# Save the figure
output_path = os.path.join(data_dir, f'ee_pose_mean_across_{len(episode_files)}_episodes.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")
print("\n" + "="*60)
print("Interactive Mode:")
print("  - Use mouse to hover over data points to see exact values")
print("  - Use toolbar buttons to zoom, pan, and reset view")
print("  - Press 'q' or close window to exit")
print("="*60)

# Print detailed statistics
print("\n" + "="*60)
print(f"Statistics Across {len(episode_files)} Episodes")
print("="*60)
print(f"\nMax trajectory length: {T} frames")
print(f"\nPosition Z Statistics (mean across episodes):")
print(f"  Mean: {np.nanmean(positions_z_mean):.4f} m")
print(f"  Std:  {np.nanmean(positions_z_std):.4f} m (avg std)")
print(f"  Range: [{np.nanmin(positions_z_mean):.4f}, {np.nanmax(positions_z_mean):.4f}] m")
print(f"\nOrientation Statistics (mean across episodes, wrapped to [-π, π]):")
print(f"  Roll  - Mean: {np.nanmean(orientations_roll_mean):.4f} rad, Avg Std: {np.nanmean(orientations_roll_std):.4f} rad")
print(f"  Pitch - Mean: {np.nanmean(orientations_pitch_mean):.4f} rad, Avg Std: {np.nanmean(orientations_pitch_std):.4f} rad")
print(f"\nDelta Statistics (mean across episodes):")
print(f"  Mean |ΔZ|:     {np.nanmean(np.abs(delta_positions_z_mean)):.6f} m")
print(f"  Max |ΔZ|:      {np.nanmax(np.abs(delta_positions_z_mean)):.6f} m")
print(f"  Mean |ΔRoll|:  {np.nanmean(np.abs(delta_orientations_roll_mean)):.6f} rad")
print(f"  Max |ΔRoll|:   {np.nanmax(np.abs(delta_orientations_roll_mean)):.6f} rad")
print(f"  Mean |ΔPitch|: {np.nanmean(np.abs(delta_orientations_pitch_mean)):.6f} rad")
print(f"  Max |ΔPitch|:  {np.nanmax(np.abs(delta_orientations_pitch_mean)):.6f} rad")

# Keep the plot window open and interactive
print("\nShowing interactive plot... (Close window or press Ctrl+C to exit)")
plt.show(block=True)
