"""
Follow pre-recorded robot keypoints with randomization for data collection.

This script loads keypoints and moves the robot through a 2-stage trajectory:
- Stage 0: Approach (with position/rotation randomization, gripper open)
- Grasp: Automatically closes gripper at end of Stage 0 and waits 3 seconds
- Stage 1: Move (with position/rotation randomization, gripper closed)

Each keypoint can have its own noise levels specified in the JSON file via
"pos_noise" (position noise in meters) and "rot_noise" (rotation noise in radians).
If not specified, defaults are used (0.02m and 0.1rad respectively).

Recording workflow:
1. Press 'h' to home - generates new randomized keypoints and moves robot to randomized first keypoint (3 seconds)
2. Press 'c' to start recording - begins trajectory execution using cached randomized keypoints
3. Robot executes continuous trajectory through all stages
4. Recording auto-stops after reaching final keypoint

The robot automatically:
- Generates new randomization each time 'h' is pressed
- Moves smoothly between waypoints at constant speed (continuous interpolation)
- Closes gripper after completing Stage 0 (Approach) and waits 3 seconds
- Keeps gripper closed during Stage 1 (Move)
- Stops recording 2 seconds after reaching the final keypoint

Usage:
python follow_keypoints_with_randomization.py \
    -k <keypoints_file.json> \
    -o <data_save_dir> \
    --robot_ip <ip_of_franka> \
    --movement_speed 0.05

Commands (type and press Enter):
- h: Home to randomized first keypoint (generates new noise for all keypoints)
- c: Start recording episode (uses cached randomized keypoints from last 'h')
- s: Stop recording episode (or auto-stops after final keypoint)
- space: Manually move to next stage (optional)
- reset: Reset to beginning of trajectory
- q: Exit program
- status: Show current status
- help: Show this help
"""

import os
import time
import threading
import queue
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import json
from pathlib import Path
import scipy.spatial.transform as st

from gendp.real_world.real_env_franka_gripper_gelsight import RealEnvFranka, CAMERA_NAMES
from gendp.common.precise_sleep import precise_wait
from gendp.real_world.keystroke_counter import KeystrokeCounter, Key, KeyCode

# Global variables for communication between threads
command_queue = queue.Queue()
robot_state = {
    'running': False,
    'recording': False,
    'episode_id': 0,
    'stage': 0,
    'keypoint_idx': 0,
    'stop': False,
    'reset_requested': False,
    'home_requested': False,
    'gripper_pos': 0.08,
    'moving_to_keypoint': False,
    'movement_start_time': 0.0,
    'movement_duration': 0.0,
    'homing': False,
    'homing_trajectory': None,
    'homing_step': 0,
    'gripper_closing': False,
    'gripper_close_time': 0.0,
    'reached_final_keypoint': False,
    'final_keypoint_time': 0.0
}

def load_keypoints(keypoints_file):
    """Load keypoints from JSON file with per-keypoint noise levels"""
    with open(keypoints_file, 'r') as f:
        data = json.load(f)
    
    keypoints = {0: [], 1: []}
    for stage_id in [0, 1]:
        stage_key = f'stage_{stage_id}'
        if stage_key in data['keypoints']:
            for kp in data['keypoints'][stage_key]:
                ee_pose = np.array(kp['ee_pose'])
                gripper_pos = kp['gripper_pos']
                # Read per-keypoint noise levels (use defaults if not specified)
                pos_noise = kp.get('pos_noise', 0.02)  # default 0.02m or [0.02, 0.02, 0.02]
                # Convert scalar to 3D array if needed
                if isinstance(pos_noise, (int, float)):
                    pos_noise = np.array([pos_noise, pos_noise, pos_noise])
                else:
                    pos_noise = np.array(pos_noise)
                    if pos_noise.shape != (3,):
                        raise ValueError(f"pos_noise must be scalar or 3D array, got shape {pos_noise.shape}")
                
                rot_noise = kp.get('rot_noise', 0.1)   # default 0.1rad
                keypoints[stage_id].append((ee_pose, gripper_pos, pos_noise, rot_noise))
    
    return keypoints, data.get('metadata', {})

def add_noise_to_pose(ee_pose, pos_noise_std, rot_noise_std):
    """
    Add Gaussian noise to EE pose.
    
    Args:
        ee_pose: [x, y, z, rx, ry, rz] in Euler angles (XYZ convention)
        pos_noise_std: Standard deviation for position noise (meters) - scalar or [x_std, y_std, z_std]
        rot_noise_std: Standard deviation for rotation noise (radians)
    
    Returns:
        Noisy EE pose with same format
    """
    noisy_pose = ee_pose.copy()
    
    # Add position noise - support both scalar and 3D array
    if isinstance(pos_noise_std, (int, float)):
        pos_noise = np.random.normal(0, pos_noise_std, size=3)
    else:
        # pos_noise_std is [x_std, y_std, z_std]
        pos_noise = np.array([
            np.random.normal(0, pos_noise_std[0]),
            np.random.normal(0, pos_noise_std[1]),
            np.random.normal(0, pos_noise_std[2])
        ])
    noisy_pose[:3] += pos_noise
    
    # Add rotation noise
    rot_noise = np.random.normal(0, rot_noise_std, size=3)
    noisy_pose[3:6] += rot_noise
    
    return noisy_pose

def interpolate_poses(pose1, pose2, alpha):
    """
    Interpolate between two poses using SLERP for rotations.
    
    Args:
        pose1, pose2: [x, y, z, rx, ry, rz] in Euler angles
        alpha: Interpolation factor [0, 1]
    
    Returns:
        Interpolated pose
    """
    # Linear interpolation for position
    pos_interp = (1 - alpha) * pose1[:3] + alpha * pose2[:3]
    
    # SLERP for rotation
    rot1 = st.Rotation.from_euler('xyz', pose1[3:6])
    rot2 = st.Rotation.from_euler('xyz', pose2[3:6])
    
    # Create rotation from pose1 to pose2
    rot_delta = rot1.inv() * rot2
    rot_interp = rot1 * st.Rotation.from_rotvec(alpha * rot_delta.as_rotvec())
    
    euler_interp = rot_interp.as_euler('xyz')
    
    return np.concatenate([pos_interp, euler_interp])

def generate_smooth_trajectory(start_pose, end_pose, num_steps=30):
    """
    Generate smooth trajectory between two EE poses using linear interpolation.
    
    Args:
        start_pose: [x, y, z, rx, ry, rz, gripper]
        end_pose: [x, y, z, rx, ry, rz, gripper]
        num_steps: Number of waypoints in trajectory
    
    Returns:
        List of waypoint poses
    """
    trajectory = []
    for i in range(num_steps):
        alpha = (i + 1) / num_steps
        # Use interpolate_poses for position and rotation (returns 6D pose)
        pose_6d = interpolate_poses(start_pose[:6], end_pose[:6], alpha)
        # Linear interpolation for gripper
        gripper = (1 - alpha) * start_pose[6] + alpha * end_pose[6]
        # Concatenate to create 7D waypoint
        waypoint = np.concatenate([pose_6d, [gripper]])
        trajectory.append(waypoint)
    return trajectory

def terminal_input_thread():
    """Handle terminal input in separate thread"""
    print("\n" + "="*60)
    print("ROBOT KEYPOINT FOLLOWER WITH RANDOMIZATION")
    print("="*60)
    print("Commands:")
    print("  c       - Move to start and begin recording episode")
    print("  s       - Stop recording episode")
    print("  space   - Next stage")
    print("  h       - Home to first keypoint")
    print("  reset   - Reset to start")
    print("  q       - Exit program")
    print("  status  - Show status")
    print("  help    - Show commands")
    print("="*60)
    print("Type commands and press Enter...")
    
    while not robot_state['stop']:
        try:
            cmd = input().strip().lower()
            if cmd:
                command_queue.put(cmd)
                if cmd == 'q':
                    break
        except (EOFError, KeyboardInterrupt):
            command_queue.put('q')
            break

def process_commands(key_counter, env, output_dir):
    """Process commands from terminal input and spacemouse"""
    global robot_state
    
    # Process terminal commands
    while not command_queue.empty():
        try:
            command = command_queue.get_nowait()
            
            if command == 'q':
                robot_state['stop'] = True
                print('🔴 Quitting...')
            elif command == 'c':
                # Start recording immediately (robot should already be at home position)
                env.start_episode(time.time(), curr_outdir=output_dir)
                key_counter.clear()
                robot_state['recording'] = True
                robot_state['stage'] = 0
                robot_state['keypoint_idx'] = 0
                robot_state['moving_to_keypoint'] = False
                robot_state['gripper_closing'] = False
                robot_state['gripper_close_time'] = 0.0
                robot_state['reached_final_keypoint'] = False
                robot_state['final_keypoint_time'] = 0.0
                print('🔴 Recording started!')
            elif command == 's':
                env.end_episode(curr_outdir=output_dir, incr_epi=True)
                key_counter.clear()
                robot_state['recording'] = False
                print('⏹️  Recording stopped!')
            elif command == 'space':
                if robot_state['stage'] < 2:
                    robot_state['stage'] += 1
                    robot_state['keypoint_idx'] = 0
                    print(f"⏭️  Moved to stage {robot_state['stage']}")
                else:
                    print("ℹ️  Already at final stage")
            elif command == 'reset':
                robot_state['stage'] = 0
                robot_state['keypoint_idx'] = 0
                print('🔄 Reset to start of trajectory')
            elif command == 'h':
                robot_state['home_requested'] = True
                print('🏠 Homing to first keypoint...')
            elif command == 'status':
                status = f"Episode: {robot_state['episode_id']}, Stage: {robot_state['stage']}"
                status += f", Keypoint: {robot_state['keypoint_idx']}"
                status += f", Recording: {'YES' if robot_state['recording'] else 'NO'}"
                print(f"📊 Status: {status}")
            elif command == 'help':
                print("\nCommands: c(record) s(stop) space(next stage) h(home) reset q(quit) status help")
            else:
                print(f"❓ Unknown command: {command}. Type 'help' for commands.")
                
        except queue.Empty:
            break
    
    # Process SpaceMouse/KeystrokeCounter commands
    press_events = key_counter.get_press_events()
    for key_stroke in press_events:
        if key_stroke == KeyCode(char='q'):
            robot_state['stop'] = True
            print('🔴 Quitting...')
        elif key_stroke == KeyCode(char='c'):
            # Start recording immediately (robot should already be at home position)
            env.start_episode(time.time(), curr_outdir=output_dir)
            key_counter.clear()
            robot_state['recording'] = True
            robot_state['stage'] = 0
            robot_state['keypoint_idx'] = 0
            robot_state['moving_to_keypoint'] = False
            robot_state['gripper_closing'] = False
            robot_state['gripper_close_time'] = 0.0
            robot_state['reached_final_keypoint'] = False
            robot_state['final_keypoint_time'] = 0.0
            print('🔴 Recording started!')
        elif key_stroke == KeyCode(char='s'):
            env.end_episode(curr_outdir=output_dir, incr_epi=True)
            key_counter.clear()
            robot_state['recording'] = False
            print('⏹️  Recording stopped!')
        elif key_stroke == Key.space:
            if robot_state['stage'] < 2:
                robot_state['stage'] += 1
                robot_state['keypoint_idx'] = 0
                print(f"⏭️  Moved to stage {robot_state['stage']}")
        elif key_stroke == KeyCode(char='h'):
            robot_state['home_requested'] = True
            print('🏠 Homing to first keypoint...')

def save_visualization_images(vis_img, output_dir, iter_idx, save_interval=30):
    """Save visualization images periodically"""
    if iter_idx % save_interval == 0:
        viz_dir = os.path.join(output_dir, 'visualization')
        os.makedirs(viz_dir, exist_ok=True)
        latest_file_name = os.path.join(viz_dir, 'latest.jpg')
        cv2.imwrite(latest_file_name, vis_img)

def compute_target_action(current_obs, target_pose, target_gripper, stage_keypoints, 
                          keypoint_idx, pos_noise_std, rot_noise_std, 
                          randomize_per_episode, episode_noise_cache, movement_speed):
    """
    Compute target action to reach the target keypoint.
    
    Args:
        current_obs: Current robot observations
        target_pose: Target EE pose [x, y, z, rx, ry, rz]
        target_gripper: Target gripper position
        stage_keypoints: All keypoints for current stage (each: pose, gripper, pos_noise, rot_noise)
        keypoint_idx: Current keypoint index
        pos_noise_std: Position noise standard deviation for this keypoint
        rot_noise_std: Rotation noise standard deviation for this keypoint
        randomize_per_episode: If True, use same noise for entire episode
        episode_noise_cache: Cache for episode-level noise
        movement_speed: Speed for moving between keypoints (m/s)
    
    Returns:
        (Target joint positions including gripper, noisy_target_pose for visualization)
    """
    # Get cache key
    cache_key = (robot_state['stage'], keypoint_idx)
    
    # Apply noise if enabled (using per-keypoint noise levels passed as parameters)
    pos_noise_enabled = np.any(pos_noise_std > 0) if isinstance(pos_noise_std, np.ndarray) else pos_noise_std > 0
    if pos_noise_enabled or rot_noise_std > 0:
        if randomize_per_episode:
            # Use cached noise for entire episode
            if cache_key not in episode_noise_cache:
                episode_noise_cache[cache_key] = add_noise_to_pose(
                    target_pose, pos_noise_std, rot_noise_std)
            noisy_target_pose = episode_noise_cache[cache_key]
        else:
            # Generate new noise each iteration
            noisy_target_pose = add_noise_to_pose(target_pose, pos_noise_std, rot_noise_std)
    else:
        noisy_target_pose = target_pose
    
    # Get current EE pose (without gripper for comparison)
    current_ee_pose_full = current_obs['ee_pose'][-1]  # [x, y, z, rx, ry, rz, gripper]
    current_ee_pose = current_ee_pose_full[:6]  # [x, y, z, rx, ry, rz]
    
    # For continuous trajectory: calculate total progress through all keypoints in stage
    num_keypoints = len(stage_keypoints)
    
    # Check if we've reached the final keypoint of the stage
    at_final_keypoint = (keypoint_idx >= num_keypoints - 1)
    
    if at_final_keypoint:
        # At or past the final keypoint of this stage
        final_pose, final_gripper, final_pos_noise, final_rot_noise = stage_keypoints[-1]
        
        # Apply noise to final target using per-keypoint noise levels
        final_pos_noise_enabled = np.any(final_pos_noise > 0) if isinstance(final_pos_noise, np.ndarray) else final_pos_noise > 0
        if final_pos_noise_enabled or final_rot_noise > 0:
            cache_key = (robot_state['stage'], num_keypoints - 1)
            if randomize_per_episode:
                if cache_key not in episode_noise_cache:
                    episode_noise_cache[cache_key] = add_noise_to_pose(
                        final_pose, final_pos_noise, final_rot_noise)
                noisy_final_pose = episode_noise_cache[cache_key]
            else:
                noisy_final_pose = add_noise_to_pose(final_pose, final_pos_noise, final_rot_noise)
        else:
            noisy_final_pose = final_pose
        
        # Check if we've reached the final position
        pos_distance = np.linalg.norm(current_ee_pose[:3] - noisy_final_pose[:3])
        rot_distance = np.linalg.norm(current_ee_pose[3:6] - noisy_final_pose[3:6])
        position_threshold = 0.025  # 2.5cm (relaxed from 1.5cm)
        rotation_threshold = 0.20   # ~11.5 degrees (relaxed from 8.6 degrees)
        
        at_final_position = (pos_distance < position_threshold and rot_distance < rotation_threshold)
        
        # Debug output when at final keypoint
        if robot_state['moving_to_keypoint']:
            print(f"🔍 At final keypoint of stage {robot_state['stage']}: pos_dist={pos_distance*100:.1f}cm, rot_dist={rot_distance:.3f}rad, at_pos={at_final_position}")
        
        if at_final_position and robot_state['moving_to_keypoint']:
            # Just reached final keypoint of this stage
            print(f"✅ Reached final keypoint of stage {robot_state['stage']}")
            robot_state['moving_to_keypoint'] = False
            
            # Advance to next stage or mark completion
            if robot_state['stage'] < 1:
                # Don't advance to Stage 1 immediately - close gripper and wait 3 seconds first
                robot_state['gripper_closing'] = True
                robot_state['gripper_close_time'] = time.time()
                print(f"✊ Closing gripper and waiting 3 seconds before Stage 1...")
            else:
                # Reached end of final stage (stage 1)
                if not robot_state['reached_final_keypoint']:
                    robot_state['reached_final_keypoint'] = True
                    robot_state['final_keypoint_time'] = time.time()
                    print(f"🏁 Reached final keypoint! Will stop recording in 1 second...")
        
        # Target the final keypoint
        interpolated_pose = noisy_final_pose
        interpolated_gripper = final_gripper
        
        if not robot_state['moving_to_keypoint'] and not robot_state['gripper_closing']:
            # Start moving toward final keypoint (only if not already in gripper closing phase)
            robot_state['moving_to_keypoint'] = True
            robot_state['movement_start_time'] = time.time()
            robot_state['movement_duration'] = max(0.1, pos_distance / movement_speed) if movement_speed > 0 else 1.0
            print(f"🎯 Starting movement to final keypoint of Stage {robot_state['stage']} (dist: {pos_distance*100:.1f}cm)")
            
    else:
        # Continuous interpolation through multiple waypoints
        # Calculate which segment we're on and progress within that segment
        
        if not robot_state['moving_to_keypoint']:
            # Initialize continuous movement through all keypoints
            robot_state['moving_to_keypoint'] = True
            robot_state['movement_start_time'] = time.time()
            
            # Calculate total path length through all waypoints
            total_distance = 0.0
            for i in range(num_keypoints - 1):
                kp1_pose, _, _, _ = stage_keypoints[i]
                kp2_pose, _, _, _ = stage_keypoints[i + 1]
                segment_dist = np.linalg.norm(kp2_pose[:3] - kp1_pose[:3])
                total_distance += segment_dist
            
            robot_state['movement_duration'] = total_distance / movement_speed if movement_speed > 0 else 1.0
        
        # Calculate progress through the entire stage
        elapsed_time = time.time() - robot_state['movement_start_time']
        overall_progress = min(1.0, elapsed_time / robot_state['movement_duration'])
        
        # Calculate cumulative distances to determine which segment we're in
        cumulative_distances = [0.0]
        for i in range(num_keypoints - 1):
            kp1_pose, _, kp1_pos_noise, kp1_rot_noise = stage_keypoints[i]
            kp2_pose, _, kp2_pos_noise, kp2_rot_noise = stage_keypoints[i + 1]
            # Apply noise to both keypoints using per-keypoint noise levels
            cache_key1 = (robot_state['stage'], i)
            cache_key2 = (robot_state['stage'], i + 1)
            
            kp1_pos_noise_enabled = np.any(kp1_pos_noise > 0) if isinstance(kp1_pos_noise, np.ndarray) else kp1_pos_noise > 0
            kp2_pos_noise_enabled = np.any(kp2_pos_noise > 0) if isinstance(kp2_pos_noise, np.ndarray) else kp2_pos_noise > 0
            if kp1_pos_noise_enabled or kp1_rot_noise > 0 or kp2_pos_noise_enabled or kp2_rot_noise > 0:
                if randomize_per_episode:
                    if cache_key1 not in episode_noise_cache:
                        episode_noise_cache[cache_key1] = add_noise_to_pose(kp1_pose, kp1_pos_noise, kp1_rot_noise)
                    if cache_key2 not in episode_noise_cache:
                        episode_noise_cache[cache_key2] = add_noise_to_pose(kp2_pose, kp2_pos_noise, kp2_rot_noise)
                    noisy_kp1 = episode_noise_cache[cache_key1]
                    noisy_kp2 = episode_noise_cache[cache_key2]
                else:
                    # For per-iteration randomization, use the same noise for both endpoints during this iteration
                    if i == 0:
                        episode_noise_cache[cache_key1] = add_noise_to_pose(kp1_pose, kp1_pos_noise, kp1_rot_noise)
                    noisy_kp1 = episode_noise_cache.get(cache_key1, kp1_pose)
                    noisy_kp2 = add_noise_to_pose(kp2_pose, kp2_pos_noise, kp2_rot_noise)
                    episode_noise_cache[cache_key2] = noisy_kp2
            else:
                noisy_kp1 = kp1_pose
                noisy_kp2 = kp2_pose
            
            segment_dist = np.linalg.norm(noisy_kp2[:3] - noisy_kp1[:3])
            cumulative_distances.append(cumulative_distances[-1] + segment_dist)
        
        total_distance = cumulative_distances[-1]
        if total_distance == 0:
            total_distance = 1.0  # Avoid division by zero
        
        # Find which segment we're in
        target_distance = overall_progress * total_distance
        segment_idx = 0
        for i in range(len(cumulative_distances) - 1):
            if target_distance <= cumulative_distances[i + 1]:
                segment_idx = i
                break
        
        # Calculate progress within current segment
        segment_start_dist = cumulative_distances[segment_idx]
        segment_end_dist = cumulative_distances[segment_idx + 1]
        segment_length = segment_end_dist - segment_start_dist
        
        if segment_length > 0:
            segment_progress = (target_distance - segment_start_dist) / segment_length
        else:
            segment_progress = 0.0
        
        # Get the two keypoints for this segment
        kp1_pose, kp1_gripper, _, _ = stage_keypoints[segment_idx]
        kp2_pose, kp2_gripper, _, _ = stage_keypoints[segment_idx + 1]
        
        # Get noisy versions
        cache_key1 = (robot_state['stage'], segment_idx)
        cache_key2 = (robot_state['stage'], segment_idx + 1)
        noisy_kp1 = episode_noise_cache.get(cache_key1, kp1_pose)
        noisy_kp2 = episode_noise_cache.get(cache_key2, kp2_pose)
        
        # Interpolate position and rotation within this segment
        interpolated_pose = interpolate_poses(noisy_kp1, noisy_kp2, segment_progress)
        interpolated_gripper = (1 - segment_progress) * kp1_gripper + segment_progress * kp2_gripper
        
        # Update keypoint_idx for visualization and completion detection
        robot_state['keypoint_idx'] = segment_idx
        
        # Check if we've completed the entire trajectory
        if overall_progress >= 1.0:
            # Set keypoint_idx to trigger final keypoint detection on next iteration
            robot_state['keypoint_idx'] = num_keypoints
            # Keep moving_to_keypoint = True so the final position check works
            print(f"🎯 Completed continuous trajectory through Stage {robot_state['stage']} (progress: {overall_progress:.2f})")
    
    # Override gripper control
    # During/after gripper closing (transition from Stage 0 to 1), force gripper closed
    if robot_state['gripper_closing']:
        elapsed_close = time.time() - robot_state['gripper_close_time']
        if elapsed_close < 3.0:
            # Still closing/waiting - force gripper to close (0.0)
            interpolated_gripper = 0.0
            if elapsed_close < 0.5:
                print(f"✊ Closing gripper... ({elapsed_close:.1f}s)")
            elif int(elapsed_close) != int(elapsed_close - 0.1):  # Print once per second
                print(f"⏳ Waiting... ({elapsed_close:.1f}s / 3.0s)")
        else:
            # 3 seconds elapsed - advance to Stage 1
            if robot_state['stage'] == 0:  # Only advance if still in stage 0
                robot_state['stage'] = 1
                robot_state['keypoint_idx'] = 0
                robot_state['moving_to_keypoint'] = False  # Reset to start Stage 1 trajectory
                print(f"⏭️  3-second grasp complete! Auto-advancing to Stage 1 (Move)")
            # Keep gripper closed
            interpolated_gripper = 0.0
    
    # In stage 1 (Move), keep gripper closed
    if robot_state['stage'] == 1:
        interpolated_gripper = 0.0
    
    # Create EE pose action [x, y, z, rx, ry, rz, gripper]
    actions = np.array([
        interpolated_pose[0], interpolated_pose[1], interpolated_pose[2],
        interpolated_pose[3], interpolated_pose[4], interpolated_pose[5],
        interpolated_gripper
    ])
    
    return actions, noisy_target_pose

@click.command()
@click.option('--keypoints_file', '-k', required=True, help='JSON file with recorded keypoints')
@click.option('--output_dir', '-o', required=True, help='Directory to save recording')
@click.option('--robot_ip', '-ri', default="192.168.1.143", help="Franka's IP address")
@click.option('--init_joints', '-j', is_flag=True, default=True, help="Whether to initialize robot joint configuration")
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving command to executing on Robot in Sec")
@click.option('--save_viz_interval', default=30, type=int, help="Save visualization every N frames")
@click.option('--randomize_per_episode', is_flag=True, default=True, help="Use same randomization for entire episode")
@click.option('--interpolation_steps', default=50, type=int, help="Number of steps to interpolate between keypoints")
@click.option('--auto_advance', is_flag=True, default=False, help="Automatically advance through keypoints")
@click.option('--movement_speed', default=0.05, type=float, help="Speed for moving between keypoints (m/s)")
def main(keypoints_file, output_dir, robot_ip, init_joints, vis_camera_idx, frequency, 
         command_latency, save_viz_interval, randomize_per_episode,
         interpolation_steps, auto_advance, movement_speed):
    
    # Load keypoints
    print(f"📂 Loading keypoints from: {keypoints_file}")
    keypoints, metadata = load_keypoints(keypoints_file)
    
    # Verify keypoints
    total_keypoints = sum(len(keypoints[i]) for i in [0, 1])
    if total_keypoints == 0:
        print("❌ No keypoints found in file!")
        return
    
    print(f"✅ Loaded {total_keypoints} keypoints:")
    stage_names = metadata.get('stage_names', ['Approach', 'Move'])
    for stage_id in [0, 1]:
        print(f"   Stage {stage_id} ({stage_names[stage_id]}): {len(keypoints[stage_id])} keypoints")
    print(f"💡 Gripper will close automatically after Stage 0 (Approach)")
    
    print(f"🎲 Using per-keypoint noise levels from JSON file")
    print(f"🔁 Randomize per episode: {randomize_per_episode}")
    print(f"🚶 Movement speed: {movement_speed} m/s (constant speed between waypoints)")
    
    dt = 1/frequency
    os.makedirs(output_dir, exist_ok=True)
    
    # Start terminal input thread
    input_thread = threading.Thread(target=terminal_input_thread, daemon=True)
    input_thread.start()
    
    # Episode-level noise cache
    episode_noise_cache = {}
    
    try:
        with SharedMemoryManager() as shm_manager:
            with KeystrokeCounter() as key_counter, \
                RealEnvFranka(
                    output_dir=output_dir,
                    robot_ip=robot_ip,
                    frequency=frequency,
                    n_obs_steps=2,
                    obs_float32=False,
                    init_joints=None,  # init_joints,
                    ctrl_mode='eef',
                    enable_multi_cam_vis=True,
                    record_raw_video=True,
                    thread_per_video=3,
                    video_crf=21,
                    shm_manager=shm_manager) as env:
                
                print('🤖 Robot ready! Manually guide robot through keypoints or use auto mode...')
                robot_state['running'] = True
                
                # Get first keypoint as home position
                first_stage_keypoints = keypoints[0]
                if len(first_stage_keypoints) > 0:
                    home_pose, home_gripper, _, _ = first_stage_keypoints[0]
                    home_pose_full = np.append(home_pose, home_gripper)
                    print(f"🏠 Home position set to first keypoint: pos=[{home_pose[0]:.3f}, {home_pose[1]:.3f}, {home_pose[2]:.3f}], gripper={home_gripper:.3f}")
                else:
                    home_pose_full = None
                    print("⚠️  No keypoints found, no home position set")
                
                time.sleep(1.0)
                t_start = time.monotonic()
                iter_idx = 0
                last_status_time = time.time()
                
                # Tracking for interpolation
                current_keypoint_target = None
                interpolation_progress = 0.0
                last_episode_id = -1
                
                while not robot_state['stop']:
                    # Calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt
                    
                    # Get observations
                    obs = env.get_obs()
                    
                    # Process commands
                    process_commands(key_counter, env, output_dir)
                    
                    # Update state
                    robot_state['episode_id'] = env.episode_id
                    
                    # Handle homing request
                    if robot_state['home_requested']:
                        if home_pose_full is not None and not robot_state['homing']:
                            # Generate new randomized noise for ALL keypoints (both stages)
                            print('🎲 Generating new randomized keypoints...')
                            episode_noise_cache.clear()
                            for stage_id in [0, 1]:
                                stage_kps = keypoints[stage_id]
                                for i in range(len(stage_kps)):
                                    kp_pose, _, kp_pos_noise, kp_rot_noise = stage_kps[i]
                                    kp_pos_noise_enabled = np.any(kp_pos_noise > 0) if isinstance(kp_pos_noise, np.ndarray) else kp_pos_noise > 0
                                    if kp_pos_noise_enabled or kp_rot_noise > 0:
                                        cache_key = (stage_id, i)
                                        episode_noise_cache[cache_key] = add_noise_to_pose(
                                            kp_pose, kp_pos_noise, kp_rot_noise)
                            
                            # Start homing motion to RANDOMIZED first keypoint
                            current_pose_full = obs['ee_pose'][-1].copy()
                            
                            # Get randomized first keypoint from cache
                            first_kp_pose, first_kp_gripper, _, _ = keypoints[0][0]
                            cache_key = (0, 0)
                            if cache_key in episode_noise_cache:
                                noisy_first_pose = episode_noise_cache[cache_key]
                            else:
                                noisy_first_pose = first_kp_pose
                            
                            noisy_home_pose_full = np.append(noisy_first_pose, first_kp_gripper)
                            
                            # Generate smooth trajectory to randomized home
                            robot_state['homing_trajectory'] = generate_smooth_trajectory(
                                current_pose_full, noisy_home_pose_full, num_steps=30)
                            robot_state['homing_step'] = 0
                            robot_state['homing'] = True
                            print(f'🏠 Homing to randomized first keypoint...')
                            print(f'   Target: [{noisy_first_pose[0]:.3f}, {noisy_first_pose[1]:.3f}, {noisy_first_pose[2]:.3f}]')
                            print(f'   Generated {len(robot_state["homing_trajectory"])} waypoints')
                        
                        robot_state['home_requested'] = False
                        robot_state['stage'] = 0
                        robot_state['keypoint_idx'] = 0
                        robot_state['moving_to_keypoint'] = False
                    
                    # Execute homing trajectory
                    if robot_state['homing']:
                        if robot_state['homing_step'] < len(robot_state['homing_trajectory']):
                            actions = robot_state['homing_trajectory'][robot_state['homing_step']]
                            
                            # Validate action shape
                            if actions.shape != (7,):
                                print(f"⚠️  WARNING: Invalid action shape {actions.shape} at step {robot_state['homing_step']}")
                                print(f"   Actions: {actions}")
                                robot_state['homing'] = False
                                robot_state['homing_trajectory'] = None
                                continue
                            
                            robot_state['homing_step'] += 1
                            
                            # Execute homing action
                            env.exec_actions(
                                actions=[actions],
                                timestamps=[t_command_target-time.monotonic()+time.time()],
                                mode='eef')
                            
                            precise_wait(t_cycle_end)
                            iter_idx += 1
                            continue  # Skip normal execution
                        else:
                            # Homing complete
                            robot_state['homing'] = False
                            robot_state['homing_trajectory'] = None
                            robot_state['homing_step'] = 0
                            print('✅ Homing motion complete - ready to record!')
                    
                    # Auto-reset when starting new episode
                    if robot_state['recording']:
                        # Check if we've reached final keypoint and should auto-stop
                        if robot_state['reached_final_keypoint']:
                            elapsed = time.time() - robot_state['final_keypoint_time']
                            if elapsed >= 2.0:
                                # Stop recording after 2 second
                                env.end_episode(curr_outdir=output_dir, incr_epi=True)
                                key_counter.clear()
                                robot_state['recording'] = False
                                robot_state['reached_final_keypoint'] = False
                                print('⏹️  Auto-stopped recording after reaching final keypoint!')
                        
                        # Check if episode changed (new episode started)
                        if env.episode_id != last_episode_id:
                            last_episode_id = env.episode_id
                            # episode_noise_cache.clear()
                            robot_state['stage'] = 0
                            robot_state['keypoint_idx'] = 0
                            robot_state['moving_to_keypoint'] = False
                            robot_state['gripper_closing'] = False
                            robot_state['gripper_close_time'] = 0.0
                            robot_state['reached_final_keypoint'] = False
                            robot_state['final_keypoint_time'] = 0.0
                            
                            # Check if we're already at the first keypoint (e.g., after homing)
                            # If so, immediately target the next keypoint
                            first_stage_kps = keypoints[0]
                            if len(first_stage_kps) > 0:
                                first_kp_pose, first_kp_gripper, _, _ = first_stage_kps[0]
                                current_ee = obs['ee_pose'][-1]
                                pos_dist = np.linalg.norm(current_ee[:3] - first_kp_pose[:3])
                                rot_dist = np.linalg.norm(current_ee[3:6] - first_kp_pose[3:6])
                                
                                if pos_dist < 0.015 and rot_dist < 0.15:  # Slightly larger threshold
                                    robot_state['keypoint_idx'] = 1
                                    print(f"✅ Starting from first keypoint, advancing to keypoint 1")
                    
                    # Get current stage keypoints
                    current_stage = robot_state['stage']
                    stage_keypoints = keypoints[current_stage]
                    
                    # Get target keypoint
                    if len(stage_keypoints) > 0:
                        kp_idx = min(robot_state['keypoint_idx'], len(stage_keypoints) - 1)
                        target_pose, target_gripper, target_pos_noise, target_rot_noise = stage_keypoints[kp_idx]
                        
                        # If not recording, just maintain current position
                        if not robot_state['recording']:
                            # Maintain current position - don't move unless recording or homing
                            ee_pose = obs['ee_pose'][-1].copy()
                            actions = np.array([ee_pose[0], ee_pose[1], ee_pose[2], 
                                               ee_pose[3], ee_pose[4], ee_pose[5], ee_pose[6]])
                            noisy_target_pose = ee_pose[:6]
                        else:
                            # Recording: compute target action with movement
                            actions, noisy_target_pose = compute_target_action(
                                obs, target_pose, target_gripper, stage_keypoints, kp_idx,
                                target_pos_noise, target_rot_noise, randomize_per_episode, episode_noise_cache, movement_speed
                            )
                            
                            # Re-fetch stage keypoints in case stage changed during compute_target_action
                            current_stage = robot_state['stage']
                            stage_keypoints = keypoints[current_stage]
                    else:
                        # No keypoints, maintain current position
                        ee_pose = obs['ee_pose'][-1].copy()
                        actions = np.array([ee_pose[0], ee_pose[1], ee_pose[2], 
                                           ee_pose[3], ee_pose[4], ee_pose[5], ee_pose[6]])
                        noisy_target_pose = ee_pose[:6]
                    
                    # Create visualization
                    rs_front = obs['camera_front_color'][-1,:,:,::-1].copy()
                    rs_right = obs['camera_right_color'][-1,:,:,::-1].copy()
                    
                    # Concatenate images
                    vis_img = np.concatenate([rs_front, rs_right], axis=1)
                    vis_img = cv2.resize(vis_img, (960, 360))
                    
                    # Add status text
                    episode_id = robot_state['episode_id']
                    stage = robot_state['stage']
                    stage_name = stage_names[stage] if stage < len(stage_names) else f"Stage {stage}"
                    kp_idx = robot_state['keypoint_idx']
                    num_kps = len(stage_keypoints)
                    
                    text = f'Ep: {episode_id}, Stage: {stage} ({stage_name}), KP: {kp_idx}/{num_kps}'
                    if robot_state['recording']:
                        text += ', Recording!'
                        # Add movement status
                        if robot_state['moving_to_keypoint'] and robot_state['movement_duration'] > 0:
                            progress = min(1.0, (time.time() - robot_state['movement_start_time']) / robot_state['movement_duration'])
                            text += f' [{progress*100:.0f}%]'
                    
                    cv2.putText(
                        vis_img,
                        text,
                        (10, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.7,
                        thickness=2,
                        color=(0, 255, 0) if robot_state['recording'] else (255, 255, 255)
                    )
                    
                    # Add current EE pose
                    ee_pose = obs['ee_pose'][-1]
                    pose_text = f'EE: [{ee_pose[0]:.3f}, {ee_pose[1]:.3f}, {ee_pose[2]:.3f}]'
                    cv2.putText(
                        vis_img,
                        pose_text,
                        (10, 60),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.6,
                        thickness=1,
                        color=(255, 255, 255)
                    )
                    
                    # Add target pose if available
                    if len(stage_keypoints) > 0:
                        target_text = f'Target: [{noisy_target_pose[0]:.3f}, {noisy_target_pose[1]:.3f}, {noisy_target_pose[2]:.3f}]'
                        # Calculate distance to target
                        ee_pose = obs['ee_pose'][-1]
                        distance = np.linalg.norm(ee_pose[:3] - noisy_target_pose[:3])
                        target_text += f' Dist: {distance*100:.1f}cm'
                        cv2.putText(
                            vis_img,
                            target_text,
                            (10, 90),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.6,
                            thickness=1,
                            color=(0, 255, 255)
                        )
                    
                    # Save visualization images periodically
                    # if save_viz_interval > 0:
                    #     save_visualization_images(vis_img, output_dir, iter_idx, save_viz_interval)
                    
                    # Validate action shape before sending
                    if actions.shape != (7,):
                        print(f"⚠️  ERROR: Invalid action shape {actions.shape}")
                        print(f"   Actions: {actions}")
                        print(f"   Recording: {robot_state['recording']}, Stage: {robot_state['stage']}, KP: {robot_state['keypoint_idx']}")
                        # Try to maintain current position instead
                        ee_pose = obs['ee_pose'][-1].copy()
                        actions = np.append(ee_pose[:6], [ee_pose[6]])
                        print(f"   Using current pose instead, new shape: {actions.shape}")
                    
                    # Execute robot actions with EEF mode
                    env.exec_actions(
                        actions=[actions],
                        timestamps=[t_command_target-time.monotonic()+time.time()],
                        mode='eef')
                    
                    precise_wait(t_cycle_end)
                    iter_idx += 1
                    
                    # Print status periodically (every 5 seconds)
                    current_time = time.time()
                    if current_time - last_status_time > 5.0:
                        status = f"📊 Iter: {iter_idx}, Ep: {episode_id}, Stage: {stage} ({stage_name}), KP: {kp_idx}/{num_kps}"
                        status += f", Recording: {'YES' if robot_state['recording'] else 'NO'}"
                        print(status)
                        last_status_time = current_time
                        
    except KeyboardInterrupt:
        print("\n🔴 Interrupted by user")
        robot_state['stop'] = True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        robot_state['stop'] = True
    finally:
        robot_state['running'] = False
        print("🏁 Robot control ended")

if __name__ == '__main__':
    main()
