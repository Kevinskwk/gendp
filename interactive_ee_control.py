#!/usr/bin/env python3
"""
Interactive End-Effector Control Script for Franka Robot

This script provides an interactive command-line interface to send end-effector 
pose commands to the Franka robot using the FrankaInterpolationController.

All poses are specified in the end-effector frame and automatically transformed
to the link8 frame for the robot controller.

Transform from link8 to end-effector:
- Translation: (0, 0, 0.03) meters
- Rotation: (0, 0, -0.7854) radians in RPY

Commands:
- move <x> <y> <z> <rx> <ry> <rz>: Move to absolute pose (position in meters, rotation in radians)
- rel <dx> <dy> <dz> <drx> <dry> <drz>: Move relative to current pose
- pos: Show current end-effector position
- vel <speed>: Set movement velocity (0.01-1.0 m/s)
- home: Move to home position
- quit/exit: Exit the program
- help: Show this help message
"""

import time
import numpy as np
import argparse
import signal
import sys
import os
from multiprocessing.managers import SharedMemoryManager
import scipy.spatial.transform as st

# Add the gendp directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'gendp'))
from gendp.real_world.franka_interpolation_controller_no_gripper import FrankaInterpolationControllerNoGripper


class InteractiveEEController:
    def __init__(self, robot_ip='192.168.1.143', robot_port=4242, initial_velocity=0.01):
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.velocity = initial_velocity  # m/s
        self.shm_manager = None
        self.controller = None
        
        # Home position (adjust as needed for your setup)
        # self.home_joints = np.array([0.0702805, -0.90773028, -0.09513126, -2.67802477, -0.0919309, 1.82060218, 0.16051947])
        self.home_joints = np.array([0.44268226623535156, -0.22802259027957916, -0.04864306002855301, -1.8856139183044434, -0.06176647171378136, 1.6909462213516235, 1.1077513694763184])
        
        # Transform from link8 to end effector
        # Translation: (0, 0, 0.03), Rotation: (0, 0, -0.7854) in RPY
        self.link8_to_ee_translation = np.array([0.0, 0.0, 0.06])
        # self.link8_to_ee_rotation = st.Rotation.from_euler('xyz', [0, 0, -0.7854])
        self.link8_to_ee_rotation = st.Rotation.from_euler('xyz', [0, 0, 1.57]) * st.Rotation.from_euler('xyz', [1.57, 0, 1.57 +0.7854])
        
        # Create transformation matrices for easier computation
        self.T_link8_to_ee = np.eye(4)
        self.T_link8_to_ee[:3, :3] = self.link8_to_ee_rotation.as_matrix()
        self.T_link8_to_ee[:3, 3] = self.link8_to_ee_translation
        
        # Inverse transformation (end effector to link8)
        self.T_ee_to_link8 = np.linalg.inv(self.T_link8_to_ee)
        
        # Set up signal handler for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        print("\nShutting down safely...")
        self.shutdown()
        sys.exit(0)
    
    def _link8_to_ee_transform(self, link8_pose):
        """Transform pose from link8 frame to end effector frame"""
        # Convert pose to homogeneous transformation matrix
        link8_pos = link8_pose[:3]
        link8_rot = st.Rotation.from_euler('xyz', link8_pose[3:])
        
        T_world_to_link8 = np.eye(4)
        T_world_to_link8[:3, :3] = link8_rot.as_matrix()
        T_world_to_link8[:3, 3] = link8_pos
        
        # Apply transformation: T_world_to_ee = T_world_to_link8 * T_link8_to_ee
        T_world_to_ee = T_world_to_link8 @ self.T_link8_to_ee
        
        # Extract pose from transformation matrix
        ee_pos = T_world_to_ee[:3, 3]
        ee_rot = st.Rotation.from_matrix(T_world_to_ee[:3, :3])
        
        # Return as 6DOF pose
        return np.concatenate([ee_pos, ee_rot.as_euler('xyz')])
    
    def _ee_to_link8_transform(self, ee_pose):
        """Transform pose from end effector frame to link8 frame"""
        # Convert pose to homogeneous transformation matrix
        ee_pos = ee_pose[:3]
        ee_rot = st.Rotation.from_euler('xyz', ee_pose[3:])
        
        T_world_to_ee = np.eye(4)
        T_world_to_ee[:3, :3] = ee_rot.as_matrix()
        T_world_to_ee[:3, 3] = ee_pos
        
        # Apply inverse transformation: T_world_to_link8 = T_world_to_ee * T_ee_to_link8
        T_world_to_link8 = T_world_to_ee @ self.T_ee_to_link8
        
        # Extract pose from transformation matrix
        link8_pos = T_world_to_link8[:3, 3]
        link8_rot = st.Rotation.from_matrix(T_world_to_link8[:3, :3])
        
        # Return as 6DOF pose
        return np.concatenate([link8_pos, link8_rot.as_euler('xyz')])
    
    def _test_transform(self, ee_pose):
        """Test transformation round-trip for debugging"""
        print(f"Input EE pose: {ee_pose}")
        
        # Forward transform: EE -> Link8
        link8_pose = self._ee_to_link8_transform(ee_pose)
        print(f"Computed Link8 pose: {link8_pose}")
        
        # Reverse transform: Link8 -> EE
        recovered_ee_pose = self._link8_to_ee_transform(link8_pose)
        print(f"Recovered EE pose: {recovered_ee_pose}")
        
        # Check difference
        diff = np.array(recovered_ee_pose) - np.array(ee_pose)
        print(f"Round-trip error: {diff}")
        
        return link8_pose, recovered_ee_pose
    
    def start(self):
        """Initialize and start the robot controller"""
        print("Starting Franka Interactive EE Controller...")
        print(f"Connecting to robot at {self.robot_ip}:{self.robot_port}")
        
        try:
            # Create shared memory manager
            self.shm_manager = SharedMemoryManager()
            self.shm_manager.start()
            
            # Create controller in EE control mode
            self.controller = FrankaInterpolationControllerNoGripper(
                shm_manager=self.shm_manager,
                robot_ip=self.robot_ip,
                robot_port=self.robot_port,
                frequency=200,
                Kx_scale=1.0,
                Kxd_scale=np.array([2.0, 1.5, 2.0, 1.0, 1.0, 1.0]),
                joints_init=self.home_joints,
                joints_init_duration=3.0,
                verbose=False,
                ctrl_mode='eef'  # End-effector control mode
            )
            
            # Start the controller
            self.controller.start(wait=False)
            self.controller.start_wait()
            time.sleep(1)
            
            if not self.controller.is_ready:
                raise RuntimeError("Failed to initialize robot controller")
            
            print("✓ Robot controller started successfully!")
            print(f"✓ Initial velocity set to {self.velocity} m/s")
            
            # Wait a moment for initialization
            time.sleep(1.0)
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to start controller: {e}")
            self.shutdown()
            return False
    
    def shutdown(self):
        """Clean shutdown of the controller"""
        if self.controller is not None:
            try:
                self.controller.stop(wait=True)
                print("✓ Robot controller stopped")
            except Exception as e:
                print(f"Warning: Error stopping controller: {e}")
            self.controller = None
        
        if self.shm_manager is not None:
            try:
                self.shm_manager.shutdown()
                print("✓ Shared memory manager shutdown")
            except Exception as e:
                print(f"Warning: Error shutting down shared memory: {e}")
            self.shm_manager = None
    
    def get_current_pose(self):
        """Get the current end-effector pose (transformed from link8)"""
        try:
            state = self.controller.get_state()
            ee_pose_data = state['ActualTCPPose']
            
            # Handle both scalar and array cases
            if np.isscalar(ee_pose_data):
                print("Warning: Received scalar pose data, trying to get latest state...")
                # Try to get the latest state with a different approach
                all_state = self.controller.get_all_state()
                if 'ActualTCPPose' in all_state and len(all_state['ActualTCPPose']) > 0:
                    link8_pose = all_state['ActualTCPPose'][-1]
                else:
                    print("Error: No pose data available")
                    return None
            else:
                # If it's an array, get the latest pose
                if hasattr(ee_pose_data, '__len__') and len(ee_pose_data) > 0:
                    link8_pose = ee_pose_data[-1] if ee_pose_data.ndim > 1 else ee_pose_data
                else:
                    link8_pose = ee_pose_data
            
            # Ensure we have a 6D pose
            if hasattr(link8_pose, '__len__') and len(link8_pose) == 6:
                # Transform from link8 to end effector frame
                ee_pose = self._link8_to_ee_transform(link8_pose)
                # ee_pose = link8_pose
                return ee_pose  # Return [x, y, z, rx, ry, rz] in EE frame
            else:
                print(f"Error: Invalid pose shape: {link8_pose}")
                return None
                
        except Exception as e:
            print(f"Error getting current pose: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def move_to_pose(self, target_pose, duration=None):
        """Move to absolute pose (input in end effector frame, converts to link8 frame)"""
        if duration is None:
            # Calculate duration based on velocity and distance
            current_pose = self.get_current_pose()
            if current_pose is not None:
                pos_distance = np.linalg.norm(target_pose[:3] - current_pose[:3])
                rot_distance = np.linalg.norm(target_pose[3:] - current_pose[3:])
                duration = max(pos_distance / self.velocity, rot_distance / (self.velocity * 5), 0.1)
            else:
                duration = 1.0
        
        try:
            # Transform end effector pose to link8 pose
            link8_pose = self._ee_to_link8_transform(target_pose)
            # link8_pose = target_pose
            self.controller.servoL(link8_pose, duration=duration)
            return True
        except Exception as e:
            print(f"Error moving to pose: {e}")
            return False
    
    def move_relative(self, delta_pose):
        """Move relative to current pose"""
        current_pose = self.get_current_pose()
        if current_pose is None:
            print("Error: Could not get current pose for relative movement")
            return False
        
        print(f"Current pose: {current_pose}")
        print(f"Delta pose: {delta_pose}")
        
        try:
            target_pose = current_pose + np.array(delta_pose)
            print(f"Target pose: {target_pose}")
            return self.move_to_pose(target_pose)
        except Exception as e:
            print(f"Error in relative movement calculation: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def move_home(self):
        """Move to home position"""
        try:
            target_time = time.time() + 3.0
            self.controller.schedule_joint_waypoint(self.home_joints, target_time)
            print("Moving to home position...")
            return True
        except Exception as e:
            print(f"Error moving home: {e}")
            return False
    
    def print_help(self):
        """Print help message"""
        help_text = """
Available Commands:
  move <x> <y> <z> <rx> <ry> <rz>  - Move to absolute pose (pos in meters, rot in radians, EE frame)
  rel <dx> <dy> <dz> <drx> <dry> <drz> - Move relative to current pose (EE frame)
  pos                              - Show current end-effector position (EE frame)
  vel <speed>                      - Set movement velocity (0.01-1.0 m/s)
  debug <x> <y> <z> <rx> <ry> <rz> - Test transformation round-trip for debugging
  home                            - Move to home position
  quit, exit                      - Exit the program
  help                            - Show this help message

Examples:
  move 0.5 0.0 0.3 0.0 3.14 0.0   - Move to position with 180° rotation around Y (EE frame)
  rel 0.0 0.0 -0.05 0.0 0.0 0.0   - Move 5cm down (EE frame)
  debug 0.0 0.0 0.0 3.14 0.0 0.0  - Test transformation for pose with 180° X rotation
  vel 0.05                        - Set velocity to 5cm/s

Note: All poses are in the end-effector frame. Transforms to link8 frame are applied automatically.
"""
        print(help_text)
    
    def run_interactive(self):
        """Run the interactive command loop"""
        if not self.start():
            return
        
        print("\n" + "="*60)
        print("Franka Interactive End-Effector Controller")
        print("="*60)
        self.print_help()
        print("Type 'help' for commands or 'quit' to exit.")
        print("="*60)
        
        try:
            while True:
                try:
                    # Get user input
                    user_input = input("\nfranka> ").strip().lower()
                    
                    if not user_input:
                        continue
                    
                    parts = user_input.split()
                    cmd = parts[0]
                    
                    if cmd in ['quit', 'exit', 'q']:
                        print("Exiting...")
                        break
                    
                    elif cmd == 'help' or cmd == 'h':
                        self.print_help()
                    
                    elif cmd == 'pos':
                        pose = self.get_current_pose()
                        if pose is not None:
                            print(f"Current EE pose:")
                            # print(f"  Position: [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}] m")
                            # print(f"  Rotation: [{pose[3]:.3f}, {pose[4]:.3f}, {pose[5]:.3f}] rad")
                            print(pose)
                        else:
                            print("Failed to get current pose")
                    
                    elif cmd == 'vel':
                        if len(parts) != 2:
                            print("Usage: vel <speed>")
                            continue
                        try:
                            new_vel = float(parts[1])
                            if 0.01 <= new_vel <= 1.0:
                                self.velocity = new_vel
                                print(f"Velocity set to {self.velocity} m/s")
                            else:
                                print("Velocity must be between 0.01 and 1.0 m/s")
                        except ValueError:
                            print("Invalid velocity value")
                    
                    elif cmd == 'move':
                        if len(parts) != 7:
                            print("Usage: move <x> <y> <z> <rx> <ry> <rz>")
                            continue
                        try:
                            target_pose = [float(x) for x in parts[1:]]
                            print(f"Moving to pose: {target_pose}")
                            if self.move_to_pose(target_pose):
                                print("✓ Movement command sent")
                            else:
                                print("✗ Failed to send movement command")
                        except ValueError:
                            print("Invalid pose values")
                    
                    elif cmd == 'rel':
                        if len(parts) != 7:
                            print("Usage: rel <dx> <dy> <dz> <drx> <dry> <drz>")
                            continue
                        try:
                            delta_pose = [float(x) for x in parts[1:]]
                            print(f"Moving relative: {delta_pose}")
                            if self.move_relative(delta_pose):
                                print("✓ Relative movement command sent")
                            else:
                                print("✗ Failed to send relative movement command")
                        except ValueError:
                            print("Invalid delta values")
                    
                    elif cmd == 'debug':
                        if len(parts) != 7:
                            print("Usage: debug <x> <y> <z> <rx> <ry> <rz>")
                            continue
                        try:
                            test_pose = [float(x) for x in parts[1:]]
                            print(f"Testing transformation for pose: {test_pose}")
                            self._test_transform(test_pose)
                        except ValueError:
                            print("Invalid pose values")
                    
                    elif cmd == 'home':
                        if self.move_home():
                            print("✓ Moving to home position")
                        else:
                            print("✗ Failed to move home")
                    
                    elif cmd == 'up':
                        delta_pose = [0, 0, 0.1, 0, 0, 0]
                        print(f"Moving relative: {delta_pose}")
                        if self.move_relative(delta_pose):
                            print("✓ Relative movement command sent")
                        else:
                            print("✗ Failed to send relative movement command")
                    
                    else:
                        print(f"Unknown command: {cmd}. Type 'help' for available commands.")
                
                except KeyboardInterrupt:
                    print("\nUse 'quit' to exit properly.")
                except EOFError:
                    print("\nExiting...")
                    break
                except Exception as e:
                    print(f"Error: {e}")
        
        finally:
            self.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--robot-ip', default='192.168.1.143', 
                       help='IP address of the robot controller (default: 192.168.1.143)')
    parser.add_argument('--robot-port', type=int, default=4242,
                       help='Port of the robot controller (default: 4242)')
    parser.add_argument('--velocity', type=float, default=0.05,
                       help='Initial movement velocity in m/s (default: 0.05)')

    args = parser.parse_args()
    
    if not (0.01 <= args.velocity <= 1.0):
        print("Error: Velocity must be between 0.01 and 1.0 m/s")
        return 1
    
    controller = InteractiveEEController(
        robot_ip=args.robot_ip,
        robot_port=args.robot_port,
        initial_velocity=args.velocity
    )
    
    try:
        controller.run_interactive()
        return 0
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
