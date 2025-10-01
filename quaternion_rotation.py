#!/usr/bin/env python3
"""
Python script to apply rotation around global x, y, or z axis to a quaternion.
Returns the new quaternion after applying the rotation.

The script supports:
- Rotation around global X, Y, or Z axis
- Input quaternion in various formats (x,y,z,w or w,x,y,z)
- Angle in degrees or radians
- Command line interface and programmatic usage
"""

import numpy as np
import scipy.spatial.transform as st
import argparse
import sys


def apply_global_rotation(quaternion, axis, angle_deg=None, angle_rad=None, quat_format='xyzw'):
    """
    Apply rotation around global x, y, or z axis to a quaternion.
    
    Args:
        quaternion (array-like): Input quaternion as [x, y, z, w] or [w, x, y, z]
        axis (str): Rotation axis - 'x', 'y', or 'z'
        angle_deg (float): Rotation angle in degrees
        angle_rad (float): Rotation angle in radians
        quat_format (str): Input quaternion format - 'xyzw' or 'wxyz'
    
    Returns:
        np.ndarray: New quaternion after rotation in the same format as input
    """
    # Validate inputs
    if axis.lower() not in ['x', 'y', 'z']:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
    
    if angle_deg is None and angle_rad is None:
        raise ValueError("Must specify either angle_deg or angle_rad")
    
    if angle_deg is not None and angle_rad is not None:
        raise ValueError("Cannot specify both angle_deg and angle_rad")
    
    # Convert angle to radians if needed
    if angle_deg is not None:
        angle = np.deg2rad(angle_deg)
    else:
        angle = angle_rad
    
    # Convert input quaternion to scipy format (x, y, z, w)
    quat = np.array(quaternion)
    if quat_format.lower() == 'wxyz':
        # Convert from [w, x, y, z] to [x, y, z, w]
        quat = np.array([quat[1], quat[2], quat[3], quat[0]])
    elif quat_format.lower() == 'xyzw':
        # Already in correct format
        pass
    else:
        raise ValueError("quat_format must be 'xyzw' or 'wxyz'")
    
    # Create the original rotation from quaternion
    original_rotation = st.Rotation.from_quat(quat)
    
    # Create the additional rotation around the specified global axis
    axis_vector = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}[axis.lower()]
    additional_rotation = st.Rotation.from_rotvec(angle * np.array(axis_vector))
    
    # Apply the additional rotation to the original rotation
    # For global rotation, we multiply the additional rotation first
    new_rotation = additional_rotation * original_rotation
    
    # Convert back to quaternion
    new_quat = new_rotation.as_quat()  # Returns [x, y, z, w]
    
    # Convert back to original format if needed
    if quat_format.lower() == 'wxyz':
        # Convert from [x, y, z, w] to [w, x, y, z]
        new_quat = np.array([new_quat[3], new_quat[0], new_quat[1], new_quat[2]])
    
    return new_quat


def apply_local_rotation(quaternion, axis, angle_deg=None, angle_rad=None, quat_format='xyzw'):
    """
    Apply rotation around local x, y, or z axis to a quaternion.
    
    Args:
        quaternion (array-like): Input quaternion as [x, y, z, w] or [w, x, y, z]
        axis (str): Rotation axis - 'x', 'y', or 'z'
        angle_deg (float): Rotation angle in degrees
        angle_rad (float): Rotation angle in radians
        quat_format (str): Input quaternion format - 'xyzw' or 'wxyz'
    
    Returns:
        np.ndarray: New quaternion after rotation in the same format as input
    """
    # Validate inputs
    if axis.lower() not in ['x', 'y', 'z']:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
    
    if angle_deg is None and angle_rad is None:
        raise ValueError("Must specify either angle_deg or angle_rad")
    
    if angle_deg is not None and angle_rad is not None:
        raise ValueError("Cannot specify both angle_deg and angle_rad")
    
    # Convert angle to radians if needed
    if angle_deg is not None:
        angle = np.deg2rad(angle_deg)
    else:
        angle = angle_rad
    
    # Convert input quaternion to scipy format (x, y, z, w)
    quat = np.array(quaternion)
    if quat_format.lower() == 'wxyz':
        # Convert from [w, x, y, z] to [x, y, z, w]
        quat = np.array([quat[1], quat[2], quat[3], quat[0]])
    elif quat_format.lower() == 'xyzw':
        # Already in correct format
        pass
    else:
        raise ValueError("quat_format must be 'xyzw' or 'wxyz'")
    
    # Create the original rotation from quaternion
    original_rotation = st.Rotation.from_quat(quat)
    
    # Create the additional rotation around the specified local axis
    axis_vector = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}[axis.lower()]
    additional_rotation = st.Rotation.from_rotvec(angle * np.array(axis_vector))
    
    # Apply the additional rotation to the original rotation
    # For local rotation, we multiply the original rotation first
    new_rotation = original_rotation * additional_rotation
    
    # Convert back to quaternion
    new_quat = new_rotation.as_quat()  # Returns [x, y, z, w]
    
    # Convert back to original format if needed
    if quat_format.lower() == 'wxyz':
        # Convert from [x, y, z, w] to [w, x, y, z]
        new_quat = np.array([new_quat[3], new_quat[0], new_quat[1], new_quat[2]])
    
    return new_quat


def print_quaternion_info(quat, label="Quaternion", quat_format='xyzw'):
    """Print quaternion information in a formatted way."""
    if quat_format.lower() == 'xyzw':
        print(f"{label}: [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}] (x, y, z, w)")
    else:
        print(f"{label}: [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}] (w, x, y, z)")
    
    # Convert to scipy format for additional info
    if quat_format.lower() == 'wxyz':
        scipy_quat = np.array([quat[1], quat[2], quat[3], quat[0]])
    else:
        scipy_quat = quat
    
    # Convert to Euler angles for reference
    rotation = st.Rotation.from_quat(scipy_quat)
    euler_xyz = rotation.as_euler('xyz', degrees=True)
    print(f"  Euler XYZ: [{euler_xyz[0]:.2f}°, {euler_xyz[1]:.2f}°, {euler_xyz[2]:.2f}°]")
    
    # Show magnitude
    magnitude = np.linalg.norm(quat)
    print(f"  Magnitude: {magnitude:.6f}")


def main():
    parser = argparse.ArgumentParser(description='Apply rotation around global or local axis to a quaternion')
    parser.add_argument('--quat', nargs=4, type=float, required=True,
                       help='Input quaternion as 4 numbers (default: x y z w format)')
    parser.add_argument('--axis', choices=['x', 'y', 'z', 'X', 'Y', 'Z'], required=True,
                       help='Rotation axis (x, y, or z)')
    parser.add_argument('--angle', type=float, required=True,
                       help='Rotation angle in degrees')
    parser.add_argument('--format', choices=['xyzw', 'wxyz'], default='xyzw',
                       help='Quaternion format (default: xyzw)')
    parser.add_argument('--rotation_type', choices=['global', 'local'], default='global',
                       help='Type of rotation - global (world frame) or local (object frame)')
    parser.add_argument('--radians', action='store_true',
                       help='Interpret angle as radians instead of degrees')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed information')
    
    args = parser.parse_args()
    
    try:
        # Prepare angle arguments
        angle_kwargs = {}
        if args.radians:
            angle_kwargs['angle_rad'] = args.angle
        else:
            angle_kwargs['angle_deg'] = args.angle
        
        # Apply rotation
        if args.rotation_type == 'global':
            new_quat = apply_global_rotation(
                args.quat, 
                args.axis.lower(), 
                quat_format=args.format,
                **angle_kwargs
            )
        else:
            new_quat = apply_local_rotation(
                args.quat, 
                args.axis.lower(), 
                quat_format=args.format,
                **angle_kwargs
            )
        
        if args.verbose:
            print("=" * 60)
            print(f"Applying {args.rotation_type} rotation around {args.axis.upper()} axis")
            angle_unit = "radians" if args.radians else "degrees"
            print(f"Rotation angle: {args.angle} {angle_unit}")
            print("=" * 60)
            print_quaternion_info(args.quat, "Original", args.format)
            print()
            print_quaternion_info(new_quat, "Result", args.format)
            print("=" * 60)
        else:
            # Simple output format
            print(f"Original: {args.quat}")
            print(f"Result:   {new_quat.tolist()}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # Example usage if run directly without arguments
    if len(sys.argv) == 1:
        print("Example usage:")
        print("python quaternion_rotation.py --quat 0 0 0 1 --axis z --angle 90")
        print("python quaternion_rotation.py --quat 0.7071 0 0 0.7071 --axis y --angle 45 --verbose")
        print("python quaternion_rotation.py --quat 1 0 0 0 --axis x --angle 1.57 --radians --format wxyz")
        print("\nFor help: python quaternion_rotation.py --help")
        
        # Run a demo
        print("\n" + "="*50)
        print("DEMO: Rotating identity quaternion 90° around Z axis")
        print("="*50)
        
        identity_quat = [0, 0, 0, 1]  # Identity quaternion in xyzw format
        result = apply_global_rotation(identity_quat, 'z', angle_deg=90)
        
        print_quaternion_info(identity_quat, "Original")
        print()
        print_quaternion_info(result, "After 90° Z rotation")
    else:
        main()
