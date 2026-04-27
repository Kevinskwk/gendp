#!/usr/bin/env python3
"""
Script to extract frames from a video file.
"""

import cv2
import argparse
import os


def extract_all_frames(video_path: str, output_dir: str = None) -> bool:
    """
    Extract all frames from a video file.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str, optional): Directory to save the extracted frames.
            If None, creates a folder named video_name_frames

    Returns:
        bool: True if successful, False otherwise.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return False
    
    # Determine output directory
    if output_dir is None:
        base, _ = os.path.splitext(video_path)
        output_dir = f"{base}_frames"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get total frame count for progress reporting
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_idx = 0
    frames_saved = 0
    
    print(f"Extracting frames from '{video_path}'")
    print(f"Total frames: {total_frames}")
    print(f"Output directory: {output_dir}")
    
    while True:
        ret, frame = cap.read()
        
        if not ret or frame is None:
            break
        
        # Save frame with zero-padded index
        output_path = os.path.join(output_dir, f"{frame_idx:06d}.jpg")
        success = cv2.imwrite(output_path, frame)
        
        if success:
            frames_saved += 1
            if (frame_idx + 1) % 100 == 0:  # Progress update every 100 frames
                print(f"  Processed {frame_idx + 1}/{total_frames} frames")
        else:
            print(f"Warning: Failed to save frame {frame_idx}")
        
        frame_idx += 1
    
    # Release the video capture object
    cap.release()
    
    print(f"Extraction complete! Saved {frames_saved} frames to '{output_dir}'")
    return frames_saved > 0


def extract_first_frame(video_path: str, output_path: str = None) -> bool:
    """
    Extract the first frame from a video file.

    Args:
        video_path (str): Path to the input video file.
        output_path (str, optional): Path to save the extracted frame.
            If None, saves as video_name_first_frame.jpg

    Returns:
        bool: True if successful, False otherwise.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return False
    
    # Read the first frame
    ret, frame = cap.read()
    
    # Release the video capture object
    cap.release()
    
    if not ret or frame is None:
        print(f"Error: Could not read the first frame from '{video_path}'")
        return False
    
    # Determine output path
    if output_path is None:
        base, _ = os.path.splitext(video_path)
        output_path = f"{base}_first_frame.jpg"
    else:
        # Ensure output has a valid extension
        _, ext = os.path.splitext(output_path)
        if not ext:
            output_path += '.jpg'
    
    # Save the frame
    success = cv2.imwrite(output_path, frame)
    
    if success:
        print(f"First frame extracted successfully!")
        print(f"Frame shape: {frame.shape} (height, width, channels)")
        print(f"Saved to: {output_path}")
        return True
    else:
        print(f"Error: Failed to save frame to '{output_path}'")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Extract frame(s) from a video file"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the input video file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to save the extracted frame or directory for all frames"
    )
    parser.add_argument(
        "-a", "--all",
        action="store_true",
        help="Extract all frames instead of just the first frame"
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' not found")
        return
    
    # Extract frames
    if args.all:
        extract_all_frames(args.video_path, args.output)
    else:
        extract_first_frame(args.video_path, args.output)


if __name__ == "__main__":
    main()
