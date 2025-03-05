import numpy as np
import cv2
# import transforms3d as t3d
from scipy.spatial.transform import Rotation

def combine_image_arrays_to_video_2x3(
    series1,  # Depth image (uint16)
    series2,  # Depth image (uint16)
    series3, 
    series4,
    series5,
    series6,
    output_path="combined_video.mp4", 
    fps=15,
    depth_min=0,
    depth_max=5000  # Default max depth in mm for typical depth cameras
):
    """
    Combines six series of images into a single video arranged in a 2x3 grid layout.
    The first two series are depth images (uint16), while the rest are regular RGB/BGR images.
    
    Parameters:
    -----------
    series1 : numpy.ndarray
        First series of depth images as numpy array of shape [num_frames, height, width] (uint16)
    series2 : numpy.ndarray
        Second series of depth images as numpy array of shape [num_frames, height, width] (uint16)
    series3-6 : numpy.ndarray
        Series of images as numpy array of shape [num_frames, height, width, channels]
    output_path : str
        Path where the output video will be saved
    fps : int
        Frames per second for the output video
    depth_min : int
        Minimum depth value for normalization (default: 0)
    depth_max : int
        Maximum depth value for normalization (default: 5000 mm)
        
    Returns:
    --------
    bool
        True if video was successfully created, False otherwise
    """
    # Check if all arrays have the same number of frames
    frame_counts = [
        series1.shape[0], series2.shape[0], series3.shape[0], 
        series4.shape[0], series5.shape[0], series6.shape[0]
    ]
    if len(set(frame_counts)) > 1:
        print(f"Warning: Series have different numbers of frames: {frame_counts}. Using the minimum.")
    
    # Find the minimum number of frames
    min_frames = min(frame_counts)
    
    # Check and print shapes
    # print(f"Series1 shape: {series1.shape}, dtype: {series1.dtype}")
    # print(f"Series2 shape: {series2.shape}, dtype: {series2.dtype}")
    # print(f"Series3 shape: {series3.shape}, dtype: {series3.dtype}")
    
    # Get dimensions of each series
    h1, w1 = series1.shape[1:3]
    h2, w2 = series2.shape[1:3]
    h3, w3 = series3.shape[1:3]
    h4, w4 = series4.shape[1:3]
    h5, w5 = series5.shape[1:3]
    h6, w6 = series6.shape[1:3]
    
    # Determine the dimensions for the grid - set uniform size for all cells
    grid_cell_width = max(w1, w2, w3, w4, w5, w6)
    grid_cell_height = max(h1, h2, h3, h4, h5, h6)
    
    # Final video dimensions (2x3 grid)
    video_width = grid_cell_width * 3
    video_height = grid_cell_height * 2
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (video_width, video_height))
    
    # Process each frame
    for i in range(min_frames):
        # Get the depth images and convert them to colorized visualization
        depth1 = series1[i].astype(np.float32)
        depth2 = series2[i].astype(np.float32)
        
        # Normalize depth data to 0-255 range for visualization
        depth1_normalized = np.clip((depth1 - depth_min) / (depth_max - depth_min) * 255, 0, 255).astype(np.uint8)
        depth2_normalized = np.clip((depth2 - depth_min) / (depth_max - depth_min) * 255, 0, 255).astype(np.uint8)
        
        # Apply colormap to depth images for better visualization (COLORMAP_JET is common for depth)
        depth1_colored = cv2.applyColorMap(depth1_normalized, cv2.COLORMAP_JET)
        depth2_colored = cv2.applyColorMap(depth2_normalized, cv2.COLORMAP_JET)
        
        # Get the regular images
        if series3.shape[3] == 3:  # If it has 3 channels (assume RGB)
            img3 = series3[i][..., ::-1].copy()  # Convert RGB to BGR
            img4 = series4[i][..., ::-1].copy()
            img5 = series5[i][..., ::-1].copy()
            img6 = series6[i][..., ::-1].copy()
        else:
            img3 = series3[i].copy()
            img4 = series4[i].copy()
            img5 = series5[i].copy()
            img6 = series6[i].copy()
        
        # Convert all to uint8 for OpenCV
        img3 = img3.astype(np.uint8)
        img4 = img4.astype(np.uint8)
        img5 = img5.astype(np.uint8)
        img6 = img6.astype(np.uint8)
        
        # Resize all images to have the same dimensions
        depth1_colored = cv2.resize(depth1_colored, (grid_cell_width, grid_cell_height))
        depth2_colored = cv2.resize(depth2_colored, (grid_cell_width, grid_cell_height))
        img3 = cv2.resize(img3, (grid_cell_width, grid_cell_height))
        img4 = cv2.resize(img4, (grid_cell_width, grid_cell_height))
        img5 = cv2.resize(img5, (grid_cell_width, grid_cell_height))
        img6 = cv2.resize(img6, (grid_cell_width, grid_cell_height))
        
        # Create the top row and bottom row
        top_row = np.hstack((depth1_colored, img3, img5))
        bottom_row = np.hstack((depth2_colored, img4, img6))
        
        # Combine top and bottom rows
        combined = np.vstack((top_row, bottom_row))
        
        # Write the combined frame to the video
        video_writer.write(combined)
    
    # Release the video writer
    video_writer.release()
    print(f"Video successfully created at {output_path}")
    return True

def get_extrinsic(pos, quat):
    extrinsic_matrix = np.eye(4)
    rot = Rotation.from_quat(quat)
    # Ensure translation is a numpy array
    t = np.array(pos, dtype=np.float64).reshape(3, 1)
    
    # Create the 3x4 extrinsic matrix [R|t]
    extrinsic_matrix[:3, :3] = rot.as_matrix()
    extrinsic_matrix[:3, 3:] = t

    extrinsic_matrix = np.linalg.inv(extrinsic_matrix)
    
    return extrinsic_matrix