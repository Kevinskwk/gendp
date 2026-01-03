"""
Visualize tactile depth data from HDF5 files as 3D point clouds.

This script:
1. Reads tactile image data from HDF5 files (similar to vis_data_2d_combined.py)
2. Processes tactile images to compute depth maps using TactileProcessor
3. Warms up depth estimation with reference images before processing actual data
4. Visualizes the depth as 3D point clouds (similar to demo_view3D.py)
"""

import os
import sys
import numpy as np
import cv2
from tqdm import tqdm

# Add project paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gendp.common.data_utils import load_dict_from_hdf5
from gendp.common.tactile_utils import TactileProcessor

sys.path.append("/home/kevin/gendp/gsrobotics")
from utilities.visualization import Visualize3D
from utilities.image_processing import (
    stack_label_above_image,
    apply_cmap,
    color_map_from_txt,
    normalize_array,
    trim_outliers,
)


def create_display_frame(frame_rgb, depth_map, contact_mask, cmap, scale, title):
    """
    Create a display frame with RGB image, contact mask, and depth visualization.
    Similar to UpdateView in demo_view3D.py
    
    Args:
        frame_rgb: RGB frame (H, W, 3)
        depth_map: Depth map (H, W)
        contact_mask: Contact mask (H, W) as boolean or float
        cmap: Colormap for depth visualization
        scale: Scale factor for display
        title: Window title
        
    Returns:
        Combined display frame
    """
    if np.isnan(depth_map).any():
        # If depth map has NaNs, just show the RGB frame
        depth_map = np.nan_to_num(depth_map, nan=0.0)
    
    # Process depth map
    depth_map_trimmed = trim_outliers(depth_map, 1, 99)
    depth_map_normalized = normalize_array(array=depth_map_trimmed, min_divider=10)
    depth_rgb = apply_cmap(data=depth_map_normalized, cmap=cmap)
    
    # Convert contact mask to 8-bit grayscale
    contact_mask = (contact_mask * 255).astype(np.uint8)
    contact_mask_rgb = cv2.cvtColor(contact_mask, cv2.COLOR_GRAY2BGR)
    
    # Apply labels above images
    frame_labeled = stack_label_above_image(frame_rgb, f"{title} - RGB", 30)
    depth_labeled = stack_label_above_image(depth_rgb, "Depth", 30)
    contact_mask_labeled = stack_label_above_image(contact_mask_rgb, "Contact Mask", 30)
    
    # Add spacing between images
    spacing_size = 30
    horizontal_spacer = np.zeros(
        (frame_labeled.shape[0], spacing_size, 3), dtype=np.uint8
    )
    
    # Stack images horizontally
    top_row = np.hstack(
        (
            frame_labeled,
            horizontal_spacer,
            contact_mask_labeled,
            horizontal_spacer,
            depth_labeled,
        )
    )
    
    # Scale the display frame
    display_frame = cv2.resize(
        top_row,
        (
            int(top_row.shape[1] * scale),
            int(top_row.shape[0] * scale),
        ),
        interpolation=cv2.INTER_NEAREST,
    )
    display_frame = display_frame.astype(np.uint8)
    
    return display_frame


def visualize_tactile_pointcloud(
    episode_range,
    data_dir,
    ref_img_left,
    ref_img_right,
    marker_config_left,
    marker_config_right,
    width=320,
    height=240,
    nn_model_path='~/gendp/gsrobotics/models/nnmini.pt',
    use_gpu=True,
    marker_mask_min=0,
    marker_mask_max=70,
    pointcloud_window_scale=3.0,
    save_pointcloud_path=None,
    cmap_txt_path='~/gendp/gsrobotics/colormaps/simple.txt',
    cmap_in_BGR_format=False,
    cv_image_scale=2.0,
):
    """
    Visualize tactile depth data from HDF5 files as 3D point clouds.
    
    Args:
        episode_range: List of episode indices to visualize
        data_dir: Directory containing HDF5 episode files
        ref_img_left: Path to reference image for left sensor
        ref_img_right: Path to reference image for right sensor
        marker_config_left: Marker configuration dict for left sensor
        marker_config_right: Marker configuration dict for right sensor
        width: Image width
        height: Image height
        nn_model_path: Path to neural network model for depth estimation
        use_gpu: Whether to use GPU for depth estimation
        marker_mask_min: Minimum marker mask threshold
        marker_mask_max: Maximum marker mask threshold
        pointcloud_window_scale: Scale factor for point cloud window size
        save_pointcloud_path: Directory to save point cloud files (None to not save)
        cmap_txt_path: Path to colormap text file for depth visualization
        cmap_in_BGR_format: Whether the colormap is in BGR format
        cv_image_scale: Scale factor for cv2 image display windows
    """
    
    print("Initializing TactileProcessors...")
    
    # Initialize tactile processors for left and right sensors
    # These will warm up with the reference images
    processor_left = TactileProcessor(
        width=width,
        height=height,
        nn_model_path=nn_model_path,
        ref_img=ref_img_left,
        marker_config=marker_config_left,
        use_gpu=use_gpu,
        marker_mask_min=marker_mask_min,
        marker_mask_max=marker_mask_max,
    )
    
    processor_right = TactileProcessor(
        width=width,
        height=height,
        nn_model_path=nn_model_path,
        ref_img=ref_img_right,
        marker_config=marker_config_right,
        use_gpu=use_gpu,
        marker_mask_min=marker_mask_min,
        marker_mask_max=marker_mask_max,
    )
    
    print("TactileProcessors initialized and warmed up!")
    
    # Initialize 3D visualizers for left and right sensors
    visualizer_left = Visualize3D(
        pointcloud_size_x=width,
        pointcloud_size_y=height,
        save_path=save_pointcloud_path if save_pointcloud_path else "",
        window_width=int(pointcloud_window_scale * width),
        window_height=int(pointcloud_window_scale * height),
    )
    
    visualizer_right = Visualize3D(
        pointcloud_size_x=width,
        pointcloud_size_y=height,
        save_path=save_pointcloud_path if save_pointcloud_path else "",
        window_width=int(pointcloud_window_scale * width),
        window_height=int(pointcloud_window_scale * height),
    )
    
    print("3D Visualizers initialized!")
    
    # Load colormap for depth visualization
    cmap_path = os.path.expanduser(cmap_txt_path)
    cmap = color_map_from_txt(path=cmap_path, is_bgr=cmap_in_BGR_format)
    
    # Create CV2 windows for RGB visualization
    cv2.namedWindow("Left Sensor", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Right Sensor", cv2.WINDOW_NORMAL)
    
    # Process each episode
    for episode_idx in episode_range:
        print(f'\nProcessing episode {episode_idx}')
        data_path = f'{data_dir}/episode_{episode_idx}.hdf5'
        
        if not os.path.exists(data_path):
            print(f"Episode file not found: {data_path}")
            continue
        
        # Load data from HDF5
        data_dict, _ = load_dict_from_hdf5(data_path)
        
        # Extract tactile images
        if 'observations' not in data_dict or 'tactile' not in data_dict['observations']:
            print(f"No tactile data found in episode {episode_idx}")
            continue
        
        tactile_left = data_dict['observations']['tactile']['tactile_img_left']
        tactile_right = data_dict['observations']['tactile']['tactile_img_right']
        
        print(f"Tactile left shape: {tactile_left.shape}")
        print(f"Tactile right shape: {tactile_right.shape}")
        
        num_frames = len(tactile_left)
        print(f"Total frames: {num_frames}")
        
        # Process each frame
        for frame_idx in tqdm(range(num_frames), desc=f"Episode {episode_idx}"):
            # Get frames (assuming BGR format from HDF5)
            frame_left = tactile_left[frame_idx]
            frame_right = tactile_right[frame_idx]
            
            # Convert to RGB if needed (reconstruction expects RGB)
            if frame_left.shape[-1] == 3:
                frame_left_rgb = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
                frame_right_rgb = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
            else:
                frame_left_rgb = frame_left
                frame_right_rgb = frame_right
            
            # Get marker positions first, then depth maps
            # This follows the same approach as viz_contact_field_with_tactile_images.py
            initial_positions_left, _ = processor_left.get_marker_flow(frame_left_rgb)
            initial_positions_right, _ = processor_right.get_marker_flow(frame_right_rgb)
            
            # Get depth maps with proper marker positions
            depth_map_left, contact_mask_left, grad_x_left, grad_y_left, _ = processor_left.get_depth(
                frame_left_rgb, marker_positions=initial_positions_left
            )
            depth_map_right, contact_mask_right, grad_x_right, grad_y_right, _ = processor_right.get_depth(
                frame_right_rgb, marker_positions=initial_positions_right
            )
            
            # Update 3D visualizations
            visualizer_left.update(
                depth_map=depth_map_left,
                gradient_x=grad_x_left,
                gradient_y=grad_y_left
            )
            
            visualizer_right.update(
                depth_map=depth_map_right,
                gradient_x=grad_x_right,
                gradient_y=grad_y_right
            )
            
            # Create 2D visualization displays (similar to demo_view3D.py)
            display_left = create_display_frame(
                frame_left_rgb, depth_map_left, contact_mask_left, 
                cmap, cv_image_scale, "Left Sensor"
            )
            display_right = create_display_frame(
                frame_right_rgb, depth_map_right, contact_mask_right,
                cmap, cv_image_scale, "Right Sensor"
            )
            
            # Show 2D visualizations
            cv2.imshow("Left Sensor", display_left)
            cv2.imshow("Right Sensor", display_right)
            
            # Check for exit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nExiting early (pressed 'q')...")
                break
            
        print(f"Completed episode {episode_idx}")
    
    # Cleanup
    print("\nVisualization complete. Closing windows...")
    cv2.destroyAllWindows()
    visualizer_left.visualizer.destroy_window()
    visualizer_right.visualizer.destroy_window()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize tactile depth data from HDF5 files as 3D point clouds"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/kevin/gendp/data/scraper_test/",
        help="Directory containing HDF5 episode files"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=[35],
        help="Episode indices to visualize (e.g., --episodes 0 1 2)"
    )
    parser.add_argument(
        "--ref-left",
        type=str,
        default="/home/kevin/gendp/data/ref_imgs/tactile_left_rgb.png",
        help="Path to reference image for left sensor"
    )
    parser.add_argument(
        "--ref-right",
        type=str,
        default="/home/kevin/gendp/data/ref_imgs/tactile_right_rgb_old.png",
        help="Path to reference image for right sensor"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=320,
        help="Image width"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=240,
        help="Image height"
    )
    parser.add_argument(
        "--nn-model",
        type=str,
        default="~/gendp/gsrobotics/models/nnmini.pt",
        help="Path to neural network model for depth estimation"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration"
    )
    parser.add_argument(
        "--marker-mask-min",
        type=int,
        default=0,
        help="Minimum marker mask threshold"
    )
    parser.add_argument(
        "--marker-mask-max",
        type=int,
        default=70,
        help="Maximum marker mask threshold"
    )
    parser.add_argument(
        "--window-scale",
        type=float,
        default=3.0,
        help="Scale factor for point cloud window size"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Directory to save point cloud files (default: don't save)"
    )
    parser.add_argument(
        "--cmap-txt",
        type=str,
        default="~/gendp/gsrobotics/cmap.txt",
        help="Path to colormap text file for depth visualization"
    )
    parser.add_argument(
        "--cmap-bgr",
        action="store_true",
        help="Colormap is in BGR format"
    )
    parser.add_argument(
        "--cv-scale",
        type=float,
        default=2.0,
        help="Scale factor for cv2 image display windows"
    )
    
    args = parser.parse_args()
    
    # Marker configurations for left and right sensors
    # These values match the settings from tactile_utils.py
    marker_config_left = {
        'N': 7,
        'M': 9,
        'fps': 10,
        'x0': 34.5,
        'y0': 37.5,
        'dx': 28.7,
        'dy': 29.6,
    }
    
    marker_config_right = {
        'N': 7,
        'M': 9,
        'fps': 10,
        'x0': 40.6,
        'y0': 46,
        'dx': 28.6,
        'dy': 29.1,
    }
    
    visualize_tactile_pointcloud(
        episode_range=args.episodes,
        data_dir=args.data_dir,
        ref_img_left=args.ref_left,
        ref_img_right=args.ref_right,
        marker_config_left=marker_config_left,
        marker_config_right=marker_config_right,
        width=args.width,
        height=args.height,
        nn_model_path=args.nn_model,
        use_gpu=not args.no_gpu,
        marker_mask_min=args.marker_mask_min,
        marker_mask_max=args.marker_mask_max,
        pointcloud_window_scale=args.window_scale,
        save_pointcloud_path=args.save_path,
        cmap_txt_path=args.cmap_txt,
        cmap_in_BGR_format=args.cmap_bgr,
        cv_image_scale=args.cv_scale,
    )
