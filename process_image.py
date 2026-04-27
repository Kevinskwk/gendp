#!/usr/bin/env python3
"""
Script to process an image using the crop_and_resize function.
Crops borders from an image and optionally resizes it.
"""

import cv2
import numpy as np
import argparse
import os


def crop_and_resize(
    image: np.ndarray,
    target_size: tuple[int, int] = None,
    border_fraction: float = 0.15,
) -> np.ndarray:
    """
    Crop a fraction of the image along the borders while keeping its ratio, and optionally resize
    the cropped image if a target size is provided.

    Args:
        image (np.ndarray): Image to modify.
        target_size (Optional[tuple[int, int]]): Tuple (target_width, target_height) to which the
            image will be resized. If None, only cropping occurs.
        border_fraction (float, optional): Fraction of the image dimensions to crop from each border.
            Is clamped to range [0, 0.49]
            Defaults to 0.15.

    Returns:
        np.ndarray: The modified image.
    """
    # clamp border fraction
    border_fraction = min(max(0, border_fraction), 0.49)
    # Calculate border sizes
    border_x = int(image.shape[0] * border_fraction)
    border_y = int(image.shape[1] * border_fraction)

    # Crop image
    modified_image = image[
        border_x : image.shape[0] - border_x, border_y : image.shape[1] - border_y
    ]

    # If a target size is provided, resize the cropped image
    if target_size is not None:
        modified_image = cv2.resize(modified_image, target_size)

    return modified_image


def main():
    parser = argparse.ArgumentParser(
        description="Process an image by cropping borders and optionally resizing"
    )
    parser.add_argument(
        "input_image",
        type=str,
        help="Path to the input image file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to save the processed image (default: input_name_processed.ext)"
    )
    parser.add_argument(
        "-b", "--border-fraction",
        type=float,
        default=0.15,
        help="Fraction of image to crop from each border (default: 0.15)"
    )
    parser.add_argument(
        "-w", "--width",
        type=int,
        default=None,
        help="Target width for resizing (requires --height)"
    )
    parser.add_argument(
        "-H", "--height",
        type=int,
        default=None,
        help="Target height for resizing (requires --width)"
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_image):
        print(f"Error: Input file '{args.input_image}' not found")
        return
    
    # Load the image
    print(f"Loading image: {args.input_image}")
    image = cv2.imread(args.input_image)
    
    if image is None:
        print(f"Error: Could not load image from '{args.input_image}'")
        return
    
    print(f"Original image shape: {image.shape} (height, width, channels)")
    
    # Rotate the image 90 degrees counter-clockwise
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    print(f"Rotated image shape: {image.shape} (height, width, channels)")
    
    # Determine target size
    target_size = None
    if args.width is not None and args.height is not None:
        target_size = (args.width, args.height)
        print(f"Target size: {target_size} (width, height)")
    elif args.width is not None or args.height is not None:
        print("Error: Both --width and --height must be specified together")
        return
    
    # Process the image
    print(f"Cropping with border fraction: {args.border_fraction}")
    processed_image = crop_and_resize(
        image,
        target_size=target_size,
        border_fraction=args.border_fraction
    )
    
    print(f"Processed image shape: {processed_image.shape} (height, width, channels)")
    
    # Determine output path
    if args.output is None:
        base, ext = os.path.splitext(args.input_image)
        if not ext:
            ext = '.jpg'  # Default extension if input has none
        output_path = f"{base}_processed{ext}"
    else:
        output_path = args.output
        # Ensure output has a valid extension
        _, ext = os.path.splitext(output_path)
        if not ext:
            output_path += '.jpg'
    
    # Save the processed image
    success = cv2.imwrite(output_path, processed_image)
    if success:
        print(f"Processed image saved to: {output_path}")
    else:
        print(f"Error: Failed to save image to: {output_path}")
        print(f"Make sure the output path has a valid image extension (e.g., .jpg, .png)")


if __name__ == "__main__":
    main()
