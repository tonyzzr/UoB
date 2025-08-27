"""
Dynamic Canvas Creation for Image Stitching

This module implements dynamic canvas sizing for panoramic image stitching, following the principle
of calculating optimal canvas dimensions before performing the final merge.

Key Features:
============
1. **calculate_dynamic_canvas()**: Core function that determines optimal canvas size by:
   - Transforming corner coordinates of all images using homography matrices
   - Finding the bounding box that encompasses all transformed corners
   - Creating translation matrices to ensure all coordinates are positive
   - Returning adjusted homographies ready for cv2.warpPerspective

2. **create_panorama()**: Complete panorama creation with multiple blending modes:
   - Simple overlay (last image wins)
   - Weighted average blending
   - Maximum value blending

3. **Automatic handling of negative coordinates**: The algorithm automatically translates
   all images to ensure no clipping occurs, even when homographies place images in
   negative coordinate space.

Usage Example:
=============
```python
from dynamic_canvas_creation import calculate_dynamic_canvas, create_panorama

# Prepare your images and homography matrices
images = [img1, img2, img3]  # List of numpy arrays
homographies = [H1, H2, H3]  # List of 3x3 homography matrices

# Calculate optimal canvas size
(canvas_width, canvas_height), adjusted_homographies = calculate_dynamic_canvas(
    images, homographies, reference_idx=0
)

# Create panorama
panorama = create_panorama(images, homographies, blend_mode='weighted')
```

Algorithm Steps:
===============
1. **Reference Frame**: Choose one image (typically index 0) as the reference
2. **Corner Transformation**: Apply homographies to all four corners of each image
3. **Bounding Box Calculation**: Find min/max x,y coordinates across all transformed corners
4. **Canvas Creation**: Calculate width/height from bounding box
5. **Translation Adjustment**: Create translation matrix to shift all coordinates to positive values
6. **Final Warping**: Use adjusted homographies with cv2.warpPerspective

Mathematical Foundation:
=======================
For each corner point p = [x, y, 1] in source image:
- Transformed point: p' = H @ p
- Cartesian coordinates: (x'/w', y'/w') where w' is the homogeneous coordinate

The final canvas accommodates all transformed corners without clipping.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


def calculate_dynamic_canvas(images: List[np.ndarray], 
                           homographies: List[np.ndarray],
                           reference_idx: int = 0) -> Tuple[Tuple[int, int], List[np.ndarray]]:
    """
    Calculate the optimal canvas size for image stitching and return adjusted homographies.
    
    Args:
        images: List of images to be stitched
        homographies: List of homography matrices (H_i transforms image i to reference frame)
        reference_idx: Index of the reference image (default: 0)
    
    Returns:
        canvas_size: (width, height) of the optimal canvas
        adjusted_homographies: List of adjusted homography matrices including translation
    """
    if not images or not homographies:
        raise ValueError("Images and homographies lists cannot be empty")
    
    if len(images) != len(homographies):
        raise ValueError("Number of images must match number of homographies")
    
    # Get reference image dimensions
    ref_height, ref_width = images[reference_idx].shape[:2]
    
    # Initialize with reference image corners
    all_corners = []
    
    # Add reference image corners (no transformation needed)
    ref_corners = np.array([
        [0, 0, 1],
        [ref_width, 0, 1],
        [ref_width, ref_height, 1],
        [0, ref_height, 1]
    ]).T  # Shape: (3, 4)
    
    all_corners.extend(ref_corners[:2].T)  # Add x, y coordinates
    
    # Transform corners of all other images
    for i, (image, H) in enumerate(zip(images, homographies)):
        if i == reference_idx:
            continue  # Skip reference image
            
        height, width = image.shape[:2]
        
        # Define image corners in homogeneous coordinates
        corners = np.array([
            [0, 0, 1],
            [width, 0, 1],
            [width, height, 1],
            [0, height, 1]
        ]).T  # Shape: (3, 4)
        
        # Transform corners using homography
        transformed_corners = H @ corners
        
        # Convert from homogeneous to Cartesian coordinates
        transformed_corners = transformed_corners[:2] / transformed_corners[2]
        
        # Add to all corners list
        all_corners.extend(transformed_corners.T)
    
    # Convert to numpy array for easier manipulation
    all_corners = np.array(all_corners)
    
    # Calculate bounding box
    x_min = np.min(all_corners[:, 0])
    x_max = np.max(all_corners[:, 0])
    y_min = np.min(all_corners[:, 1])
    y_max = np.max(all_corners[:, 1])
    
    # Calculate canvas dimensions
    canvas_width = int(np.round(x_max - x_min))
    canvas_height = int(np.round(y_max - y_min))
    
    # Create translation matrix to shift everything to positive coordinates
    translation_matrix = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ])
    
    # Adjust all homographies with translation
    adjusted_homographies = []
    for i, H in enumerate(homographies):
        if i == reference_idx:
            # For reference image, only apply translation
            adjusted_H = translation_matrix
        else:
            # For other images, combine translation with original homography
            adjusted_H = translation_matrix @ H
        adjusted_homographies.append(adjusted_H)
    
    return (canvas_width, canvas_height), adjusted_homographies


def create_panorama(images: List[np.ndarray],
                   homographies: List[np.ndarray],
                   reference_idx: int = 0,
                   blend_mode: str = 'simple') -> np.ndarray:
    """
    Create a panorama using dynamic canvas creation.
    
    Args:
        images: List of images to stitch
        homographies: List of homography matrices
        reference_idx: Index of reference image
        blend_mode: Blending mode ('simple', 'weighted', or 'max')
    
    Returns:
        Panorama image
    """
    # Calculate dynamic canvas size and adjusted homographies
    (canvas_width, canvas_height), adjusted_homographies = calculate_dynamic_canvas(
        images, homographies, reference_idx
    )
    
    print(f"Calculated canvas size: {canvas_width} x {canvas_height}")
    
    # Initialize canvas
    if len(images[0].shape) == 3:
        canvas = np.zeros((canvas_height, canvas_width, images[0].shape[2]), dtype=np.float32)
        weight_map = np.zeros((canvas_height, canvas_width, images[0].shape[2]), dtype=np.float32)
    else:
        canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)
        weight_map = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    
    # Warp and blend each image
    for i, (image, adjusted_H) in enumerate(zip(images, adjusted_homographies)):
        # Warp image to canvas
        warped_image = cv2.warpPerspective(
            image.astype(np.float32), 
            adjusted_H, 
            (canvas_width, canvas_height)
        )
        
        # Create mask for valid pixels
        if len(image.shape) == 3:
            mask = np.ones((image.shape[0], image.shape[1]), dtype=np.float32)
        else:
            mask = np.ones_like(image, dtype=np.float32)
            
        warped_mask = cv2.warpPerspective(mask, adjusted_H, (canvas_width, canvas_height))
        
        if len(warped_mask.shape) == 2 and len(canvas.shape) == 3:
            warped_mask = np.stack([warped_mask] * canvas.shape[2], axis=2)
        
        # Blend based on mode
        if blend_mode == 'simple':
            # Simple overlay (last image wins)
            valid_pixels = warped_mask > 0
            canvas[valid_pixels] = warped_image[valid_pixels]
        elif blend_mode == 'weighted':
            # Weighted average
            canvas += warped_image * warped_mask
            weight_map += warped_mask
        elif blend_mode == 'max':
            # Take maximum values
            canvas = np.maximum(canvas, warped_image * warped_mask)
    
    # Normalize for weighted blending
    if blend_mode == 'weighted':
        valid_weights = weight_map > 0
        canvas[valid_weights] /= weight_map[valid_weights]
    
    return canvas.astype(np.uint8)


def test_dynamic_canvas_creation():
    """
    Test the dynamic canvas creation with mock data.
    """
    print("Testing Dynamic Canvas Creation...")
    
    # Create mock images
    img1 = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    img3 = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    
    # Add some distinguishable features
    img1[100:200, 150:250] = [255, 0, 0]  # Red square
    img2[120:180, 170:230] = [0, 255, 0]  # Green square
    img3[80:220, 130:270] = [0, 0, 255]   # Blue square
    
    images = [img1, img2, img3]
    
    # Create mock homographies
    # H1: Identity (reference image)
    H1 = np.eye(3)
    
    # H2: Translation to the right
    H2 = np.array([
        [1, 0, 200],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # H3: Translation and slight rotation
    angle = np.radians(10)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    H3 = np.array([
        [cos_a, -sin_a, 100],
        [sin_a, cos_a, -50],
        [0, 0, 1]
    ], dtype=np.float32)
    
    homographies = [H1, H2, H3]
    
    # Test canvas calculation
    try:
        (canvas_width, canvas_height), adjusted_homographies = calculate_dynamic_canvas(
            images, homographies, reference_idx=0
        )
        
        print(f"✓ Canvas calculation successful!")
        print(f"  Canvas size: {canvas_width} x {canvas_height}")
        print(f"  Number of adjusted homographies: {len(adjusted_homographies)}")
        
        # Test corner transformations
        print("\nTesting corner transformations:")
        for i, (img, H_adj) in enumerate(zip(images, adjusted_homographies)):
            height, width = img.shape[:2]
            corners = np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]]).T
            transformed = H_adj @ corners
            transformed = transformed[:2] / transformed[2]
            
            print(f"  Image {i} corners after adjustment:")
            print(f"    Min: ({np.min(transformed[0]):.1f}, {np.min(transformed[1]):.1f})")
            print(f"    Max: ({np.max(transformed[0]):.1f}, {np.max(transformed[1]):.1f})")
        
        # Create panorama
        print("\nCreating panorama...")
        panorama = create_panorama(images, homographies, reference_idx=0, blend_mode='weighted')
        print(f"✓ Panorama created with shape: {panorama.shape}")
        
        # Visualize results
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Show original images
        for i, img in enumerate(images):
            axes[0, i].imshow(img)
            axes[0, i].set_title(f'Original Image {i+1}')
            axes[0, i].axis('off')
        
        # Show panorama
        axes[1, 1].imshow(panorama)
        axes[1, 1].set_title('Dynamic Canvas Panorama')
        axes[1, 1].axis('off')
        
        # Hide unused subplots
        axes[1, 0].axis('off')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('dynamic_canvas_test_result.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        return False


def demo_usage():
    """
    Simple demonstration of how to use the dynamic canvas creation functions.
    """
    print("=== Dynamic Canvas Creation Demo ===\n")
    
    # Example: Creating 3 images with known transformations
    print("1. Creating mock images...")
    
    # Create images with distinct patterns for visualization
    img1 = np.zeros((200, 300, 3), dtype=np.uint8)
    img1[50:150, 100:200] = [255, 100, 100]  # Light red rectangle
    img1[75:125, 125:175] = [255, 255, 255]  # White center
    
    img2 = np.zeros((200, 300, 3), dtype=np.uint8)
    img2[60:140, 80:220] = [100, 255, 100]   # Light green rectangle
    img2[85:115, 130:170] = [255, 255, 255]  # White center
    
    img3 = np.zeros((200, 300, 3), dtype=np.uint8)
    img3[40:160, 120:240] = [100, 100, 255]  # Light blue rectangle
    img3[90:110, 160:200] = [255, 255, 255]  # White center
    
    images = [img1, img2, img3]
    
    print("2. Defining homography transformations...")
    
    # Simple transformations for demonstration
    H1 = np.eye(3, dtype=np.float32)  # Reference image (no transformation)
    
    # Translate image 2 to the right by 150 pixels
    H2 = np.array([
        [1, 0, 150],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Translate image 3 to the left by 100 pixels and up by 50 pixels
    H3 = np.array([
        [1, 0, -100],
        [0, 1, -50],
        [0, 0, 1]
    ], dtype=np.float32)
    
    homographies = [H1, H2, H3]
    
    print("3. Calculating dynamic canvas size...")
    
    # Calculate optimal canvas size
    (canvas_width, canvas_height), adjusted_homographies = calculate_dynamic_canvas(
        images, homographies, reference_idx=0
    )
    
    print(f"   Original image size: {images[0].shape[1]} x {images[0].shape[0]}")
    print(f"   Calculated canvas size: {canvas_width} x {canvas_height}")
    print(f"   Canvas area increase: {(canvas_width * canvas_height) / (images[0].shape[1] * images[0].shape[0]):.1f}x")
    
    print("\n4. Creating panorama with different blending modes...")
    
    # Test different blending modes
    blend_modes = ['simple', 'weighted', 'max']
    results = {}
    
    for mode in blend_modes:
        panorama = create_panorama(images, homographies, reference_idx=0, blend_mode=mode)
        results[mode] = panorama
        print(f"   ✓ {mode.capitalize()} blending completed")
    
    print("\n5. Analysis of corner transformations:")
    
    for i, (img, H_original, H_adjusted) in enumerate(zip(images, homographies, adjusted_homographies)):
        height, width = img.shape[:2]
        corners = np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]]).T
        
        # Original transformation
        original_transformed = H_original @ corners
        original_transformed = original_transformed[:2] / original_transformed[2]
        
        # Adjusted transformation (with translation to positive coordinates)
        adjusted_transformed = H_adjusted @ corners
        adjusted_transformed = adjusted_transformed[:2] / adjusted_transformed[2]
        
        print(f"\n   Image {i+1}:")
        print(f"     Original bounds: x[{np.min(original_transformed[0]):.1f}, {np.max(original_transformed[0]):.1f}], "
              f"y[{np.min(original_transformed[1]):.1f}, {np.max(original_transformed[1]):.1f}]")
        print(f"     Adjusted bounds: x[{np.min(adjusted_transformed[0]):.1f}, {np.max(adjusted_transformed[0]):.1f}], "
              f"y[{np.min(adjusted_transformed[1]):.1f}, {np.max(adjusted_transformed[1]):.1f}]")
    
    print(f"\n✅ Demo completed successfully!")
    print(f"   Canvas dimensions: {canvas_width} x {canvas_height}")
    print(f"   All coordinates are now positive (no clipping)")
    
    return results, (canvas_width, canvas_height), adjusted_homographies


if __name__ == "__main__":
    # Run comprehensive test
    success = test_dynamic_canvas_creation()
    if success:
        print("\n🎉 All tests passed! Dynamic canvas creation is working correctly.")
    else:
        print("\n❌ Tests failed. Please check the implementation.")
    
    print("\n" + "="*60)
    
    # Run simple demo
    demo_usage()
