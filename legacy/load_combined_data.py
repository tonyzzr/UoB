import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import sys

# Add the project root to Python path
project_root = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(project_root))
sys.path.append(str(project_root.parent))

def load_and_visualize_combined_data(combined_path: str, start_frame: int = 0, num_frames: int = 3):
    """
    Load and visualize frames from the combined dataset pickle file.
    Shows all 8 views for both LFTX and HFTX for multiple frames.
    
    Args:
        combined_path: Path to the combined pickle file
        start_frame: Starting frame index to visualize (default: 0)
        num_frames: Number of consecutive frames to visualize (default: 3)
    """
    # Load the combined data
    with open(combined_path, 'rb') as f:
        combined_data = pickle.load(f)
    
    # Create a figure with 2*num_frames rows and 8 columns
    fig, axes = plt.subplots(2*num_frames, 8, figsize=(20, 5*num_frames))
    
    # Plot frames for both LFTX and HFTX
    for frame_offset in range(num_frames):
        frame_idx = start_frame + frame_offset
        
        # Plot LFTX data - all 8 views
        for view_idx in range(8):
            lftx_frame = combined_data['lftx']['view_images'][frame_idx, view_idx, :, :]
            im = axes[2*frame_offset, view_idx].imshow(lftx_frame, cmap='gray')
            axes[2*frame_offset, view_idx].set_title(f'LFTX Frame {frame_idx} View {view_idx}')
            axes[2*frame_offset, view_idx].axis('off')
            plt.colorbar(im, ax=axes[2*frame_offset, view_idx])
        
        # Plot HFTX data - all 8 views
        for view_idx in range(8):
            hftx_frame = combined_data['hftx']['view_images'][frame_idx, view_idx, :, :]
            im = axes[2*frame_offset + 1, view_idx].imshow(hftx_frame, cmap='gray')
            axes[2*frame_offset + 1, view_idx].set_title(f'HFTX Frame {frame_idx} View {view_idx}')
            axes[2*frame_offset + 1, view_idx].axis('off')
            plt.colorbar(im, ax=axes[2*frame_offset + 1, view_idx])
    
    plt.suptitle(f'Frames {start_frame} to {start_frame + num_frames - 1}')
    plt.tight_layout()
    plt.show()

def print_dataset_info(combined_path: str):
    """
    Print information about the combined dataset.
    
    Args:
        combined_path: Path to the combined pickle file
    """
    with open(combined_path, 'rb') as f:
        combined_data = pickle.load(f)
        
        print("\nCombined Dataset Information:")
        print("\nLFTX Information:")
        print(f"Number of frames: {combined_data['lftx']['view_images'].shape[0]}")
        print(f"Number of views: {combined_data['lftx']['view_images'].shape[1]}")
        print(f"Frame shape: {combined_data['lftx']['view_images'].shape[2:]}")
        
        print("\nHFTX Information:")
        print(f"Number of frames: {combined_data['hftx']['view_images'].shape[0]}")
        print(f"Number of views: {combined_data['hftx']['view_images'].shape[1]}")
        print(f"Frame shape: {combined_data['hftx']['view_images'].shape[2:]}")

if __name__ == '__main__':
    # Example usage
    current_dir = Path.cwd()
    combined_path = str(current_dir / 'dataset/recording_2022-08-17_trial2-arm/combined_data.pkl')
    
    # Print dataset information
    print_dataset_info(combined_path)
    
    # Load and visualize frames 0-2
    load_and_visualize_combined_data(combined_path, start_frame=0, num_frames=3) 