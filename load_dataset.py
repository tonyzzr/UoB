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

def load_and_visualize_frame(mvbv_path: str, frame_idx: int = 0):
    """
    Load and visualize a specific frame from a MultiViewBmodeVideo pickle file.
    Shows all 8 views for both LFTX and HFTX.
    
    Args:
        mvbv_path: Path to the pickle file containing the MultiViewBmodeVideo dictionary
        frame_idx: Index of the frame to visualize (default: 0)
    """
    # Load the MultiViewBmodeVideo dictionary
    with open(mvbv_path, 'rb') as f:
        mvbvs = pickle.load(f)
    
    # Create a figure with 2x8 subplots (LFTX on top row, HFTX on bottom row)
    fig, axes = plt.subplots(2, 8, figsize=(20, 5))
    
    # Plot LFTX data - all 8 views
    for view_idx in range(8):
        lftx_frame = mvbvs['lftx'].view_images[frame_idx, view_idx, :, :]
        im = axes[0, view_idx].imshow(lftx_frame, cmap='gray')
        axes[0, view_idx].set_title(f'LFTX View {view_idx}')
        axes[0, view_idx].axis('off')
        plt.colorbar(im, ax=axes[0, view_idx])
    
    # Plot HFTX data - all 8 views
    for view_idx in range(8):
        hftx_frame = mvbvs['hftx'].view_images[frame_idx, view_idx, :, :]
        im = axes[1, view_idx].imshow(hftx_frame, cmap='gray')
        axes[1, view_idx].set_title(f'HFTX View {view_idx}')
        axes[1, view_idx].axis('off')
        plt.colorbar(im, ax=axes[1, view_idx])
    
    plt.suptitle(f'Frame {frame_idx}')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Example usage
    current_dir = Path.cwd()
    mvbv_path = str(current_dir / 'dataset/recording_2022-08-17_trial2-arm/1_mvbv.pkl')
    
    # Load and visualize the first frame (index 0)
    load_and_visualize_frame(mvbv_path, frame_idx=0)
    
    # Print some information about the dataset
    with open(mvbv_path, 'rb') as f:
        mvbvs = pickle.load(f)
        print("\nDataset Information:")
        print(f"Number of frames in LFTX: {mvbvs['lftx'].view_images.shape[0]}")
        print(f"Number of views in LFTX: {mvbvs['lftx'].view_images.shape[1]}")
        print(f"Frame shape LFTX: {mvbvs['lftx'].view_images.shape[2:]}")
        print(f"\nNumber of frames in HFTX: {mvbvs['hftx'].view_images.shape[0]}")
        print(f"Number of views in HFTX: {mvbvs['hftx'].view_images.shape[1]}")
        print(f"Frame shape HFTX: {mvbvs['hftx'].view_images.shape[2:]}")
