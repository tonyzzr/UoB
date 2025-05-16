import os
import sys
from pathlib import Path
# Add the project root to Python path
project_root = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(project_root))
sys.path.append(str(project_root.parent))

import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from third_party.FeatUp.featup.util import norm, unnorm
from sklearn.decomposition import PCA
import psutil
import gc

def print_memory_usage():
    """Print current memory usage of the program."""
    process = psutil.Process(os.getpid())
    print(f"RAM Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    if torch.cuda.is_available():
        print(f"CUDA Memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
        print(f"CUDA Memory cached: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")

def clear_memory():
    """Clear both CPU and GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def normalize_ultrasound(image):
    """Normalize ultrasound image for deep learning model input."""
    # Convert to float and scale to [0, 1]
    image = image.float() / image.max()
    
    # Convert grayscale to RGB by repeating channels
    if len(image.shape) == 2:
        image = image.unsqueeze(0).repeat(3, 1, 1)
    
    return image

def apply_pca_to_features(features, pca=None):
    """Apply PCA to feature maps for visualization."""
    # Features should be in shape [C, H, W] or [H, W, C]
    if features.shape[0] > features.shape[-1]:
        # If in [C, H, W] format, transpose to [H, W, C]
        features = np.transpose(features, (1, 2, 0))
    
    H, W, C = features.shape
    
    # Reshape to [H*W, C]
    features_reshaped = features.reshape(-1, C)
    
    if pca is None:
        # Apply PCA if not provided
        pca = PCA(n_components=3)
        features_pca = pca.fit_transform(features_reshaped)
    else:
        # Use provided PCA
        features_pca = pca.transform(features_reshaped)
    
    # Reshape back to [H, W, 3]
    features_pca = features_pca.reshape(H, W, 3)
    
    # Normalize each channel independently to [0, 1]
    for i in range(3):
        channel = features_pca[..., i]
        min_val = channel.min()
        max_val = channel.max()
        if max_val > min_val:
            features_pca[..., i] = (channel - min_val) / (max_val - min_val)
        else:
            features_pca[..., i] = 0
    
    return features_pca, pca

def compute_correspondence_matrix(source_feats, query_feats, temperature=0.1):
    """
    Compute correspondence matrix between source and query features.
    Args:
        source_feats: tensor of shape [C, H, W]
        query_feats: tensor of shape [C, H, W]
        temperature: softmax temperature (lower -> sharper)
    Returns:
        correspondence: tensor of shape [H, W, H, W]
    """
    C, H, W = source_feats.shape
    
    # Reshape features to [H*W, C]
    source_feats = source_feats.reshape(C, -1).transpose(0, 1)  # [H*W, C]
    query_feats = query_feats.reshape(C, -1).transpose(0, 1)    # [H*W, C]
    
    # Normalize features for cosine similarity
    source_feats = F.normalize(source_feats, p=2, dim=1)
    query_feats = F.normalize(query_feats, p=2, dim=1)
    
    # Compute similarity matrix
    similarity = torch.mm(source_feats, query_feats.transpose(0, 1))  # [H*W, H*W]
    
    # Apply temperature and softmax
    similarity = similarity / temperature
    correspondence = F.softmax(similarity, dim=1)
    
    # Reshape to [H, W, H, W]
    correspondence = correspondence.reshape(H, W, H, W)
    
    return correspondence

def visualize_sequence_features(frame_sequence, upsampler, view_idx=0, num_frames=5, temperature=0.1, device='cpu'):
    """
    Visualize features and correlations across sequential frames of the same view.
    Args:
        frame_sequence: Sequence of frames from the dataset
        upsampler: FeatUp upsampler model
        view_idx: Index of the view to visualize
        num_frames: Number of sequential frames to visualize
        temperature: Temperature for correspondence computation
        device: Device to run computations on
    """
    # Check if we have enough frames
    available_frames = len(frame_sequence)
    if available_frames < num_frames:
        print(f"Warning: Only {available_frames} frames available, reducing num_frames from {num_frames}")
        num_frames = available_frames
    
    # Check if view_idx is valid
    if view_idx >= len(frame_sequence[0]):
        raise ValueError(f"view_idx {view_idx} is out of bounds. Available views: {len(frame_sequence[0])}")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = plt.GridSpec(3, num_frames)
    
    # Lists to store features for joint PCA
    all_lr_features = []
    all_hr_features = []
    
    # First pass: collect all features
    for frame_idx in range(num_frames):
        # Get frame for the specified view
        frame = frame_sequence[frame_idx][view_idx]
        
        # Convert tensor to numpy array if needed
        if torch.is_tensor(frame):
            frame = frame.cpu().numpy()
        
        # Normalize image to [0, 1]
        frame = (frame - frame.min()) / (frame.max() - frame.min())
        
        # Convert to PIL Image
        pil_img = Image.fromarray((frame * 255).astype(np.uint8))
        
        # Transform image
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            norm
        ])
        
        # Process image
        image_tensor = transform(pil_img.convert("RGB")).unsqueeze(0).to(device)
        
        # Get features
        with torch.no_grad():
            hr_feats = upsampler(image_tensor)
            lr_feats = upsampler.model(image_tensor)
        
        # Store features
        all_lr_features.append(lr_feats[0].cpu().numpy())
        all_hr_features.append(hr_feats[0].cpu().numpy())
        
        # Plot original image
        ax = fig.add_subplot(gs[0, frame_idx])
        ax.imshow(frame, cmap='gray')
        ax.set_title(f'Frame {frame_idx}')
        ax.axis('off')
    
    # Fit joint PCA
    lr_pca = PCA(n_components=3)
    hr_pca = PCA(n_components=3)
    
    # Reshape features for PCA
    lr_features_reshaped = np.concatenate([f.reshape(-1, f.shape[0]) for f in all_lr_features], axis=0)
    hr_features_reshaped = np.concatenate([f.reshape(-1, f.shape[0]) for f in all_hr_features], axis=0)
    
    # Fit PCA
    lr_pca.fit(lr_features_reshaped)
    hr_pca.fit(hr_features_reshaped)
    
    # Second pass: visualize features
    for frame_idx in range(num_frames):
        # Plot low-resolution features
        lr_pca_vis, _ = apply_pca_to_features(all_lr_features[frame_idx], pca=lr_pca)
        ax = fig.add_subplot(gs[1, frame_idx])
        ax.imshow(lr_pca_vis)
        ax.set_title(f'LR Features Frame {frame_idx}')
        ax.axis('off')
        
        # Plot high-resolution features
        hr_pca_vis, _ = apply_pca_to_features(all_hr_features[frame_idx], pca=hr_pca)
        ax = fig.add_subplot(gs[2, frame_idx])
        ax.imshow(hr_pca_vis)
        ax.set_title(f'HR Features Frame {frame_idx}')
        ax.axis('off')
    
    plt.suptitle(f'Feature Visualization Across {num_frames} Sequential Frames (View {view_idx})')
    plt.tight_layout()
    plt.show()
    
    # Only create correspondence visualization if we have at least 2 frames
    if num_frames >= 2:
        # Create new figure for correspondences
        fig = plt.figure(figsize=(20, 12))
        gs = plt.GridSpec(2, num_frames)
        
        # Compute and visualize correspondences between consecutive frames
        for frame_idx in range(num_frames - 1):
            # Get features for current and next frame
            source_feats = torch.from_numpy(all_hr_features[frame_idx]).to(device)
            query_feats = torch.from_numpy(all_hr_features[frame_idx + 1]).to(device)
            
            # Compute correspondence matrix
            corr_matrix = compute_correspondence_matrix(source_feats, query_feats, temperature)
            
            # Get original images
            source_frame = frame_sequence[frame_idx][view_idx]
            query_frame = frame_sequence[frame_idx + 1][view_idx]
            
            if torch.is_tensor(source_frame):
                source_frame = source_frame.cpu().numpy()
            if torch.is_tensor(query_frame):
                query_frame = query_frame.cpu().numpy()
            
            source_frame = (source_frame - source_frame.min()) / (source_frame.max() - source_frame.min())
            query_frame = (query_frame - query_frame.min()) / (query_frame.max() - query_frame.min())
            
            # Plot source and query frames
            ax = fig.add_subplot(gs[0, frame_idx])
            ax.imshow(source_frame, cmap='gray')
            ax.set_title(f'Source Frame {frame_idx}')
            ax.axis('off')
            
            ax = fig.add_subplot(gs[0, frame_idx + 1])
            ax.imshow(query_frame, cmap='gray')
            ax.set_title(f'Query Frame {frame_idx + 1}')
            ax.axis('off')
            
            # Plot correspondence map
            ax = fig.add_subplot(gs[1, frame_idx + 1])
            corr_map = corr_matrix[112, 112].cpu().numpy()  # Center point correspondence
            ax.imshow(query_frame, cmap='gray')
            im = ax.imshow(corr_map, cmap='hot', alpha=0.5)
            ax.set_title(f'Correspondence {frame_idx}→{frame_idx + 1}')
            ax.axis('off')
            
            # Add colorbar
            if frame_idx == num_frames - 2:  # Only add colorbar for last correspondence
                plt.colorbar(im, ax=ax)
        
        plt.suptitle(f'Correspondence Visualization Between Consecutive Frames (View {view_idx})')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize FeatUp upsampler
    print("Loading FeatUp upsampler...")
    use_norm = True
    upsampler = torch.hub.load("mhamilton723/FeatUp", 'dino16', use_norm=use_norm).to(device)
    upsampler.eval()
    
    # Load dataset
    current_dir = Path.cwd()
    mvbv_path = str(current_dir / 'dataset/recording_2022-08-17_trial2-arm/1_mvbv.pkl')
    
    print("Loading dataset...")
    with open(mvbv_path, 'rb') as f:
        mvbvs = pickle.load(f)
    
    # Process LFTX views
    print("Processing LFTX views...")
    lftx_frames = mvbvs['lftx'].view_images  # Get all available frames
    visualize_sequence_features(lftx_frames, upsampler, view_idx=0, num_frames=5, temperature=0.1, device=device)
    
    # Process HFTX views
    print("Processing HFTX views...")
    hftx_frames = mvbvs['hftx'].view_images  # Get all available frames
    visualize_sequence_features(hftx_frames, upsampler, view_idx=0, num_frames=5, temperature=0.1, device=device)
