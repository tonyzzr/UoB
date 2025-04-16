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
from pathlib import Path
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torchvision import transforms
from PIL import Image
from third_party.FeatUp.featup.util import norm, unnorm
from third_party.FeatUp.featup.plotting import plot_feats

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

def fit_joint_pca(features_list):
    """Fit PCA on concatenated features from all views."""
    # Convert all features to [H*W, C] format
    reshaped_features = []
    for features in features_list:
        if features.shape[0] > features.shape[-1]:
            features = np.transpose(features, (1, 2, 0))
        H, W, C = features.shape
        reshaped_features.append(features.reshape(-1, C))
    
    # Concatenate all features
    all_features = np.concatenate(reshaped_features, axis=0)
    
    # Fit PCA
    pca = PCA(n_components=3)
    pca.fit(all_features)
    
    return pca

def visualize_features(frame_data, upsampler, device='cpu', use_joint_pca=True):
    """Apply FeatUp features and visualize results for all views."""
    n_views = frame_data.shape[0]
    
    # Lists to store features for joint PCA
    all_lr_features = []
    all_hr_features = []
    
    # First pass: collect all features
    for view_idx in range(n_views):
        # Get original image
        orig_img = frame_data[view_idx]
        
        # Convert tensor to numpy array if needed
        if torch.is_tensor(orig_img):
            orig_img = orig_img.cpu().numpy()
        
        # Normalize image to [0, 1]
        orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min())
        
        # Convert to PIL Image
        pil_img = Image.fromarray((orig_img * 255).astype(np.uint8))
        
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
    
    # Fit joint PCA if requested
    if use_joint_pca:
        lr_pca = fit_joint_pca(all_lr_features)
        hr_pca = fit_joint_pca(all_hr_features)
    else:
        lr_pca = None
        hr_pca = None
    
    # Create figure
    fig, axes = plt.subplots(3, n_views, figsize=(20, 8))
    
    # Second pass: visualize with joint PCA
    for view_idx in range(n_views):
        # Get original image
        orig_img = frame_data[view_idx]
        if torch.is_tensor(orig_img):
            orig_img = orig_img.cpu().numpy()
        orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min())
        
        # Plot original image
        axes[0, view_idx].imshow(orig_img, cmap='gray')
        axes[0, view_idx].set_title(f'Original View {view_idx}')
        axes[0, view_idx].axis('off')
        
        # Plot low-resolution features
        lr_pca_vis, _ = apply_pca_to_features(all_lr_features[view_idx], pca=lr_pca)
        axes[1, view_idx].imshow(lr_pca_vis)
        axes[1, view_idx].set_title(f'Low-Res Features View {view_idx}')
        axes[1, view_idx].axis('off')
        
        # Plot high-resolution features
        hr_pca_vis, _ = apply_pca_to_features(all_hr_features[view_idx], pca=hr_pca)
        axes[2, view_idx].imshow(hr_pca_vis)
        axes[2, view_idx].set_title(f'High-Res Features View {view_idx}')
        axes[2, view_idx].axis('off')
    
    plt.suptitle('Joint PCA across views' if use_joint_pca else 'Independent PCA per view')
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
    
    # Process LFTX views with joint PCA
    print("Processing LFTX views...")
    lftx_frame = mvbvs['lftx'].view_images[0]  # Get first frame, all views
    visualize_features(lftx_frame, upsampler, device, use_joint_pca=True)
    
    # Process HFTX views with joint PCA
    print("Processing HFTX views...")
    hftx_frame = mvbvs['hftx'].view_images[0]  # Get first frame, all views
    visualize_features(hftx_frame, upsampler, device, use_joint_pca=True)
