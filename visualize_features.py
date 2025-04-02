import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torchvision import transforms
from featup.featurizers.dinov2 import DINOv2Featurizer

import sys
sys.path.append('/Users/zhuoruizhang/Desktop/')

def normalize_ultrasound(image):
    """Normalize ultrasound image for deep learning model input."""
    # Convert to float and scale to [0, 1]
    image = image.float() / image.max()
    
    # Apply standard normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
    
    # Convert grayscale to RGB by repeating channels
    if len(image.shape) == 2:
        image = image.unsqueeze(0).repeat(3, 1, 1)
    
    return normalize(image)

def apply_pca_to_features(features, n_components=3):
    """Apply PCA to feature maps and return first n_components."""
    # Reshape features to 2D array (pixels x channels)
    feat_shape = features.shape
    features_2d = features.reshape(-1, feat_shape[0])
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features_2d)
    
    # Reshape back to spatial dimensions
    features_pca = features_pca.reshape(feat_shape[1], feat_shape[2], n_components)
    
    # Normalize to [0, 1] for visualization
    features_pca = (features_pca - features_pca.min()) / (features_pca.max() - features_pca.min())
    
    return features_pca

def get_intermediate_features(model, x, layer_name='blocks.11'):
    """Extract intermediate features from the model."""
    features = {}
    
    def hook_fn(module, input, output):
        features['feat'] = output
    
    # Register hook
    for name, module in model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(hook_fn)
            break
    
    # Forward pass
    with torch.no_grad():
        model(x)
    
    # Remove hook
    handle.remove()
    
    return features['feat']

def visualize_pca_features(frame_data, model, device='cpu'):
    """Apply DINOv2 features and visualize PCA results for all views."""
    n_views = frame_data.shape[0]
    
    # Create figure
    fig, axes = plt.subplots(2, n_views, figsize=(20, 5))
    
    # Process each view
    for view_idx in range(n_views):
        # Get original image
        orig_img = frame_data[view_idx]
        
        # Normalize image
        norm_img = normalize_ultrasound(torch.tensor(orig_img))
        
        # Add batch dimension and move to device
        input_tensor = norm_img.unsqueeze(0).to(device)
        
        # Get intermediate features
        features = get_intermediate_features(model, input_tensor)
        
        # Convert features to numpy and apply PCA
        features_np = features[0].permute(1, 2, 0).cpu().numpy()
        pca_features = apply_pca_to_features(features_np)
        
        # Plot original image
        axes[0, view_idx].imshow(orig_img, cmap='gray')
        axes[0, view_idx].set_title(f'Original View {view_idx}')
        axes[0, view_idx].axis('off')
        
        # Plot PCA features
        axes[1, view_idx].imshow(pca_features)
        axes[1, view_idx].set_title(f'PCA Features View {view_idx}')
        axes[1, view_idx].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize DINOv2 model
    print("Loading DINOv2 model...")
    model = dinov2_vits14(pretrained=True)
    model = model.to(device)
    model.eval()
    
    # Load dataset
    current_dir = Path.cwd()
    mvbv_path = str(current_dir / 'dataset/recording_2022-08-17_trial2-arm/1_mvbv.pkl')
    
    print("Loading dataset...")
    with open(mvbv_path, 'rb') as f:
        mvbvs = pickle.load(f)
    
    # Process LFTX views
    print("Processing LFTX views...")
    lftx_frame = mvbvs['lftx'].view_images[0]  # Get first frame, all views
    visualize_pca_features(lftx_frame, model, device)
    
    # Process HFTX views
    print("Processing HFTX views...")
    hftx_frame = mvbvs['hftx'].view_images[0]  # Get first frame, all views
    visualize_pca_features(hftx_frame, model, device)
