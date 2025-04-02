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

def normalize_ultrasound(image):
    """Normalize ultrasound image for deep learning model input."""
    # Convert to float and scale to [0, 1]
    image = image.float() / image.max()
    
    # Convert grayscale to RGB by repeating channels
    if len(image.shape) == 2:
        image = image.unsqueeze(0).repeat(3, 1, 1)
    
    return image

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

def visualize_correspondences(frame_data, upsampler, source_view_idx=0, point_of_interest=None, temperature=0.1, device='cpu'):
    """
    Visualize correspondence matrices between source view and all views.
    
    Args:
        frame_data: Input frames of shape [N, H, W]
        upsampler: FeatUp upsampler model
        source_view_idx: Index of the source view
        point_of_interest: Tuple of (y_frac, x_frac) normalized coordinates in source image [0,1]. 
                          If None, uses center point (0.5, 0.5)
        temperature: Temperature for softmax normalization
        device: Device to run computations on
    """
    n_views = frame_data.shape[0]
    
    # Process source view
    source_img = frame_data[source_view_idx]
    if torch.is_tensor(source_img):
        source_img = source_img.cpu().numpy()
    source_img = (source_img - source_img.min()) / (source_img.max() - source_img.min())
    
    # Get image dimensions
    h, w = source_img.shape
    
    # Set point of interest
    if point_of_interest is None:
        poi_y_frac, poi_x_frac = 0.5, 0.5  # Center point
    else:
        poi_y_frac, poi_x_frac = point_of_interest
        # Validate normalized coordinates
        if not (0 <= poi_y_frac <= 1 and 0 <= poi_x_frac <= 1):
            raise ValueError(f"Point of interest {(poi_y_frac, poi_x_frac)} should be in range [0,1]")
    
    # Convert normalized coordinates to pixel coordinates
    poi_h = int(poi_y_frac * h)
    poi_w = int(poi_x_frac * w)
    
    # Convert to PIL and process
    source_pil = Image.fromarray((source_img * 255).astype(np.uint8))
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        norm
    ])
    source_tensor = transform(source_pil.convert("RGB")).unsqueeze(0).to(device)
    
    # Calculate scale factors for point transformation
    scale_h = 224 / h
    scale_w = 224 / w
    
    # Transform point coordinates to feature space
    feat_h = int(poi_h * scale_h)
    feat_w = int(poi_w * scale_w)
    
    # Get source features
    with torch.no_grad():
        source_hr_feats = upsampler(source_tensor)[0]
    
    # Create figure for visualization
    fig = plt.figure(figsize=(20, 8))
    gs = plt.GridSpec(2, n_views)
    
    # Plot source image with green point at POI
    ax_source = fig.add_subplot(gs[0, source_view_idx])
    ax_source.imshow(source_img, cmap='gray')
    ax_source.plot(poi_w, poi_h, 'go', markersize=10)  # Add green point at POI
    ax_source.set_title(f'Source View {source_view_idx}\nPOI: ({poi_y_frac:.2f}, {poi_x_frac:.2f})')
    ax_source.axis('off')
    
    # Process each view and compute correspondences
    for view_idx in range(n_views):
        # Process query image
        query_img = frame_data[view_idx]
        if torch.is_tensor(query_img):
            query_img = query_img.cpu().numpy()
        query_img = (query_img - query_img.min()) / (query_img.max() - query_img.min())
        
        # Convert to PIL and process
        query_pil = Image.fromarray((query_img * 255).astype(np.uint8))
        query_tensor = transform(query_pil.convert("RGB")).unsqueeze(0).to(device)
        
        # Get query features
        with torch.no_grad():
            query_hr_feats = upsampler(query_tensor)[0]
        
        # Compute correspondence matrix
        hr_corr = compute_correspondence_matrix(source_hr_feats, query_hr_feats, temperature)
        
        # Plot query image
        if view_idx != source_view_idx:
            ax = fig.add_subplot(gs[0, view_idx])
            ax.imshow(query_img, cmap='gray')
            ax.set_title(f'Query View {view_idx}')
            ax.axis('off')
        
        # Plot high-res correspondence map for POI
        ax_hr = fig.add_subplot(gs[1, view_idx])
        corr_map = hr_corr[feat_h, feat_w].cpu().numpy()
        im = ax_hr.imshow(corr_map, cmap='hot')
        ax_hr.set_title(f'Correspondence View {view_idx}')
        ax_hr.axis('off')
    
    plt.suptitle(f'Correspondence Maps (temp={temperature})')
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
    
    # Define points of interest to visualize (normalized coordinates)
    points_of_interest = [
        None,           # Center point (0.5, 0.5)
        (0.3, 0.6),    # Upper middle region
        (0.7, 0.4),    # Lower middle region
    ]
    
    # Process LFTX views
    print("Processing LFTX views...")
    lftx_frame = mvbvs['lftx'].view_images[0]  # Get first frame, all views
    
    # Try different points of interest
    for poi in points_of_interest:
        visualize_correspondences(lftx_frame, upsampler, source_view_idx=0, 
                                point_of_interest=poi, temperature=0.1, device=device)
    
    # Process HFTX views
    print("Processing HFTX views...")
    hftx_frame = mvbvs['hftx'].view_images[0]  # Get first frame, all views 
    
    # Try different points of interest
    for poi in points_of_interest:
        visualize_correspondences(hftx_frame, upsampler, source_view_idx=0, 
                                point_of_interest=poi, temperature=0.1, device=device)
