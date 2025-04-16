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

def compute_and_save_correspondence_matrices(frame_data, upsampler, source_view_idx=0, temperature=0.1, device='cpu', save_dir='correspondence_cache'):
    """
    Compute and save correspondence matrices between source view and all views.
    """
    os.makedirs(save_dir, exist_ok=True)
    n_views = frame_data.shape[0]
    
    print("\nInitial memory state:")
    print_memory_usage()
    
    # Process source view
    source_img = frame_data[source_view_idx]
    if torch.is_tensor(source_img):
        source_img = source_img.cpu().numpy()
    source_img = (source_img - source_img.min()) / (source_img.max() - source_img.min())
    
    # Convert to PIL and process
    source_pil = Image.fromarray((source_img * 255).astype(np.uint8))
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        norm
    ])
    source_tensor = transform(source_pil.convert("RGB")).unsqueeze(0).to(device)
    
    # Get source features
    with torch.no_grad():
        source_hr_feats = upsampler(source_tensor)[0]
        # Move source features to CPU to save GPU memory
        source_hr_feats = source_hr_feats.cpu()
        clear_memory()
    
    print("\nAfter source feature computation:")
    print_memory_usage()
    
    # Process each view and compute correspondences
    for view_idx in range(n_views):
        print(f"\nProcessing view {view_idx}/{n_views}")
        
        # Skip if correspondence file already exists
        save_path = os.path.join(save_dir, f'correspondence_{view_idx}.npy')
        if os.path.exists(save_path):
            print(f"Skipping view {view_idx} - file already exists")
            continue
            
        # Process query image
        query_img = frame_data[view_idx]
        if torch.is_tensor(query_img):
            query_img = query_img.cpu().numpy()
        query_img = (query_img - query_img.min()) / (query_img.max() - query_img.min())
        
        # Convert to PIL and process
        query_pil = Image.fromarray((query_img * 255).astype(np.uint8))
        query_tensor = transform(query_pil.convert("RGB")).unsqueeze(0).to(device)
        
        # Get query features and compute correspondence
        with torch.no_grad():
            # Move source features back to GPU for computation
            source_hr_feats_gpu = source_hr_feats.to(device)
            
            query_hr_feats = upsampler(query_tensor)[0]
            hr_corr = compute_correspondence_matrix(source_hr_feats_gpu, query_hr_feats, temperature)
            
            # Save correspondence matrix to disk and clear from GPU memory
            np.save(save_path, hr_corr.cpu().numpy())
            
            # Clear GPU memory
            del hr_corr, query_hr_feats, source_hr_feats_gpu, query_tensor
            clear_memory()
        
        print_memory_usage()
    
    print("\nFinal memory state:")
    print_memory_usage()

def resize_and_pad_correspondence_map(corr_map, original_size, crop_size=224):
    """
    Resize correspondence map accounting for center crop and add padding.
    
    Args:
        corr_map: The correspondence map from feature space
        original_size: Tuple of (height, width) of original image
        crop_size: Size used for center crop in preprocessing
    
    Returns:
        Padded and properly aligned correspondence map
    """
    h, w = original_size
    
    # Calculate resize dimensions preserving aspect ratio
    if h > w:
        resize_h = int(crop_size * h / w)
        resize_w = crop_size
    else:
        resize_h = crop_size
        resize_w = int(crop_size * w / h)
    
    # Calculate crop offsets (same as in center crop)
    crop_y = (resize_h - crop_size) // 2
    crop_x = (resize_w - crop_size) // 2
    
    # First resize the correspondence map to the crop_size
    corr_map_resized = np.array(Image.fromarray(corr_map).resize((crop_size, crop_size), Image.BILINEAR))
    
    # Create padded version at resize dimensions
    padded_map = np.zeros((resize_h, resize_w))
    
    # Calculate where to place the resized map
    y_start = crop_y
    y_end = y_start + crop_size
    x_start = crop_x
    x_end = x_start + crop_size
    
    # Place the resized map in the correct position
    padded_map[y_start:y_end, x_start:x_end] = corr_map_resized
    
    # Final resize to original image dimensions
    final_map = np.array(Image.fromarray(padded_map).resize((w, h), Image.BILINEAR))
    
    return final_map

def visualize_correspondences(frame_data, upsampler, source_view_idx=0, point_of_interest=None, temperature=0.1, device='cpu', save_dir='correspondence_cache'):
    """
    Visualize correspondence matrices between source view and all views.
    """
    n_views = frame_data.shape[0]
    
    # Check if correspondence matrices need to be computed
    if not os.path.exists(save_dir) or len(os.listdir(save_dir)) < n_views:
        compute_and_save_correspondence_matrices(frame_data, upsampler, source_view_idx, temperature, device, save_dir)
    
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
    
    # Calculate scale factors for point transformation
    # First calculate resize dimensions preserving aspect ratio
    if h > w:
        resize_h = int(224 * h / w)
        resize_w = 224
    else:
        resize_h = 224
        resize_w = int(224 * w / h)
    
    # Calculate crop offsets
    crop_y = (resize_h - 224) // 2
    crop_x = (resize_w - 224) // 2
    
    # Transform point coordinates to feature space (accounting for resize and crop)
    feat_h = int((poi_h * resize_h / h - crop_y))
    feat_w = int((poi_w * resize_w / w - crop_x))
    
    # Clamp to valid feature space coordinates
    feat_h = max(0, min(feat_h, 223))
    feat_w = max(0, min(feat_w, 223))
    
    # Create figure for visualization
    fig = plt.figure(figsize=(20, 8))
    gs = plt.GridSpec(2, n_views)
    
    # Plot source image with green point at POI
    ax_source = fig.add_subplot(gs[0, source_view_idx])
    ax_source.imshow(source_img, cmap='gray')
    ax_source.plot(poi_w, poi_h, 'go', markersize=10)  # Add green point at POI
    ax_source.set_title(f'Source View {source_view_idx}\nPOI: ({poi_y_frac:.2f}, {poi_x_frac:.2f})')
    ax_source.axis('off')
    
    # Process each view and visualize
    for view_idx in range(n_views):
        # Load correspondence matrix from disk
        corr_path = os.path.join(save_dir, f'correspondence_{view_idx}.npy')
        hr_corr = np.load(corr_path, mmap_mode='r')  # Use memory mapping for large files
        
        # Get correspondence map for POI
        corr_map = hr_corr[feat_h, feat_w].copy()  # Make a copy of the specific slice we need
        del hr_corr  # Free memory
        
        # Process query image
        query_img = frame_data[view_idx]
        if torch.is_tensor(query_img):
            query_img = query_img.cpu().numpy()
        query_img = (query_img - query_img.min()) / (query_img.max() - query_img.min())
        
        # Plot query image
        if view_idx != source_view_idx:
            ax = fig.add_subplot(gs[0, view_idx])
            ax.imshow(query_img, cmap='gray')
            ax.set_title(f'Query View {view_idx}')
            ax.axis('off')
        
        # Resize and pad correspondence map properly
        corr_map_resized = resize_and_pad_correspondence_map(corr_map, (h, w))
        
        # Plot query image with overlaid correspondence map
        ax_hr = fig.add_subplot(gs[1, view_idx])
        ax_hr.imshow(query_img, cmap='gray')
        im = ax_hr.imshow(corr_map_resized, cmap='hot', alpha=0.5)
        ax_hr.set_title(f'Correspondence View {view_idx}')
        ax_hr.axis('off')
        
        clear_memory()
    
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
        # None,           # Center point (0.5, 0.5)
        (0.55, 0.55),    # Center point
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
    
    # # Process HFTX views
    # print("Processing HFTX views...")
    # hftx_frame = mvbvs['hftx'].view_images[0]  # Get first frame, all views 
    
    # # Try different points of interest
    # for poi in points_of_interest:
    #     visualize_correspondences(hftx_frame, upsampler, source_view_idx=0, 
    #                             point_of_interest=poi, temperature=1, device=device)
