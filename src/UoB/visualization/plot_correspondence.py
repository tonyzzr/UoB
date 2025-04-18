"""Functions for visualizing feature correspondences between images."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional, Tuple, List, Dict, Union
import io

# Ensure src package is findable if running from project root
import sys, os
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"Added project root to sys.path: {project_root}")


def plot_correspondences(
    image_s: np.ndarray,
    image_q: np.ndarray,
    points_s: np.ndarray,
    points_q: np.ndarray,
    scores: Optional[np.ndarray] = None,
    max_points: Optional[int] = 50,
    show_indices: bool = False,
    line_color: str | List = 'lime',
    point_color: str = 'red',
    point_size: int = 5,
    line_width: float = 1.0,
    score_cmap: str = 'viridis',
    title: Optional[str] = "Feature Correspondences",
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots corresponding points between two images and draws lines connecting them.

    Args:
        image_s: Source image (H_s, W_s) or (H_s, W_s, 3/4).
        image_q: Query image (H_q, W_q) or (H_q, W_q, 3/4).
        points_s: Source points coordinates (N, 2) as [row, col] or [y, x].
        points_q: Query points coordinates (N, 2) as [row, col] or [y, x].
                  Must have the same number of points N as points_s.
        scores: Optional scores for each correspondence (N,). If provided, lines
                can be color-coded using score_cmap.
        max_points: Maximum number of points/lines to plot. If None, plot all.
                    If specified, points are randomly sampled.
        show_indices: If True, display the index number next to each point.
        line_color: Color for the connecting lines. If scores are provided,
                    this is ignored unless score_cmap is None.
        point_color: Color for the plotted points.
        point_size: Marker size for the points.
        line_width: Width of the connecting lines.
        score_cmap: Colormap name to use if scores are provided.
        title: Optional title for the plot figure.
        ax: Optional matplotlib Axes to plot on. If None, a new figure/axes is created.

    Returns:
        Tuple containing the matplotlib Figure and Axes objects.
    """
    if points_s.shape[0] != points_q.shape[0]:
        raise ValueError("Number of source points must match number of query points.")
    if points_s.ndim != 2 or points_s.shape[1] != 2 or \
       points_q.ndim != 2 or points_q.shape[1] != 2:
        raise ValueError("Points must be of shape (N, 2).")

    num_points = points_s.shape[0]
    indices = np.arange(num_points)

    if max_points is not None and num_points > max_points:
        print(f"Sampling {max_points} points out of {num_points} for visualization.")
        indices = np.random.choice(num_points, max_points, replace=False)
        points_s = points_s[indices]
        points_q = points_q[indices]
        if scores is not None:
            scores = scores[indices]
        num_points = max_points

    if scores is not None and score_cmap is not None:
        norm = plt.Normalize(vmin=scores.min(), vmax=scores.max())
        cmap = cm.get_cmap(score_cmap)
        line_colors = cmap(norm(scores))
    else:
        line_colors = [line_color] * num_points

    # Ensure images are suitable for imshow (e.g., handle single channel)
    def prep_img(img):
        if img.ndim == 3 and img.shape[-1] == 1:
            return img.squeeze(-1)
        return img

    image_s_disp = prep_img(image_s)
    image_q_disp = prep_img(image_q)
    cmap_s = 'gray' if image_s_disp.ndim == 2 else None
    cmap_q = 'gray' if image_q_disp.ndim == 2 else None

    # Create figure if axes not provided
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    else:
        fig = ax.figure

    # Concatenate images horizontally
    h_s, w_s = image_s_disp.shape[:2]
    h_q, w_q = image_q_disp.shape[:2]
    max_h = max(h_s, h_q)
    
    # Pad shorter image if heights differ
    if h_s < max_h:
        pad_s = np.zeros((max_h - h_s, w_s) + image_s_disp.shape[2:], dtype=image_s_disp.dtype)
        image_s_disp = np.vstack((image_s_disp, pad_s))
    if h_q < max_h:
        pad_q = np.zeros((max_h - h_q, w_q) + image_q_disp.shape[2:], dtype=image_q_disp.dtype)
        image_q_disp = np.vstack((image_q_disp, pad_q))
        
    concatenated_image = np.hstack((image_s_disp, image_q_disp))
    ax.imshow(concatenated_image, cmap=cmap_s or cmap_q) # Use cmap if either original was gray
    ax.axis('off') # Turn off axes for cleaner look

    # Plot points and lines
    # Adjust query points x-coordinates by source image width
    points_q_shifted = points_q.copy()
    points_q_shifted[:, 1] += w_s

    # Note: points are [row, col], but plot uses [x, y]
    ax.scatter(points_s[:, 1], points_s[:, 0], c=point_color, s=point_size, label="Source Points")
    ax.scatter(points_q_shifted[:, 1], points_q_shifted[:, 0], c=point_color, s=point_size, label="Query Points")

    for i in range(num_points):
        p_s = points_s[i] # row, col
        p_q = points_q_shifted[i] # row, col (shifted)
        ax.plot([p_s[1], p_q[1]], [p_s[0], p_q[0]], color=line_colors[i], linewidth=line_width)
        if show_indices:
            ax.text(p_s[1], p_s[0], str(indices[i]), color='white', ha='center', va='center', bbox=dict(facecolor=point_color, alpha=0.5, pad=0.1), fontsize=6)
            ax.text(p_q[1], p_q[0], str(indices[i]), color='white', ha='center', va='center', bbox=dict(facecolor=point_color, alpha=0.5, pad=0.1), fontsize=6)

    if title:
        fig.suptitle(title)

    # plt.tight_layout() # Often not needed with axis off and can interfere
    return fig, ax

# TODO: Add function to plot correspondence heatmap (like legacy)

def get_marker_size(coords, default_size=150):
    """
    Calculate marker size based on coordinate position.
    
    Args:
        coords: [y, x] normalized coordinates (0-1)
        default_size: Base marker size
        
    Returns:
        Appropriate marker size based on position
    """
    if not coords or len(coords) != 2:
        return default_size
        
    y, x = coords
    
    # Convert to percentage for easier comparison
    x_pct = x * 100
    y_pct = y * 100
    
    # Stars near corners might be low confidence matches
    if ((x_pct <= 5 and y_pct <= 5) or 
        (x_pct >= 95 and y_pct <= 5) or 
        (x_pct <= 5 and y_pct >= 95) or 
        (x_pct >= 95 and y_pct >= 95)):
        return default_size * 0.6  # 60% size for low confidence
    
    # For [0,0] values or very near edges (which often indicate failed matches)
    if (x_pct <= 2 and y_pct <= 2) or x_pct >= 98 or y_pct >= 98:
        return default_size * 0.5  # 50% size for likely incorrect matches
    
    return default_size  # Normal size for higher confidence matches

def plot_correspondence_grid(
    images: List[np.ndarray],
    source_view_index: int,
    poi_coords: List[float],
    match_coords: Dict[int, Optional[List[float]]],
    view_labels: Optional[List[str]] = None,
    num_cols: int = 8,
    figsize_scale: float = 1.0,
    source_color: str = 'red',
    match_color: str = 'lime',
    marker_size: int = 150,
    return_bytes: bool = False
) -> Optional[bytes]:
    """
    Plots correspondence matches across a grid of images with star markers.
    
    Args:
        images: List of images corresponding to different views
        source_view_index: Index of the source view in the images list
        poi_coords: Normalized [y, x] coordinates of the POI in the source view (0-1)
        match_coords: Dict mapping view indices to normalized [y, x] coordinates of matches (0-1)
                     or None for invalid matches
        view_labels: Optional list of labels for each view ("LF 0", "HF 1", etc.)
        num_cols: Number of columns in the grid layout
        figsize_scale: Scaling factor for the figure size
        source_color: Color for the source POI marker
        match_color: Color for the match markers
        marker_size: Size of the marker stars
        return_bytes: If True, returns PNG image bytes, otherwise shows the plot
        
    Returns:
        Optional[bytes]: PNG image bytes if return_bytes is True, otherwise None
    """
    num_total_views = len(images)
    if not view_labels:
        view_labels = [f"View {i}" for i in range(num_total_views)]
    if len(view_labels) != num_total_views:
        raise ValueError(f"Number of view labels ({len(view_labels)}) must match number of images ({num_total_views})")
    
    # Calculate grid dimensions
    num_rows = (num_total_views + num_cols - 1) // num_cols
    
    # Create figure
    fig, axes = plt.subplots(
        num_rows, num_cols, 
        figsize=(num_cols * figsize_scale * 2, num_rows * figsize_scale * 2),
        squeeze=False
    )
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    # Plot each image
    for i in range(num_total_views):
        ax = axes_flat[i]
        row, col = i // num_cols, i % num_cols
        
        # Get the image and normalize if needed
        img = images[i]
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)
        
        # Show image
        cmap = 'gray' if img.ndim == 2 else None
        ax.imshow(img, cmap=cmap)
        
        # Highlight source view
        if i == source_view_index:
            ax.patch.set_edgecolor('green')
            ax.patch.set_linewidth(4)
            
            # Add POI marker (red star) in source view
            y, x = poi_coords
            ax.scatter(
                x * img.shape[1], y * img.shape[0],
                marker='*', s=marker_size, color=source_color,
                edgecolors='white', linewidths=1, zorder=10
            )
        
        # Add match marker (green star) if available
        elif i in match_coords and match_coords[i] is not None:
            y, x = match_coords[i]
            
            # Calculate appropriate marker size based on position
            size = get_marker_size([y, x], default_size=marker_size)
            
            # Add alpha transparency based on confidence
            alpha = 0.6 if size < marker_size else 1.0
            
            ax.scatter(
                x * img.shape[1], y * img.shape[0],
                marker='*', s=size, color=match_color,
                edgecolors='black', linewidths=1, zorder=10,
                alpha=alpha
            )
        
        # Set view label
        ax.set_title(view_labels[i])
        ax.axis('off')  # Hide axes
    
    # Hide any empty subplots
    for i in range(num_total_views, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    # Add overall title
    source_label = view_labels[source_view_index]
    plt.suptitle(f"Correspondence Matches from Source View: {source_label}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
    
    # Return bytes or show plot
    if return_bytes:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)  # Close the figure to free memory
        buf.seek(0)
        return buf.getvalue()
    else:
        plt.show()
        plt.close(fig)  # Close the figure after showing
        return None

if __name__ == '__main__':
    import pickle
    import torch
    import time
    # Use tomllib (Python 3.11+) or install toml
    try:
        import tomllib
    except ImportError:
        try:
            import toml as tomllib
        except ImportError:
            raise ImportError("Please install toml ('pip install toml') or use Python 3.11+ for tomllib.")

    # Import necessary components from the project
    from src.UoB.features.upsamplers import build_feature_upsampler
    from src.UoB.data.formats import MultiViewBmodeVideo
    from src.UoB.features.matching import compute_similarity_matrix, find_mutual_nearest_neighbors, find_k_nearest_neighbors
    from src.UoB.visualization.plot_features import apply_pca_to_features, fit_joint_pca

    # --- Configuration ---
    project_root = Path(__file__).resolve().parents[3]
    data_path = project_root / "data" / "processed" / "recording_2022-08-17_trial2-arm" / "combined_mvbv.pkl"
    config_path = project_root / "configs" / "features" / "jbu_dino16.toml"
    frame_index = 0
    source_view_freq = 'lftx'
    source_view_idx = 0
    query_view_freq = 'lftx'
    query_view_idx = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Project Root: {project_root}")
    print(f"Data Path: {data_path.resolve()}")
    print(f"Config Path: {config_path.resolve()}")
    print(f"Using Device: {device}")
    print(f"Using Frame Index: {frame_index}")
    print(f"Source View: {source_view_freq} View {source_view_idx}")
    print(f"Query View: {query_view_freq} View {query_view_idx}")

    # --- Load Data ---
    print(f"\nLoading data from {data_path}...")
    if not data_path.exists():
         raise FileNotFoundError(f"Data file not found: {data_path.resolve()}")
    try:
        with open(data_path, 'rb') as f:
            mvbv_data: dict[str, MultiViewBmodeVideo] = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load pickle file {data_path}: {e}")

    # --- Load Upsampler/Feature Extractor ---
    print(f"\nLoading config from {config_path}...")
    if not config_path.exists():
         raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            upsampler_config = tomllib.load(f)
        upsampler = build_feature_upsampler(upsampler_config)
        upsampler.to(device)
        upsampler.eval()
        transform = upsampler.get_preprocessing_transform()
        print(f"Upsampler '{upsampler_config['name']}' built.")
    except Exception as e:
        raise RuntimeError(f"Failed to load config or build upsampler: {e}")

    # --- Select Images & Preprocess ---
    try:
        source_mvbv = mvbv_data[source_view_freq]
        query_mvbv = mvbv_data[query_view_freq]

        if not (0 <= frame_index < source_mvbv.n_frame and 0 <= frame_index < query_mvbv.n_frame):
            raise IndexError(f"Frame index {frame_index} out of bounds.")
        if not (0 <= source_view_idx < source_mvbv.n_view):
            raise IndexError(f"Source view index {source_view_idx} out of bounds.")
        if not (0 <= query_view_idx < query_mvbv.n_view):
            raise IndexError(f"Query view index {query_view_idx} out of bounds.")

        img_s_orig = source_mvbv.view_images[frame_index, source_view_idx]
        img_q_orig = query_mvbv.view_images[frame_index, query_view_idx]

        # Preprocess images (ensure float tensor, add channel dim if needed)
        img_s_tensor = img_s_orig.float().unsqueeze(0) if img_s_orig.ndim == 2 else img_s_orig.float()
        img_q_tensor = img_q_orig.float().unsqueeze(0) if img_q_orig.ndim == 2 else img_q_orig.float()

        img_s_prep = transform(img_s_tensor) # C, H, W
        img_q_prep = transform(img_q_tensor)

    except (KeyError, IndexError, AttributeError) as e:
        raise ValueError(f"Error selecting or processing image views: {e}")

    # --- Extract Features ---
    print("\nExtracting features...")
    start_feat_time = time.time()
    with torch.no_grad():
        feats_s = upsampler(img_s_prep.unsqueeze(0).to(device))[0] # Extract batch, get [C, H, W]
        feats_q = upsampler(img_q_prep.unsqueeze(0).to(device))[0]
    feats_s = feats_s.cpu()
    feats_q = feats_q.cpu()
    end_feat_time = time.time()
    print(f"Feature extraction took {end_feat_time - start_feat_time:.2f}s. Shape: {feats_s.shape}")

    # --- Visualize Features using JOINT PCA ---
    print("\nVisualizing features using Joint PCA...")
    start_pca_time = time.time()
    # Fit PCA jointly on source and query features
    features_list_np = [feats_s.unsqueeze(0).numpy(), feats_q.unsqueeze(0).numpy()]
    joint_pca_model = fit_joint_pca(features_list_np, n_components=3)
    # Apply the joint model to each feature map
    pca_s_vis, _ = apply_pca_to_features(features_list_np[0], pca_model=joint_pca_model)
    pca_q_vis, _ = apply_pca_to_features(features_list_np[1], pca_model=joint_pca_model)
    # Result is [N, H, W, C], squeeze N dim -> [H, W, 3]
    pca_s_img = pca_s_vis[0]
    pca_q_img = pca_q_vis[0]
    end_pca_time = time.time()
    print(f"Joint PCA visualization took {end_pca_time - start_pca_time:.2f}s. Shape: {pca_s_img.shape}")

    # --- Compute Similarity ---
    print("\nComputing similarity matrix...")
    start_sim_time = time.time()
    # Use cosine similarity (normalize=True)
    similarity_matrix = compute_similarity_matrix(feats_s, feats_q, normalize=True)
    end_sim_time = time.time()
    print(f"Similarity computation took {end_sim_time - start_sim_time:.2f}s. Shape: {similarity_matrix.shape}")

    # --- Define Points of Interest (POIs) in Source Feature Map ---
    feat_h, feat_w = feats_s.shape[1], feats_s.shape[2]
    k = 10 # Number of nearest neighbors
    pois_coords = torch.tensor([
        [int(feat_h * 0.25), int(feat_w * 0.50)], # Top-center
        [int(feat_h * 0.50), int(feat_w * 0.50)], # Middle-left
        [int(feat_h * 0.75), int(feat_w * 0.50)]  # Bottom-right
    ])
    # Convert 2D POI coords to flat 1D indices
    pois_indices_flat = pois_coords[:, 0] * feat_w + pois_coords[:, 1]
    num_pois = pois_indices_flat.shape[0]
    print(f"\nDefined {num_pois} POIs (feature coords):\n{pois_coords.numpy()}")
    print(f"Corresponding flat indices: {pois_indices_flat.numpy()}")

    # --- Find K-Nearest Neighbors for POIs ---
    print(f"\nFinding k={k} Nearest Neighbors for POIs...")
    start_match_time = time.time()
    # Find kNNs only for the POIs
    knn_scores, knn_indices_flat = find_k_nearest_neighbors(similarity_matrix, k=k, source_indices=pois_indices_flat)
    # knn_scores shape: [num_pois, k]
    # knn_indices_flat shape: [num_pois, k]
    end_match_time = time.time()
    print(f"kNN search took {end_match_time - start_match_time:.2f}s.")
    print(f"kNN Scores shape: {knn_scores.shape}")
    print(f"kNN Indices shape: {knn_indices_flat.shape}")

    # --- Prepare points for plotting --- 
    # We need to repeat source POI coords k times and flatten knn results
    
    # Repeat source POI coords k times: [num_pois, 2] -> [num_pois, k, 2] -> [num_pois * k, 2]
    points_s_plot = pois_coords.unsqueeze(1).repeat(1, k, 1).reshape(-1, 2) 
    
    # Flatten kNN query indices: [num_pois, k] -> [num_pois * k]
    knn_q_indices_flat_flat = knn_indices_flat.reshape(-1)
    
    # Convert flattened query indices to 2D coords
    knn_q_coords_r = torch.div(knn_q_indices_flat_flat, feat_w, rounding_mode='floor')
    knn_q_coords_c = knn_q_indices_flat_flat % feat_w
    points_q_plot = torch.stack((knn_q_coords_r, knn_q_coords_c), dim=1)
    
    # Flatten scores: [num_pois, k] -> [num_pois * k]
    scores_plot = knn_scores.reshape(-1)

    # Convert tensors to numpy for plotting
    points_s_np = points_s_plot.numpy()
    points_q_np = points_q_plot.numpy()
    scores_np = scores_plot.numpy()

    # --- Plot Correspondences --- 
    title_prefix=f"kNN (k={k}) Correspondences ({source_view_freq} {source_view_idx} vs {query_view_freq} {query_view_idx}, Frame {frame_index})"
    common_plot_args = dict(
        points_s=points_s_np,
        points_q=points_q_np,
        scores=scores_np,
        max_points=100,
        show_indices=False,
        point_size=10,
        line_width=1.0,
        score_cmap='viridis'
    )

    # Plot 1: On Original Preprocessed Images
    print("\nPlotting kNN correspondences on Original Images...")
    # Prepare original preprocessed images for plotting
    img_s_plot = img_s_prep.permute(1, 2, 0).cpu().numpy() # C,H,W -> H,W,C
    img_q_plot = img_q_prep.permute(1, 2, 0).cpu().numpy()
    def normalize_for_plot(img):
        min_val, max_val = img.min(), img.max()
        if max_val > min_val:
            return (img - min_val) / (max_val - min_val)
        return np.clip(img, 0, 1) # Clip just in case, ensure range
    img_s_plot = normalize_for_plot(img_s_plot)
    img_q_plot = normalize_for_plot(img_q_plot)

    fig1, ax1 = plot_correspondences(
        image_s=img_s_plot,
        image_q=img_q_plot,
        title=f"{title_prefix} - Original Images",
        **common_plot_args
    )
    # Add markers for the original POIs
    ax1.scatter(pois_coords[:, 1], pois_coords[:, 0], marker='*', s=150, facecolor='none', edgecolors='yellow', linewidth=1.5, label='POIs')
    ax1.legend()

    # Plot 2: On Joint PCA Features
    print("\nPlotting kNN correspondences on Joint PCA Features...")
    fig2, ax2 = plot_correspondences(
        image_s=pca_s_img, # Use PCA image
        image_q=pca_q_img, # Use PCA image
        title=f"{title_prefix} - Joint PCA Features",
        **common_plot_args
    )
    # Add markers for the original POIs
    ax2.scatter(pois_coords[:, 1], pois_coords[:, 0], marker='*', s=150, facecolor='none', edgecolors='yellow', linewidth=1.5, label='POIs')
    ax2.legend()

    plt.show() # Show both figures
    print("\nExample finished.")
