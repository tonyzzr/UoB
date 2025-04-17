"""Functions for visualizing extracted features, including PCA."""

import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
print(f"Project root: {project_root}")
sys.path.insert(0, project_root)

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# TODO: Add type hints using data format classes once defined/imported

def apply_pca_to_features(features: np.ndarray, n_components: int = 3, pca_model: PCA | None = None) -> tuple[np.ndarray, PCA]:
    """
    Applies PCA to a batch of feature maps.

    Args:
        features: Numpy array of features, expected shape [N, C, H, W] or [N, H, W, C].
        n_components: Number of principal components (usually 3 for RGB viz).
        pca_model: Optional pre-fitted PCA model to use for transformation.

    Returns:
        Tuple containing:
            - features_pca: PCA-transformed features, shape [N, H, W, n_components], scaled to [0, 1].
            - pca_model: The fitted or provided PCA model.
    """
    if not isinstance(features, np.ndarray):
        raise TypeError("Input features must be a NumPy array.")

    if features.ndim != 4:
         raise ValueError(f"Expected 4D features [N, C, H, W], got {features.ndim}D")

    N, C_in, H_in, W_in = features.shape # Assume input is NCHW

    # --- Correction: Always transpose NCHW to NHWC for PCA --- 
    # Original logic had a faulty condition
    features_nhwc = np.transpose(features, (0, 2, 3, 1)) # -> [N, H, W, C]
    N, H, W, C = features_nhwc.shape
    # -------------------------------------------------------

    # Reshape to [N*H*W, C] for PCA
    features_reshaped = features_nhwc.reshape(-1, C)

    if pca_model is None:
        print(f"Fitting PCA with {n_components} components on {features_reshaped.shape[0]} samples...")
        pca_model = PCA(n_components=n_components)
        features_pca_flat = pca_model.fit_transform(features_reshaped)
        print("PCA fitting complete.")
    else:
        print(f"Transforming features using provided PCA model...")
        features_pca_flat = pca_model.transform(features_reshaped)
        print("PCA transformation complete.")

    # Reshape back to [N, H, W, n_components]
    features_pca = features_pca_flat.reshape(N, H, W, n_components)

    # Normalize each channel independently to [0, 1] for visualization
    normalized_features_pca = np.zeros_like(features_pca)
    for i in range(n_components):
        channel = features_pca[..., i]
        min_val = channel.min()
        max_val = channel.max()
        if max_val > min_val:
            normalized_features_pca[..., i] = (channel - min_val) / (max_val - min_val)
        # else: normalized_features_pca remains 0

    return normalized_features_pca, pca_model


def fit_joint_pca(features_list: list[np.ndarray], n_components: int = 3) -> PCA:
    """
    Fits a single PCA model on a list of feature sets concatenated together.

    Args:
        features_list: A list where each element is a feature set (np.ndarray)
                       expected in shape [N, C, H, W] or [N, H, W, C].
                       N can vary between elements.
        n_components: Number of principal components.

    Returns:
        A fitted PCA model.
    """
    reshaped_features_all = []
    for features in features_list:
        if not isinstance(features, np.ndarray):
             raise TypeError("Input features must be a list of NumPy arrays.")
        if features.ndim != 4:
             raise ValueError(f"Expected 4D features [N, C, H, W], got {features.ndim}D")

        # --- Correction: Always transpose NCHW to NHWC --- 
        features_nhwc = np.transpose(features, (0, 2, 3, 1)) # -> [N, H, W, C]
        N, H, W, C = features_nhwc.shape
        # ------------------------------------------------

        reshaped_features_all.append(features_nhwc.reshape(-1, C))

    # Concatenate all features
    all_features_flat = np.concatenate(reshaped_features_all, axis=0)

    print(f"Fitting joint PCA with {n_components} components on {all_features_flat.shape[0]} total samples...")
    pca_model = PCA(n_components=n_components)
    pca_model.fit(all_features_flat)
    print("Joint PCA fitting complete.")

    return pca_model


def plot_feature_pca_comparison(
    input_images_to_plot: list, # List of preprocessed images (Tensors or ndarray)
    features: list[np.ndarray], # List of feature arrays [1, C, H, W]
    feature_type_label: str = "Features",
    view_labels: list[str] | None = None, # Should have 16 labels (LF0-7, HF0-7)
    use_joint_pca: bool = True,
    pca_n_components: int = 3,
    num_cols: int = 8, # Number of columns in the plot grid
    figsize_scale: float = 2.0 # Adjusted scale for potentially denser plot
):
    """
    Plots preprocessed input images alongside PCA visualizations of features.
    Arranges plots in a grid, assuming features correspond to LF and HF views.

    Args:
        input_images_to_plot: List of preprocessed images (e.g., padded square tensors)
                              corresponding to the features. Expected length 16.
        features: List of feature tensors (N=1 assumed). Expected length 16.
        feature_type_label: Label for the feature type (e.g., "HR Features").
        view_labels: List of 16 labels (e.g., "LF 0"..."HF 7").
        use_joint_pca: If True, fit one PCA model jointly across all 16 feature sets.
        pca_n_components: Number of PCA components.
        num_cols: Number of columns for the subplot grid (usually 8 for LF/HF).
        figsize_scale: Scaling factor for figure size.
    """
    num_total_views = len(input_images_to_plot)
    if len(features) != num_total_views:
        raise ValueError(f"Number of feature sets ({len(features)}) must match number of images ({num_total_views}).")
    if not view_labels or len(view_labels) != num_total_views:
        raise ValueError(f"Must provide view_labels list with {num_total_views} elements.")
    if num_total_views % num_cols != 0:
         print(f"Warning: Total views ({num_total_views}) not divisible by num_cols ({num_cols}). Layout might be uneven.")

    # Determine grid layout (assuming 2 rows for images, 2 for PCA = 4 rows total)
    num_img_types = num_total_views // num_cols # Should be 2 (LF, HF)
    rows = num_img_types * 2 # Image row + PCA row for each type
    fig, axes = plt.subplots(rows, num_cols, figsize=(num_cols * figsize_scale, rows * figsize_scale), squeeze=False)

    # --- PCA Fitting and Transformation (Jointly across all 16 views if specified) --- 
    pca_model = None
    pca_features_all = [] # List to store the PCA results [H, W, 3] for each view
    features_for_pca = [f[0:1] for f in features] # Extract N=0 slice, keep 4D [1, C, H, W]

    if use_joint_pca:
        print(f"Fitting joint PCA on {len(features_for_pca)} sets of {feature_type_label.lower()}...")
        pca_model = fit_joint_pca(features_for_pca, n_components=pca_n_components)
        print(f"Transforming features using the joint PCA model...")
        for i, feat_pca_input in enumerate(features_for_pca):
            pca_vis, _ = apply_pca_to_features(feat_pca_input, pca_model=pca_model)
            pca_features_all.append(pca_vis[0]) # Store the [H, W, 3] result
    else:
        print(f"Fitting independent PCA for each set of {feature_type_label.lower()}...")
        for i, feat_pca_input in enumerate(features_for_pca):
             pca_vis, _ = apply_pca_to_features(feat_pca_input, n_components=pca_n_components, pca_model=None)
             pca_features_all.append(pca_vis[0])
    print("PCA processing complete.")

    # --- Plotting --- 
    print("Plotting results...")
    for view_idx in range(num_total_views):
        # Determine row and column based on view index
        img_type_row_offset = (view_idx // num_cols) * 2 # 0 for LF (0-7), 2 for HF (8-15)
        col_idx = view_idx % num_cols
        label = view_labels[view_idx]

        # Plot Input Image (Row 0 or 2)
        img_row_idx = img_type_row_offset
        ax_img = axes[img_row_idx, col_idx]
        plot_img = input_images_to_plot[view_idx]
        if torch.is_tensor(plot_img):
            plot_img = plot_img.cpu().numpy()
        # Ensure image is in HWC format for imshow
        if plot_img.ndim == 3 and plot_img.shape[0] in [1, 3]: # CHW -> HWC
             plot_img = np.transpose(plot_img, (1, 2, 0))
        if plot_img.ndim == 3 and plot_img.shape[-1] == 1: # Grayscale HWC -> HW
             plot_img = plot_img.squeeze(-1)

        # Normalize intensity to [0, 1] for consistent display if it's float
        if plot_img.dtype == np.float32 or plot_img.dtype == np.float64:
            min_val, max_val = plot_img.min(), plot_img.max()
            if max_val > min_val:
                 plot_img = (plot_img - min_val) / (max_val - min_val)
            else:
                 plot_img = np.zeros_like(plot_img)

        cmap = 'gray' if plot_img.ndim == 2 else None
        ax_img.imshow(plot_img, cmap=cmap)
        ax_img.set_title(f"Input {label}") # Changed title
        ax_img.axis('off')

        # Plot Feature PCA (Row 1 or 3)
        pca_row_idx = img_type_row_offset + 1
        ax_pca = axes[pca_row_idx, col_idx]
        pca_vis = pca_features_all[view_idx]
        ax_pca.imshow(pca_vis)
        ax_pca.set_title(f"PCA {label}") # Simplified title
        ax_pca.axis('off')

    pca_mode_str = "Joint PCA" if use_joint_pca else "Independent PCA"
    fig.suptitle(f"{feature_type_label} Visualization ({pca_mode_str})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust rect for suptitle
    plt.show()


# Example Usage (using real data)
if __name__ == '__main__':
    import pickle
    import os
    import time
    from pathlib import Path
    # Use tomllib (Python 3.11+) or install toml
    try:
        import tomllib
    except ImportError:
        try:
            import toml as tomllib
        except ImportError:
            raise ImportError("Please install toml ('pip install toml') or use Python 3.11+ for tomllib.")

    # Assuming running from project root or src is in path
    from src.UoB.features.upsamplers import build_feature_upsampler

    # --- Configuration ---
    # Find project root dynamically
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent # Assumes plot_features.py is in src/UoB/visualization

    # Define paths relative to project root
    # data_path = project_root / "data" / "processed" / "recording_2022-08-17_trial2-arm" / "combined_mvbv.pkl"
    # Adjust path based on your actual data location if different - using placeholder if directly in data/processed
    # TODO: This path might need adjustment based on where the script is run from or project structure
    data_path = Path("data/processed/recording_2022-08-17_trial2-arm/combined_mvbv.pkl")
    config_path = Path("configs/features/jbu_dino16.toml")
    frame_index = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Project Root (estimated): {project_root}")
    print(f"Data Path: {data_path.resolve()}")
    print(f"Config Path: {config_path.resolve()}")
    print(f"Using Device: {device}")
    print(f"Using Frame Index: {frame_index}")

    # --- Load Data ---
    print(f"\nLoading data from {data_path}...")
    if not data_path.exists():
         raise FileNotFoundError(f"Data file not found at {data_path.resolve()}. Please check the path.")
    try:
        with open(data_path, 'rb') as f:
            # Assuming the pkl structure is a dict like {'lftx': mvbv_obj, 'hftx': mvbv_obj}
            # where mvbv_obj.view_images is [T, N_Views, H, W]
            mvbv_data = pickle.load(f)
        lftx_images = mvbv_data['lftx'].view_images # Shape [T, 8, H, W]
        hftx_images = mvbv_data['hftx'].view_images # Shape [T, 8, H, W]
        print(f"Data loaded successfully. LF shape: {lftx_images.shape}, HF shape: {hftx_images.shape}")
    except Exception as e:
        raise RuntimeError(f"Failed to load or parse pickle file {data_path}: {e}")

    # --- Load Upsampler --- 
    print(f"\nLoading config from {config_path}...")
    if not config_path.exists():
         raise FileNotFoundError(f"Config file not found at {config_path.resolve()}. Please check the path.")
    try:
        # Open in text mode ('r') for the toml package
        with open(config_path, 'r', encoding='utf-8') as f:
            upsampler_config = tomllib.load(f)
        print("Config loaded:", upsampler_config)
        print(f"Building upsampler '{upsampler_config['name']}'...")
        upsampler = build_feature_upsampler(upsampler_config)
        print(f"Moving upsampler to {device}...")
        upsampler.to(device)
        upsampler.eval()
        print("Upsampler built and moved to device.")
    except Exception as e:
        raise RuntimeError(f"Failed to load config or build upsampler: {e}")

    # --- Prepare Data & Extract Features --- 
    transform = upsampler.get_preprocessing_transform()
    num_lf_views = lftx_images.shape[1]
    num_hf_views = hftx_images.shape[1]
    num_total_views = num_lf_views + num_hf_views

    processed_images_list = [] # Store preprocessed images for plotting
    features_list = []
    view_labels = [f"LF {i}" for i in range(num_lf_views)] + [f"HF {i}" for i in range(num_hf_views)]

    print(f"\nProcessing {num_total_views} views for frame {frame_index}...")
    start_extract_time = time.time()

    # Check frame index bounds
    if frame_index >= lftx_images.shape[0] or frame_index >= hftx_images.shape[0]:
        max_frames = min(lftx_images.shape[0], hftx_images.shape[0])
        raise IndexError(f"Frame index {frame_index} out of bounds for data with {max_frames} frames.")

    # Process LF images
    print("Processing LF views...")
    with torch.no_grad():
        for i in range(num_lf_views):
            view_label = f"LF {i}"
            print(f"  Processing {view_label}...")
            original_img = lftx_images[frame_index, i]
            # --- Debug: Print ORIGINAL tensor stats ---
            print(f"    Original Tensor Stats (min/max/mean): {original_img.min():.3f} / {original_img.max():.3f} / {original_img.mean():.3f}")
            # -----------------------------------------
            # original_images_list.append(original_img.cpu()) # No longer needed

            # Preprocess (PadToSquareAndAlign handles tensor input)
            img_for_transform = original_img.unsqueeze(0) if original_img.ndim == 2 else original_img
            preprocessed_img = transform(img_for_transform)
            processed_images_list.append(preprocessed_img.cpu()) # Store preprocessed image

            # Add batch dim and move to device
            input_tensor = preprocessed_img.unsqueeze(0).to(device)

            # --- Debug: Print input tensor stats ---
            print(f"    Input Tensor Stats (min/max/mean): {input_tensor.min():.3f} / {input_tensor.max():.3f} / {input_tensor.mean():.3f}")
            # ---------------------------------------

            # Run forward pass
            feat = upsampler(input_tensor)

            # 5. Store features (CPU, NumPy) - Keep Batch Dimension!
            features_list.append(feat.cpu().numpy())
            print(f"    Features stored, shape: {features_list[-1].shape}")

    # Process HF images
    print("\nProcessing HF views...")
    with torch.no_grad():
        for i in range(num_hf_views):
            view_label = f"HF {i}"
            print(f"  Processing {view_label}...")
            original_img = hftx_images[frame_index, i]
            # --- Debug: Print ORIGINAL tensor stats ---
            print(f"    Original Tensor Stats (min/max/mean): {original_img.min():.3f} / {original_img.max():.3f} / {original_img.mean():.3f}")
            # -----------------------------------------
            # original_images_list.append(original_img.cpu()) # No longer needed

            # Preprocess (PadToSquareAndAlign handles tensor input)
            img_for_transform = original_img.unsqueeze(0) if original_img.ndim == 2 else original_img
            preprocessed_img = transform(img_for_transform)
            processed_images_list.append(preprocessed_img.cpu()) # Store preprocessed image

            # Add batch dim and move to device
            input_tensor = preprocessed_img.unsqueeze(0).to(device)

            # --- Debug: Print input tensor stats ---
            print(f"    Input Tensor Stats (min/max/mean): {input_tensor.min():.3f} / {input_tensor.max():.3f} / {input_tensor.mean():.3f}")
            # ---------------------------------------

            # Run forward pass
            feat = upsampler(input_tensor)

            # 5. Store features (CPU, NumPy) - Keep Batch Dimension!
            features_list.append(feat.cpu().numpy())
            print(f"    Features stored, shape: {features_list[-1].shape}")

    end_extract_time = time.time()
    print(f"\nFeature extraction for {num_total_views} views completed in {end_extract_time - start_extract_time:.2f} seconds.")

    # --- Plot Features --- 
    print("\nPlotting results in 4x8 grid (Joint PCA)...")
    plot_feature_pca_comparison(
        input_images_to_plot=processed_images_list,
        features=features_list,
        feature_type_label="HR Features (JBU-DINO16)",
        view_labels=view_labels,
        use_joint_pca=True,
        num_cols=8,
        figsize_scale=1.8
    )

    print("\nVisualization finished.")
