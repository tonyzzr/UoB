"""Functions for visualizing extracted features, including PCA."""

import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
print(f"Project root: {project_root}")
sys.path.insert(0, project_root)

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Optional
import io # Add io import for BytesIO

# TODO: Add type hints using data format classes once defined/imported

def apply_pca_to_features(
    features: np.ndarray,
    n_components: int = 3,
    pca_model: PCA | None = None,
    mask: Optional[np.ndarray] = None
) -> tuple[np.ndarray, PCA]:
    """
    Applies PCA to a batch of feature maps, optionally using a mask.

    Args:
        features: Numpy array of features, expected shape [N, C, H, W] or [N, H, W, C].
        n_components: Number of principal components (usually 3 for RGB viz).
        pca_model: Optional pre-fitted PCA model to use for transformation.
        mask: Optional boolean Numpy array with shape [N, H, W] or [H, W] if N=1.
              If provided, only pixels where mask is True are used for PCA transformation,
              and the output for masked pixels is set to 0.

    Returns:
        Tuple containing:
            - features_pca: PCA-transformed features, shape [N, H, W, n_components], scaled to [0, 1].
            - pca_model: The fitted or provided PCA model.
    """
    if not isinstance(features, np.ndarray):
        raise TypeError("Input features must be a NumPy array.")

    if features.ndim != 4:
         raise ValueError(f"Expected 4D features [N, C, H, W], got {features.ndim}D")

    N_in, C_in, H_in, W_in = features.shape # Assume input is NCHW

    # --- Correction: Always transpose NCHW to NHWC for PCA --- 
    # Original logic had a faulty condition
    features_nhwc = np.transpose(features, (0, 2, 3, 1)) # -> [N, H, W, C]
    N, H, W, C = features_nhwc.shape
    # -------------------------------------------------------

    # Reshape to [N*H*W, C] for PCA
    features_reshaped = features_nhwc.reshape(-1, C)

    # --- Mask Handling ---
    mask_flat = None
    if mask is not None:
        if not isinstance(mask, np.ndarray) or mask.dtype != bool:
             raise TypeError("Mask must be a boolean NumPy array.")
        # Handle cases where mask is [H, W] for N=1 input
        if mask.ndim == 2 and N == 1:
            mask = mask[np.newaxis, :, :] # Add batch dim -> [1, H, W]
        if mask.shape != (N, H, W):
            raise ValueError(f"Mask shape {mask.shape} must match features spatial dims {(N, H, W)}.")
        mask_flat = mask.reshape(-1) # Flatten to [N*H*W]
        print(f"Applying mask: Selecting {mask_flat.sum()} / {mask_flat.size} pixels for PCA.")
        features_to_process = features_reshaped[mask_flat]
    else:
        features_to_process = features_reshaped
    # --------------------

    if pca_model is None:
        print(f"Fitting PCA with {n_components} components on {features_to_process.shape[0]} samples...")
        if features_to_process.shape[0] < n_components:
            raise ValueError(f"Number of valid samples ({features_to_process.shape[0]}) is less than n_components ({n_components}). Cannot fit PCA.")
        pca_model = PCA(n_components=n_components)
        features_pca_flat_masked = pca_model.fit_transform(features_to_process)
        print("PCA fitting complete.")
    else:
        print(f"Transforming {features_to_process.shape[0]} samples using provided PCA model...")
        features_pca_flat_masked = pca_model.transform(features_to_process)
        print("PCA transformation complete.")

    # Create output array, initialized to 0
    features_pca_flat = np.zeros((N * H * W, n_components), dtype=features_pca_flat_masked.dtype)

    # Place transformed features into the correct locations if mask was used
    if mask_flat is not None:
        features_pca_flat[mask_flat] = features_pca_flat_masked
    else:
        features_pca_flat = features_pca_flat_masked

    # Reshape back to [N, H, W, n_components]
    features_pca = features_pca_flat.reshape(N, H, W, n_components)

    # Normalize each channel independently to [0, 1] for visualization
    # Only normalize based on the values in the non-masked areas
    normalized_features_pca = np.zeros_like(features_pca)
    for i in range(n_components):
        channel = features_pca[..., i]
        if mask is not None:
            valid_channel_pixels = channel[mask] # Select only valid pixels based on original mask shape
            if valid_channel_pixels.size == 0: # Handle case where mask is all False
                continue
            min_val = valid_channel_pixels.min()
            max_val = valid_channel_pixels.max()
        else:
            min_val = channel.min()
            max_val = channel.max()

        if max_val > min_val:
            # Apply normalization only to valid pixels, keep masked areas 0
            if mask is not None:
                normalized_channel = np.zeros_like(channel)
                normalized_channel[mask] = (channel[mask] - min_val) / (max_val - min_val)
                normalized_features_pca[..., i] = normalized_channel
            else:
                normalized_features_pca[..., i] = (channel - min_val) / (max_val - min_val)
        # else: normalized_features_pca remains 0

    return normalized_features_pca, pca_model


def fit_joint_pca(
    features_list: list[np.ndarray],
    n_components: int = 3,
    masks_list: Optional[list[np.ndarray]] = None
) -> PCA:
    """
    Fits a single PCA model on a list of feature sets concatenated together,
    optionally using corresponding masks.

    Args:
        features_list: A list where each element is a feature set (np.ndarray)
                       expected in shape [N, C, H, W] or [N, H, W, C].
                       N can vary between elements but is typically 1 for visualization.
        n_components: Number of principal components.
        masks_list: Optional list of boolean masks (np.ndarray), corresponding to features_list.
                    Each mask should have shape [N, H, W] matching the feature set's spatial dims.
                    Only pixels where mask is True are used for fitting PCA.

    Returns:
        A fitted PCA model.
    """
    reshaped_features_all_masked = []

    if masks_list is not None and len(masks_list) != len(features_list):
        raise ValueError("Length of masks_list must match length of features_list.")

    for i, features in enumerate(features_list):
        if not isinstance(features, np.ndarray):
             raise TypeError("Input features must be a list of NumPy arrays.")
        if features.ndim != 4:
             raise ValueError(f"Expected 4D features [N, C, H, W], got {features.ndim}D")

        # --- Correction: Always transpose NCHW to NHWC --- 
        features_nhwc = np.transpose(features, (0, 2, 3, 1)) # -> [N, H, W, C]
        N, H, W, C = features_nhwc.shape
        # ------------------------------------------------

        features_reshaped = features_nhwc.reshape(-1, C) # [N*H*W, C]

        # --- Mask Handling ---
        mask_flat = None
        if masks_list is not None:
            mask = masks_list[i]
            if not isinstance(mask, np.ndarray) or mask.dtype != bool:
                raise TypeError(f"Mask at index {i} must be a boolean NumPy array.")
            # Handle cases where mask is [H, W] for N=1 input
            if mask.ndim == 2 and N == 1:
                 mask = mask[np.newaxis, :, :] # Add batch dim -> [1, H, W]
            if mask.shape != (N, H, W):
                raise ValueError(f"Mask {i} shape {mask.shape} must match features spatial dims {(N, H, W)}.")
            mask_flat = mask.reshape(-1) # Flatten to [N*H*W]
            features_to_append = features_reshaped[mask_flat]
        else:
            features_to_append = features_reshaped

        if features_to_append.shape[0] > 0: # Only append if there are valid pixels
            reshaped_features_all_masked.append(features_to_append)
        elif masks_list is not None:
            print(f"Warning: Mask for feature set {i} resulted in zero valid pixels. Skipping for joint PCA fitting.")
        # --------------------

    if not reshaped_features_all_masked:
         raise ValueError("No valid feature pixels found to fit PCA (possibly due to masks).")

    # Concatenate all valid features
    all_features_flat_masked = np.concatenate(reshaped_features_all_masked, axis=0)

    print(f"Fitting joint PCA with {n_components} components on {all_features_flat_masked.shape[0]} total valid samples...")
    if all_features_flat_masked.shape[0] < n_components:
        raise ValueError(f"Total number of valid samples ({all_features_flat_masked.shape[0]}) is less than n_components ({n_components}). Cannot fit PCA.")
    pca_model = PCA(n_components=n_components)
    pca_model.fit(all_features_flat_masked)
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
    figsize_scale: float = 2.0, # Adjusted scale for potentially denser plot
    masks: Optional[list[np.ndarray]] = None, # Added optional masks list
    return_bytes: bool = False # Added flag to return bytes instead of showing
):
    """
    Plots preprocessed input images alongside PCA visualizations of features.
    Arranges plots in a grid, assuming features correspond to LF and HF views.
    Optionally applies masks during PCA.
    Can either show the plot or return it as PNG bytes.

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
        masks: Optional list of boolean masks (numpy arrays, HW) corresponding to features.
               If provided, used during PCA fitting and transformation. Length must match features.
        return_bytes: If True, saves the plot to a BytesIO buffer and returns the bytes.
                      If False (default), calls plt.show().

    Returns:
        bytes | None: PNG image bytes if return_bytes is True, otherwise None.
    """
    num_total_views = len(input_images_to_plot)
    if len(features) != num_total_views:
        raise ValueError(f"Number of feature sets ({len(features)}) must match number of images ({num_total_views}).")
    if not view_labels or len(view_labels) != num_total_views:
        raise ValueError(f"Must provide view_labels list with {num_total_views} elements.")
    if num_total_views % num_cols != 0:
         print(f"Warning: Total views ({num_total_views}) not divisible by num_cols ({num_cols}). Layout might be uneven.")
    if masks is not None and len(masks) != num_total_views:
        raise ValueError(f"Number of masks ({len(masks)}) must match number of images/features ({num_total_views}).")

    # Determine grid layout (assuming 2 rows for images, 2 for PCA = 4 rows total)
    num_img_types = num_total_views // num_cols # Should be 2 (LF, HF)
    rows = num_img_types * 2 # Image row + PCA row for each type
    fig, axes = plt.subplots(rows, num_cols, figsize=(num_cols * figsize_scale, rows * figsize_scale), squeeze=False)

    # --- PCA Fitting and Transformation ---
    pca_model = None
    pca_features_all = [] # List to store the PCA results [H, W, 3] for each view
    features_for_pca = [f[0:1] for f in features] # Extract N=0 slice, keep 4D [1, C, H, W]
    masks_for_pca = masks # Use the provided masks directly (should be HW or NHW with N=1)

    if use_joint_pca:
        print(f"Fitting joint PCA on {len(features_for_pca)} sets of {feature_type_label.lower()}...")
        try:
            pca_model = fit_joint_pca(
                features_for_pca,
                n_components=pca_n_components,
                masks_list=masks_for_pca # Pass masks here
            )
            print(f"Transforming features using the joint PCA model...")
            for i, feat_pca_input in enumerate(features_for_pca):
                mask_i = masks_for_pca[i] if masks_for_pca else None
                pca_vis, _ = apply_pca_to_features(
                    feat_pca_input,
                    pca_model=pca_model,
                    mask=mask_i # Pass individual mask here
                )
                pca_features_all.append(pca_vis[0]) # Store the [H, W, 3] result
        except ValueError as e:
            print(f"Error during joint PCA: {e}. Skipping PCA visualization.")
            # Fill with dummy data or handle error appropriately
            pca_features_all = [np.zeros((features_for_pca[0].shape[2], features_for_pca[0].shape[3], pca_n_components)) for _ in features_for_pca]

    else:
        print(f"Fitting independent PCA for each set of {feature_type_label.lower()}...")
        for i, feat_pca_input in enumerate(features_for_pca):
            mask_i = masks_for_pca[i] if masks_for_pca else None
            try:
                pca_vis, _ = apply_pca_to_features(
                    feat_pca_input,
                    n_components=pca_n_components,
                    pca_model=None, # Fit independently
                    mask=mask_i # Pass individual mask here
                )
                pca_features_all.append(pca_vis[0])
            except ValueError as e:
                 print(f"Error during independent PCA for view {i}: {e}. Skipping PCA visualization for this view.")
                 pca_features_all.append(np.zeros((feat_pca_input.shape[2], feat_pca_input.shape[3], pca_n_components)))

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
    # plt.show()

    # --- Return bytes or show plot ---
    if return_bytes:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig) # Close the figure to free memory
        buf.seek(0)
        return buf.getvalue()
    else:
        plt.show()
        plt.close(fig) # Close the figure after showing
        return None
    # --------------------------------


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
    from src.UoB.data.formats import MultiViewBmodeVideo # Import the data format
    from torchvision import transforms # Need this for Compose check

    # --- Configuration ---
    # Find project root dynamically
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent # Assumes plot_features.py is in src/UoB/visualization

    # Define paths relative to project root
    # Adjust path based on your actual data location if different
    # Ensure this path points to a combined_mvbv.pkl file containing MultiViewBmodeVideo objects
    # with the 'view_masks' attribute.
    data_path = project_root / "data" / "processed" / "recording_2022-08-17_trial2-arm" / "combined_mvbv.pkl"
    config_path = project_root / "configs" / "features" / "jbu_dino16.toml"
    frame_index = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MASK_THRESHOLD = 0.5 # Threshold for converting processed mask to binary

    print(f"Project Root (estimated): {project_root}")
    print(f"Data Path: {data_path.resolve()}")
    print(f"Config Path: {config_path.resolve()}")
    print(f"Using Device: {device}")
    print(f"Using Frame Index: {frame_index}")
    print(f"Using Mask Threshold: {MASK_THRESHOLD}")

    # --- Load Data --- 
    print(f"\nLoading data from {data_path}...")
    if not data_path.exists():
         raise FileNotFoundError(f"Data file not found at {data_path.resolve()}. Please check the path.")
    try:
        with open(data_path, 'rb') as f:
            # Assuming the pkl structure is a dict like {'lftx': mvbv_obj, 'hftx': mvbv_obj}
            mvbv_data: dict[str, MultiViewBmodeVideo] = pickle.load(f)

        # Validate data structure and mask presence
        if not isinstance(mvbv_data, dict) or 'lftx' not in mvbv_data or 'hftx' not in mvbv_data:
            raise ValueError("Pickle file does not contain expected keys 'lftx' and 'hftx'.")
        if not isinstance(mvbv_data['lftx'], MultiViewBmodeVideo) or not isinstance(mvbv_data['hftx'], MultiViewBmodeVideo):
            raise TypeError("Values for 'lftx' and 'hftx' are not MultiViewBmodeVideo objects.")
        if not hasattr(mvbv_data['lftx'], 'view_masks') or not hasattr(mvbv_data['hftx'], 'view_masks') or \
           mvbv_data['lftx'].view_masks is None or mvbv_data['hftx'].view_masks is None:
           raise AttributeError("MultiViewBmodeVideo objects must contain the 'view_masks' attribute.")

        lftx_images = mvbv_data['lftx'].view_images # Shape [T, 8, H, W]
        hftx_images = mvbv_data['hftx'].view_images # Shape [T, 8, H, W]
        lftx_masks = mvbv_data['lftx'].view_masks # Shape [T, 8, H, W] or [1, 8, H, W]
        hftx_masks = mvbv_data['hftx'].view_masks # Shape [T, 8, H, W] or [1, 8, H, W]
        print(f"Data loaded successfully. LF shape: {lftx_images.shape}, HF shape: {hftx_images.shape}")
        print(f"Masks loaded successfully. LF shape: {lftx_masks.shape}, HF shape: {hftx_masks.shape}")

        # Basic shape validation (allow masks to be static: T=1)
        if lftx_images.shape[1:] != lftx_masks.shape[1:] or \
           (lftx_masks.shape[0] != 1 and lftx_masks.shape[0] != lftx_images.shape[0]):
             raise ValueError(f"LF Image shape {lftx_images.shape} and Mask shape {lftx_masks.shape} are incompatible.")
        if hftx_images.shape[1:] != hftx_masks.shape[1:] or \
           (hftx_masks.shape[0] != 1 and hftx_masks.shape[0] != hftx_images.shape[0]):
             raise ValueError(f"HF Image shape {hftx_images.shape} and Mask shape {hftx_masks.shape} are incompatible.")

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

    # --- Prepare Data, Masks & Extract Features --- 
    transform = upsampler.get_preprocessing_transform()
    if not isinstance(transform, transforms.Compose):
        print("Warning: Expected transform to be torchvision.transforms.Compose. Mask processing might need adjustment.")

    num_lf_views = lftx_images.shape[1]
    num_hf_views = hftx_images.shape[1]
    num_total_views = num_lf_views + num_hf_views

    processed_images_list = [] # Store preprocessed images for plotting
    processed_masks_list = [] # Store preprocessed boolean masks for PCA
    features_list = []
    view_labels = [f"LF {i}" for i in range(num_lf_views)] + [f"HF {i}" for i in range(num_hf_views)]

    print(f"\nProcessing {num_total_views} views for frame {frame_index}...")
    start_extract_time = time.time()

    # Check frame index bounds
    if frame_index >= lftx_images.shape[0] or frame_index >= hftx_images.shape[0]:
        max_frames = min(lftx_images.shape[0], hftx_images.shape[0])
        raise IndexError(f"Frame index {frame_index} out of bounds for data with {max_frames} frames.")

    # Process LF images and masks
    print("Processing LF views...")
    with torch.no_grad():
        for i in range(num_lf_views):
            view_label = f"LF {i}"
            print(f"  Processing {view_label}...")
            # --- Image Processing ---
            original_img = lftx_images[frame_index, i]
            # Ensure image is float tensor C H W for transform
            # img_tensor = torch.from_numpy(original_img).float()
            img_tensor = original_img.float() # Directly convert type
            img_tensor = img_tensor.unsqueeze(0) if img_tensor.ndim == 2 else img_tensor # Add channel if needed
            preprocessed_img = transform(img_tensor) # C H W
            processed_images_list.append(preprocessed_img.cpu()) # Store preprocessed image
            input_tensor = preprocessed_img.unsqueeze(0).to(device) # Add batch N C H W
            # -----------------------

            # --- Mask Processing ---
            # Use correct attribute name 'view_masks'
            # Handle static mask (shape [1, n_view, H, W]) vs per-frame mask
            mask_frame_idx = 0 if lftx_masks.shape[0] == 1 else frame_index
            original_mask = lftx_masks[mask_frame_idx, i]
            # Ensure mask is float tensor C H W for transform (transforms usually expect float)
            # mask_tensor = torch.from_numpy(original_mask).float()
            mask_tensor = original_mask.float() # Directly convert type
            mask_tensor = mask_tensor.unsqueeze(0) # Add channel dim C=1
            # Apply the *same* transform
            preprocessed_mask = transform(mask_tensor) # Should output C H W (C=1, or C=3 if normalized)
            # Convert back to numpy, remove channel dim, threshold to boolean
            if preprocessed_mask.shape[0] == 3:
                # Select N=0, C=0 -> H W
                preprocessed_mask_np = preprocessed_mask[0].cpu().numpy()
            elif preprocessed_mask.shape[0] == 1:
                 # Select N=0, C=0 -> H W
                preprocessed_mask_np = preprocessed_mask.squeeze(0).cpu().numpy()
            else:
                 # Should not happen if transform is standard
                 raise ValueError(f"Unexpected mask shape after transform: {preprocessed_mask.shape}")
            binary_mask = preprocessed_mask_np > MASK_THRESHOLD
            processed_masks_list.append(binary_mask) # Store boolean mask [H, W]
            # ---------------------

            # --- Feature Extraction ---
            feat = upsampler(input_tensor) # [1, FeatureC, H, W]
            features_list.append(feat.cpu().numpy()) # Store features [1, C, H, W]
            print(f"    Features stored, shape: {features_list[-1].shape}")
            print(f"    Processed Mask stored, shape: {processed_masks_list[-1].shape}, True pixels: {processed_masks_list[-1].sum()}")
            # ------------------------

    # Process HF images and masks
    print("\nProcessing HF views...")
    with torch.no_grad():
        for i in range(num_hf_views):
            view_label = f"HF {i}"
            print(f"  Processing {view_label}...")
            # --- Image Processing ---
            original_img = hftx_images[frame_index, i]
            # img_tensor = torch.from_numpy(original_img).float()
            img_tensor = original_img.float() # Directly convert type
            img_tensor = img_tensor.unsqueeze(0) if img_tensor.ndim == 2 else img_tensor
            preprocessed_img = transform(img_tensor)
            processed_images_list.append(preprocessed_img.cpu())
            input_tensor = preprocessed_img.unsqueeze(0).to(device)
            # -----------------------

            # --- Mask Processing ---
            # Use correct attribute name 'view_masks'
            # Handle static mask (shape [1, n_view, H, W]) vs per-frame mask
            mask_frame_idx = 0 if hftx_masks.shape[0] == 1 else frame_index
            original_mask = hftx_masks[mask_frame_idx, i]
            # mask_tensor = torch.from_numpy(original_mask).float()
            mask_tensor = original_mask.float() # Directly convert type
            mask_tensor = mask_tensor.unsqueeze(0)
            # Apply the *same* transform
            preprocessed_mask = transform(mask_tensor)
            # --- FIX: Take only first channel if C=3, then remove N and C dims ---
            if preprocessed_mask.shape[0] == 3:
                # Select N=0, C=0 -> H W
                preprocessed_mask_np = preprocessed_mask[0].cpu().numpy()
            elif preprocessed_mask.shape[0] == 1:
                 # Select N=0, C=0 -> H W
                preprocessed_mask_np = preprocessed_mask.squeeze(0).cpu().numpy()
            else:
                 # Should not happen if transform is standard
                 raise ValueError(f"Unexpected mask shape after transform: {preprocessed_mask.shape}")
            # -------------------------------------------------------------------
            binary_mask = preprocessed_mask_np > MASK_THRESHOLD
            processed_masks_list.append(binary_mask)
            # ---------------------

            # --- Feature Extraction ---
            feat = upsampler(input_tensor)
            features_list.append(feat.cpu().numpy())
            print(f"    Features stored, shape: {features_list[-1].shape}")
            print(f"    Processed Mask stored, shape: {processed_masks_list[-1].shape}, True pixels: {processed_masks_list[-1].sum()}")
            # ------------------------

    end_extract_time = time.time()
    print(f"\nProcessing for {num_total_views} views completed in {end_extract_time - start_extract_time:.2f} seconds.")

    # --- Plot Features with Masks --- 
    print("\nPlotting results in 4x8 grid (Joint PCA with Masks)...")
    plot_start_time = time.time()
    plot_feature_pca_comparison(
        input_images_to_plot=processed_images_list,
        features=features_list,
        masks=processed_masks_list, # Pass the processed masks here
        feature_type_label="HR Features (JBU-DINO16)",
        view_labels=view_labels,
        use_joint_pca=True,
        num_cols=8,
        figsize_scale=1.8,
        return_bytes=False # Keep showing the plot for local testing
    )
    plot_end_time = time.time()
    print(f"Plotting took {plot_end_time - plot_start_time:.2f} seconds.")

    print("\nVisualization finished.")
