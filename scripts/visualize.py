#!/usr/bin/env python
"""Script for running various visualization tasks."""

import typer
import sys
from pathlib import Path
import pickle
import time

# Ensure src package is findable
project_root = Path(__file__).resolve().parents[1] # scripts -> project root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"Added project root to sys.path: {project_root}")

# Conditional imports based on availability or specific subcommands
try:
    import matplotlib.pyplot as plt
    import torch
    import numpy as np
    from src.UoB.features.matching import compute_similarity_matrix, find_k_nearest_neighbors
    from src.UoB.visualization.plot_correspondence import plot_correspondences
    # Add other necessary imports for data loading, feature extraction etc.
    from src.UoB.features.upsamplers import build_feature_upsampler
    from src.UoB.data.formats import MultiViewBmodeVideo
    # Use tomllib (Python 3.11+) or install toml
    try:
        import tomllib
    except ImportError:
        try:
            import toml as tomllib
        except ImportError:
            # Provide a clearer error message if toml is needed
            print("Error: 'tomllib' not found. Please install 'toml' (`pip install toml`) or use Python 3.11+.", file=sys.stderr)
            sys.exit(1)
    VIS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import necessary libraries for visualization: {e}", file=sys.stderr)
    VIS_AVAILABLE = False

# Create a Typer app instance
app = typer.Typer(
    name="visualize",
    help="Run visualization tasks for the Multi-View Ultrasound project.",
    add_completion=False
)

@app.command()
def correspondence(
    config: Path = typer.Option(
        ..., 
        "--config", "-c", 
        help="Path to the TOML configuration file for visualization.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    source_view: str = typer.Option("lftx:0", help="Source view identifier (e.g., 'lftx:0', 'hftx:3')"),
    query_view: str = typer.Option("lftx:1", help="Query view identifier (e.g., 'lftx:1', 'hftx:0')"),
    frame: int = typer.Option(0, "--frame", "-f", help="Frame index to visualize."),
    k: int = typer.Option(10, "--k", help="Number of nearest neighbors to find and plot per POI."),
    num_pois: int = typer.Option(5, "--num-pois", help="Number of Points of Interest (POIs) to sample randomly."),
    # TODO: Add option for specific POI coordinates?
    match_type: str = typer.Option("knn", help="Type of matching to visualize ('knn', 'mnn'). Currently only knn supported here.")
):
    """
    Visualizes feature correspondences (kNN or MNN) between two views.
    
    Loads data, extracts features, computes similarity, finds matches,
    and plots the results based on the provided configuration file.
    """
    if not VIS_AVAILABLE:
        print("Error: Necessary libraries not installed. Cannot run visualization.", file=sys.stderr)
        raise typer.Exit(code=1)
        
    print(f"Running correspondence visualization...")
    print(f" - Config: {config}")
    print(f" - Source: {source_view}")
    print(f" - Query: {query_view}")
    print(f" - Frame: {frame}")
    print(f" - Match Type: {match_type}")
    print(f" - k: {k}")
    print(f" - Num POIs: {num_pois}")

    # 1. Load configuration (example - needs refinement)
    try:
        with open(config, 'r', encoding='utf-8') as f:
            vis_config = tomllib.load(f)
        print("\nLoaded visualization config:")
        print(vis_config)
        # TODO: Extract data path, feature config path, etc. from vis_config
        # Example structure expected in config:
        # [data]
        # recording_id = "recording_2022-08-17_trial2-arm"
        # data_root = "data/processed" # Optional, defaults relative to project
        # [features]
        # config_path = "configs/features/jbu_dino16.toml"
        # [visualization]
        # device = "cuda"
        data_cfg = vis_config.get('data', {})
        feat_cfg = vis_config.get('features', {})
        viz_cfg = vis_config.get('visualization', {})

        recording_id = data_cfg.get('recording_id')
        data_root = Path(data_cfg.get('data_root', project_root / 'data' / 'processed'))
        feature_config_path = Path(feat_cfg.get('config_path', project_root / 'configs' / 'features' / 'jbu_dino16.toml'))
        device_str = viz_cfg.get('device', "auto").lower() # Read from config, default to auto, ensure lowercase
        # Resolve "auto" device
        if device_str == "auto":
            resolved_device_str = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Device 'auto' resolved to '{resolved_device_str}'")
        else:
            resolved_device_str = device_str # Use the specified device
        device = torch.device(resolved_device_str)
        
        if not recording_id:
            raise ValueError("Missing 'recording_id' in [data] section of config.")
        if not feature_config_path.exists():
             raise FileNotFoundError(f"Feature config path not found: {feature_config_path}")
        data_path = data_root / recording_id / "combined_mvbv.pkl"
        if not data_path.exists():
             raise FileNotFoundError(f"Data path not found: {data_path}")
             
    except (tomllib.TomlDecodeError, ValueError, FileNotFoundError) as e:
        print(f"Error loading or parsing config file {config}: {e}", file=sys.stderr)
        raise typer.Exit(code=1)
        
    # 2. Parse view identifiers (e.g., "lftx:0")
    try:
        source_freq, source_idx_str = source_view.split(':')
        query_freq, query_idx_str = query_view.split(':')
        source_idx = int(source_idx_str)
        query_idx = int(query_idx_str)
        if source_freq not in ['lftx', 'hftx'] or query_freq not in ['lftx', 'hftx']:
             raise ValueError("Frequency must be 'lftx' or 'hftx'")
    except ValueError as e:
        print(f"Error parsing view identifiers ('{source_view}', '{query_view}'): {e}. Expected format 'freq:index' (e.g., 'lftx:0').", file=sys.stderr)
        raise typer.Exit(code=1)
        
    # 3. Load Data (similar to example in plot_correspondence)
    print(f"\nLoading data from {data_path}...")
    try:
        with open(data_path, 'rb') as f:
            mvbv_data: dict[str, MultiViewBmodeVideo] = pickle.load(f)
    except Exception as e:
        print(f"Error loading data pickle file {data_path}: {e}", file=sys.stderr)
        raise typer.Exit(code=1)

    # 4. Load Feature Extractor
    print(f"\nLoading feature extractor from {feature_config_path}...")
    try:
        with open(feature_config_path, 'r', encoding='utf-8') as f:
            upsampler_config = tomllib.load(f)
        upsampler = build_feature_upsampler(upsampler_config)
        upsampler.to(device)
        upsampler.eval()
        transform = upsampler.get_preprocessing_transform()
        print(f"Upsampler '{upsampler_config['name']}' built.")
    except Exception as e:
        print(f"Error loading config or building upsampler: {e}", file=sys.stderr)
        raise typer.Exit(code=1)

    # 5. Select Images & Preprocess
    print(f"\nSelecting and preprocessing images...")
    try:
        source_mvbv = mvbv_data[source_freq]
        query_mvbv = mvbv_data[query_freq]

        if not (0 <= frame < source_mvbv.n_frame and 0 <= frame < query_mvbv.n_frame):
            raise IndexError(f"Frame index {frame} out of bounds.")
        if not (0 <= source_idx < source_mvbv.n_view):
            raise IndexError(f"Source view index {source_idx} out of bounds.")
        if not (0 <= query_idx < query_mvbv.n_view):
            raise IndexError(f"Query view index {query_idx} out of bounds.")

        img_s_orig = source_mvbv.view_images[frame, source_idx]
        img_q_orig = query_mvbv.view_images[frame, query_idx]

        img_s_tensor = img_s_orig.float().unsqueeze(0) if img_s_orig.ndim == 2 else img_s_orig.float()
        img_q_tensor = img_q_orig.float().unsqueeze(0) if img_q_orig.ndim == 2 else img_q_orig.float()

        img_s_prep = transform(img_s_tensor) # C, H, W
        img_q_prep = transform(img_q_tensor)

    except (KeyError, IndexError, AttributeError, TypeError) as e:
        print(f"Error selecting or processing image views: {e}", file=sys.stderr)
        raise typer.Exit(code=1)

    # 6. Extract Features
    print("\nExtracting features...")
    start_feat_time = time.time()
    try:
        with torch.no_grad():
            feats_s = upsampler(img_s_prep.unsqueeze(0).to(device))[0]
            feats_q = upsampler(img_q_prep.unsqueeze(0).to(device))[0]
        feats_s = feats_s.cpu()
        feats_q = feats_q.cpu()
        end_feat_time = time.time()
        print(f"Feature extraction took {end_feat_time - start_feat_time:.2f}s. Shape: {feats_s.shape}")
    except Exception as e:
        print(f"Error during feature extraction: {e}", file=sys.stderr)
        raise typer.Exit(code=1)

    # 7. Compute Similarity
    print("\nComputing similarity matrix...")
    start_sim_time = time.time()
    try:
        similarity_matrix = compute_similarity_matrix(feats_s, feats_q, normalize=True)
        end_sim_time = time.time()
        print(f"Similarity computation took {end_sim_time - start_sim_time:.2f}s. Shape: {similarity_matrix.shape}")
    except Exception as e:
        print(f"Error computing similarity: {e}", file=sys.stderr)
        raise typer.Exit(code=1)
    
    # 8. Define POIs (randomly sample for now)
    feat_h, feat_w = feats_s.shape[1], feats_s.shape[2]
    if num_pois > (feat_h * feat_w):
        print(f"Warning: num_pois ({num_pois}) > total pixels ({feat_h * feat_w}). Clamping.", file=sys.stderr)
        num_pois = feat_h * feat_w
        
    all_indices_flat = torch.arange(feat_h * feat_w)
    # Ensure we sample unique indices if num_pois is less than total
    replace_sample = False if num_pois <= (feat_h * feat_w) else True 
    pois_indices_flat = all_indices_flat[torch.randperm(feat_h * feat_w)[:num_pois]] if not replace_sample else torch.randint(0, feat_h * feat_w, (num_pois,))

    print(f"\nRandomly selected {num_pois} POIs (flat indices): {pois_indices_flat.numpy()}")

    # 9. Find Matches (kNN or MNN)
    if match_type.lower() == 'knn':
        print(f"\nFinding k={k} Nearest Neighbors for POIs...")
        knn_scores, knn_indices_flat = find_k_nearest_neighbors(
            similarity_matrix, k=k, source_indices=pois_indices_flat
        )
        # Prepare points for plotting kNN
        pois_coords_r = torch.div(pois_indices_flat, feat_w, rounding_mode='floor')
        pois_coords_c = pois_indices_flat % feat_w
        pois_coords = torch.stack((pois_coords_r, pois_coords_c), dim=1)
        
        points_s_plot = pois_coords.unsqueeze(1).repeat(1, k, 1).reshape(-1, 2)
        knn_q_indices_flat_flat = knn_indices_flat.reshape(-1)
        knn_q_coords_r = torch.div(knn_q_indices_flat_flat, feat_w, rounding_mode='floor')
        knn_q_coords_c = knn_q_indices_flat_flat % feat_w
        points_q_plot = torch.stack((knn_q_coords_r, knn_q_coords_c), dim=1)
        scores_plot = knn_scores.reshape(-1)
        plot_title = f"kNN (k={k}) Correspondences ({source_view} vs {query_view}, Frame {frame})"

    elif match_type.lower() == 'mnn':
        print("\nFinding Mutual Nearest Neighbors...")
        # ... (Add MNN finding and point preparation logic here) ... 
        # mnn_s_indices_flat, mnn_q_indices_flat = ...
        # points_s_plot = ...
        # points_q_plot = ...
        # scores_plot = ...
        print("MNN plotting not fully implemented in script yet.")
        raise typer.Exit(code=1)
    else:
        print(f"Error: Unknown match_type '{match_type}'. Use 'knn' or 'mnn'.", file=sys.stderr)
        raise typer.Exit(code=1)
        
    # Convert to numpy for plotting
    points_s_np = points_s_plot.cpu().numpy()
    points_q_np = points_q_plot.cpu().numpy()
    scores_np = scores_plot.cpu().numpy()
    
    # 10. Generate PCA visualizations (Joint PCA)
    print("\nGenerating PCA visualizations...")
    start_pca_time = time.time()
    try:
        from src.UoB.visualization.plot_features import apply_pca_to_features, fit_joint_pca
        features_list_np = [feats_s.unsqueeze(0).numpy(), feats_q.unsqueeze(0).numpy()]
        joint_pca_model = fit_joint_pca(features_list_np, n_components=3)
        pca_s_vis, _ = apply_pca_to_features(features_list_np[0], pca_model=joint_pca_model)
        pca_q_vis, _ = apply_pca_to_features(features_list_np[1], pca_model=joint_pca_model)
        pca_s_img = pca_s_vis[0]
        pca_q_img = pca_q_vis[0]
        end_pca_time = time.time()
        print(f"Joint PCA visualization took {end_pca_time - start_pca_time:.2f}s.")
    except Exception as e:
        print(f"Error during PCA processing: {e}", file=sys.stderr)
        # Continue without PCA plot if it fails?
        pca_s_img = np.zeros((feat_h, feat_w, 3))
        pca_q_img = np.zeros((feat_h, feat_w, 3))

    # 11. Prepare Original Images for Plotting
    print("\nPreparing original images for plotting...")
    try:
        img_s_plot = img_s_prep.permute(1, 2, 0).cpu().numpy()
        img_q_plot = img_q_prep.permute(1, 2, 0).cpu().numpy()
        def normalize_for_plot(img):
            min_val, max_val = img.min(), img.max()
            if max_val > min_val:
                return (img - min_val) / (max_val - min_val)
            return np.clip(img, 0, 1)
        img_s_plot = normalize_for_plot(img_s_plot)
        img_q_plot = normalize_for_plot(img_q_plot)
    except Exception as e:
        print(f"Error preparing original images for plot: {e}", file=sys.stderr)
        # Use placeholders if error
        img_s_plot = np.zeros((feat_h, feat_w, 3))
        img_q_plot = np.zeros((feat_h, feat_w, 3))
        
    # 12. Plot Results
    common_plot_args = dict(
        points_s=points_s_np,
        points_q=points_q_np,
        scores=scores_np,
        max_points=num_pois * (k if match_type.lower() == 'knn' else 1), # Show all matches for POIs
        show_indices=False,
        point_size=10,
        line_width=1.0,
        score_cmap='viridis'
    )
    
    # Plot 1: On Original Images
    print("\nPlotting correspondences on Original Images...")
    fig1, ax1 = plot_correspondences(
        image_s=img_s_plot,
        image_q=img_q_plot,
        title=f"{plot_title} - Original Images",
        **common_plot_args
    )
    # Add POI markers if kNN
    if match_type.lower() == 'knn':
        ax1.scatter(pois_coords[:, 1].cpu().numpy(), pois_coords[:, 0].cpu().numpy(), marker='*', s=150, facecolor='none', edgecolors='yellow', linewidth=1.5, label='POIs')
        ax1.legend()

    # Plot 2: On PCA Features
    print("\nPlotting correspondences on PCA features...")
    fig2, ax2 = plot_correspondences(
        image_s=pca_s_img, 
        image_q=pca_q_img,
        title=plot_title + " - Joint PCA Features",
        **common_plot_args
    )
    # Add POI markers if kNN
    if match_type.lower() == 'knn':
        ax2.scatter(pois_coords[:, 1].cpu().numpy(), pois_coords[:, 0].cpu().numpy(), marker='*', s=150, facecolor='none', edgecolors='yellow', linewidth=1.5, label='POIs')
        ax2.legend()
    plt.show()

    print("\nVisualization script finished.")


@app.command()
def dummy():
    """A dummy command for testing the CLI structure."""
    print("Dummy command executed successfully.")


if __name__ == "__main__":
    app() 