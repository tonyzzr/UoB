# scripts/debug_mat_converter_steps.py
import sys
import os

from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
import warnings

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.UoB.utils import processing # Import the processing module directly

# --- Add project root to sys.path ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    print(f"Added PROJECT_ROOT to sys.path: {PROJECT_ROOT}")

    from src.UoB.preprocessing.mat_converter import MatConverter
    from src.UoB.data.readers import RecordingLoader
    from src.UoB.data.formats import BmodeConfig, MatData # Import necessary formats

    print("Successfully imported necessary UoB modules.")

except ImportError as e:
    print(f"Error: Could not import required UoB modules.", file=sys.stderr)
    print(f"Check paths and ensure necessary class definitions exist.", file=sys.stderr)
    print(f"Import error details: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error setting up sys.path or initial imports: {e}", file=sys.stderr)
    sys.exit(1)


def visualize_steps(
    config_path: Path,
    recording_dir: Path,
    freq_key: str = 'lftx',
    frame_index: int = 0,
    view_index: int = 0
):
    """Loads raw data, runs processing steps, and visualizes intermediate results."""

    print(f"\n--- Initializing MatConverter ---")
    print(f"Config path: {config_path}")
    if not config_path.is_file():
         print(f"Error: Config file not found at {config_path}", file=sys.stderr)
         sys.exit(1)
    try:
         converter = MatConverter(config_path=config_path)
         bmode_config = converter.lf_bmode_config if freq_key == 'lftx' else converter.hf_bmode_config
    except Exception as e:
        print(f"Error initializing MatConverter or getting config: {e}", file=sys.stderr); sys.exit(1)

    print(f"\n--- Loading Raw Data ---")
    print(f"Recording directory: {recording_dir}")
    if not recording_dir.is_dir():
        print(f"Error: Recording directory not found at {recording_dir}", file=sys.stderr)
        sys.exit(1)
    try:
        loader = RecordingLoader(recording_dir)
        raw_lf_data, raw_hf_data = loader.load_combined_mat_data()
        raw_mat_data = raw_lf_data if freq_key == 'lftx' else raw_hf_data
        if raw_mat_data is None: raise ValueError(f"Could not load raw data for {freq_key}")
        print(f"Successfully loaded raw data for '{freq_key}'.")
    except Exception as e:
        print(f"Error loading raw data: {e}", file=sys.stderr); sys.exit(1)

    # --- Visualize RAW Data Directly ---
    print(f"\n--- Visualizing Raw Loaded Data (Before Any Processing) ---")
    try:
        # Access the imgdata dictionary for the specified spatial view
        if view_index not in raw_mat_data.imgdata:
            raise ValueError(f"Spatial view index {view_index} not found in raw_mat_data.imgdata keys: {list(raw_mat_data.imgdata.keys())}")
        
        raw_view_data = raw_mat_data.imgdata[view_index]
        print(f"Raw data array shape for view {view_index}: {raw_view_data.shape}") # Expected: (n_frames, ?, h_raw, w_raw)

        # Validate frame index
        raw_n_frames = raw_view_data.shape[0]
        if not (0 <= frame_index < raw_n_frames):
            raise ValueError(f"Invalid frame index {frame_index} for raw data. Max is {raw_n_frames-1}.")

        # Select the specific frame and squeeze potential singleton dim (axis 1)
        raw_frame_data = np.squeeze(raw_view_data[frame_index, ...]) # Squeeze potential singleton dim
        print(f"Raw frame data shape after squeeze: {raw_frame_data.shape}") # Expected: (h_raw, w_raw)

        if raw_frame_data.ndim != 2:
            print(f"Warning: Raw frame data is not 2D after squeeze (shape: {raw_frame_data.shape}). Plotting might be incorrect.")

        plt.figure(figsize=(7, 7))
        plt.imshow(raw_frame_data, cmap='gray')
        plt.title(f"0. Raw Loaded Data\nRec: {recording_dir.name}, Frame: {frame_index}, Freq: {freq_key}, View: {view_index}")
        plt.xlabel("Raw Width"); plt.ylabel("Raw Height")
        plt.colorbar(label='Raw Intensity')
        print(f"Raw Data range (min, max): ({np.min(raw_frame_data)}, {np.max(raw_frame_data)})")
        print("Displaying raw data plot...")
        plt.show() # Show this plot immediately

    except Exception as e:
        print(f"Error during Raw Data Visualization: {e}", file=sys.stderr)
        import traceback; traceback.print_exc(); sys.exit(1)

    print(f"\n--- Step 1: Preparing Initial Image Sequence ---")
    initial_img_seq = None
    target_h, target_w = 0, 0
    n_views = 0 # Initialize
    try:
        target_h, target_w = converter._calculate_image_size(raw_mat_data, bmode_config.scale_bar)
        print(f"Calculated Target Size (h, w): ({target_h}, {target_w})")
        initial_img_seq = converter._prepare_initial_image_sequence(raw_mat_data.imgdata, (target_h, target_w))
        if initial_img_seq is None: raise ValueError("Prep returned None")
        print(f"Shape after _prepare_initial_image_sequence: {initial_img_seq.shape}")

        n_frames = initial_img_seq.shape[3]
        n_views = initial_img_seq.shape[2]
        if not (0 <= frame_index < n_frames): raise ValueError(f"Invalid frame index {frame_index}. Max is {n_frames-1}.")
        if not (0 <= view_index < n_views): raise ValueError(f"Invalid spatial view index {view_index}. Max is {n_views-1}.")

        frame_to_visualize_initial = initial_img_seq[:, :, view_index, frame_index]
        print(f"Initial Data range (min, max): ({np.min(frame_to_visualize_initial)}, {np.max(frame_to_visualize_initial)})")

        # Visualize Initial Prep
        plt.figure(figsize=(12, 6)) # Wider figure for two plots
        plt.subplot(1, 2, 1)
        plt.imshow(frame_to_visualize_initial, cmap='gray')
        plt.title(f"1. After Initial Prep\nFrame:{frame_index}, Freq:{freq_key}, View:{view_index}")
        plt.xlabel("Width"); plt.ylabel("Height")
        plt.colorbar(label='Raw/Resized Intensity')

    except Exception as e:
        print(f"Error during Step 1 (Initial Prep): {e}", file=sys.stderr)
        import traceback; traceback.print_exc(); sys.exit(1)

    print(f"\n--- Step 2: Calculating Masks & Applying B-Mode Processing (Step-by-Step) ---")
    mask_seq = None
    proc_img = initial_img_seq.astype(float) # Start with float version of initial image
    try:
        # Dependencies for mask calculation
        print("Calculating transducer positions...")
        trans_pos = converter._calculate_transducer_positions(raw_mat_data, bmode_config.scale_bar)
        num_trans = len(trans_pos)
        if num_trans != n_views: # Sanity check
            warnings.warn(f"Mismatch: num_trans from pos ({num_trans}) != n_views from initial_img ({n_views})")

        # Calculate masks
        print("Calculating masks...")
        mask_seq = converter._calculate_masks(num_trans, target_h, target_w, trans_pos, bmode_config.mask_setting)
        print(f"Calculated mask_seq shape: {mask_seq.shape}") # Expected: (1, n_views, h, w)

        # Define and Apply Processing Steps Sequentially
        processing_steps_def = [
            ("Log Compression", processing.log_compression, bmode_config.log_compression_setting),
            ("Speckle Reduction", processing.speckle_reduction, bmode_config.speckle_reduction_setting),
            ("Reject Grating Lobe", processing.reject_grating_lobe_artifact, bmode_config.reject_grating_lobe_setting, mask_seq),
            ("Apply TGC", processing.apply_tgc, bmode_config.time_gain_compensation_setting),
            ("Histogram Match", processing.histogram_match, bmode_config.histogram_match_setting),
        ]
        
        num_plots = 1 + sum(1 for _, _, setting, *_ in processing_steps_def if setting.enable) # Initial + enabled steps
        plt.figure(figsize=(6 * num_plots, 6))
        plot_index = 1

        # Plot Initial
        plt.subplot(1, num_plots, plot_index)
        plt.imshow(frame_to_visualize_initial, cmap='gray')
        plt.title(f"{plot_index}. Initial Prep\nFrame:{frame_index}, Freq:{freq_key}, View:{view_index}")
        plt.xlabel("W"); plt.ylabel("H")
        plt.colorbar(label='Intensity')
        plot_index += 1

        print("\nApplying B-mode processing steps individually:")
        for name, func, setting, *args in processing_steps_def:
            if setting.enable:
                print(f"  Applying: {name}...")
                prev_shape = proc_img.shape # Store shape before applying
                try:
                    if args:
                        proc_img = func(proc_img, *args, setting=setting)
                    else:
                        proc_img = func(proc_img, setting=setting)
                    
                    # ---> Add shape check <--- 
                    current_shape = proc_img.shape
                    print(f"    Shape after {name}: {current_shape} (Previous: {prev_shape})")
                    if current_shape[:2] != prev_shape[:2]:
                         print(f"    WARNING: Shape mismatch detected after {name}! Expected H,W: {prev_shape[:2]}, Got: {current_shape[:2]}")
                    # ----------------------

                    # Visualize after this step
                    current_frame_view = proc_img[:, :, view_index, frame_index]
                    data_min, data_max = np.min(current_frame_view), np.max(current_frame_view)
                    print(f"    Data range after {name}: ({data_min}, {data_max})")
                    
                    plt.subplot(1, num_plots, plot_index)
                    plt.imshow(current_frame_view, cmap='gray')
                    plt.title(f"{plot_index}. After {name}\n(Range: {data_min:.1f}-{data_max:.1f})")
                    plt.xlabel("W"); plt.ylabel("H")
                    plt.colorbar(label='Intensity')
                    plot_index += 1

                except Exception as step_e:
                    print(f"    ERROR during step '{name}': {step_e}")
                    import traceback; traceback.print_exc()
                    # Continue to next step if possible, or break/exit
                    break # Stop processing if one step fails
            else:
                print(f"  Skipping: {name} (disabled in config).")

    except Exception as e:
        print(f"Error during Step 2 (Processing/Masks): {e}", file=sys.stderr)
        import traceback; traceback.print_exc(); sys.exit(1)

    # Final display
    plt.tight_layout()
    print("\nDisplaying plots...")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug MatConverter steps by visualizing intermediate data.")
    parser.add_argument("config_file", type=str, help="Path to the preprocessing TOML config file.")
    parser.add_argument("recording_dir", type=str, help="Path to the raw recording directory (containing LF/HF .mat files).")
    parser.add_argument("-f", "--frame", type=int, default=0, help="Frame index to visualize (default: 0).")
    parser.add_argument("-q", "--freq", type=str, default="lftx", choices=["lftx", "hftx"], help="Frequency key ('lftx' or 'hftx') to visualize (default: lftx).")
    parser.add_argument("-v", "--view", type=int, default=0, help="Spatial view index to visualize (default: 0).")

    args = parser.parse_args()

    visualize_steps(Path(args.config_file), Path(args.recording_dir), args.freq, args.frame, args.view)