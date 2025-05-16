# scripts/inspect_mvbv.py
import pickle
import sys
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np

# --- Add project root to sys.path to find src.UoB ---
try:
    # Assumes the script is run from the project root (e.g., python scripts/inspect_mvbv.py ...)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    SRC_PATH = PROJECT_ROOT / 'src'
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT)) # Add project root for 'import src.UoB...'
    print(f"Added PROJECT_ROOT to sys.path: {PROJECT_ROOT}")

    # Import the necessary class definition *after* modifying sys.path
    # Make sure this matches the actual class name and location used during pickling
    from src.UoB.data.formats import MultiViewBmodeVideo
    print("Successfully imported MultiViewBmodeVideo.")

except ImportError as e:
    print(f"Error: Could not import MultiViewBmodeVideo.", file=sys.stderr)
    print(f"Ensure the class definition exists at 'src/UoB/data/formats.py' and you have run necessary installations.", file=sys.stderr)
    print(f"Import error details: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error setting up sys.path or initial imports: {e}", file=sys.stderr)
    sys.exit(1)


def main(pkl_path: Path, frame_index: int, freq_key: str, view_index: int):
    """Loads and inspects the MultiViewBmodeVideo data."""

    if not pkl_path.is_file():
        print(f"Error: File not found at {pkl_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading data from: {pkl_path}")
    try:
        with open(pkl_path, 'rb') as f:
            # The structure is expected to be {'lftx': mvbv_object, 'hftx': mvbv_object}
            combined_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n--- Basic Info ---")
    if not isinstance(combined_data, dict):
        print("Error: Loaded data is not a dictionary.", file=sys.stderr)
        sys.exit(1)

    available_freqs = list(combined_data.keys())
    print(f"Available Frequency Keys: {available_freqs}")

    if not available_freqs:
        print("Error: No frequency data found in the dictionary.", file=sys.stderr)
        sys.exit(1)

    # Use the first available frequency key to get representative metadata
    # (assuming metadata like frame count is consistent)
    rep_key = available_freqs[0]
    if not isinstance(combined_data[rep_key], MultiViewBmodeVideo):
         print(f"Error: Object for key '{rep_key}' is not a MultiViewBmodeVideo instance.", file=sys.stderr)
         sys.exit(1)

    rep_mvbv = combined_data[rep_key]
    frame_count = getattr(rep_mvbv, 'n_frame', 'N/A')
    num_spatial_views = getattr(rep_mvbv, 'n_view', 'N/A')
    image_shape = getattr(rep_mvbv, 'image_shape', 'N/A')

    print(f"Frame Count: {frame_count}")
    print(f"Spatial Views (per freq): {num_spatial_views}")
    print(f"Image Shape (h, w): {image_shape}")

    # --- Visualization ---
    print(f"\n--- Visualizing ---")
    print(f"Requested: Frame={frame_index}, Freq={freq_key}, Spatial View={view_index}")

    if freq_key not in combined_data:
        print(f"Error: Requested frequency '{freq_key}' not found in data. Available: {available_freqs}", file=sys.stderr)
        sys.exit(1)

    mvbv_to_show = combined_data[freq_key]
    
    # Validate indices
    fc = getattr(mvbv_to_show, 'n_frame', 0)
    nsv = getattr(mvbv_to_show, 'n_view', 0)
    if not (isinstance(fc, int) and 0 <= frame_index < fc):
         print(f"Error: Invalid frame index {frame_index}. Max is {fc-1}.", file=sys.stderr)
         sys.exit(1)
    if not (isinstance(nsv, int) and 0 <= view_index < nsv):
         print(f"Error: Invalid spatial view index {view_index}. Max is {nsv-1}.", file=sys.stderr)
         sys.exit(1)

    # Access the image data
    view_images_tensor = getattr(mvbv_to_show, 'view_images', None)
    if view_images_tensor is None:
        print("Error: Could not find 'view_images' attribute.", file=sys.stderr)
        sys.exit(1)

    try:
        # Select frame/view and convert tensor to numpy
        # Assuming tensor shape [n_frame, n_view, h, w]
        frame_data_tensor = view_images_tensor[frame_index, view_index, :, :]
        # Check if it's a torch tensor and move to cpu/convert
        if hasattr(frame_data_tensor, 'cpu') and hasattr(frame_data_tensor, 'numpy'):
             frame_data_np = frame_data_tensor.cpu().numpy()
             print("Converted Torch tensor to NumPy.")
        elif isinstance(frame_data_tensor, np.ndarray):
             frame_data_np = frame_data_tensor # Already numpy
             print("Data is already NumPy array.")
        else:
             print(f"Warning: Unexpected data type for frame: {type(frame_data_tensor)}. Attempting direct plot.", file=sys.stderr)
             frame_data_np = frame_data_tensor # Try plotting directly

        # Display the image
        plt.figure(figsize=(8, 8))
        # print(frame_data_np)
        plt.imshow(frame_data_np, cmap='gray') # Use grayscale colormap
        plt.title(f"Recording: {pkl_path.parent.name}\nFrame: {frame_index}, Freq: {freq_key}, Spatial View: {view_index}")
        plt.xlabel("Width (pixels)")
        plt.ylabel("Height (pixels)")
        plt.colorbar(label='Intensity')
        print("Displaying plot...")
        plt.show()

    except Exception as e:
        print(f"Error during image data access or plotting: {e}", file=sys.stderr)
        print(f"Expected tensor shape: ({fc}, {nsv}, H, W)")
        print(f"Shape of view_images attribute: {getattr(view_images_tensor, 'shape', 'N/A')}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect and visualize data from a combined_mvbv.pkl file.")
    parser.add_argument("pkl_file", type=str, help="Path to the combined_mvbv.pkl file.")
    parser.add_argument("-f", "--frame", type=int, default=0, help="Frame index to visualize (default: 0).")
    parser.add_argument("-q", "--freq", type=str, default="lftx", choices=["lftx", "hftx"], help="Frequency key ('lftx' or 'hftx') to visualize (default: lftx).")
    parser.add_argument("-v", "--view", type=int, default=0, help="Spatial view index to visualize (default: 0).")

    args = parser.parse_args()

    main(Path(args.pkl_file), args.frame, args.freq, args.view) 