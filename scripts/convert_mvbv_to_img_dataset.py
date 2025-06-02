import argparse
import logging
from pathlib import Path
import pickle
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import csv

# --- Add project root to sys.path to find src.UoB ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    SRC_PATH = PROJECT_ROOT / 'src'
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.UoB.data.formats import MultiViewBmodeVideo
except ImportError as e:
    print(f"Error: Could not import MultiViewBmodeVideo.", file=sys.stderr)
    print(f"Ensure the class definition exists at 'src/UoB/data/formats.py' and you have run necessary installations.", file=sys.stderr)
    print(f"Import error details: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error setting up sys.path or initial imports: {e}", file=sys.stderr)
    sys.exit(1)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Convert combined_mvbv.pkl to an image dataset.")
    parser.add_argument("pkl_file", type=str, help="Path to the combined_mvbv.pkl file.")
    return parser.parse_args()


def get_output_paths(pkl_path: Path) -> dict:
    """Determine output directories for images and ROI masks based on the input file's parent folder name."""
    recording_name = pkl_path.parent.name
    root = Path("data/formatted") / recording_name
    images = root / "images"
    roi_masks = root / "roi_masks"
    return {"root": root, "images": images, "roi_masks": roi_masks}


def load_mvbv_data(pkl_path: Path) -> dict:
    """Load the pickle file and return the frequency-keyed dictionary of MultiViewBmodeVideo objects."""
    logging.info(f"Loading data from: {pkl_path}")
    if not pkl_path.is_file():
        logging.error(f"File not found: {pkl_path}")
        sys.exit(1)
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            logging.error("Loaded data is not a dictionary.")
            sys.exit(1)
        logging.info(f"Loaded frequency keys: {list(data.keys())}")
        return data
    except Exception as e:
        logging.error(f"Error loading pickle file: {e}")
        sys.exit(1)


def ensure_directories_exist(output_dirs: dict):
    """Create output directories if they do not exist."""
    for key, path in output_dirs.items():
        if key == "root":
            continue
        path.mkdir(parents=True, exist_ok=True)


def save_png(array, path):
    """Save a numpy array or torch tensor as a PNG file."""
    if hasattr(array, 'cpu') and hasattr(array, 'numpy'):
        array = array.cpu().numpy()
    # Normalize to 0-255 for uint8 PNG
    arr_min, arr_max = np.min(array), np.max(array)
    if arr_max > arr_min:
        norm = (array - arr_min) / (arr_max - arr_min)
    else:
        norm = np.zeros_like(array)
    img_uint8 = (norm * 255).astype(np.uint8)
    im = Image.fromarray(img_uint8)
    im.save(path)


def extract_and_save_images(mvbv_dict: dict, output_dirs: dict):
    """Iterate over all frequency keys, frames, and views, extract image data, and save as PNGs in the images/ folder."""
    images_dir = output_dirs["images"]
    for freq_key, mvbv in mvbv_dict.items():
        n_frame = getattr(mvbv, 'n_frame', 0)
        n_view = getattr(mvbv, 'n_view', 0)
        view_images = getattr(mvbv, 'view_images', None)
        if view_images is None:
            logging.error(f"No 'view_images' attribute for frequency {freq_key}.")
            continue
        logging.info(f"Saving images for frequency '{freq_key}' ({n_frame} frames, {n_view} views)...")
        for frame_idx in tqdm(range(n_frame), desc=f"{freq_key} frames"):
            for view_idx in range(n_view):
                try:
                    img = view_images[frame_idx, view_idx, :, :]
                    out_name = f"f{frame_idx}_v{view_idx}_{freq_key}.png"
                    out_path = images_dir / out_name
                    save_png(img, out_path)
                except Exception as e:
                    logging.error(f"Error saving image for frame {frame_idx}, view {view_idx}, freq {freq_key}: {e}")


def extract_and_save_roi_masks(mvbv_dict: dict, output_dirs: dict):
    """For each frequency key, extract the ROI mask for each view (from any frame), and save as PNGs in the roi_masks/ folder."""
    roi_dir = output_dirs["roi_masks"]
    for freq_key, mvbv in mvbv_dict.items():

        # import pdb; pdb.set_trace()

        n_view = getattr(mvbv, 'n_view', 0)
        # Try to get roi_masks attribute (should be [n_view, h, w] or similar)
        roi_masks = getattr(mvbv, 'view_masks', None)
        if roi_masks is None:
            logging.warning(f"No 'roi_masks' attribute for frequency {freq_key}. Skipping.")
            continue
        logging.info(f"Saving ROI masks for frequency '{freq_key}' ({n_view} views)...")
        for view_idx in tqdm(range(n_view), desc=f"{freq_key} roi views"):
            try:
                # Roi_masks is [1, n_view, h, w], just index by view
                mask = roi_masks[0, view_idx]
                out_name = f"f0_v{view_idx}_{freq_key}.png"  # f0 is arbitrary since mask is same for all frames
                out_path = roi_dir / out_name
                save_png(mask, out_path)
            except Exception as e:
                logging.error(f"Error saving ROI mask for view {view_idx}, freq {freq_key}: {e}")


def generate_frame_list_csv(mvbv_dict: dict, output_dirs: dict):
    """Generate a CSV file listing all frames with metadata: frame_name, tx_mode, frame, view, annotated."""
    csv_path = output_dirs["root"] / "list_frames.csv"
    images_dir = output_dirs["images"]
    rows = []
    for freq_key, mvbv in mvbv_dict.items():
        n_frame = getattr(mvbv, 'n_frame', 0)
        n_view = getattr(mvbv, 'n_view', 0)
        for frame_idx in range(n_frame):
            for view_idx in range(n_view):
                frame_name = f"f{frame_idx}_v{view_idx}_{freq_key}.png"
                rows.append({
                    "frame_name": frame_name,
                    "tx_mode": freq_key,
                    "frame": frame_idx,
                    "view": view_idx,
                    "annotated": False,
                })
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ["frame_name", "tx_mode", "frame", "view", "annotated"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logging.info(f"Frame list CSV generated at: {csv_path}")


def generate_empty_matched_points_csv(output_dirs: dict):
    """Generate an empty matched_points.csv file with the required headers."""
    csv_path = output_dirs["root"] / "matched_points.csv"
    fieldnames = ["frame_name", "tx_mode", "view", "x", "y", "point", "frame"]
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    logging.info(f"Empty matched_points.csv generated at: {csv_path}")


def main():
    """Main entry point for the script."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    args = parse_args()
    pkl_path = Path(args.pkl_file)
    logging.info("Starting conversion of combined_mvbv.pkl to image dataset...")
    output_dirs = get_output_paths(pkl_path)
    ensure_directories_exist(output_dirs)
    mvbv_dict = load_mvbv_data(pkl_path)
    extract_and_save_images(mvbv_dict, output_dirs)
    extract_and_save_roi_masks(mvbv_dict, output_dirs)
    generate_frame_list_csv(mvbv_dict, output_dirs)
    generate_empty_matched_points_csv(output_dirs)
    logging.info(f"Finished conversion. Dataset saved to: {output_dirs['root']}")


if __name__ == "__main__":
    main()
