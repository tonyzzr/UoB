import argparse
import logging
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import csv
import warnings
from typing import List, Dict, Tuple

# --- Add project root to sys.path to find src.UoB ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    SRC_PATH = PROJECT_ROOT / 'src'
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.UoB.preprocessing.mat_converter import MatConverter
    from src.UoB.data.readers import RecordingLoader
    from src.UoB.data.formats import MultiViewBmodeVideo, MatData
    from src.UoB.data.readers import MatDataLoader
except ImportError as e:
    print(f"Error: Could not import required UoB modules.", file=sys.stderr)
    print(f"Ensure the class definitions exist and you have run necessary installations.", file=sys.stderr)
    print(f"Import error details: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error setting up sys.path or initial imports: {e}", file=sys.stderr)
    sys.exit(1)


class MatConverterMemEfficient:
    """Memory-efficient version that processes .mat files one by one instead of loading all into memory."""
    
    def __init__(self, config_path: str | Path):
        """Initialize with the same config as regular MatConverter."""
        # Use the existing MatConverter for its configuration logic
        self.base_converter = MatConverter(config_path)
        self.config_path = Path(config_path)
        
    def find_mat_file_pairs(self, recording_dir: Path) -> List[Dict[str, Path]]:
        """Find pairs of HF/LF .mat files in the recording directory."""
        if not recording_dir.is_dir():
            raise FileNotFoundError(f"Recording directory not found: {recording_dir}")
        
        all_files = list(recording_dir.glob('*.mat'))
        files_dict: Dict[int, Dict[str, Path]] = {}
        
        for f in all_files:
            name = f.stem  # Filename without extension
            parts = name.split('_')
            if len(parts) == 2 and parts[0].isdigit() and parts[1] in ['HF', 'LF']:
                index = int(parts[0])
                ftype = parts[1].lower()  # 'hf' or 'lf'
                if index not in files_dict:
                    files_dict[index] = {}
                files_dict[index][f'{ftype}_path'] = f
            else:
                warnings.warn(f"Ignoring file with unexpected name format: {f.name}")
        
        # Convert to list of pairs, sorted by index
        pairs = []
        sorted_indices = sorted(files_dict.keys())
        for i in sorted_indices:
            if 'hf_path' in files_dict[i] and 'lf_path' in files_dict[i]:
                pairs.append({
                    'index': i,
                    'hf_path': files_dict[i]['hf_path'],
                    'lf_path': files_dict[i]['lf_path']
                })
            else:
                warnings.warn(f"Incomplete HF/LF pair for index {i}, skipping.")
        
        logging.info(f"Found {len(pairs)} complete HF/LF pairs")
        return pairs
    
    def load_single_mat_file(self, mat_path: Path) -> MatData:
        """Load a single .mat file using MatDataLoader."""
        with MatDataLoader(mat_path) as loader:
            return loader.build_mat_data()
    
    def process_single_file_pair(self, pair: Dict[str, Path], recording_id: str, global_frame_offset: int, 
                                output_dirs: dict, csv_rows: list) -> int:
        """
        Process a single pair of HF/LF .mat files and save images immediately.
        Returns the number of frames processed.
        """
        pair_index = pair['index']
        logging.info(f"Processing file pair {pair_index}: {pair['lf_path'].name} & {pair['hf_path'].name}")
        
        try:
            # Load individual .mat files
            lf_mat_data = self.load_single_mat_file(pair['lf_path'])
            hf_mat_data = self.load_single_mat_file(pair['hf_path'])
            
            # Process each frequency
            lf_success, lf_frames = self.process_single_frequency_from_pair(
                lf_mat_data, self.base_converter.lf_bmode_config, 'lftx', 
                recording_id, global_frame_offset, output_dirs, csv_rows
            )
            
            hf_success, hf_frames = self.process_single_frequency_from_pair(
                hf_mat_data, self.base_converter.hf_bmode_config, 'hftx', 
                recording_id, global_frame_offset, output_dirs, csv_rows
            )
            
            if not lf_success or not hf_success:
                logging.error(f"Failed to process pair {pair_index}")
                return 0
                
            if lf_frames != hf_frames:
                logging.warning(f"Frame count mismatch in pair {pair_index}: LF={lf_frames}, HF={hf_frames}")
            
            return lf_frames  # Return number of frames processed
            
        except Exception as e:
            logging.error(f"Error processing file pair {pair_index}: {e}")
            return 0
    
    def process_single_frequency_from_pair(self, mat_data: MatData, bmode_config, freq_key: str, 
                                          recording_id: str, global_frame_offset: int, 
                                          output_dirs: dict, csv_rows: list) -> Tuple[bool, int]:
        """Process a single frequency from a mat file pair and save images immediately."""
        try:
            # Process using the base converter's methods
            processed_bmode = self.base_converter._process_single_frequency(mat_data, bmode_config)
            if processed_bmode is None:
                logging.error(f"Processing failed for {freq_key} data.")
                return False, 0
                
            # Convert to MultiViewBmodeVideo
            mvbv = self.base_converter._convert_bmode_to_mvbv(processed_bmode, recording_id)
            if mvbv is None:
                logging.error(f"Conversion to MVBV failed for {freq_key} data.")
                return False, 0
            
            import pdb; pdb.set_trace()
            
            # Extract and save images with adjusted frame indices
            n_frames = self.extract_and_save_images_with_offset(
                mvbv, freq_key, global_frame_offset, output_dirs["images"]
            )
            
            # Extract and save ROI masks (only need to do this once per frequency, not per file)
            if global_frame_offset == 0:  # Only save masks for the first file
                self.extract_and_save_roi_masks_from_mvbv(mvbv, freq_key, output_dirs["roi_masks"])
            
            # Add frame entries to CSV list with adjusted frame indices
            self.add_frames_to_csv_list_with_offset(
                mvbv, freq_key, global_frame_offset, csv_rows
            )
            
            return True, n_frames
            
        except Exception as e:
            logging.error(f"Error processing {freq_key} data: {e}")
            return False, 0
    
    def extract_and_save_images_with_offset(self, mvbv: MultiViewBmodeVideo, freq_key: str, 
                                           frame_offset: int, images_dir: Path) -> int:
        """Extract and save images with global frame indexing."""
        n_frame = mvbv.n_frame
        n_view = mvbv.n_view
        view_images = mvbv.view_images
        
        for frame_idx in range(n_frame):
            global_frame_idx = frame_offset + frame_idx
            for view_idx in range(n_view):
                try:
                    img = view_images[frame_idx, view_idx, :, :]
                    out_name = f"f{global_frame_idx}_v{view_idx}_{freq_key}.png"
                    out_path = images_dir / out_name
                    save_png(img, out_path)
                except Exception as e:
                    logging.error(f"Error saving image for frame {global_frame_idx}, view {view_idx}, freq {freq_key}: {e}")
        
        return n_frame
    
    def extract_and_save_roi_masks_from_mvbv(self, mvbv: MultiViewBmodeVideo, freq_key: str, roi_dir: Path):
        """Extract and save ROI masks from a MultiViewBmodeVideo object."""
        n_view = mvbv.n_view
        view_masks = mvbv.view_masks
        
        for view_idx in range(n_view):
            try:
                # view_masks is [1, n_view, h, w], just index by view
                mask = view_masks[0, view_idx]
                out_name = f"f0_v{view_idx}_{freq_key}.png"  # f0 is arbitrary since mask is same for all frames
                out_path = roi_dir / out_name
                save_png(mask, out_path)
            except Exception as e:
                logging.error(f"Error saving ROI mask for view {view_idx}, freq {freq_key}: {e}")
    
    def add_frames_to_csv_list_with_offset(self, mvbv: MultiViewBmodeVideo, freq_key: str, 
                                          frame_offset: int, csv_rows: list):
        """Add frame entries to the CSV rows list with global frame indexing."""
        n_frame = mvbv.n_frame
        n_view = mvbv.n_view
        
        for frame_idx in range(n_frame):
            global_frame_idx = frame_offset + frame_idx
            for view_idx in range(n_view):
                frame_name = f"f{global_frame_idx}_v{view_idx}_{freq_key}.png"
                csv_rows.append({
                    "frame_name": frame_name,
                    "tx_mode": freq_key,
                    "frame": global_frame_idx,
                    "view": view_idx,
                    "annotated": False,
                })


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess raw .mat files and convert directly to image dataset.")
    parser.add_argument("input_dir", type=str, help="Path to the raw recording directory containing *_LF.mat and *_HF.mat files.")
    parser.add_argument("--output_dir", type=str, default="data/formatted", help="Base directory where the dataset will be created.")
    parser.add_argument("--config_path", type=str, default="configs/preprocessing/default.toml", help="Path to the TOML configuration file for preprocessing.")
    return parser.parse_args()


def get_output_paths(input_dir: Path, output_base: Path) -> dict:
    """Determine output directories for images and ROI masks based on the input directory name."""
    recording_name = input_dir.name
    root = output_base / recording_name
    images = root / "images"
    roi_masks = root / "roi_masks"
    return {"root": root, "images": images, "roi_masks": roi_masks}


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


def extract_and_save_images_from_mvbv(mvbv: MultiViewBmodeVideo, freq_key: str, images_dir: Path):
    """Extract and save images from a MultiViewBmodeVideo object."""
    n_frame = mvbv.n_frame
    n_view = mvbv.n_view
    view_images = mvbv.view_images
    
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


def extract_and_save_roi_masks_from_mvbv(mvbv: MultiViewBmodeVideo, freq_key: str, roi_dir: Path):
    """Extract and save ROI masks from a MultiViewBmodeVideo object."""
    n_view = mvbv.n_view
    view_masks = mvbv.view_masks
    
    logging.info(f"Saving ROI masks for frequency '{freq_key}' ({n_view} views)...")
    for view_idx in tqdm(range(n_view), desc=f"{freq_key} roi views"):
        try:
            # view_masks is [1, n_view, h, w], just index by view
            mask = view_masks[0, view_idx]
            out_name = f"f0_v{view_idx}_{freq_key}.png"  # f0 is arbitrary since mask is same for all frames
            out_path = roi_dir / out_name
            save_png(mask, out_path)
        except Exception as e:
            logging.error(f"Error saving ROI mask for view {view_idx}, freq {freq_key}: {e}")


def add_frames_to_csv_list(mvbv: MultiViewBmodeVideo, freq_key: str, csv_rows: list):
    """Add frame entries to the CSV rows list."""
    n_frame = mvbv.n_frame
    n_view = mvbv.n_view
    
    for frame_idx in range(n_frame):
        for view_idx in range(n_view):
            frame_name = f"f{frame_idx}_v{view_idx}_{freq_key}.png"
            csv_rows.append({
                "frame_name": frame_name,
                "tx_mode": freq_key,
                "frame": frame_idx,
                "view": view_idx,
                "annotated": False,
            })


def generate_frame_list_csv(csv_rows: list, output_dirs: dict):
    """Generate a CSV file listing all frames with metadata."""
    csv_path = output_dirs["root"] / "list_frames.csv"
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ["frame_name", "tx_mode", "frame", "view", "annotated"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
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
    
    input_dir = Path(args.input_dir)
    output_base = Path(args.output_dir)
    config_path = Path(args.config_path)
    
    logging.info("Starting memory-efficient conversion from raw .mat files to image dataset...")
    
    # Validate inputs
    if not input_dir.is_dir():
        logging.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
        
    if not config_path.is_file():
        logging.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    # Setup output directories
    output_dirs = get_output_paths(input_dir, output_base)
    ensure_directories_exist(output_dirs)
    recording_id = input_dir.name
    
    # Initialize memory-efficient converter
    logging.info(f"Initializing MatConverterMemEfficient with config: {config_path}")
    converter = MatConverterMemEfficient(config_path=config_path)
    
    # Find all mat file pairs
    mat_pairs = converter.find_mat_file_pairs(input_dir)
    if not mat_pairs:
        logging.error("No valid .mat file pairs found in the input directory.")
        sys.exit(1)
    
    # Initialize CSV rows list
    csv_rows = []
    total_frames_processed = 0
    
    # Process each file pair sequentially
    logging.info(f"Processing {len(mat_pairs)} file pairs...")
    for pair in tqdm(mat_pairs, desc="Processing file pairs"):
        frames_processed = converter.process_single_file_pair(
            pair, recording_id, total_frames_processed, output_dirs, csv_rows
        )
        total_frames_processed += frames_processed
    
    if total_frames_processed == 0:
        logging.error("No frames were successfully processed.")
        sys.exit(1)
    
    # Generate CSV files
    generate_frame_list_csv(csv_rows, output_dirs)
    generate_empty_matched_points_csv(output_dirs)
    
    logging.info(f"Finished conversion. Dataset saved to: {output_dirs['root']}")
    logging.info(f"Total frames processed: {total_frames_processed}")
    logging.info(f"Total image files created: {len(csv_rows)}")


if __name__ == "__main__":
    main()
