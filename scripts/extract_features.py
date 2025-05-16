#!/usr/bin/env python3
"""
Feature extraction script for UoB project.

This script extracts features from MultiViewBmodeVideo data and saves them to disk.
"""

import sys
import os
import argparse
import pickle
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

# Use tomllib (Python 3.11+) or toml package
try:
    import tomllib
except ImportError:
    try:
        import toml as tomllib
    except ImportError:
        raise ImportError("Please install toml ('pip install toml') or use Python 3.11+ for tomllib.")

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

# Import UoB modules
from src.UoB.features.upsamplers import build_feature_upsampler


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract features from ultrasound data.")
    parser.add_argument(
        "recording_id", 
        type=str, 
        help="Recording ID (directory name in data/processed/)"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/features/jbu_dino16.toml",
        help="Path to TOML config for feature extraction"
    )
    parser.add_argument(
        "--frame_start", 
        type=int, 
        default=0,
        help="Starting frame index to process"
    )
    parser.add_argument(
        "--frame_end", 
        type=int, 
        default=-1,
        help="Ending frame index to process (inclusive, -1 for all frames)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for processing (cuda/cpu)"
    )
    return parser.parse_args()


def ensure_directory(path):
    """Ensure directory exists, create if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def main():
    """Main function for feature extraction."""
    args = parse_args()
    device = torch.device(args.device)
    
    # Set up paths
    recording_path = Path("data/processed") / args.recording_id
    data_path = recording_path / "combined_mvbv.pkl"
    config_path = Path(args.config)
    
    # Get feature set name from config filename (without extension)
    feature_set_name = config_path.stem
    features_dir = recording_path / "features" / feature_set_name
    
    print(f"Recording ID: {args.recording_id}")
    print(f"Data Path: {data_path}")
    print(f"Config Path: {config_path}")
    print(f"Feature Set Name: {feature_set_name}")
    print(f"Features Output Dir: {features_dir}")
    print(f"Device: {device}")
    
    # Validate paths
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Create output directory if it doesn't exist
    ensure_directory(features_dir)
    
    # Load data
    print(f"Loading data from {data_path}...")
    try:
        with open(data_path, 'rb') as f:
            mvbv_data = pickle.load(f)
        lftx_images = mvbv_data['lftx'].view_images  # Shape [T, 8, H, W]
        hftx_images = mvbv_data['hftx'].view_images  # Shape [T, 8, H, W]
        num_frames = min(lftx_images.shape[0], hftx_images.shape[0])
        num_lf_views = lftx_images.shape[1]
        num_hf_views = hftx_images.shape[1]
        print(f"Data loaded. LF shape: {lftx_images.shape}, HF shape: {hftx_images.shape}")
        print(f"Number of frames: {num_frames}")
    except Exception as e:
        raise RuntimeError(f"Failed to load or parse pickle file {data_path}: {e}")
    
    # Determine frame range
    frame_start = args.frame_start
    frame_end = args.frame_end if args.frame_end >= 0 else num_frames - 1
    if frame_start < 0 or frame_start >= num_frames:
        raise ValueError(f"Invalid frame_start: {frame_start}. Valid range: 0-{num_frames-1}")
    if frame_end < frame_start or frame_end >= num_frames:
        raise ValueError(f"Invalid frame_end: {frame_end}. Valid range: {frame_start}-{num_frames-1}")
    
    frames_to_process = list(range(frame_start, frame_end + 1))
    print(f"Processing frames {frame_start} to {frame_end} (total: {len(frames_to_process)} frames)")
    
    # Load feature extractor/upsampler config
    print(f"Loading feature extraction config from {config_path}...")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            feature_config = tomllib.load(f)
        print(f"Config loaded: {feature_config}")
    except Exception as e:
        raise RuntimeError(f"Failed to load feature config: {e}")
    
    # Build feature extractor
    print(f"Building feature extractor '{feature_config['name']}'...")
    try:
        extractor = build_feature_upsampler(feature_config)
        extractor.to(device)
        extractor.eval()
        print("Feature extractor built and moved to device.")
        transform = extractor.get_preprocessing_transform()
    except Exception as e:
        raise RuntimeError(f"Failed to build feature extractor: {e}")
    
    # Setup for processing
    total_processed = 0
    start_time = time.time()
    
    # Process all frames
    for frame_idx in tqdm(frames_to_process, desc="Processing frames"):
        # Process LF views
        for view_idx in range(num_lf_views):
            output_path = features_dir / f"frame{frame_idx}_view{view_idx}_lf.pt"
            
            # Skip if feature file already exists
            if output_path.exists():
                print(f"Skipping existing feature file: {output_path}")
                continue
            
            # Get image, preprocess, and extract features
            with torch.no_grad():
                original_img = lftx_images[frame_idx, view_idx]
                img_for_transform = original_img.unsqueeze(0) if original_img.ndim == 2 else original_img
                preprocessed_img = transform(img_for_transform)
                input_tensor = preprocessed_img.unsqueeze(0).to(device)
                features = extractor(input_tensor)
                
                # Save features to disk
                torch.save(features.cpu(), output_path)
                total_processed += 1
        
        # Process HF views
        for view_idx in range(num_hf_views):
            output_path = features_dir / f"frame{frame_idx}_view{view_idx}_hf.pt"
            
            # Skip if feature file already exists
            if output_path.exists():
                print(f"Skipping existing feature file: {output_path}")
                continue
            
            # Get image, preprocess, and extract features
            with torch.no_grad():
                original_img = hftx_images[frame_idx, view_idx]
                img_for_transform = original_img.unsqueeze(0) if original_img.ndim == 2 else original_img
                preprocessed_img = transform(img_for_transform)
                input_tensor = preprocessed_img.unsqueeze(0).to(device)
                features = extractor(input_tensor)
                
                # Save features to disk
                torch.save(features.cpu(), output_path)
                total_processed += 1
    
    # Print summary
    end_time = time.time()
    elapsed_time = end_time - start_time
    features_per_second = total_processed / elapsed_time if elapsed_time > 0 else 0
    
    print(f"\nFeature extraction complete!")
    print(f"Processed {total_processed} feature maps in {elapsed_time:.2f} seconds")
    print(f"Average processing speed: {features_per_second:.2f} features per second")
    print(f"Features saved to: {features_dir}")


if __name__ == "__main__":
    main() 