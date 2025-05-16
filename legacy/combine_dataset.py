import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(project_root))
sys.path.append(str(project_root.parent))

import pickle
import numpy as np
from typing import Dict, List

def combine_pkl_files(dataset_dir: str, output_file: str = None) -> Dict:
    """
    Combine all .pkl files in a dataset directory into a single dictionary.
    
    Args:
        dataset_dir: Path to the directory containing .pkl files
        output_file: Optional path to save the combined data. If None, no file is saved.
    
    Returns:
        Dictionary containing the combined data from all .pkl files
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise ValueError(f"Dataset directory {dataset_dir} does not exist")
    
    # Find all .pkl files in the directory except combined_data.pkl
    pkl_files = [f for f in dataset_dir.glob('*.pkl') if f.name != 'combined_data.pkl']
    if not pkl_files:
        raise ValueError(f"No .pkl files found in {dataset_dir}")
    
    # Sort files based on numeric part of filename
    def get_file_number(file_path):
        return int(file_path.stem.split('_')[0])
    
    pkl_files = sorted(pkl_files, key=get_file_number)
    print(f"Found {len(pkl_files)} .pkl files to combine")
    
    # Initialize combined data structure
    combined_data = {
        'lftx': {
            'view_images': []
        },
        'hftx': {
            'view_images': []
        }
    }
    
    # Load and combine data from each file
    for pkl_file in pkl_files:
        print(f"Processing {pkl_file.name}...")
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            
            # Combine LFTX data
            if 'lftx' in data:
                combined_data['lftx']['view_images'].append(data['lftx'].view_images)
            
            # Combine HFTX data
            if 'hftx' in data:
                combined_data['hftx']['view_images'].append(data['hftx'].view_images)
    
    # Convert lists to numpy arrays
    for tx_type in ['lftx', 'hftx']:
        if combined_data[tx_type]['view_images']:
            combined_data[tx_type]['view_images'] = np.concatenate(combined_data[tx_type]['view_images'], axis=0)
    
    # Save combined data if output file is specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(combined_data, f)
        print(f"Combined data saved to {output_path}")
    
    return combined_data

if __name__ == '__main__':
    # Example usage
    dataset_dir = 'dataset/recording_2022-08-17_trial2-arm'
    output_file = 'dataset/recording_2022-08-17_trial2-arm/combined_data.pkl'
    
    # Combine all .pkl files
    combined_data = combine_pkl_files(dataset_dir, output_file)
    
    # Print some information about the combined dataset
    print("\nCombined Dataset Information:")
    for tx_type in ['lftx', 'hftx']:
        if combined_data[tx_type]['view_images'].size > 0:  # Check if array has elements
            print(f"\n{tx_type.upper()} Information:")
            print(f"Number of frames: {combined_data[tx_type]['view_images'].shape[0]}")
            print(f"Number of views: {combined_data[tx_type]['view_images'].shape[1]}")
            print(f"Frame shape: {combined_data[tx_type]['view_images'].shape[2:]}")
