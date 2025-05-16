import argparse
from pathlib import Path
import sys
import os

# Add project root to the Python path to find the src module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.UoB.preprocessing.mat_converter import MatConverter

def main():
    parser = argparse.ArgumentParser(description="Preprocess a recording directory (.mat files) into a combined .pkl file.")
    
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True,
        help="Path to the raw recording directory containing *_LF.mat and *_HF.mat files."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/processed",
        help="Base directory where the processed output folder for the recording will be created."
    )
    parser.add_argument(
        "--config_path", 
        type=str, 
        default="configs/preprocessing/default.toml",
        help="Path to the TOML configuration file for preprocessing."
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="combined_mvbv.pkl",
        help="Name of the output pickle file."
    )

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    config_path = Path(args.config_path)

    if not input_path.is_dir():
        print(f"Error: Input directory not found: {input_path}")
        sys.exit(1)
        
    if not config_path.is_file():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    try:
        print(f"Initializing MatConverter with config: {config_path}")
        converter = MatConverter(config_path=config_path)
        
        print(f"Starting conversion...")
        converter.convert_recording(
            recording_dir=input_path,
            output_dir=output_path,
            output_filename=args.output_filename
        )
        print(f"Conversion finished successfully.")
        
    except Exception as e:
        print(f"An error occurred during conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 