"""
Preprocess raw recording data (.mat files) into combined .pkl files.

Usage:
  python preprocess_data.py                    # Interactive selection from data/raw/
  python preprocess_data.py --input_dir /path  # Specify input directory directly
  python preprocess_data.py --help             # Show all options

Dependencies:
  pip install inquirer  # For interactive CLI selection

The script looks for directories containing *_LF.mat and *_HF.mat files in data/raw/
"""

import argparse
from pathlib import Path
import sys
import os

# Add project root to the Python path to find the src module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.UoB.preprocessing.mat_converter import MatConverter

def get_available_raw_recordings(raw_data_dir: str = "data/raw") -> list:
    """
    Get a list of available raw recording directories from the raw data directory.
    Looks for directories containing *_LF.mat and *_HF.mat files.
    """
    raw_path = Path(raw_data_dir)
    if not raw_path.exists():
        print(f"Raw data directory {raw_data_dir} does not exist!")
        return []
    
    recordings = []
    for item in raw_path.iterdir():
        if item.is_dir():
            # Check if it contains .mat files (specifically LF and HF files)
            mat_files = list(item.glob("*.mat"))
            lf_files = list(item.glob("*_LF.mat"))
            hf_files = list(item.glob("*_HF.mat"))
            
            if lf_files and hf_files:
                recordings.append(item.name)
            elif mat_files:
                print(f"Warning: Directory {item.name} contains .mat files but no *_LF.mat or *_HF.mat files")
            else:
                print(f"Info: Directory {item.name} exists but doesn't contain .mat files")
    
    return sorted(recordings)


def select_recording_interactively() -> str:
    """
    Present an interactive menu to select a raw recording using arrow keys.
    """
    try:
        import inquirer
    except ImportError:
        print("Error: 'inquirer' package is required for interactive selection.")
        print("Install it with: pip install inquirer")
        sys.exit(1)
    
    recordings = get_available_raw_recordings()
    
    if not recordings:
        print("No valid raw recordings found in data/raw/")
        print("Make sure the directory exists and contains folders with *_LF.mat and *_HF.mat files.")
        sys.exit(1)
    
    questions = [
        inquirer.List('recording',
                     message="Select a raw recording to preprocess",
                     choices=recordings,
                     carousel=True)  # Allow wrapping around with arrow keys
    ]
    
    answers = inquirer.prompt(questions)
    if answers is None:  # User pressed Ctrl+C
        print("\nOperation cancelled.")
        sys.exit(0)
    
    return answers['recording']


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


def main_user_selection():
    """
    Interactive version of main() that uses arrow key selection instead of command line arguments.
    """
    # Get recording selection interactively
    recording_name = select_recording_interactively()
    
    # Set up paths
    input_path = Path("data/raw") / recording_name
    output_path = Path("data/processed")
    config_path = Path("configs/preprocessing/default.toml")
    output_filename = "combined_mvbv.pkl"
    
    print(f"Selected recording: {recording_name}")
    print(f"Input directory: {input_path}")
    print(f"Output directory: {output_path}")
    print(f"Config file: {config_path}")
    print(f"Output filename: {output_filename}")
    
    # Validation
    if not input_path.is_dir():
        print(f"Error: Input directory not found: {input_path}")
        sys.exit(1)
        
    if not config_path.is_file():
        print(f"Error: Config file not found: {config_path}")
        print("Make sure the config file exists. You can use default settings.")
        sys.exit(1)

    try:
        print(f"Initializing MatConverter with config: {config_path}")
        converter = MatConverter(config_path=config_path)
        
        print(f"Starting conversion...")
        converter.convert_recording(
            recording_dir=input_path,
            output_dir=output_path,
            output_filename=output_filename
        )
        print(f"Conversion finished successfully.")
        print(f"Output saved to: {output_path / recording_name / output_filename}")
        
    except Exception as e:
        print(f"An error occurred during conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Check if command line arguments are provided
    if len(sys.argv) > 1:
        # Use original argparse-based main function
        main()
    else:
        # Use interactive selection
        main_user_selection() 