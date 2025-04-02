import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(project_root))
sys.path.append(str(project_root.parent))

from mat_to_mvbv_converter import convert_mat_directory

if __name__ == '__main__':
    # Get the current directory for bmode configs
    current_dir = Path(os.getcwd())
    
    # Get the MatFiles directory path
    mat_base_dir = Path(os.path.expanduser('~/Desktop/1-Zhuorui/MIT/Code/Image_Registration'))

    # Define paths to your bmode configs (using current directory)
    bmode_config_paths = {
        'lftx': str(current_dir / 'examples/lftx_bmode_config_default.pkl'),
        'hftx': str(current_dir / 'examples/hftx_bmode_config_default.pkl')
    }

    # Define input and output directories (using mat_base_dir)
    mat_dir = str(mat_base_dir / 'data/MatFiles/2022-08-17/trial2 -arm')
    output_dir = str(current_dir / 'dataset/recording_2022-08-17_trial2-arm')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Convert all mat files in the directory
    convert_mat_directory(
        mat_dir=mat_dir,
        bmode_config_paths=bmode_config_paths,
        output_dir=output_dir
    )