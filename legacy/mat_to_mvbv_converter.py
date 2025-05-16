import os
import pickle
from pathlib import Path
from typing import Dict, Optional

# Import UoB modules needed for unpickling
# import UoB
from data import bmode, process
from data import vsx_mat as mat

from data.multiview_bmode import MultiViewBmodeVideo, Bmode2MultiViewBmodeVideo
from data.bmode import BmodeConfig

class MatToMVBVConverter:
    """Converts a directory of mat files to MultiViewBmodeVideo objects."""
    
    def __init__(
        self,
        mat_dir: str,
        bmode_config_paths: Dict[str, str],
        output_dir: Optional[str] = None
    ):
        """
        Initialize the converter.
        
        Args:
            mat_dir: Directory containing pairs of HF and LF .mat files
            bmode_config_paths: Dict with paths to bmode configs for 'lftx' and 'hftx'
            output_dir: Directory to save the converted files. If None, uses mat_dir/converted/
        """
        self.mat_dir = Path(mat_dir)
        self.output_dir = Path(output_dir) if output_dir else self.mat_dir / 'converted'
        self.bmode_config_paths = bmode_config_paths
        
        # Load bmode configs
        self.bmode_configs = {}
        for key in ['lftx', 'hftx']:
            try:
                with open(bmode_config_paths[key], 'rb') as f:
                    self.bmode_configs[key] = pickle.load(f)
            except ModuleNotFoundError as e:
                print(f"Error loading bmode config for {key}: {e}")
                print("Make sure UoB module is in your PYTHONPATH")
                raise
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _get_mat_pairs(self):
        """Get pairs of HF and LF mat files from the directory."""
        pairs = []
        
        # List all mat files
        mat_files = sorted([f for f in os.listdir(self.mat_dir) if f.endswith('.mat')])
        
        # Group them by number
        for i in range(1, len(mat_files) // 2 + 1):
            hf_file = f"{i}_HF.mat"
            lf_file = f"{i}_LF.mat"
            
            if hf_file in mat_files and lf_file in mat_files:
                pairs.append({
                    'hftx': self.mat_dir / hf_file,
                    'lftx': self.mat_dir / lf_file,
                    'index': i
                })
        
        return pairs
    
    def convert_pair(self, pair: Dict):
        """Convert a pair of mat files to MultiViewBmodeVideo objects."""
        # Load mat data
        mat_data = {}
        for key in ['lftx', 'hftx']:
            mat_data[key] = mat.MatDataLoader(str(pair[key])).build_mat_data()
        
        # Convert to B-mode
        b_mode = {}
        for key in mat_data:
            b_mode[key] = bmode.BmodeBuilder(
                mat_data=mat_data[key],
                config=self.bmode_configs[key]
            ).build_b_mode()
        
        # Convert to MultiViewBmodeVideo
        mvbvs = {}
        for key in ['lftx', 'hftx']:
            mvbvs[key] = Bmode2MultiViewBmodeVideo(b_mode[key]).convert(
                mat_file_dir=str(self.mat_dir),
                bmode_config_path=self.bmode_configs[key],
            )
        
        return mvbvs
    
    # def save_mvbvs(self, mvbvs: Dict, index: int):
    #     """Save MultiViewBmodeVideo objects to files."""
    #     for key in mvbvs:
    #         output_path = self.output_dir / f"{index}_{key}_mvbv.pkl"
    #         with open(output_path, 'wb') as f:
    #             pickle.dump(mvbvs[key], f)
    
    def save_mvbvs(self, mvbvs: Dict, index: int):
        """Save MultiViewBmodeVideo objects to a single file."""
        output_path = self.output_dir / f"{index}_mvbv.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(mvbvs, f)
        print(f"Saved mvbvs dictionary to {output_path}")


    def convert_all(self):
        """Convert all mat file pairs in the directory."""
        pairs = self._get_mat_pairs()
        
        for pair in pairs:
            print(f"Converting pair {pair['index']}...")
            mvbvs = self.convert_pair(pair)
            self.save_mvbvs(mvbvs, pair['index'])
            print(f"Saved pair {pair['index']}")


def convert_mat_directory(
    mat_dir: str,
    bmode_config_paths: Dict[str, str],
    output_dir: Optional[str] = None
):
    """
    Convenience function to convert a directory of mat files.
    
    Args:
        mat_dir: Directory containing pairs of HF and LF .mat files
        bmode_config_paths: Dict with paths to bmode configs for 'lftx' and 'hftx'
        output_dir: Directory to save the converted files. If None, uses mat_dir/converted/
    """
    converter = MatToMVBVConverter(mat_dir, bmode_config_paths, output_dir)
    converter.convert_all() 