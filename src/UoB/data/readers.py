import h5py
import numpy as np
from pathlib import Path
import glob
import os
from typing import Dict, Tuple, List
import warnings

# Assuming formats.py is in the same directory or path is handled
from .formats import MatData

class MatDataLoader:
    """Loads data from a single Verasonics .mat file (v7.1 format)."""
    def __init__(self, path: str | Path) -> None:
        self.filepath = Path(path)
        if not self.filepath.exists():
            raise FileNotFoundError(f"MAT file not found: {self.filepath}")
        # Use try-finally to ensure file is closed even if errors occur during loading
        self.data = None
        try:
            self.data = h5py.File(str(self.filepath), 'r')
        except OSError as e:
            raise OSError(f"Could not open HDF5 file {self.filepath}: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self.data:
            self.data.close()
            self.data = None

    def build_mat_data(self) -> MatData:
        """Extracts PData, ImgData, and Trans structures into a MatData object."""
        if not self.data:
             raise ValueError("HDF5 file is not open or already closed.")
        try:
            pdata = self.__load_pdata()
            imgdata = self.__load_imgdata()
            trans = self.__load_trans()
        except Exception as e:
             self.close() # Ensure file is closed on error
             raise RuntimeError(f"Error reading data from {self.filepath}: {e}") from e
             
        # No need to close here if called outside context manager, 
        # but good practice if used standalone. Let context manager handle it.
        # self.close()
        return MatData(pdata=pdata, imgdata=imgdata, trans=trans)

    def __load_pdata(self) -> dict:
        """Loads the PData field."""
        if 'PData' not in self.data:
             raise KeyError("'PData' field not found in HDF5 file.")
        
        PData = {}
        PDataRef = self.data['PData']

        # Check for expected structure (simple check)
        if not all(key in PDataRef for key in ['PDelta', 'Size', 'Origin']):
            raise KeyError("Missing one or more required keys ('PDelta', 'Size', 'Origin') in PData.")
            
        # Assuming PDelta determines the number of views/structures
        PDataNum = len(PDataRef['PDelta'])

        for i in range(PDataNum):
            try:
                pdelta_ref = PDataRef['PDelta'][i][0]
                size_ref = PDataRef['Size'][i][0]
                origin_ref = PDataRef['Origin'][i][0]

                PDelta = np.array(self.data[pdelta_ref]).T
                Size = np.array(self.data[size_ref]).T
                Origin = np.array(self.data[origin_ref]).T
                
                # Basic shape validation
                if PDelta.shape[1] < 3 or Origin.shape[1] < 3 or Size.shape[1] < 2:
                     warnings.warn(f"Unexpected shape in PData substructure {i} in {self.filepath}")

                PDataDict = {
                    'idx'    : i,
                    'PDelta' : PDelta[0], # Assuming first row is relevant
                    'Size'   : Size[0],
                    'Origin' : Origin[0],
                    'dx_wl'  : PDelta[0][0],
                    'dz_wl'  : PDelta[0][2],
                    'Ox_wl'  : Origin[0][0],
                    'Oz_wl'  : Origin[0][2],
                    'nx'     : int(Size[0][1]), # Ensure integer
                    'nz'     : int(Size[0][0]), # Ensure integer
                }
                PData[i] = PDataDict
            except Exception as e:
                 warnings.warn(f"Could not load PData structure {i} from {self.filepath}: {e}")
                 PData[i] = {'idx': i} # Add placeholder if loading fails?
        
        return PData

    def __load_imgdata(self) -> dict:
        """Loads the ImgData field, handling the Verasonics reference structure."""
        if 'ImgData' not in self.data:
            raise KeyError("'ImgData' field not found in HDF5 file.")
            
        ImgData = {}
        try:
            img_data_ref_container = self.data['ImgData']
            if len(img_data_ref_container.shape) != 2 or img_data_ref_container.shape[0] != 1:
                raise ValueError(f"Unexpected shape for ImgData reference container: {img_data_ref_container.shape}")
            num_views = img_data_ref_container.shape[1]
        except Exception as e:
             raise ValueError(f"Unexpected structure reading ImgData reference container in {self.filepath}: {e}")

        for i in range(num_views):
            try:
                # 1. Get the reference to the intermediate dataset
                intermediate_ref = img_data_ref_container[0, i]
                intermediate_dataset = self.data[intermediate_ref]

                # 2. Check if intermediate dataset holds the final reference
                final_data_ref = None
                if intermediate_dataset.dtype == h5py.ref_dtype:
                    # Assuming the reference is stored at [0, 0] within this dataset
                    # This might need adjustment if structure varies
                    if intermediate_dataset.shape == (1, 1) or intermediate_dataset.size == 1:
                        final_data_ref = intermediate_dataset[0, 0] 
                    else:
                         warnings.warn(f"Intermediate dataset {intermediate_dataset.name} has unexpected shape {intermediate_dataset.shape} for holding a reference.")
                         # Attempt to get ref from first element anyway?
                         final_data_ref = intermediate_dataset.item() # Try getting scalar ref
                else:
                    # If the intermediate dataset *is* the data (no double reference)
                    # This case might not occur based on observed structure, but included for robustness
                    target_dataset = intermediate_dataset
                    
                # 3. Dereference the final reference (if found)
                if final_data_ref is not None:
                    target_dataset = self.data[final_data_ref]
                elif target_dataset is None: # Check if we failed to find the target
                     raise ValueError(f"Could not resolve final data reference for view {i}")

                # 4. Convert the target dataset to a numpy array
                data = np.array(target_dataset)
                ImgData[i] = data
                
            except Exception as e:
                 warnings.warn(f"Could not load ImgData array for view index {i} from {self.filepath}: {e}")
                 ImgData[i] = np.array([]) # Placeholder
        return ImgData

    def __load_trans(self) -> dict:
        """Loads the Trans field."""
        if 'Trans' not in self.data:
            raise KeyError("'Trans' field not found in HDF5 file.")
            
        Trans = {}
        TransVar = self.data['Trans']
        TransKeys = list(TransVar.keys())

        for key in TransKeys:
            try:
                # Attempt to read value, handle potential references
                value_ref = TransVar[key]
                value = np.array(value_ref).T # Load and transpose
                
                # Simplify if single value or 1D array
                if value.size == 1:
                    value = value.item() # Get scalar value
                elif value.shape[0] == 1 or value.shape[1] == 1:
                     value = value.flatten()
                     
                Trans[key] = value
            except Exception as e:
                 warnings.warn(f"Could not load Trans field '{key}' from {self.filepath}: {e}")
                 Trans[key] = None # Placeholder
        
        # Calculate wavelength if possible - Requires frequency
        try:
            if 'frequency' not in Trans or Trans['frequency'] is None or Trans['frequency'] <= 0:
                raise ValueError(f"Invalid or missing 'frequency' ({Trans.get('frequency')}) in Trans metadata.")
            
            sound_speed = Trans.get('SoundSpeed')
            if sound_speed is None:
                 # warnings.warn(f"'SoundSpeed' missing in Trans metadata for {self.filepath}. Defaulting to 1540 m/s.") # Suppressed warning
                 sound_speed = 1540.0
            elif sound_speed <= 1000:
                 # warnings.warn(f"'SoundSpeed' ({sound_speed}) seems low in {self.filepath}. Assuming m/s and using default 1540 m/s.") # Suppressed warning
                 sound_speed = 1540.0
                
            Trans['wavelengthMm'] = (sound_speed * 1000) / (Trans['frequency'] * 1e6)
            
        except Exception as e:
             Trans['wavelengthMm'] = None
             warnings.warn(f"Error calculating wavelengthMm for {self.filepath}: {e}")
             
        return Trans


class RecordingLoader:
    """Loads and combines MatData from a directory of HF/LF .mat file pairs."""
    
    def __init__(self, recording_dir: str | Path):
        self.recording_dir = Path(recording_dir)
        if not self.recording_dir.is_dir():
            raise FileNotFoundError(f"Recording directory not found: {self.recording_dir}")
        self.hf_files: List[Path] = []
        self.lf_files: List[Path] = []
        self._find_mat_files()

    def _find_mat_files(self):
        """Finds and sorts HF and LF .mat files based on numerical prefix."""
        all_files = list(self.recording_dir.glob('*.mat'))
        files_dict: Dict[int, Dict[str, Path]] = {}

        for f in all_files:
            name = f.stem # Filename without extension
            parts = name.split('_')
            if len(parts) == 2 and parts[0].isdigit() and parts[1] in ['HF', 'LF']:
                index = int(parts[0])
                ftype = parts[1].lower() # 'hf' or 'lf'
                if index not in files_dict:
                    files_dict[index] = {}
                if ftype == 'hf':
                     files_dict[index]['hf_path'] = f
                else:
                     files_dict[index]['lf_path'] = f
            else:
                warnings.warn(f"Ignoring file with unexpected name format: {f.name}")

        # Sort by index and store paths
        sorted_indices = sorted(files_dict.keys())
        self.hf_files = [files_dict[i]['hf_path'] for i in sorted_indices if 'hf_path' in files_dict[i]]
        self.lf_files = [files_dict[i]['lf_path'] for i in sorted_indices if 'lf_path' in files_dict[i]]
        
        if not self.hf_files or not self.lf_files:
             warnings.warn(f"No complete HF/LF pairs found in {self.recording_dir}")
        elif len(self.hf_files) != len(self.lf_files):
             warnings.warn(f"Mismatch in number of HF ({len(self.hf_files)}) and LF ({len(self.lf_files)}) files found.")

    def load_combined_mat_data(self) -> Tuple[MatData | None, MatData | None]:
        """Loads all LF and HF files and concatenates their imgdata.

        Returns:
            A tuple containing (combined_lf_matdata, combined_hf_matdata).
            Returns None for a type if no files were found or loading failed.
        """
        combined_lf = self._load_and_combine('lf', self.lf_files)
        combined_hf = self._load_and_combine('hf', self.hf_files)
        return combined_lf, combined_hf

    def _load_and_combine(self, file_type: str, file_list: List[Path]) -> MatData | None:
        """Helper to load and combine data for a specific type (LF or HF)."""
        if not file_list:
            return None

        all_mat_data: List[MatData] = []
        for fpath in file_list:
            try:
                # Use context manager for MatDataLoader
                with MatDataLoader(fpath) as loader:
                     mat_data = loader.build_mat_data()
                     all_mat_data.append(mat_data)
            except Exception as e:
                warnings.warn(f"Failed to load or process {file_type} file {fpath}: {e}")
                continue # Skip this file
        
        if not all_mat_data:
             warnings.warn(f"No {file_type} files could be successfully loaded.")
             return None

        # Use the first file's pdata and trans as representative (assuming they are consistent)
        # Could add checks for consistency here if needed.
        combined_pdata = all_mat_data[0].pdata
        combined_trans = all_mat_data[0].trans
        
        # Concatenate imgdata
        first_imgdata = all_mat_data[0].imgdata
        combined_imgdata: Dict[int, np.ndarray] = {}
        view_indices = list(first_imgdata.keys())

        for view_ind in view_indices:
            try:
                arrays_to_concat = [
                    md.imgdata[view_ind] for md in all_mat_data if view_ind in md.imgdata
                ]
                if arrays_to_concat:
                     # Assuming concatenation along the first axis (frames)
                     combined_imgdata[view_ind] = np.concatenate(arrays_to_concat, axis=0)
                else:
                     warnings.warn(f"No data found for view index {view_ind} in {file_type} files.")
                     combined_imgdata[view_ind] = np.array([]) # Placeholder
            except Exception as e:
                warnings.warn(f"Error concatenating imgdata for view {view_ind} in {file_type} files: {e}")
                combined_imgdata[view_ind] = np.array([]) # Placeholder

        return MatData(pdata=combined_pdata, imgdata=combined_imgdata, trans=combined_trans)
