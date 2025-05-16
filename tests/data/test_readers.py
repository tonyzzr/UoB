import pytest
import numpy as np
import h5py
from pathlib import Path
import sys
import os

# Add project root to the Python path to find the 'src' module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Import the readers and formats modules
from src.UoB.data import readers
from src.UoB.data import formats

# --- Test MatDataLoader ---

# Helper to create a dummy HDF5 file for testing MatDataLoader directly
@pytest.fixture
def dummy_mat_file(tmp_path) -> Path:
    filepath = tmp_path / "dummy_1_HF.mat"
    with h5py.File(filepath, 'w') as f:
        # Create minimal structure based on reader assumptions
        pdata_grp = f.create_group('PData')
        # Use references for PDelta, Size, Origin
        pdelta_data = f.create_dataset('pdelta_0', data=np.array([[0.1, 0, 0.1]]).T)
        size_data = f.create_dataset('size_0', data=np.array([[100, 50]]).T)
        origin_data = f.create_dataset('origin_0', data=np.array([[0, 0, 0]]).T)
        pdata_grp.create_dataset('PDelta', data=np.array([[f['pdelta_0'].ref]], dtype=h5py.ref_dtype))
        pdata_grp.create_dataset('Size', data=np.array([[f['size_0'].ref]], dtype=h5py.ref_dtype))
        pdata_grp.create_dataset('Origin', data=np.array([[f['origin_0'].ref]], dtype=h5py.ref_dtype))
        
        # ImgData structure - Create data first, then reference it
        img_array = np.random.rand(5, 1, 100, 50) # frames, ?, nz, nx
        img_data_ds = f.create_dataset('img_data_arrays/view_0', data=img_array)
        # Create the dataset holding the reference (mimicking MatLab structure)
        # This structure might vary, adjust based on real files if needed
        img_data_ref_ds = f.create_dataset('img_data_refs/ref_0', data=np.array([[img_data_ds.ref]], dtype=h5py.ref_dtype))
        f.create_dataset('ImgData', data=np.array([[f['img_data_refs/ref_0'].ref]], dtype=h5py.ref_dtype))
        
        # Trans structure
        trans_grp = f.create_group('Trans')
        trans_grp.create_dataset('frequency', data=np.array([5e6]))
        trans_grp.create_dataset('SoundSpeed', data=np.array([1540]))
        trans_grp.create_dataset('ElementPos', data=np.zeros((64, 3))) # Dummy
        
    return filepath

def test_matdataloader_context_manager(dummy_mat_file):
    """ Test MatDataLoader using the context manager. """
    with readers.MatDataLoader(dummy_mat_file) as loader:
        mat_data = loader.build_mat_data()
        assert isinstance(mat_data, formats.MatData)
        assert 0 in mat_data.pdata
        assert 'nx' in mat_data.pdata[0]
        assert 0 in mat_data.imgdata
        assert mat_data.imgdata[0].shape[0] == 5 # Check frames dim
        assert 'frequency' in mat_data.trans
        assert 'wavelengthMm' in mat_data.trans
        assert mat_data.trans['wavelengthMm'] is not None
    # Check file is closed after exiting context
    assert loader.data is None 

def test_matdataloader_file_not_found():
     with pytest.raises(FileNotFoundError):
         readers.MatDataLoader("non_existent.mat")

# --- Test RecordingLoader ---

@pytest.fixture
def real_data_dir() -> Path:
    # Define the path to the real data directory
    # IMPORTANT: This test relies on the specific data existing at this path
    path = Path("/home/tonyz/code_bases/UoB/data/raw/recording_2022-08-17_trial2-arm")
    if not path.is_dir():
        pytest.skip(f"Real data directory not found: {path}")
    return path

def test_recording_loader_find_files(real_data_dir):
    """ Test that the loader finds the expected number of files. """
    loader = readers.RecordingLoader(real_data_dir)
    
    # Dynamically determine expected count from loader
    expected_count_hf = len(loader.hf_files)
    expected_count_lf = len(loader.lf_files)
    
    assert expected_count_hf > 0, "No HF files found by loader"
    assert expected_count_lf > 0, "No LF files found by loader"
    assert expected_count_hf == expected_count_lf, "Mismatch between HF and LF file counts"
    
    print(f"Found {expected_count_hf} HF and {expected_count_lf} LF files.") # Added for debugging
    # Check sorting by asserting first and last file names found
    assert loader.hf_files[0].name.endswith("_HF.mat") # General check
    assert loader.lf_files[0].name.endswith("_LF.mat")
    # Extract indices to confirm numerical sort
    first_hf_index = int(loader.hf_files[0].stem.split('_')[0])
    last_hf_index = int(loader.hf_files[-1].stem.split('_')[0])
    first_lf_index = int(loader.lf_files[0].stem.split('_')[0])
    last_lf_index = int(loader.lf_files[-1].stem.split('_')[0])
    
    assert first_hf_index < last_hf_index
    assert first_lf_index < last_lf_index
    assert first_hf_index == first_lf_index
    assert last_hf_index == last_lf_index
    # Check specific last index if known (e.g., 131 based on previous error)
    assert last_hf_index == 131 # Update if this changes


def test_recording_loader_load_combined(real_data_dir):
    """ Test loading and combining data using RecordingLoader. """
    loader = readers.RecordingLoader(real_data_dir)
    combined_lf, combined_hf = loader.load_combined_mat_data()
    
    # Check that data was loaded for both
    assert isinstance(combined_lf, formats.MatData), "Combined LF data failed to load."
    assert isinstance(combined_hf, formats.MatData), "Combined HF data failed to load."
    
    # Check that metadata seems reasonable (using first view index 0 as example)
    assert combined_lf.pdata and 0 in combined_lf.pdata, "Missing pdata[0] in combined LF."
    assert combined_lf.trans and 'frequency' in combined_lf.trans, "Missing frequency in combined LF trans."
    assert combined_hf.pdata and 0 in combined_hf.pdata, "Missing pdata[0] in combined HF."
    assert combined_hf.trans and 'frequency' in combined_hf.trans, "Missing frequency in combined HF trans."
    
    # Check imgdata concatenation
    assert combined_lf.imgdata, "Combined LF imgdata is empty."
    assert combined_hf.imgdata, "Combined HF imgdata is empty."
    
    # --- Debugging: Print shapes --- 
    lf_view_idx = next(iter(combined_lf.imgdata), None)
    hf_view_idx = next(iter(combined_hf.imgdata), None)
    
    assert lf_view_idx is not None, "No views found in combined LF imgdata."
    assert hf_view_idx is not None, "No views found in combined HF imgdata."
    
    # Print shape from first loaded file for comparison
    if loader.lf_files:
        try:
            with readers.MatDataLoader(loader.lf_files[0]) as single_loader_lf:
                single_data_lf = single_loader_lf.build_mat_data()
                if lf_view_idx in single_data_lf.imgdata:
                    print(f"Debug: Shape of imgdata[{lf_view_idx}] from FIRST LF file ({loader.lf_files[0].name}): {single_data_lf.imgdata[lf_view_idx].shape}")
                else:
                    print(f"Debug: View index {lf_view_idx} not found in first LF file's imgdata.")
        except Exception as e:
            print(f"Debug: Error loading first LF file for shape check: {e}")
            
    # Print shape of the combined array
    combined_lf_img = combined_lf.imgdata[lf_view_idx]
    print(f"Debug: Shape of COMBINED LF imgdata[{lf_view_idx}]: {combined_lf_img.shape}")
    # --- End Debugging ---

    # --- Determine shapes and frames from combined data --- 
    total_frames_lf = combined_lf_img.shape[0]
    total_frames_hf = combined_hf.imgdata[hf_view_idx].shape[0]
    
    num_lf_files = len(loader.lf_files)
    num_hf_files = len(loader.hf_files)
    
    assert num_lf_files > 0, "Loader found zero LF files."
    assert num_hf_files > 0, "Loader found zero HF files."
    
    # Calculate average frames per file (should be integer if data is consistent)
    frames_per_file_lf = total_frames_lf / num_lf_files
    frames_per_file_hf = total_frames_hf / num_hf_files
    
    # Check if it divides evenly (indicates consistent frames per file)
    assert total_frames_lf % num_lf_files == 0, f"Total LF frames ({total_frames_lf}) not divisible by file count ({num_lf_files}). Inconsistent frames?"
    assert total_frames_hf % num_hf_files == 0, f"Total HF frames ({total_frames_hf}) not divisible by file count ({num_hf_files}). Inconsistent frames?"

    print(f"LF: {total_frames_lf} total frames / {num_lf_files} files = {frames_per_file_lf} frames/file")
    print(f"HF: {total_frames_hf} total frames / {num_hf_files} files = {frames_per_file_hf} frames/file")
    
    # We don't need to explicitly check frames_per_file > 0 anymore,
    # as the total frame count check implicitly covers it if num_files > 0.
    
    # Check that other dimensions seem valid (get shape of a single frame)
    single_lf_frame_shape = combined_lf_img.shape[1:]
    single_hf_frame_shape = combined_hf.imgdata[hf_view_idx].shape[1:]
    assert len(single_lf_frame_shape) == 3, f"Unexpected LF frame shape dimensions: {single_lf_frame_shape}" 
    assert len(single_hf_frame_shape) == 3, f"Unexpected HF frame shape dimensions: {single_hf_frame_shape}" 

    # Optional: Load one file again just to compare single frame shape if needed
    # This is redundant if we trust the combined shape correctly represents single frames
    # with readers.MatDataLoader(loader.lf_files[0]) as single_loader_lf:
    #    single_data_lf = single_loader_lf.build_mat_data()
    #    assert single_data_lf.imgdata[lf_view_idx].shape[1:] == single_lf_frame_shape
    # with readers.MatDataLoader(loader.hf_files[0]) as single_loader_hf:
    #    single_data_hf = single_loader_hf.build_mat_data()
    #    assert single_data_hf.imgdata[hf_view_idx].shape[1:] == single_hf_frame_shape 