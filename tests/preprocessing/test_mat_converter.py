import pytest
import pickle
from pathlib import Path
import sys
import os
import torch
import numpy as np

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Imports from our project
from src.UoB.preprocessing.mat_converter import MatConverter
from src.UoB.data.formats import MultiViewBmodeVideo
from src.UoB.data.readers import RecordingLoader # Needed to get expected frames

# --- Fixtures ---

@pytest.fixture
def config_path() -> Path:
    path = Path("configs/preprocessing/default.toml")
    if not path.exists():
        pytest.fail(f"Config file not found for testing: {path}")
    return path

@pytest.fixture
def real_data_dir() -> Path:
    path = Path("/home/tonyz/code_bases/UoB/data/raw/recording_2022-08-17_trial2-arm")
    if not path.is_dir():
        pytest.skip(f"Real data directory not found, skipping integration test: {path}")
    return path

# --- Test Function ---

def test_mat_converter_run(config_path, real_data_dir, tmp_path):
    """ Integration test for MatConverter using real data directory. """
    # Arrange
    converter = MatConverter(config_path=config_path)
    output_dir = tmp_path / "processed" # Save to a subdirectory in tmp_path
    expected_output_filename = "combined_mvbv.pkl"
    expected_output_path = output_dir / real_data_dir.name / expected_output_filename
    
    # --- Determine expected shapes/counts from source --- 
    # (Copied logic from test_readers to avoid re-running loader just for info)
    loader = RecordingLoader(real_data_dir)
    num_lf_files = len(loader.lf_files)
    num_hf_files = len(loader.hf_files)
    assert num_lf_files > 0 and num_hf_files > 0, "Test setup failed: No LF or HF files found by loader."
    assert num_lf_files == num_hf_files, "Test setup failed: Mismatched LF/HF file counts."
    
    # Get frames per file (use loader directly)
    raw_lf, raw_hf = loader.load_combined_mat_data()
    assert raw_lf is not None and raw_hf is not None, "Failed to load raw data for shape check."
    assert raw_lf.imgdata and raw_hf.imgdata, "imgdata missing after load."
    lf_view_idx = next(iter(raw_lf.imgdata))
    hf_view_idx = next(iter(raw_hf.imgdata))
    
    # Calculate expected total frames AFTER successful load
    expected_total_frames_lf = raw_lf.imgdata[lf_view_idx].shape[0]
    expected_total_frames_hf = raw_hf.imgdata[hf_view_idx].shape[0]
    assert expected_total_frames_lf > 0 and expected_total_frames_hf > 0, "Loaded raw imgdata has 0 frames."
    
    # Get expected number of views (should be same for LF/HF)
    expected_n_view = len(raw_lf.imgdata)
    assert expected_n_view == len(raw_hf.imgdata), "Mismatch in number of views between LF and HF raw data."
    assert expected_n_view > 0, "No views found in loaded raw data."
    # -----------------------------------------------------
    
    # Act
    # Suppress warnings during conversion for cleaner test output if needed
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    converter.convert_recording(
        recording_dir=real_data_dir, 
        output_dir=output_dir, 
        output_filename=expected_output_filename
    )
    
    # Assert: Check output file exists
    assert expected_output_path.exists(), f"Output file was not created: {expected_output_path}"
    assert expected_output_path.is_file(), f"Expected output is not a file: {expected_output_path}"
    
    # Assert: Load and check content
    with open(expected_output_path, 'rb') as f:
        loaded_data = pickle.load(f)
        
    assert isinstance(loaded_data, dict), "Saved data is not a dictionary."
    assert 'lftx' in loaded_data, "Missing 'lftx' key in saved data."
    assert 'hftx' in loaded_data, "Missing 'hftx' key in saved data."
    
    # Check LF data
    mvbv_lf = loaded_data['lftx']
    assert isinstance(mvbv_lf, MultiViewBmodeVideo), "'lftx' value is not a MultiViewBmodeVideo object."
    assert mvbv_lf.source_data_identifier == real_data_dir.name
    assert mvbv_lf.n_frame == expected_total_frames_lf
    assert mvbv_lf.n_view == expected_n_view
    assert mvbv_lf.view_images.shape == (expected_total_frames_lf, expected_n_view, mvbv_lf.image_shape[0], mvbv_lf.image_shape[1])
    assert mvbv_lf.view_masks.shape == (1, expected_n_view, mvbv_lf.image_shape[0], mvbv_lf.image_shape[1])
    assert isinstance(mvbv_lf.view_images, torch.Tensor)
    assert isinstance(mvbv_lf.view_masks, torch.Tensor)
    assert mvbv_lf.scale_bar == converter.lf_bmode_config.scale_bar # Check scale bar matches config
    
    # Check HF data
    mvbv_hf = loaded_data['hftx']
    assert isinstance(mvbv_hf, MultiViewBmodeVideo), "'hftx' value is not a MultiViewBmodeVideo object."
    assert mvbv_hf.source_data_identifier == real_data_dir.name
    assert mvbv_hf.n_frame == expected_total_frames_hf
    assert mvbv_hf.n_view == expected_n_view
    assert mvbv_hf.view_images.shape == (expected_total_frames_hf, expected_n_view, mvbv_hf.image_shape[0], mvbv_hf.image_shape[1])
    assert mvbv_hf.view_masks.shape == (1, expected_n_view, mvbv_hf.image_shape[0], mvbv_hf.image_shape[1])
    assert isinstance(mvbv_hf.view_images, torch.Tensor)
    assert isinstance(mvbv_hf.view_masks, torch.Tensor)
    assert mvbv_hf.scale_bar == converter.hf_bmode_config.scale_bar # Check scale bar matches config
