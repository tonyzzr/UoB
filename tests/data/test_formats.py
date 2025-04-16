import pytest
import numpy as np
import torch
from dataclasses import is_dataclass, fields

import sys
import os

# Add project root to the Python path to find the 'src' module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
print(project_root)
sys.path.insert(0, project_root)

# Import the formats module
from src.UoB.data import formats

# --- Test Settings Dataclasses Default Values ---

@pytest.mark.parametrize("setting_cls, expected_defaults", [
    (formats.MaskSetting, {'enable': False, 'main_lobe_beamwidth': 30.0, 'soft_boundary': True, 'softness': 0.3}),
    (formats.LogCompressionSetting, {'enable': False, 'dynamic_range': 60.0, 'max_value': 1024.0}),
    (formats.SpeckleReductionSetting, {'enable': False, 'med_blur_kernal': 3, 'nlm_h': 9.0, 'nlm_template_window_size': 7, 'nlm_search_window_size': 11}),
    (formats.RejectGratingLobeSetting, {'enable': False}),
    (formats.HistogramMatchSetting, {'enable': False, 'ref_ind': 0, 'background_removal': True}),
    (formats.ApplyTGCSetting, {'enable': False, 'tgc_threshold': 0.8, 'tgc_slope': 10.0}),
    (formats.ZeroPadSetting, {'enable': False, 'top_padding_ratio': 0.0, 'bottom_padding_ratio': 0.0, 'left_padding_ratio': 0.0, 'right_padding_ratio': 0.0}),
])
def test_settings_defaults(setting_cls, expected_defaults):
    """ Test that settings dataclasses have the expected default values. """
    instance = setting_cls()
    assert is_dataclass(instance)
    for field_name, expected_value in expected_defaults.items():
        assert hasattr(instance, field_name)
        assert getattr(instance, field_name) == expected_value

# --- Test Data Structure Instantiation and Properties ---

def test_matdata_instantiation():
    pdata_dummy = {0: {'nx': 100, 'nz': 50}}
    imgdata_dummy = {0: np.zeros((10, 1, 50, 100))}
    trans_dummy = {'frequency': 5e6}
    instance = formats.MatData(pdata=pdata_dummy, imgdata=imgdata_dummy, trans=trans_dummy)
    assert instance.pdata == pdata_dummy
    assert instance.imgdata == imgdata_dummy
    assert instance.trans == trans_dummy
    assert isinstance(str(instance), str) # Check __str__ runs

def test_transpos_instantiation_and_props():
    left = np.array([[10], [20], [1]])
    right = np.array([[50], [20], [1]])
    instance = formats.TransPos(left_edge_coord=left, right_edge_coord=right)
    assert np.array_equal(instance.left_edge_coord, left)
    assert np.array_equal(instance.right_edge_coord, right)
    # Test properties
    assert instance.length == pytest.approx(40.0)
    centroid = instance.centroid
    assert centroid.shape == (2, 1)
    assert np.allclose(centroid, np.array([[30], [20]]))

# --- Test BmodeConfig --- 

@pytest.fixture
def sample_config_dict():
    """ Provides a sample dictionary mimicking loaded TOML data. """
    return {
        'general': {
            'scale_bar': 0.05
        },
        'mask': {
            'enable': True,
            'main_lobe_beamwidth': 25.0,
            'soft_boundary': False
            # softness default should be used
        },
        'log_compression': {
            'enable': True,
            'dynamic_range': 50.0
            # max_value default should be used
        },
        'speckle_reduction': {
            'enable': True,
            'med_blur_kernal': 5
            # Other nlm defaults should be used
        },
        'reject_grating_lobe': {
            'enable': True
        },
        'histogram_match': {
            'enable': False
            # Other defaults should be used
        },
        'time_gain_compensation': {
            'enable': True,
            'tgc_threshold': 0.7,
            'tgc_slope': 12.0
        }
        # zero_padding section omitted, should use defaults or be ignored
    }

def test_bmodeconfig_from_dict(sample_config_dict):
    """ Test creating BmodeConfig from a dictionary. """
    config = formats.BmodeConfig.from_dict(sample_config_dict)
    
    assert isinstance(config, formats.BmodeConfig)
    assert config.scale_bar == 0.05
    
    # Check nested MaskSetting
    assert isinstance(config.mask_setting, formats.MaskSetting)
    assert config.mask_setting.enable is True
    assert config.mask_setting.main_lobe_beamwidth == 25.0
    assert config.mask_setting.soft_boundary is False
    assert config.mask_setting.softness == 0.3 # Check default used
    
    # Check nested LogCompressionSetting
    assert isinstance(config.log_compression_setting, formats.LogCompressionSetting)
    assert config.log_compression_setting.enable is True
    assert config.log_compression_setting.dynamic_range == 50.0
    assert config.log_compression_setting.max_value == 1024.0 # Check default used
    
    # Check nested SpeckleReductionSetting
    assert isinstance(config.speckle_reduction_setting, formats.SpeckleReductionSetting)
    assert config.speckle_reduction_setting.enable is True
    assert config.speckle_reduction_setting.med_blur_kernal == 5
    assert config.speckle_reduction_setting.nlm_h == 9.0 # Check default
    
    # Check other settings
    assert isinstance(config.reject_grating_lobe_setting, formats.RejectGratingLobeSetting)
    assert config.reject_grating_lobe_setting.enable is True
    
    assert isinstance(config.histogram_match_setting, formats.HistogramMatchSetting)
    assert config.histogram_match_setting.enable is False
    assert config.histogram_match_setting.ref_ind == 0 # Check default
    
    assert isinstance(config.time_gain_compensation_setting, formats.ApplyTGCSetting)
    assert config.time_gain_compensation_setting.enable is True
    assert config.time_gain_compensation_setting.tgc_threshold == 0.7
    assert config.time_gain_compensation_setting.tgc_slope == 12.0

def test_bmodeconfig_from_dict_missing_sections(sample_config_dict):
    """ Test that missing sections in dict use defaults. """
    # Remove a section, e.g., speckle reduction
    del sample_config_dict['speckle_reduction']
    config = formats.BmodeConfig.from_dict(sample_config_dict)
    
    # Check that the corresponding setting uses defaults
    assert isinstance(config.speckle_reduction_setting, formats.SpeckleReductionSetting)
    assert config.speckle_reduction_setting.enable is False # Default
    assert config.speckle_reduction_setting.med_blur_kernal == 3 # Default

# --- Test Processed/Final Data Structures Instantiation ---

@pytest.fixture
def dummy_bmode_config():
    """ Creates a default BmodeConfig for testing Bmode instantiation. """
    # Minimal config dict using defaults for nested settings
    cfg_dict = {'general': {'scale_bar': 0.1}}
    return formats.BmodeConfig.from_dict(cfg_dict)

def test_bmode_instantiation(dummy_bmode_config):
    n_frame, n_trans, h, w = 5, 2, 50, 60
    img_seq = np.zeros((n_frame, n_trans, h, w))
    mask_seq = np.ones((1, n_trans, h, w))
    trans_pos_dict = {
        0: formats.TransPos(np.array([[0],[0],[1]]), np.array([[40],[0],[1]])),
        1: formats.TransPos(np.array([[50],[0],[1]]), np.array([[90],[0],[1]])),
    }
    instance = formats.Bmode(
        num_trans=n_trans,
        scale_bar=0.1,
        b_img_seq=img_seq,
        trans_pos=trans_pos_dict,
        mask_seq=mask_seq,
        config=dummy_bmode_config
    )
    assert instance.num_trans == n_trans
    assert instance.scale_bar == 0.1
    assert instance.b_img_seq.shape == (n_frame, n_trans, h, w)
    assert instance.mask_seq.shape == (1, n_trans, h, w)
    assert isinstance(instance.config, formats.BmodeConfig)
    assert isinstance(str(instance), str) # Check __str__ runs

def test_multiviewbmodevideo_instantiation():
    n_frame, n_view, h, w = 5, 2, 50, 60
    images = torch.zeros((n_frame, n_view, h, w))
    masks = torch.ones((1, n_view, h, w))
    instance = formats.MultiViewBmodeVideo(
        n_view=n_view,
        image_shape=(h, w),
        origin=(10.0, 0.0),
        aperture_size=40.0,
        scale_bar=0.1,
        source_data_identifier="/path/to/data",
        processing_config={'log_compression': {'enable': True, 'dynamic_range': 50}},
        n_frame=n_frame,
        view_images=images,
        view_masks=masks
    )
    assert instance.n_frame == n_frame
    assert instance.n_view == n_view
    assert instance.image_shape == (h, w)
    assert instance.view_images.shape == (n_frame, n_view, h, w)
    assert instance.view_masks.shape == (1, n_view, h, w)
    assert instance.source_data_identifier == "/path/to/data"
    assert instance.processing_config['log_compression']['dynamic_range'] == 50
    assert isinstance(str(instance), str) # Check __str__ runs
