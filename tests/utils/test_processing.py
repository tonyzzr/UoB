import pytest
import numpy as np
import cv2
import types # For SimpleNamespace to mock settings easily
from skimage import exposure # Needed for histogram matching test setup

import sys
import os

# Add project root to the Python path to find the 'src' module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
print(project_root)
sys.path.insert(0, project_root)

# Assuming the tests directory is discoverable relative to src
# Adjust the import path if necessary based on your test runner setup
from src.UoB.utils import processing

# --- Test Data Fixtures ---

@pytest.fixture
def dummy_src_image():
    """ Creates a dummy source image sequence (h, w, ntrans, nfrm). """
    h, w, ntrans, nfrm = 50, 60, 2, 3
    # Create data with varying ranges for different tests
    data = np.random.rand(h, w, ntrans, nfrm) * 1000 + 100 # Example range
    return data.astype(np.float32) # Use float32 for more realistic input

@pytest.fixture
def dummy_mask():
    """ Creates a dummy mask (1, ntrans, h, w). """
    h, w, ntrans = 50, 60, 2
    # Create a simple gradient mask for testing
    mask_data = np.linspace(0, 1, h*w*ntrans).reshape((1, ntrans, h, w))
    return mask_data.astype(np.float32)

@pytest.fixture
def dummy_theta_v():
    """ Creates dummy theta_v data (h, w). """
    h, w = 50, 60
    # Example angle data
    theta_v = np.random.rand(h, w) * 90
    return theta_v.astype(np.float32)


# --- Helper for Settings ---

def create_setting(**kwargs):
    """ Creates a SimpleNamespace object to mimic setting dataclasses. """
    return types.SimpleNamespace(**kwargs)

# --- Test Functions ---

# 1. resize_in_scale
def test_resize_in_scale_shape(dummy_src_image):
    h_in, w_in, ntrans, nfrm = dummy_src_image.shape
    dst_size = (h_in // 2, w_in * 2) # Example different output size
    result = processing.resize_in_scale(dummy_src_image, dst_size)
    assert result.shape == (dst_size[0], dst_size[1], ntrans, nfrm)

def test_resize_in_scale_dtype(dummy_src_image):
    dst_size = (50, 60)
    result = processing.resize_in_scale(dummy_src_image, dst_size)
    assert result.dtype == dummy_src_image.dtype


# 2. generate_mask
def test_generate_mask_disabled(dummy_theta_v):
    setting = create_setting(enable=False)
    result = processing.generate_mask(dummy_theta_v, setting)
    assert result.shape == dummy_theta_v.shape
    assert np.all(result == 1.0) # Should be all ones when disabled

def test_generate_mask_enabled_shape(dummy_theta_v):
    setting = create_setting(enable=True, main_lobe_beamwidth=30, soft_boundary=True, softness=0.3)
    result = processing.generate_mask(dummy_theta_v, setting)
    assert result.shape == dummy_theta_v.shape
    assert result.dtype == np.float64 # Soft boundary uses floats, ensure float output

def test_generate_mask_hard_boundary(dummy_theta_v):
     setting = create_setting(enable=True, main_lobe_beamwidth=45, soft_boundary=False)
     result = processing.generate_mask(dummy_theta_v, setting)
     assert result.shape == dummy_theta_v.shape
     assert result.dtype == np.float64 # Uses floats internally
     assert np.all((result == 0.0) | (result == 1.0)) # Hard boundary should be 0s and 1s


# 3. log_compression
def test_log_compression_disabled(dummy_src_image):
    setting = create_setting(enable=False)
    result = processing.log_compression(dummy_src_image, setting)
    np.testing.assert_array_equal(result, dummy_src_image)
    assert result.shape == dummy_src_image.shape

def test_log_compression_enabled_shape_dtype(dummy_src_image):
    setting = create_setting(enable=True, dynamic_range=60, max_value=1024)
    result = processing.log_compression(dummy_src_image, setting)
    assert result.shape == dummy_src_image.shape
    assert result.dtype == np.float64 # Calculation results in float

def test_log_compression_range(dummy_src_image):
    # Use smaller range data to better test clipping
    src = np.random.rand(*dummy_src_image.shape) * 2000
    setting = create_setting(enable=True, dynamic_range=40, max_value=1500)
    result = processing.log_compression(src, setting)
    assert np.all(result >= 0)
    assert np.all(result <= 255)

def test_log_compression_range_flexible_max(dummy_src_image):
    # Use smaller range data
    src = np.random.rand(*dummy_src_image.shape) * 500 + 50
    setting = create_setting(enable=True, dynamic_range=50, max_value=None)
    result = processing.log_compression(src, setting)
    assert np.all(result >= 0)
    assert np.all(result <= 255)


# 4. speckle_reduction
def test_speckle_reduction_disabled(dummy_src_image):
    setting = create_setting(enable=False)
    result = processing.speckle_reduction(dummy_src_image, setting)
    np.testing.assert_array_equal(result, dummy_src_image)
    assert result.shape == dummy_src_image.shape

def test_speckle_reduction_enabled_shape_dtype(dummy_src_image):
    setting = create_setting(enable=True, med_blur_kernal=3, nlm_h=9, nlm_template_window_size=7, nlm_search_window_size=11)
    # Reduce size for faster testing
    small_src = dummy_src_image[:10, :10, :, :1]
    result = processing.speckle_reduction(small_src, setting)
    assert result.shape == small_src.shape
    # Output dtype depends on processing, NLM outputs same as input if float, but operates on uint8
    # Let's check if it's numeric
    assert np.issubdtype(result.dtype, np.number)


# 5. reject_grating_lobe_artifact
def test_reject_grating_lobe_disabled(dummy_src_image, dummy_mask):
    setting = create_setting(enable=False)
    result = processing.reject_grating_lobe_artifact(dummy_src_image, dummy_mask, setting)
    np.testing.assert_array_equal(result, dummy_src_image)
    assert result.shape == dummy_src_image.shape

def test_reject_grating_lobe_enabled_shape_dtype(dummy_src_image, dummy_mask):
    setting = create_setting(enable=True)
    result = processing.reject_grating_lobe_artifact(dummy_src_image, dummy_mask, setting)
    assert result.shape == dummy_src_image.shape
    assert result.dtype == dummy_src_image.dtype # Simple multiplication


# 6. histogram_match
# Mock skimage if not installed (though needed for test setup here)
try:
    from skimage import exposure
except ImportError:
    pytest.skip("skimage not installed, skipping histogram_match tests", allow_module_level=True)

def test_histogram_match_disabled(dummy_src_image):
    setting = create_setting(enable=False)
    result = processing.histogram_match(dummy_src_image, setting)
    np.testing.assert_array_equal(result, dummy_src_image)
    assert result.shape == dummy_src_image.shape

def test_histogram_match_enabled_shape_dtype(dummy_src_image):
    setting = create_setting(enable=True, ref_ind=0, background_removal=True)
    result = processing.histogram_match(dummy_src_image.astype(np.uint8), setting) # Needs uint8 or similar for skimage input
    assert result.shape == dummy_src_image.shape
    # Background removal step introduces floats
    assert np.issubdtype(result.dtype, np.floating)


# 7. apply_tgc
def test_apply_tgc_disabled(dummy_src_image):
    setting = create_setting(enable=False)
    result = processing.apply_tgc(dummy_src_image, setting)
    np.testing.assert_array_equal(result, dummy_src_image)
    assert result.shape == dummy_src_image.shape

def test_apply_tgc_enabled_shape_dtype(dummy_src_image):
    setting = create_setting(enable=True, tgc_threshold=0.8, tgc_slope=10)
    result = processing.apply_tgc(dummy_src_image, setting)
    assert result.shape == dummy_src_image.shape
    # Multiplication with float64 mask promotes result to float64
    assert result.dtype == np.float64

def test_apply_tgc_effect(dummy_src_image):
    setting = create_setting(enable=True, tgc_threshold=0.5, tgc_slope=20)
    result = processing.apply_tgc(dummy_src_image, setting)
    # Check if deeper parts (higher row index) are generally attenuated
    # Compare mean of top rows vs bottom rows
    h = dummy_src_image.shape[0]
    mean_top = np.mean(result[:h//4, ...])
    mean_bottom = np.mean(result[3*h//4:, ...])
    assert mean_bottom < mean_top # Should be attenuated at bottom
