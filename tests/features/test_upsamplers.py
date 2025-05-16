import pytest
import torch
from PIL import Image

import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Use tomllib (Python 3.11+) or install toml
try:
    import tomllib
except ImportError:
    # Fallback for older Python or if tomllib is not available
    try:
        import toml as tomllib
    except ImportError:
        raise ImportError("Please install toml ('pip install toml') or use Python 3.11+ for tomllib.")

# Assuming src is in PYTHONPATH or tests are run from root
from src.registries import FEATURE_UPSAMPLER_REGISTRY
from src.UoB.features.upsamplers import BaseFeatureUpsampler, JointBilateralUpsampler, build_feature_upsampler

@pytest.fixture
def upsampler_config_path():
    # Construct path relative to project root
    return os.path.join(project_root, 'configs', 'features', 'jbu_dino16.toml')

@pytest.fixture
def upsampler_config(upsampler_config_path):
    # Load config from TOML file
    try:
        # Open in text mode ('r') for the toml package (used as fallback)
        with open(upsampler_config_path, 'r', encoding='utf-8') as f:
            config = tomllib.load(f)
        return config
    except FileNotFoundError:
        pytest.fail(f"Config file not found: {upsampler_config_path}")
    except Exception as e:
        pytest.fail(f"Failed to load or parse config file {upsampler_config_path}: {e}")

def test_upsampler_registration(upsampler_config):
    """Test if the upsampler is registered correctly."""
    print("\n---> Starting test_upsampler_registration...")
    assert upsampler_config['name'] in FEATURE_UPSAMPLER_REGISTRY
    print("<--- Finished test_upsampler_registration.")

def test_upsampler_instantiation_direct(upsampler_config):
    """Test direct instantiation via registry.get()."""
    print("\n---> Starting test_upsampler_instantiation_direct...")
    config_params = upsampler_config.get('params', {})
    UpsamplerClass = FEATURE_UPSAMPLER_REGISTRY.get(upsampler_config['name'])
    upsampler = UpsamplerClass(
        backbone_hub_id=config_params.get('backbone_hub_id', 'dino16'),
        use_norm=config_params.get('use_norm', True),
        target_size=config_params.get('target_size', 224)
    )
    assert isinstance(upsampler, JointBilateralUpsampler)
    assert upsampler.backbone_hub_id == config_params.get('backbone_hub_id', 'dino16')
    print("<--- Finished test_upsampler_instantiation_direct.")

def test_upsampler_factory(upsampler_config):
    """Test instantiation via the build_feature_upsampler factory function."""
    print("\n---> Starting test_upsampler_factory...")
    upsampler = build_feature_upsampler(upsampler_config)
    assert isinstance(upsampler, JointBilateralUpsampler)
    assert upsampler.backbone_hub_id == upsampler_config['params']['backbone_hub_id']
    assert upsampler.use_norm == upsampler_config['params']['use_norm']
    assert upsampler.target_size == upsampler_config['params']['target_size']
    print("<--- Finished test_upsampler_factory.")

def test_upsampler_preprocessing(upsampler_config):
    """Test the preprocessing pipeline of the upsampler."""
    print("\n---> Starting test_upsampler_preprocessing...")
    upsampler = build_feature_upsampler(upsampler_config)
    transform = upsampler.get_preprocessing_transform()
    assert transform is not None
    h, w = 100, 150
    target_size = upsampler_config['params']['target_size']
    dummy_pil = Image.new('L', (w, h), color=128)
    transformed_img = transform(dummy_pil)
    assert transformed_img.shape == (3, target_size, target_size)
    assert transformed_img.min() < 0 or transformed_img.max() > 1
    assert not torch.allclose(transformed_img, torch.zeros_like(transformed_img))
    print("<--- Finished test_upsampler_preprocessing.")

def test_upsampler_forward_pass(upsampler_config):
    """Test the forward pass with the actual loaded model (requires network)."""
    print("\n---> Starting test_upsampler_forward_pass...")

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    try:
        print("  Instantiating upsampler...")
        upsampler = build_feature_upsampler(upsampler_config)
        print("  Upsampler instantiated.")
        # Move model to device
        print(f"  Moving model to {device}...")
        upsampler.to(device)
        print("  Model moved to device.")
    except Exception as e:
        pytest.fail(f"Failed to instantiate or move JointBilateralUpsampler to device: {e}")

    target_size = upsampler_config['params']['target_size']
    # Move input tensor to device
    dummy_input = torch.rand(1, 3, target_size, target_size).to(device)
    print(f"  Created dummy input tensor on {device}: {dummy_input.shape}")

    try:
        print("  Setting model to eval mode...")
        upsampler.eval()
        print("  Model in eval mode.")
        print("  Starting forward pass with torch.no_grad()...")
        with torch.no_grad():
             features = upsampler(dummy_input)
        print(f"  Forward pass completed. Output shape: {features.shape}")
    except Exception as e:
        pytest.fail(f"Upsampler forward pass failed: {e}")

    print("  Asserting output type and shape...")
    assert isinstance(features, torch.Tensor)
    # Check if output is on the correct device
    assert features.device.type == device.type
    expected_dim = upsampler.upsampled_feature_dim
    assert expected_dim == 384 # We expect dim 384 for dino16 backbone
    assert features.shape == (1, expected_dim, target_size, target_size)
    print("  Assertions passed.")

    assert hasattr(upsampler, 'extractor_model')
    assert upsampler.extractor_model is not None
    print("<--- Finished test_upsampler_forward_pass.") 