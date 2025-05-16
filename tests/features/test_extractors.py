import pytest
import torch
from PIL import Image

import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Assuming src is in PYTHONPATH or tests are run from root
from src.registries import FEATURE_EXTRACTOR_REGISTRY
from src.UoB.features.extractors import BaseFeatureExtractor, DinoV1Extractor, build_feature_extractor

@pytest.fixture
def extractor_config():
    return {
        'name': 'dino_vits16',
        'params': {
            'patch_size': 16,
            'target_size': 224
        }
    }

def test_extractor_registration():
    """Test if the extractor is registered correctly."""
    assert 'dino_vits16' in FEATURE_EXTRACTOR_REGISTRY

def test_extractor_instantiation_direct():
    """Test direct instantiation via registry.get()."""
    ExtractorClass = FEATURE_EXTRACTOR_REGISTRY.get('dino_vits16')
    extractor = ExtractorClass(model_name='dino_vits16_override', patch_size=16, target_size=224)
    assert isinstance(extractor, DinoV1Extractor)
    assert extractor.model_name == 'dino_vits16_override'

def test_extractor_factory(extractor_config):
    """Test instantiation via the build_feature_extractor factory function."""
    extractor = build_feature_extractor(extractor_config)
    assert isinstance(extractor, DinoV1Extractor)
    assert extractor.patch_size == extractor_config['params']['patch_size']
    assert extractor.target_size == extractor_config['params']['target_size']

def test_extractor_preprocessing(extractor_config):
    """Test the preprocessing pipeline of the extractor."""
    extractor = build_feature_extractor(extractor_config)
    transform = extractor.get_preprocessing_transform()
    assert transform is not None

    # Test with a dummy PIL image (grayscale)
    h, w = 100, 150
    target_size = extractor_config['params']['target_size']
    dummy_pil = Image.new('L', (w, h), color=128)
    transformed_img = transform(dummy_pil)

    assert transformed_img.shape == (3, target_size, target_size)
    # Check normalization occurred (values differ from 0-1 range significantly)
    # (Using a loose check as exact values depend on padding)
    assert transformed_img.min() < 0 or transformed_img.max() > 1
    assert not torch.allclose(transformed_img, torch.zeros_like(transformed_img)) # Ensure not all zeros

def test_extractor_forward_pass(extractor_config):
    """Test the forward pass (placeholder)."""
    extractor = build_feature_extractor(extractor_config)
    target_size = extractor_config['params']['target_size']
    # Create a dummy input tensor that matches the expected preprocessed shape
    dummy_input = torch.rand(1, 3, target_size, target_size)
    # Run forward pass (currently uses nn.Identity placeholder)
    features = extractor(dummy_input)
    # Basic check on output shape (will need adjustment when real model is loaded)
    assert features.shape == dummy_input.shape # Because placeholder is Identity
    assert extractor.feature_dim == 384 # Check declared feature dim 