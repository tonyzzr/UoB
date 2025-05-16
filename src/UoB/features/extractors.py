"""Defines base classes and specific implementations for feature extractors."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import torchvision.transforms.v2 as transforms

import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
print(f"Project root: {project_root}")
sys.path.insert(0, project_root)


# Import the specific registry
from src.registries import FEATURE_EXTRACTOR_REGISTRY
# Import the custom transform
from src.UoB.utils.transforms import PadToSquareAndAlign

# TODO: Integrate with a registry system (e.g., from src/registries.py)
# FEATURE_EXTRACTOR_REGISTRY = {} -> Replaced by import

class BaseFeatureExtractor(nn.Module, ABC):
    """Abstract base class for feature extractors."""
    def __init__(self, model_name: str, **kwargs):
        super().__init__()
        self.model_name = model_name
        # kwargs might include config paths, specific model params, etc.
        self.model = self._load_model(**kwargs)

    @abstractmethod
    def _load_model(self, **kwargs) -> nn.Module:
        """Loads the underlying feature extraction model."""
        pass

    @abstractmethod
    def get_preprocessing_transform(self):
        """Returns the specific preprocessing steps required for this model."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts features from the preprocessed input tensor."""
        pass

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Returns the dimension of the features produced by the model."""
        pass


# Example Implementation (Placeholder - requires actual model loading logic)
# Use the registry decorator
@FEATURE_EXTRACTOR_REGISTRY.register('dino_vits16')
class DinoV1Extractor(BaseFeatureExtractor):
    def __init__(self, model_name: str = 'dino_vits16', patch_size: int = 16, target_size: int = 224, **kwargs):
        self.patch_size = patch_size
        self.target_size = target_size
        # Define standard ImageNet normalization constants
        self.img_mean = [0.485, 0.456, 0.406]
        self.img_std = [0.229, 0.224, 0.225]
        super().__init__(model_name=model_name, **kwargs)
        self._feature_dim = 384 # Example for ViT-S/16

    def _load_model(self, **kwargs) -> nn.Module:
        print(f"Placeholder: Loading DINO model: {self.model_name}...")
        # TODO: Implement actual model loading
        dummy_model = nn.Identity()
        print(f"Placeholder: Model {self.model_name} loaded.")
        return dummy_model

    def get_preprocessing_transform(self):
        print(f"Returning DINO preprocessing transform (Pad to {self.target_size}, Normalize)...")
        return transforms.Compose([
            PadToSquareAndAlign(size=self.target_size, fill=0),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0]==1 else x),
            transforms.Normalize(mean=self.img_mean, std=self.img_std),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f"Placeholder: Extracting features with {self.model_name}...")
        # Input x should already be preprocessed by the transform
        features = self.model(x)
        # TODO: Adapt based on actual DINO model output
        print(f"Placeholder: Feature extraction complete. Output shape: {features.shape}")
        return features

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

# Add other extractors here (e.g., DinoV2Extractor)

# Factory function example (alternative to direct class usage)
def build_feature_extractor(config: dict) -> BaseFeatureExtractor:
    """Builds a feature extractor from a config dict."""
    extractor_name = config.get('name')
    if not extractor_name:
        raise ValueError("Config must contain an extractor 'name'")

    extractor_class = FEATURE_EXTRACTOR_REGISTRY.get(extractor_name)
    # Pass the rest of the config as kwargs to the constructor
    instance = extractor_class(**config.get('params', {}))
    return instance
