"""Defines base classes and specific implementations for feature upsamplers (e.g., FeatUp)."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import torchvision.transforms.v2 as transforms
import time

# Import the specific registry
from src.registries import FEATURE_UPSAMPLER_REGISTRY
# Import base extractor type hint for factory function
from .extractors import BaseFeatureExtractor
# Import the custom transform
from src.UoB.utils.transforms import PadToSquareAndAlign
# Import FeatUp normalization utility
from third_party.FeatUp.featup.util import norm

# TODO: Integrate with a registry system
# FEATURE_UPSAMPLER_REGISTRY = {} -> Replaced by import

class BaseFeatureUpsampler(nn.Module, ABC):
    """Abstract base class for feature upsamplers."""
    def __init__(self, model_name: str, extractor_model: nn.Module | None = None, **kwargs):
        super().__init__()
        self.model_name = model_name
        # Upsamplers often depend on or wrap a backbone extractor
        self.extractor_model = extractor_model
        # kwargs might include config paths, specific model params, etc.
        self.upsampler_model = self._load_model(**kwargs)

    @abstractmethod
    def _load_model(self, **kwargs) -> nn.Module:
        """Loads the underlying upsampling model/module."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Upsamples features extracted from the input tensor x.

        Note: The exact input requirements might vary.
        Some upsamplers might take the image tensor `x` directly,
        others might require pre-computed low-res features.
        Use kwargs flexibly.
        """
        pass

    @property
    @abstractmethod
    def upsampled_feature_dim(self) -> int:
        """Returns the dimension of the upsampled features."""
        pass

    # Add abstract method for preprocessing
    @abstractmethod
    def get_preprocessing_transform(self):
        """Returns the specific preprocessing steps required for this model."""
        pass


# Renamed class and updated registry key
@FEATURE_UPSAMPLER_REGISTRY.register('featup_jbu')
class JointBilateralUpsampler(BaseFeatureUpsampler):
    """ Implements FeatUp's Joint Bilateral Upsampler.

    Loads the specified backbone via torch.hub.
    """
    # Changed signature: Primarily rely on kwargs passed from config['params']
    def __init__(self, backbone_hub_id: str, use_norm: bool = True, target_size: int = 224, model_name: str = 'featup_jbu', **kwargs):
        # Store essential parameters directly
        self.backbone_hub_id = backbone_hub_id
        self.use_norm = use_norm
        self.target_size = target_size

        # Construct a descriptive model name
        full_model_name = f"{model_name}_{backbone_hub_id}"
        if not use_norm:
            full_model_name += "_noNorm"

        # Define normalization stats (could also be part of config later)
        # These are standard ImageNet stats, might not be used if FeatUp norm is different
        self.img_mean = [0.485, 0.456, 0.406]
        self.img_std = [0.229, 0.224, 0.225]

        # Call base init, which triggers _load_model
        # Pass only the constructed model name and potentially other base class args from kwargs
        super().__init__(model_name=full_model_name, extractor_model=None, **kwargs)

        # Determine feature dimension AFTER the backbone model is loaded
        self._upsampled_feature_dim = self._get_feature_dim_from_backbone(self.extractor_model)
        print(f"  Determined feature dimension: {self._upsampled_feature_dim}")

    def _load_model(self, **kwargs) -> nn.Module:
        # Use self.backbone_hub_id and self.use_norm here
        print(f"Loading FeatUp model via torch.hub: backbone='{self.backbone_hub_id}', use_norm={self.use_norm}...")
        try:
            # Add logging before the hub call
            print(f"  --> Starting torch.hub.load('mhamilton723/FeatUp', '{self.backbone_hub_id}', use_norm={self.use_norm})...")
            start_time = time.time() # Optional: Time the operation

            upsampler = torch.hub.load(
                "mhamilton723/FeatUp",
                self.backbone_hub_id,
                use_norm=self.use_norm,
                force_reload=False # Set to True to bypass cache if needed
            )

            # Add logging after the hub call
            end_time = time.time()
            print(f"  <-- torch.hub.load finished in {end_time - start_time:.2f} seconds.")

            # Store the reference to the internal backbone model
            if hasattr(upsampler, 'model') and isinstance(upsampler.model, nn.Module):
                 self.extractor_model = upsampler.model
                 print(f"  Stored internal backbone: {type(self.extractor_model).__name__}")
            else:
                 print("Warning: Could not find internal backbone model 'upsampler.model'")
                 self.extractor_model = None # Or handle as error?

            print(f"FeatUp model ({self.backbone_hub_id}, use_norm={self.use_norm}) loaded successfully.")
            return upsampler
        except Exception as e:
            print(f"ERROR loading FeatUp model from torch.hub: {e}")
            # Consider raising the error or returning a dummy model for testing
            # For now, let's raise it to make failures explicit
            raise RuntimeError(f"Failed to load FeatUp model '{self.backbone_hub_id}' from torch.hub") from e

    # Called from __init__ AFTER _load_model
    def _get_feature_dim_from_backbone(self, backbone_model: nn.Module | None) -> int:
        if backbone_model is None:
            print(f"Warning: Cannot determine feature dimension as backbone model was not loaded.")
            return 0 # Or raise error?

        # Try common attribute names for feature dimension
        if hasattr(backbone_model, 'embed_dim'): # Common in ViTs
            return backbone_model.embed_dim
        elif hasattr(backbone_model, 'num_features'): # Common in timm models
            return backbone_model.num_features
        # Add more checks if needed (e.g., checking output shape of last layer)
        else:
            # Fallback or raise error
            print(f"Warning: Could not automatically determine feature dimension for {self.backbone_hub_id}'s backbone {type(backbone_model).__name__}. Attempting fallback based on name.")
            if self.backbone_hub_id == 'dino8': return 384 # ViT-S/8 DINOv1
            if self.backbone_hub_id == 'dino16': return 384 # ViT-S/16 DINOv1
            if self.backbone_hub_id == 'dinov2': return 768 # ViT-S/14 DINOv2
            if self.backbone_hub_id == 'clip': return 512 # ViT-B/16 CLIP
            if self.backbone_hub_id == 'vit': return 768 # ViT-B/16 supervised
            if self.backbone_hub_id == 'resnet50': return 2048 # ResNet50
            # If none match, raise an error as we can't be sure
            raise AttributeError(f"Cannot determine feature dimension for backbone {type(backbone_model).__name__} from FeatUp model '{self.backbone_hub_id}'")

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Applies the loaded FeatUp upsampler model to the input tensor.

        Args:
            x: Input tensor, expected to be preprocessed (padded, normalized).
            **kwargs: Optional keyword arguments (currently unused by FeatUp's JBU).

        Returns:
            Upsampled feature tensor.
        """
        # The loaded FeatUp model handles backbone extraction and upsampling
        upsampled_features = self.upsampler_model(x)
        return upsampled_features

    @property
    def upsampled_feature_dim(self) -> int:
        # Return the dimension determined during init
        return self._upsampled_feature_dim

    def get_preprocessing_transform(self):
        # Use FeatUp's specific normalization function
        print(f"Returning JBU ({self.backbone_hub_id}) preprocessing transform (dtype->scale[0,1]->pad->channels->norm)...")

        # Custom transform to scale tensor to [0, 1]
        class ScaleTo01(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                min_val = torch.min(x)
                max_val = torch.max(x)
                if max_val > min_val:
                    # Add small epsilon for stability if max_val is close to min_val
                    return (x - min_val) / (max_val - min_val + 1e-6)
                else:
                    # Handle constant image (return all zeros or original constant)
                    return torch.zeros_like(x)
            def __repr__(self) -> str:
                 return f"{self.__class__.__name__}()"

        return transforms.Compose([
            # 1. Ensure float32 type first
            transforms.ToDtype(torch.float32, scale=False),
            # 2. Manually scale to [0, 1]
            ScaleTo01(),
            # 3. Pad to square, align top-center
            PadToSquareAndAlign(size=self.target_size, fill=0),
            # 4. Ensure 3 channels
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0]==1 else x),
            # 5. Apply FeatUp's normalization
            norm,
        ])

# Factory function remains largely the same, but interacts with new class/params
def build_feature_upsampler(config: dict, extractor: BaseFeatureExtractor | None = None) -> BaseFeatureUpsampler:
    """Builds a feature upsampler from a config dict."""
    upsampler_name = config.get('name') # Should be 'featup_jbu'
    params = config.get('params', {})   # Params should include 'backbone_hub_id'

    if not upsampler_name:
        raise ValueError("Config must contain an upsampler 'name'")
    if upsampler_name == 'featup_jbu' and 'backbone_hub_id' not in params:
        raise ValueError("Config for 'featup_jbu' must specify 'backbone_hub_id' in params")

    upsampler_class = FEATURE_UPSAMPLER_REGISTRY.get(upsampler_name)

    # Pass extractor if provided (might be needed for some future upsamplers)
    if extractor:
        params['extractor_model'] = extractor.model

    instance = upsampler_class(**params)
    return instance
