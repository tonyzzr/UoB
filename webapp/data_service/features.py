# webapp/data_service/features.py
import torch
import torch.nn.functional as F
import time
from typing import Dict, Any, List, Tuple, Optional
from fastapi import HTTPException

# Import from other modules
from .config import (PROJECT_ROOT, DEFAULT_FEATURE_CONFIG, DEVICE, 
                     UOB_AVAILABLE, MultiViewBmodeVideo)
from .cache import (get_features_from_cache, add_features_to_cache, 
                    FEATURE_CACHE, FEATURE_CACHE_MAX_FRAMES) # Need direct cache access for cleanup
from .utils import load_pickle_data

# --- Initialization Function ---
def init_feature_extractor():
    """Loads the feature extractor and transform, returning them."""
    
    extractor = None
    transform = None
    
    if not UOB_AVAILABLE:
        print("[Features] Feature extractor initialization skipped: UoB modules not available.")
        return None, None # Return None if not available
    
    # Import UoB modules only if available and needed
    try:
        from src.UoB.features.upsamplers import build_feature_upsampler
        try:
            import tomllib
        except ImportError:
            import toml as tomllib
    except ImportError as e:
         print(f"[Features] Error importing necessary modules for init: {e}", file=sys.stderr)
         # Set to None to prevent endpoint usage
         return None, None # Return None on import error
         
    try:
        config_path = PROJECT_ROOT / 'configs' / 'features' / f"{DEFAULT_FEATURE_CONFIG}.toml"
        print(f"[Features] Loading feature extractor config from: {config_path}")
        
        if not config_path.exists():
            print(f"[Features] Warning: Feature extractor config file not found: {config_path}", file=sys.stderr)
            return None, None
            
        with open(config_path, 'r', encoding='utf-8') as f:
            feature_config = tomllib.load(f)
            
        # Build feature extractor
        extractor = build_feature_upsampler(feature_config)
        extractor.to(DEVICE)
        extractor.eval()
        
        # Get preprocessing transform
        transform = extractor.get_preprocessing_transform()
        
        print(f"[Features] Feature extractor '{DEFAULT_FEATURE_CONFIG}' loaded and transform obtained.")
        return extractor, transform # Return the loaded objects
    except Exception as e:
        print(f"[Features] Error initializing feature extractor: {e}", file=sys.stderr)
        return None, None # Return None on error

# --- Feature Extraction Helper --- 
# Note: _extract_features (single image) is not currently used by endpoints, keep or remove?
# Let's keep it for now, but comment it out.
# def _extract_features(img_tensor):
#     """(Deprecated?) Extract features from a single image tensor using the loaded feature extractor"""
#     if FEATURE_EXTRACTOR is None or FEATURE_TRANSFORM is None:
#         raise ValueError("Feature extractor not initialized")
#     # ... (rest of the logic for single image)

# --- Frame Feature Computation and Caching Helper ---
async def get_or_compute_frame_features(
    recording_id: str, 
    frame_index: int,
    extractor: torch.nn.Module,
    transform: torch.nn.Module
) -> Dict[int, torch.Tensor]:
    """Gets features for all 16 views of a frame, using cache if available."""
    
    # Check cache first
    cached_features = get_features_from_cache(recording_id, frame_index)
    if cached_features is not None:
        return cached_features
        
    print(f"[Features] Feature cache miss for ({recording_id}, {frame_index}). Computing features...")
    
    if not UOB_AVAILABLE:
         raise HTTPException(status_code=503, detail="Feature extraction module not available or initialized.")
    # This check might become redundant if we fetch from app.state in endpoints
    # Let's keep it for now, but it assumes the caller passed valid objects
    # Or better: pass extractor/transform as arguments?
    # For now, let's remove this check - it will be done in the endpoint
    # if FEATURE_EXTRACTOR is None or FEATURE_TRANSFORM is None:
    #     raise HTTPException(status_code=503, detail="Feature extractor or transform not initialized.")
        
    # Load raw data
    try:
        combined_data: Dict[str, MultiViewBmodeVideo] = load_pickle_data(recording_id)
        lftx_mvbv = combined_data['lftx']
        hftx_mvbv = combined_data['hftx']
    except HTTPException as e:
         # Propagate HTTP errors from data loading
         raise e
    except Exception as e:
        print(f"[Features] Unexpected error loading data for feature extraction: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail="Internal error loading data before feature extraction.")

    # Validate frame index
    try:
        num_frames = min(lftx_mvbv.n_frame, hftx_mvbv.n_frame)
        if not (0 <= frame_index < num_frames):
            # Raise specific error that can be caught
            raise IndexError(f"Frame index {frame_index} out of bounds ({num_frames} frames) for feature extraction.")
    except AttributeError as e:
        print(f"[Features] Error accessing n_frame attribute: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail="Data structure missing frame count information.")
        
    features_all_views: Dict[int, torch.Tensor] = {}
    try:
        num_lf_views = lftx_mvbv.n_view
        num_hf_views = hftx_mvbv.n_view
    except AttributeError as e:
        print(f"[Features] Error accessing n_view attribute: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail="Data structure missing view count information.")
        
    start_time = time.time()

    try:
        with torch.no_grad():
            # LF Views (Indices 0 to num_lf_views-1)
            for i in range(num_lf_views):
                img_tensor_raw = lftx_mvbv.view_images[frame_index, i].float() # Shape [H, W]
                img_tensor_chw = img_tensor_raw.unsqueeze(0) if img_tensor_raw.ndim == 2 else img_tensor_raw # Shape [1, H, W]
                # Apply transform to [C, H, W] tensor
                preprocessed_img_chw = transform(img_tensor_chw) # Replace with passed transform
                # Add batch dim and move to device for extractor
                input_tensor_nchw = preprocessed_img_chw.unsqueeze(0).to(DEVICE) # Shape [1, 3, target_size, target_size]
                extractor_output = extractor(input_tensor_nchw)
                feat = extractor_output[0] # Get C, H, W from N=1 output
                features_all_views[i] = feat.cpu() # Store features on CPU

            # HF Views (Indices num_lf_views to num_lf_views + num_hf_views - 1)
            for i in range(num_hf_views):
                img_tensor_raw = hftx_mvbv.view_images[frame_index, i].float()
                img_tensor_chw = img_tensor_raw.unsqueeze(0) if img_tensor_raw.ndim == 2 else img_tensor_raw
                # Apply transform to [C, H, W] tensor
                preprocessed_img_chw = transform(img_tensor_chw) # Replace with passed transform
                # Add batch dim and move to device for extractor
                input_tensor_nchw = preprocessed_img_chw.unsqueeze(0).to(DEVICE) # Shape [1, 3, target_size, target_size]
                extractor_output = extractor(input_tensor_nchw)
                feat = extractor_output[0]
                features_all_views[num_lf_views + i] = feat.cpu()
    except Exception as e:
         print(f"[Features] Error during torch feature extraction loop: {e}", file=sys.stderr)
         # Consider logging traceback
         raise HTTPException(status_code=500, detail="Internal error during feature computation.")
            
    end_time = time.time()
    print(f"[Features] Computed features for ({recording_id}, {frame_index}) in {end_time - start_time:.2f}s")
    
    # Add to cache
    add_features_to_cache(recording_id, frame_index, features_all_views)
    return features_all_views 