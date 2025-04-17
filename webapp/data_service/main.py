# webapp/data_service/main.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse # Added JSONResponse
import uvicorn
import pickle
import os
import sys
from pathlib import Path
import io # For in-memory image data
import numpy as np
from PIL import Image # Import Pillow
import time # For cache expiry
from typing import Dict, Any, List
import base64 # For Data URIs

# --- Add project root to sys.path to find UoB ---
# This assumes the script is run from the project root
# or that the working directory is correctly set.
# We might need to adjust this based on how the service is run (e.g., Docker).
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2] # data_service -> webapp -> UoB (project root)
    # Add the Project Root to sys.path, so 'import src.UoB...' works
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    print(f"Added PROJECT_ROOT to sys.path: {PROJECT_ROOT}")
    # Now imports like 'from src.UoB.data.formats import MultiViewBmodeVideo' should work if used during pickling
    # Placeholder: Replace with your actual class if needed for unpickling
    # from src.UoB.data.formats import MultiViewBmodeVideo # Example using src prefix
except ImportError as e:
    print(f"Error importing UoB modules: {e}", file=sys.stderr)
    # Decide how to handle this - maybe raise an error or proceed without UoB?
    # For now, we'll allow it to proceed but log the error.
    pass
except Exception as e:
    print(f"Error setting up sys.path: {e}", file=sys.stderr)
    pass

# --- Try importing necessary UoB and external libraries ---
try:
    import torch
    import torchvision.transforms.v2 as transforms # Use v2 if available
    from sklearn.decomposition import PCA
    from src.UoB.visualization.plot_features import apply_pca_to_features, fit_joint_pca
    from src.UoB.utils.transforms import PadToSquareAndAlign # Assuming this exists
    from src.UoB.features.upsamplers import build_feature_upsampler
    # Placeholder for data formats if needed for unpickling features later
    # from src.UoB.data.formats import MultiViewBmodeVideo
    UOB_AVAILABLE = True
    print("Successfully imported torch, sklearn, and UoB modules for PCA.")
except ImportError as e:
    UOB_AVAILABLE = False
    print(f"Warning: Could not import UoB modules or dependencies for feature visualization: {e}", file=sys.stderr)
    print("Feature visualization endpoint will be disabled.", file=sys.stderr)

app = FastAPI()

# --- Configuration ---
# Determine the data directory relative to the project root
try:
    PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
    print(f"Expecting processed data in: {PROCESSED_DATA_DIR}")
    if not PROCESSED_DATA_DIR.is_dir():
         print(f"Warning: PROCESSED_DATA_DIR does not exist or is not a directory.", file=sys.stderr)
except NameError:
     print("Error: PROJECT_ROOT not defined, cannot set PROCESSED_DATA_DIR.", file=sys.stderr)
     # Provide a default or raise an error if PROJECT_ROOT is critical
     PROCESSED_DATA_DIR = Path('../data/processed').resolve() # Fallback guess
     print(f"Falling back to PROCESSED_DATA_DIR: {PROCESSED_DATA_DIR}", file=sys.stderr)

# --- Cache for PKL files ---
# Simple in-memory cache for pickled data
PKL_CACHE: Dict[str, Dict[str, Any]] = {}  # {recording_id: {'data': loaded_data, 'last_access': timestamp}}
CACHE_TTL = 3600  # Time to live in seconds (1 hour)
MAX_CACHE_SIZE = 5  # Maximum number of recordings to keep in cache

# --- Initialize feature extractor at startup ---
FEATURE_EXTRACTOR = None
FEATURE_TRANSFORM = None
DEFAULT_FEATURE_CONFIG = "jbu_dino16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def init_feature_extractor():
    """Initialize the feature extractor (upsampler) on server startup"""
    global FEATURE_EXTRACTOR, FEATURE_TRANSFORM
    
    if not UOB_AVAILABLE:
        print("Feature extractor initialization skipped due to missing dependencies.")
        return
    
    try:
        config_path = PROJECT_ROOT / 'configs' / 'features' / f"{DEFAULT_FEATURE_CONFIG}.toml"
        print(f"Loading feature extractor config from: {config_path}")
        
        if not config_path.exists():
            print(f"Warning: Feature extractor config file not found: {config_path}", file=sys.stderr)
            return
            
        # Load config
        try:
            import tomllib
        except ImportError:
            import toml as tomllib
            
        with open(config_path, 'r', encoding='utf-8') as f:
            feature_config = tomllib.load(f)
            
        # Build feature extractor
        FEATURE_EXTRACTOR = build_feature_upsampler(feature_config)
        FEATURE_EXTRACTOR.to(DEVICE)
        FEATURE_EXTRACTOR.eval()
        
        # Get preprocessing transform
        FEATURE_TRANSFORM = FEATURE_EXTRACTOR.get_preprocessing_transform()
        
        print(f"Feature extractor '{DEFAULT_FEATURE_CONFIG}' loaded and moved to {DEVICE}")
    except Exception as e:
        print(f"Error initializing feature extractor: {e}", file=sys.stderr)

# Call initialization at startup
if UOB_AVAILABLE:
    init_feature_extractor()

# --- Helper functions for cache management ---
def _get_from_cache(recording_id: str) -> dict:
    """Get data from cache if available and not expired"""
    if recording_id in PKL_CACHE:
        # Update last access time
        PKL_CACHE[recording_id]['last_access'] = time.time()
        print(f"Cache hit for recording: {recording_id}")
        return PKL_CACHE[recording_id]['data']
    return None

def _add_to_cache(recording_id: str, data: dict) -> None:
    """Add data to cache, managing cache size"""
    # If cache is full, remove least recently used entry
    if len(PKL_CACHE) >= MAX_CACHE_SIZE and recording_id not in PKL_CACHE:
        # Find least recently accessed recording
        oldest_id = min(PKL_CACHE.keys(), key=lambda k: PKL_CACHE[k]['last_access'])
        print(f"Cache full, removing oldest entry: {oldest_id}")
        del PKL_CACHE[oldest_id]
    
    # Add or update cache entry
    PKL_CACHE[recording_id] = {
        'data': data,
        'last_access': time.time()
    }
    print(f"Added/updated cache for recording: {recording_id}")

def _cleanup_cache() -> None:
    """Remove expired entries from cache"""
    current_time = time.time()
    expired_keys = [
        k for k, v in PKL_CACHE.items() 
        if current_time - v['last_access'] > CACHE_TTL
    ]
    for key in expired_keys:
        print(f"Removing expired cache entry: {key}")
        del PKL_CACHE[key]

# --- Modified helper function to load data (with caching) ---
def _load_pickle_data(recording_id: str) -> dict:
    """Loads the combined_mvbv.pkl file for a given recording ID with caching."""
    try:
        # First check if data is in cache
        cached_data = _get_from_cache(recording_id)
        if cached_data is not None:
            return cached_data

        # If not in cache, load from file
        recording_dir = PROCESSED_DATA_DIR / recording_id
        pkl_file_path = recording_dir / 'combined_mvbv.pkl'
        print(f"Cache miss - Loading PKL file: {pkl_file_path}")
        
        if not pkl_file_path.is_file():
            print(f"Helper Error: PKL file not found at {pkl_file_path}")
            raise HTTPException(status_code=404, detail=f"Pickle file for recording '{recording_id}' not found.")

        with open(pkl_file_path, 'rb') as f:
            combined_data = pickle.load(f)
        
        if not isinstance(combined_data, dict) or not all(k in combined_data for k in ['lftx', 'hftx']):
            raise TypeError(f"Loaded data from {pkl_file_path} is not a dictionary with 'lftx' and 'hftx' keys.")
            
        # Add to cache before returning
        _add_to_cache(recording_id, combined_data)
        return combined_data
    except (pickle.UnpicklingError, ModuleNotFoundError, AttributeError) as e:
         print(f"Helper Error: Error unpickling file {pkl_file_path}: {e}", file=sys.stderr)
         raise HTTPException(status_code=500, detail=f"Error reading data file for recording '{recording_id}': {e}")
    except TypeError as e:
         print(f"Helper Error: Data structure mismatch in {pkl_file_path}: {e}", file=sys.stderr)
         raise HTTPException(status_code=500, detail=f"Data structure mismatch in file for recording '{recording_id}'.")
    except HTTPException as http_exc:
        raise http_exc # Re-raise specific HTTP errors
    except Exception as e:
        print(f"Helper Error: Unexpected error loading {pkl_file_path}: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Internal server error loading data for '{recording_id}'.")

# --- Helper function to extract features on-demand ---
def _extract_features(img_tensor):
    """Extract features from an image tensor using the loaded feature extractor"""
    if FEATURE_EXTRACTOR is None or FEATURE_TRANSFORM is None:
        raise ValueError("Feature extractor not initialized")
        
    with torch.no_grad():
        # Ensure input is properly preprocessed
        if img_tensor.ndim == 2:  # Add channel dimension if needed
            img_tensor = img_tensor.unsqueeze(0)
            
        # Apply preprocessing transform
        preprocessed_img = FEATURE_TRANSFORM(img_tensor)
        
        # Add batch dimension and move to device
        input_tensor = preprocessed_img.unsqueeze(0).to(DEVICE)
        
        # Extract features
        features = FEATURE_EXTRACTOR(input_tensor)
        
        # Return as CPU numpy array
        return features.cpu().numpy()

# --- Helper function to convert numpy array to Base64 PNG Data URI ---
def numpy_to_data_uri(img_np: np.ndarray) -> str:
    """Converts a NumPy array (H, W) or (H, W, C) to a PNG Data URI."""
    if img_np.dtype != np.uint8:
         # Normalize if not uint8 (e.g., PCA output scaled 0-1)
         min_val, max_val = np.min(img_np), np.max(img_np)
         if max_val > min_val:
             img_np = ((img_np - min_val) / (max_val - min_val) * 255)
         else:
             img_np = np.zeros_like(img_np)
         img_np = img_np.astype(np.uint8)
    
    mode = 'L' if img_np.ndim == 2 else 'RGB' # Grayscale or RGB
    if img_np.ndim == 3 and img_np.shape[-1] == 1:
        img_np = img_np.squeeze(-1) # Convert HWC grayscale to HW
        mode = 'L'
        
    img = Image.fromarray(img_np, mode=mode)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

@app.get("/ping")
async def ping():
    """Simple endpoint to check if the service is running."""
    return {"message": "Data service is running!"}

@app.get("/recordings/{recording_id}/details")
async def get_recording_details(recording_id: str):
    """Loads metadata details for a specific recording from its pkl file."""
    # Perform cache cleanup before processing
    _cleanup_cache()
    
    print(f"Request received for recording ID details: {recording_id}")
    try:
        combined_data = _load_pickle_data(recording_id)
        lftx_data = combined_data['lftx']

        frame_count = getattr(lftx_data, 'n_frame', 'N/A')
        num_spatial_views = getattr(lftx_data, 'n_view', 'N/A')
        available_freq_views = list(combined_data.keys())

        details = {
            "id": recording_id,
            # "pkl_path": str(PROCESSED_DATA_DIR / recording_id / 'combined_mvbv.pkl'), # Maybe omit path from response
            "frame_count": frame_count,
            "num_spatial_views": num_spatial_views,
            "available_freq_views": available_freq_views,
        }
        print(f"Successfully extracted details for {recording_id}.")
        return details
    except HTTPException as http_exc:
        raise http_exc
    except AttributeError as e:
         print(f"Error accessing expected attributes (e.g., n_frame, n_view) on unpickled object for {recording_id}: {e}", file=sys.stderr)
         raise HTTPException(status_code=500, detail=f"Data structure mismatch in file for recording '{recording_id}'.")
    except Exception as e:
        print(f"Error extracting details from unpickled object for {recording_id}: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Error processing data structure for recording '{recording_id}'.")

@app.get("/recordings/{recording_id}/frames/{frame_index}")
async def get_frame_image(
    recording_id: str,
    frame_index: int,
    freq: str = Query(..., description="Frequency view ('lftx' or 'hftx')"),
    view: int = Query(..., description="Spatial view index (0-based)")
):
    """Loads a specific frame, converts it to PNG, and returns the image."""
    print(f"Request received for frame: rec={recording_id}, frame={frame_index}, freq={freq}, view={view}")
    try:
        combined_data = _load_pickle_data(recording_id)

        if freq not in combined_data:
            raise HTTPException(status_code=400, detail=f"Invalid frequency view '{freq}'. Available: {list(combined_data.keys())}")

        mvbv_data = combined_data[freq]
        
        # Validate indices
        num_frames = getattr(mvbv_data, 'n_frame', 0)
        num_views = getattr(mvbv_data, 'n_view', 0)
        if not isinstance(num_frames, int) or not isinstance(num_views, int):
             raise TypeError("Could not determine valid frame/view counts from data object.")
             
        if not (0 <= frame_index < num_frames):
            raise HTTPException(status_code=400, detail=f"Invalid frame index {frame_index}. Must be 0 to {num_frames - 1}.")
        if not (0 <= view < num_views):
            raise HTTPException(status_code=400, detail=f"Invalid spatial view index {view}. Must be 0 to {num_views - 1}.")

        # Access the image data (expected shape: [n_frame, n_view, h, w])
        img_data_tensor = getattr(mvbv_data, 'view_images', None)
        if img_data_tensor is None:
             raise AttributeError("Could not find 'view_images' attribute in data object.")
        
        # Convert tensor to numpy if necessary, select the frame/view
        # Assuming it's already a torch tensor from conversion
        frame_np = img_data_tensor[frame_index, view, :, :].cpu().numpy() # Select frame/view, move to CPU, convert to numpy
        
        # Normalize image data to 0-255 for PNG conversion
        min_val, max_val = np.min(frame_np), np.max(frame_np)
        if max_val > min_val:
            normalized_frame = ((frame_np - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            normalized_frame = np.zeros_like(frame_np, dtype=np.uint8) # Handle flat image

        # Convert numpy array to PNG image in memory
        img = Image.fromarray(normalized_frame, mode='L') # 'L' for grayscale
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0) # Rewind buffer to start

        print(f"Successfully generated PNG for frame: rec={recording_id}, frame={frame_index}, freq={freq}, view={view}")
        # Return the image data as a streaming response
        return StreamingResponse(img_byte_arr, media_type="image/png")

    except HTTPException as http_exc:
        raise http_exc
    except (AttributeError, IndexError, TypeError) as e:
         print(f"Error accessing image data for frame request: {e}", file=sys.stderr)
         raise HTTPException(status_code=500, detail=f"Error accessing frame data structure: {e}")
    except Exception as e:
        print(f"Unexpected error generating frame image: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail="Internal server error generating frame image.")

# --- Modified Endpoint for Feature Visualization with On-Demand Feature Extraction --- 
@app.get("/recordings/{recording_id}/visualize_features/{frame_index}")
async def visualize_features(recording_id: str, frame_index: int):
    if not UOB_AVAILABLE:
        raise HTTPException(status_code=501, detail="Feature visualization disabled due to missing dependencies.")
    
    if FEATURE_EXTRACTOR is None or FEATURE_TRANSFORM is None:
        raise HTTPException(status_code=501, detail="Feature extractor not initialized. The server may need to be restarted.")
        
    print(f"Request received for feature visualization: rec={recording_id}, frame={frame_index}")
    _cleanup_cache()
    
    try:
        # 1. Load Raw Image Data
        combined_data = _load_pickle_data(recording_id)
        lftx_data_obj = combined_data.get('lftx')
        hftx_data_obj = combined_data.get('hftx')
        if not lftx_data_obj or not hftx_data_obj:
             raise HTTPException(status_code=404, detail="LF/HF data not found in PKL.")

        lftx_images = getattr(lftx_data_obj, 'view_images', None) # [T, N_Views, H, W]
        hftx_images = getattr(hftx_data_obj, 'view_images', None)
        if lftx_images is None or hftx_images is None:
             raise AttributeError("Could not find 'view_images' attribute.")
             
        num_lf_views = lftx_images.shape[1]
        num_hf_views = hftx_images.shape[1]
        num_total_views = num_lf_views + num_hf_views
        if num_total_views == 0:
             raise ValueError("No views found in the data.")
             
        # Validate frame index
        num_frames = lftx_images.shape[0]
        if not (0 <= frame_index < num_frames):
            raise HTTPException(status_code=400, detail=f"Invalid frame index {frame_index}. Must be 0 to {num_frames - 1}.")

        # 2. Preprocess Input Images & Extract Features On-Demand
        input_images_preprocessed = []
        extracted_features_list = []
        view_labels = [f"LF {i}" for i in range(num_lf_views)] + [f"HF {i}" for i in range(num_hf_views)]
        
        print(f"Processing {num_total_views} views for frame {frame_index}...")
        start_extract_time = time.time()

        # --- Process LF images ---
        for i in range(num_lf_views):
            # Get image
            img = lftx_images[frame_index, i]
            
            # Preprocess for display and input
            img_tensor = img.unsqueeze(0) if img.ndim == 2 else img
            input_images_preprocessed.append(FEATURE_TRANSFORM(img_tensor))
            
            # Extract features on-demand
            try:
                features = _extract_features(img)
                extracted_features_list.append(features)
            except Exception as extract_err:
                raise HTTPException(status_code=500, detail=f"Error extracting features for LF view {i}: {extract_err}")

        # --- Process HF images ---
        for i in range(num_hf_views):
            # Get image
            img = hftx_images[frame_index, i]
            
            # Preprocess for display and input
            img_tensor = img.unsqueeze(0) if img.ndim == 2 else img
            input_images_preprocessed.append(FEATURE_TRANSFORM(img_tensor))
            
            # Extract features on-demand
            try:
                features = _extract_features(img)
                extracted_features_list.append(features)
            except Exception as extract_err:
                raise HTTPException(status_code=500, detail=f"Error extracting features for HF view {i}: {extract_err}")
        
        end_extract_time = time.time()
        print(f"Feature extraction completed in {end_extract_time - start_extract_time:.2f} seconds")
        
        # Get dimensions from the first preprocessed image for logging/sanity check
        if not input_images_preprocessed:
            raise ValueError("Preprocessing failed, no images generated.")
        _, H_proc, W_proc = input_images_preprocessed[0].shape
        print(f"Generated {len(extracted_features_list)} feature sets. Preprocessed image size: H={H_proc}, W={W_proc}")

        # 3. Apply Joint PCA (using the extracted features)
        print("Applying Joint PCA on extracted features...")
        start_pca_time = time.time()
        try:
            joint_pca_model = fit_joint_pca(extracted_features_list, n_components=3)
            pca_results_list = []
            for features_np in extracted_features_list:
                pca_img_normalized, _ = apply_pca_to_features(features_np, pca_model=joint_pca_model)
                pca_results_list.append(pca_img_normalized[0]) # Get [H, W, 3]
        except Exception as pca_error:
             print(f"Error during PCA processing: {pca_error}", file=sys.stderr)
             raise HTTPException(status_code=500, detail=f"Error during PCA: {pca_error}")
        end_pca_time = time.time()
        print(f"PCA completed in {end_pca_time - start_pca_time:.2f} seconds.")
        
        # 4. Convert Images to Data URIs
        input_image_uris = []
        for img_tensor in input_images_preprocessed:
            # Convert tensor to numpy HWC for visualization
            img_np = img_tensor.cpu().numpy()
            if img_np.ndim == 3 and img_np.shape[0] in [1, 3]: # CHW -> HWC
                img_np = np.transpose(img_np, (1, 2, 0))
            input_image_uris.append(numpy_to_data_uri(img_np))
        
        pca_image_uris = [numpy_to_data_uri(pca_img) for pca_img in pca_results_list]
        print("Image conversion complete.")

        # 5. Return JSON Response
        return JSONResponse(content={
            "recording_id": recording_id,
            "frame_index": frame_index,
            "input_image_uris": input_image_uris, # List of 16 Data URIs
            "pca_image_uris": pca_image_uris,     # List of 16 Data URIs
            "view_labels": view_labels, # Pass labels generated earlier
            "feature_extractor": DEFAULT_FEATURE_CONFIG,
            "processing_time": {
                "extraction_ms": int((end_extract_time - start_extract_time) * 1000),
                "pca_ms": int((end_pca_time - start_pca_time) * 1000)
            }
        })

    except HTTPException as http_exc:
        raise http_exc
    except (AttributeError, IndexError, TypeError, ValueError) as e:
         print(f"Error processing data for feature visualization: {e}", file=sys.stderr)
         raise HTTPException(status_code=500, detail=f"Error processing data: {e}")
    except Exception as e:
        print(f"Unexpected error during feature visualization: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail="Internal server error during feature visualization.")

# New endpoint to view cache status
@app.get("/cache/status")
async def get_cache_status():
    """Returns information about the current cache state."""
    return {
        "cache_size": len(PKL_CACHE),
        "max_cache_size": MAX_CACHE_SIZE,
        "cache_ttl_seconds": CACHE_TTL,
        "cached_recordings": [
            {
                "id": rec_id,
                "last_accessed": PKL_CACHE[rec_id]['last_access'],
                "age_seconds": time.time() - PKL_CACHE[rec_id]['last_access']
            }
            for rec_id in PKL_CACHE
        ],
        "feature_extractor": {
            "name": DEFAULT_FEATURE_CONFIG,
            "initialized": FEATURE_EXTRACTOR is not None,
            "device": str(DEVICE)
        }
    }

# New endpoint to manually clear the cache
@app.post("/cache/clear")
async def clear_cache():
    """Clears the entire PKL cache."""
    global PKL_CACHE
    cache_size = len(PKL_CACHE)
    PKL_CACHE = {}
    return {"message": f"Cache cleared. {cache_size} entries removed."}

# --- Run the server (for local development) ---
if __name__ == "__main__":
    # Run using Uvicorn. 'main:app' refers to the 'app' instance in the 'main.py' file.
    # --reload enables auto-reloading on code changes.
    # Port 8000 is common for secondary services.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 