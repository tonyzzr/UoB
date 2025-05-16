import pickle
import io
import base64
import numpy as np
from PIL import Image
from pathlib import Path
from fastapi import HTTPException
from typing import Dict, Any

# Import necessary functions and variables from other modules
from .config import PROCESSED_DATA_DIR, MultiViewBmodeVideo # Assuming MultiViewBmodeVideo type hint is useful
from .cache import get_pkl_from_cache, add_pkl_to_cache

# --- Data Loading Utility (depends on config, cache) ---
def load_pickle_data(recording_id: str) -> Dict[str, Any]:
    """Loads the combined_mvbv.pkl file for a given recording ID with caching."""
    try:
        # First check if data is in cache
        cached_data = get_pkl_from_cache(recording_id)
        if cached_data is not None:
            return cached_data

        # If not in cache, load from file
        recording_dir = PROCESSED_DATA_DIR / recording_id
        pkl_file_path = recording_dir / 'combined_mvbv.pkl'
        print(f"[Utils] Loading PKL file: {pkl_file_path}")
        
        if not pkl_file_path.is_file():
            print(f"[Utils] Error: PKL file not found at {pkl_file_path}")
            raise HTTPException(status_code=404, detail=f"Pickle file for recording '{recording_id}' not found.")

        with open(pkl_file_path, 'rb') as f:
            combined_data = pickle.load(f)
        
        # Simple validation (can be enhanced)
        if not isinstance(combined_data, dict) or not all(k in combined_data for k in ['lftx', 'hftx']): # Check for expected keys
             if MultiViewBmodeVideo is not None and not all(isinstance(v, MultiViewBmodeVideo) for v in combined_data.values()):
                 raise TypeError(f"Loaded data values are not all MultiViewBmodeVideo objects.")
             elif MultiViewBmodeVideo is None: # If UoB not available, just check basic structure
                 if not isinstance(combined_data.get('lftx'), object) or not isinstance(combined_data.get('hftx'), object):
                      raise TypeError(f"Loaded data does not contain expected objects for 'lftx' and 'hftx'.")
             # Fallback if keys are present but MultiViewBmodeVideo check fails/not possible
             # raise TypeError(f"Loaded data from {pkl_file_path} does not conform to expected structure.")
            
        # Add to cache before returning
        add_pkl_to_cache(recording_id, combined_data)
        return combined_data
    except (pickle.UnpicklingError, ModuleNotFoundError, AttributeError) as e:
         print(f"[Utils] Error: Error unpickling file {pkl_file_path}: {e}", file=sys.stderr)
         raise HTTPException(status_code=500, detail=f"Error reading data file for recording '{recording_id}': {e}")
    except TypeError as e:
         print(f"[Utils] Error: Data structure mismatch in {pkl_file_path}: {e}", file=sys.stderr)
         raise HTTPException(status_code=500, detail=f"Data structure mismatch in file for recording '{recording_id}'.")
    except HTTPException as http_exc:
        raise http_exc # Re-raise specific HTTP errors
    except Exception as e:
        print(f"[Utils] Error: Unexpected error loading {pkl_file_path}: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Internal server error loading data for '{recording_id}'.")


# --- Image Utility ---
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