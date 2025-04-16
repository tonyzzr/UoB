# webapp/data_service/main.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse # To return image data
import uvicorn
import pickle
import os
import sys
from pathlib import Path
import io # For in-memory image data
import numpy as np
from PIL import Image # Import Pillow

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

# --- Helper function to load data (avoids repetition) ---
def _load_pickle_data(recording_id: str) -> dict:
    """Loads the combined_mvbv.pkl file for a given recording ID."""
    try:
        recording_dir = PROCESSED_DATA_DIR / recording_id
        pkl_file_path = recording_dir / 'combined_mvbv.pkl'
        print(f"Helper: Attempting to load PKL file: {pkl_file_path}")
        if not pkl_file_path.is_file():
            print(f"Helper Error: PKL file not found at {pkl_file_path}")
            raise HTTPException(status_code=404, detail=f"Pickle file for recording '{recording_id}' not found.")

        with open(pkl_file_path, 'rb') as f:
            combined_data = pickle.load(f)
        
        if not isinstance(combined_data, dict) or not all(k in combined_data for k in ['lftx', 'hftx']):
            raise TypeError(f"Loaded data from {pkl_file_path} is not a dictionary with 'lftx' and 'hftx' keys.")
            
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

@app.get("/ping")
async def ping():
    """Simple endpoint to check if the service is running."""
    return {"message": "Data service is running!"}

@app.get("/recordings/{recording_id}/details")
async def get_recording_details(recording_id: str):
    """Loads metadata details for a specific recording from its pkl file."""
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

# --- Run the server (for local development) ---
if __name__ == "__main__":
    # Run using Uvicorn. 'main:app' refers to the 'app' instance in the 'main.py' file.
    # --reload enables auto-reloading on code changes.
    # Port 8000 is common for secondary services.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 