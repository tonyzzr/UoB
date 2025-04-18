# webapp/data_service/routers/recordings.py
import os
import io
import time
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import APIRouter, HTTPException, Query, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from PIL import Image

# Import dependencies from other service modules
from ..config import PROCESSED_DATA_DIR, UOB_AVAILABLE, MultiViewBmodeVideo, DEVICE, DEFAULT_FEATURE_CONFIG
from ..utils import load_pickle_data # numpy_to_data_uri (now used only in visualize_features)
from ..features import get_or_compute_frame_features # Keep helper import

# Import UoB visualization functions if available
if UOB_AVAILABLE:
    try:
        from src.UoB.visualization.plot_features import plot_feature_pca_comparison
        from src.UoB.visualization.plot_correspondence import plot_correspondence_grid
    except ImportError:
        print("[Router] Warning: Could not import plot_feature_pca_comparison or plot_correspondence_grid.")
        plot_feature_pca_comparison = None
        plot_correspondence_grid = None
else:
    plot_feature_pca_comparison = None
    plot_correspondence_grid = None

router = APIRouter(
    prefix="/recordings/{recording_id}",
    tags=["recordings"]
)

# --- Pydantic Models --- (Moved CorrespondenceRequest here)
class CorrespondenceRequest(BaseModel):
    source_view_index: int = Field(..., ge=0, le=15, description="Index (0-15) of the source view (LF 0-7, HF 8-15)")
    poi_normalized: List[float] = Field(..., min_items=2, max_items=2, description="Normalized POI coordinates [y_norm, x_norm] (0.0-1.0)")
    # k: Optional[int] = Field(1, ge=1, description="Number of nearest neighbors to find (default: 1)") # k > 1 not implemented yet

# --- Endpoints --- 

@router.get("/details")
async def get_recording_details(recording_id: str):
    """Loads metadata details for a specific recording from its pkl file."""
    print(f"[Router] Request received for recording ID details: {recording_id}")
    try:
        combined_data = load_pickle_data(recording_id)
        lftx_data = combined_data['lftx']

        # Use getattr for safer access
        frame_count = getattr(lftx_data, 'n_frame', 'N/A')
        num_spatial_views = getattr(lftx_data, 'n_view', 'N/A')
        available_freq_views = list(combined_data.keys())

        details = {
            "id": recording_id,
            "frame_count": frame_count,
            "num_spatial_views": num_spatial_views,
            "available_freq_views": available_freq_views,
        }
        print(f"[Router] Successfully extracted details for {recording_id}.")
        return details
    except HTTPException as http_exc:
        raise http_exc
    except AttributeError as e:
         print(f"[Router] Error accessing expected attributes (e.g., n_frame, n_view) on unpickled object for {recording_id}: {e}", file=sys.stderr)
         raise HTTPException(status_code=500, detail=f"Data structure mismatch in file for recording '{recording_id}'.")
    except Exception as e:
        print(f"[Router] Error extracting details from unpickled object for {recording_id}: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Error processing data structure for recording '{recording_id}'.")

@router.get("/frames/{frame_index}")
async def get_frame_image(
    recording_id: str,
    frame_index: int,
    freq: str = Query(..., description="Frequency view ('lftx' or 'hftx')"),
    view: int = Query(..., description="Spatial view index (0-based)")
):
    """Loads a specific frame, converts it to PNG, and returns the image."""
    print(f"[Router] Request received for frame: rec={recording_id}, frame={frame_index}, freq={freq}, view={view}")
    try:
        combined_data = load_pickle_data(recording_id)

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
        
        # Select frame/view, move to CPU, convert to numpy
        frame_np = img_data_tensor[frame_index, view, :, :].cpu().numpy() 
        
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

        print(f"[Router] Successfully generated PNG for frame: rec={recording_id}, frame={frame_index}, freq={freq}, view={view}")
        # Return the image data as a streaming response
        return StreamingResponse(img_byte_arr, media_type="image/png")

    except HTTPException as http_exc:
        raise http_exc
    except (AttributeError, IndexError, TypeError) as e:
         print(f"[Router] Error accessing image data for frame request: {e}", file=sys.stderr)
         raise HTTPException(status_code=500, detail=f"Error accessing frame data structure: {e}")
    except Exception as e:
        print(f"[Router] Unexpected error generating frame image: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail="Internal server error generating frame image.")

@router.get("/visualize_features/{frame_index}")
async def visualize_features(
    request: Request, # Add request object to access app state
    recording_id: str, 
    frame_index: int
):
    """
    Computes features and PCA visualization for all 16 views of a frame
    and returns the resulting comparison plot as a PNG image.
    Uses masks to improve PCA quality.
    """
    start_time = time.time()
    
    # Access extractor and transform from app state via request
    feature_extractor = request.app.state.feature_extractor
    feature_transform = request.app.state.feature_transform

    print(f"[Visualize Check] UOB_AVAILABLE: {UOB_AVAILABLE}") # Keep checks for debugging
    print(f"[Visualize Check] feature_extractor is None: {feature_extractor is None}") 
    print(f"[Visualize Check] feature_transform is None: {feature_transform is None}")

    if not UOB_AVAILABLE or feature_extractor is None or feature_transform is None:
        print("[Visualize Check] Condition met, raising 503...")
        raise HTTPException(status_code=503, detail="Feature extraction module not available or correctly initialized in app state.")
    if plot_feature_pca_comparison is None:
        raise HTTPException(status_code=501, detail="Visualization function (plot_feature_pca_comparison) not available.")
        
    # Check if recording exists (basic check)
    recording_dir = PROCESSED_DATA_DIR / recording_id
    if not recording_dir.is_dir(): # More robust than os.listdir
        raise HTTPException(status_code=404, detail=f"Recording '{recording_id}' not found.")
        
    try:
        print("[Visualize] Endpoint entered.") # DEBUG
        # 1. Load data using the helper function
        combined_data: Dict[str, MultiViewBmodeVideo] = load_pickle_data(recording_id)
        lftx_mvbv = combined_data['lftx']
        hftx_mvbv = combined_data['hftx']
        print("[Visualize] Data loaded.") # DEBUG

        # Validate frame index
        num_frames = min(lftx_mvbv.n_frame, hftx_mvbv.n_frame)
        if not (0 <= frame_index < num_frames):
            raise HTTPException(status_code=404, detail=f"Frame index {frame_index} out of bounds (0-{num_frames-1}).")

        # Validate mask presence
        if not hasattr(lftx_mvbv, 'view_masks') or not hasattr(hftx_mvbv, 'view_masks') or \
           lftx_mvbv.view_masks is None or hftx_mvbv.view_masks is None:
           raise HTTPException(status_code=500, detail="View masks not found in the loaded data.")

        print("[Visualize] Starting feature/mask extraction loop...") # DEBUG
        # 2. Prepare lists for images, features, masks, and labels
        processed_images_list = []
        features_list = []
        processed_masks_list = [] # For boolean masks
        view_labels = [f"LF {i}" for i in range(lftx_mvbv.n_view)] + \
                      [f"HF {i}" for i in range(hftx_mvbv.n_view)]
        num_total_views = lftx_mvbv.n_view + hftx_mvbv.n_view
        MASK_THRESHOLD = 0.5 

        # 3. Extract features, preprocess images and masks for all views
        print(f"[Router] Processing features, images, and masks for recording '{recording_id}', frame {frame_index}...")
        extract_start_time = time.time()
        # Re-implement feature/mask extraction loop here, as it's specific to this endpoint's needs
        # (different from get_or_compute_frame_features which only gets features)
        with torch.no_grad():
             # LF Views
            for i in range(lftx_mvbv.n_view):
                img_tensor_raw = lftx_mvbv.view_images[frame_index, i].float()
                img_tensor_chw = img_tensor_raw.unsqueeze(0) if img_tensor_raw.ndim == 2 else img_tensor_raw
                preprocessed_img_chw = feature_transform(img_tensor_chw) 
                processed_images_list.append(preprocessed_img_chw.cpu()) # Store preprocessed image for plotting
                input_tensor_nchw = preprocessed_img_chw.unsqueeze(0).to(DEVICE)
                feat = feature_extractor(input_tensor_nchw)
                features_list.append(feat.cpu().numpy()) # Store features [1, C, H, W]
                
                mask_frame_idx = 0 if lftx_mvbv.view_masks.shape[0] == 1 else frame_index
                mask_tensor_raw = lftx_mvbv.view_masks[mask_frame_idx, i].float()
                mask_tensor_chw = mask_tensor_raw.unsqueeze(0)
                preprocessed_mask = feature_transform(mask_tensor_chw)
                if preprocessed_mask.shape[0] == 3:
                    preprocessed_mask_np = preprocessed_mask[0].cpu().numpy()
                else: # Assume 1 channel
                    preprocessed_mask_np = preprocessed_mask.squeeze(0).cpu().numpy()
                binary_mask = preprocessed_mask_np > MASK_THRESHOLD
                processed_masks_list.append(binary_mask)

            # HF Views
            for i in range(hftx_mvbv.n_view):
                img_tensor_raw = hftx_mvbv.view_images[frame_index, i].float()
                img_tensor_chw = img_tensor_raw.unsqueeze(0) if img_tensor_raw.ndim == 2 else img_tensor_raw
                preprocessed_img_chw = feature_transform(img_tensor_chw)
                processed_images_list.append(preprocessed_img_chw.cpu())
                input_tensor_nchw = preprocessed_img_chw.unsqueeze(0).to(DEVICE)
                feat = feature_extractor(input_tensor_nchw)
                features_list.append(feat.cpu().numpy())
                
                mask_frame_idx = 0 if hftx_mvbv.view_masks.shape[0] == 1 else frame_index
                mask_tensor_raw = hftx_mvbv.view_masks[mask_frame_idx, i].float()
                mask_tensor_chw = mask_tensor_raw.unsqueeze(0)
                preprocessed_mask = feature_transform(mask_tensor_chw)
                if preprocessed_mask.shape[0] == 3:
                    preprocessed_mask_np = preprocessed_mask[0].cpu().numpy()
                else: # Assume 1 channel
                    preprocessed_mask_np = preprocessed_mask.squeeze(0).cpu().numpy()
                binary_mask = preprocessed_mask_np > MASK_THRESHOLD
                processed_masks_list.append(binary_mask)
                
        extract_end_time = time.time()
        print(f"[Router] Feature/Mask extraction took {extract_end_time - extract_start_time:.2f}s")
        print("[Visualize] Feature/mask extraction loop finished.") # DEBUG

        # 4. Generate PCA comparison plot with masks, returning bytes
        print("[Router] Generating PCA comparison plot...")
        plot_start_time = time.time()
        try:
            plot_bytes = plot_feature_pca_comparison(
                input_images_to_plot=processed_images_list,
                features=features_list,
                masks=processed_masks_list, 
                feature_type_label=f"Features ({DEFAULT_FEATURE_CONFIG.upper()})",
                view_labels=view_labels,
                use_joint_pca=True,
                pca_n_components=3,
                num_cols=8, 
                return_bytes=True 
            )
            if plot_bytes is None:
                 raise RuntimeError("plot_feature_pca_comparison returned None unexpectedly.")
        except ValueError as e:
             print(f"[Router] Error during PCA plot generation: {e}", file=sys.stderr)
             raise HTTPException(status_code=500, detail=f"Error generating PCA plot: {e}")
             
        plot_end_time = time.time()
        print(f"[Router] PCA Plot generation took {plot_end_time - plot_start_time:.2f}s")

        print("[Visualize] Plot generated, preparing response...") # DEBUG
        # 5. Return the plot as a PNG image
        end_time = time.time()
        print(f"[Router] Total time for visualize_features request: {end_time - start_time:.2f}s")
        return StreamingResponse(io.BytesIO(plot_bytes), media_type="image/png")

    except HTTPException as http_exc:
        raise http_exc # Re-raise HTTP exceptions
    except Exception as e:
        print(f"[Router] Error processing visualize_features request for {recording_id}, frame {frame_index}: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Internal server error visualizing features: {e}")


@router.post("/correspondence/{frame_index}")
async def get_correspondence(
    request: Request, # Add request object
    recording_id: str,
    frame_index: int,
    req_body: CorrespondenceRequest # Rename Pydantic model param to avoid conflict
):
    """
    Finds the best matching point(s) in query views for a given source Point of Interest (POI).
    Assumes features for the frame are cached or computes them.
    Returns coordinates of the best match (1-NN) for each query view.
    """
    start_time = time.time()
    print(f"[Router] Correspondence request: rec={recording_id}, frame={frame_index}, req={req_body}")
    
    # Access extractor and transform from app state via request
    feature_extractor = request.app.state.feature_extractor
    feature_transform = request.app.state.feature_transform

    if feature_extractor is None or feature_transform is None:
        print("[Correspond Check] Extractor/Transform missing in app state, raising 503...")
        raise HTTPException(status_code=503, detail="Feature extraction module not correctly initialized in app state.")

    try:
        # 1. Get features for the requested frame (uses cache)
        # Use await since the helper function is async
        all_features = await get_or_compute_frame_features(
            recording_id, frame_index, feature_extractor, feature_transform
        )

        # 2. Get source features and dimensions
        source_view_idx = req_body.source_view_index
        if source_view_idx not in all_features:
             raise HTTPException(status_code=500, detail=f"Source features for view {source_view_idx} not found after computation.")
        feats_s = all_features[source_view_idx]
        C, feat_h, feat_w = feats_s.shape
        
        # 3. Convert normalized POI to feature map coords
        y_norm, x_norm = req_body.poi_normalized
        if not (0.0 <= y_norm <= 1.0 and 0.0 <= x_norm <= 1.0):
            raise HTTPException(status_code=400, detail="Normalized POI coordinates must be between 0.0 and 1.0.")
        poi_r = int(y_norm * (feat_h - 1))
        poi_c = int(x_norm * (feat_w - 1))
        poi_r = max(0, min(poi_r, feat_h - 1))
        poi_c = max(0, min(poi_c, feat_w - 1))
        print(f"[Router] Source View: {source_view_idx}. Feature Map Size: ({feat_h}x{feat_w}). Converted POI ({y_norm:.3f}, {x_norm:.3f}) to feature coords ({poi_r}, {poi_c})")

        # 4. Isolate POI feature vector
        poi_flat_idx = poi_r * feat_w + poi_c
        feats_s_flat = feats_s.reshape(C, -1)
        poi_feat_vec = feats_s_flat[:, poi_flat_idx].clone()
        poi_feat_vec = poi_feat_vec.unsqueeze(1).to(DEVICE)
        
        # 1. Normalize POI vector (L2 norm along feature dim)
        poi_feat_vec_norm = F.normalize(poi_feat_vec, p=2, dim=0) # Shape [C, 1]

        # 5. Find best match in each query view
        match_results: Dict[int, List[int]] = {}
        num_views = len(all_features)
        match_start_time = time.time()

        for q_idx in range(num_views):
            if q_idx == source_view_idx:
                continue 
                
            feats_q = all_features[q_idx].to(DEVICE)
            C_q, feat_h_q, feat_w_q = feats_q.shape
            if feat_h_q != feat_h or feat_w_q != feat_w:
                 print(f"[Router] Warning: Feature map size mismatch between source view {source_view_idx} ({feat_h}x{feat_w}) and query view {q_idx} ({feat_h_q}x{feat_w_q}). Skipping match.", file=sys.stderr)
                 match_results[q_idx] = [-1, -1] 
                 continue
                 
            feats_q_flat = feats_q.reshape(C_q, -1) 
            
            # 2. Normalize query features (L2 norm along feature dim)
            feats_q_flat_norm = F.normalize(feats_q_flat, p=2, dim=0) # Shape [C, H*W]

            # 3. Compute dot product of normalized vectors (Cosine Similarity)
            similarities = torch.matmul(poi_feat_vec_norm.T, feats_q_flat_norm).squeeze()
            
            # Find best match (1-NN)
            match_val, match_flat_idx = torch.max(similarities, dim=0)
            
            # Convert flat index to 2D coords
            match_r = torch.div(match_flat_idx, feat_w, rounding_mode='floor')
            match_c = match_flat_idx % feat_w
            
            # Store results as feature map coordinates [row, column]
            match_results[q_idx] = [match_r.item(), match_c.item()]
        
        match_end_time = time.time()
        print(f"[Router] Computed matches for {len(match_results)} query views in {match_end_time - match_start_time:.2f}s")

        # 6. Return results (feature map coordinates)
        end_time = time.time()
        print(f"[Router] Total time for correspondence request: {end_time - start_time:.2f}s")
        return JSONResponse(content=match_results)

    except IndexError as e:
         print(f"[Router] Error during feature caching/retrieval or POI indexing: {e}", file=sys.stderr)
         raise HTTPException(status_code=404, detail=str(e))
    except HTTPException as http_exc:
        raise http_exc 
    except Exception as e:
        print(f"[Router] Error processing correspondence request: {e}", file=sys.stderr)
        # Consider logging traceback here
        raise HTTPException(status_code=500, detail=f"Internal server error finding correspondence: {e}") 

@router.post("/visualize_correspondence/{frame_index}")
async def visualize_correspondence(
    request: Request,
    recording_id: str,
    frame_index: int,
    req_body: CorrespondenceRequest
):
    """
    Generates a matplotlib visualization of computed correspondences across all 16 views.
    Takes the same input as the correspondence endpoint but returns a PNG image instead of JSON data.
    """
    start_time = time.time()
    print(f"[Router] Correspondence visualization request: rec={recording_id}, frame={frame_index}, req={req_body}")
    
    # Access extractor and transform from app state
    feature_extractor = request.app.state.feature_extractor
    feature_transform = request.app.state.feature_transform
    
    if feature_extractor is None or feature_transform is None:
        raise HTTPException(status_code=503, detail="Feature extraction module not correctly initialized in app state.")
    
    # Import visualization function
    if UOB_AVAILABLE:
        try:
            from src.UoB.visualization.plot_correspondence import plot_correspondence_grid
        except ImportError:
            print("[Router] Warning: Could not import plot_correspondence_grid.")
            raise HTTPException(status_code=501, detail="Correspondence visualization function not available.")
    else:
        raise HTTPException(status_code=501, detail="UoB visualization module not available.")

    try:
        # 1. Load data
        combined_data: Dict[str, MultiViewBmodeVideo] = load_pickle_data(recording_id)
        lftx_mvbv = combined_data['lftx']
        hftx_mvbv = combined_data['hftx']
        
        # Validate frame index
        num_frames = min(lftx_mvbv.n_frame, hftx_mvbv.n_frame)
        if not (0 <= frame_index < num_frames):
            raise HTTPException(status_code=404, detail=f"Frame index {frame_index} out of bounds (0-{num_frames-1}).")
            
        # 2. Get features for all views
        all_features = await get_or_compute_frame_features(
            recording_id, frame_index, feature_extractor, feature_transform
        )
        
        # 3. Extract images for all views
        all_images = []
        view_labels = []
        
        # LF Images (0-7)
        for i in range(lftx_mvbv.n_view):
            img = lftx_mvbv.view_images[frame_index, i].float()
            # Convert to numpy with proper normalization
            img_np = img.numpy() if not torch.is_tensor(img) else img.cpu().numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            all_images.append(img_np)
            view_labels.append(f"LF {i}")
            
        # HF Images (8-15)
        for i in range(hftx_mvbv.n_view):
            img = hftx_mvbv.view_images[frame_index, i].float()
            img_np = img.numpy() if not torch.is_tensor(img) else img.cpu().numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            all_images.append(img_np)
            view_labels.append(f"HF {i}")
        
        # 4. Compute correspondence matches
        source_view_idx = req_body.source_view_index
        y_norm, x_norm = req_body.poi_normalized
        feats_s = all_features[source_view_idx]
        C, feat_h, feat_w = feats_s.shape
        
        # Convert normalized POI to feature map coords
        poi_r = int(y_norm * (feat_h - 1))
        poi_c = int(x_norm * (feat_w - 1))
        poi_r = max(0, min(poi_r, feat_h - 1))
        poi_c = max(0, min(poi_c, feat_w - 1))
        
        # Compute matches for all views
        match_results = {}
        
        # Get POI feature vector
        poi_flat_idx = poi_r * feat_w + poi_c
        feats_s_flat = feats_s.reshape(C, -1)
        poi_feat_vec = feats_s_flat[:, poi_flat_idx].clone()
        poi_feat_vec = poi_feat_vec.unsqueeze(1).to(DEVICE)
        poi_feat_vec_norm = F.normalize(poi_feat_vec, p=2, dim=0)
        
        # For each query view, compute best match
        for q_idx in range(len(all_features)):
            if q_idx == source_view_idx:
                match_results[q_idx] = [poi_r, poi_c]  # Self-match
                continue
                
            feats_q = all_features[q_idx].to(DEVICE)
            C_q, feat_h_q, feat_w_q = feats_q.shape
            
            if feat_h_q != feat_h or feat_w_q != feat_w:
                print(f"[Router] Warning: Feature map size mismatch between views. Skipping match.")
                match_results[q_idx] = [-1, -1]
                continue
                
            feats_q_flat = feats_q.reshape(C_q, -1)
            feats_q_flat_norm = F.normalize(feats_q_flat, p=2, dim=0)
            
            # Compute cosine similarity
            similarities = torch.matmul(poi_feat_vec_norm.T, feats_q_flat_norm).squeeze()
            match_val, match_flat_idx = torch.max(similarities, dim=0)
            
            # Convert flat index to 2D coords
            match_r = torch.div(match_flat_idx, feat_w, rounding_mode='floor')
            match_c = match_flat_idx % feat_w
            
            match_results[q_idx] = [match_r.item(), match_c.item()]
        
        # 5. Generate visualization
        print(f"[Router] Generating correspondence visualization with matplotlib...")
        plot_start_time = time.time()
        
        # Convert feature map coordinates to normalized image coordinates
        # First create normalized POI coordinates
        poi_coords_norm = [y_norm, x_norm]
        
        # Convert all match coords to normalized
        match_coords_norm = {}
        for q_idx, coords in match_results.items():
            # Skip invalid matches
            if coords[0] < 0 or coords[1] < 0:
                match_coords_norm[q_idx] = None
                continue
                
            # Convert feature map coords to normalized (0-1)
            y_norm_q = coords[0] / (feat_h - 1)
            x_norm_q = coords[1] / (feat_w - 1)
            match_coords_norm[q_idx] = [y_norm_q, x_norm_q]
        
        # Call plotting function
        plot_bytes = plot_correspondence_grid(
            images=all_images,
            source_view_index=source_view_idx,
            poi_coords=poi_coords_norm,
            match_coords=match_coords_norm,
            view_labels=view_labels,
            return_bytes=True
        )
        
        if plot_bytes is None:
            raise RuntimeError("plot_correspondence_grid returned None unexpectedly.")
            
        plot_end_time = time.time()
        print(f"[Router] Correspondence visualization took {plot_end_time - plot_start_time:.2f}s")
        
        # 6. Return the plot as PNG
        end_time = time.time()
        print(f"[Router] Total time for correspondence visualization: {end_time - start_time:.2f}s")
        return StreamingResponse(io.BytesIO(plot_bytes), media_type="image/png")
        
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"[Router] Error processing correspondence visualization: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error visualizing correspondence: {e}") 