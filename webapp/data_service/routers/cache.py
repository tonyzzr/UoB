# webapp/data_service/routers/cache.py
import time
from fastapi import APIRouter, Depends, Request

# Import dependencies from other service modules
from ..cache import (PKL_CACHE, FEATURE_CACHE, MAX_CACHE_SIZE, CACHE_TTL, 
                    FEATURE_CACHE_MAX_FRAMES, clear_all_caches, cleanup_pkl_cache)
from ..config import DEVICE, DEFAULT_FEATURE_CONFIG # Import config directly

router = APIRouter(
    prefix="/cache",
    tags=["cache"]
)

# --- Endpoints --- 

# Dependency to run PKL cache cleanup before status check
async def run_pkl_cleanup():
    cleanup_pkl_cache()

@router.get("/status", dependencies=[Depends(run_pkl_cleanup)])
async def get_cache_status(request: Request):
    """Returns information about the current cache state after cleaning expired PKL entries."""
    # PKL Cache Info
    pkl_cache_info = [
        {
            "id": rec_id,
            "last_accessed": PKL_CACHE[rec_id]['last_access'],
            "age_seconds": time.time() - PKL_CACHE[rec_id]['last_access']
        }
        for rec_id in PKL_CACHE
    ]
    
    # Feature Cache Info (Basic)
    feature_cache_keys = list(FEATURE_CACHE.keys())
    
    return {
        "pkl_cache": {
            "current_size": len(PKL_CACHE),
            "max_size": MAX_CACHE_SIZE,
            "ttl_seconds": CACHE_TTL,
            "entries": pkl_cache_info,
        },
        "feature_cache": {
            "current_frames": len(FEATURE_CACHE),
            "max_frames": FEATURE_CACHE_MAX_FRAMES,
            "keys": feature_cache_keys # List keys for basic info
        },
        "feature_extractor": {
            "name": DEFAULT_FEATURE_CONFIG,
            "initialized": hasattr(request.app.state, 'feature_extractor') and request.app.state.feature_extractor is not None,
            "device": str(DEVICE)
        }
    }

@router.post("/clear")
async def clear_cache_endpoint(): # Renamed function to avoid conflict
    """Clears the entire PKL and Feature caches."""
    pkl_removed, feat_removed = clear_all_caches()
    return {"message": f"Caches cleared. PKL: {pkl_removed} entries removed. Features: {feat_removed} frame entries removed."} 