import time
from typing import Dict, Any, List, Tuple, Optional
import torch # For feature cache type hint

# --- Cache Configuration ---
CACHE_TTL = 3600  # Time to live in seconds (1 hour)
MAX_CACHE_SIZE = 5  # Maximum number of recordings to keep in PKL cache
FEATURE_CACHE_MAX_FRAMES = 10 # Max number of frames to cache features for

# --- Cache Storage ---
# PKL Cache: {recording_id: {'data': loaded_data, 'last_access': timestamp}}
PKL_CACHE: Dict[str, Dict[str, Any]] = {}  
# Feature Cache: {(recording_id, frame_index): {view_index_0_15: features_tensor}}
FEATURE_CACHE: Dict[Tuple[str, int], Dict[int, torch.Tensor]] = {} 

# --- Helper functions for PKL cache management ---
def get_pkl_from_cache(recording_id: str) -> Optional[Dict[str, Any]]:
    """Get data from PKL cache if available and not expired"""
    if recording_id in PKL_CACHE:
        # Update last access time
        PKL_CACHE[recording_id]['last_access'] = time.time()
        print(f"[Cache] PKL Cache hit for recording: {recording_id}")
        return PKL_CACHE[recording_id]['data']
    print(f"[Cache] PKL Cache miss for recording: {recording_id}")
    return None

def add_pkl_to_cache(recording_id: str, data: dict) -> None:
    """Add data to PKL cache, managing cache size"""
    # If cache is full, remove least recently used entry
    if len(PKL_CACHE) >= MAX_CACHE_SIZE and recording_id not in PKL_CACHE:
        # Find least recently accessed recording
        try:
            oldest_id = min(PKL_CACHE.keys(), key=lambda k: PKL_CACHE[k]['last_access'])
            print(f"[Cache] PKL Cache full, removing oldest entry: {oldest_id}")
            del PKL_CACHE[oldest_id]
        except ValueError: # Cache might be empty
            pass
    
    # Add or update cache entry
    PKL_CACHE[recording_id] = {
        'data': data,
        'last_access': time.time()
    }
    print(f"[Cache] Added/updated PKL cache for recording: {recording_id}")

def cleanup_pkl_cache() -> None:
    """Remove expired entries from PKL cache"""
    current_time = time.time()
    expired_keys = [
        k for k, v in PKL_CACHE.items() 
        if current_time - v['last_access'] > CACHE_TTL
    ]
    for key in expired_keys:
        print(f"[Cache] Removing expired PKL cache entry: {key}")
        try:
            del PKL_CACHE[key]
        except KeyError:
            pass # Might have been removed already

# --- Helper functions for Feature cache management (could be enhanced later) ---
def get_features_from_cache(recording_id: str, frame_index: int) -> Optional[Dict[int, torch.Tensor]]:
    """Get features from cache if available."""
    cache_key = (recording_id, frame_index)
    if cache_key in FEATURE_CACHE:
        print(f"[Cache] Feature cache hit for {cache_key}")
        # TODO: Implement LRU or TTL for feature cache if needed
        return FEATURE_CACHE[cache_key]
    print(f"[Cache] Feature cache miss for {cache_key}")
    return None

def add_features_to_cache(recording_id: str, frame_index: int, features: Dict[int, torch.Tensor]) -> None:
    """Add features to cache, managing size."""
    cache_key = (recording_id, frame_index)
    # Manage cache size (simple strategy: remove one entry if full)
    if len(FEATURE_CACHE) >= FEATURE_CACHE_MAX_FRAMES and cache_key not in FEATURE_CACHE:
        # This requires storing access times or using LRU logic, 
        # for simplicity, let's just clear one entry if full (not ideal)
        try:
             oldest_key = next(iter(FEATURE_CACHE))
             print(f"[Cache] Feature cache full ({len(FEATURE_CACHE)}/{FEATURE_CACHE_MAX_FRAMES}), removing entry for {oldest_key}")
             del FEATURE_CACHE[oldest_key]
        except StopIteration:
             pass # Cache was already empty
             
    FEATURE_CACHE[cache_key] = features
    print(f"[Cache] Added features to cache for {cache_key}")

def clear_all_caches():
    """Clears both PKL and Feature caches."""
    global PKL_CACHE, FEATURE_CACHE
    pkl_size = len(PKL_CACHE)
    feat_size = len(FEATURE_CACHE)
    PKL_CACHE = {}
    FEATURE_CACHE = {}
    print(f"[Cache] Cleared PKL ({pkl_size} entries) and Feature ({feat_size} entries) caches.")
    return pkl_size, feat_size 