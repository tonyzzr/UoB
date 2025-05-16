# webapp/data_service/main.py
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

# Import the initialization function and routers
# Use absolute import relative to webapp dir when running uvicorn from webapp/

import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f"Project root: {project_root}")
sys.path.insert(0, project_root)

from data_service.features import init_feature_extractor
from data_service.routers import recordings, cache, misc
from data_service.cache import cleanup_pkl_cache

# --- Lifespan Management for Initialization/Cleanup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize the feature extractor
    print("[Main] Application startup: Initializing feature extractor...")
    extractor, transform = init_feature_extractor()
    if extractor is None or transform is None:
        print("[Main] Error: Feature extractor failed to initialize. Endpoints relying on it will fail.")
        # Optionally raise an error here to prevent startup if features are critical
    app.state.feature_extractor = extractor
    app.state.feature_transform = transform
    print("[Main] Feature extractor initialization complete.")
    yield
    # Shutdown: Optional cleanup (e.g., clearing caches explicitly if needed)
    print("[Main] Application shutdown: Cleaning up PKL cache...")
    cleanup_pkl_cache()
    # Clear other resources if necessary
    print("[Main] Cleanup complete.")

# --- FastAPI App Instance ---
# Pass the lifespan context manager to the FastAPI app
app = FastAPI(lifespan=lifespan)

# --- Include Routers ---
# The paths defined within these routers will be added to the app
app.include_router(misc.router)        # Includes /ping
app.include_router(recordings.router)  # Includes /recordings/...
app.include_router(cache.router)       # Includes /cache/...

# --- Run the server (for local development) ---
if __name__ == "__main__":
    # This block is mainly for direct execution (python main.py)
    # It might not be the primary way to run when using the full webapp structure.
    # When running with `uvicorn data_service.main:app` from the webapp/ dir,
    # uvicorn handles the app loading directly.
    print("[Main] Starting Uvicorn server via __main__ block...")
    # Pass the app object directly to uvicorn here
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
