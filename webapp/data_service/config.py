# webapp/data_service/config.py
import os
import sys
from pathlib import Path
import torch

# --- Path Setup ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2] # config.py -> data_service -> webapp -> UoB (project root)
    # Add the Project Root to sys.path, so 'import src.UoB...' works
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    print(f"[Config] Added PROJECT_ROOT to sys.path: {PROJECT_ROOT}")
except Exception as e:
    print(f"[Config] Error setting up PROJECT_ROOT sys.path: {e}", file=sys.stderr)
    PROJECT_ROOT = Path('.').resolve().parents[1] # Fallback guess
    print(f"[Config] Falling back to PROJECT_ROOT: {PROJECT_ROOT}", file=sys.stderr)

# --- Data Directory ---
try:
    PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
    print(f"[Config] Expecting processed data in: {PROCESSED_DATA_DIR}")
    if not PROCESSED_DATA_DIR.is_dir():
         print(f"[Config] Warning: PROCESSED_DATA_DIR does not exist or is not a directory.", file=sys.stderr)
except NameError:
     print("[Config] Error: PROJECT_ROOT not defined, cannot set PROCESSED_DATA_DIR.", file=sys.stderr)
     PROCESSED_DATA_DIR = Path('../data/processed').resolve() # Fallback guess
     print(f"[Config] Falling back to PROCESSED_DATA_DIR: {PROCESSED_DATA_DIR}", file=sys.stderr)

# --- Feature Configuration ---
DEFAULT_FEATURE_CONFIG = "jbu_dino16"

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Config] Using device: {DEVICE}")

# --- UoB Availability Check (moved here for central check) ---
try:
    from src.UoB.data.formats import MultiViewBmodeVideo # Check if this specific class can be imported
    UOB_AVAILABLE = True
    print("[Config] Successfully imported UoB modules.")
except ImportError as e:
    UOB_AVAILABLE = False
    MultiViewBmodeVideo = None # Define as None if import fails
    print(f"[Config] Warning: Could not import UoB modules: {e}", file=sys.stderr)
    print("[Config] Feature-related endpoints might be disabled.", file=sys.stderr) 