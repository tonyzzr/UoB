import toml # Use third-party toml library for < Python 3.11
from pathlib import Path
from typing import Dict, Any


def load_toml_config(config_path: str | Path) -> Dict[str, Any]:
    """Loads a TOML configuration file.

    Args:
        config_path: Path to the TOML configuration file.

    Returns:
        A dictionary containing the parsed configuration.
        
    Raises:
        FileNotFoundError: If the config file does not exist.
        toml.TomlDecodeError: If the file is not valid TOML.
    """

    print(f"Loading config from: {config_path}")    

    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Use text mode 'r' for the toml library
    with open(config_path, "r") as f:
        try:
            print(f"Attempting to load config from: {config_path}")
            config_data = toml.load(f)
            print(f"Successfully loaded config from: {config_path}")
        except toml.TomlDecodeError as e:
            # Just re-raise the original error
            print(f"Error loading config from: {config_path}")
            print(f"Error: {e}")
            raise e

    print(f"Loaded config: {config_data}")
    return config_data

# --- Potential future additions ---
# def save_pickle(data: Any, path: str | Path):
#     ...

# def load_pickle(path: str | Path) -> Any:
#     ...
