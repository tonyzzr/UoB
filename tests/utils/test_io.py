import pytest
import sys
import os
import toml # Use third-party toml library for < Python 3.11
from pathlib import Path

# Add project root to the Python path to find the 'src' module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Import the io module
from src.UoB.utils import io

# --- Test load_toml_config ---

def test_load_toml_config_success():
    """ Test loading a valid TOML file (using the default preprocessing config). """
    # Assuming this test runs from the project root or pytest handles paths correctly
    valid_toml_path = Path("configs/preprocessing/default.toml")
    assert valid_toml_path.exists(), f"Test setup error: {valid_toml_path} not found."
    
    config_data = io.load_toml_config(valid_toml_path)
    
    assert isinstance(config_data, dict)
    # Check for some expected top-level keys
    assert "general" in config_data
    assert "lftx" in config_data
    assert "hftx" in config_data
    # Check for a nested key
    assert "mask" in config_data["lftx"]
    assert "enable" in config_data["lftx"]["mask"]

def test_load_toml_config_file_not_found():
    """ Test that FileNotFoundError is raised for a non-existent file. """
    invalid_path = Path("non_existent_config_file.toml")
    assert not invalid_path.exists() # Ensure it doesn't exist
    
    with pytest.raises(FileNotFoundError):
        io.load_toml_config(invalid_path)

def test_load_toml_config_decode_error(tmp_path):
    """ Test that TomlDecodeError is raised for an invalid TOML file. """
    invalid_toml_content = "this is not valid toml content = ["
    invalid_toml_file = tmp_path / "invalid.toml"
    invalid_toml_file.write_text(invalid_toml_content)
    
    with pytest.raises(toml.TomlDecodeError):
        io.load_toml_config(invalid_toml_file)

# --- Add tests for other io functions (e.g., pickle) when implemented --- 
