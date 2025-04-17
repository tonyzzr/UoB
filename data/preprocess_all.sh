#!/bin/bash

# Set the base directory relative to the script's location
BASE_DIR=$(dirname "$(dirname "$(readlink -f "$0")")") 
RAW_DATA_DIR="$BASE_DIR/data/raw"
PROCESSED_DATA_DIR="$BASE_DIR/data/processed"
SCRIPT_PATH="$BASE_DIR/scripts/preprocess_data.py"
CONFIG_PATH="$BASE_DIR/configs/preprocessing/default.toml"

# Ensure the processed data directory exists
mkdir -p "$PROCESSED_DATA_DIR"

# Check if the script path exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Preprocessing script not found at $SCRIPT_PATH"
    exit 1
fi

# Check if the config path exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found at $CONFIG_PATH"
    exit 1
fi

# Check if the raw data directory exists
if [ ! -d "$RAW_DATA_DIR" ]; then
    echo "Error: Raw data directory not found at $RAW_DATA_DIR"
    exit 1
fi

echo "Starting batch preprocessing..."
echo "Raw data directory: $RAW_DATA_DIR"
echo "Processed data directory: $PROCESSED_DATA_DIR"
echo "Script path: $SCRIPT_PATH"
echo "Config path: $CONFIG_PATH"

# Find all directories directly under RAW_DATA_DIR and process them
find "$RAW_DATA_DIR" -mindepth 1 -maxdepth 1 -type d | while IFS= read -r recording_dir; do
    recording_name=$(basename "$recording_dir")
    echo "----------------------------------------"
    echo "Processing recording: $recording_name"
    echo "Input directory: $recording_dir"
    
    # Define the specific output directory for this recording
    output_subdir="$PROCESSED_DATA_DIR/$recording_name"
    
    # Run the preprocessing script
    python "$SCRIPT_PATH" \
        --input_dir "$recording_dir" \
        --output_dir "$PROCESSED_DATA_DIR" \
        --config_path "$CONFIG_PATH" \
        # The output filename is handled by the script creating a subdirectory named after the recording
        # inside the base output_dir and placing the default 'combined_mvbv.pkl' there.
        
    if [ $? -eq 0 ]; then
        echo "Successfully processed $recording_name. Output in $output_subdir"
    else
        echo "Error processing $recording_name."
    fi
    echo "----------------------------------------"
done

echo "Batch preprocessing finished." 