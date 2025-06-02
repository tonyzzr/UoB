#!/bin/bash

# Set the base directory relative to the script's location
BASE_DIR=$(dirname "$(dirname "$(readlink -f "$0")")")
PROCESSED_DATA_DIR="$BASE_DIR/data/processed"
SCRIPT_PATH="$BASE_DIR/scripts/convert_mvbv_to_img_dataset.py"

# Check if the script path exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Conversion script not found at $SCRIPT_PATH"
    exit 1
fi

# Check if the processed data directory exists
if [ ! -d "$PROCESSED_DATA_DIR" ]; then
    echo "Error: Processed data directory not found at $PROCESSED_DATA_DIR"
    exit 1
fi

echo "Starting batch formatting..."
echo "Processed data directory: $PROCESSED_DATA_DIR"
echo "Script path: $SCRIPT_PATH"

# Find all directories directly under PROCESSED_DATA_DIR and process them
find "$PROCESSED_DATA_DIR" -mindepth 1 -maxdepth 1 -type d | while IFS= read -r recording_dir; do
    recording_name=$(basename "$recording_dir")
    pkl_file="$recording_dir/combined_mvbv.pkl"
    echo "----------------------------------------"
    echo "Formatting recording: $recording_name"
    echo "Input PKL file: $pkl_file"
    
    if [ -f "$pkl_file" ]; then
        python "$SCRIPT_PATH" "$pkl_file"
        if [ $? -eq 0 ]; then
            echo "Successfully formatted $recording_name."
        else
            echo "Error formatting $recording_name."
        fi
    else
        echo "combined_mvbv.pkl not found in $recording_dir, skipping."
    fi
    echo "----------------------------------------"
done

echo "Batch formatting finished."
