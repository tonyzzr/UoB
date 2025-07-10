#!/bin/bash

# Set the base directory relative to the script's location
BASE_DIR=$(dirname "$(dirname "$(readlink -f "$0")")") 
RAW_DATA_DIR="$BASE_DIR/data/raw"
FORMATTED_DATA_DIR="$BASE_DIR/data/formatted"
SCRIPT_PATH="$BASE_DIR/scripts/preprocess_and_convert_to_dataset.py"
CONFIG_PATH="$BASE_DIR/configs/preprocessing/default.toml"

# Ensure the formatted data directory exists
mkdir -p "$FORMATTED_DATA_DIR"

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

echo "Starting batch conversion of raw data to PNG datasets..."
echo "Raw data directory: $RAW_DATA_DIR"
echo "Formatted data directory: $FORMATTED_DATA_DIR"
echo "Script path: $SCRIPT_PATH"
echo "Config path: $CONFIG_PATH"
echo ""

# Count total directories to process
total_dirs=$(find "$RAW_DATA_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
current_dir=0

echo "Found $total_dirs recording directories to process..."
echo ""

# Find all directories directly under RAW_DATA_DIR and process them
find "$RAW_DATA_DIR" -mindepth 1 -maxdepth 1 -type d | sort | while IFS= read -r recording_dir; do
    current_dir=$((current_dir + 1))
    recording_name=$(basename "$recording_dir")
    echo "=========================================="
    echo "Processing ($current_dir/$total_dirs): $recording_name"
    echo "Input directory: $recording_dir"
    
    # Define the expected output directory for this recording
    output_subdir="$FORMATTED_DATA_DIR/$recording_name"
    
    # Check if output already exists and ask user
    if [ -d "$output_subdir" ]; then
        echo "Warning: Output directory already exists: $output_subdir"
        echo "Do you want to overwrite it? [y/N]"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping $recording_name..."
            echo "=========================================="
            echo ""
            continue
        fi
        echo "Removing existing output directory..."
        rm -rf "$output_subdir"
    fi
    
    echo "Starting conversion to PNG dataset..."
    start_time=$(date +%s)
    
    # Run the memory-efficient preprocessing and conversion script
    python "$SCRIPT_PATH" \
        "$recording_dir" \
        --output_dir "$FORMATTED_DATA_DIR" \
        --config_path "$CONFIG_PATH"
        
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Successfully processed $recording_name in ${duration}s"
        echo "📁 Dataset saved to: $output_subdir"
        
        # Show some statistics about the generated dataset
        if [ -d "$output_subdir" ]; then
            num_images=$(find "$output_subdir/images" -name "*.png" 2>/dev/null | wc -l)
            num_masks=$(find "$output_subdir/roi_masks" -name "*.png" 2>/dev/null | wc -l)
            echo "📊 Generated: $num_images images, $num_masks ROI masks"
        fi
    else
        echo ""
        echo "❌ Error processing $recording_name after ${duration}s"
        echo "Check the logs above for details."
    fi
    echo "=========================================="
    echo ""
done

echo ""
echo "🎉 Batch conversion to PNG datasets finished!"
echo ""
echo "📁 All datasets are saved in: $FORMATTED_DATA_DIR"
echo "📋 Each recording contains:"
echo "   - images/     : PNG images for each frame, view, and frequency"
echo "   - roi_masks/  : ROI mask images"
echo "   - list_frames.csv      : Metadata for all frames"
echo "   - matched_points.csv   : Empty file for annotations" 