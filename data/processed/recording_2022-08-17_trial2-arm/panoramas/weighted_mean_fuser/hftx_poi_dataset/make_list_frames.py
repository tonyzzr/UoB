#!/usr/bin/env python3
"""
This script generates the list_frames.csv file for the dataset.
It loads all images in the ./images folder and matches them with
annotations from matched_points.csv to determine which frames are annotated.

The output CSV has columns: frame_name,tx_mode,frame,subject,annotated,video_name
"""

import os
import pandas as pd
import re
from typing import Set, List, Dict


def extract_frame_number(filename: str) -> int:
    """Extract frame number from filename like 'frame_0123.png'"""
    match = re.match(r'^frame_(\d+)\.png$', filename)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot extract frame number from filename: {filename}")


def get_annotated_frames(matched_points_path: str) -> Set[str]:
    """Read matched_points.csv and return set of annotated frame names"""
    try:
        df = pd.read_csv(matched_points_path)
        return set(df['frame_name'].unique())
    except Exception as e:
        print(f"Warning: Could not read matched_points.csv: {e}")
        return set()


def get_all_image_files(images_dir: str) -> List[str]:
    """Get all PNG image files from the images directory"""
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    image_files = []
    for filename in os.listdir(images_dir):
        if filename.endswith('.png') and filename.startswith('frame_'):
            image_files.append(filename)
    
    return sorted(image_files)


def generate_list_frames(images_dir: str = './images', 
                        matched_points_path: str = './matched_points.csv',
                        output_path: str = './list_frames.csv',
                        tx_mode: str = 'hftx',
                        subject: int = 0,
                        video_name: str = 'recording_2022-08-17_trial2-arm') -> pd.DataFrame:
    """
    Generate the list_frames.csv file.
    
    Args:
        images_dir: Path to directory containing image files
        matched_points_path: Path to matched_points.csv file
        output_path: Path for output CSV file
        tx_mode: Default tx_mode value for all frames
        subject: Default subject number for all frames
        video_name: Default video name for all frames
    
    Returns:
        DataFrame with the generated data
    """
    # Get annotated frames
    annotated_frames = get_annotated_frames(matched_points_path)
    print(f"Found {len(annotated_frames)} annotated frames")
    
    # Get all image files
    image_files = get_all_image_files(images_dir)
    print(f"Found {len(image_files)} image files")
    
    # Generate data for each image file
    data: List[Dict] = []
    for filename in image_files:
        try:
            frame_number = extract_frame_number(filename)
            is_annotated = filename in annotated_frames
            
            data.append({
                'frame_name': filename,
                'tx_mode': tx_mode,
                'frame': frame_number,
                'subject': subject,
                'annotated': is_annotated,
                'video_name': video_name
            })
        except ValueError as e:
            print(f"Warning: Skipping file {filename}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by frame number
    df = df.sort_values('frame').reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Generated {output_path} with {len(df)} rows")
    
    # Print summary
    annotated_count = df['annotated'].sum()
    print(f"Summary: {annotated_count} annotated frames, {len(df) - annotated_count} non-annotated frames")
    
    return df


def main():
    """Main function to generate list_frames.csv"""
    try:
        df = generate_list_frames()
        print("\nFirst 10 rows:")
        print(df.head(10))
        print("\nLast 10 rows:")
        print(df.tail(10))
        print(f"\nAnnotated frames sample:")
        annotated_frames = df[df['annotated'] == True]
        if len(annotated_frames) > 0:
            print(annotated_frames.head())
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())