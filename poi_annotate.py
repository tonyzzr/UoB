'''
This script is used to annotate the POI of the tissue maps using an
interactive GUI.

The POI is a list of tuples, each containing:
- frame_name: str, e.g. 'frame_0000_hftx.png'
- x: int, e.g. 100
- y: int, e.g. 100
- points: str, e.g. 'p1,p2,p3'
- frame: int, e.g. 0
- video_name: str, e.g. 'recording_2022-08-17_trial2-arm'
- subject: int, e.g. 0

The GUI is a simple interface that allows the user to click on the image to
annotate the POI. There should be a button to save the current frame's POI
into the CSV file. Also, considering the continuous nature of the POI, there 
should also be a button to show the POI of the closest frame to the current frame.

Since the video is very long, we will make a slider to select the frame, and
the default frame interval will be 10 frames. There should also be a number
input box to set the frame interval.

'''

def main(recording_name: str):
    import logging
    import os

    logging.info("Starting POI annotation UI")

    dataset_dir = "data/processed"
    tx_mode = 'hftx'
    panorama_dir = os.path.join(dataset_dir, 
                                recording_name,
                                f'panoramas/weighted_mean_fuser/{tx_mode}')
    poi_save_path = os.path.join(dataset_dir, 
                                 recording_name,
                                 'poi.csv')

    panoramas = load_panoramas(panorama_dir)
    
    if not panoramas:
        logging.error(f"No panoramas found in {panorama_dir}")
        return

    annotate_poi(panoramas, poi_save_path, recording_name)
    
    return

def load_panoramas(panorama_dir):
    '''
    Load the panoramas from the directory.

    we need to load the filenames with .png extension, and then sort them
    by the filenames. The filenames are like 'frame_0000_hftx.png'.
    '''
    import imageio.v2 as imageio  # Use v2 to avoid deprecation warning
    import os
    import re
    from collections import OrderedDict

    panoramas = OrderedDict()
    
    if not os.path.exists(panorama_dir):
        print(f"Error: Directory {panorama_dir} does not exist")
        return panoramas

    # Get all PNG files and sort them by frame number
    png_files = [f for f in os.listdir(panorama_dir) if f.endswith('.png')]
    
    # Extract frame numbers and sort
    frame_files = []
    for filename in png_files:
        # Extract frame number from filename like 'frame_0000_hftx.png' or 'frame_0000.png'
        match = re.search(r'frame_(\d+)', filename)
        if match:
            frame_num = int(match.group(1))
            frame_files.append((frame_num, filename))
    
    # Sort by frame number
    frame_files.sort(key=lambda x: x[0])
    
    print(f"Loading {len(frame_files)} panorama frames...")
    
    # Load images
    for frame_num, filename in frame_files:
        file_path = os.path.join(panorama_dir, filename)
        try:
            image = imageio.imread(file_path)
            panoramas[frame_num] = {
                'image': image,
                'filename': filename,
                'frame_num': frame_num
            }
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    print(f"Successfully loaded {len(panoramas)} frames")
    return panoramas

def save_poi(poi_list, poi_save_path):
    '''
    Save the POI to a CSV file.
    
    Args:
        poi_list: List of POI dictionaries to save
        poi_save_path: Path to save the POI
    
    Returns:
        None

    Headers: frame_name,x,y,points,frame,video_name,subject
    Example: frame_0000_hftx.png,100,100,p1,0,recording_2022-08-17_trial2-arm,0
    '''
    import pandas as pd
    import os

    if not poi_list:
        print("No POI data to save")
        return
    
    # Ensure directory exists (only if there's a directory component)
    dir_path = os.path.dirname(poi_save_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    df = pd.DataFrame(poi_list)
    
    # Ensure correct column order
    expected_columns = ['frame_name', 'x', 'y', 'points', 'frame', 'video_name', 'subject']
    df = df.reindex(columns=expected_columns)
    
    df.to_csv(poi_save_path, index=False)
    print(f"Saved {len(poi_list)} POI records to {poi_save_path}")
    return

def load_existing_poi_csv(poi_save_path):
    '''
    Load existing POI data from CSV file in CSV format (list of dicts).
    
    Returns:
        list: List of POI dictionaries ready for CSV saving
    '''
    import pandas as pd
    import os
    
    if not os.path.exists(poi_save_path):
        return []
    
    try:
        df = pd.read_csv(poi_save_path)
        if df.empty:
            return []
        
        # Convert DataFrame to list of dictionaries
        poi_list = []
        for _, row in df.iterrows():
            poi_list.append({
                'frame_name': row['frame_name'],
                'x': int(row['x']),
                'y': int(row['y']),
                'points': row['points'],
                'frame': int(row['frame']),
                'video_name': row['video_name'],
                'subject': int(row['subject'])
            })
        
        print(f"Loaded {len(poi_list)} existing POI records from CSV")
        return poi_list
        
    except Exception as e:
        print(f"Error loading existing POI CSV data: {e}")
        return []

def load_existing_poi(poi_save_path):
    '''
    Load existing POI data from CSV file.
    
    Returns:
        dict: {frame_num: [poi_dict, ...]}
    '''
    import pandas as pd
    import os
    
    if not os.path.exists(poi_save_path):
        return {}
    
    try:
        df = pd.read_csv(poi_save_path)
        poi_data = {}
        
        for _, row in df.iterrows():
            frame_num = int(row['frame'])
            if frame_num not in poi_data:
                poi_data[frame_num] = []
            
            poi_data[frame_num].append({
                'x': int(row['x']),
                'y': int(row['y']),
                'points': row['points']
            })
        
        print(f"Loaded existing POI data for {len(poi_data)} frames")
        return poi_data
        
    except Exception as e:
        print(f"Error loading existing POI data: {e}")
        return {}

def annotate_poi(panoramas, poi_save_path, recording_name):
    '''
    Annotate the POI of the panoramas using an interactive GUI.
    
    Args:
        panoramas: dict of panoramas
        poi_save_path: Path to save POI data
        recording_name: Name of the recording
    
    Returns:
        None
    '''
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.widgets import Slider, Button, TextBox
    import numpy as np
    
    # Load existing POI data
    existing_poi = load_existing_poi(poi_save_path)
    
    # GUI state
    class AnnotationState:
        def __init__(self):
            self.current_frame = 0
            self.frame_interval = 10
            self.current_pois = []  # [(x, y, point_id), ...]
            self.poi_counter = 1
            self.all_poi_data = {}  # {frame_num: [(x,y,point_id),...]}
            self.zoom_factor = 1.0
            self.pan_x = 0
            self.pan_y = 0
            
            # Load existing POI data into state
            for frame_num, pois in existing_poi.items():
                self.all_poi_data[frame_num] = []
                for poi in pois:
                    point_id = poi['points']
                    self.all_poi_data[frame_num].append((poi['x'], poi['y'], point_id))
    
    state = AnnotationState()
    
    # Get frame info
    frame_nums = sorted(panoramas.keys())
    if not frame_nums:
        print("No frames available for annotation")
        return
    
    max_frame = max(frame_nums)
    min_frame = min(frame_nums)
    
    # Create figure and layout
    fig = plt.figure(figsize=(16, 10))
    
    # Main image axes (larger)
    ax_img = plt.axes([0.1, 0.25, 0.6, 0.65])
    ax_img.set_title("Click to add POI, Right-click to remove nearest POI")
    
    # Control panel area
    ax_controls = plt.axes([0.75, 0.25, 0.2, 0.65], frameon=True)
    ax_controls.set_xlim(0, 1)
    ax_controls.set_ylim(0, 1)
    ax_controls.axis('off')
    
    # Frame slider
    ax_slider = plt.axes([0.1, 0.15, 0.6, 0.03])
    frame_slider = Slider(ax_slider, 'Frame', min_frame, max_frame, 
                         valinit=min_frame, valfmt='%d', valstep=1)
    
    # Interval slider
    ax_interval = plt.axes([0.1, 0.10, 0.6, 0.03])
    interval_slider = Slider(ax_interval, 'Interval', 1, 50, 
                           valinit=10, valfmt='%d', valstep=1)
    
    # Buttons
    ax_save = plt.axes([0.75, 0.15, 0.08, 0.04])
    btn_save = Button(ax_save, 'Save Frame')
    
    ax_clear = plt.axes([0.85, 0.15, 0.08, 0.04])
    btn_clear = Button(ax_clear, 'Clear Frame')
    
    ax_load_ref = plt.axes([0.75, 0.10, 0.08, 0.04])
    btn_load_ref = Button(ax_load_ref, 'Load Ref')
    
    ax_save_all = plt.axes([0.85, 0.10, 0.08, 0.04])
    btn_save_all = Button(ax_save_all, 'Save All')
    
    ax_prev = plt.axes([0.75, 0.05, 0.08, 0.04])
    btn_prev = Button(ax_prev, 'Prev')
    
    ax_next = plt.axes([0.85, 0.05, 0.08, 0.04])
    btn_next = Button(ax_next, 'Next')
    
    # Initialize display
    current_img = None
    poi_artists = []
    ref_poi_artists = []  # Separate list for reference POIs
    
    def update_display():
        nonlocal current_img, poi_artists, ref_poi_artists
        
        # Clear previous POI markers (but NOT reference POIs)
        for artist in poi_artists:
            artist.remove()
        poi_artists = []
        
        # Get current frame data
        if state.current_frame in panoramas:
            img_data = panoramas[state.current_frame]['image']
            
            # Update image
            if current_img is None:
                current_img = ax_img.imshow(img_data, cmap='gray')
                ax_img.set_xlim(0, img_data.shape[1])
                ax_img.set_ylim(img_data.shape[0], 0)
            else:
                current_img.set_data(img_data)
                current_img.set_extent([0, img_data.shape[1], img_data.shape[0], 0])
        
        # Load POIs for current frame
        if state.current_frame in state.all_poi_data:
            state.current_pois = state.all_poi_data[state.current_frame].copy()
        else:
            state.current_pois = []
        
        # Update POI counter
        if state.current_pois:
            max_poi_num = max([int(poi[2][1:]) for poi in state.current_pois if poi[2].startswith('p')])
            state.poi_counter = max_poi_num + 1
        else:
            state.poi_counter = 1
        
        # Draw POI markers
        for i, (x, y, point_id) in enumerate(state.current_pois):
            # Draw small precise dot marker
            circle = patches.Circle((x, y), radius=2, color='red', fill=True, linewidth=1)
            ax_img.add_patch(circle)
            poi_artists.append(circle)
            
            # Draw text label
            text = ax_img.text(x+6, y-8, point_id, color='red', fontsize=9, weight='bold')
            poi_artists.append(text)
        
        # Update title
        ax_img.set_title(f"Frame {state.current_frame} - POIs: {len(state.current_pois)} - "
                        f"Click to add POI, Right-click to remove")
        
        # Update control panel text
        ax_controls.clear()
        ax_controls.set_xlim(0, 1)
        ax_controls.set_ylim(0, 1)
        ax_controls.axis('off')
        
        # Display POI list
        info_text = f"Frame: {state.current_frame}\n"
        info_text += f"POIs: {len(state.current_pois)}\n"
        info_text += f"Next ID: p{state.poi_counter}\n\n"
        info_text += "POI List:\n"
        
        for x, y, point_id in state.current_pois[-10:]:  # Show last 10 POIs
            info_text += f"{point_id}: ({x},{y})\n"
        
        if len(state.current_pois) > 10:
            info_text += f"... and {len(state.current_pois)-10} more"
        
        ax_controls.text(0.05, 0.95, info_text, transform=ax_controls.transAxes, 
                        verticalalignment='top', fontsize=9, family='monospace')
        
        plt.draw()
    
    def on_click(event):
        if event.inaxes != ax_img:
            return
        
        x, y = int(event.xdata), int(event.ydata)
        
        if event.button == 1:  # Left click - add POI
            point_id = f"p{state.poi_counter}"
            state.current_pois.append((x, y, point_id))
            state.poi_counter += 1
            
            # Update stored data
            state.all_poi_data[state.current_frame] = state.current_pois.copy()
            
            print(f"Added POI {point_id} at ({x}, {y})")
            update_display()
            
        elif event.button == 3:  # Right click - remove nearest POI
            if state.current_pois:
                # Find nearest POI
                distances = [((x-px)**2 + (y-py)**2)**0.5 for px, py, _ in state.current_pois]
                nearest_idx = distances.index(min(distances))
                
                if distances[nearest_idx] < 10:  # Within 10 pixels
                    removed_poi = state.current_pois.pop(nearest_idx)
                    state.all_poi_data[state.current_frame] = state.current_pois.copy()
                    print(f"Removed POI {removed_poi[2]} at ({removed_poi[0]}, {removed_poi[1]})")
                    update_display()
    
    def on_frame_change(val):
        nonlocal ref_poi_artists
        
        # Clear reference POIs when changing frames
        for artist in ref_poi_artists:
            artist.remove()
        ref_poi_artists = []
        
        state.current_frame = int(frame_slider.val)
        update_display()
    
    def on_interval_change(val):
        state.frame_interval = int(interval_slider.val)
    
    def on_save_frame(event):
        if not state.current_pois:
            print("No POIs to save for current frame")
            return
        
        # Load existing POI data from CSV file
        existing_poi_csv = load_existing_poi_csv(poi_save_path)
        
        # Remove any existing data for current frame (to replace with new data)
        existing_poi_csv = [poi for poi in existing_poi_csv if poi['frame'] != state.current_frame]
        
        # Convert current frame POIs to save format
        filename = panoramas[state.current_frame]['filename']
        
        for x, y, point_id in state.current_pois:
            existing_poi_csv.append({
                'frame_name': filename,
                'x': x,
                'y': y,
                'points': point_id,
                'frame': state.current_frame,
                'video_name': recording_name,
                'subject': 0
            })
        
        # Save all data (existing + current frame)
        save_poi(existing_poi_csv, poi_save_path)
        
        # Update in-memory state to ensure consistency (for Load Ref and Save All functions)
        state.all_poi_data[state.current_frame] = state.current_pois.copy()
        
        print(f"Saved {len(state.current_pois)} POIs for frame {state.current_frame}")
    
    def on_clear_frame(event):
        state.current_pois = []
        if state.current_frame in state.all_poi_data:
            del state.all_poi_data[state.current_frame]
        state.poi_counter = 1
        print(f"Cleared all POIs from frame {state.current_frame}")
        update_display()
    
    def on_load_reference(event):
        nonlocal ref_poi_artists
        
        # Find closest frame with POI data
        current = state.current_frame
        closest_frame = None
        min_distance = float('inf')
        
        for frame_num in state.all_poi_data:
            distance = abs(frame_num - current)
            if distance < min_distance and frame_num != current:
                min_distance = distance
                closest_frame = frame_num
        
        if closest_frame is not None:
            ref_pois = state.all_poi_data[closest_frame]
            print(f"Loaded {len(ref_pois)} reference POIs from frame {closest_frame}")
            
            # Show reference POIs as bright green circles
            for artist in ref_poi_artists:
                artist.remove()
            ref_poi_artists = []
            
            for x, y, point_id in ref_pois:
                circle = patches.Circle((x, y), radius=2, color='lime', fill=False, 
                                      linewidth=1, alpha=0.9)
                ax_img.add_patch(circle)
                ref_poi_artists.append(circle)
                
                # Draw reference POI label
                text = ax_img.text(x+6, y-8, point_id, color='lime', fontsize=9, weight='bold', alpha=0.9)
                ref_poi_artists.append(text)
            
            plt.draw()
        else:
            print("No reference frame found")
    
    def on_save_all(event):
        # Save all POI data
        all_poi_list = []
        
        for frame_num, pois in state.all_poi_data.items():
            if frame_num in panoramas:
                filename = panoramas[frame_num]['filename']
                for x, y, point_id in pois:
                    all_poi_list.append({
                        'frame_name': filename,
                        'x': x,
                        'y': y,
                        'points': point_id,
                        'frame': frame_num,
                        'video_name': recording_name,
                        'subject': 0
                    })
        
        save_poi(all_poi_list, poi_save_path)
    
    def on_prev_frame(event):
        nonlocal ref_poi_artists
        
        new_frame = state.current_frame - state.frame_interval
        if new_frame >= min_frame:
            # Clear reference POIs when changing frames
            for artist in ref_poi_artists:
                artist.remove()
            ref_poi_artists = []
            
            state.current_frame = new_frame
            frame_slider.set_val(state.current_frame)
            update_display()
    
    def on_next_frame(event):
        nonlocal ref_poi_artists
        
        new_frame = state.current_frame + state.frame_interval
        if new_frame <= max_frame:
            # Clear reference POIs when changing frames
            for artist in ref_poi_artists:
                artist.remove()
            ref_poi_artists = []
            
            state.current_frame = new_frame
            frame_slider.set_val(state.current_frame)
            update_display()
    
    # Connect events
    fig.canvas.mpl_connect('button_press_event', on_click)
    frame_slider.on_changed(on_frame_change)
    interval_slider.on_changed(on_interval_change)
    btn_save.on_clicked(on_save_frame)
    btn_clear.on_clicked(on_clear_frame)
    btn_load_ref.on_clicked(on_load_reference)
    btn_save_all.on_clicked(on_save_all)
    btn_prev.on_clicked(on_prev_frame)
    btn_next.on_clicked(on_next_frame)
    
    # Initial display
    update_display()
    
    plt.show()

if __name__ == "__main__":
    import sys
    from omegaconf import OmegaConf
    from manual_reg_ui import select_recording_interactively
    import logging

    # Check if recording_name is provided via command line
    if len(sys.argv) > 1:
        try:
            cfg_from_cli = OmegaConf.from_cli(sys.argv[1:])
            if hasattr(cfg_from_cli, 'recording_name'):
                recording_name = cfg_from_cli.recording_name
            else:
                # If no recording_name in args, show interactive selection
                recording_name = select_recording_interactively()
        except Exception as e:
            logging.warning(f"Error parsing command line arguments: {e}")
            # Fall back to interactive selection
            recording_name = select_recording_interactively()
    else:
        # No command line arguments, show interactive selection
        recording_name = select_recording_interactively()
    
    print(f"Selected recording: {recording_name}")
    main(recording_name)
    # TODO:
    # 1. add a button to save the current frame as a reference frame
    # 2. add the Image rendering panels
    # 3. add the pose model panel



