'''
This script is used to manually register the frames of the dataset.

We will firstly load the dataset and display the first frame.

There will be a slider bar to select the view pair (adjacent views) to be registered.

We will show the views with red and green color so can be overlapped for easy observation.

We will then save a first_frame_pose.csv file to record the manual registration result.
The csv file will have the following columns:
frame_no, theta_0, theta_1, ..., theta_7 (8 values)

The theta_0 is the rotation angle of the first view, theta_1 is the rotation angle of the second view, etc.
The thata_0 should always be 0.

Usage:
  python manual_reg_ui.py                          # Interactive selection of recording
  python manual_reg_ui.py recording_name=my_recording  # Specify recording directly

Dependencies:
  pip install inquirer  # For interactive CLI selection

'''
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Slider

import logging
logging.basicConfig(level=logging.INFO)
from legacy.load_dataset import load_and_visualize_frame

def main(recording_name:str = "recording_2022-08-17_trial2-arm"):
    logging.info("Starting manual registration UI")

    dataset_dir = "data/processed"
    dataset_path = os.path.join(dataset_dir, 
                                recording_name,
                                'combined_mvbv.pkl')

    mvbvs = load_and_visualize_frame(dataset_path,
                                     do_plot=False)

    logging.info("\nDataset Information:")
    logging.info(f"Number of frames in LFTX: {mvbvs['lftx'].view_images.shape[0]}")
    logging.info(f"Number of views in LFTX: {mvbvs['lftx'].view_images.shape[1]}")
    logging.info(f"Frame shape LFTX: {mvbvs['lftx'].view_images.shape[2:]}")
    logging.info(f"\nNumber of frames in HFTX: {mvbvs['hftx'].view_images.shape[0]}")
    logging.info(f"Number of views in HFTX: {mvbvs['hftx'].view_images.shape[1]}")
    logging.info(f"Frame shape HFTX: {mvbvs['hftx'].view_images.shape[2:]}")

    # Create a slider for selecting the frame number
    frm4reg = select_frame_for_registration(mvbvs)

    # import pdb; pdb.set_trace()

    # frm4reg = 0
    logging.info(f"Frame {frm4reg} selected for registration")

    save_path = os.path.join(dataset_dir, 
                             recording_name,
                             'ref_frame_pose.csv')

    register_views(mvbvs, frm4reg, save_path)


    return


def find_transducer_edges_xy(mvbvs, tx_mode:str = 'lftx') -> tuple:
    '''
    Find the center of rotation of the views of the frame.
    We will return in the format of (x, y) i.e., width, height.
    '''
    
    mvbv = mvbvs[tx_mode]
    h, w = mvbv.image_shape

    transducer_edge_right_xy = (mvbv.origin[0] + mvbv.aperture_size,
                                    mvbv.origin[1])

    transducer_edge_left_xy = (mvbv.origin[0],
                               mvbv.origin[1])

    return np.array(transducer_edge_right_xy), np.array(transducer_edge_left_xy)

from typing import Dict
def apply_pose(mvbvs, frm4reg:int = 0, 
               view_pair:tuple = (0, 1), 
               theta_values:list = [0.0] * 8) -> Dict:
    '''
    Apply the pose to the views of the frame.
    the theta_valus are in degree, and is clockwise rotation.
    '''
    import cv2

    reg_result = {}

    for tx_mode in ['lftx', 'hftx']:
        mvbv = mvbvs[tx_mode]
        h, w = mvbv.image_shape

        aperture_size = mvbv.aperture_size
        origin = mvbv.origin
        right_edge_x = origin[0] + aperture_size
        right_edge_y = origin[1]
        
        # firstly we need to shift the second view's left edge to the first view's right edge
        view_1 = mvbv.view_images[frm4reg, view_pair[0], :, :].numpy()
        view_2 = mvbv.view_images[frm4reg, view_pair[1], :, :].numpy()

        shift_trans_mat = np.array([
            [1, 0, aperture_size],
            [0, 1, 0],
            [0, 0, 1]
        ])
        
        # second we need to rotate the new view_2 to the first view's right edge
        # the theta_values are in degree, and is clockwise rotation.
        # we need to convert it to counter-clockwise rotation.
        theta_degrees = -theta_values[view_pair[1]]
        rotate_trans_mat = cv2.getRotationMatrix2D((right_edge_x, right_edge_y), 
                                                    theta_degrees, 
                                                    scale=1.0)
        
        combined_trans_mat = rotate_trans_mat @ shift_trans_mat
        view_2_rotated = cv2.warpAffine(view_2, 
                                        combined_trans_mat, 
                                        (w, h))
        
        reg_result[tx_mode] = {
            'view_1': view_1,
            'view_2': view_2_rotated
        }

    return reg_result
  

def register_views(mvbvs, frm4reg:int = 0, save_path:str = None) -> np.ndarray:
    '''
    Register the views of the frame. We will create a UI for this.

    The UI will have 8 sliders, one for each view.
    The sliders will be used to select the rotation angle of the view:
    for the first view, the slider will be a absolute rotation angle respect to the center of the image.
    for the other views, the slider will be a relative rotation angle respect to the previous view, we 
    will also specify the center of rotation.
    Slider will be continuous, and the value will be in degree, within the range of -15 to 45.

    We will have five panels of plots:

    - the first panel will show the LFTX image pair selected for registration 
    (for example, view 0 and view 1, or view 2 and view 3, etc., the smaller one is the reference view)
    - the second panel will show the HFTX image pair selected for registration

    - the third panel will show the overall rendered image of the 8 views of LFTX,
    the rendering method will be separately provided, for example, by taking the average of the 8 views.
    - the fourth panel will show the overall rendered image of the 8 views of HFTX,
    the rendering method will be separately provided, for example, by taking the average of the 8 views.

    - we will also have a panel to show the rigid link model of the overall pose

    We will have a button to save the registration result, which is the 8 theta values, and the first
    theta value should be 0.

    '''
    from matplotlib.widgets import Slider, Button
    import matplotlib.gridspec as gridspec
    
    n_views = mvbvs['lftx'].view_images.shape[1]
    
    # Initialize rotation angles (theta_0 = 0, others start at 0)
    theta_values = [0.0] * n_views
    
    # Create the main figure with custom layout
    fig = plt.figure(figsize=(20, 12))
    
    # Create a grid layout
    # Top row: 5 panels for visualization
    # Bottom area: sliders and button
    gs = gridspec.GridSpec(3, 5, height_ratios=[3, 3, 1], width_ratios=[1, 1, 1, 1, 1])
    
    # Create the 5 visualization panels
    ax_lftx_pair = fig.add_subplot(gs[0, 0])
    ax_hftx_pair = fig.add_subplot(gs[0, 1]) 
    ax_lftx_rendered = fig.add_subplot(gs[0, 2])
    ax_hftx_rendered = fig.add_subplot(gs[0, 3])
    ax_pose_model = fig.add_subplot(gs[0, 4])
    
    # Set titles for the panels
    ax_lftx_pair.set_title('LFTX Image Pair')
    ax_hftx_pair.set_title('HFTX Image Pair')
    ax_lftx_rendered.set_title('LFTX Rendered')
    ax_hftx_rendered.set_title('HFTX Rendered')
    ax_pose_model.set_title('Pose Model')
    
    # Turn off axes for image panels
    for ax in [ax_lftx_pair, ax_hftx_pair, ax_lftx_rendered, ax_hftx_rendered]:
        ax.axis('off')
    
    # Create horizontal view pair selector above sliders
    pair_buttons = []
    pair_button_axes = []
    current_pair_selection = [0]  # Use list to make it mutable in nested functions
    
    # Create horizontal buttons for view pair selection
    button_width = 0.08
    button_height = 0.04
    start_left = 0.15
    button_spacing = 0.09
    button_top = 0.35
    
    # Add label for the buttons
    ax_pair_label = plt.axes([0.05, button_top, 0.08, button_height])
    ax_pair_label.text(0.5, 0.5, 'View Pair:', ha='center', va='center', 
                       transform=ax_pair_label.transAxes, fontsize=10, fontweight='bold')
    ax_pair_label.axis('off')
    
    # Create buttons with clear visual feedback
    for i in range(n_views-1):
        left = start_left + i * button_spacing
        ax_button = plt.axes([left, button_top, button_width, button_height])
        
        # Create a regular button for each pair
        button = Button(ax_button, f"({i},{i+1})")
        
        # Set initial appearance - selected button has different color
        if i == 0:
            button.color = 'lightblue'  # Selected color
            button.hovercolor = 'lightblue'
        else:
            button.color = 'lightgray'  # Unselected color
            button.hovercolor = 'gray'
        
        pair_buttons.append(button)
        pair_button_axes.append(ax_button)
    
    # Create sliders area (moved down)
    slider_axes = []
    sliders = []
    
    # Create 8 sliders below the pair selector
    for i in range(n_views):
        # Calculate position for each slider
        left = 0.05 + i * 0.11  # Spacing them across the width
        bottom = 0.20  # Moved down to make room for pair selector
        width = 0.1
        height = 0.03
        
        # Create slider axis
        ax_slider = plt.axes([left, bottom, width, height])
        
        # Create slider
        # theta_0 is absolute (0 degrees), others are relative (-15 to 45 degrees)
        if i == 0:
            slider = Slider(
                ax=ax_slider,
                label=f'θ{i}\n(abs)',
                valmin=-0.1,  # Small range to avoid warning
                valmax=0.1,   # Small range to avoid warning
                valinit=0,
                valstep=0.1,
                valfmt='%.1f°'
            )
            slider.set_active(False)  # Disable the first slider
        else:
            slider = Slider(
                ax=ax_slider,
                label=f'θ{i}\n(rel)',
                valmin=-15,
                valmax=45,
                valinit=0,
                valstep=0.1,
                valfmt='%.1f°'
            )
        
        slider_axes.append(ax_slider)
        sliders.append(slider)
    
    # Create save button
    ax_save_button = plt.axes([0.85, 0.10, 0.1, 0.05])
    save_button = Button(ax_save_button, 'Save Result')
    
    # Create reset button
    ax_reset_button = plt.axes([0.75, 0.10, 0.08, 0.05])
    reset_button = Button(ax_reset_button, 'Reset')
    
    # Helper function to manage button selection and visual feedback
    def select_pair_button(selected_index):
        """Ensure only one pair button is selected at a time with clear visual feedback"""
        current_pair_selection[0] = selected_index
        
        # Update button colors to show selection state
        for i, button in enumerate(pair_buttons):
            if i == selected_index:
                # Selected button: blue background
                button.color = 'lightblue'
                button.hovercolor = 'lightblue'
            else:
                # Unselected buttons: gray background
                button.color = 'lightgray'
                button.hovercolor = 'gray'
        
        # Force redraw to update button colors
        fig.canvas.draw_idle()
        update_visualization()
    
    # Update functions
    def update_visualization():
        """Update all visualization panels based on current theta values"""
        current_pair = current_pair_selection[0]

        reg_result = apply_pose(mvbvs, 
                                frm4reg, 
                                (current_pair, current_pair+1),
                                theta_values)

        
        # Update LFTX pair (views current_pair and current_pair+1)
        ax_lftx_pair.clear()
        lftx_view1 = reg_result['lftx']['view_1']
        lftx_view2 = reg_result['lftx']['view_2']
        
        # Show as overlay with different colors
        combined_lftx = np.zeros((*lftx_view1.shape, 3))
        combined_lftx[:, :, 0] = lftx_view1 / lftx_view1.max()  # Red channel
        combined_lftx[:, :, 1] = lftx_view2 / lftx_view2.max()  # Green channel
        
        ax_lftx_pair.imshow(combined_lftx)
        ax_lftx_pair.set_title(f'LFTX Views {current_pair},{current_pair+1}')
        ax_lftx_pair.axis('off')
        
        # Update HFTX pair
        ax_hftx_pair.clear()
        hftx_view1 = reg_result['hftx']['view_1']
        hftx_view2 = reg_result['hftx']['view_2']
        
        combined_hftx = np.zeros((*hftx_view1.shape, 3))
        combined_hftx[:, :, 0] = hftx_view1 / hftx_view1.max()  # Red channel  
        combined_hftx[:, :, 1] = hftx_view2 / hftx_view2.max()  # Green channel
        
        ax_hftx_pair.imshow(combined_hftx)
        ax_hftx_pair.set_title(f'HFTX Views {current_pair},{current_pair+1}')
        ax_hftx_pair.axis('off')
        
        # TODO: Update rendered images (placeholder for now)
        ax_lftx_rendered.clear()
        ax_lftx_rendered.text(0.5, 0.5, 'LFTX\nRendered\n(TODO)', 
                              ha='center', va='center', transform=ax_lftx_rendered.transAxes)
        ax_lftx_rendered.set_title('LFTX Rendered')
        
        ax_hftx_rendered.clear()
        ax_hftx_rendered.text(0.5, 0.5, 'HFTX\nRendered\n(TODO)', 
                              ha='center', va='center', transform=ax_hftx_rendered.transAxes)
        ax_hftx_rendered.set_title('HFTX Rendered')
        
        # TODO: Update pose model (placeholder for now)
        ax_pose_model.clear()
        ax_pose_model.text(0.5, 0.5, 'Pose Model\n(TODO)', 
                           ha='center', va='center', transform=ax_pose_model.transAxes)
        ax_pose_model.set_title('Pose Model')
        
        fig.canvas.draw_idle()
    
    def update_theta(val):
        """Update theta values when sliders change"""
        for i, slider in enumerate(sliders):
            if i == 0:
                theta_values[i] = 0.0  # Always keep theta_0 at 0
            else:
                theta_values[i] = slider.val
        update_visualization()
    
    def create_pair_callback(pair_index):
        """Create a callback function for a specific pair button"""
        def callback(event):
            select_pair_button(pair_index)
        return callback
    
    def save_result(event):
        """Save the current registration result"""

        save_reg_result(frm4reg, np.array(theta_values), save_path)



        logging.info("Registration result saved!")
        logging.info(f"Frame: {frm4reg}")
        logging.info(f"Theta values: {theta_values}")
        # Close the figure to return the result
        plt.close(fig)
    
    def reset_values(event):
        """Reset all theta values to 0"""
        for i, slider in enumerate(sliders):
            if i > 0:  # Don't reset theta_0, it's always 0
                slider.reset()
        update_visualization()
    
    # Connect callbacks
    for slider in sliders[1:]:  # Skip the first disabled slider
        slider.on_changed(update_theta)
    
    # Connect each pair button to its callback
    for i, button in enumerate(pair_buttons):
        button.on_clicked(create_pair_callback(i))
    
    save_button.on_clicked(save_result)
    reset_button.on_clicked(reset_values)
    
    # Initial visualization
    update_visualization()
    
    # Add instructions
    fig.suptitle(f'Manual Registration - Frame {frm4reg}\n'
                 f'Adjust θ1-θ7 (relative rotations), select view pairs to register\n'
                 f'Red=View N, Green=View N+1. Use sliders to align overlapping regions.', 
                 fontsize=14)
    
    plt.show()

    return np.array(theta_values)

def save_reg_result(frm4reg, reg_result, save_path):
    '''
    Save the registration result to a csv file
    The csv file will have the following columns:
    frame_no, theta_0, theta_1, ..., theta_7 (8 values)

    The theta_0 is the rotation angle of the first view, theta_1 is the rotation angle of the second view, etc.
    The thata_0 should always be 0.
    '''
    data_dict = {
        'frame_no': frm4reg,
    }
    for i in range(len(reg_result)):
        data_dict[f'theta_{i}'] = reg_result[i]
    
    df = pd.DataFrame([data_dict])  # Wrap in list to create single row
    df.to_csv(save_path, index=False)
    logging.info(f"Registration result saved to: {save_path}")
    return

def select_frame_for_registration(mvbvs) -> int:
    from legacy.load_dataset import visualize_frame
    '''
    Create a slider for selecting the frame number
    '''
    n_frames = mvbvs['lftx'].view_images.shape[0]
    n_views = mvbvs['lftx'].view_images.shape[1]

    fig, axes = plt.subplots(2, n_views, figsize=(16, 6))
    plt.subplots_adjust(bottom=0.25)  # Make more space for the slider

    # Initial visualization
    visualize_frame(mvbvs, 0, axes)
    
    # Create slider axis (position: [left, bottom, width, height])
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    frame_slider = Slider(
        ax=ax_slider,
        label='Frame',
        valmin=0,
        valmax=n_frames - 1,
        valinit=0,
        valstep=1,
        valfmt='%d'
    )

    # Update function for the slider
    def update_frame(val):
        frame_idx = int(frame_slider.val)
        # Clear the axes and redraw
        for i in range(2):
            for j in range(n_views):
                axes[i, j].clear()
                axes[i, j].axis('off')
        
        # Update LFTX data - all views
        for view_idx in range(n_views):
            lftx_frame = mvbvs['lftx'].view_images[frame_idx, view_idx, :, :]
            axes[0, view_idx].imshow(lftx_frame, cmap='gray')
            axes[0, view_idx].set_title(f'LFTX View {view_idx}')
            axes[0, view_idx].axis('off')
        
        # Update HFTX data - all views
        for view_idx in range(n_views):
            hftx_frame = mvbvs['hftx'].view_images[frame_idx, view_idx, :, :]
            axes[1, view_idx].imshow(hftx_frame, cmap='gray')
            axes[1, view_idx].set_title(f'HFTX View {view_idx}')
            axes[1, view_idx].axis('off')
            
        # Update the main title to show current frame
        fig.suptitle(f'Frame {frame_idx} (Total: {n_frames} frames)')
        fig.canvas.draw_idle()

    # Connect the slider to the update function
    frame_slider.on_changed(update_frame)
    
    plt.suptitle(f'Frame Selection (Total: {n_frames} frames)')
    plt.show()

    return int(frame_slider.val)
    

def get_available_recordings(dataset_dir: str = "data/processed") -> list:
    """
    Get a list of available recording names from the dataset directory.
    """
    if not os.path.exists(dataset_dir):
        logging.error(f"Dataset directory {dataset_dir} does not exist!")
        return []
    
    recordings = []
    for item in os.listdir(dataset_dir):
        item_path = os.path.join(dataset_dir, item)
        if os.path.isdir(item_path):
            # Check if it contains the expected combined_mvbv.pkl file
            pkl_file = os.path.join(item_path, 'combined_mvbv.pkl')
            if os.path.exists(pkl_file):
                recordings.append(item)
            else:
                logging.warning(f"Directory {item} exists but doesn't contain combined_mvbv.pkl")
    
    return sorted(recordings)


def select_recording_interactively() -> str:
    """
    Present an interactive menu to select a recording using arrow keys.
    """
    try:
        import inquirer
    except ImportError:
        print("Error: 'inquirer' package is required for interactive selection.")
        print("Install it with: pip install inquirer")
        sys.exit(1)
    
    recordings = get_available_recordings()
    
    if not recordings:
        print("No valid recordings found in data/processed/")
        print("Make sure the directory exists and contains folders with combined_mvbv.pkl files.")
        sys.exit(1)
    
    questions = [
        inquirer.List('recording',
                     message="Select a recording to process",
                     choices=recordings,
                     carousel=True)  # Allow wrapping around with arrow keys
    ]
    
    answers = inquirer.prompt(questions)
    if answers is None:  # User pressed Ctrl+C
        print("\nOperation cancelled.")
        sys.exit(0)
    
    return answers['recording']


if __name__ == "__main__":
    import sys
    from omegaconf import OmegaConf

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








