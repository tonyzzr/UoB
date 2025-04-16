import pickle
from pathlib import Path
from typing import Dict, Any, Tuple
import warnings
import numpy as np
import torch
from tqdm import tqdm # Import tqdm

# Internal imports
from ..data.formats import (
    MatData, BmodeConfig, MultiViewBmodeVideo, TransPos, Bmode, MaskSetting # Add MaskSetting
)
from ..data.readers import RecordingLoader
from ..utils.io import load_toml_config
from ..utils import processing # Import the whole module

class MatConverter:
    """Converts a directory of raw Verasonics .mat files (LF/HF pairs) 
       for a single recording into a processed MultiViewBmodeVideo object 
       saved as a single pickle file.
    """

    def __init__(
        self,
        config_path: str | Path,
    ):
        """
        Initializes the converter with the path to the preprocessing TOML config.

        Args:
            config_path: Path to the TOML configuration file.
        """
        self.config_path = Path(config_path)
        self.raw_config = load_toml_config(self.config_path)
        
        # Prepare specific configs for LF and HF, handling inheritance from [general]
        self.lf_config = self._prepare_frequency_config('lftx')
        self.hf_config = self._prepare_frequency_config('hftx')

        # Instantiate BmodeConfig objects
        self.lf_bmode_config = BmodeConfig.from_dict(self.lf_config)
        self.hf_bmode_config = BmodeConfig.from_dict(self.hf_config)
        
        print(f"Loaded and prepared configuration from: {self.config_path}")
        # print("LF Config:", self.lf_bmode_config)
        # print("HF Config:", self.hf_bmode_config)

    def _prepare_frequency_config(self, freq_key: str) -> Dict[str, Any]:
        """Merges general config with frequency-specific config."""
        if freq_key not in ['lftx', 'hftx']:
            raise ValueError(f"Invalid frequency key: {freq_key}")

        general_cfg = self.raw_config.get('general', {})
        freq_cfg = self.raw_config.get(freq_key, {})

        # Start with general config as base
        merged_config = {
            'general': general_cfg.copy() 
        }

        # Merge/override with frequency-specific sections
        for section in [
            'mask', 'log_compression', 'speckle_reduction', 
            'reject_grating_lobe', 'histogram_match', 'time_gain_compensation'
            # Add other sections if needed (e.g., zero_padding)
        ]:
            # Start with general section settings (if they exist, unlikely but possible)
            merged_section = general_cfg.get(section, {}).copy()
            # Update/override with frequency-specific settings
            merged_section.update(freq_cfg.get(section, {}))
            merged_config[section] = merged_section
            
        # Handle potential override of top-level general keys (e.g., scale_bar)
        # Check if freq_cfg has a specific 'general' section or top-level keys
        # For simplicity now, assume only sections override, scale_bar is just in [general]
        # If freq_cfg has eg. 'scale_bar', merge it into merged_config['general']
        if 'scale_bar' in freq_cfg:
             merged_config['general']['scale_bar'] = freq_cfg['scale_bar']
             
        # Ensure the 'general' section exists if only frequency sections were provided
        if 'general' not in merged_config:
             merged_config['general'] = {}
             if 'scale_bar' not in merged_config['general']:
                  raise ValueError(f"Missing required 'scale_bar' in [general] or [{freq_key}] config.")
        elif 'scale_bar' not in merged_config['general']:
             raise ValueError(f"Missing required 'scale_bar' in [general] config.")

        return merged_config

    def convert_recording(
        self,
        recording_dir: str | Path,
        output_dir: str | Path,
        output_filename: str = "combined_mvbv.pkl"
    ):
        """Processes a single recording directory and saves the combined result."""
        recording_dir = Path(recording_dir)
        output_dir = Path(output_dir)
        recording_id = recording_dir.name
        output_path = output_dir / recording_id / output_filename

        print(f"Starting conversion for recording: {recording_id}")
        print(f"Input directory: {recording_dir}")
        print(f"Output file: {output_path}")

        # Ensure output subdirectory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Load Raw Data
        print("Loading raw data...")
        loader = RecordingLoader(recording_dir)
        raw_lf_data, raw_hf_data = loader.load_combined_mat_data()

        if raw_lf_data is None or raw_hf_data is None:
            warnings.warn(f"Could not load complete LF/HF data for {recording_id}. Skipping conversion.")
            return

        # 2. Process LF Data
        print("Processing LF data...")
        processed_lf_bmode = self._process_single_frequency(raw_lf_data, self.lf_bmode_config)
        if processed_lf_bmode is None:
            warnings.warn(f"Processing failed for LF data in {recording_id}. Skipping conversion.")
            return
            
        # 3. Process HF Data
        print("Processing HF data...")
        processed_hf_bmode = self._process_single_frequency(raw_hf_data, self.hf_bmode_config)
        if processed_hf_bmode is None:
            warnings.warn(f"Processing failed for HF data in {recording_id}. Skipping conversion.")
            return

        # 4. Convert Bmode to MultiViewBmodeVideo
        print("Converting to MultiViewBmodeVideo format...")
        mvbv_lf = self._convert_bmode_to_mvbv(processed_lf_bmode, recording_id)
        mvbv_hf = self._convert_bmode_to_mvbv(processed_hf_bmode, recording_id)
        
        if mvbv_lf is None or mvbv_hf is None:
            warnings.warn(f"Conversion to MVBV failed for {recording_id}. Skipping saving.")
            return

        # 5. Save Combined Result
        print(f"Saving combined data to {output_path}...")
        combined_data = {
            'lftx': mvbv_lf,
            'hftx': mvbv_hf
        }
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(combined_data, f)
            print(f"Successfully saved combined data for {recording_id}.")
        except Exception as e:
            warnings.warn(f"Failed to save pickle file {output_path}: {e}")

    # -------------------------------------
    # Helper methods for processing steps
    # -------------------------------------
    def _process_single_frequency(self, raw_mat_data: MatData, config: BmodeConfig) -> Bmode | None:
        """Applies all preprocessing steps to the data for one frequency."""
        try:
            print(f"    Starting _process_single_frequency...") # DEBUG
            # Check wavelength
            if raw_mat_data.trans.get('wavelengthMm') is None:
                raise ValueError("Cannot process data: 'wavelengthMm' could not be calculated from Trans metadata (missing frequency or SoundSpeed?).")
            
            print(f"    Calculating transducer positions...") # DEBUG
            trans_pos = self._calculate_transducer_positions(raw_mat_data, config.scale_bar)
            num_trans = len(trans_pos)
            print(f"      -> Found {num_trans} transducers.") # DEBUG

            print(f"    Calculating image size...") # DEBUG
            target_h, target_w = self._calculate_image_size(raw_mat_data, config.scale_bar)
            print(f"      -> Target size: ({target_h}, {target_w})") # DEBUG
            
            print(f"    Preparing initial image sequence (stack, squeeze, transpose, resize)...") # DEBUG
            initial_img_seq = self._prepare_initial_image_sequence(raw_mat_data.imgdata, (target_h, target_w))
            if initial_img_seq is None:
                 raise ValueError("Could not prepare initial image sequence.")
            print(f"      -> Prepared initial_img_seq with shape: {initial_img_seq.shape}") # DEBUG
            
            print(f"    Calculating masks...") # DEBUG
            mask_seq = self._calculate_masks(num_trans, target_h, target_w, trans_pos, config.mask_setting)
            print(f"      -> Calculated mask_seq with shape: {mask_seq.shape}") # DEBUG
            
            print(f"    Calling _apply_processing_steps...") # DEBUG
            processed_img_seq = self._apply_processing_steps(initial_img_seq, mask_seq, config)
            print(f"      -> Returned from _apply_processing_steps.") # DEBUG
            
            # Final transpose to match Bmode format: (f, ntrans, h, w)
            # Input to processing steps is (h, w, ntrans, f)
            final_img_seq = np.transpose(processed_img_seq, (3, 2, 0, 1)) 
            
            # Transpose mask to match Bmode format: (1, ntrans, h, w)
            final_mask_seq = np.transpose(mask_seq, (0, 1, 2, 3)) # Already (1, ntrans, h, w)? Check calc.
            # Check if _calculate_masks returns (1, ntrans, h, w) - if so, no transpose needed
            final_mask_seq = mask_seq # Assume _calculate_masks returns correct shape for Bmode

            return Bmode(
                num_trans=num_trans,
                scale_bar=config.scale_bar,
                b_img_seq=final_img_seq, # Shape (f, ntrans, h, w)
                trans_pos=trans_pos,
                mask_seq=final_mask_seq, # Shape (1, ntrans, h, w)
                config=config
            )

        except Exception as e:
            # Log the specific error type and message
            warnings.warn(f"Error during processing ({type(e).__name__}): {e}")
            return None

    def _calculate_transducer_positions(self, mat_data: MatData, scale_bar: float) -> Dict[int, TransPos]:
        """Calculates transducer edge positions in pixel coordinates.
           (Logic adapted from legacy bmode.py::BmodeBuilder.__calc_trans_pos)
        """
        trans_meta = mat_data.trans
        pdata_meta = mat_data.pdata
        num_trans = len(pdata_meta)
        
        if not all(k in trans_meta for k in ['ElementPos', 'wavelengthMm']):
             raise ValueError("Missing 'ElementPos' or 'wavelengthMm' in Trans metadata.")
             
        # Assuming ElementPos gives position of each element center in mm
        # Shape (n_elements_total, 3 or similar?)
        element_pos_x = trans_meta['ElementPos'][:, 0]
        wl_mm = trans_meta['wavelengthMm']
        
        # Need number of elements per aperture (transducer view)
        # This was hardcoded as 32 in legacy code. Is it in metadata?
        # Check Trans or PData keys. Defaulting to 32 for now.
        n_elements_per_aperture = trans_meta.get('n_elements_per_aperture', 32) 
        if 'n_elements_per_aperture' not in trans_meta:
             warnings.warn("'n_elements_per_aperture' not found in Trans metadata, defaulting to 32.")

        trans_pos_dict = {}
        for view_idx in range(num_trans):
            if view_idx not in pdata_meta:
                warnings.warn(f"PData for view index {view_idx} not found.")
                continue
                
            pdata = pdata_meta[view_idx]
            if 'Ox_wl' not in pdata:
                 warnings.warn(f"'Ox_wl' missing in PData for view {view_idx}. Cannot calculate relative position.")
                 continue
                 
            # Origin of the view's coordinate system relative to transducer (in wavelengths)
            view_origin_x_wl = pdata['Ox_wl']
            view_origin_x_mm = view_origin_x_wl * wl_mm
            
            # Find the global indices for the first and last element of this aperture
            first_element_idx = n_elements_per_aperture * view_idx
            last_element_idx = n_elements_per_aperture * (view_idx + 1) - 1
            
            if last_element_idx >= len(element_pos_x):
                 raise IndexError(f"Calculated element index {last_element_idx} out of bounds for ElementPos (size {len(element_pos_x)}). Check n_elements_per_aperture.")
                 
            # Get global positions (in mm) of the edge elements
            left_edge_global_mm = element_pos_x[first_element_idx]
            right_edge_global_mm = element_pos_x[last_element_idx]
            
            # Calculate position relative to the view's origin (in mm)
            left_edge_relative_mm = left_edge_global_mm - view_origin_x_mm
            right_edge_relative_mm = right_edge_global_mm - view_origin_x_mm
            
            # Convert to pixels
            left_edge_px = left_edge_relative_mm / scale_bar
            right_edge_px = right_edge_relative_mm / scale_bar
            
            # Assume Z position is 0 in the image plane
            trans_pos_dict[view_idx] = TransPos(
                left_edge_coord=np.array([[left_edge_px], [0], [1]]),
                right_edge_coord=np.array([[right_edge_px], [0], [1]]),
            )
            
        if not trans_pos_dict:
            raise ValueError("Could not calculate any transducer positions.")
            
        return trans_pos_dict

    def _calculate_image_size(self, mat_data: MatData, scale_bar: float) -> Tuple[int, int]:
        """Calculates target image size (h, w) in pixels based on metadata.
           (Logic adapted from legacy bmode.py::BmodeBuilder.img_size)
        """
        # Use metadata from the first view as representative
        pdata_0 = mat_data.pdata.get(0)
        trans_meta = mat_data.trans
        
        if pdata_0 is None or not all(k in pdata_0 for k in ['nz', 'nx', 'dz_wl', 'dx_wl']):
            raise ValueError("Missing required keys in PData[0] to calculate image size.")
        if 'wavelengthMm' not in trans_meta or trans_meta['wavelengthMm'] is None:
             raise ValueError("Missing 'wavelengthMm' in Trans metadata to calculate image size.")
             
        wl_mm = trans_meta['wavelengthMm']
        dx_mm = pdata_0['dx_wl'] * wl_mm
        dz_mm = pdata_0['dz_wl'] * wl_mm
        
        # Calculate size in mm
        height_mm = pdata_0['nz'] * dz_mm
        width_mm = pdata_0['nx'] * dx_mm
        
        # Convert to pixels
        height_px = int(round(height_mm / scale_bar))
        width_px = int(round(width_mm / scale_bar))
        
        if height_px <= 0 or width_px <= 0:
             raise ValueError(f"Calculated non-positive image size: ({height_px}, {width_px})")
             
        return height_px, width_px

    def _prepare_initial_image_sequence(self, imgdata_dict: Dict[int, np.ndarray], target_size: Tuple[int, int]) -> np.ndarray | None:
        """Extracts, combines views, checks shape, squeezes singleton dim, and resizes image data."""
        # 1. Combine views into a single array (frames, views, h_raw, w_raw)
        view_indices = sorted(imgdata_dict.keys())
        if not view_indices:
            warnings.warn("imgdata_dict is empty.")
            return None
            
        # Stack arrays along a new 'view' dimension
        try:
            # Ensure all arrays have the same shape (except possibly frames)
            base_shape = imgdata_dict[view_indices[0]].shape
            num_frames = base_shape[0]
            raw_h, raw_w = base_shape[-2], base_shape[-1] # Assuming frames, ?, h, w
            
            list_of_arrays = []
            for idx in view_indices:
                 arr = imgdata_dict[idx]
                 # Basic check: must have same non-frame dimensions
                 if arr.shape[1:] != base_shape[1:]:
                      raise ValueError(f"Inconsistent shapes in imgdata for view {idx}: {arr.shape} vs {base_shape}")
                 list_of_arrays.append(arr)
                 
            # Stack along axis 1 to create (frames, views, 1, h_raw, w_raw)
            combined_raw_seq = np.stack(list_of_arrays, axis=1)
            
            # --- Squeeze the singleton dimension (axis 2) --- 
            if combined_raw_seq.shape[2] == 1:
                 squeezed_seq = np.squeeze(combined_raw_seq, axis=2)
                 # Shape is now (frames, views, h_raw, w_raw)
            else:
                 warnings.warn(f"Expected singleton dimension at axis 2 after stacking views, but got shape {combined_raw_seq.shape}. Proceeding without squeeze.")
                 squeezed_seq = combined_raw_seq # Hope for the best?
            # ---------------------------------------------
                 
        except Exception as e:
            warnings.warn(f"Error combining/squeezing image data views: {e}")
            return None
            
        # 2. Transpose for processing: (h, w, views, frames)
        # Original squeezed_seq shape: (f, v, w_raw, h_raw) based on visualization
        # We want output shape for resizing: (h_raw, w_raw, v, f) ??? No, resize expects (h_in, w_in, ...)
        # Let's target (h_raw, w_raw, v, f) directly from (f, v, w_raw, h_raw)
        print(f"Transposing from shape {squeezed_seq.shape} using axes (3, 2, 1, 0)")
        transposed_seq = np.transpose(squeezed_seq, (3, 2, 1, 0))
        print(f"Transposed shape: {transposed_seq.shape}") # Should be (h_raw, w_raw, v, f)
        
        # 3. Resize
        target_h, target_w = target_size
        print(f"Resizing to H={target_h}, W={target_w}...")
        # resize_in_scale expects (h_in, w_in, ntrans, nfrm)
        resized_seq = processing.resize_in_scale(transposed_seq, (target_h, target_w))
        # Output shape: (target_h, target_w, views, frames)
        
        return resized_seq
        
    def _calculate_masks(self, num_trans: int, h: int, w: int, trans_pos: Dict[int, TransPos], setting: MaskSetting) -> np.ndarray:
        """Calculates the angular masks for all views.
           (Logic adapted from legacy bmode.py::BmodeBuilder.__build_mask_seq)
           Returns shape: (1, num_trans, h, w)
        """
        if not setting.enable:
             # Return dummy masks if not enabled (shape needs to match expected output)
             return np.ones((1, num_trans, h, w), dtype=float)
             
        angle_deg = setting.main_lobe_beamwidth
        angle_rad = np.deg2rad(angle_deg) # Convert to radians for trig
        
        mask_seq_list = []
        for i in range(num_trans):
            if i not in trans_pos:
                 warnings.warn(f"TransPos not found for view {i}, cannot calculate mask. Using default mask.")
                 mask_seq_list.append(np.ones((h, w))) # Default mask for this view
                 continue
                 
            # Get transducer edge coordinates (x is index 0)
            left_edge_x = trans_pos[i].left_edge_coord[0, 0]
            right_edge_x = trans_pos[i].right_edge_coord[0, 0]
            
            # Calculate virtual apex (assuming z=0 is transducer face)
            # Ensure right_edge > left_edge? Assume correct for now.
            aperture_width = right_edge_x - left_edge_x
            if aperture_width <= 0:
                 warnings.warn(f"Invalid aperture width ({aperture_width}) for view {i}. Using default mask.")
                 mask_seq_list.append(np.ones((h, w)))
                 continue
                 
            apex_x = (left_edge_x + right_edge_x) / 2.0
            # Use tan(half_angle) = (half_aperture) / |apex_z|
            apex_z = - (aperture_width / 2.0) / np.tan(angle_rad / 2.0) 
            
            # Create coordinate grids relative to apex
            # Z coordinates (depth) range from 0 to h-1
            z_coords = np.arange(h) - apex_z
            # X coordinates range from 0 to w-1
            x_coords = np.arange(w) - apex_x
            xv, zv = np.meshgrid(x_coords, z_coords) # Meshgrid gives x variation across columns, z across rows
            
            # Calculate angle theta relative to the apex Z-axis
            # Avoid division by zero at apex_x line
            # theta_v = np.arctan(np.abs(xv) / zv) # Angle with Z axis, need care with quadrants/signs
            # Using atan2 is safer
            theta_rad = np.arctan2(np.abs(xv), zv) # Angle measured from positive Z axis towards X axis
            theta_deg = np.rad2deg(theta_rad)
            
            # Generate the mask based on angle (using the processing util function)
            view_mask = processing.generate_mask(theta_deg, setting)
            mask_seq_list.append(view_mask)
            
        # Stack masks and add frame dimension -> (num_trans, h, w)
        mask_array = np.stack(mask_seq_list, axis=0)
        # Reshape to (1, num_trans, h, w) for Bmode format
        mask_seq = mask_array.reshape((1, num_trans, h, w))
        return mask_seq
        
    def _apply_processing_steps(self, img_seq: np.ndarray, mask_seq: np.ndarray, config: BmodeConfig) -> np.ndarray:
        """Applies the configured processing steps sequentially.
           Input img_seq shape: (h, w, ntrans, f)
           Input mask_seq shape: (1, ntrans, h, w)
           Output shape: (h, w, ntrans, f)
        """
        proc_img = img_seq.astype(float)
        
        processing_steps = [
            ("Log Compression", processing.log_compression, config.log_compression_setting),
            ("Speckle Reduction", processing.speckle_reduction, config.speckle_reduction_setting),
            ("Reject Grating Lobe", processing.reject_grating_lobe_artifact, config.reject_grating_lobe_setting, mask_seq),
            ("Apply TGC", processing.apply_tgc, config.time_gain_compensation_setting),
            ("Histogram Match", processing.histogram_match, config.histogram_match_setting),
        ]

        # Wrap the loop with tqdm
        print("Applying B-mode processing steps:")
        for name, func, setting, *args in tqdm(processing_steps, desc="Processing Steps"):
            print(f"  Running: {name}...") # Print current step
            if setting.enable:
                if args: # Handle functions needing the mask
                    proc_img = func(proc_img, *args, setting=setting)
                else:
                    proc_img = func(proc_img, setting=setting)
            else:
                print(f"    Skipped (disabled in config).")
                
        return proc_img
        
    def _convert_bmode_to_mvbv(self, bmode_data: Bmode, recording_id: str) -> MultiViewBmodeVideo | None:
        """Converts a processed Bmode object to a MultiViewBmodeVideo object.
           (Logic adapted from legacy multiview_bmode.py::Bmode2MultiViewBmodeVideo.convert)
        """
        try:
            n_frame, n_view, h, w = bmode_data.b_img_seq.shape
            
            # Get origin and aperture from the first transducer's position
            if 0 not in bmode_data.trans_pos:
                 raise ValueError("TransPos for view index 0 not found.")
                 
            trans_pos_0 = bmode_data.trans_pos[0]
            origin_x = trans_pos_0.left_edge_coord[0, 0]
            origin_y = trans_pos_0.left_edge_coord[1, 0]
            aperture_size_px = trans_pos_0.length # Use property
            
            # Convert numpy arrays to torch tensors
            view_images_tensor = torch.from_numpy(bmode_data.b_img_seq).float()
            view_masks_tensor = torch.from_numpy(bmode_data.mask_seq).float()
            
            # Store the used config settings (convert dataclasses back to dict?)
            # For now, store the BmodeConfig object itself, but dict might be better for serialization
            # Simplification: Just store the path or a basic identifier for now.
            # processing_config_dict = dataclasses.asdict(bmode_data.config) # Requires import
            processing_config_dict = {"config_path": str(self.config_path)} # Example placeholder

            return MultiViewBmodeVideo(
                n_view=n_view,
                image_shape=(h, w),
                origin=(float(origin_x), float(origin_y)),
                aperture_size=float(aperture_size_px),
                scale_bar=bmode_data.scale_bar,
                n_frame=n_frame,
                view_images=view_images_tensor, # Shape (n_frame, n_view, h, w)
                view_masks=view_masks_tensor,     # Shape (1, n_view, h, w)
                source_data_identifier=recording_id,
                processing_config=processing_config_dict 
            )
        except Exception as e:
             warnings.warn(f"Error converting Bmode to MultiViewBmodeVideo: {e}")
             return None
