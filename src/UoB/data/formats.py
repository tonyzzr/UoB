import numpy as np
import torch # For MultiViewBmodeVideo
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple # Using Any for BmodeConfig initially

# --- Data Structures from Raw Files ---

@dataclass
class MatData:
    """ Holds raw data loaded from a .mat file. """
    pdata: dict   # Dict containing PData structures (metadata per view)
    imgdata: dict # Dict containing image data arrays (e.g., {0: np.ndarray(...)})
    trans: dict   # Dict containing transducer properties

    def __str__(self) -> str:
        # Basic string representation, might need refinement
        pdata_keys = list(self.pdata.get(0, {}).keys()) if self.pdata else 'N/A'
        imgdata_shape = self.imgdata.get(0, np.array([])).shape if self.imgdata else 'N/A'
        trans_keys = list(self.trans.keys()) if self.trans else 'N/A'
        return f'MatData(pdata_keys={pdata_keys}, imgdata_shape={imgdata_shape}, trans_keys={trans_keys})'

# --- Intermediate Data Structures (Preprocessing) ---

@dataclass
class TransPos:
    """
    Coordinates of left and right edges of a transducer 
    relative to the B-mode image origin. Unit in pixels.
    Uses homogeneous coordinates (x, y, 1).
    """
    left_edge_coord : np.ndarray # Shape (3, 1)
    right_edge_coord: np.ndarray # Shape (3, 1)

    @property
    def length(self) -> float:
        """ Calculate the length of the transducer face in pixels. """
        vec = self.left_edge_coord - self.right_edge_coord
        return float(np.linalg.norm(vec[:2])) # Use only x, y for length

    @property
    def centroid(self) -> np.ndarray:
        """ Calculate the centroid of the transducer face (x, y). """
        return (self.left_edge_coord[:2] + self.right_edge_coord[:2]) / 2

# --- Settings for Processing Steps ---

@dataclass
class MaskSetting:
    """ Settings for generating the angular mask. """
    enable: bool = False
    main_lobe_beamwidth: float = 30.0 # Angle in degrees
    soft_boundary: bool = True
    softness: float = 0.3 # Parameter for sigmoid transition

@dataclass
class LogCompressionSetting:
    """ Settings for log compression. """
    enable: bool = False
    dynamic_range: float = 60.0 # Dynamic range in dB
    max_value: float | None = 1024.0 # Reference max value (if None, calculated from data)

@dataclass
class SpeckleReductionSetting:
    """ Settings for speckle reduction (Median Blur + NLM). """
    enable: bool = False
    med_blur_kernal: int = 3 # Kernel size for median blur (must be odd > 1)
    nlm_h: float = 9.0 # Parameter controlling filter strength for NLM
    nlm_template_window_size: int = 7 # Size of template patch for NLM
    nlm_search_window_size: int = 11 # Size of search window for NLM

@dataclass
class RejectGratingLobeSetting:
    """ Settings for rejecting grating lobe artifacts using a mask. """
    enable: bool = False

@dataclass
class HistogramMatchSetting:
    """ Settings for histogram matching between views. """
    enable: bool = False
    ref_ind: int = 0 # Index of the reference view/transducer
    background_removal: bool = True # Attempt to remove background shift post-matching

@dataclass
class ApplyTGCSetting:
    """ Settings for applying Time Gain Compensation (TGC). """
    enable: bool = False
    tgc_threshold: float = 0.8 # Normalized depth threshold for TGC curve inflection
    tgc_slope: float = 10.0 # Slope of the sigmoid TGC curve

@dataclass
class ZeroPadSetting:
    """ Settings for applying zero padding (if needed post-processing). """
    enable: bool = False
    top_padding_ratio: float = 0.0
    bottom_padding_ratio: float = 0.0
    left_padding_ratio: float = 0.0
    right_padding_ratio: float = 0.0

# --- Configuration Container ---

@dataclass
class BmodeConfig:
    """ Container for all B-mode processing settings, loaded from TOML. """
    # General
    scale_bar: float # Pixel size in mm/pixel (or pixel/mm, needs consistency)

    # Individual step settings
    mask_setting: MaskSetting
    log_compression_setting: LogCompressionSetting
    speckle_reduction_setting: SpeckleReductionSetting
    reject_grating_lobe_setting: RejectGratingLobeSetting
    histogram_match_setting: HistogramMatchSetting
    time_gain_compensation_setting: ApplyTGCSetting
    # zero_pad_setting: ZeroPadSetting # Padding might not be part of initial Bmode conversion

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BmodeConfig':
        """ Creates a BmodeConfig instance from a dictionary representing a single frequency config (e.g., merged general + lftx). """
        
        # Extract sections with defaults
        general_cfg = config_dict.get('general', {})
        mask_cfg = config_dict.get('mask', {})
        log_comp_cfg = config_dict.get('log_compression', {})
        speckle_cfg = config_dict.get('speckle_reduction', {})
        grating_cfg = config_dict.get('reject_grating_lobe', {})
        hist_cfg = config_dict.get('histogram_match', {})
        tgc_cfg = config_dict.get('time_gain_compensation', {})
        
        # Validate essential keys (example: scale_bar)
        if 'scale_bar' not in general_cfg:
            print("Warning: Missing 'scale_bar' in [general] config. Using default 0.1")
            general_cfg['scale_bar'] = 0.1 

        # Handle explicit null max_value for log compression
        log_max_value = log_comp_cfg.get('max_value') # Get value, could be number, None, or missing
        if log_max_value == 'null': # Handle potential string 'null' just in case
             log_max_value = None

        # Create LogCompressionSetting instance manually to control None passing
        try:
             log_setting = LogCompressionSetting(
                 enable=log_comp_cfg.get('enable', False), # Get other fields or use defaults
                 dynamic_range=float(log_comp_cfg.get('dynamic_range', 60.0)),
                 max_value=None if log_max_value is None else float(log_max_value) # Pass None or float
             )
        except (ValueError, TypeError) as e:
             raise TypeError(f"Invalid value in [log_compression] config: {e}. Config: {log_comp_cfg}") from e

        try:
            return cls(
                scale_bar=float(general_cfg['scale_bar']),
                mask_setting=MaskSetting(**mask_cfg),
                log_compression_setting=log_setting, # Use the manually created instance
                speckle_reduction_setting=SpeckleReductionSetting(**speckle_cfg),
                reject_grating_lobe_setting=RejectGratingLobeSetting(**grating_cfg),
                histogram_match_setting=HistogramMatchSetting(**hist_cfg),
                time_gain_compensation_setting=ApplyTGCSetting(**tgc_cfg)
            )
        except TypeError as e:
            # Catch errors if unexpected keys are passed to dataclasses
            raise TypeError(f"Error creating BmodeConfig from dict. Check config keys: {e}\nConfig provided: {config_dict}") from e

# --- Processed Data Structures ---

@dataclass
class Bmode:
    """ Represents processed B-mode data for a single frequency (LF/HF) over time. """
    num_trans: int               # Number of transducers/views
    scale_bar: float             # Pixel size (e.g., mm/pixel)
    b_img_seq: np.ndarray        # Processed image sequence (n_frame, n_trans, h, w)
    trans_pos: Dict[int, TransPos] # Dictionary mapping transducer index to its position
    mask_seq: np.ndarray         # Angular mask sequence (1, n_trans, h, w) - usually static
    config: BmodeConfig          # Configuration used for processing

    # Removed __post_init__ calculating trans_pos_gap for now, can be added if needed

    def __str__(self) -> str:
        return (f'Bmode(num_trans={self.num_trans}, scale_bar={self.scale_bar}, ' \
                f'b_img_seq.shape={self.b_img_seq.shape}, mask_seq.shape={self.mask_seq.shape})')

# --- Final Output Data Structure ---

# Base class definition (similar to legacy MultiViewBmode but adapted)
# @dataclass
# class MultiViewBmodeBase:
#     """ Base structure for multi-view B-mode data, potentially single frame. """
#     n_view: int                 # Number of views (transducers)
#     image_shape: Tuple[int, int] # (h, w) of the processed images
#     origin: Tuple[float, float]  # (x, y) origin of the first transducer's coordinate system in pixels
#     aperture_size: float        # Width of the transducer aperture in pixels
#     scale_bar: float            # Pixel size (e.g., mm/pixel), inherited/passed
#     
#     # Metadata
#     source_data_identifier: str = "" # E.g., path to raw dir or specific file ID
#     processing_config: Dict[str, Any] = field(default_factory=dict) # Store the config dict used

@dataclass
class MultiViewBmodeVideo:
    """ Represents the full multi-view B-mode video sequence for one frequency (LF/HF). """
    # Fields without defaults first
    n_view: int
    image_shape: Tuple[int, int]
    origin: Tuple[float, float]
    aperture_size: float
    scale_bar: float
    n_frame: int
    view_images: torch.Tensor   # Shape (n_frame, n_view, h, w)
    view_masks: torch.Tensor    # Shape (1, n_view, h, w) - static mask
    
    # Fields with defaults last
    source_data_identifier: str = "" # E.g., path to raw dir or specific file ID
    processing_config: Dict[str, Any] = field(default_factory=dict) # Store the config dict used

    # Can add methods for frame access, padding, resizing etc. later

    def __str__(self) -> str:
        return (f'MultiViewBmodeVideo(n_frame={self.n_frame}, n_view={self.n_view}, ' \
                f'image_shape={self.image_shape}, view_images.shape={self.view_images.shape})')
