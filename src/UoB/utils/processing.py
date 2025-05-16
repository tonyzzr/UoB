import numpy as np
import cv2
from tqdm import tqdm
from typing import Any # Using Any for settings for now, will be replaced by specific dataclasses later

# --- RESIZE IMAGE --- #
def resize_in_scale(src: np.ndarray, dst_size: tuple) -> np.ndarray:
    '''
        Return B mode image sequences in scale.
        Input shape: (h_in, w_in, ntrans, nfrm)
        Output shape: (h_out, w_out, ntrans, nfrm)
    '''
    _, _, ntrans, nfrm = src.shape
    h, w = dst_size

    # Use src.dtype for the destination array
    dst = np.zeros((h, w, ntrans, nfrm), dtype=src.dtype)

    # cv2.resize can only take 3 dimensional images (w, h, c) or 2D
    # Iterate through frames and transducers
    for f in range(nfrm):
        for t in range(ntrans):
            dst[..., t, f] = cv2.resize(src[..., t, f],
                                        dsize=(w, h), # Note: dsize is (width, height) for cv2
                                        interpolation=cv2.INTER_LINEAR)
            
    return dst # (h, w, ntrans, f)

# --- PRODUCE MASKS --- #
# Note: MaskSetting dataclass definition will be moved to src/UoB/data/formats.py
def generate_mask(theta_v: np.ndarray, setting: Any) -> np.ndarray:
    ''' Generates a mask based on angle theta_v and MaskSetting. '''
    if not setting.enable:
        return np.ones(theta_v.shape)

    angle = setting.main_lobe_beamwidth

    if setting.soft_boundary:
            mask = 1 - 1 / (1 + np.exp(-setting.softness * (theta_v - angle)))
    else:
            mask = theta_v < angle
            
    return mask.astype(float) # Ensure output is float for multiplication

# --- LOG COMPRESSION --- #
# Note: LogCompressionSetting dataclass definition will be moved to src/UoB/data/formats.py
def log_compression(src: np.ndarray, setting: Any) -> np.ndarray:
    ''' Applies log compression to the image sequence. '''
    if not setting.enable:
        return src

    DR = setting.dynamic_range
    data = src.copy()

    eps = 1e-9 # Use a small epsilon to avoid log(0)
    data = data + eps

    # Ensure data is positive before log
    data[data <= 0] = eps

    if setting.max_value is None:
        # Flexible maximum value, calculate per-frame, per-transducer max?
        # Or global max? For now, assume global max for simplicity, might need refinement.
        # Avoid division by zero if max is zero.
        max_val = np.amax(data)
        if max_val <= 0: max_val = eps
        img = 20 * np.log10(data / max_val) + DR
    else:
        # Fixed maximum value
        max_val = setting.max_value
        if max_val <= 0: max_val = eps # Avoid division by zero/negative
        img = 20 * np.log10(data / max_val) + DR

    # Normalize to 0-255 range
    img = img / DR * 255.0
    img[img < 0] = 0 # Clip values below 0
    img[img > 255] = 255 # Clip values above 255
    
    # Ensure output is float64 as calculations likely promoted it
    return img.astype(np.float64)

# --- SPECKLE REDUCTION --- #
# Note: SpeckleReductionSetting dataclass definition will be moved to src/UoB/data/formats.py
def speckle_reduction(src: np.ndarray, setting: Any) -> np.ndarray:
    ''' Applies speckle reduction using median blur and NLM denoising. '''
    if not setting.enable:
        return src

    h, w, ntrans, nfrm = src.shape
    dst = np.zeros(src.shape)

    # NLM requires uint8 input
    src_uint8 = src.astype(np.uint8)

    for i in tqdm(range(nfrm), desc="Speckle Reduction"):
        for c in range(ntrans):
            img = src_uint8[..., c, i]

            # Median Blur
            if setting.med_blur_kernal > 1:
                img = cv2.medianBlur(img, setting.med_blur_kernal)

            # NLM Denoising
            if setting.nlm_h > 0:
                 img = cv2.fastNlMeansDenoising(img,
                                             h                  = float(setting.nlm_h), # h needs to be float
                                             templateWindowSize = setting.nlm_template_window_size,
                                             searchWindowSize   = setting.nlm_search_window_size)
            
            dst[..., c, i] = img

    return dst

# --- REJECT GRATING LOBE ARTIFACT --- #
# Note: RejectGratingLobeSetting dataclass definition will be moved to src/UoB/data/formats.py
def reject_grating_lobe_artifact(src: np.ndarray, mask: np.ndarray, setting: Any) -> np.ndarray:
    ''' Applies the pre-calculated mask to reject grating lobes. '''
    # Input shapes:
    # src: (h, w, ntrans, f)
    # mask: (1, ntrans, h, w) -> transpose needed

    if not setting.enable:
        return src

    # Transpose mask to match src dimensions for broadcasting: (h, w, ntrans, 1)
    mask_transposed = np.transpose(mask, (2, 3, 1, 0))

    return mask_transposed * src # Element-wise multiplication

# --- HISTOGRAM MATCHING --- #
# Note: HistogramMatchSetting dataclass definition will be moved to src/UoB/data/formats.py
def histogram_match(src: np.ndarray, setting: Any) -> np.ndarray:
    ''' Matches the histogram of each view/frame to a reference view/frame. '''
    if not setting.enable:
        return src

    h, w, ntrans, nfrm = src.shape
    dst = np.zeros(src.shape)
    
    # Assuming reference index is for the transducer dimension
    ref_ind = setting.ref_ind
    if ref_ind < 0 or ref_ind >= ntrans:
        print(f"Warning: Invalid ref_ind {ref_ind} for histogram matching. Using 0.")
        ref_ind = 0
        
    # Reference can be the average of the reference transducer over all frames,
    # or a specific frame. Let's use the first frame of the reference transducer for now.
    ref_image = src[..., ref_ind, 0].copy().astype(src.dtype) # Match to first frame of ref transducer
    
    try:
        from skimage import exposure
    except ImportError:
        print("Warning: scikit-image not installed. Histogram matching will be skipped.")
        print("Please install it: pip install scikit-image")
        return src

    for i in tqdm(range(ntrans), desc="Histogram Matching"):
        if i == ref_ind: # Skip matching the reference to itself
            dst[..., i, :] = src[..., i, :]
            continue
            
        # Match each frame of the current transducer to the reference image
        for f in range(nfrm):
            img_to_match = src[..., i, f].copy().astype(src.dtype)
            
            # Match histograms
            matched = exposure.match_histograms(img_to_match, ref_image) # channel_axis is deprecated/not needed for 2D
            
            if setting.background_removal:
                 # Remove potential background shift, ensure non-negative
                 min_val = np.amin(matched)
                 matched = matched - min_val
                 matched[matched < 0] = 0

            dst[..., i, f] = matched.astype(src.dtype)
            
    return dst

# --- APPLY TGC --- #
# Note: ApplyTGCSetting dataclass definition will be moved to src/UoB/data/formats.py
def apply_tgc(src: np.ndarray, setting: Any) -> np.ndarray:
    ''' Applies Time Gain Compensation (TGC) based on depth (h). '''
    if not setting.enable:
        return src
    
    h, w, ntrans, nfrm = src.shape
    
    # Create TGC curve based on depth (h dimension)
    tgc_depth_norm = np.linspace(0, 1, h).reshape((-1, 1)) # Normalize depth from 0 to 1
    
    # Sigmoid function for TGC curve
    tgc_gain = 1 / (1 + np.exp(-(tgc_depth_norm - setting.tgc_threshold) * setting.tgc_slope))
    
    # Invert gain (attenuate deeper regions more) and reshape for broadcasting
    tgc_mask = (1 - tgc_gain).reshape((h, 1, 1, 1)) # Reshape for (h, w, ntrans, nfrm) broadcasting

    # Apply TGC mask
    return src * tgc_mask 