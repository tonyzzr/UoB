from dataclasses import dataclass
import numpy as np
import cv2

from tqdm import tqdm

# --- RESIZE IMAGE --- #
def resize_in_scale(src:np.ndarray(None), dst_size:tuple) -> np.ndarray(None):
    '''
        Return B mode image sequences in scale. 
        
    '''

    _, _, ntrans, nfrm = src.shape
    h, w = dst_size

    dst = np.zeros((h, w, ntrans, nfrm))

    # cv2.resize can only take 3 dimensional images (w, h, c)
    # so we use for loop to do this here (in the future we may convert c = f * ntrans, and reshape later)
        
    for i in range(nfrm):
        dst[..., i] = cv2.resize(src[..., i],
                                dsize=(w, h),
                                interpolation=cv2.INTER_LINEAR
                            )
        
    return dst # (h, w, ntrans, f)


# --- PRODUCE MASKS --- #
@dataclass
class MaskSetting:
    enable: bool = False
    main_lobe_beamwidth: float = 30
    soft_boundary: bool = True
    softness: float = 0.3

def generate_mask(theta_v, setting: MaskSetting):
    
    # not enabled - no masking
    if not setting.enable:
        return np.ones(theta_v.shape)

    # enabled
    angle = setting.main_lobe_beamwidth    

    if setting.soft_boundary:
            mask = 1 - 1 / (1 + np.exp(-setting.softness * (theta_v - angle)))
    else:
            mask = theta_v < angle
            
    return mask


# --- LOG COMPRESSION --- #
@dataclass
class LogCompressionSetting:
    enable: bool = False
    dynamic_range: int = 60
    max_value: int = 1024

def log_compression(src, setting:LogCompressionSetting):

    if not setting.enable:
        return src

    DR = setting.dynamic_range
    data = src.copy()

    eps = 0.01
    data = data + eps

    if setting.max_value is None:
        # flexible maximum value
        img = 20 * np.log10(data / np.amax(data)) + DR
    else:
        # fixed maximum value
        img = 20 * np.log10(data /setting.max_value) + DR

    img = img / DR * 255
    img = img * (img > 0)
    
    return img



# --- SPECKLE REDUCTION --- #
@dataclass
class SpeckleReductionSetting:
    enable: bool = False
    med_blur_kernal: int = 3
    nlm_h: int = 9
    nlm_template_window_size: int = 7
    nlm_search_window_size: int = 11

def speckle_reduction(src, setting:SpeckleReductionSetting):

    if not setting.enable:
        return src

    _, _, ntrans, nfrm = src.shape
    dst = np.zeros(src.shape)

    for i in tqdm(range(nfrm)):
        for c in range(ntrans):
            img = np.array(src[..., c, i]).astype(np.uint8)

            img = cv2.medianBlur(img, setting.med_blur_kernal)
            img = cv2.fastNlMeansDenoising(img, 
                                        h                  = setting.nlm_h, 
                                        templateWindowSize = setting.nlm_template_window_size, 
                                        searchWindowSize   = setting.nlm_search_window_size,
                                        )
            
            dst[..., c, i] = img

    return dst



# --- REJECT GRATING LOBE ARTIFACT --- #
@dataclass
class RejectGratingLobeSetting:
    enable: bool = False

def reject_grating_lobe_artifact(src      :np.ndarray(None),
                                 mask     :np.ndarray(None),
                                 setting  :RejectGratingLobeSetting):

    # disabled
    if not setting.enable:
        return src

    # enabled
        # shape of the mask (1 x ntrans x h x w)
        # shape of the src  (h x w x ntrans x f)

    __mask = np.transpose(mask, (2, 3, 1, 0))

    return __mask * src # element-wise multiplication
    



# --- HISTOGRAM MATCHING --- #
@dataclass
class HistogramMatchSetting:
    enable: bool = False
    ref_ind: int = 0
    background_removal: bool = True

def histogram_match(src, setting:HistogramMatchSetting):

    if not setting.enable:
        return src

    _, _, ntrans, _ = src.shape
    dst = np.zeros(src.shape)
    ref = src[..., setting.ref_ind, :].copy()
    
    from skimage import exposure
    
    for i in range(ntrans):
        img = src[..., i, :]
        matched = exposure.match_histograms(img, ref, channel_axis=-1)
        matched = matched - np.amin(matched) # remove the background imposed by histogram matching
        dst[..., i, :] = matched
       
    return dst   



# --- APPLY TGC --- #
@dataclass
class ApplyTGCSetting:
    enable: bool = False
    tgc_threshold: float = 0.8
    tgc_slope: float = 10

def apply_tgc(src, setting:ApplyTGCSetting):
    if not setting.enable:
        return src
    
    h, w, _, _, = src.shape
    
    tgc_x = np.linspace(0, 1, h).reshape((-1, 1))
    tgc_y = 1 / (1 + \
                    np.exp(-(tgc_x - setting.tgc_threshold) * setting.tgc_slope)
                    )
    tgc_mask = np.repeat(1 - tgc_y, w, axis=1).reshape((h, w, 1, 1))

    return src * tgc_mask


def main():
    return

if __name__ == "__main__":
    main()
