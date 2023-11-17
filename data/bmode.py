from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

from .mat import MatData
from . import process

@dataclass
class TransPos:
    '''
        Coordinates of left edge and righr edge
        of a transducer relative to the B-mode
        image. Unit in pixels.
    '''
    left_edge_coord : np.ndarray((3, 1)) # use homogenous coordinates (x, y, 1)
    right_edge_coord: np.ndarray((3, 1))  

    
    @property
    def length(self) -> float:
        vec = self.left_edge_coord - self.right_edge_coord
        return np.linalg.norm(vec)

    @property
    def centroid(self) -> np.ndarray((2,1)):
        return (self.left_edge_coord + self.right_edge_coord) / 2

# --- ZERO PADDING SETTING --- #
# zero padding is put here as we need to simultanenously
# opertate the trans_pos and B-mode images, using Bmode instance
# as input argument
@dataclass
class ZeroPadSetting:
    enable: bool = False
    top_padding_ratio: float = 0.
    bottom_padding_ratio: float = 0.
    left_padding_ratio: float = 0.
    right_padding_ratio: float = 0.
    need_transfomation_matrix: bool = False


@dataclass
class BmodeConfig:
    scale_bar: float

    mask_setting:                       process.MaskSetting
    log_compression_setting:            process.LogCompressionSetting
    speckle_reduction_setting:          process.SpeckleReductionSetting
    reject_grating_lobe_setting:        process.RejectGratingLobeSetting
    histogram_match_setting:            process.HistogramMatchSetting
    time_gain_compensation_setting:     process.ApplyTGCSetting



@dataclass
class Bmode:
    num_trans: int
    scale_bar: float            # how large is a pixel in mm
    b_img_seq: np.ndarray(None) # (f x ntrans x h x w)
    trans_pos: dict             # {1:<TransPos>, ...}
    mask_seq:  np.ndarray(None) # (1 x ntrans x h x w)
    config:    BmodeConfig

    def __post_init__(self):
        self.trans_pos_gap = {}
        gap = 1.0 / self.scale_bar # should be calculated from mat data in future

        for idx in range(self.num_trans):
            # print(self.trans_pos[idx].left_edge_coord)

            left_edge_coord  = self.trans_pos[idx].left_edge_coord + np.array([-gap/2, 0, 0]).reshape((3, 1))
            right_edge_coord = self.trans_pos[idx].right_edge_coord + np.array([gap/2, 0, 0]).reshape((3, 1))

            self.trans_pos_gap[idx] = TransPos(
                left_edge_coord =  left_edge_coord,
                right_edge_coord =  right_edge_coord,
                )

    def __str__(self) -> str:
        return f'num_trans: {self.num_trans} \n' + \
               f'scale_bar (mm per pixel): {self.scale_bar} \n' + \
               f'b_img_seq (frame, trans, w, h): {self.b_img_seq.shape} \n' + \
               f'trans_pos: {self.trans_pos} \n' +\
               f'mask_seq (frame, trans, w, h): {self.mask_seq.shape}'
    
    def show_b_img_seq(self, frame = 0):
        print(self.b_img_seq.shape)

        nrows = int(np.sqrt(self.num_trans))
        ncols = self.num_trans // nrows
        
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
        for i in range(self.num_trans):
            r = i // ncols
            c = i - r * ncols

            ax[r, c].imshow(self.b_img_seq[frame, i, ...])
            ax[r, c].set_title('AS_%s' % i)
    
    def show_mask_seq(self):
        print(self.mask_seq.shape)

        nrows = int(np.sqrt(self.num_trans))
        ncols = self.num_trans // nrows
        
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
        for i in range(self.num_trans):
            r = i // ncols
            c = i - r * ncols
            ax[r, c].imshow(self.mask_seq[0, i, ...])
            ax[r, c].set_title('AS_%s' % i)
        
    def show_trans_pos(self, frame = 0):
        '''
            Plot the transducer positions.
        '''
        nrows = int(np.sqrt(self.num_trans))
        ncols = self.num_trans // nrows
        
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
        for i in range(self.num_trans):
            r = i // ncols
            c = i - r * ncols

            ax[r, c].imshow(self.b_img_seq[frame, i, ...])
            ax[r, c].set_title('AS_%s' % i)

            ax[r, c].plot(self.trans_pos[i].left_edge_coord[0, 0], \
                          self.trans_pos[i].left_edge_coord[1, 0], 'r.')

            ax[r, c].plot(self.trans_pos[i].right_edge_coord[0, 0], \
                          self.trans_pos[i].right_edge_coord[1, 0], 'r.')

        plt.show()
        pass

class BmodeBuilder:
    def __init__(self, mat_data:MatData, config:BmodeConfig) -> None:
        self.mat_data   = mat_data
        self.config     = config

    
    def build_b_mode(self) -> Bmode:
        self.trans_pos = self.__calc_trans_pos()
        self.mask_seq  = self.__build_mask_seq()
        self.b_img_seq = self.__build_b_img_seq()
        

        return Bmode(
                config    = self.config,
                num_trans = self.num_trans,
                scale_bar = self.config.scale_bar,

                trans_pos = self.trans_pos,

                mask_seq  = self.mask_seq,
                b_img_seq = self.b_img_seq,
                )

    @property
    def num_trans(self) -> int:
        return len(self.mat_data.pdata.keys())

    @property
    def img_size(self) -> tuple:
        ''' 
            Calculate size of in-scaled image (h, w).
        '''
        wl = self.mat_data.trans['wavelengthMm']
        PData = self.mat_data.pdata[0]

        dx = PData['dx_wl'] * wl
        dz = PData['dz_wl'] * wl       
        h = int(PData['nz'] * dz / self.config.scale_bar)
        w = int(PData['nx'] * dx / self.config.scale_bar)
        return (h, w)


    def __extract_imgdata(self, mat_data:MatData) -> np.ndarray(None):
        '''
            Extract imgdata from mat_data and build b_img_seq in size of (w x h x ntrans x f).
        '''
        b_img_seq = []
        for i in range(self.num_trans):
            img_data = mat_data.imgdata[i][:, 0, ...]               # (f x 1 x h x w) -> (f x h x w)
            b_img_seq.append(img_data)                             # [ntrans x (f x h x w)]

        b_img_seq = np.array(b_img_seq)                            # (ntrans x f x h x w)
        b_img_seq = np.transpose(b_img_seq, (3, 2, 0, 1))          # (ntrans x f x h x w) -> (w x h x ntrans x f)

        return b_img_seq

    def __calc_trans_pos(self) -> dict: 
        '''
            Transducer edge positions in the (in-scale) image.
            The output is of type: dict[int] = TransPos
        '''
        Trans = self.mat_data.trans
        scale_bar = self.config.scale_bar

        Trans_xPos = Trans['ElementPos'][:, 0]
        # Trans_zPos = Trans['ElementPos'][:, 2]
        
        wl = Trans['wavelengthMm']
        PDataNum = self.num_trans

        nEleApe = 32 # this should be modified per different mat data
        relaTransXPos = {}
        
        for PDataIdx in range(PDataNum):            

            PData = self.mat_data.pdata[PDataIdx]
            PDataLeft  = PData['Ox_wl'] * wl


            left_edge  = Trans_xPos[nEleApe * PDataIdx]  - PDataLeft
            right_edge = Trans_xPos[nEleApe * (PDataIdx+1) - 1] - PDataLeft


            relaTransXPos[PDataIdx] = TransPos(
                left_edge_coord  = np.array([left_edge  / scale_bar, 0, 1]).reshape((3, 1)),
                right_edge_coord = np.array([right_edge / scale_bar, 0, 1]).reshape((3, 1)),
             ) # relative transducer positions (unit in pixel)
             # the gap between two is not considered here
        
        return relaTransXPos
    
    def __build_mask_seq(self) -> np.array(None):
        '''
            Mask sequence calculated based on angular beamwidth, image size, and trans_pos.
        
            mask_seq -> (1, ntrans, h, w)
        '''
        angle = self.config.mask_setting.main_lobe_beamwidth
        h, w = self.img_size

        mask_seq = []

        for i in range(self.num_trans):

            # step 1 - calculate the virtual apex
            trans_pos = self.trans_pos[i]
            left_edge = trans_pos.left_edge_coord[0, 0]
            right_edge = trans_pos.right_edge_coord[0, 0]

            # print(left_edge, right_edge)

            apex_x = (left_edge + right_edge) / 2
            apex_z = - (1 /np.tan(angle/180 * np.pi)) * (right_edge - left_edge) / 2

            # print(apex_x, apex_z)

            # step 2 - calculate the position matrix of each pixel point
            z = np.linspace(0 - apex_z, h - apex_z -1, h)
            x = np.linspace(0 - apex_x, w - apex_x -1, w)
            xv, zv = np.meshgrid(x, z, indexing = 'xy')

            # plt.imshow(xv)
            # plt.title('xv')
            # plt.colorbar()
            # plt.show()

            # plt.imshow(zv)
            # plt.title('zv')
            # plt.colorbar()
            # plt.show()


            theta_v = np.arctan(zv / np.abs(xv))
            theta_v = 90 - (theta_v / np.pi * 180)

            # plt.imshow(theta_v)
            # plt.title('theta_v')
            # plt.colorbar()
            # plt.show()


            # step 3 - generate the mask
            mask = process.generate_mask(theta_v, self.config.mask_setting)

            # plt.imshow(mask)
            # plt.title('mask')
            # plt.colorbar()
            # plt.show()
            
            mask_seq.append(mask)

        # (ntrans, h, w) -> (1 x ntrans x h x w)
        mask_seq = np.array(mask_seq)
        mask_seq = np.reshape(mask_seq, (1, self.num_trans, h, w)) # (1, ntrans, h, w)
        return mask_seq

    def __build_b_img_seq(self) -> np.ndarray(None):
        config = self.config

        b_img_seq = self.__extract_imgdata(self.mat_data) 


        # image processing (h, w, ntrans, f) -- suits cv2 functions better
        b_img_seq = process.resize_in_scale(src       = b_img_seq, 
                                            dst_size  = self.img_size, # (h, w).
                                           ) # output shape (h, w, ntrans, f)
        print('b_img_seq_shape', b_img_seq.shape) 
        
        b_img_seq = process.log_compression(src     = b_img_seq,
                                            setting = config.log_compression_setting,
                                           )                                
        
        b_img_seq = process.speckle_reduction(src     = b_img_seq,
                                              setting = config.speckle_reduction_setting,
                                             )

        print('mask_seq_shape', self.mask_seq.shape)
        b_img_seq = process.reject_grating_lobe_artifact(src       = b_img_seq,
                                                         mask      = self.mask_seq,
                                                         setting   = config.reject_grating_lobe_setting,
                                                        )                                      

        b_img_seq = process.apply_tgc(src = b_img_seq,
                                      setting = config.time_gain_compensation_setting,
                                     )

        b_img_seq = process.histogram_match(src       = b_img_seq,
                                            setting   = config.histogram_match_setting,
                                           )

        # (h, w, ntrans, f) -> (f, ntrans, h, w)
        b_img_seq = np.transpose(b_img_seq, (3, 2, 0, 1)) 
        return b_img_seq


# --- ZERO PADDING  --- #
def zero_padding(bmode:Bmode, setting:ZeroPadSetting):
    
    '''
        This should take a Bmode as input since we need to update 
        the images and the trans_pos together.

        b_img_seq.shape = (frame x ntrans x h x w)

    '''

    import copy

    # copy bmode object
    bmode_padded = copy.deepcopy(bmode)

    b_img_seq = bmode.b_img_seq
    mask_seq = bmode.mask_seq

    # 1 - create a zero_padding image frame
    frm, n_trans, h, w = b_img_seq.shape

    h_padded = int(h * (1 + setting.top_padding_ratio + setting.bottom_padding_ratio))
    w_padded = int(w * (1 + setting.left_padding_ratio + setting.right_padding_ratio))

    # 2 - align left top corner
    left_margin = int(w *  setting.left_padding_ratio)
    top_margin = int(h * setting.top_padding_ratio)

    transf_matrix = np.float32([[1, 0, left_margin], 
                            [0, 1, top_margin], 
                            [0, 0, 1]])


    b_img_seq_padded = np.zeros((frm, n_trans, h_padded, w_padded))
    b_img_seq_padded[..., top_margin:top_margin+h, left_margin:left_margin+w] = b_img_seq

    mask_seq_padded = np.zeros((1, n_trans, h_padded, w_padded))
    mask_seq_padded[..., top_margin:top_margin+h, left_margin:left_margin+w] = mask_seq

    bmode_padded.b_img_seq = b_img_seq_padded
    bmode_padded.mask_seq = mask_seq_padded

    # 3 - change trans_pos accordingly
    # print(transf_matrix.shape)


    for i in bmode.trans_pos.keys():
        trans_pos = bmode.trans_pos[i]

        left_edge_coord_padded = transf_matrix @ trans_pos.left_edge_coord.copy()
        right_edge_coord_padded = transf_matrix @ trans_pos.right_edge_coord.copy()

        bmode_padded.trans_pos[i] = TransPos(
            left_edge_coord = left_edge_coord_padded,
            right_edge_coord = right_edge_coord_padded,
        )
    bmode_padded.__post_init__()

    if setting.need_transfomation_matrix:
        return bmode_padded, transf_matrix
    else:
        return bmode_padded


# ------ ------ #




def main():
    return

if __name__ == "__main__":
    main()
