import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass

from . import bmode

@dataclass
class SegData:
  mat_directory: str
  frame_index: int
  lf_imgs: list
  hf_imgs: list
  lf_segs: list
  hf_segs: list
  lf_pad: bmode.ZeroPadSetting
  hf_pad: bmode.ZeroPadSetting

  @property
  def n_view(self):
    return len(lf_imgs)

  def show_seg_data(self,):
    fig, ax = plt.subplots(4, 8, figsize=(8, 4))
    for i in range(8):
      ax[0, i].imshow(self.lf_imgs[i])
      ax[1, i].imshow(self.lf_segs[i])
      ax[2, i].imshow(self.hf_imgs[i])
      ax[3, i].imshow(self.hf_segs[i])

      for j in range(4):
        ax[j, i].axis('off')

    plt.show()

  def __str__(self):
    return f'mat_dir = {self.mat_directory} \n' + \
           f'frame_index = {self.frame_index} \n'

class SegDataLoader():
  def __init__(self, flagging_dir):
    self.flagging_dir = flagging_dir
    self.df = pd.read_csv(Path(flagging_dir) / 'log.csv')

  def get_line_data(self, line_index):
    data_path = self.df.iloc[line_index]['output 2']

    data_dict = self._load_data_dict(data_path)
    results = data_dict['results']
    imgs, segs = [np.array(pil) for pil in results['pil_images']], \
                                         results['parts_imgs_regulated']

    configs = data_dict['configs']
    img_directory = configs['data_directory']  
    frame_index = configs['frame_index'] 
    lf_pad = configs['LF_zero_padding_setting']
    hf_pad = configs['HF_zero_padding_setting']


    line_data = {
        'mat_directory': img_directory,
        'frame_index': frame_index,
        'lf_imgs' : imgs[:8],
        'hf_imgs' : imgs[8:],
        'lf_segs' : segs[:8],
        'hf_segs' : segs[8:],
        'lf_pad' : lf_pad,
        'hf_pad' : hf_pad,
    }
                     
    return SegData(**line_data)


  def _load_data_dict(self, data_path):
    with open(data_path, 'rb') as f:
      data_dict = pickle.load(f)

    return data_dict

