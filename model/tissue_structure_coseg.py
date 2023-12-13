import torch
import numpy as np

import yaml
import random
from operator import itemgetter

from part_cosegmentation import find_part_cosegmentation


class TissueStructureCosegmentation:
  def __init__(self,
               image_paths_dict:dict = {},
               config_path:str = './UoB/examples/coseg_configs.yaml'):

    self.image_paths_dict = image_paths_dict
    self.image_paths = self._convert_image_paths_dict_to_list()

    self.configs = self._load_configs_from_yaml(config_path)

    # initialize
    self.part_imgs, self.pil_imgs = None, None
    self.segmentation_masks = {
        key: [] for key in self.image_paths_dict.keys()
    }

  def _convert_image_paths_dict_to_list(self, ):
    path_list = []
    for key in self.image_paths_dict.keys():
      path_list += self.image_paths_dict[key]

    path_list.sort()
    # print(path_list)

    return path_list

  def _load_configs_from_yaml(self, config_path:str):
    with open(config_path, 'r') as f:
      configs = yaml.safe_load(f)

    if configs['load_size'] == '':
      configs['load_size'] = None

    return configs

  def _run_part_cosegmentation(self, ):

    part_imgs, pil_imgs = find_part_cosegmentation(
        image_paths = self.image_paths,
        **self.configs,
    )
    return part_imgs, pil_imgs

  def _find_mask_centroids(self, class_no = 0):
    '''
    find centroid of a segmentation mask of a class
    '''
    rs, cs = [], []
    for view_index in range(len(self.image_paths)):
      masks = np.array(self.part_imgs[view_index])
      r, c = np.where(masks == class_no)

      rs.append(r)
      cs.append(c)

    rs = np.concatenate(rs)
    cs = np.concatenate(cs)
    centroid = np.array([np.mean(rs), np.mean(cs)])

    return centroid


  def _sort_class_no_by_mask_centroid(self, ):
    class_nos = list(range(self.configs['num_parts']))
    r_centroids = [self._find_mask_centroids(class_no = class_no)[0] \
                   for class_no in class_nos]

    zipped = zip(class_nos, r_centroids)
    sorted_zipped = sorted(zipped, key=lambda x: x[1])
    sorted_class_nos, r_centroids = zip(*sorted_zipped)

    return sorted_class_nos

  def _fill_segmentation_masks_with_sorted_class_no(self, ):
    '''
      sorted_class_nos = [c0', c1', c2', ...] where r0'<=r1'<=r2', ...
      r0' means the smallest r_centroid --- c0' is the value of segmentation mask on the top --- so, new_mask[original_mask == c0'] <- 0

      in general:
        new_mask[original_mask == c_j'] <- j
    '''
    
    sorted_class_nos = self._sort_class_no_by_mask_centroid()
    new_masks = [np.full_like(mask, np.nan) for mask in self.part_imgs]

    for j in range(len(sorted_class_nos)):
      original_label = sorted_class_nos[j]
      new_label = j
      print(f'original_label = {original_label}, new_label = {new_label}')
      
      for i, new_mask in enumerate(new_masks): 
        new_mask[self.part_imgs[i] == original_label] = new_label
        
    return new_masks

  def _set_random_seed(self, seed=6):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Set the random seed for CUDA (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return

  def _part_imgs_to_segmentation_masks(self, ):

    masks = self._fill_segmentation_masks_with_sorted_class_no()

    for key in self.image_paths_dict.keys():
      indices = [i for i, path in enumerate(self.image_paths) \
                  if key in path]
      # print(key, indices, self.image_paths[i])

      masks_tuple = itemgetter(*indices)(masks)
      masks_tensor = torch.tensor(masks_tuple)

      self.segmentation_masks[key] = masks_tensor

    return


  def run(self, ):
    self._set_random_seed()
    self.part_imgs, self.pil_imgs = self._run_part_cosegmentation()
    self._part_imgs_to_segmentation_masks()

    return
