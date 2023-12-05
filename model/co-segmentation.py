#@title TissueStructureCosegmentation class
import yaml
import random
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

    return path_list.sort()

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

  def _find_mask_centroids(self, ):
    raise NotImplementedError

  def _sort_class_no_by_mask_centroid(self, ):
    # raise NotImplementedError
    return self.part_imgs # temporal solution - update later

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

    masks = self._sort_class_no_by_mask_centroid(self.part_imgs)

    for key in self.image_paths_dict.keys():
      indices = [i for i, path in enumerate(self.image_paths) \
                  if key in path]
      self.segmentation_masks[key] = masks[indices]

    return


  def run(self, ):
    self._set_random_seed()
    self.part_imgs, self.pil_imgs = self._run_part_cosegmentation()
    self.segmentation_masks = self._part_imgs_to_segmentation_masks()



    return self.segmentation_masks
