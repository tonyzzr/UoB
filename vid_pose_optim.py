'''
This script is used to optimize the pose of the video frames using
the PixelDomainOptimization class.

We will take a reference frame, read the pose data from the csv file,
and then optimize the pose of subsequent frames using the loaded one 
as initial guess.

We will firstly optimize on the lftx and then the hftx, and save the
optimized pose data to a csv file, respectively.

'''

def wrap_mvbvs_to_mvbs(mvbvs, frame_ind):
    from legacy.data.multiview_bmode import MultiViewBmode

    mvbs = {}

    for key in mvbvs.keys():
        mvbs[key] = MultiViewBmode(
        n_view = mvbvs[key].n_view,
        image_shape = mvbvs[key].image_shape,
        origin = mvbvs[key].origin,
        aperture_size = mvbvs[key].aperture_size,
        view_images = mvbvs[key].view_images[frame_ind, ...], # this part we may need to do normalization
        view_masks = mvbvs[key].view_masks[0, ...],
        mat_source_file = '',#mvbvs[key].mat_source_file,
        bmode_config_file = '',#mvbvs[key].bmode_config_file,
        )

    return mvbs

import torchvision
def save_tissue_maps_image(tissue_maps, save_path):
    import torch

    for key in tissue_maps.keys():
        tissue_map = tissue_maps[key]
        tissue_map_max = torch.max(tissue_map)
        tissue_map /= tissue_map_max

        torchvision.utils.save_image(tissue_map, save_path + f'{key}.jpg')

    return

def save_pdo_results(work_dir, pdo, tissue_maps, best_poses, frame_index):
    '''
     We will save the tissue maps and the optimized poses to the dataset folder.
     For poses, we will create a new csv file with the optimized poses.
     For tissue maps, we will create a new folder with the tissue maps,
     in which each frame is a .png file and the name of the file is the frame index
     and the tx_mode.

     input: 
     best_poses: a np.array of shape (n_theta, )
    '''
    import os
    import pandas as pd

    # tissue maps
    tissue_maps_save_dir = os.path.join(work_dir, 'tissue_maps')
    os.makedirs(tissue_maps_save_dir, exist_ok=True)

    tissue_maps_save_path = os.path.join(tissue_maps_save_dir, 
                                         f'frame_{frame_index}_')
    # save_tissue_maps_image(tissue_maps, tissue_maps_save_path)

    # poses
    import numpy as np
    best_poses_in_radian = best_poses
    best_poses_in_degree = np.rad2deg(best_poses_in_radian)
    poses_save_path = os.path.join(work_dir, 'all_frame_poses.csv')
    if os.path.exists(poses_save_path) and (frame_index > 0):
        pose_df = pd.read_csv(poses_save_path)
        # if this is not the first frame, we will append the new row to the dataframe
        # if this is the first frame, we will create a new dataframe
        # and overwrite the existing file
    else:
        columns = ['frame_no'] + [f'theta_{i}' 
                                  for i in range(len(best_poses_in_degree))]
        pose_df = pd.DataFrame(columns=columns)

    # add a new row to the dataframe
    new_row = {'frame_no': frame_index}
    new_row.update({f'theta_{i}': best_poses_in_degree[i] 
                    for i in range(len(best_poses_in_degree))})
    pose_df = pd.concat([pose_df, pd.DataFrame([new_row])], ignore_index=True)

    pose_df.to_csv(poses_save_path, index=False)

    return
    
def run_pdo_loop(mvbvs, init_rela_poses, work_dir):
    import torch
    from tqdm import tqdm

    # work_dir = ''
    n_frame = mvbvs['lftx'].n_frame

    for frame_index in tqdm(range(n_frame)):
        # input('Press Enter to continue...')
        print(f'frame {frame_index}')

        # clear cache
        torch.cuda.empty_cache()

        # wrap a new mvbs
        mvbs = wrap_mvbvs_to_mvbs(mvbvs, frame_ind=frame_index)

        # find inital pose
        if frame_index == 0:
            rela_poses_init = init_rela_poses
            rela_poses_from_last_frame = None
        else:
            rela_poses_init = rela_poses_from_last_frame


        # pdo
        # try:
        tissue_maps, best_poses, pdo = run_pdo(mvbs, rela_poses_init)

        # save results
        save_pdo_results(work_dir, pdo, tissue_maps, best_poses, frame_index)
        # rela_poses_from_last_frame = wrap_pose_data_to_file(best_poses, '')
        from legacy.data.poses import RelativePoses
        rela_poses_from_last_frame = RelativePoses(
            thetas = torch.tensor(best_poses),
            notes = {
                'batched_pose_opt': True,
                'frame_index': frame_index,
            },
        )
        # import pdb; pdb.set_trace()


        # except Exception as e:
        #     print(f"An error occurred with frame {frame_index}: {e}")
        #     continue
        # pass

def run_pdo(mvbs, rela_poses_init):
    from legacy.model.spatial_map import PixelDomainOptimization, SpatialMapNet
    from legacy.model.spatial_map import LaplacianVarianceLoss
    from legacy.model.image_fusion import mean_fuser
    import torch
    import numpy as np


    pdo = PixelDomainOptimization(mvbs = mvbs,
                                    init_rela_poses = rela_poses_init,
                                    ref_rela_poses = rela_poses_init)

    pdo.set_model(model_class=SpatialMapNet,
                    fuser = mean_fuser)
    pdo.set_optimizer(optimizer_class=torch.optim.Adam,
                        lr=1e-3, betas=(0.9, 0.999),
                        weight_decay=10,
                        )
    pdo.set_loss_func(loss_func=LaplacianVarianceLoss(n_level=5,
                                                        target_level=1),
                        train_data_key='hftx',
    )
    pdo.set_trainer(n_epochs=100, device='cuda')
    pdo.set_ref_poses(ref_poses = rela_poses_init) # set pose difference criteria here

    tissue_maps = pdo.run()

    best_poses_ind = np.argmin(np.array(pdo.training_history['loss']))
    best_poses = pdo.training_history['rela_poses_in_radian'][best_poses_ind]


    return tissue_maps, best_poses, pdo

def main(recording_name:str):
    import os
    import pandas as pd
    from legacy.load_dataset import load_and_visualize_frame

    logging.info(f"Selected recording: {recording_name}")
    logging.info(f"Loading data...")
    dataset_dir = 'data/processed'
    dataset_path = os.path.join(dataset_dir, 
                                recording_name,
                                'combined_mvbv.pkl')
    work_dir = os.path.join(dataset_dir, 
                            recording_name)

    mvbvs = load_and_visualize_frame(dataset_path)

    logging.info("\nDataset Information:")
    logging.info(f"Number of frames in LFTX: {mvbvs['lftx'].view_images.shape[0]}")
    logging.info(f"Number of views in LFTX: {mvbvs['lftx'].view_images.shape[1]}")
    logging.info(f"Frame shape LFTX: {mvbvs['lftx'].view_images.shape[2:]}")
    logging.info(f"\nNumber of frames in HFTX: {mvbvs['hftx'].view_images.shape[0]}")
    logging.info(f"Number of views in HFTX: {mvbvs['hftx'].view_images.shape[1]}")
    logging.info(f"Frame shape HFTX: {mvbvs['hftx'].view_images.shape[2:]}")

    logging.info(f"Loading reference frame pose...")
    ref_frame_pose_path = os.path.join(dataset_dir, 
                                recording_name,
                                'ref_frame_pose.csv')
    pose_df = pd.read_csv(ref_frame_pose_path)

    logging.info(f"Reference frame index: {pose_df.iloc[0, 0]}")
    logging.info(f"Reference frame pose: {pose_df.iloc[0, 1:].values}")

    from legacy.data.poses import RelativePoses
    import torch
    import numpy as np

    thetas_init_in_degree = pose_df.iloc[0, 1:].values
    thetas_init_in_radian = np.deg2rad(thetas_init_in_degree)

    rela_poses_init = RelativePoses(
        thetas = torch.tensor(thetas_init_in_radian).double(),
        notes = {
            'batched_pose_opt': True,
            'frame_index': pose_df.iloc[0, 0],
        },
    )

    run_pdo_loop(mvbvs, rela_poses_init, work_dir)


if __name__ == "__main__":
    import sys

    import logging
    logging.basicConfig(level=logging.INFO)


    from omegaconf import OmegaConf
    from manual_reg_ui import select_recording_interactively

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
    
    main(recording_name)