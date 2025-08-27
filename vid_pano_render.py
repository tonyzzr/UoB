import logging
import os
from manual_reg_ui import select_recording_interactively
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

    # load all_frames_pose.csv
    import pandas as pd
    pose_csv_path = os.path.join(dataset_dir, 
                                 recording_name,
                                 'all_frame_poses.csv')
    pose_df = pd.read_csv(pose_csv_path)
    n_frame = pose_df.shape[0]
    n_views = mvbvs['lftx'].view_images.shape[1]

    fuser = 'weighted_mean'

    import torch
    from tqdm import tqdm
    from legacy.model.rigid_link import RigidLink
    

    pbar = tqdm(range(n_frame))
    for frame_index in pbar:
        # import pdb; pdb.set_trace()
        logging.info(f"Processing frame {frame_index}")
        theta_values = pose_df.iloc[frame_index, 1:].values
        rl, panoramas = {}, {}
        for tx_mode in ['lftx', 'hftx']:
            mvbv = mvbvs[tx_mode]
            rl = RigidLink(n=mvbv.view_images.shape[1], 
                           length=mvbv.aperture_size)
            rl.set_origin(mvbv.origin[0], mvbv.origin[1])
            rl.set_thetas(torch.tensor(theta_values).float())
            rl.calc_rela_poses()
            rl.calc_global_poses()
            rl.calc_joint_locations()
            homographies = [
                rl.global_poses[i].matrix()[[0,1,3],:][:, [0,1,3]].numpy() for i in range(n_views)
            ]
            images = [
                mvbv.view_images[frame_index, i, :, :].numpy() for i in range(n_views)
            ]
            masks = [
                mvbv.view_masks[0, i, :, :].numpy() for i in range(n_views)
            ]
            panoramas[tx_mode] = create_panorama(images, masks, homographies)
            # import pdb; pdb.set_trace()
            save_panorama(panoramas[tx_mode], 
                          dataset_dir, recording_name, 
                          fuser, frame_index, tx_mode)


    return

def save_panorama(panorama, dataset_dir, 
                  recording_name, fuser, frame_index, tx_mode):
    import imageio
    import numpy as np
    save_path = os.path.join(dataset_dir, 
                                    recording_name,
                                    f'panoramas/{fuser}_fuser/{tx_mode}',
                                    f'frame_{frame_index:04d}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    panorama_np = panorama.numpy()
    panorama_np /= panorama_np.max()
    panorama_np = (panorama_np * 255).astype(np.uint8)
    panorama_np = np.repeat(panorama_np[..., np.newaxis], 3, axis=-1)
    imageio.imwrite(save_path, panorama_np)

    # import pdb; pdb.set_trace()
    return



def create_panorama(images, masks, homographies, fuser='weighted_mean'):
    import cv2
    import numpy as np
    from dynamic_canvas_creation import calculate_dynamic_canvas

    (canvas_width, canvas_height), adjusted_homographies = calculate_dynamic_canvas(
        images=images,
        homographies=homographies
    )
    warped_images = []
    warped_masks = []

    for i, (image, adjusted_H) in enumerate(zip(images, adjusted_homographies)):
        # Warp image to canvas
        warped_image = cv2.warpPerspective(
            image.astype(np.float32), 
            adjusted_H, 
            (canvas_width, canvas_height)
        )
        warped_images.append(warped_image)

        warped_mask = cv2.warpPerspective(
            masks[i].astype(np.float32), 
            adjusted_H, 
            (canvas_width, canvas_height)
        )
        warped_masks.append(warped_mask)

    import torch
    warped_images_tensor = torch.from_numpy(np.stack(warped_images))
    warped_masks_tensor = torch.from_numpy(np.stack(warped_masks))

    from legacy.model.image_fusion import weighted_mean_fuser, mean_fuser

    if fuser == 'weighted_mean':
        panorama = weighted_mean_fuser(warped_images_tensor, 
                                        warped_masks_tensor)
    elif fuser == 'mean':
        panorama = mean_fuser(warped_images_tensor)
    else:
        raise ValueError(f"Invalid fuser: {fuser}")
    return panorama




if __name__ == '__main__':
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
