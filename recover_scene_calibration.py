'''
This script exploits VGGT model to transform a set of unposed frames in a calibrated NeRF-like dataset.
'''

# Import standard modules
import torch
import numpy as np
import os
import json
from tqdm import tqdm

# Import VGGT modules
from vggt.models.vggt import VGGT # type: ignore
from vggt.utils.load_fn import load_and_preprocess_images # type: ignore
from vggt.utils.pose_enc import pose_encoding_to_extri_intri # type: ignore

# Import parameters
from cfg import params


# Align ground truth
def align_gt(
        gt_poses: dict,
        max_dist: float
):
    # Define the t_mtx_0 as the first frame in the ground truth
    t_mtx_ref = gt_poses["000.png"]

    # This implementation shall be modified according to the ground truth convention
    for key in gt_poses:

        T = np.linalg.inv(t_mtx_ref) @ gt_poses[key]
        T[1, 2] *= -1
        T[2, 1] *= -1
        T[:3, 3] /= max_dist
        T[2, 3] *= -1
        gt_poses[key] = T

        return gt_poses


# Main function
def main(
        device: str,
        model_name: str,
        src_folder: str,
        dst_folder: str,
        batch_size: int,
        loop_closure: bool 
):
    # ===================================================================================
    # MODEL SETUP & DATASET LOADING
    # ===================================================================================

    # Load VGGT model and set target device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = VGGT.from_pretrained(model_name).to(device)

    # Load uncalibrated dataset (images only) and sort images (acquisition is progressive)
    image_folder = os.path.join(src_folder, "images/")
    image_names = sorted(
        os.listdir(image_folder),
        key=lambda x: int(os.path.splitext(x)[0])
    )

    # ===================================================================================
    # LOADING & POSTPROCESSING OF GROUND TRUTH
    # ===================================================================================

    # Initialize output variables
    estimated_poses = dict()
    gt_poses = dict()

    # Load ground truth (normalized to the maximum distance)
    with open(os.path.join(src_folder, "transforms.json"), "r") as f:
        frames_objs = json.load(f)["frames"]

    max_dist = 0
    for frame in frames_objs:

        # Load the homogeneous camera matrix and store in ground truth dictionary
        t_mtx = np.array(frame["transform_matrix"]).reshape(4, 4)
        gt_poses[frame["file_path"]] = t_mtx

        # Update the maximum distance found
        max_dist = max(max_dist, np.linalg.norm(t_mtx[:3, 3]))

    # Align the ground truth to the VGGT convention
    gt_poses_aligned = align_gt(
        gt_poses=gt_poses,
        max_dist=max_dist
    )

    # ===================================================================================
    # VGGT INFERENCE
    # ===================================================================================

    # Define the step so that each batch starts from the last frame of the previous batch
    step = batch_size - 1
    last_frame_name = None

    # Estimate camera poses for each batch and link batches according to the last frame
    for start in tqdm(range(0, len(image_names), step), desc="Inference"):

        # Define the batch boundaries and define the batch
        end = min(start + batch_size, len(image_names))
        batch_files = image_names[start:end]

        images_path = [os.path.join(image_folder, f) for f in batch_files]
        images = load_and_preprocess_images(images_path).to(device)

        # Estimate camera poses for the batch
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype):

                outputs = model(images)
                extrinsic, _ = pose_encoding_to_extri_intri(
                    outputs["pose_enc"],
                    images.shape[-2:]
                )

                # Transform the batch in numpy array and store it
                batch_poses = extrinsic[0].cpu().numpy()

                local_T = {}
                for i, pose in enumerate(batch_poses):

                    T = np.eye(4)
                    T[:3, :3] = pose[:, :3]
                    T[:3, 3] = pose[:, 3]

                    local_T[batch_files[i]] = T

                # If it is the first batch, keep it as it is
                if last_frame_name is None:

                    for name in batch_files:
                        estimated_poses[name] = local_T[name]

                    last_frame_name = batch_files[-1]

                # For all the other batches, link the batch to the last camera pose estimated
                else:

                    prev_name = last_frame_name

                    T_prev_global = estimated_poses[prev_name]
                    T_prev_local = local_T[prev_name]

                    T_delta = T_prev_global @ np.linalg.inv(T_prev_local)

                    for name in batch_files:

                        if name in estimated_poses:
                            continue

                        estimated_poses[name] = T_delta @ local_T[name]

                    last_frame_name = batch_files[-1]


if __name__ == "__main__":
    main(**params)