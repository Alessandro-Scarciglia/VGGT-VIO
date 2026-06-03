"""
This script exploits the VGGT model to transform a set of downsampled unposed 
frames into a calibrated dataset, maintaining original local chaining, refining 
global placement via Umeyama alignment, and normalizing to a [-1, 1] bounding box.
"""

# Import standard modules
import os
import json
import numpy as np
import torch
from tqdm import tqdm

# Custom modules
from visual import plot_trajectory

# Import VGGT modules
from vggt.models.vggt import VGGT  # type: ignore
from vggt.utils.load_fn import load_and_preprocess_images  # type: ignore
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore

# Import parameters
from cfg import params


def rotation_error(T_gt: np.ndarray, T_est: np.ndarray) -> float:
    """
    Computes the geodesic rotation error between two transformation matrices.
    """
    R_err = T_gt[:3, :3] @ T_est[:3, :3].T
    val = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
    return float(np.arccos(val))


def umeyama_alignment(
        X: np.ndarray, 
        Y: np.ndarray, 
        with_scale: bool = True
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Computes the optimal rigid similarity transformation (Rotation, Translation, Scale)
    that maps point set X onto point set Y: Y = s * R * X + t.
    """
    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    
    X_c = X - mu_x
    Y_c = Y - mu_y
    
    N = X.shape[0]
    C = (Y_c.T @ X_c) / N
    
    U, S, Vt = np.linalg.svd(C)
    
    S_mtx = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S_mtx[2, 2] = -1
        
    R = U @ S_mtx @ Vt
    
    if with_scale:
        var_x = np.var(X_c, axis=0).sum()
        s = float(np.trace(np.diag(S) @ S_mtx) / (var_x + 1e-10))
    else:
        s = 1.0
        
    t = mu_y - s * R @ mu_x
    
    return R, t, s


def main(
        device: str,
        model_name: str,
        src_folder: str,
        dst_folder: str,
        batch_size: int,
        do_plot: bool,
        plot_gt: bool,
        offset: float, 
        sampling_deg: float,  # NUOVO PARAMETRO: Caricato dal dizionario params
) -> None:
    """
    Main calibration pipeline. Downsamples the incoming image sequence based on the
    desired degrees per frame, runs VGGT inference, aligns via Umeyama, and normalizes.
    """
    # -----------------------------------------------------------------------------------
    # MODEL SETUP & DATASET LOADING WITH DOWNSAMPLING
    # -----------------------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    model = VGGT.from_pretrained(model_name).to(device)

    image_folder = os.path.join(src_folder, "masked_rgb/")
    all_image_names = sorted(
        os.listdir(image_folder),
        key=lambda x: int(os.path.splitext(x)[0])
    )
    
    # Calcolo dello stride: quante immagini corrispondono ai gradi richiesti.
    # Assumendo un'orbita completa a 360 gradi.
    total_frames = len(all_image_names)
    deg_per_frame = 360.0 / total_frames
    stride = int(round(sampling_deg / deg_per_frame))
    stride = max(1, stride)  # Evita passi pari a zero
    
    # Applichiamo il downsampling tramite slicing di liste nativo
    image_names = all_image_names[::stride]
    print(f"[INFO] Dataset downsampled using stride {stride} (requested {sampling_deg}°/frame).")
    print(f"[INFO] Processing {len(image_names)} frames out of {total_frames} total available.")

    # -----------------------------------------------------------------------------------
    # GROUND TRUTH LOADING
    # -----------------------------------------------------------------------------------
    estimated_poses = dict()
    gt_poses = dict()

    with open(os.path.join(src_folder, "transforms.json"), "r") as f:
        gt_ds = json.load(f)
        frames_objs = gt_ds["frames"]

    for frame in frames_objs:
        base_name = os.path.basename(frame["masked_rgb_path"])
        t_mtx = np.array(frame["transform_matrix"]).reshape(4, 4)
        gt_poses[base_name] = t_mtx

    # -----------------------------------------------------------------------------------
    # VGGT INFERENCE & ORIGINAL LOCAL CHAINING
    # -----------------------------------------------------------------------------------
    step = batch_size - 1
    last_frame_name = None
    raw_estimated_poses = dict()

    for start in tqdm(range(0, len(image_names), step), desc="Inference"):
        end = min(start + batch_size, len(image_names))
        batch_files = image_names[start:end]

        images_path = [os.path.join(image_folder, f) for f in batch_files]
        images = load_and_preprocess_images(images_path).to(device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype):
                outputs = model(images)
                extrinsic, _ = pose_encoding_to_extri_intri(
                    outputs["pose_enc"],
                    images.shape[-2:]
                )
                batch_poses = extrinsic[0].cpu().numpy()

                local_T = {}
                for i, pose in enumerate(batch_poses):
                    T = np.eye(4)
                    T[:3, :3] = pose[:, :3]
                    T[:3, 3] = pose[:, 3] + np.array(offset)
                    local_T[batch_files[i]] = T

                if last_frame_name is None:
                    for name in batch_files:
                        raw_estimated_poses[name] = local_T[name]
                    last_frame_name = batch_files[-1]
                else:
                    prev_name = last_frame_name
                    T_prev_global = raw_estimated_poses[prev_name]
                    T_prev_local = local_T[prev_name]
                    
                    T_delta = T_prev_global @ np.linalg.inv(T_prev_local)

                    for name in batch_files:
                        if name in raw_estimated_poses:
                            continue
                        raw_estimated_poses[name] = T_delta @ local_T[name]

                    last_frame_name = batch_files[-1]

    # -----------------------------------------------------------------------------------
    # GLOBAL SIMILARITY ALIGNMENT (UMEYAMA)
    # -----------------------------------------------------------------------------------
    common_keys = [k for k in raw_estimated_poses.keys() if k in gt_poses]
    
    if len(common_keys) > 3:
        X_centers = np.array([raw_estimated_poses[k][:3, 3] for k in common_keys])
        Y_centers = np.array([gt_poses[k][:3, 3] for k in common_keys])
        
        R_align, t_align, s_align = umeyama_alignment(X_centers, Y_centers, with_scale=True)
        
        for key, T_local in raw_estimated_poses.items():
            T_corrected = np.eye(4)
            T_corrected[:3, :3] = R_align @ T_local[:3, :3]
            T_corrected[:3, 3] = s_align * (R_align @ T_local[:3, 3]) + t_align
            estimated_poses[key] = T_corrected
    else:
        print("\n[WARNING] Not enough common poses found. Skipping Umeyama Alignment.")
        estimated_poses = raw_estimated_poses

    # -----------------------------------------------------------------------------------
    # SCENE NORMALIZATION (CENTER TO 0, SCALE TO [-1, 1])
    # -----------------------------------------------------------------------------------
    centers = np.array([T[:3, 3] for T in estimated_poses.values()])
    scene_center = centers.mean(axis=0)
    
    centered_centers = centers - scene_center
    max_distance = np.max(np.linalg.norm(centered_centers, axis=1))
    
    scale_factor = 1.0 / (max_distance + 1e-10)
    
    for key in estimated_poses.keys():
        estimated_poses[key][:3, 3] -= scene_center
        estimated_poses[key][:3, 3] *= scale_factor

    # -----------------------------------------------------------------------------------
    # TRAJECTORY VISUALIZATION
    # -----------------------------------------------------------------------------------
    if do_plot:
        plot_trajectory(
            estimated_poses=estimated_poses,
            gt_poses=gt_poses,
            plot_gt=plot_gt
        )

    # -----------------------------------------------------------------------------------
    # EXPORT DATASET
    # -----------------------------------------------------------------------------------
    new_ds = {
        "camera_angle_x": gt_ds["camera_angle_x"],
        "resolution": gt_ds["resolution"],
        "num_ch": gt_ds["num_ch"],
        "frames": []
    }

    for key, val in estimated_poses.items():
        frame = {
            "file_path": f"rgb/{key}",
            "depth_path": f"depth/{key.split('.')[0]}_d.png",
            "mask_path": f"mask/{key.split('.')[0]}_m.png",
            "marked_rgb_path": f"masked_rgb/{key}",
            "transform_matrix": val.tolist()
        }
        new_ds["frames"].append(frame)

    output_path = os.path.join(dst_folder, "vggt_transforms.json")
    with open(output_path, "w") as fopen:
        json.dump(new_ds, fopen, indent=4)


if __name__ == "__main__":
    main(**params)