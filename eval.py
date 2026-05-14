'''
Evaluation of results (ATE, RPE)
'''

# Standard import
import numpy as np
import os


# Function to compute ATE and RPE 
def eval_trajectory(
        estimated_poses: dict,
        gt_poses: dict
):
    # Initialize score buffers
    ate = []
    rpe = []

    # Select only the subset of frames employed for the estimation
    keys = sorted(estimated_poses.keys(), key=lambda x: int(os.path.splitext(x)[0]))
    valid = [k for k in keys if k in gt_poses]

    for i, n in enumerate(valid):

        # Compute the RMSE of the path (linear error)
        ate.append(np.linalg.norm(gt_poses[n][:3,3] - estimated_poses[n][:3,3]))

        # Likewise, compute the error on rotations (angular error)
        if i < len(valid) - 1:
            dgt = np.linalg.inv(gt_poses[valid[i]]) @ gt_poses[valid[i+1]]
            dest = np.linalg.inv(estimated_poses[valid[i]]) @ estimated_poses[valid[i+1]]
            rpe.append(np.linalg.norm((dgt @ np.linalg.inv(dest))[:3,3]))

    estimate_ate = np.sqrt(np.mean(np.array(ate)**2))
    estimate_rpe = np.sqrt(np.mean(np.array(rpe)**2))

    print("\nATE RMSE:", estimate_ate, " [%]")
    print("RPE RMSE:", estimate_rpe, "[deg]")

    return estimate_ate, estimate_rpe