'''
Functions to visualize Ground Truth vs. Estimated Camera Poses.
'''

# Standard import
import matplotlib.pyplot as plt
import numpy as np


# Function to plot a single frame
def plot_frame(
        ax,
        T,
        color='b',
        scale=0.05
):
    R = T[:3, :3]
    t = T[:3, 3]

    ax.quiver(t[0], t[1], t[2], R[0, 0], R[1, 0], R[2, 0], color=color, length=scale)
    ax.quiver(t[0], t[1], t[2], R[0, 1], R[1, 1], R[2, 1], color=color, length=scale)
    ax.quiver(t[0], t[1], t[2], R[0, 2], R[1, 2], R[2, 2], color=color, length=scale)


# Function to plot the full trajectory
def plot_trajectory(
        estimated_poses: dict,
        gt_poses: dict,
        plot_gt: bool = True
):

    # =========================
    # Plot ESTIMATED
    # =========================
    fig_est = plt.figure()
    ax_est = fig_est.add_subplot(111, projection='3d')

    est_pts = []

    for _, T_est in estimated_poses.items():
        plot_frame(ax_est, T_est, color='blue')
        est_pts.append(T_est[:3, 3])

    est_pts = np.array(est_pts)

    if len(est_pts) > 0:
        ax_est.set_xlim(est_pts[:, 0].min(), est_pts[:, 0].max())
        ax_est.set_ylim(est_pts[:, 1].min(), est_pts[:, 1].max())
        ax_est.set_zlim(est_pts[:, 2].min(), est_pts[:, 2].max())

    ax_est.set_xlabel("X")
    ax_est.set_ylabel("Y")
    ax_est.set_zlabel("Z")
    ax_est.set_box_aspect([1, 1, 1])
    ax_est.set_title("Estimated Trajectory")

    plt.tight_layout()
    plt.axis("equal")
    plt.show()

    # =========================
    # Plot GROUND TRUTH (optional)
    # =========================
    if plot_gt:
        fig_gt = plt.figure()
        ax_gt = fig_gt.add_subplot(111, projection='3d')

        gt_pts = []

        for _, T_gt in gt_poses.items():
            plot_frame(ax_gt, T_gt, color='red')
            gt_pts.append(T_gt[:3, 3])

        gt_pts = np.array(gt_pts)

        if len(gt_pts) > 0:
            ax_gt.set_xlim(gt_pts[:, 0].min(), gt_pts[:, 0].max())
            ax_gt.set_ylim(gt_pts[:, 1].min(), gt_pts[:, 1].max())
            ax_gt.set_zlim(gt_pts[:, 2].min(), gt_pts[:, 2].max())

        ax_gt.set_xlabel("X")
        ax_gt.set_ylabel("Y")
        ax_gt.set_zlabel("Z")
        ax_gt.set_box_aspect([1, 1, 1])
        ax_gt.set_title("Ground Truth Trajectory")

        plt.tight_layout()
        plt.axis("equal")
        plt.show()