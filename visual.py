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

    ax.quiver(t[0], t[1], t[2], R[0,0], R[1,0], R[2,0], color=color, length=scale)
    ax.quiver(t[0], t[1], t[2], R[0,1], R[1,1], R[2,1], color=color, length=scale)
    ax.quiver(t[0], t[1], t[2], R[0,2], R[1,2], R[2,2], color=color, length=scale)


# Function to plot the full trajectory
def plot_trajectory(
        estimated_poses: dict,
        gt_poses: dict
):
    # Generate the plot window
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Place frames
    all_pts = []
    for name, T_est in estimated_poses.items():

        if name not in gt_poses:
            continue

        T_gt = gt_poses[name]

        plot_frame(ax, T_est, color='blue')
        plot_frame(ax, T_gt, color='red')

        ax.plot(
            [T_est[0,3], T_gt[0,3]],
            [T_est[1,3], T_gt[1,3]],
            [T_est[2,3], T_gt[2,3]],
            'k--', linewidth=0.7
        )

        all_pts.append(T_est[:3,3])
        all_pts.append(T_gt[:3,3])


    all_pts = np.array(all_pts)

    ax.set_xlim(all_pts[:,0].min(), all_pts[:,0].max())
    ax.set_ylim(all_pts[:,1].min(), all_pts[:,1].max())
    ax.set_zlim(all_pts[:,2].min(), all_pts[:,2].max())

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1,1,1])

    plt.title("VGGT Pose Estimation (GT vs Prediction)")
    plt.tight_layout()
    plt.axis("equal")
    plt.show()
