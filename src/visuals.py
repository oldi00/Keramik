"""..."""

from matplotlib import pyplot as plt
import numpy as np
import cv2


def get_match_overlay_fig(typ_path, dist_map, shard_points):
    """Generates the plot and returns the figure object for Streamlit."""

    # 1. Initialize Figure and Axis explicitly
    fig, ax = plt.subplots()

    img = cv2.imread(typ_path, cv2.IMREAD_GRAYSCALE)
    ax.imshow(img, cmap="gray", alpha=0.5)

    points_int = np.rint(shard_points).astype(int)
    h, w = dist_map.shape

    dists = []
    for p in points_int:
        x, y = p
        val = dist_map[y, x] if 0 <= x < w and 0 <= y < h else 50
        dists.append(val)

    # 2. Use ax.scatter instead of plt.scatter
    ax.scatter(
        shard_points[:, 0], shard_points[:, 1],
        c=dists, cmap='RdYlGn_r',
        edgecolors='none',
        vmin=0, vmax=30,
        s=3
    )

    ax.axis('off')
    fig.tight_layout()

    return fig
