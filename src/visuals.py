from matplotlib import pyplot as plt
import io
import numpy as np
import cv2


def get_match_overlay(typ_path, dist_map, points):

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

    img = cv2.imread(typ_path, cv2.IMREAD_GRAYSCALE)
    ax.imshow(img, cmap="gray", alpha=0.6)

    h, w = dist_map.shape

    points_int = np.rint(points).astype(int)
    dists = []
    for p in points_int:
        x, y = p
        val = dist_map[y, x] if 0 <= x < w and 0 <= y < h else 50
        dists.append(val)

    ax.scatter(
        points[:, 0], points[:, 1],
        c=dists, cmap='RdYlGn_r',
        edgecolors='black',
        linewidths=0.3,
        vmin=0, vmax=30,
        s=15,
        alpha=0.9,
    )

    ax.axis('off')
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    return buf
