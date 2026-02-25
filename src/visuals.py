import io
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors


def get_match_overlay(typ_path, dist_map, points):
    """Generates a PNG byte buffer overlaying a color-coded distance path on a grayscale image."""
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

    img_array = np.fromfile(typ_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
    ax.imshow(img, cmap="gray", alpha=0.6)

    h, w = dist_map.shape
    x, y = np.rint(points).astype(int).T

    # Fallback to a distance of 50 for out-of-bounds coordinates to prevent indexing crashes.
    valid_mask = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    dists_np = np.full(len(points), 50.0)
    dists_np[valid_mask] = dist_map[y[valid_mask], x[valid_mask]]

    pts = points.reshape(-1, 1, 2)
    segments = np.concatenate([pts[:-1], pts[1:]], axis=1)

    diffs = np.linalg.norm(pts[1:] - pts[:-1], axis=2).flatten()
    mask = diffs < 5  # Adjust '15' based on your average 'step' size

    segments = segments[mask]
    segment_colors = ((dists_np[:-1] + dists_np[1:]) / 2.0)[mask]

    outline = LineCollection(
        segments, colors='black', linewidths=3, alpha=0.8,
        capstyle='round', joinstyle='round'
    )
    ax.add_collection(outline)

    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=5, vmax=17)
    lc = LineCollection(
        segments, cmap='RdYlGn_r', norm=norm, linewidths=2,
        capstyle='round', joinstyle='round'
    )
    lc.set_array(segment_colors)
    # lc.set_clim(0, 80)
    ax.add_collection(lc)

    ax.axis('off')

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    return buf
