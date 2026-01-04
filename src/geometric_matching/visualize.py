"""A collection of visualization functions."""

from preprocess import DIR_SHARDS_CLEAN_PNG, DIR_SHARDS_PROFILES, \
    DIR_TYP_SKELETONS, DIR_TYP_CROPS
from utils import get_dist_map, get_points
from solver import get_transformation, apply_transformation, get_chamfer_score
import io
import cv2
import imageio.v2 as imageio
import numpy as np
from matplotlib import pyplot as plt


def show_img(title, img, pos):
    """Reusable code to plot an image."""

    plt.subplot(*pos)
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.xticks([])
    plt.yticks([])


def show_preprocess_shards(shard_id):
    """Plots the preprocessing pipeline for shards."""

    img_raw = cv2.imread(f"data/raw/png/{shard_id}.png", cv2.IMREAD_GRAYSCALE)
    show_img("Raw", img_raw, (1, 3, 1))

    img_clean = cv2.imread(DIR_SHARDS_CLEAN_PNG / f"{shard_id}.png", cv2.IMREAD_GRAYSCALE)
    show_img("Clean", img_clean, (1, 3, 2))

    img_profile = cv2.imread(DIR_SHARDS_PROFILES / f"{shard_id}.png", cv2.IMREAD_GRAYSCALE)
    show_img("Profile", img_profile, (1, 3, 3))

    plt.suptitle("Preprocessing Pipeline for Shards")
    plt.tight_layout()
    plt.show()


def show_preprocess_typology(typ_name):
    """Plots the preprocessing pipeline for typology images."""

    img_raw = cv2.imread(f"data/typology/{typ_name}.png", cv2.IMREAD_GRAYSCALE)
    show_img("Raw", img_raw, (1, 3, 1))

    img_skeleton = cv2.imread(DIR_TYP_SKELETONS / f"{typ_name}.png", cv2.IMREAD_GRAYSCALE)
    show_img("Skeleton", img_skeleton, (1, 3, 2))

    img_crop = cv2.imread(DIR_TYP_CROPS / f"{typ_name}.png", cv2.IMREAD_GRAYSCALE)
    show_img("Crop", img_crop, (1, 3, 3))

    plt.suptitle("Preprocessing Pipeline for Typology Images")
    plt.tight_layout()
    plt.show()


def show_dist_map(img_path):
    """Visualize the distance map of the given image."""

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    show_img("Original", img, (1, 2, 1))

    dist_map = get_dist_map(img_path)
    show_img("Distance Map", dist_map, (1, 2, 2))

    plt.tight_layout()
    plt.show()


def save_heatmap_overlay(typ_path, dist_map, points, out_path):
    """Overlay shard points on the typology colored by distance and save to given path."""

    fig, ax = plt.subplots(figsize=(10, 10))

    img_typ = cv2.imread(typ_path, cv2.IMREAD_GRAYSCALE)
    img_typ = cv2.erode(img_typ, np.ones((3, 3), np.uint8))
    ax.imshow(img_typ, cmap='gray', origin='upper', alpha=0.5)

    points_int = np.rint(points).astype(int)
    h, w = dist_map.shape

    dists = []
    for p in points_int:
        x, y = p
        val = dist_map[y, x] if 0 <= x < w and 0 <= y < h else 50
        dists.append(val)

    ax.scatter(
        points[:, 0], points[:, 1],
        c=dists, cmap='RdYlGn_r',
        edgecolors='none',
        vmin=0, vmax=30,
    )

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_ransac_gif(path_typology, path_shard, out_gif_path, n_frames=50):
    """Generate and store a GIF that visualizes the RANSAC process."""

    points_typ = get_points(str(path_typology))
    points_shard = get_points(str(path_shard))
    dist_map = get_dist_map(str(path_typology))

    h, w = dist_map.shape
    frames = []
    best_score = np.inf

    count = 0
    max_attempts = n_frames * 100
    attempts = 0

    while count < n_frames and attempts < max_attempts:

        attempts += 1

        idx_s = np.random.choice(len(points_shard), 2, replace=False)
        idx_t = np.random.choice(len(points_typ), 2, replace=False)
        p1, p2 = points_shard[idx_s]
        q1, q2 = points_typ[idx_t]

        if np.array_equal(p1, p2) or np.array_equal(q1, q2):
            continue

        len_s = np.linalg.norm(p1 - p2)
        len_t = np.linalg.norm(q1 - q2)
        if len_s == 0:
            continue

        est_scale = len_t / len_s
        if not (0.5 <= est_scale <= 1.5):
            continue

        scale, rotation, translation = get_transformation(p1, p2, q1, q2)
        if abs(rotation) > np.deg2rad(20):
            continue

        t_points = apply_transformation(points_shard, scale, rotation, translation)
        score = get_chamfer_score(t_points, dist_map)

        is_best = score < best_score
        if is_best:
            best_score = score

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(dist_map, cmap='gray_r', origin='upper')

        if is_best:
            color = '#00ff00'
            alpha = 1.0
            marker_size = 25
        else:
            color = '#ff0040'
            alpha = 0.5
            marker_size = 10

        scatter_config = {
            "c": color,
            "s": marker_size,
            "alpha": alpha,
            "edgecolors": 'none'
        }

        ax.scatter(t_points[:, 0], t_points[:, 1], **scatter_config)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=80)
        buf.seek(0)

        img = imageio.imread(buf)
        frames.append(img)

        plt.close(fig)
        buf.close()

        count += 1

    imageio.mimsave(str(out_gif_path), frames, fps=3, loop=0)


if __name__ == "__main__":

    # show_preprocess_shards("10001")
    # show_preprocess_typology("Drag33")
    # show_dist_map("data/preprocess/typology_crops/Drag33.png")

    gif_config = {
        "path_typology": "data/preprocess/typology_crops/drag33.png",
        "path_shard": "data/preprocess/shards_profiles/10004.png",  # Beispiel ID
        "out_gif_path": "data/results/visualizations/ransac.gif",
        "n_frames": 20
    }
    create_ransac_gif(**gif_config)
