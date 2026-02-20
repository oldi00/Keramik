"""Provide utility functions for image operations, directory creation and config file."""

from pathlib import Path
import re
import shutil
import yaml
import cv2
import numpy as np


def normalize_name(text):
    """Normalize the given text."""

    return re.sub(r"[^a-z0-9]", "", text.lower())


def load_image_gray(img_path):
    """Load the image at the given path in grayscale color space."""

    with open(img_path, "rb") as f:
        file_bytes = bytearray(f.read())

    array = np.asarray(file_bytes, dtype=np.uint8)

    img = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)

    return img


def load_config(config_path="config.yaml"):
    """Load a YAML config file based on the given path."""

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Convert strings to Path objects automatically.
    for key, val in config["paths"].items():
        config["paths"][key] = Path(val)

    return config


def safe_to_delete(dir_path: str) -> bool:
    """
    Check if the given directory is safe to delete.

    A directory is considered safe to delete if it is inside the
    project folder and is different from the project root.
    """

    target = Path(dir_path).resolve()
    project_root = Path.cwd().resolve()

    if not target.is_relative_to(project_root):
        raise ValueError(
            f"Security check failed: Path '{target}' is outside the project scope."
        )

    if target == project_root:
        raise ValueError("Operation refused: Cannot delete the project root directory.")

    return True


def create_dir(path: Path, force: bool = False) -> None:
    """Create an empty directory at the given path."""

    if path.exists() and safe_to_delete(path) and force:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def get_points(img: np.ndarray, step=5) -> np.ndarray:
    """
    Extract points from black pixels in the image, downsampled for speed.

    Args:
        img (np.ndarray): Grayscale image (White background, Black lines).
        step (int): Downsample factor (take every n-th point).

    Returns:
        np.ndarray: List of (x, y) coordinates with shape (N, 2).
    """

    img_inv = cv2.bitwise_not(img)

    points = cv2.findNonZero(img_inv)
    points = points.reshape(-1, 2)

    # Choose only every n-th point. This keeps the shape, but
    # reduces the necessary computational resources later.
    points = points[::step]

    return points


def apply_transformation(points, scale, rotation, translation):
    """Apply a transformation to a (N, 2) array of points."""

    p_x, p_y = points[:, 0], points[:, 1]
    t_x, t_y = translation

    cos, sin = np.cos(rotation), np.sin(rotation)

    # Transform all shard points. Apply scale and rotation
    # first, then translate to the new position.
    x = (scale * (p_x * cos - p_y * sin)) + t_x
    y = (scale * (p_x * sin + p_y * cos)) + t_y

    return np.array([x, y]).T


def get_dist_map(img: np.ndarray) -> np.ndarray:
    """Compute the distance map from an image."""

    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    dist_map = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)

    return dist_map


def crop_to_content(img, padding=8):
    """Crop an image to its non-white content."""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img

    x, y, w, h = cv2.boundingRect(coords)

    img_h, img_w = img.shape[:2]
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(img_w, x + w + padding)
    y_end = min(img_h, y + h + padding)

    return img[y_start:y_end, x_start:x_end]
