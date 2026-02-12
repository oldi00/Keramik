"""Provide utility functions for image operations, directory creation and config file."""

from pathlib import Path
import re
import shutil
import yaml
import cv2
import numpy as np


def normalize_name(text):
    """Normalize the given text."""

    return re.sub(r'[^a-z0-9]', '', text.lower())


def load_image_gray(img_path):
    """Load the image at the given path in grayscale color space."""

    with open(img_path, "rb") as f:
        file_bytes = bytearray(f.read())

    array = np.asarray(file_bytes, dtype=np.uint8)

    img = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)

    return img


def load_config(config_path="config.yaml"):
    """Load a YAML config file based on the given path."""

    with open(config_path, "r", encoding='utf-8') as file:
        config = yaml.safe_load(file)

    # Convert strings to Path objects automatically.
    for key, val in config['paths'].items():
        config['paths'][key] = Path(val)

    return config


def safe_to_delete(dir_path):
    """
    Check if the given directory is safe to delete.

    A directory is considered safe to delete if it is inside the
    project folder and is different from the project root.
    """

    target = Path(dir_path).resolve()
    project_root = Path.cwd().resolve()

    if not target.is_relative_to(project_root):
        raise ValueError(f"Security check failed: Path '{target}' is outside the project scope.")

    if target == project_root:
        raise ValueError("Operation refused: Cannot delete the project root directory.")

    return True


def create_dir(path, force=False):
    """Create an empty directory at the given path."""

    if path.exists() and safe_to_delete(path) and force:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def get_points(img_path, step=5):
    """Return a list of points of black pixels, downsampled for speed."""

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.bitwise_not(img)  # switch black/white pixels

    points = cv2.findNonZero(img)
    points = points.reshape(-1, 2)

    # Choose only every n-th point. This keeps the shape, but
    # reduces the necessary computational resources later.
    points = points[::step]

    return points


def get_dist_map(img_path):
    """Load an image and returns the distance map. Assumes the image is binarized."""

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    dist_map = cv2.distanceTransform(img, cv2.DIST_L2, 5)

    return dist_map
