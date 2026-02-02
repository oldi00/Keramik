"""A collection of utility functions."""

import re
import shutil
import cv2
import yaml
from pathlib import Path


def normalize_name(text):
    """Returns the normalized form of the given text."""

    return re.sub(r'[^a-z0-9]', '', text.lower())


def load_config(config_path="config.yaml"):
    """Load a config file based on the given path."""

    with open(config_path, "r", encoding='utf-8') as file:
        config = yaml.safe_load(file)

    # Convert strings to Path objects automatically.
    for key, val in config['paths'].items():
        config['paths'][key] = Path(val)

    return config


def safe_to_delete(dir_path):
    """Checks if the given directory is safe to delete."""

    target = Path(dir_path).resolve()
    project_root = Path.cwd().resolve()

    if not target.is_relative_to(project_root):
        raise ValueError(f"Security check failed: Path '{target}' is outside the project scope.")

    if target == project_root:
        raise ValueError("Operation refused: Cannot delete the project root directory.")

    return True


def create_dir(path, override=False):
    """Creates a directory at the given path."""

    if path.exists() and safe_to_delete(path) and override:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def get_points(img_path, step=5):
    """Returns a list of points of black pixels, downsampled for speed."""

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.bitwise_not(img)  # switch black/white pixels

    points = cv2.findNonZero(img)
    points = points.reshape(-1, 2)

    # Choose only every n-th point. This keeps the shape, but
    # reduces the necessary computational resources later.
    points = points[::step]

    return points


def get_dist_map(img_path):
    """Loads an image and returns the distance map. Assumes the image is binarized."""

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    dist_map = cv2.distanceTransform(img, cv2.DIST_L2, 5)

    return dist_map
