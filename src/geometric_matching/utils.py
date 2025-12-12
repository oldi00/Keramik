"""A collection of utility functions."""

import shutil
import cv2
from pathlib import Path


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


def get_points(img_path):
    """Returns a list of points of all black pixels."""

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.bitwise_not(img)  # switch black/white pixels

    points = cv2.findNonZero(img)
    points = points.squeeze()  # removes axes of length=1

    return points


def get_dist_map(img_path):
    """Loads an image and returns the distance map. Assumes the image is binarized."""

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    dist_map = cv2.distanceTransform(img, cv2.DIST_L2, 5)

    return dist_map
