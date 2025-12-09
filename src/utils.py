"""A collection of utility functions."""

import shutil
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
