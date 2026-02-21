"""
Clean and extract geometric profiles from raw shard and typology scans.

Usage:
    Run as a script to process entire directories:
    $ python src/preprocessing.py --only_shards
    $ python src/preprocessing.py --overwrite --debug

    Or import functions to process single images in your app:
    >>> from src.preprocessing import preprocess_shard
    >>> profile = preprocess_shard("path/to/shard.svg")

CLI Arguments:
    -n, --limit N      Process only N images (e.g., -n 5). Useful for testing.
    -f, --force        Force re-processing even if output file exists.
    --only_shards      Process only the shard dataset.
    --only_typology    Process only the typology dataset.
"""

from src.utils import load_config, load_image_gray, create_dir, crop_to_content
from typing import Union, BinaryIO
import logging
import argparse
import cairosvg
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

CONFIG = load_config()

DIR_SHARDS_RAW = CONFIG["paths"]["shards_raw"]
DIR_SHARDS_CLEAN = CONFIG["paths"]["shards_clean"]
DIR_TYPOLOGY_RAW = CONFIG["paths"]["typology_raw"]
DIR_TYPOLOGY_CLEAN = CONFIG["paths"]["typology_clean"]

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s",)
logger = logging.getLogger(__name__)

SourceType = Union[str, Path, BinaryIO]


def remove_artifacts(svg_source: SourceType) -> bytes:
    """Remove artifacts (ID, scale, etc.) from the given shard SVG."""

    tree = ET.parse(svg_source)
    root = tree.getroot()

    namespace = "{http://www.w3.org/2000/svg}"
    tags_to_remove = ['image', 'text', 'rect', f'{namespace}image', f'{namespace}text']

    for parent in tree.iter():
        for child in list(parent):
            if any(tag in child.tag for tag in tags_to_remove):
                parent.remove(child)

    return ET.tostring(root, encoding='utf-8')


def convert_svg2array(svg_bytes: bytes) -> np.ndarray:
    """Convert SVG bytes into a numpy array."""

    png_bytes = cairosvg.svg2png(
        bytestring=svg_bytes,
        write_to=None,
        scale=2.0,
        background_color="white"
    )

    img_array = np.frombuffer(png_bytes, np.uint8)

    return cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)


def get_profile(img: np.ndarray) -> np.ndarray:
    """Extract the outer contour (profile) of the given shard image."""

    _, img_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)

    profile = np.ones_like(img) * 255
    cv2.drawContours(profile, [largest_contour], -1, (0, 0, 0), thickness=1)

    return profile


def get_skeleton(img: np.ndarray) -> np.ndarray:
    """Compute the skeleton of the given image."""

    blur = cv2.GaussianBlur(img, (5, 5), 0)

    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    skeleton = cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    skeleton = cv2.bitwise_not(skeleton)

    return skeleton


def crop_typology(img: np.ndarray) -> np.ndarray:
    """Crop the given typology image to the upper-left corner."""

    h, w = img.shape

    crop = img[:, :int(w * 0.45)]
    crop = crop[:int(h * 0.75), :]

    return crop


def preprocess_shard(source: SourceType) -> np.ndarray:
    """Clean the shard (SVG) and extract its profile."""

    if isinstance(source, (str, Path)):
        if Path(source).suffix != ".svg":
            logger.warning(f"Skipping '{Path(source).name}': Not an SVG file.")
            return None

    clean_svg = remove_artifacts(source)

    img_array = convert_svg2array(clean_svg)

    profile = get_profile(img_array)
    profile = crop_to_content(profile)

    return profile


def preprocess_typology(typology_path: str) -> np.ndarray:
    """Extract skeleton and crop the typology image."""

    path_obj = Path(typology_path)

    if path_obj.suffix not in [".png", ".jpg", ".jpeg", ".svg"]:
        logger.warning(f"Skipping '{path_obj.name}': Not an image file.")
        return None

    if path_obj.suffix == ".svg":
        clean_svg = remove_artifacts(typology_path)
        img_array = convert_svg2array(clean_svg)
    else:
        img_array = load_image_gray(typology_path)

    if np.all(img_array == 0) or np.all(img_array == 255):
        logger.warning(f"Skipping '{path_obj.name}': Image is pure white/black.")
        return None

    skeleton = get_skeleton(img_array)
    skeleton = crop_to_content(skeleton)

    return skeleton


def preprocess_batch(batch_type: str, limit: int = None, force: bool = False) -> None:
    """
    Process a batch of images (either shards or typologies).

    Skips images that already exist in the output directory unless 'force' is True.

    Args:
        batch_type (str): Must be 'shard' or 'typology'.
        limit (int): If set, stops processing after N images (useful for testing).
        force (bool): If True, re-processes images even if they already exist.
    """

    if batch_type == "shard":
        input_dir = DIR_SHARDS_RAW
        output_dir = DIR_SHARDS_CLEAN
        process_func = preprocess_shard
        valid_suffixes = {'.svg'}
    elif batch_type == "typology":
        input_dir = DIR_TYPOLOGY_RAW
        output_dir = DIR_TYPOLOGY_CLEAN
        process_func = preprocess_typology
        valid_suffixes = {'.png', '.jpg', '.jpeg', '.svg'}
    else:
        raise ValueError(f"Unknown batch_type: {batch_type}")

    create_dir(output_dir, force)

    files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in valid_suffixes]
    files_to_process = files[:limit] if limit else files

    descr = f"Preprocessing {batch_type.title()} Batch"
    with logging_redirect_tqdm():
        for file_path in tqdm(files_to_process, unit="img", desc=descr):

            save_name = f"{file_path.stem}.png".replace("recons_", "")
            save_path = output_dir / save_name

            if save_path.exists() and not force:
                continue

            try:
                result_img = process_func(str(file_path))
                if result_img is not None:
                    cv2.imwrite(str(save_path), result_img)
            except Exception:
                logger.warning(f"Skipping {file_path.name}: Preprocessing failed.")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--limit", type=int)
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("--only_shards", action="store_true")
    parser.add_argument("--only_typology", action="store_true")

    args = parser.parse_args()

    if not args.only_shards and not args.only_typology:
        process_shards = True
        process_typology = True
    else:
        process_shards = args.only_shards
        process_typology = args.only_typology

    if process_shards:
        preprocess_batch("shard", args.limit, args.force)

    if process_typology:
        preprocess_batch("typology", args.limit, args.force)


if __name__ == "__main__":
    main()
