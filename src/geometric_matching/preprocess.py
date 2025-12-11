"""Cleans and extracts geometric profiles from raw shard and typology scans."""

from utils import create_dir
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

DIR_SHARDS_CLEAN_SVG = Path("data/preprocess/shards_clean_svg")
DIR_SHARDS_CLEAN_PNG = Path("data/preprocess/shards_clean_png")
DIR_SHARDS_PROFILES = Path("data/preprocess/shards_profiles")
DIR_TYP_SKELETONS = Path("data/preprocess/typology_skeletons")
DIR_TYP_CROPS = Path("data/preprocess/typology_crops")

# todo: return white lines on black background (+ fix get_points in utils.py)


def clean_shard(img_path, output_dir):
    """Removes ID and scale from the raw SVG image."""

    namespace = "{http://www.w3.org/2000/svg}"

    tree = ET.parse(img_path)
    root = tree.getroot()

    # Find and remove ID/scale via specific SVG tags.
    for child in list(root):
        if child.tag in [f"{namespace}text", f"{namespace}rect"]:
            root.remove(child)

    output_path = Path(output_dir) / img_path.name
    tree.write(output_path)


def convert_svg2png(img_path, output_dir):
    """Converts the given SVG file to PNG format."""

    try:
        drawing = svg2rlg(str(img_path))
        out_path = output_dir / f"{img_path.stem}.png"
        renderPM.drawToFile(drawing, out_path, fmt="PNG", dpi=300)
    except Exception as e:
        print(f"\n[SKIP] Could not convert {img_path.name}: {e}")


def extract_profile_shard(img_path, output_dir):
    """Extracts the profile (left side) from the given shard image as a single-line contour."""

    # todo: improve the code
    # todo: add visualization of the steps?

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, img_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)

    thresh_clean = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)

    shard = np.ones_like(img) * 255
    cv2.drawContours(shard, [largest_contour], -1, (0, 0, 0), thickness=1)

    save_path = Path(output_dir) / Path(img_path).name
    cv2.imwrite(str(save_path), shard)


def get_skeleton(img_path, output_dir):
    """Compute the skeleton of the given image."""

    # todo: improve code

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    skeleton = cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    skeleton = cv2.bitwise_not(skeleton)

    out_path = Path(output_dir) / Path(img_path).name
    cv2.imwrite(out_path, skeleton)


def crop_typology(img_path, output_dir):
    """Crop the given typology image to the upper-left corner."""

    # todo: improve code

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    h, w = img.shape

    crop = img[:, :int(w * 0.45)]
    crop = crop[:int(h * 0.75), :]

    out_path = Path(output_dir) / Path(img_path).name
    cv2.imwrite(out_path, crop)


def preprocess_shards(debug=False):
    """Cleans and standardizes raw shards for model training."""

    # Create the necessary folders to store the processed images.
    for dir_path in [DIR_SHARDS_CLEAN_SVG, DIR_SHARDS_CLEAN_PNG, DIR_SHARDS_PROFILES]:
        create_dir(dir_path, override=False)

    # Remove non-relevant artifacts (IDs, scale) from raw SVGs so the model can focus on geometry.
    for svg_path in tqdm(list(Path("data/raw/svg").iterdir()), desc="Clean Shards", unit="img"):
        clean_shard(svg_path, DIR_SHARDS_CLEAN_SVG)
        if debug:
            break

    # Rename the clean shards for easier identification later.
    for file in DIR_SHARDS_CLEAN_SVG.iterdir():
        file.replace(file.with_name(file.name.replace("recons_", "")))
        if debug:
            break

    # Transform clean SVGs to PNG format since it is easier to work with later.
    for svg_path in tqdm(list(DIR_SHARDS_CLEAN_SVG.iterdir()), desc="Convert SVGs", unit="img"):
        if not (DIR_SHARDS_CLEAN_PNG / f"{svg_path.stem}.png").exists():
            convert_svg2png(svg_path, DIR_SHARDS_CLEAN_PNG)
        if debug:
            break

    # Extract the profile (left side) from each shard.
    shard_paths = list(DIR_SHARDS_CLEAN_PNG.iterdir())
    for shard_path in tqdm(shard_paths, desc="Extract Shard Profiles", unit="img"):
        extract_profile_shard(shard_path, DIR_SHARDS_PROFILES)
        if debug:
            break


def preprocess_typology(debug=False):
    """Cleans and standardizes typology snapshots for model training."""

    # Create the necessary folders to store the processed images.
    for dir_path in [DIR_TYP_SKELETONS, DIR_TYP_CROPS]:
        create_dir(dir_path, override=True)

    # Compute the skeleton to get clean, single-pixel geometric paths.
    for typ_path in tqdm(list(Path("data/typology").iterdir()), desc="Get Skeletons", unit="img"):
        get_skeleton(typ_path, DIR_TYP_SKELETONS)
        if debug:
            break

    # Crop the typology images since only the upper-left side is relevant.
    for typ_path in tqdm(list(DIR_TYP_SKELETONS.iterdir()), desc="Crop Typology", unit="img"):
        crop_typology(typ_path, DIR_TYP_CROPS)
        if debug:
            break


if __name__ == "__main__":
    preprocess_shards(debug=False)
    preprocess_typology(debug=False)
