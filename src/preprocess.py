"""
Clean and extract geometric profiles from raw shard and typology scans.

Arguments:
  -d, --debug        Run in debug mode (process single images only).
  -o, --overwrite    Force regeneration (delete and recreate output dirs).
  --only_shards      Process only the shard dataset.
  --only_typology    Process only the typology dataset.
"""

from utils import load_config, create_dir
import tempfile
import argparse
import logging
import cv2
import cairosvg
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

CONFIG = load_config()

DIR_SHARDS_RAW = CONFIG["paths"]["shards_raw"]
DIR_SHARDS_CLEAN = CONFIG["paths"]["shards_clean"]
DIR_TYPOLOGY_RAW = CONFIG["paths"]["typology_raw"]
DIR_TYPOLOGY_CLEAN = CONFIG["paths"]["typology_clean"]

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def clean_shard(img_path: Path, output_dir: Path) -> None:
    """Remove ID and scale from the raw SVG image and rename file."""

    namespace = "{http://www.w3.org/2000/svg}"

    tree = ET.parse(img_path)
    root = tree.getroot()

    # Find and remove ID/scale via specific SVG tags.
    for child in list(root):
        if child.tag in [f"{namespace}text", f"{namespace}rect"]:
            root.remove(child)

    clean_name = img_path.name.replace("recons_", "")

    output_path = Path(output_dir) / clean_name
    tree.write(output_path)


def convert_svg2png(img_path: Path, output_dir: Path) -> None:
    """
    Convert the given SVG image to PNG format.
    Follow instruction on CairoSVG website (https://cairosvg.org/) to
    ensure this functions works with no issues.
    """

    out_path = output_dir / f"{img_path.stem}.png"

    try:

        tree = ET.parse(img_path)
        root = tree.getroot()

        for parent in tree.iter():
            for child in list(parent):
                if "image" in child.tag:
                    parent.remove(child)
                    logger.debug(f"Removed embedded image from {img_path.name}")

        clean_svg_string = ET.tostring(root, encoding="utf-8")

        cairosvg.svg2png(
            bytestring=clean_svg_string,
            write_to=str(out_path),
            background_color="white",
            scale=2.0,
        )
        return out_path

    except Exception:
        logger.warning(f"Failed to convert '{img_path.name}' into PNG format.")


def get_profile_shard(img_path: Path):
    """Extract the profile (left side) from the given shard image as a single-line contour."""

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, img_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)

    thresh_clean = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(
        thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    largest_contour = max(contours, key=cv2.contourArea)

    profile = np.ones_like(img) * 255
    cv2.drawContours(profile, [largest_contour], -1, (0, 0, 0), thickness=1)

    return profile


def get_skeleton(img_path, output_dir):
    """Compute the skeleton of the given image."""

    file_bytes = np.fromfile(str(img_path), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    skeleton = cv2.ximgproc.thinning(
        binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
    )

    skeleton = cv2.bitwise_not(skeleton)

    out_path = Path(output_dir) / Path(img_path).name
    cv2.imwrite(out_path, skeleton)


def crop_typology(img_path, output_dir):
    """Crop the given typology image to the upper-left corner."""

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    h, w = img.shape

    crop = img[:, : int(w * 0.45)]
    crop = crop[: int(h * 0.75), :]

    out_path = Path(output_dir) / Path(img_path).name
    cv2.imwrite(out_path, crop)


def preprocess_shards(debug: bool) -> None:
    """
    Standardize raw shards for model training.

    The pipeline follows these steps:
    - Clean the raw SVG images (remove artifacts and rename)
    - Convert cleaned SVG images into PNG format
    - Extract profile from cleaned PNG images
    """

    temp_dir = tempfile.TemporaryDirectory()
    temp_path = Path(temp_dir.name)

    temp_dir_clean_svg = temp_path / "clean_svg"
    temp_dir_clean_svg.mkdir()

    temp_dir_clean_png = temp_path / "clean_png"
    temp_dir_clean_png.mkdir()

    # Remove non-relevant artifacts (IDs, scale) from raw SVGs so the model can focus on geometry.
    for svg_path in tqdm(
        list(DIR_SHARDS_RAW.iterdir()), desc="Clean Shards", unit="img"
    ):
        clean_shard(svg_path, temp_dir_clean_svg)
        if debug:
            break

    # Transform clean SVGs to PNG format since it is easier to work with later.
    for svg_path in tqdm(
        list(temp_dir_clean_svg.iterdir()), desc="Convert SVGs", unit="img"
    ):
        if not (temp_dir_clean_png / f"{svg_path.stem}.png").exists():
            convert_svg2png(svg_path, temp_dir_clean_png)
        if debug:
            break

    # Extract the profile (left side) from each shard.
    shard_paths = list(temp_dir_clean_png.iterdir())
    for shard_path in tqdm(shard_paths, desc="Extract Shard Profiles", unit="img"):

        profile = get_profile_shard(shard_path)

        save_path = Path(DIR_SHARDS_CLEAN) / Path(shard_path).name
        cv2.imwrite(str(save_path), profile)

        if debug:
            break

    temp_dir.cleanup()


def preprocess_typology(debug: bool) -> None:
    """
    Clean and standardize typology snapshots for model training.

    The pipeline follows these steps:
    - Compute the skeletons for the typology images
    - Crop the skeleton images
    """

    temp_dir = tempfile.TemporaryDirectory()
    temp_path = Path(temp_dir.name)

    temp_dir_converted = temp_path / "converted_pngs"
    temp_dir_converted.mkdir()

    temp_dir_skeletons = temp_path / "skeletons"
    temp_dir_skeletons.mkdir()

    # Compute the skeleton to get clean, single-pixel geometric paths.
    typ_paths = [p for p in DIR_TYPOLOGY_RAW.rglob("*") if p.is_file()]
    for typ_path in tqdm(typ_paths, desc="Get Skeletons", unit="img"):
        # ! temporary fix
        umlauts = ["ä", "ö", "ü", "Ä", "Ö", "Ü", "ß"]
        valid_extensions = {".jpg", ".jpeg", ".png", ".svg"}
        if (
            typ_path.name.startswith(".")
            or any(x in typ_path.name for x in umlauts)
            or typ_path.suffix.lower() not in valid_extensions
        ):
            continue

        processing_path = typ_path

        if typ_path.suffix.lower() == ".svg":
            convert_svg2png(typ_path, temp_dir_converted)
            processing_path = temp_dir_converted / f"{typ_path.stem}.png"

        get_skeleton(processing_path, temp_dir_skeletons)
        if debug:
            break

    # Crop the typology images since only the upper-left side is relevant.
    for typ_path in tqdm(
        list(temp_dir_skeletons.iterdir()), desc="Crop Typology", unit="img"
    ):
        crop_typology(typ_path, DIR_TYPOLOGY_CLEAN)
        if debug:
            break

    temp_dir.cleanup()


def get_args():

    parser = argparse.ArgumentParser(
        description="Clean and extract geometric profiles from raw shard and typology scans."
    )

    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Run in debug mode: Process only a single shard and typology image for testing.",
    )

    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Force regeneration: Delete and re-create all output directories from scratch.",
    )

    parser.add_argument(
        "--only_shards", action="store_true", help="Process only the shard dataset."
    )

    parser.add_argument(
        "--only_typology",
        action="store_true",
        help="Process only the typology dataset.",
    )

    return parser.parse_args()


def main(args):

    if args.debug and args.overwrite:
        logger.warning("Debug mode enabled. Disabling overwrite to protect data.")
        args.overwrite = False

    if not args.only_shards and not args.only_typology:
        process_shards = True
        process_typology = True
    else:
        process_shards = args.only_shards
        process_typology = args.only_typology

    if process_shards:
        create_dir(DIR_SHARDS_CLEAN, args.overwrite)
        preprocess_shards(args.debug)

    if process_typology:
        create_dir(DIR_TYPOLOGY_CLEAN, args.overwrite)
        preprocess_typology(args.debug)


if __name__ == "__main__":
    main(get_args())
