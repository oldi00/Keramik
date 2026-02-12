"""..."""

from utils import get_points, get_dist_map, load_config, load_image_gray
from ransac import find_coarse_match
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import pickle
import logging
import numpy as np
from tqdm import tqdm

CONFIG = load_config()

TYPOLOGY_DIR = CONFIG["paths"]["typology_clean"]
CACHE_FILE = CONFIG["paths"]["typology_cache"]

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s",)
logger = logging.getLogger(__name__)


def build_typology_cache() -> None:
    """Calculate points and distance map for all typology files and store them."""

    Path(CACHE_FILE).parent.mkdir(parents=True, exist_ok=True)

    typology_files = list(Path(TYPOLOGY_DIR).glob("*.png"))

    if not typology_files:
        logger.warning(f"No PNG files found in {TYPOLOGY_DIR}. Cache will be empty!")
        return []

    cache = []
    for path in typology_files:
        img = load_image_gray(path)
        cache.append({
            "name": path.stem,
            "path": str(path),
            "points": get_points(img),
            "dist_map": get_dist_map(img)
        })

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)

    logger.info(f"Successfully saved {len(cache)} entries to {CACHE_FILE}")
    return cache


def load_typology_data() -> None:
    """
    Load typology cache or build cache if path is invalid, cache could
    not be loaded or cache is outdated.
    """

    if not Path(CACHE_FILE).exists():
        logger.info("Cache file not found. Triggering build...")
        return build_typology_cache()

    try:
        with open(CACHE_FILE, "rb") as f:
            cached_data = pickle.load(f)
    except (EOFError, pickle.UnpicklingError):
        logger.warning("Cache file is corrupted or empty. Rebuilding...")
        return build_typology_cache()

    current_files = {p.name for p in Path(TYPOLOGY_DIR).glob("*.png")}
    cached_files = {Path(item["path"]).name for item in cached_data}

    if current_files != cached_files:
        diff_count = len(current_files.symmetric_difference(cached_files))
        logger.info(f"Cache is stale ({diff_count} file difference detected). Rebuilding...")
        return build_typology_cache()

    logger.info(f"Cache is valid. Loaded {len(cached_data)} items.")
    return cached_data


def match_single_entry(typology_entry: dict, points_shard: np.ndarray) -> dict:
    """Match a single shard against a single typology on a separate CPU core."""

    score, params = find_coarse_match(
        points_shard,
        typology_entry["points"],
        typology_entry["dist_map"]
    )

    return {
        "name": typology_entry["name"],
        "path": typology_entry["path"],
        "score": score,
        "params": params,
    }


def find_top_matches(shard_img: np.ndarray, top_k: int = 3):
    """..."""

    points_shard = get_points(shard_img)
    typology_data = load_typology_data()

    num_cores = cpu_count()
    worker_func = partial(match_single_entry, points_shard=points_shard)

    with Pool(processes=num_cores) as pool:
        candidates = list(tqdm(
            pool.imap(worker_func, typology_data),
            total=len(typology_data),
            unit="match"
        ))

    candidates.sort(key=lambda x: x["score"])

    # todo: integrate ICP with the top 10 candidates
    top_matches = candidates[:top_k]

    return top_matches


if __name__ == "__main__":

    shard_path = "data/processed/shards/recons_10004.png"
    shard_img = load_image_gray(shard_path)

    top_matches = find_top_matches(shard_img)

    print(top_matches)
