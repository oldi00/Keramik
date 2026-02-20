"""..."""

from utils import get_points, get_dist_map, load_config, load_image_gray, normalize_name
from ransac import find_coarse_match
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import pickle
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s",)
logger = logging.getLogger(__name__)


def build_typology_cache(typology_dir: str, cache_file: str) -> None:
    """Calculate points and distance map for all typology files and store them."""

    Path(cache_file).parent.mkdir(parents=True, exist_ok=True)

    typology_files = list(Path(typology_dir).glob("*.png"))

    if not typology_files:
        logger.warning(f"No PNG files found in {typology_dir}. Cache will be empty!")
        return []

    cache = {}
    for path in typology_files:

        img = load_image_gray(path)

        name_normalized = normalize_name(path.stem)
        cache[name_normalized] = {
            "name": name_normalized,
            "path": str(path),
            "points": get_points(img),
            "dist_map": get_dist_map(img)
        }

    with open(cache_file, "wb") as f:
        pickle.dump(cache, f)

    logger.info(f"Successfully saved {len(cache)} entries to {cache_file}")
    return cache


def load_typology_data(config: dict) -> None:
    """
    Load typology cache or build cache if path is invalid, cache could
    not be loaded or cache is outdated.
    """

    typology_dir = config["paths"]["typology_clean"]
    cache_file = config["paths"]["typology_cache"]

    if not Path(cache_file).exists():
        logger.info("Cache file not found. Triggering build...")
        return build_typology_cache(typology_dir, cache_file)

    try:
        with open(cache_file, "rb") as f:
            cached_data = pickle.load(f)
    except (EOFError, pickle.UnpicklingError):
        logger.warning("Cache file is corrupted or empty. Rebuilding...")
        return build_typology_cache(typology_dir, cache_file)

    current_files = {p.name for p in Path(typology_dir).glob("*.png")}
    cached_files = {Path(item["path"]).name for item in cached_data.values()}

    if current_files != cached_files:
        diff_count = len(current_files.symmetric_difference(cached_files))
        logger.info(f"Cache is stale ({diff_count} file difference detected). Rebuilding...")
        return build_typology_cache(typology_dir, cache_file)

    logger.info(f"Cache is valid. Loaded {len(cached_data)} items.")
    return cached_data


def match_single_entry(typology_entry: dict, points_shard: np.ndarray, config: dict) -> dict:
    """Match a single shard against a single typology on a separate CPU core."""

    ransac_params = config["parameters"]["ransac"]

    score, params = find_coarse_match(
        points_shard,
        typology_entry["points"],
        typology_entry["dist_map"],
        config=ransac_params
    )

    return {
        "name": normalize_name(typology_entry["name"]),
        "path": typology_entry["path"],
        "score": score,
        "params": params,
    }


def find_top_matches(shard_img: np.ndarray, typology_data=None, config: dict = None):
    """..."""

    if config is None:
        config = load_config()

    if not typology_data:
        typology_data = load_typology_data(config)

    points_shard = get_points(shard_img)

    num_cores = cpu_count()
    worker_func = partial(match_single_entry, points_shard=points_shard, config=config)

    with Pool(processes=num_cores) as pool:
        candidates = list(pool.imap(worker_func, typology_data.values()))

    candidates.sort(key=lambda x: x["score"])

    # todo: integrate ICP with the top 10 candidates

    top_k = config.get("parameters", {}).get("top_k", 10)
    top_matches = candidates[:top_k]

    return top_matches
