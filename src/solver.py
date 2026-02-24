"""..."""

from src.ransac import ransac
from src.icp import icp
import src.utils as utils
from pathlib import Path
import pickle
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s",)
logger = logging.getLogger(__name__)


def build_typology_cache(config: dict) -> None:
    """Calculate points and distance map for all typology files and store them."""

    typology_dir = config["paths"]["typology_clean"]
    cache_file = config["paths"]["typology_cache"]
    squared_dist_map = config["parameters"]["ransac"]["squared_dist_map"]

    Path(cache_file).parent.mkdir(parents=True, exist_ok=True)

    typology_files = list(Path(typology_dir).glob("*.png"))

    if not typology_files:
        logger.warning(f"No PNG files found in {typology_dir}. Cache will be empty!")
        return []

    cache = {}
    for path in typology_files:

        img = utils.load_image_gray(path)

        name_normalized = utils.normalize_name(path.stem)
        cache[name_normalized] = {
            "name": name_normalized,
            "path": str(path),
            "points": utils.get_points(img),
            "dist_map": utils.get_dist_map(img, squared_dist_map)
        }

    cache_payload = {
        "config": config,
        "entries": cache
    }

    with open(cache_file, "wb") as f:
        pickle.dump(cache_payload, f)

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
        return build_typology_cache(config)

    try:
        with open(cache_file, "rb") as f:
            cached_payload = pickle.load(f)
    except (EOFError, pickle.UnpicklingError):
        logger.warning("Cache file is corrupted or empty. Rebuilding...")
        return build_typology_cache(config)

    if "config" not in cached_payload or cached_payload["config"] != config:
        logger.info("Cache parameters changed or format is outdated. Rebuilding...")
        return build_typology_cache(config)

    cached_data = cached_payload["entries"]
    current_files = {p.name for p in Path(typology_dir).glob("*.png")}
    cached_files = {Path(item["path"]).name for item in cached_data.values()}

    if current_files != cached_files:
        diff_count = len(current_files.symmetric_difference(cached_files))
        logger.info(f"Cache is stale ({diff_count} file difference detected). Rebuilding...")
        return build_typology_cache(config)

    logger.info(f"Cache is valid. Loaded {len(cached_data)} items.")
    return cached_data


def find_top_matches(shard_img: np.ndarray, typology_data=None, config: dict = None):
    """Find the top typology matches for the given shard using RANSAC and ICP."""

    if config is None:
        config = utils.load_config()

    ransac_config = config["parameters"]["ransac"]
    icp_config = config["parameters"]["icp"]

    if not typology_data:
        typology_data = load_typology_data(config)

    points_shard = utils.get_points(shard_img)
    if config["parameters"]["remove_shard_base"]:
        points_shard = utils.remove_shard_base(points_shard)

    # --- RANSAC ---

    candidates = []
    for typology_entry in typology_data.values():

        score, params = ransac(
            points_shard,
            typology_entry["points"],
            typology_entry["dist_map"],
            config=ransac_config
        )

        candidates.append({
            "name": utils.normalize_name(typology_entry["name"]),
            "path": typology_entry["path"],
            "score": score,
            "params": params,
        })

    candidates.sort(key=lambda x: x["score"])
    candidates = candidates[:10]  # todo: integrate this parameter into config?

    # --- ICP ---

    top_matches = []
    for candidate in candidates:

        typology_name = candidate["name"]
        ransac_params = candidate["params"]
        points_typology = typology_data[typology_name]["points"]

        init_pose = utils.params_to_matrix(*ransac_params)
        final_T, distances, _ = icp(
            source_points=points_shard,
            target_points=points_typology,
            config=icp_config,
            init_pose=init_pose,
        )

        mean_icp_error = np.mean(distances)
        icp_params = utils.matrix_to_params(final_T)

        top_matches.append({
            "name": typology_name,
            "shard_img": shard_img,
            "ransac_score": candidate["score"],
            "ransac_params": ransac_params,
            "icp_error": mean_icp_error,
            "icp_params": icp_params,
        })

    top_matches.sort(key=lambda x: x["icp_error"])

    top_k = config.get("parameters", {}).get("top_k", 10)
    top_matches = top_matches[:top_k]

    return top_matches
