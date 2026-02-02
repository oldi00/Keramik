"""
PoC pipeline for classifying shards by geometrically matching
them against reference typologies using RANSAC and Chamfer distance.
"""

from utils import get_points, get_dist_map, normalize_name, create_dir
from visualize import save_heatmap_overlay
from solver import solve_matching, apply_transformation
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed


EXCEL_PATH = "data/raw/Gesamt_DB_export.xlsx"
TYPOLOGY_FOLDER = "data/preprocess/typology_crops"

TARGET_TYPES = ["drag32", "drag33", "drag1831", "drag37"]


def load_test_set_from_excel():
    """
    Parses the Excel file to build a list of shards with their
    corresponding types, ids and paths.
    """

    df = pd.read_excel(EXCEL_PATH)
    df.columns = df.columns.str.replace('Sample.', '', regex=False)

    test_set = []

    for row in df.itertuples():

        shard_id = str(row.Id)
        shard_type = str(row.Typ)

        path = Path(f"data/preprocess/shards_profiles/{shard_id}.png")

        # Filter: Keep only the specific target types (drag32, etc.) to focus the PoC scope,
        # but verify the image file actually exists to avoid I/O errors later.
        if normalize_name(shard_type) not in TARGET_TYPES or not path.exists():
            continue

        test_set.append({
            "id": shard_id,
            "type": normalize_name(shard_type),
            "path": path
        })

    return test_set


def load_ground_truth_typology():
    """
    Loads point clouds and distance maps for all reference typologies
    to serve as the matching database.
    """

    typology = {}

    # Load all typologies, including those outside TARGET_TYPES, to act as
    # "distractors" and test the model's robustness against false positives.
    for typ_path in Path(TYPOLOGY_FOLDER).rglob("*"):

        points = get_points(str(typ_path))
        dist_map = get_dist_map(str(typ_path))

        typology[normalize_name(typ_path.stem)] = {
            "points": points,
            "dist_map": dist_map,
            "path": str(typ_path)
        }

    return typology


def process_single_match(typ_name, typ_data, points_shard):
    """
    Executes a single matching computation and packages the result
    with the typology name for parallel processing.
    """

    score, params = solve_matching(points_shard, typ_data["points"], typ_data["dist_map"])

    return {"typ_name": typ_name, "score": score, "params": params}


def main():

    test_set = load_test_set_from_excel()
    typology = load_ground_truth_typology()

    out_dir_correct = Path("data/results/correct")
    out_dir_wrong = Path("data/results/wrong")
    create_dir(out_dir_correct, override=True)
    create_dir(out_dir_wrong, override=True)

    top_1, top_3 = 0, 0
    with tqdm(test_set, unit="shard") as bar:

        for idx, shard in enumerate(bar, 1):

            bar.set_description(f"Processing shard with ID={shard['id']}")

            points_shard = get_points(shard["path"])

            # Compare the shard points with all typology references in parallel
            # by using multiple CPU kernels to speed up computation.
            results = Parallel(n_jobs=-1)(
                delayed(process_single_match)(name, data, points_shard)
                for name, data in typology.items()
            )

            # Sort by ascending score because Chamfer Distance measures error.
            # This means 0 is a perfect match.
            results.sort(key=lambda x: x['score'])

            top_3_types = [x["typ_name"] for x in results[:3]]

            is_top1 = shard["type"] == top_3_types[0]
            is_top3 = shard["type"] in top_3_types

            if is_top1:
                top_1 += 1
            if is_top3:
                top_3 += 1

            curr_acc1 = top_1 / idx
            curr_acc3 = top_3 / idx
            bar.set_postfix({"Top-1": f"{curr_acc1:.2%}", "Top-3": f"{curr_acc3:.2%}"})

            points_shard = get_points(shard["path"], step=1)
            out_dir = out_dir_correct if is_top1 else out_dir_wrong

            heatmap_config = {
                "typ_path": typology[results[0]["typ_name"]]["path"],
                "dist_map": typology[results[0]["typ_name"]]["dist_map"],
                "points": apply_transformation(points_shard, *results[0]["params"]),
                "out_path": out_dir / f"{shard["id"]}.png"
            }
            save_heatmap_overlay(**heatmap_config)

    print("-" * 30)
    print(f"Final Top-1 Accuracy: {top_1 / len(test_set):.2%}")
    print(f"Final Top-3 Accuracy: {top_3 / len(test_set):.2%}")


if __name__ == "__main__":
    main()
