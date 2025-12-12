"""
PoC pipeline for classifying shards by geometrically matching
them against reference typologies using RANSAC and Chamfer distance.
"""

from utils import get_points, get_dist_map, normalize_name
from solver import solve_matching
import pandas as pd
from pathlib import Path
from tqdm import tqdm


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


def main():

    test_set = load_test_set_from_excel()
    typology = load_ground_truth_typology()

    top_1, top_3 = 0, 0
    with tqdm(test_set, unit="shard") as bar:

        for idx, shard in enumerate(bar, 1):

            bar.set_description(f"Processing shard with ID={shard['id']}")

            points_shard = get_points(shard["path"])

            results = []
            for typ_name, typ_data in typology.items():
                points_typ, dist_map = typ_data["points"], typ_data["dist_map"]
                score, params = solve_matching(points_shard, points_typ, dist_map)
                results.append({"typ_name": typ_name, "score": score, "params": params})

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

    print("-" * 30)
    print(f"Final Top-1 Accuracy: {top_1 / len(test_set):.2%}")
    print(f"Final Top-3 Accuracy: {top_3 / len(test_set):.2%}")


if __name__ == "__main__":
    main()
