import numpy as np
import matplotlib.pyplot as plt

from solver import find_top_matches
from utils import load_image_gray, get_points, apply_transformation, get_dist_map
from icp import icp
from visuals import get_match_overlay_fig


def params_to_matrix(scale, rotation, translation):
    """
    Helper: Converts Solver's (scale, rotation, translation) tuple
    into a 3x3 Homogeneous Transformation Matrix for ICP.
    """
    c, s = np.cos(rotation), np.sin(rotation)
    tx, ty = translation

    # Construct standard similarity matrix
    # [ s*cos  -s*sin   tx ]
    # [ s*sin   s*cos   ty ]
    # [   0       0      1 ]
    T = np.eye(3)
    T[0, 0] = scale * c
    T[0, 1] = -scale * s
    T[0, 2] = tx
    T[1, 0] = scale * s
    T[1, 1] = scale * c
    T[1, 2] = ty

    return T


def matrix_to_params(T):
    """
    Helper: Decomposes a 3x3 Matrix back into (scale, rotation, translation)
    so we can use the existing visualize/apply_transformation functions.
    """
    # 1. Translation is the last column
    tx = T[0, 2]
    ty = T[1, 2]

    # 2. Scale is the magnitude of the first column vector
    # col0 = [s*cos, s*sin]
    sx = np.sqrt(T[0, 0] ** 2 + T[1, 0] ** 2)

    # 3. Rotation is atan2 of the rotation components
    # The scale cancels out in atan2
    rotation = np.arctan2(T[1, 0], T[0, 0])

    return sx, rotation, (tx, ty)


def main():
    shard_path = "data/processed/shards/10004.png"
    shard_img = load_image_gray(shard_path)
    top_matches = find_top_matches(shard_img)
    points_shard = get_points(shard_img)

    for i, match in enumerate(top_matches):
        params = match["params"]
        typ_path = match["path"]
        typology_img = load_image_gray(match["path"])
        points_typology = get_points(typology_img)
        dist_map = get_dist_map(typology_img)
        transformed_shard_ransac = apply_transformation(points_shard, *match["params"])

        break

    print("\n--- Phase 2: Running ICP (Fine Optimization) ---")

    # A. Bridge: Convert RANSAC params to Matrix
    init_pose = params_to_matrix(*params)
    final_T, distances, iterations = icp(
        source_points=points_shard,
        target_points=points_typology,
        init_pose=init_pose,
        max_iterations=100,
        tolerance=1e-6,
    )

    # C. Analyze Result
    mean_icp_error = np.mean(distances)  # This is point-to-point error, not Chamfer
    print(f"ICP converged in {iterations} iterations.")
    print(f"ICP Mean Euclidian Error: {mean_icp_error:.4f}")

    # D. Bridge: Convert Matrix back to Params for visualization
    final_scale, final_rot, final_trans = matrix_to_params(final_T)

    print(
        f"Final Params: Scale={final_scale:.2f}, Rot={np.degrees(final_rot):.2f}°, Trans={final_trans}"
    )

    transformed_shard_icp = apply_transformation(
        points_shard, final_scale, final_rot, final_trans
    )

    get_match_overlay_fig(typ_path, dist_map, transformed_shard_ransac)
    plt.show()

    get_match_overlay_fig(typ_path, dist_map, transformed_shard_icp)
    plt.show()


if __name__ == "__main__":
    main()
