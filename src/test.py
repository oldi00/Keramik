import numpy as np
from pathlib import Path

# Import your existing modules
from solver import solve_matching, apply_transformation
from utils import get_points, get_dist_map
from visualize import save_heatmap_overlay
from icp import icp  # The new optimization-based ICP


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
    # 1. Setup Paths
    path_shard = "data/processed/shards/10017.png"  # 10004 10011 10014 10017
    path_typology = "data/processed/typology/drag33.png"

    print(f"Loading shard: {path_shard}")
    print(f"Loading typology: {path_typology}")

    # 2. Load Data
    points_shard = get_points(path_shard)
    points_typology = get_points(path_typology)
    dist_map = get_dist_map(path_typology)

    # ---------------------------------------------------------
    # STEP 3: Run RANSAC (The Solver)
    # ---------------------------------------------------------
    print("\n--- Phase 1: Running RANSAC (Coarse Matching) ---")
    ransac_score, ransac_params = solve_matching(
        points_shard, points_typology, dist_map, iterations=20000
    )

    print(f"RANSAC Score (Chamfer): {ransac_score:.4f}")
    print(
        f"RANSAC Params: Scale={ransac_params[0]:.2f}, Rot={np.degrees(ransac_params[1]):.2f}°, Trans={ransac_params[2]}"
    )

    # Visualization 1: RANSAC Result
    transformed_shard_ransac = apply_transformation(points_shard, *ransac_params)
    save_heatmap_overlay(
        path_typology, dist_map, transformed_shard_ransac, "test_result_1_ransac.png"
    )
    print("Saved 'test_result_1_ransac.png'")

    # ---------------------------------------------------------
    # STEP 4: Run ICP (The Optimization)
    # ---------------------------------------------------------
    print("\n--- Phase 2: Running ICP (Fine Optimization) ---")

    # A. Bridge: Convert RANSAC params to Matrix
    init_pose = params_to_matrix(*ransac_params)

    # B. Run ICP using RANSAC result as starting guess
    # The ICP will internally decompose this matrix into (r, theta, size, z)
    # and optimize those specific parameters as per your R logic.
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

    # Visualization 2: Final ICP Result
    # We reuse apply_transformation since we converted the matrix back to compatible params
    transformed_shard_icp = apply_transformation(
        points_shard, final_scale, final_rot, final_trans
    )
    save_heatmap_overlay(
        path_typology, dist_map, transformed_shard_icp, "test_result_2_icp.png"
    )
    print("Saved 'test_result_2_icp.png'")


if __name__ == "__main__":
    main()
