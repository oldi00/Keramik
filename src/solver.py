"""
Implements RANSAC-based geometric alignment followed by ICP fine-tuning
to match shard points onto the larger typology contour.
"""

import numpy as np

# Stelle sicher, dass icp.py im selben Ordner liegt oder im Python-Pfad ist
try:
    from src.geometric_matching.icp import icp
except ImportError:
    from icp import icp

MIN_SCALE = 0.5
MAX_SCALE = 1.5
MAX_ROTATION = np.deg2rad(20)  # +/- 20 Grad Toleranz


def get_transformation(p1, p2, q1, q2):
    """
    Calculates the affine transformation (scale, rotation, translation) needed to
    map two shard points (p1, p2) onto two typology points (q1, q2).
    """
    vec_p = p2 - p1
    vec_q = q2 - q1

    len_s = np.linalg.norm(vec_p)
    len_q = np.linalg.norm(vec_q)

    # Avoid division by zero
    if len_s == 0:
        return 1.0, 0.0, (0.0, 0.0)

    # Compute scale
    scale = len_q / len_s

    angle_p = np.arctan2(vec_p[1], vec_p[0])
    angle_q = np.arctan2(vec_q[1], vec_q[0])

    # Compute rotation
    rotation = angle_q - angle_p

    cos, sin = np.cos(rotation), np.sin(rotation)

    # Compute translation
    t_x = q1[0] - (scale * (p1[0] * cos - p1[1] * sin))
    t_y = q1[1] - (scale * (p1[0] * sin + p1[1] * cos))

    return scale, rotation, (t_x, t_y)


def apply_transformation(points, scale, rotation, translation):
    """Apply a transformation to a (N, 2) array of points."""
    p_x, p_y = points[:, 0], points[:, 1]
    t_x, t_y = translation

    cos, sin = np.cos(rotation), np.sin(rotation)

    # Transform: Scale -> Rotate -> Translate
    x = (scale * (p_x * cos - p_y * sin)) + t_x
    y = (scale * (p_x * sin + p_y * cos)) + t_y

    return np.array([x, y]).T


def get_chamfer_score(transformed_points, dist_map, penalty_factor=1):
    """Computes the Chamfer score with penalties for out-of-bounds points."""
    points_int = np.rint(transformed_points).astype(int)

    p_x = points_int[:, 0]
    p_y = points_int[:, 1]

    h, w = dist_map.shape

    # Identify valid points inside map
    valid_mask = (p_x >= 0) & (p_x < w) & (p_y >= 0) & (p_y < h)

    penalty_value = np.max(dist_map) * penalty_factor
    all_distances = np.full(points_int.shape[0], penalty_value)

    if np.any(valid_mask):
        valid_x = p_x[valid_mask]
        valid_y = p_y[valid_mask]
        all_distances[valid_mask] = dist_map[valid_y, valid_x]

    return np.mean(all_distances)


def params_to_matrix(scale, rotation, translation):
    """Converts (scale, rot, trans) to a 3x3 Homogeneous Transformation Matrix."""
    c, s = np.cos(rotation), np.sin(rotation)
    tx, ty = translation

    # T = [[s*cos, -s*sin, tx],
    #      [s*sin,  s*cos, ty],
    #      [0,      0,      1]]

    T = np.eye(3)
    T[0, 0] = scale * c
    T[0, 1] = -scale * s
    T[0, 2] = tx
    T[1, 0] = scale * s
    T[1, 1] = scale * c
    T[1, 2] = ty

    return T


def matrix_to_params(T):
    """Decomposes a 3x3 Matrix back into (scale, rotation, translation)."""
    # Translation
    tx = T[0, 2]
    ty = T[1, 2]

    # Scale (Magnitude of the first column vector)
    # col0 = [s*cos, s*sin]
    sx = np.sqrt(T[0, 0] ** 2 + T[1, 0] ** 2)

    # Rotation (arctan of the rotation components)
    # The scale cancels out in atan2
    rotation = np.arctan2(T[1, 0], T[0, 0])

    return sx, rotation, (tx, ty)


def solve_matching(points_shard, points_typology, dist_map, iterations=5000):
    """
    Finds the best transformation using RANSAC (coarse) followed by ICP (fine).
    """

    # --- PHASE 1: RANSAC (Coarse Search) ---
    best_ransac_score = np.inf
    best_ransac_params = None

    for _ in range(iterations):
        # Sample points
        idxs_shard = np.random.choice(len(points_shard), 2, replace=False)
        idxs_typ = np.random.choice(len(points_typology), 2, replace=False)

        p1, p2 = points_shard[idxs_shard]
        q1, q2 = points_typology[idxs_typ]

        # Fast checks
        if np.array_equal(p1, p2) or np.array_equal(q1, q2):
            continue

        # Fast Reject: Scale constraints
        len_shard = np.linalg.norm(p1 - p2)
        len_typ = np.linalg.norm(q1 - q2)
        if len_shard == 0:
            continue

        estimated_scale = len_typ / len_shard
        if not (MIN_SCALE <= estimated_scale <= MAX_SCALE):
            continue

        # Compute transform
        scale, rotation, translation = get_transformation(p1, p2, q1, q2)

        # Constraints check
        if abs(rotation) > MAX_ROTATION:  # Assumption: Shards are roughly oriented
            continue

        # Verify Score
        transformed = apply_transformation(points_shard, scale, rotation, translation)
        score = get_chamfer_score(transformed, dist_map)

        if score < best_ransac_score:
            best_ransac_score = score
            best_ransac_params = (scale, rotation, translation)

    # If RANSAC failed to find anything reasonable
    if best_ransac_params is None:
        return np.inf, (1.0, 0.0, (0.0, 0.0))

    # --- PHASE 2: ICP (Fine Tuning) ---

    # 1. Convert RANSAC params to Matrix (Init Pose)
    s_init, r_init, t_init = best_ransac_params
    init_pose_matrix = params_to_matrix(s_init, r_init, t_init)

    # 2. Run ICP
    # Note: Standard ICP is rigid (doesn't change scale), but it will optimize
    # rotation and translation *while keeping* the scale found by RANSAC.
    final_T_matrix, distances, _ = icp(
        source_points=points_shard,
        target_points=points_typology,
        init_pose=init_pose_matrix,
        max_iterations=30,  # Fast refinement
        tolerance=1e-4,
    )

    # 3. Decompose Matrix back to Params
    # We need this because run.py expects (s, r, t) for visualization
    final_scale, final_rotation, final_translation = matrix_to_params(final_T_matrix)

    # 4. Calculate Final Score (Standardized via Chamfer/DistMap)
    # While ICP minimizes point-to-point distance, we return the Chamfer score
    # to maintain consistency across the pipeline comparison.
    final_transformed = apply_transformation(
        points_shard, final_scale, final_rotation, final_translation
    )
    final_score = get_chamfer_score(final_transformed, dist_map)

    # Safety Check: If ICP somehow made it worse (rare, but possible with local minima),
    # revert to RANSAC result.
    if final_score > best_ransac_score:
        return best_ransac_score, best_ransac_params

    return final_score, (final_scale, final_rotation, final_translation)
