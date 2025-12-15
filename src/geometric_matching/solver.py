"""
Implements RANSAC-based geometric alignment to match
shard points onto the larger typology contour.
"""

import numpy as np


MIN_SCALE = 0.5
MAX_SCALE = 1.5
MAX_ROTATION = np.deg2rad(20)


def get_transformation(p1, p2, q1, q2):
    """
    Calculates the affine transformation (scale, rotation, translation) needed to
    map two shard points (p1, p2) onto two typology points (q1, q2).
    """

    vec_p = p2 - p1
    vec_q = q2 - q1

    len_s = np.linalg.norm(vec_p)
    len_q = np.linalg.norm(vec_q)

    # Compute the ratio to resize the shard so it matches the
    # physical scale of the reference typology.
    scale = len_q / len_s

    angle_p = np.arctan2(vec_p[1], vec_p[0])
    angle_q = np.arctan2(vec_q[1], vec_q[0])

    # Determine the angle needed to align the shard's orientation with the typology.
    rotation = angle_q - angle_p

    cos, sin = np.cos(rotation), np.sin(rotation)

    # Calculate the translation to shift the shard so that p1 exactly overlaps with q1.
    t_x = q1[0] - (scale * (p1[0] * cos - p1[1] * sin))
    t_y = q1[1] - (scale * (p1[0] * sin + p1[1] * cos))

    return scale, rotation, (t_x, t_y)


def apply_transformation(points, scale, rotation, translation):
    """Apply a transformation to a (N, 2) array of points."""

    p_x, p_y = points[:, 0], points[:, 1]
    t_x, t_y = translation

    cos, sin = np.cos(rotation), np.sin(rotation)

    # Transform all shard points. Apply scale and rotation
    # first, then translate to the new position.
    x = (scale * (p_x * cos - p_y * sin)) + t_x
    y = (scale * (p_x * sin + p_y * cos)) + t_y

    return np.array([x, y]).T


def get_chamfer_score(transformed_points, dist_map, penalty_factor=1):
    """Computes the Chamfer score with penalties for out-of-bounds points."""

    points_int = np.rint(transformed_points).astype(int)

    p_x = points_int[:, 0]
    p_y = points_int[:, 1]

    h, w = dist_map.shape

    # Identify which points are inside the distance map bounds.
    valid_mask = (p_x >= 0) & (p_x < w) & (p_y >= 0) & (p_y < h)

    # Prepare a container with all distances and initialize with the penalty value.
    penalty_value = np.max(dist_map) * penalty_factor
    all_distances = np.full(points_int.shape[0], penalty_value)

    # Fill in actual distances for the valid points.
    if np.any(valid_mask):
        valid_x = p_x[valid_mask]
        valid_y = p_y[valid_mask]
        all_distances[valid_mask] = dist_map[valid_y, valid_x]

    return np.mean(all_distances)


def solve_matching(points_shard, points_typology, dist_map, iterations=10000):
    """
    Finds the best transformation to align shard points to typology
    using RANSAC and Chamfer distance.
    """

    best_score, best_params = np.inf, None

    for _ in range(iterations):

        # Sample 2 random points from each set.
        p1, p2 = points_shard[np.random.choice(len(points_shard), 2, replace=False)]
        q1, q2 = points_typology[np.random.choice(len(points_typology), 2, replace=False)]

        # Prevent division by zero if input data has duplicate coordinates.
        if np.array_equal(p1, p2) or np.array_equal(q1, q2):
            continue

        # Fast Reject: Check if length ratio is within bounds before computing full transform.
        len_shard = np.linalg.norm(p1 - p2)
        len_typ = np.linalg.norm(q1 - q2)

        estimated_scale = len_typ / len_shard
        if not (MIN_SCALE <= estimated_scale <= MAX_SCALE):
            continue

        scale, rotation, translation = get_transformation(p1, p2, q1, q2)

        # Check rotation bounds (and double-check scale for safety).
        if scale < MIN_SCALE or scale > MAX_SCALE or abs(rotation) > MAX_ROTATION:
            continue

        # Heavy operations: Transform all points and compute pixel-wise score.
        transformed_points_shard = apply_transformation(points_shard, scale, rotation, translation)
        score = get_chamfer_score(transformed_points_shard, dist_map)

        if score < best_score:
            best_score = score
            best_params = (scale, rotation, translation)

    return best_score, best_params
