"""
Run RANSAC-based geometric matching to estimate similarity transformations
(scale, rotation, translation) between shard and typology point sets.
"""

import numpy as np


def generate_batch_indices(n_shard, n_typ, iterations):
    """Generate random point pair indices for RANSAC hypothesis generation."""

    idx_p = np.random.randint(0, n_shard, (iterations, 2))
    idx_q = np.random.randint(0, n_typ, (iterations, 2))

    return idx_p, idx_q


def estimate_geometric_params(p1, p2, q1, q2, config):
    """Estimate vectorized similarity transformation parameters (S, R, T) from point pairs."""

    vec_p = p2 - p1
    vec_q = q2 - q1

    len_p = np.hypot(vec_p[:, 0], vec_p[:, 1])
    len_q = np.hypot(vec_q[:, 0], vec_q[:, 1])

    # Prevent division by zero or unstable estimates from coincident/collinear points.
    valid_mask = (len_p > 1e-6)

    scale = np.zeros_like(len_p)
    np.divide(len_q, len_p, out=scale, where=valid_mask)

    valid_mask &= (scale >= config["min_scale"]) & (scale <= config["max_scale"])

    ang_p = np.arctan2(vec_p[:, 1], vec_p[:, 0])
    ang_q = np.arctan2(vec_q[:, 1], vec_q[:, 0])
    rotation = ang_q - ang_p

    # Normalize angles to [-pi, pi] to ensure valid_mask checks minimal rotation distance.
    rotation = (rotation + np.pi) % (2 * np.pi) - np.pi

    max_rotation = np.deg2rad(config["max_rotation_deg"])
    valid_mask &= (np.abs(rotation) <= max_rotation)

    cos_r = np.cos(rotation)
    sin_r = np.sin(rotation)

    t_x = q1[:, 0] - scale * (p1[:, 0] * cos_r - p1[:, 1] * sin_r)
    t_y = q1[:, 1] - scale * (p1[:, 0] * sin_r + p1[:, 1] * cos_r)

    return scale, rotation, t_x, t_y, valid_mask


def score_survivors(points_shard, dist_map, scale, rotation, t_x, t_y):
    """Compute Chamfer scores for transformed points using the distance map."""

    H, W = dist_map.shape

    # Add new axis to enable broadcasting: (1, N_points) vs (N_hypotheses, 1)
    # This generates a matrix of size (N_hypotheses, N_points).
    src_x = points_shard[:, 0][None, :]
    src_y = points_shard[:, 1][None, :]

    s_bc = scale[:, None]
    c_bc = np.cos(rotation)[:, None]
    sn_bc = np.sin(rotation)[:, None]
    tx_bc = t_x[:, None]
    ty_bc = t_y[:, None]

    dst_x = s_bc * (src_x * c_bc - src_y * sn_bc) + tx_bc
    dst_y = s_bc * (src_x * sn_bc + src_y * c_bc) + ty_bc

    ix = np.rint(dst_x).astype(np.int32)
    iy = np.rint(dst_y).astype(np.int32)

    is_inside = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)

    # Assign max penalty to points falling outside the distance map boundaries.
    penalty = np.max(dist_map)
    scores = np.full(ix.shape, penalty, dtype=np.float32)

    if np.any(is_inside):
        scores[is_inside] = dist_map[iy[is_inside], ix[is_inside]]

    base_score = np.mean(scores, axis=1)

    min_y = np.min(dst_y, axis=1)

    rim_threshold = 12.0

    distance_below_rim = np.maximum(0, min_y - rim_threshold)
    distance_above_top = np.maximum(0, -min_y)

    y_penalty = 0.5 * (distance_below_rim**2 + distance_above_top**2)

    return base_score + y_penalty


def ransac(points_shard, points_typology, dist_map, config):
    """Execute RANSAC pipeline to identify the best geometric alignment."""

    idx_p, idx_q = generate_batch_indices(
        len(points_shard),
        len(points_typology),
        iterations=config["iterations"]
    )

    p1 = points_shard[idx_p[:, 0]]
    p2 = points_shard[idx_p[:, 1]]
    q1 = points_typology[idx_q[:, 0]]
    q2 = points_typology[idx_q[:, 1]]

    scale, rot, tx, ty, valid_mask = estimate_geometric_params(p1, p2, q1, q2, config)

    # Early exit if nothing matches constraints.
    if not np.any(valid_mask):
        return np.inf, None

    # Optimization: Filter hypotheses before heavy scoring to reduce memory/compute.
    scale = scale[valid_mask]
    rot = rot[valid_mask]
    tx = tx[valid_mask]
    ty = ty[valid_mask]

    scores = score_survivors(points_shard, dist_map, scale, rot, tx, ty)

    best_idx = np.argmin(scores)

    params = [
        scale[best_idx],
        rot[best_idx],
        (tx[best_idx], ty[best_idx]),
    ]

    return scores[best_idx], params
