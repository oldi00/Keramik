"""..."""  # todo

import numpy as np


def get_transformation(p1, p2, q1, q2):
    """
    Calculates the transformation to map the shard
    points (p1, p2) onto the typology points (q1, q2).
    """

    vec_p = p2 - p1
    vec_q = q2 - q1

    len_s = np.linalg.norm(vec_p)
    len_q = np.linalg.norm(vec_q)

    scale = len_q / len_s

    angle_p = np.arctan2(vec_p[1], vec_p[0])
    angle_q = np.arctan2(vec_q[1], vec_q[0])

    rotation = angle_q - angle_p

    cos, sin = np.cos(rotation), np.sin(rotation)

    t_x = q1[0] - (scale * (p1[0] * cos - p1[1] * sin))
    t_y = q1[1] - (scale * (p1[0] * sin + p1[1] * cos))

    return scale, rotation, (t_x, t_y)


def apply_transformation(points, scale, rotation, translation):
    """Apply the transformation to a (N, 2) array of points."""

    p_x, p_y = points[:, 0], points[:, 1]
    t_x, t_y = translation

    cos, sin = np.cos(rotation), np.sin(rotation)

    x = (scale * (p_x * cos - p_y * sin)) + t_x
    y = (scale * (p_x * sin + p_y * cos)) + t_y

    return np.array([x, y]).T


def get_chamfer_score(transformed_points, dist_map):
    """..."""

    points = np.rint(transformed_points).astype(int)

    p_x = points[:, 0]
    p_y = points[:, 1]

    h, w = dist_map.shape

    valid_mask = (p_x >= 0) & (p_x < w) & (p_y >= 0) & (p_y < h)
    print(valid_mask)

    if np.sum(valid_mask) == 0:
        return np.inf

    valid_x = p_x[valid_mask]
    valid_y = p_y[valid_mask]

    distances = dist_map[valid_y, valid_x]

    return np.mean(distances)


def main():

    from utils import get_points, get_dist_map
    import random

    shard_path = "data/preprocess/shards_profiles/10004.png"
    typology_path = "data/preprocess/typology_crops/Drag33.png"

    points_shard = get_points(shard_path)
    points_typology = get_points(typology_path)

    p1, p2 = random.sample(points_shard, 2)

    # Select 2 unique random points from the typology list
    q1, q2 = random.sample(points_typology, 2)

    scale, rotation, translation = get_transformation(p1, p2, q1, q2)

    transformed_points = apply_transformation(points_shard, scale, rotation, translation)

    dist_map = get_dist_map(typology_path)

    score = get_chamfer_score(transformed_points, dist_map)
    print(score)


if __name__ == "__main__":
    main()
