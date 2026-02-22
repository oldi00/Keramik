import numpy as np
from scipy.optimize import minimize
from sklearn.neighbors import KDTree


def transform_rtsz(points, x_move, theta, scale, y_move):
    # Get the x and y points
    p_x = points[:, 0].astype(float)
    p_y = points[:, 1].astype(float)

    # First move by some value on the x-axis
    p_x = p_x + x_move

    # Calculate the rotation by the value theta
    c, s = np.cos(theta), np.sin(theta)
    rot_x = p_x * c - p_y * s
    rot_y = p_y * c + p_x * s

    # Apply scaling to the points
    scaled_x = rot_x * scale
    scaled_y = rot_y * scale

    # Finally apply y axis transformation
    final_x = scaled_x
    final_y = scaled_y + y_move

    return np.column_stack((final_x, final_y))


def matrix_from_rtsz(r_tr, theta, size, z_tr):
    c, s = np.cos(theta), np.sin(theta)

    T = np.eye(3)
    T[0, 0] = size * c
    T[0, 1] = -size * s
    T[0, 2] = size * c * r_tr

    T[1, 0] = size * s
    T[1, 1] = size * c
    T[1, 2] = size * s * r_tr + z_tr

    return T


def rtsz_from_matrix(T):
    A, _, C = T[0, 0], T[0, 1], T[0, 2]
    D, _, F = T[1, 0], T[1, 1], T[1, 2]
    size = np.sqrt(A**2 + D**2)
    if size == 0:
        size = 1.0

    theta = np.arctan2(D, A)

    if abs(A) > 1e-6:
        r_tr = C / A
    else:
        r_tr = 0.0  # Fallback

    z_tr = F - (D * r_tr)

    return np.array([r_tr, theta, size, z_tr])


def nearest_neighbor_distances(source, target_tree):

    # This looks for nearest neighbor for every source point through
    # every target points. Euclidean distance.
    dists, _ = target_tree.query(source)
    return dists


def objective_function(params, source_points, target_tree, minfun="rmsd"):
    x_move, theta, scale, y_move = params

    transformed_source = transform_rtsz(source_points, x_move, theta, scale, y_move)
    dists = nearest_neighbor_distances(transformed_source, target_tree)

    if minfun == "rmsd":
        return np.sqrt(np.mean(dists**2))
    elif minfun == "ssd":
        return np.sum(dists**2)
    elif minfun == "var":
        return np.var(dists)
    else:
        return np.mean(dists)


def icp(
    source_points,
    target_points,
    init_pose=None,
    max_iterations=100,
    tolerance=1e-6,
    delta_x=3.0,
    delta_y=1.0,
    delta_scale=0.05,  # In prozent
    delta_rotation=0.07,  # In radiant -> 0.07 Rad == 4 Grad
):

    target_tree = KDTree(target_points)
    if init_pose is not None:
        initial_params = rtsz_from_matrix(init_pose)
    else:
        initial_params = np.array([0.0, 0.0, 1.0, 0.0])

    # Define the dynamic boundaries here
    bounds = (
        (initial_params[0] - delta_x, initial_params[0] + delta_x),
        (initial_params[1] - delta_rotation, initial_params[1] + delta_rotation),
        (initial_params[2] - delta_scale, initial_params[2] + delta_scale),
        (initial_params[3] - delta_y, initial_params[3] + delta_y),
    )

    result = minimize(
        objective_function,
        initial_params,
        args=(source_points, target_tree, "rmsd"),
        method="Powell",  # Diese Methode unterstützt Grenzen
        bounds=bounds,
        tol=tolerance,
        options={"maxiter": max_iterations},
    )

    best_params = result.x
    final_transformed = transform_rtsz(source_points, *best_params)
    final_dists, _ = target_tree.query(final_transformed)

    final_T = matrix_from_rtsz(*best_params)

    return final_T, final_dists.ravel(), result.nfev
