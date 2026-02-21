import numpy as np
from scipy.optimize import minimize


def transform_rtsz(points, r_tr, theta, size, z_tr):
    """
    Applies the specific 'rtsz' transformation order defined in the R code:
    1. Translation in R (x-axis)
    2. Rotation by theta
    3. Scaling by size
    4. Translation in Z (y-axis)

    R Logic:
    pro[,1] <- pro[,1]+r.tr
    pro <- rot2dM(pro,theta)
    pro <- pro*siz
    pro[,2] <- pro[,2]+z.tr
    """
    # 1. Translate R (x-axis only)
    p_x = points[:, 0] + r_tr
    p_y = points[:, 1]

    # 2. Rotate
    c, s = np.cos(theta), np.sin(theta)
    # Note: R's rot2dM: x' = x*cos - y*sin, y' = y*cos + x*sin
    rot_x = p_x * c - p_y * s
    rot_y = p_y * c + p_x * s

    # 3. Scale
    scaled_x = rot_x * size
    scaled_y = rot_y * size

    # 4. Translate Z (y-axis only)
    final_x = scaled_x
    final_y = scaled_y + z_tr

    return np.column_stack((final_x, final_y))


def matrix_from_rtsz(r_tr, theta, size, z_tr):
    """
    Constructs the equivalent 3x3 Homogeneous Transformation Matrix
    from the 4 'rtsz' parameters.
    """
    c, s = np.cos(theta), np.sin(theta)

    # Based on the math of transform_rtsz:
    # x' = (size*c)*x + (-size*s)*y + (size*c*r_tr)
    # y' = (size*s)*x + (size*c)*y + (size*s*r_tr + z_tr)

    T = np.eye(3)
    T[0, 0] = size * c
    T[0, 1] = -size * s
    T[0, 2] = size * c * r_tr

    T[1, 0] = size * s
    T[1, 1] = size * c
    T[1, 2] = size * s * r_tr + z_tr

    return T


def rtsz_from_matrix(T):
    """
    Extracts approximate (r_tr, theta, size, z_tr) parameters from a
    standard 3x3 affine matrix to seed the optimization.
    """
    # T = [[A, B, C], [D, E, F], [0, 0, 1]]
    A, _, C = T[0, 0], T[0, 1], T[0, 2]
    D, _, F = T[1, 0], T[1, 1], T[1, 2]

    # 1. Extract Scale (Size)
    # Assuming uniform scaling: size = sqrt(A^2 + D^2)
    size = np.sqrt(A**2 + D**2)
    if size == 0:
        size = 1.0

    # 2. Extract Rotation (Theta)
    theta = np.arctan2(D, A)

    # 3. Extract r_tr (Translation R)
    # From matrix def: C = size * cos(theta) * r_tr  => C = A * r_tr
    if abs(A) > 1e-6:
        r_tr = C / A
    else:
        r_tr = 0.0  # Fallback

    # 4. Extract z_tr (Translation Z)
    # From matrix def: F = size * sin(theta) * r_tr + z_tr => F = D * r_tr + z_tr
    z_tr = F - (D * r_tr)

    return np.array([r_tr, theta, size, z_tr])


def nearest_neighbor_distances(source, target_tree):
    """
    Calculates Euclidean distances from each source point to the
    nearest point in the target using a pre-built KDTree.
    """
    dists, _ = target_tree.query(source)
    return dists


def objective_function(params, source_points, target_tree, minfun="rmsd"):
    """
    The cost function to minimize.
    Applies transform -> Calculates distances -> Returns Error Metric
    """
    r_tr, theta, size, z_tr = params

    # 1. Transform
    transformed_source = transform_rtsz(source_points, r_tr, theta, size, z_tr)

    # 2. Calculate Distances
    dists = nearest_neighbor_distances(transformed_source, target_tree)

    # 3. Compute Metric (matching R code 'minfun')
    if minfun == "rmsd" or minfun == "rmsd.cv" or minfun == "rmsd.norm":
        # Root Mean Square Deviation
        return np.sqrt(np.mean(dists**2))
    elif minfun == "ssd":
        # Sum of Squared Distances
        return np.sum(dists**2)
    elif minfun == "var":
        return np.var(dists)
    else:
        # Default to mean distance
        return np.mean(dists)


def icp(
    source_points, target_points, init_pose=None, max_iterations=100, tolerance=1e-6
):
    """
    Optimization-based ICP that mimics the R code logic.
    Optimizes 4 parameters: [r_tr, theta, size, z_tr].

    Args:
        source_points: (N, 2) numpy array
        target_points: (M, 2) numpy array
        init_pose: (3, 3) matrix (optional, e.g. from RANSAC)

    Returns:
        final_T: (3, 3) Transformation Matrix
        distances: Final error distances per point
        iterations: Number of function evaluations
    """
    # 1. Setup efficient Nearest Neighbor search
    # Using KDTree for O(log M) lookups inside the optimization loop
    from sklearn.neighbors import KDTree

    target_tree = KDTree(target_points)

    # 2. Initial Guess
    if init_pose is not None:
        # Convert RANSAC matrix to our 4 parameters
        initial_params = rtsz_from_matrix(init_pose)
    else:
        # Default: r=0, theta=0, size=1, z=0
        initial_params = np.array([0.0, 0.0, 1.0, 0.0])

    # 3. Run Optimization (replaces the 'hclimbing' or standard ICP loop)
    # Nelder-Mead is a robust, derivative-free optimizer good for this type of problem
    result = minimize(
        objective_function,
        initial_params,
        args=(source_points, target_tree, "rmsd"),
        method="Nelder-Mead",
        tol=tolerance,
        options={"maxiter": max_iterations},
    )

    best_params = result.x

    # 4. Prepare Result
    # Re-calculate final distances for reporting
    final_transformed = transform_rtsz(source_points, *best_params)
    final_dists, _ = target_tree.query(final_transformed)

    # Convert optimized params back to Matrix for compatibility with solver.py/run.py
    final_T = matrix_from_rtsz(*best_params)

    return final_T, final_dists.ravel(), result.nfev
