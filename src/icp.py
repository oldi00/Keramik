import numpy as np
from sklearn.neighbors import NearestNeighbors


def best_fit_transform(A, B):
    """
    Berechnet die starre Transformation (Rotation + Translation),
    die A auf B abbildet (mittels SVD).
    """
    assert A.shape == B.shape
    m = A.shape[1]

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # Spiegelung verhindern
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    t = centroid_B.T - np.dot(R, centroid_A.T)

    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T


def nearest_neighbor(src, dst):
    """Findet die nächsten Nachbarn in dst für jeden Punkt in src."""
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(
    source_points, target_points, init_pose=None, max_iterations=50, tolerance=1e-5
):
    """
    Führt den ICP-Algorithmus aus.

    Args:
        source_points: (N, 2) Array der Scherbe
        target_points: (M, 2) Array des Profils (Referenz)
        init_pose: (3, 3) Transformationsmatrix (z.B. von RANSAC)

    Returns:
        T: (3, 3) Finale Transformationsmatrix
        distances: Abstände der Punkte nach der Konvergenz
    """
    m = source_points.shape[1]

    # Homogene Koordinaten erstellen (N, 3)
    src = np.ones((m + 1, source_points.shape[0]))
    src[:m, :] = np.copy(source_points.T)

    dst = target_points  # Target bleibt (M, 2)

    # Initiale Pose anwenden (WICHTIG: Startpunkt für ICP)
    if init_pose is not None:
        src = np.dot(init_pose, src)

    # Start-Transformation ist die init_pose (oder Identität)
    final_T = init_pose if init_pose is not None else np.identity(m + 1)

    prev_error = 0

    for i in range(max_iterations):
        # 1. Matching (nur x,y Koordinaten nutzen)
        src_xyz = src[:m, :].T
        distances, indices = nearest_neighbor(src_xyz, dst)

        # 2. Beste Transformation für DIESE Zuordnung berechnen
        # Wir nehmen nur die 'src' Punkte und ihre gefundenen Partner in 'dst'
        matched_dst = dst[indices]
        T_step = best_fit_transform(src_xyz, matched_dst)

        # 3. Update der Punkte
        src = np.dot(T_step, src)

        # 4. Update der Gesamt-Matrix (Matrix-Multiplikation von links)
        final_T = np.dot(T_step, final_T)

        # 5. Abbruchkriterium
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    return final_T, distances, i
