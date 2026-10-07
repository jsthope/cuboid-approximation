"""Bounded orientation proposals from planar support and identifiable PCA axes.

PCA alone cannot recover the axes of a cube: its covariance is isotropic. Hull
facet areas provide geometric evidence that is independent of sampling density
and remain useful when opposite faces share the same eigenvalues.
"""

import itertools

import numpy as np
from scipy.spatial import ConvexHull, QhullError


def equivalent_frame(left, right, cosine):
    """Ignore signs and axis permutations, which do not change a cuboid frame."""
    return bool(np.all(np.max(np.abs(left.T @ right), axis=0) >= cosine))


def proper_frame(frame):
    """Return an orthonormal, right-handed copy without mutating its source."""
    u, _, vt = np.linalg.svd(frame)
    result = u @ vt
    if np.linalg.det(result) < 0:
        result[:, 2] *= -1
    return result


def _pca(points):
    centered = points - points.mean(axis=0)
    values, axes = np.linalg.eigh(centered.T @ centered / max(1, len(points)))
    # Both eigengaps must be nonzero to identify all three axes. In particular,
    # a cube's arbitrary eigenbasis must not outrank observed face directions.
    confidence = float(max(0, np.diff(values).min()) / max(values[-1], 1e-30))
    return values, axes[:, ::-1], confidence


def _normal_groups(normals, weights):
    """Area-weighted, unoriented normal votes with bounded work on curved hulls."""
    remaining = np.ones(len(weights), bool)
    total = float(weights.sum())
    cosine = np.cos(np.deg2rad(10))
    groups = []
    for _ in range(min(24, len(weights))):
        if not remaining.any():
            break
        seed = normals[np.argmax(np.where(remaining, weights, -1))]
        members = remaining & (np.abs(normals @ seed) >= cosine)
        signed = np.where(normals[members] @ seed >= 0, 1.0, -1.0)
        normal = (normals[members] * (weights[members] * signed)[:, None]).sum(0)
        normal /= np.linalg.norm(normal)
        members = remaining & (np.abs(normals @ normal) >= cosine)
        mass = float(weights[members].sum()) / total
        remaining[members] = False
        # Small patches of a curved hull are not evidence of a dominant plane.
        if mass >= 0.035:
            groups.append((normal, mass, []))
    return groups


def _hull_directions(points, values, pca):
    """Recover dominant support directions, including a planar square's edges."""
    if len(points) < 4 or values[-1] <= 0:
        return []
    # Bound the hull with one spatial representative per cell (at most 20^3).
    # Coordinate tie breaks make this independent of input order and duplicate
    # density; index-based subsampling can lose sparsely acquired faces.
    if len(points) > 8192:
        lower = points.min(0)
        width = np.ptp(points, axis=0).max() / 20
        local = (points - lower) / width
        cells = np.minimum(np.floor(local).astype(int), 19)
        ids = cells @ np.array([400, 20, 1])
        distances = np.sum((local - cells - 0.5) ** 2, axis=1)
        order = np.lexsort((points[:, 2], points[:, 1], points[:, 0], distances, ids))
        representatives = order[np.r_[True, np.diff(ids[order]) != 0]]
        # Retain true extrema even if the nearest cell representative lies
        # inside the hull. Resolve tied extrema by coordinates, not row order.
        extremes = []
        for axis in range(3):
            for value in (points[:, axis].min(), points[:, axis].max()):
                tied = np.flatnonzero(points[:, axis] == value)
                tie_order = np.lexsort((points[tied, 2], points[tied, 1], points[tied, 0]))
                extremes.append(tied[tie_order[0]])
        points = np.unique(points[np.r_[representatives, extremes]], axis=0)
    centered = points - points.mean(0)
    centered /= max(np.linalg.norm(centered, axis=1).max(), 1e-30)
    try:
        if values[0] <= values[-1] * 1e-12:
            if values[1] <= values[-1] * 1e-12:
                return []
            local = centered @ pca[:, :2]
            hull = ConvexHull(local)
            edges = local[hull.simplices[:, 1]] - local[hull.simplices[:, 0]]
            lengths = np.linalg.norm(edges, axis=1)
            directions = (edges / lengths[:, None]) @ pca[:, :2].T
            return [(pca[:, 2], 1.0, [])] + _normal_groups(directions, lengths)
        hull = ConvexHull(centered)
    except QhullError:
        # Degenerate inputs still have their PCA and region proposals. Do not
        # perturb them with Qhull's joggle option, which invents facet normals.
        return []
    triangles = centered[hull.simplices]
    areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    ) / 2
    valid = areas > 1e-15
    return _normal_groups(hull.equations[valid, :3], areas[valid])


def orientation_proposals(points, faces):
    """Rank geometric normal pairs before ambiguous global/local PCA frames."""
    points = np.asarray(points, float)
    if len(points) < 3:
        return []
    values, global_frame, confidence = _pca(points)
    proposed = [(1.0 + confidence, global_frame, [])]
    directions = _hull_directions(points, values, global_frame)
    face_count = max(1, sum(face["points"] for face in faces))
    region_directions = [
        (face["normal"], face["points"] / face_count, [face["region_id"]])
        for face in sorted(faces, key=lambda face: -face["points"])[:32]
    ]
    # Combining region normals before consuming the budget avoids arbitrary
    # in-plane PCA axes on square faces. Do not cross unrelated evidence sets.
    for group in (directions, region_directions):
        for (a, wa, ia), (b, wb, ib) in itertools.combinations(group, 2):
            alignment = abs(float(a @ b))
            if alignment > np.sin(np.deg2rad(12)):
                continue
            frame = proper_frame(np.column_stack((a, b, np.cross(a, b))))
            proposed.append((3.0 + (wa + wb) * (1 - alignment), frame, ia + ib))
    for face in faces:
        proposed.append((2.0 + face["points"] / face_count, face["frame"], [face["region_id"]]))
    if len(points) >= 48:
        # Spatial fallback follows the global frame instead of fixed world
        # octants. Its low rank leaves the budget to actual planar evidence.
        local = (points - points.mean(0)) @ global_frame
        cells = (local >= np.median(local, axis=0)).astype(int) @ np.array([1, 2, 4])
        for cell in np.unique(cells):
            patch = points[cells == cell]
            if len(patch) >= 12:
                _, frame, confidence = _pca(patch)
                proposed.append((confidence, frame, []))
    return sorted(proposed, key=lambda item: -item[0])
