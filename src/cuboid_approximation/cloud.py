"""PLY input and spatially sampled local normal estimation."""

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.special import expit
from scipy.spatial import cKDTree


def load_points(path, min_opacity=0.1, outlier_distance_factor=50.0):
    if not 0 <= min_opacity <= 1:
        raise ValueError("min_opacity must be in [0,1].")
    if not np.isfinite(outlier_distance_factor) or outlier_distance_factor < 0:
        raise ValueError("outlier_distance_factor must be finite and nonnegative.")
    data = PlyData.read(str(path))
    if "vertex" not in data:
        raise ValueError("PLY must contain a vertex element.")
    vertex = data["vertex"]
    names = set(vertex.data.dtype.names)
    if not {"x", "y", "z"} <= names:
        raise ValueError("PLY must contain XYZ vertices.")
    points = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(float)
    valid = np.isfinite(points).all(axis=1)
    shape_fields = {f"scale_{i}" for i in range(3)} | {f"rot_{i}" for i in range(4)}
    gaussian = shape_fields | {"opacity"} <= names
    alpha = np.ones(len(points))
    if gaussian:
        alpha = expit(np.asarray(vertex["opacity"], dtype=float))
        valid &= np.isfinite(alpha) & (alpha >= min_opacity)
    if gaussian and {f"f_dc_{i}" for i in range(3)} <= names:
        colors = 0.5 + 0.28209479177387814 * np.column_stack(
            [vertex[f"f_dc_{i}"] for i in range(3)]
        )
        color_source = "gaussian_sh_dc"
    elif {"red", "green", "blue"} <= names:
        color_source = "vertex_rgb"
        colors = np.column_stack([vertex[name] for name in ("red", "green", "blue")]).astype(float)
        float_rgb = all(vertex.data.dtype[name].kind == "f" for name in ("red", "green", "blue"))
        finite_rgb = colors[np.isfinite(colors)]
        normalized = float_rgb and (
            not len(finite_rgb) or (finite_rgb.min() >= 0 and finite_rgb.max() <= 1)
        )
        if not normalized:
            colors /= 255
    else:
        colors = np.full_like(points, 0.65)
        color_source = "neutral_gray_no_source_color"
    colors = np.clip(np.nan_to_num(colors, nan=0.65, posinf=1, neginf=0), 0, 1)
    rejected = np.empty(0, dtype=int)
    if outlier_distance_factor and valid.sum() >= 20:
        ids = np.flatnonzero(valid)
        distinct, inverse = np.unique(points[ids], axis=0, return_inverse=True)
        if len(distinct) >= 20:
            distances, neighbors = cKDTree(distinct).query(
                distinct, k=min(9, len(distinct)), workers=-1
            )
            local = distances[:, 3]
            neighborhood_scale = np.median(local[neighbors[:, 1:]], axis=1)
            isolated = local > outlier_distance_factor * neighborhood_scale
            rejected = ids[isolated[inverse]]
            valid[rejected] = False
    splats = None
    if shape_fields <= names:
        splats = dict(
            log_scales=np.column_stack([vertex[f"scale_{i}"][valid] for i in range(3)]).astype(
                float
            ),
            quaternions_wxyz=np.column_stack([vertex[f"rot_{i}"][valid] for i in range(4)]).astype(
                float
            ),
        )
    retained = points[valid]
    distinct, first, inverse = np.unique(retained, axis=0, return_index=True, return_inverse=True)
    if len(distinct) < 20:
        raise ValueError("At least 20 distinct retained points are required.")
    distances, _ = cKDTree(distinct).query(distinct, k=2, workers=-1)
    spacing = float(np.median(distances[:, 1]))
    return dict(
        points=retained,
        distinct_indices=np.sort(first),
        spacing=spacing,
        spacing_per_point=distances[inverse, 1],
        colors=colors[valid],
        source_indices=np.flatnonzero(valid),
        source_count=len(points),
        gaussian_splats=gaussian,
        min_opacity=min_opacity,
        opacity=alpha[valid],
        splats=splats,
        color_source=color_source,
        outlier_source_indices=rejected,
        outlier_distance_factor=outlier_distance_factor,
    )


def write_points(
    path, points, colors, source_indices, scores=None, mask=None, valid=None, normals=None
):
    fields = [(name, "f8") for name in ("x", "y", "z")]
    fields += [(name, "u1") for name in ("red", "green", "blue")] + [("source_index", "i4")]
    if scores is not None:
        fields += [("edge_score", "f4"), ("is_edge", "u1"), ("score_valid", "u1")]
    if normals is not None:
        fields += [(name, "f4") for name in ("nx", "ny", "nz")]
    records = np.empty(len(points), dtype=fields)
    for i, name in enumerate(("x", "y", "z")):
        records[name] = points[:, i]
    for i, name in enumerate(("red", "green", "blue")):
        records[name] = np.rint(np.clip(colors[:, i], 0, 1) * 255).astype(np.uint8)
    records["source_index"] = source_indices
    if scores is not None:
        records["edge_score"], records["is_edge"], records["score_valid"] = scores, mask, valid
    if normals is not None:
        for i, name in enumerate(("nx", "ny", "nz")):
            records[name] = normals[:, i]
    PlyData([PlyElement.describe(records, "vertex")]).write(str(path))


def cell_keys(cells):
    """Collision-free scalar voxel keys, avoiding repeated structured-array sorts."""
    shape = cells.max(0) + 1
    if int(shape[0]) * int(shape[1]) * int(shape[2]) > np.iinfo(np.int64).max:
        # Extremely disparate coordinates: retain the full tuple without overflow.
        return (
            np.ascontiguousarray(cells).view(np.dtype((np.void, cells.dtype.itemsize * 3))).ravel()
        )
    return (cells[:, 0] * shape[1] + cells[:, 1]) * shape[2] + cells[:, 2]


def spatial_representatives(points, maximum, distinct_indices=None):
    """Return original point indices, at most one nearest-to-center point per cell."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Expected finite N-by-3 points.")
    if isinstance(maximum, bool) or not isinstance(maximum, (int, np.integer)) or maximum < 20:
        raise ValueError("The sample budget must be an integer of at least 20.")
    if len(points) < 20 or np.ptp(points, axis=0).max() <= 0:
        raise ValueError("At least 20 points with nonzero extent are required.")
    # Duplicates must not change which sampling branch or cell size is selected.
    if distinct_indices is None:
        _, original_ids = np.unique(points, axis=0, return_index=True)
        original_ids.sort()
    else:
        original_ids = np.asarray(distinct_indices)
    if len(original_ids) < 20:
        raise ValueError("At least 20 distinct points are required.")
    if len(original_ids) <= maximum:
        return original_ids, 0.0
    points = points[original_ids]
    origin = points.min(axis=0)
    size = np.linalg.norm(np.ptp(points, axis=0)) / 400
    low, high = 0.0, None
    best = None
    for _ in range(40):
        cells = np.floor((points - origin) / size).astype(np.int64)
        count = len(np.unique(cell_keys(cells)))
        if count <= maximum:
            if best is None or count > best[0]:
                best = (count, size)
            high = size
            if count >= maximum * 0.95:
                break
        else:
            low = size
        if high is None:
            size *= max(1.08, np.sqrt(count / maximum) * 1.02)
        elif low == 0:
            size /= 2
        else:
            if (high - low) / high < 1e-4:
                break
            size = (low + high) / 2
    if best is None or best[0] < 20:
        raise ValueError("Could not retain at least 20 distinct spatial representatives.")
    size = best[1]
    cells = np.floor((points - origin) / size).astype(np.int64)
    _, inverse = np.unique(cell_keys(cells), return_inverse=True)
    residual = (points - origin) / size - cells - 0.5
    priority = np.lexsort((np.arange(len(points)), np.sum(residual**2, axis=1), inverse))
    first = np.r_[True, inverse[priority[1:]] != inverse[priority[:-1]]]
    return original_ids[np.sort(priority[first])], float(size)


def local_point_spacing(points, fallback=None):
    """Nearest distinct-point distances, aligned with the original point order.

    Coincident samples share a scale instead of collapsing their neighborhood
    radius. A global spacing is retained by callers only for legacy consumers.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Expected finite N-by-3 points.")
    distinct, inverse = np.unique(points, axis=0, return_inverse=True)
    if len(distinct) < 2:
        if fallback is None or not np.isfinite(fallback) or fallback <= 0:
            raise ValueError("At least two distinct points are required for local spacing.")
        return np.full(len(points), fallback, dtype=float)
    distances, _ = cKDTree(distinct).query(distinct, k=2, workers=-1)
    return distances[inverse, 1]


def local_geometry(points, neighbors=30, radius_factor=6.0):
    """Locally radius-bounded, duplicate-invariant PCA with unsigned normals."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Expected finite N-by-3 points.")
    distinct, first, inverse = np.unique(points, axis=0, return_index=True, return_inverse=True)
    if len(distinct) < 3:
        raise ValueError("At least three distinct points are required for normals.")
    tree = cKDTree(distinct)
    distances, ids = tree.query(distinct, k=min(neighbors + 1, len(distinct)), workers=-1)
    scales = distances[:, 1]
    spacing = float(np.median(scales))
    radius = radius_factor * scales
    valid_neighbors = (
        (ids != np.arange(len(distinct))[:, None]) & (distances > 0)
        # A point on the radius must remain a neighbor after a rigid rotation;
        # rounding of both the nearest distance and the cutoff can exclude it.
        & (distances <= radius[:, None] * (1 + 1e-10))
    )
    counts = valid_neighbors.sum(axis=1)
    normals = np.zeros_like(distinct)
    eigenvalues = np.zeros_like(distinct)
    for start in range(0, len(distinct), 2048):
        part = slice(start, start + 2048)
        delta = (distinct[ids[part]] - distinct[part, None]) / scales[part, None, None]
        delta *= valid_neighbors[part, :, None]
        total = counts[part] + 1  # Include the central point in the covariance.
        mean = delta.sum(axis=1) / total[:, None]
        cov = np.einsum("nki,nkj->nij", delta, delta) / total[:, None, None]
        cov -= np.einsum("ni,nj->nij", mean, mean)
        values, vectors = np.linalg.eigh(cov)
        eigenvalues[part] = np.maximum(values, 0)  # Ascending: smallest first.
        normals[part] = vectors[:, :, 0]
    trace = eigenvalues.sum(axis=1)
    valid = (counts >= 8) & (trace > 0) & (eigenvalues[:, 1] > trace * 1e-5)
    valid &= eigenvalues[:, 0] <= 0.3 * eigenvalues[:, 1]
    return dict(
        points=points,
        tree=cKDTree(points),
        ids=first[ids[inverse]],
        distances=distances[inverse],
        neighbor_mask=valid_neighbors[inverse],
        neighbor_counts=counts[inverse],
        spacing=spacing,
        spacing_per_point=scales[inverse],
        radius=radius[inverse],
        normals=normals[inverse],
        eigenvalues=eigenvalues[inverse],
        valid=valid[inverse],
    )


def normal_variation(geometry):
    """90th percentile of unsigned neighbor-normal angles, in degrees."""
    g = geometry
    dots = np.einsum("nki,ni->nk", g["normals"][g["ids"]], g["normals"])
    angles = np.degrees(np.arccos(np.clip(np.abs(dots), 0, 1)))
    mask = g["neighbor_mask"] & g["valid"][g["ids"]]
    count = mask.sum(axis=1)
    angles[~mask] = np.inf
    angles.sort(axis=1)
    quantile_index = np.floor(0.9 * np.maximum(count - 1, 0)).astype(int)
    scores = angles[np.arange(len(angles)), quantile_index]
    valid = g["valid"] & (count >= 6)
    scores[~valid] = 0
    return scores, valid, {}
