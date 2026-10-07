"""Connected normal-compatible regions used to propose cuboid orientations."""

import colorsys
from dataclasses import asdict, dataclass
from time import perf_counter
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


@dataclass
class SurfaceConfig:
    neighbors: int = 16
    neighbor_radius_factor: float = 3.5
    line_barrier_factor: float = 2.0
    normal_angle_degrees: float = 25.0
    tangent_distance_factor: float = 0.75
    min_region_points: int = 50

    def validate(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("neighbors", "min_region_points"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer.")
        if self.neighbor_radius_factor >= 2 * self.line_barrier_factor:
            raise ValueError("The neighbor radius must not jump across the line barrier.")
        if self.normal_angle_degrees >= 90:
            raise ValueError("Normal angles must be less than 90 degrees.")


def distance_to_segments(points, segments):
    """Euclidean distance to finite segments, including degenerate endpoints."""
    result = np.full(len(points), np.inf)
    for start, end in segments:
        direction = end - start
        squared_length = direction @ direction
        relative = points - start
        t = np.clip(relative @ direction / squared_length, 0, 1) if squared_length > 0 else 0
        offset = relative - np.asarray(t)[..., None] * direction
        result = np.minimum(result, np.linalg.norm(offset, axis=1))
    return result


def segment_surfaces(
    points, normals, normal_valid, edge_mask, segments, spacing, config=None, attach_borders=False,
    spacing_per_point=None,
):
    """Cut a local surface graph into core regions used by orientation fitting.

    Labels >= 0 identify estimated surface regions, not certified closed polygons.
    assignment_kind: 0 = unassigned, 1 = core.
    All normal comparisons are invariant to normal sign. No points are generated.
    """
    if attach_borders:
        raise ValueError(
            "attach_borders=True is no longer supported: orientation fitting uses only "
            "core regions. Remove this option; uncertain border points remain unassigned."
        )
    started = perf_counter()
    config = config or SurfaceConfig()
    config.validate()
    points, normals = np.asarray(points, dtype=float), np.asarray(normals, dtype=float)
    normal_valid, edge_mask = np.asarray(normal_valid, bool), np.asarray(edge_mask, bool)
    segments = np.asarray(segments, dtype=float).reshape(-1, 2, 3)
    if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape:
        raise ValueError("Expected points and normals with shape (N, 3).")
    if normal_valid.shape != (len(points),) or edge_mask.shape != (len(points),):
        raise ValueError("Masks must align with the points.")
    if (
        not np.isfinite(spacing)
        or spacing <= 0
        or not np.isfinite(points).all()
        or not np.isfinite(segments).all()
    ):
        raise ValueError("Finite coordinates and a positive spacing are required.")
    local_spacing = (
        np.full(len(points), spacing) if spacing_per_point is None
        else np.asarray(spacing_per_point, dtype=float)
    )
    if local_spacing.shape != (len(points),) or not np.isfinite(local_spacing).all() or np.any(local_spacing <= 0):
        raise ValueError("Local spacing must be finite, positive and aligned with points.")
    lengths = np.linalg.norm(normals, axis=1)
    usable = normal_valid & np.isfinite(normals).all(axis=1) & (lengths > 1e-12)
    unit = np.zeros_like(normals)
    unit[usable] = normals[usable] / lengths[usable, None]
    line_distance = distance_to_segments(points, segments)
    core_mask = usable & ~edge_mask & (line_distance > local_spacing * config.line_barrier_factor)
    core_ids = np.flatnonzero(core_mask)
    labels = np.full(len(points), -1, dtype=np.int32)
    assignment = np.zeros(len(points), dtype=np.uint8)
    if len(core_ids):
        distances, neighbors = cKDTree(points[core_ids]).query(
            points[core_ids],
            k=list(range(1, min(config.neighbors + 1, len(core_ids)) + 1)),
            workers=-1,
        )
        row = np.broadcast_to(np.arange(len(core_ids))[:, None], neighbors.shape)
        keep = np.isfinite(distances) & (neighbors != row)
        first, second = row[keep], neighbors[keep]
        a, b = core_ids[first], core_ids[second]
        pair_spacing = np.minimum(local_spacing[a], local_spacing[b])
        near = distances[keep] <= pair_spacing * config.neighbor_radius_factor
        delta = points[b] - points[a]
        angle_ok = np.abs(np.einsum("ij,ij->i", unit[a], unit[b])) >= np.cos(
            np.deg2rad(config.normal_angle_degrees)
        )
        residual = np.maximum(
            np.abs(np.einsum("ij,ij->i", delta, unit[a])),
            np.abs(np.einsum("ij,ij->i", delta, unit[b])),
        )
        keep = near & angle_ok & (residual <= pair_spacing * config.tangent_distance_factor)
        graph = csr_matrix(
            (np.ones(keep.sum(), dtype=np.uint8), (first[keep], second[keep])),
            shape=(len(core_ids), len(core_ids)),
        )
        count, component = connected_components(graph, directed=False)
        sizes = np.bincount(component, minlength=count)
        order = np.argsort(-sizes, kind="stable")
        order = order[sizes[order] >= config.min_region_points]
        mapping = np.full(count, -1, dtype=np.int32)
        mapping[order] = np.arange(len(order))
        labels[core_ids] = mapping[component]
        assignment[labels >= 0] = 1

    return dict(
        labels=labels,
        assignment_kind=assignment,
        line_distance=line_distance,
        core_candidate_mask=core_mask,
        seconds=perf_counter() - started,
        config=asdict(config),
    )


def region_colors(labels):
    count = int(labels.max(initial=-1)) + 1
    palette = np.array(
        [
            colorsys.hsv_to_rgb(
                (0.07 + 0.61803398875 * i) % 1, 0.62 + 0.09 * (i % 3), 0.84 + 0.15 * ((i // 3) % 2)
            )
            for i in range(count)
        ]
    ).reshape(-1, 3)
    colors = np.tile([0.61, 0.64, 0.68], (len(labels), 1))
    colors[labels >= 0] = palette[labels[labels >= 0]]
    return colors, palette
