"""Voxelized observed support, local interior reconstruction and proximity checks."""

import math
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .parallel import current_workers


class CoverageNotReachedError(RuntimeError):
    """A completed geometric stage cannot satisfy the requested point target."""

    def __init__(self, message, **details):
        super().__init__(message)
        self.details = details


def center_spacing(points, h):
    points = np.unique(points, axis=0)
    distances, _ = cKDTree(points).query(points, k=min(2, len(points)), workers=current_workers())
    positive = distances[:, 1][distances[:, 1] > 0] if len(points) > 1 else np.empty(0)
    return float(np.median(positive)) if len(positive) else h


def check_grid_memory(shape, maximum_mb, stage, bytes_per_cell=64, fixed_bytes=0):
    """Reject oversized dense workspaces before their allocation."""
    estimate = math.prod(int(n) for n in shape) * bytes_per_cell + fixed_bytes
    if any(n <= 0 for n in shape) or estimate > maximum_mb * 1024**2:
        raise ValueError(
            f"{stage}: grid {tuple(int(n) for n in shape)} needs an estimated "
            f"{estimate / 1024**2:.0f} MiB (budget {maximum_mb} MiB). "
            "Lower --resolution, inspect isolated points, or increase --max-memory-mb."
        )


def splat_geometry(points, shapes, h, spacing, sigma):
    """Validate once; derive bounds from the actual capped ellipsoids."""
    logs = np.asarray(shapes["log_scales"], float)
    quats = np.asarray(shapes["quaternions_wxyz"], float)
    if logs.shape != (len(points), 3) or quats.shape != (len(points), 4):
        raise ValueError("Splat shapes must align with the retained source points.")
    finite = np.isfinite(logs).all(axis=1) & np.isfinite(quats).all(axis=1)
    quaternion_scale = np.max(np.abs(quats), axis=1)
    finite &= quaternion_scale > 0
    finite &= (logs > -700).all(axis=1) & (logs < 700).all(axis=1)
    cap = max(h, 4 * spacing)
    scales = np.exp(np.minimum(logs[finite], np.log(cap)))
    usable = scales.min(axis=1) >= h * 1e-12
    finite[np.flatnonzero(finite)[~usable]] = False
    scales = scales[usable]
    rotations = (
        Rotation.from_quat(
            quats[finite][:, [1, 2, 3, 0]] / quaternion_scale[finite, None]
        ).as_matrix()
        if finite.any()
        else np.empty((0, 3, 3))
    )
    radii = sigma * np.sqrt(np.einsum("nij,nj->ni", rotations**2, scales**2))
    inverse = np.einsum("nik,nk,njk->nij", rotations, 1 / scales**2, rotations)
    return (
        finite,
        radii,
        inverse,
        dict(
            method="splat_ellipsoids",
            sigma=float(sigma),
            valid_splats=int(finite.sum()),
            center_only_splats=int((~finite).sum()),
            capped_splats=int(np.any(logs[finite] > np.log(cap), axis=1).sum()),
            maximum_principal_scale=cap,
            full_cloud_median_spacing=spacing,
        ),
    )


def splat_occupancy(points, shapes, origin, h, grid_shape, sigma=2.5, spacing=None, geometry=None):
    """Rasterize bounded ellipsoids using the PLY's scale/rotation convention.

    Kernels are only shrunk by the density cap, never enlarged. A cap of four
    median nearest-center spacings prevents a few oversized splats from defining
    the whole solid. Original center voxels are retained for sub-voxel kernels.
    """
    spacing = center_spacing(points, h) if spacing is None else spacing
    finite, radii, inverse, metadata = (
        splat_geometry(points, shapes, h, spacing, sigma) if geometry is None else geometry
    )
    occupied = np.zeros(grid_shape, bool)
    point_ids = np.floor((points - origin) / h).astype(int)
    occupied[tuple(point_ids.T)] = True
    positions = points[finite]
    for start in range(0, len(positions), 128):
        p = positions[start : start + 128]
        extent = radii[start : start + 128]
        low = np.maximum(np.ceil((p - extent - origin) / h - 0.5).astype(int), 0)
        high = np.minimum(
            np.floor((p + extent - origin) / h - 0.5).astype(int), np.asarray(grid_shape) - 1
        )
        spans = np.maximum(high - low + 1, 0)
        counts = np.prod(spans, axis=1)
        ends = np.cumsum(counts)
        for offset_start in range(0, int(ends[-1]), 65536):
            flat = np.arange(offset_start, min(offset_start + 65536, int(ends[-1])))
            owner = np.searchsorted(ends, flat, side="right")
            offset = flat - (ends - counts)[owner]
            sy, sz = spans[owner, 1], spans[owner, 2]
            idx = low[owner] + np.column_stack(
                (offset // (sy * sz), (offset // sz) % sy, offset % sz)
            )
            delta = origin + (idx + 0.5) * h - p[owner]
            metric = np.einsum("ni,nij,nj->n", delta, inverse[start : start + 128][owner], delta)
            occupied[tuple(idx[metric <= sigma * sigma].T)] = True
    return occupied, metadata


def reconstruction_layers(spacing, h, config):
    # Close sampling gaps at a physical scale, independent of grid refinement.
    return max(config.shell_layers, int(np.ceil(0.75 * spacing / h)))


def reconstruct_interior(occupied, layers, config):
    neighborhood = np.ones((3, 3, 3), bool)
    barrier = ndi.binary_dilation(occupied, structure=neighborhood, iterations=layers)
    filled = (
        ndi.binary_fill_holes(barrier, structure=neighborhood)
        if config.reconstruction_mode == "solid"
        else barrier
    )
    safe = ndi.binary_erosion(filled, structure=neighborhood, iterations=layers + 1)
    labels, count = ndi.label(safe)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    keep = sizes >= config.min_component_voxels
    keep[0] = False
    discarded = int(safe.sum() - sizes[keep].sum())
    return (
        keep[labels],
        discarded,
        int(keep.sum()),
        dict(
            mode=config.reconstruction_mode,
            occupied_voxels=int(occupied.sum()),
            barrier_voxels=int(barrier.sum()),
            enclosed_empty_voxels=int(filled.sum() - barrier.sum()),
            reconstruction_layers=layers,
            safe_voxels=int(sizes[keep].sum()),
        ),
    )


def preflight_grid(points, config, spacing, splats=None):
    """Check support and padded envelope bounds before normals and dense allocations."""
    h = float(np.ptp(points, axis=0).max()) / config.resolution
    if not np.isfinite(h) or h <= 0:
        raise ValueError("A finite, nonzero cloud extent is required.")
    low, high = points.min(0), points.max(0)
    if splats is not None:
        valid, radii, _, _ = splat_geometry(points, splats, h, spacing, config.splat_sigma)
        if valid.any():
            low = np.minimum(low, (points[valid] - radii).min(0))
            high = np.maximum(high, (points[valid] + radii).max(0))
    padding = reconstruction_layers(spacing, h, config) + 3
    shape = np.ceil((high - low) / h).astype(int) + 2 * padding + 1
    check_grid_memory(shape, config.max_memory_mb, "support preflight")
    pad = int(np.ceil(config.approximation_distance_factor * max(h, spacing) / h)) + 1
    check_grid_memory(shape + 2 * pad, config.max_memory_mb, "envelope preflight")
    return shape


def solid_from_splats(points, shapes, config, spacing=None):
    h = float(np.ptp(points, axis=0).max()) / config.resolution
    spacing = center_spacing(points, h) if spacing is None else spacing
    geometry = splat_geometry(points, shapes, h, spacing, config.splat_sigma)
    finite, radii, _, _ = geometry
    low, high = points.min(0), points.max(0)
    if finite.any():
        low = np.minimum(low, (points[finite] - radii).min(0))
        high = np.maximum(high, (points[finite] + radii).max(0))
    layers = reconstruction_layers(spacing, h, config)
    padding = layers + 3
    origin = low - padding * h
    grid_shape = tuple(np.ceil((high - origin) / h).astype(int) + padding + 1)
    check_grid_memory(grid_shape, config.max_memory_mb, "splat support")
    occupied, metadata = splat_occupancy(
        points, shapes, origin, h, grid_shape, config.splat_sigma, spacing, geometry
    )
    safe, discarded, count, reconstruction = reconstruct_interior(occupied, layers, config)
    metadata.update(reconstruction)
    return dict(
        safe=safe,
        observed=occupied,
        origin=origin,
        voxel_size=h,
        discarded_voxels=discarded,
        source_surface_voxels=int(occupied.sum()),
        components=count,
        reconstruction=metadata,
    )


def conservative_solid(points, config=None, splats=None, spacing=None):
    """Flood only enclosed empty voxels; never fill the convex hull or silhouettes.

    A narrow dilation of occupied cells forms a sampling barrier. Exterior flood
    uses 26-connectivity, including diagonal leak paths. The filled solid is
    eroded by MORE than the dilation, retreating from its outer boundary. Inner
    noisy splat centers do not become artificial cavities. Holes larger than
    the local sampling band stay open; no global convex hull fill is performed.
    """
    if config is None:
        from .cuboids import CuboidConfig

        config = CuboidConfig()
    config.validate()
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Expected nonempty finite XYZ points.")
    h = float(np.ptp(points, axis=0).max()) / config.resolution
    if h <= 0:
        raise ValueError("The cloud must have a nonzero extent.")
    spacing = center_spacing(points, h) if spacing is None else spacing
    if splats is not None:
        return solid_from_splats(points, splats, config, spacing)
    layers = reconstruction_layers(spacing, h, config)
    padding = layers + 3
    origin = points.min(axis=0) - padding * h
    ids = np.floor((points - origin) / h).astype(int)
    shape = tuple(ids.max(axis=0) + padding + 1)
    check_grid_memory(shape, config.max_memory_mb, "point support")
    occupied = np.zeros(shape, bool)
    occupied[tuple(ids.T)] = True
    safe, discarded, count, metadata = reconstruct_interior(occupied, layers, config)
    return dict(
        safe=safe,
        observed=occupied,
        origin=origin,
        voxel_size=h,
        discarded_voxels=discarded,
        source_surface_voxels=int(occupied.sum()),
        components=count,
        reconstruction=dict(method="center_voxels", **metadata),
    )
