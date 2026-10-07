"""Memory-bounded local search grids over the unchanged global support envelope."""

import math

import numpy as np
from scipy import ndimage as ndi

from .cuboids import candidate_boxes, summed_volume, volume_sum
from .geometry import CORNER_SIGNS
from .volume import check_grid_memory


_CHUNK_CELLS = 16384
_STREAM_BYTES = 4 * 1024**2
_WORKSPACE_BYTES_PER_CELL = 112


def _array_bytes(solid):
    """Count existing array storage once, including views and cached integrals."""
    roots = {}
    for value in solid.values():
        if isinstance(value, np.ndarray):
            while isinstance(value.base, np.ndarray):
                value = value.base
            roots[id(value)] = value.nbytes
    return sum(roots.values())


def _slice(low, high):
    return tuple(slice(int(a), int(b)) for a, b in zip(low, high))


def _trim(safe, low, high):
    block = safe[_slice(low, high)]
    occupied = []
    for axis in range(3):
        positions = np.flatnonzero(np.any(block, axis=tuple(i for i in range(3) if i != axis)))
        if not len(positions):
            return None
        occupied.append((positions[0], positions[-1] + 1))
    occupied = np.asarray(occupied)
    return low + occupied[:, 0], low + occupied[:, 1]


def _layout(low, high, frame):
    if np.array_equal(frame, np.eye(3)):
        return low.copy(), high - low
    corners = (low + high) / 2 + CORNER_SIGNS * (high - low - 1) / 2
    projected = corners @ frame
    minimum = np.floor(projected.min(0)).astype(int) - 1
    shape = np.floor(projected.max(0)).astype(int) - minimum + 2
    return minimum, shape


def _plan_tiles(safe, frame, available_bytes):
    """Split before allocating: empty gaps first, then occupied spatial medians."""
    initial = _trim(safe, np.zeros(3, int), np.asarray(safe.shape))
    if initial is None:
        return []
    pending = [initial]
    tiles = []
    while pending:
        low, high = pending.pop()
        # Overlap neighboring search tiles so their boundaries do not remove
        # every useful seed. Final growth/certification uses the global solid.
        padded_low = np.maximum(low - 2, 0)
        padded_high = np.minimum(high + 2, safe.shape)
        _, shape = _layout(padded_low, padded_high, frame)
        cells = math.prod(int(n) for n in shape)
        block = safe[_slice(low, high)]
        count = int(np.count_nonzero(block))
        oversized = cells * _WORKSPACE_BYTES_PER_CELL > available_bytes
        sparse = block.size > 32768 and count < block.size * 0.12
        if not oversized and not sparse:
            tiles.append((padded_low, padded_high, count, cells))
            continue
        axis = int(np.argmax(high - low))
        if high[axis] - low[axis] <= 1:
            raise ValueError("Candidate tiles cannot fit the remaining memory budget.")
        marginal = np.sum(block, axis=tuple(i for i in range(3) if i != axis), dtype=np.int64)
        occupied = np.flatnonzero(marginal)
        gaps = np.diff(occupied)
        if len(gaps) and gaps.max() > 2:
            gap = int(np.argmax(gaps))
            split = int((occupied[gap] + occupied[gap + 1] + 1) // 2)
        else:
            split = int(np.searchsorted(np.cumsum(marginal), count / 2) + 1)
        split = int(np.clip(split, 1, len(marginal) - 1)) + low[axis]
        left_high, right_low = high.copy(), low.copy()
        left_high[axis], right_low[axis] = split, split
        for a, b in ((right_low, high), (low, left_high)):
            trimmed = _trim(safe, a, b)
            if trimmed is not None:
                pending.append(trimmed)
    return tiles


def _occupied_chunks(safe, low, high):
    """Never flatten a noncontiguous full block or materialize its N×3 centers."""
    rows = max(1, _CHUNK_CELLS // int(high[2] - low[2]))
    for x in range(int(low[0]), int(high[0])):
        for y in range(int(low[1]), int(high[1]), rows):
            slab = safe[x, y : min(y + rows, high[1]), low[2] : high[2]]
            flat = np.flatnonzero(slab)
            if len(flat):
                yield np.column_stack(
                    (
                        np.full(len(flat), x),
                        flat // slab.shape[1] + y,
                        flat % slab.shape[1] + low[2],
                    )
                )


def _rotated_tile(solid, frame, low, high, fixed_bytes):
    minimum, shape = _layout(low, high, frame)
    check_grid_memory(
        shape, solid["max_memory_mb"], "local candidate workspace",
        _WORKSPACE_BYTES_PER_CELL, fixed_bytes + _STREAM_BYTES,
    )
    h, origin, safe = solid["voxel_size"], solid["origin"], solid["safe"]
    if np.array_equal(frame, np.eye(3)):
        return safe[_slice(low, high)], origin + low * h
    proposals = np.zeros(tuple(shape), bool)
    for indices in _occupied_chunks(safe, low, high):
        projected = np.floor((indices + 0.5) @ frame).astype(int) - minimum
        proposals[tuple(projected.T)] = True
    ndi.binary_dilation(proposals, structure=np.ones((3, 3, 3), bool), output=proposals)
    result = np.zeros_like(proposals)
    half_aabb = np.abs(frame).sum(axis=1) / 2
    # Every local cell is checked against the global support integral. Tile
    # boundaries never become an alternative or weaker containment certificate.
    for start in range(0, proposals.size, _CHUNK_CELLS):
        flat = np.flatnonzero(proposals.ravel()[start : start + _CHUNK_CELLS]) + start
        indices = np.column_stack(np.unravel_index(flat, tuple(shape)))
        centers = (minimum + indices + 0.5) @ frame.T
        a = np.floor(centers - half_aabb + 1e-9).astype(int)
        b = np.ceil(centers + half_aabb - 1e-9).astype(int)
        bounded = np.all((a >= 0) & (b <= safe.shape), axis=1)
        accepted = np.zeros(len(indices), bool)
        accepted[bounded] = volume_sum(
            solid["safe_prefix"], a[bounded], b[bounded]
        ) == np.prod(b[bounded] - a[bounded], axis=1)
        result[tuple(indices[accepted].T)] = True
    return result, origin @ frame + minimum * h


def _seed_allocations(tiles, budget):
    """Share the frame budget spatially; a small component gets a seed first."""
    allocations = np.zeros(len(tiles), int)
    if len(tiles) > budget:
        allocations[np.linspace(0, len(tiles) - 1, budget, dtype=int)] = 1
        return allocations
    allocations[:] = 1
    remaining = budget - len(tiles)
    if remaining:
        weights = np.sqrt([tile[2] for tile in tiles])
        extra = remaining * weights / weights.sum()
        allocations += np.floor(extra).astype(int)
        leftover = budget - int(allocations.sum())
        allocations[np.argsort(-(extra % 1), kind="stable")[:leftover]] += 1
    return allocations


def ensure_support_prefix(solid, config, external_bytes=0):
    """Build the certification accelerator once, with the same memory guard on resume."""
    if "safe_prefix" not in solid:
        safe = solid["safe"]
        prefix_shape = np.asarray(safe.shape) + 1
        dtype_bytes = 4 if safe.size <= np.iinfo(np.int32).max else 8
        check_grid_memory(
            prefix_shape, config.max_memory_mb, "support integral", dtype_bytes * 2,
            _array_bytes(solid) + external_bytes + _STREAM_BYTES,
        )
        solid["safe_prefix"] = summed_volume(safe)


def generate_candidates(solid, frames, config, rng, surface_points, progress, point_bytes=0):
    """Keep compact search grids; automatically tile sparse or oversized ones.

    The support envelope and voxel size are unchanged. This bounds candidate
    workspaces, not total process RSS or the dense reconstruction stage.
    """
    safe = solid["safe"]
    maximum = config.max_memory_mb * 1024**2
    solid["max_memory_mb"] = config.max_memory_mb
    # Account for point projections and masks during each tile's surface seeds.
    external = int(point_bytes) + surface_points.nbytes * 4
    ensure_support_prefix(solid, config, external)
    output = []
    seen = set()
    metadata = dict(method="adaptive local grids", frames=[], maximum_grid_cells=0)
    for frame_id, frame in enumerate(frames):
        # Include accumulated proposals and the maximum temporary proposal list
        # for this frame (two seed sets, three radii, six growth orders).
        fixed = _array_bytes(solid) + external + (len(output) + 36 * config.seeds_per_frame) * 1024
        available = maximum - fixed - _STREAM_BYTES
        if available <= 0:
            raise ValueError("No candidate workspace remains within --max-memory-mb.")
        tiles = _plan_tiles(safe, frame, available)
        if not tiles:
            continue
        allocations = _seed_allocations(tiles, config.seeds_per_frame)
        start = len(output)
        for (low, high, _, cells), budget in zip(tiles, allocations):
            if not budget:
                continue
            mask, base = _rotated_tile(solid, frame, low, high, fixed)
            world_low = solid["origin"] + low * solid["voxel_size"]
            world_high = solid["origin"] + high * solid["voxel_size"]
            local_points = surface_points[
                np.all((surface_points >= world_low) & (surface_points <= world_high), axis=1)
            ]
            proposed = candidate_boxes(
                mask, base, frame, solid["voxel_size"], int(budget), rng, local_points,
                config.max_memory_mb, fixed + _STREAM_BYTES,
            )
            for box in proposed:
                key = np.r_[box["center"], box["dimensions"], frame.ravel()].tobytes()
                if key not in seen:
                    output.append(box)
                    seen.add(key)
            metadata["maximum_grid_cells"] = max(metadata["maximum_grid_cells"], cells)
            del mask, proposed
        metadata["frames"].append(dict(
            tiles=len(tiles), searched_tiles=int(np.count_nonzero(allocations)),
            seed_budget=int(allocations.sum()), proposals=len(output) - start,
        ))
        progress(
            f"Frame {frame_id + 1}/{len(frames)}: {len(output) - start} proposals "
            f"from {np.count_nonzero(allocations)}/{len(tiles)} local grids"
        )
    solid["candidate_search"] = metadata
    return output
