"""Incremental cuboid selection with bounded, penalized approximation."""

from dataclasses import dataclass
import itertools
import numpy as np
from scipy import ndimage as ndi
from plyfile import PlyData, PlyElement

from .geometry import BOX_QUADS, BOX_TRIANGLES, CORNER_SIGNS


@dataclass
class CuboidConfig:
    approximation_distance_factor: float = 2.0
    approximation_exterior_penalty: float = 2.0
    approximation_detail_factor: float = 2.0
    splat_sigma: float = 2.5
    resolution: int = 160
    shell_layers: int = 1
    max_frames: int = 32
    seeds_per_frame: int = 96
    target_coverage: float = 0.999
    target_surface_support: float = 0.95
    target_component_coverage: float = 0.0
    surface_max_evaluations: int = 65536
    residual_interval: int = 4
    max_cuboids: int = 128
    min_component_voxels: int = 16
    frame_angle_degrees: float = 2.0
    seed: int = 42
    reconstruction_mode: str = "solid"
    max_memory_mb: int = 2048
    point_tolerance: float | None = None

    def validate(self):
        if self.reconstruction_mode not in ("solid", "surface"):
            raise ValueError("reconstruction_mode must be solid or surface.")
        if self.point_tolerance is not None and (
            not np.isfinite(self.point_tolerance) or self.point_tolerance <= 0
        ):
            raise ValueError("point_tolerance must be finite and positive.")
        for name in ("approximation_distance_factor", "approximation_detail_factor"):
            if not np.isfinite(getattr(self, name)) or not 0 < getattr(self, name) <= 8:
                raise ValueError(f"{name} must be finite, positive and at most eight.")
        if (
            not np.isfinite(self.approximation_exterior_penalty)
            or self.approximation_exterior_penalty < 0
        ):
            raise ValueError("The exterior penalty must be finite and nonnegative.")
        if not np.isfinite(self.splat_sigma) or not 1 <= self.splat_sigma <= 3:
            raise ValueError("Splat support must be between one and three standard deviations.")
        for name in (
            "resolution",
            "residual_interval",
            "shell_layers",
            "max_frames",
            "seeds_per_frame",
            "max_cuboids",
            "min_component_voxels",
            "seed",
            "max_memory_mb",
            "surface_max_evaluations",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        if not 0 <= self.target_surface_support <= 1 or self.residual_interval < 1:
            raise ValueError("Invalid surface target or residual interval.")
        if not 0 <= self.target_component_coverage <= 1:
            raise ValueError("target_component_coverage must be in [0, 1].")
        if self.resolution < 32 or self.resolution > 768:
            raise ValueError("Resolution must be between 32 and 768.")
        if self.max_memory_mb < 64:
            raise ValueError("max_memory_mb must be at least 64.")
        if self.surface_max_evaluations < 1:
            raise ValueError("surface_max_evaluations must be positive.")
        if self.shell_layers > 8:
            raise ValueError("The local sampling barrier is limited to eight voxel layers.")
        if (
            min(
                self.shell_layers,
                self.max_frames,
                self.seeds_per_frame,
                self.max_cuboids,
                self.min_component_voxels,
            )
            < 1
        ):
            raise ValueError("Search budgets must be positive.")
        if not 0 < self.target_coverage <= 1 or not 0 < self.frame_angle_degrees < 45:
            raise ValueError("Invalid coverage or frame angle.")


def summed_volume(mask):
    dtype = np.int32 if mask.size <= np.iinfo(np.int32).max else np.int64
    result = np.pad(np.asarray(mask, dtype), ((1, 0), (1, 0), (1, 0)))
    for axis in range(3):
        np.cumsum(result, axis=axis, dtype=dtype, out=result)
    return result


def volume_sum(prefix, low, high):
    """Vectorized half-open integer box queries."""
    low, high = np.asarray(low), np.asarray(high)
    total = np.zeros(low.shape[:-1], np.int64)
    for bits in itertools.product((0, 1), repeat=3):
        idx = np.where(bits, high, low)
        total += (-1 if (3 - sum(bits)) % 2 else 1) * prefix[tuple(idx.T)]
    return total


def region_frames(points, labels, assignment_kind, spacing, config):
    """Propose orientations from dominant support planes, regions and PCA."""
    if config.max_frames == 1:
        return np.eye(3)[None], [[]], []
    from .orientations import equivalent_frame, orientation_proposals, proper_frame

    faces = []
    for label in np.unique(labels[labels >= 0]):
        sample = points[(labels == label) & (assignment_kind == 1)]
        if len(sample) < 12:
            continue
        center = sample.mean(axis=0)
        values, axes = np.linalg.eigh(np.cov(sample.T))
        if np.sqrt(max(values[0], 0)) > 2 * spacing:
            continue
        normal = axes[:, 0]
        major = axes[:, 2]
        frame = np.column_stack([normal, major, np.cross(normal, major)])
        faces.append(
            dict(
                region_id=int(label),
                points=len(sample),
                center=center,
                normal=normal,
                frame=frame,
                rms=float(np.sqrt(max(values[0], 0))),
            )
        )
    frames, supports = [np.eye(3)], [[]]
    cosine = np.cos(np.deg2rad(config.frame_angle_degrees))
    for _, frame, region_ids in orientation_proposals(points, faces):
        if len(frames) >= config.max_frames:
            break
        frame = proper_frame(frame)
        if any(equivalent_frame(old, frame, cosine) for old in frames):
            continue
        frames.append(frame)
        supports.append(region_ids)
    return np.array(frames), supports, faces


def candidate_boxes(
    mask,
    origin,
    frame,
    h,
    seed_count,
    rng,
    surface_points=None,
    max_memory_mb=2048,
    fixed_memory_bytes=0,
):
    if not mask.any():
        return []
    from .volume import check_grid_memory

    check_grid_memory(
        mask.shape, max_memory_mb, "candidate grid", 12, fixed_memory_bytes + int(mask.sum()) * 80
    )
    clearance = ndi.distance_transform_cdt(mask, metric="chessboard")
    ids = np.argwhere(mask)
    # Spatially diverse seeds, retaining high-clearance locations in each bin.
    quality = clearance[tuple(ids.T)]
    order = np.argsort(-quality, kind="stable")
    bin_size = max(2.0, (len(ids) / seed_count) ** (1 / 3))
    bins = np.floor(ids[order] / bin_size).astype(int)
    from .cloud import cell_keys

    _, positions = np.unique(cell_keys(bins), return_index=True)
    order = order[np.sort(positions)]
    if len(order) > seed_count:
        first = order[: seed_count // 2]
        rest = rng.choice(order[seed_count // 2 :], seed_count - len(first), replace=False)
        order = np.r_[first, rest]
    seeds = ids[order]
    if surface_points is not None:
        # Thin sheets and small appendages may have no deep-interior seed.
        surface_ids = np.unique(np.floor((surface_points @ frame - origin) / h).astype(int), axis=0)
        valid = ((surface_ids >= 0) & (surface_ids < mask.shape)).all(axis=1)
        surface_ids = surface_ids[valid]
        surface_ids = surface_ids[mask[tuple(surface_ids.T)]]
        if len(surface_ids) > seed_count:
            surface_ids = surface_ids[rng.choice(len(surface_ids), seed_count, replace=False)]
        seeds = np.unique(np.vstack([seeds, surface_ids]), axis=0)
    radii = np.maximum(clearance[tuple(seeds.T)] - 1, 0)
    # Narrow starts can extend down a tapering limb where a maximal initial cube
    # would stop immediately. Thick starts keep broad boxes on the main body.
    seeds = np.tile(seeds, (3, 1))
    radii = np.concatenate([np.zeros_like(radii), radii // 2, radii])
    initial_low = seeds - radii[:, None]
    initial_high = seeds + radii[:, None] + 1
    prefix = summed_volume(mask)
    output = []
    # Axis order changes the admissible aspect ratio. Start with thick cubes,
    # then maximize face extents, instead of growing needles from lone points.
    for order in itertools.permutations(range(3)):
        low, high = initial_low.copy(), initial_high.copy()
        for axis in order:
            for direction in (-1, 1):
                left = np.zeros(len(low), int)
                right = (
                    low[:, axis] if direction == -1 else mask.shape[axis] - high[:, axis]
                ).copy()
                while np.any(left < right):
                    middle = (left + right + 1) // 2
                    a, b = low.copy(), high.copy()
                    if direction == -1:
                        a[:, axis] -= middle
                    else:
                        b[:, axis] += middle
                    valid = volume_sum(prefix, a, b) == np.prod(b - a, axis=1)
                    left = np.where(valid, middle, left)
                    right = np.where(valid, right, middle - 1)
                if direction == -1:
                    low[:, axis] -= left
                else:
                    high[:, axis] += left
        for a, b in zip(low, high):
            dimensions = (b - a) * h
            center = (origin + (a + b) * h / 2) @ frame.T
            output.append(
                dict(
                    center=center,
                    dimensions=dimensions,
                    rotation=frame,
                    volume=float(np.prod(dimensions)),
                )
            )
    return output


def box_membership(points, box, tolerance=0.0):
    return np.all(
        np.abs((points - box["center"]) @ box["rotation"]) <= box["dimensions"] / 2 + tolerance,
        axis=1,
    )


def distance_to_box(points, box):
    return np.linalg.norm(
        np.maximum(np.abs((points - box["center"]) @ box["rotation"]) - box["dimensions"] / 2, 0),
        axis=1,
    )


def _sat_overlaps_voxels(centers, box, h, epsilon=0.0):
    """Exact separating-axis OBB/AABB intersection, not vertex sampling."""
    if not len(centers):
        return np.zeros(0, bool)
    frame, half = box["rotation"], box["dimensions"] / 2
    delta = centers - box["center"]
    axes = [*np.eye(3), *frame.T]
    axes += [np.cross(a, b) for a in np.eye(3) for b in frame.T]
    overlapping = np.ones(len(centers), bool)
    for axis in axes:
        length = np.linalg.norm(axis)
        if length < 1e-10:
            continue
        axis = axis / length
        radius = np.sum(np.abs(frame.T @ axis) * half) + 0.5 * h * np.abs(axis).sum()
        # Face-only contact is allowed; epsilon is a scale-relative roundoff tolerance.
        overlapping &= np.abs(delta @ axis) < radius - epsilon
        if not overlapping.any():
            break
    return overlapping


def certify_box(box, solid, strict=False):
    """Enumerate EVERY excluded reference voxel in the box AABB, then apply SAT."""
    h, origin, safe = solid["voxel_size"], solid["origin"], solid["safe"]
    half_aabb = np.abs(box["rotation"]) @ (box["dimensions"] / 2)
    rounding = 0 if strict else 1e-9
    low = np.floor((box["center"] - half_aabb - origin) / h + rounding).astype(int)
    high = np.ceil((box["center"] + half_aabb - origin) / h - rounding).astype(int)
    if np.any(low < 0) or np.any(high > safe.shape) or np.any(high <= low):
        return False
    prefix = solid.get("safe_prefix")
    if prefix is not None and volume_sum(prefix, low[None], high[None])[0] == np.prod(high - low):
        return True
    for x in range(low[0], high[0], 8):
        a = np.array([x, low[1], low[2]])
        b = np.array([min(x + 8, high[0]), high[1], high[2]])
        region = safe[tuple(slice(i, j) for i, j in zip(a, b))]
        outside = np.argwhere(~region) + a
        centers = origin + (outside + 0.5) * h
        if _sat_overlaps_voxels(centers, box, h, epsilon=0 if strict else h * 1e-8).any():
            return False
    return True


def grow_certified(box, solid, score=None):
    """Expand continuous cuboid faces using whole-volume SAT, without the local-grid margin."""
    result = dict(box)
    result["center"] = box["center"].copy()
    result["dimensions"] = box["dimensions"].copy()
    for axis in np.argsort(-result["dimensions"]):
        for sign in (-1, 1):
            low, high = 0.0, solid["voxel_size"] * max(solid["safe"].shape)
            for _ in range(12):
                distance = (low + high) / 2
                trial = dict(result)
                trial["center"] = (
                    result["center"] + sign * distance / 2 * result["rotation"][:, axis]
                )
                trial["dimensions"] = result["dimensions"].copy()
                trial["dimensions"][axis] += distance
                if certify_box(trial, solid):
                    low = distance
                else:
                    high = distance
            distance = low
            if score is not None and low > 0:
                # All intermediate expansions remain inside the certified maximum.
                choices = []
                for fraction in (0, 0.25, 0.5, 0.75, 1):
                    trial = dict(
                        result,
                        center=result["center"].copy(),
                        dimensions=result["dimensions"].copy(),
                    )
                    trial["center"] += sign * low * fraction / 2 * result["rotation"][:, axis]
                    trial["dimensions"][axis] += low * fraction
                    trial["volume"] = float(np.prod(trial["dimensions"]))
                    choices.append(score(trial))
                distance = low * (0, 0.25, 0.5, 0.75, 1)[int(np.argmax(choices))]
            result["center"] += sign * distance / 2 * result["rotation"][:, axis]
            result["dimensions"][axis] += distance
    result["volume"] = float(np.prod(result["dimensions"]))
    result["expanded"] = True
    return result


def refine_box(box, points, solid, score):
    """Fit, shrink, translate and locally rotate a proposal before it is committed."""
    if not len(points):
        return box
    result, best = box, score(box)
    minimum = solid["voxel_size"] * 1e-4

    def consider(frame, low, high):
        nonlocal result, best
        dimensions = np.maximum(high - low, minimum)
        trial = dict(
            center=((low + high) / 2) @ frame.T,
            dimensions=dimensions,
            rotation=frame.copy(),
            volume=float(np.prod(dimensions)),
        )
        if not certify_box(trial, solid):
            return
        value = score(trial)
        if value > best + max(1e-12, abs(best) * 1e-10):
            result, best = trial, value

    # Use the proposal center as a nearby origin for stable projected coordinates.
    origin = box["center"].copy()
    shifted = points - origin

    def fit_frame(frame):
        local = shifted @ frame
        # A small adjacent surface can contaminate several proposal bounds at
        # once. Include the same 5% trim available to individual face moves:
        # those moves alone cannot escape a zero-score burial plateau when
        # multiple faces must be corrected together. Certification and the
        # unchanged full-cloud objective still decide whether to keep the fit.
        for q in (0, 0.005, 0.02, 0.05):
            low, high = np.quantile(local, [q, 1 - q], axis=0)
            consider(frame, low + origin @ frame, high + origin @ frame)

    fit_frame(box["rotation"])
    # Independent face moves also improve proposals containing multiple surfaces.
    for _ in range(2):
        frame = result["rotation"]
        local = shifted @ frame + origin @ frame
        bounds = np.array(
            [
                result["center"] @ frame - result["dimensions"] / 2,
                result["center"] @ frame + result["dimensions"] / 2,
            ]
        )
        for axis in range(3):
            for side, quantiles in ((0, (0, 0.01, 0.05)), (1, (1, 0.99, 0.95))):
                for q in quantiles:
                    trial = bounds.copy()
                    trial[side, axis] = np.quantile(local[:, axis], q)
                    if np.all(trial[1] > trial[0]):
                        consider(frame, *trial)
        # Small local rotations; the exact support score chooses whether to keep them.
    from scipy.spatial.transform import Rotation

    frame = result["rotation"].copy()
    for axis in range(3):
        for angle in (-2.0, 2.0):
            vector = np.eye(3)[axis] * np.deg2rad(angle)
            fit_frame(frame @ Rotation.from_rotvec(vector).as_matrix())
    return dict(result, refined=True)


def box_corners(box):
    return (CORNER_SIGNS * box["dimensions"] / 2) @ box["rotation"].T + box["center"]


def select_incremental_boxes(
    candidates,
    reference,
    points,
    tolerance,
    config,
    solid,
    progress=print,
    propose=None,
    checkpoint=None,
    export_certificate=None,
):
    """Append certified boxes using the shared volume/surface selection objective."""
    from .selection import select_incremental_boxes as select

    return select(
        candidates, reference, points, tolerance, config, solid, progress,
        propose=propose, checkpoint=checkpoint, export_certificate=export_certificate,
    )


def write_cuboid_mesh(output, boxes, colors):
    vertices = np.concatenate([box_corners(b) for b in boxes])
    vertex = np.empty(
        len(vertices),
        dtype=[
            ("x", "f8"),
            ("y", "f8"),
            ("z", "f8"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("cuboid_id", "i4"),
        ],
    )
    for i, key in enumerate("xyz"):
        vertex[key] = vertices[:, i]
    for i, key in enumerate(("red", "green", "blue")):
        vertex[key] = np.repeat(np.rint(colors[:, i] * 255).astype(np.uint8), 8)
    vertex["cuboid_id"] = np.repeat(np.arange(len(boxes)), 8)
    triangles = np.concatenate([BOX_TRIANGLES + 8 * i for i in range(len(boxes))])
    face = np.empty(len(triangles), dtype=[("vertex_indices", "i4", (3,)), ("cuboid_id", "i4")])
    face["vertex_indices"], face["cuboid_id"] = triangles, np.repeat(np.arange(len(boxes)), 12)
    PlyData(
        [PlyElement.describe(vertex, "vertex"), PlyElement.describe(face, "face")],
        text=False,
        comments=["See report.json for approximation bounds; overlaps allowed"],
    ).write(str(output / "cuboids.ply"))
    with (
        (output / "cuboids.obj").open("w", encoding="utf-8") as obj,
        (output / "cuboids.mtl").open("w", encoding="utf-8") as mtl,
    ):
        obj.write("mtllib cuboids.mtl\n")
        for p in vertices:
            obj.write("v " + " ".join(f"{v:.17g}" for v in p) + "\n")
        for i, color in enumerate(colors):
            mtl.write(f"newmtl cuboid_{i}\nKd " + " ".join(f"{v:.6f}" for v in color) + "\n\n")
            obj.write(f"o cuboid_{i}\nusemtl cuboid_{i}\n")
            for quad in BOX_QUADS + 8 * i + 1:
                obj.write("f " + " ".join(map(str, quad)) + "\n")
