"""Construct, select and certify cuboids in an immutable insertion order."""

from time import perf_counter
from dataclasses import replace

import numpy as np

from .approximation import (
    approximation_solid,
    evaluation_tolerance,
    surface_metrics,
    surface_support_reached,
)
from .geometry import box_surface_distance
from .candidate_grids import ensure_support_prefix, generate_candidates
from .cuboids import (
    CuboidConfig,
    certify_box,
    distance_to_box,
    region_frames,
    select_incremental_boxes,
)
from .volume import CoverageNotReachedError, conservative_solid, center_spacing


def fit_cuboids(
    full_points,
    regions,
    spacing,
    config=None,
    splats=None,
    progress=print,
    reference=(),
    cached_candidates=None,
    reference_local=None,
    full_spacing=None,
    checkpoint=None,
    cached_solid=None,
    cached_frames=None,
):
    config = config or CuboidConfig()
    config.validate()
    start = perf_counter()
    offset = full_points.min(axis=0)
    scale = float(np.ptp(full_points, axis=0).max())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("A finite, nonzero cloud extent is required.")
    full_points = (full_points - offset) / scale
    regions = dict(regions, points=(regions["points"] - offset) / scale)
    spacing /= scale
    if splats is not None:
        splats = dict(splats, log_scales=splats["log_scales"] - np.log(scale))
    config = replace(
        config,
        point_tolerance=(
            None if config.point_tolerance is None else config.point_tolerance / scale
        ),
    )

    def local_box(b):
        return dict(
            b,
            center=(b["center"] - offset) / scale,
            dimensions=b["dimensions"] / scale,
            volume=b["volume"] / scale**3,
        )

    def world_box(b):
        return dict(
            b,
            center=b["center"] * scale + offset,
            dimensions=b["dimensions"] * scale,
            volume=b["volume"] * scale**3,
        )

    full_spacing = (
        center_spacing(full_points, 1 / config.resolution)
        if full_spacing is None
        else full_spacing / scale
    )
    solid = cached_solid
    if solid is None:
        solid = approximation_solid(
            conservative_solid(full_points, config, splats, full_spacing),
            full_points,
            config,
            spacing=full_spacing,
        )
    else:
        ensure_support_prefix(
            solid, config, external_bytes=full_points.nbytes + regions["points"].nbytes * 4
        )
    solid["max_memory_mb"] = config.max_memory_mb
    tolerance = evaluation_tolerance(config, full_spacing, 1.0)
    progress(
        f"Approximation distance <= {solid['approximation']['distance_limit'] * scale:.6g}; point tolerance {tolerance * scale:.6g}"
    )
    # Observed center cells are retained by construction; no voxel KD-tree needed.
    ids = np.floor((full_points - solid["origin"]) / solid["voxel_size"]).astype(int)
    if not solid["safe"][tuple(ids.T)].all():
        raise RuntimeError("The envelope lost an observed center cell.")
    h = solid["voxel_size"]
    if cached_frames is None:
        frames, _, _ = region_frames(
            regions["points"], regions["labels"], regions["assignment_kind"], spacing, config
        )
    else:
        frames = cached_frames
    from .surface_proposals import SurfacePanelProposer

    panel_proposer = None

    def surface_panels(points):
        nonlocal panel_proposer
        if config.target_surface_support == 0:
            return []
        if panel_proposer is None:
            panel_proposer = SurfacePanelProposer(full_points, frames, tolerance, solid, config)
        return panel_proposer(points)

    rng = np.random.default_rng(config.seed)
    if cached_candidates is None:
        candidates = []
        single_box = False
        # Include tight source bounds before considering maximally grown grid boxes.
        # This lets a single admissible box win for a cuboid or two close sheets,
        # even when extra orientations introduce distracting maximal candidates.
        for frame in frames:
            local = full_points @ frame
            low, high = local.min(0), local.max(0)
            dimensions = np.maximum(high - low, h * 1e-4)
            tight = dict(
                center=((low + high) / 2) @ frame.T,
                dimensions=dimensions,
                rotation=frame.copy(),
                volume=float(np.prod(dimensions)),
            )
            if certify_box(tight, solid):
                candidates.append(tight)
                if (
                    not reference
                    and not reference_local
                    and (
                        config.reconstruction_mode != "surface"
                        or np.min(dimensions) <= 2 * tolerance
                    )
                    and np.max(box_surface_distance(full_points, tight)) <= h * 1e-3
                ):
                    support_target = max(0.95, config.target_surface_support)
                    quality = surface_metrics(
                        [tight],
                        full_points,
                        tolerance,
                        target_support=support_target,
                        support_max_evaluations=config.surface_max_evaluations,
                    )
                    if quality[
                        "source_surface_coverage"
                    ] >= config.target_coverage and surface_support_reached(
                        quality, support_target
                    ):
                        candidates = [tight]
                        single_box = True
                        progress("Certified single-box surface fit; skipping orientation grids")
                        break
        if not single_box:
            candidates.extend(
                generate_candidates(
                    solid,
                    frames,
                    config,
                    rng,
                    regions["points"],
                    progress,
                    point_bytes=full_points.nbytes,
                )
            )
            panels = surface_panels(full_points)
            candidates.extend(panels)
            progress(f"Observed surface panels: {len(panels)} proposals")
    else:
        candidates = list(cached_candidates)
        progress(f"Reusing {len(candidates)} candidate boxes")

    def propose(residual):
        from scipy.spatial import cKDTree

        if not len(residual):
            return []
        cells = np.unique(np.floor((residual - solid["origin"]) / h).astype(int), axis=0)
        # Uniform spatial cells, not raw acquisition populations.
        if len(cells) > config.seeds_per_frame:
            cells = cells[np.linspace(0, len(cells) - 1, config.seeds_per_frame, dtype=int)]
        tree = cKDTree(residual)
        output = surface_panels(residual)
        for cell in cells:
            center = solid["origin"] + (cell + 0.5) * h
            output.append(
                dict(
                    center=center,
                    dimensions=np.full(3, h),
                    rotation=np.eye(3),
                    volume=h**3,
                    proposal_kind="local_cell",
                )
            )
            ids = tree.query_ball_point(center, max(4 * h, 4 * full_spacing))
            if len(ids) < 3:
                continue
            patch = residual[ids]
            _, axes = np.linalg.eigh(np.cov((patch - center).T))
            if np.linalg.det(axes) < 0:
                axes[:, 2] *= -1
            for frame in (np.eye(3), axes):
                projected = (patch - center) @ frame
                low, high = projected.min(0), projected.max(0)
                dimensions = np.maximum(high - low, h * 1e-4)
                box = dict(
                    center=center + ((low + high) / 2) @ frame.T,
                    dimensions=dimensions,
                    rotation=frame,
                    volume=float(np.prod(dimensions)),
                    proposal_kind="local_fit",
                )
                if certify_box(box, solid):
                    output.append(box)
        return output

    def save_checkpoint(boxes, pool):
        if checkpoint is not None:
            checkpoint(boxes, pool, solid, frames, offset, scale)

    save_checkpoint(reference_local or [local_box(b) for b in reference], candidates)
    export_solid = dict(
        solid, origin=solid["origin"] * scale + offset, voxel_size=solid["voxel_size"] * scale
    )
    boxes, curve, _, count, selection = select_incremental_boxes(
        candidates,
        ([local_box(b) for b in reference] if reference_local is None else reference_local),
        full_points,
        tolerance,
        config,
        solid,
        progress,
        propose=propose,
        checkpoint=save_checkpoint,
        export_certificate=lambda b: certify_box(world_box(b), export_solid, strict=True),
    )
    if not boxes:
        raise CoverageNotReachedError(
            "No admissible candidate covers retained source points.", reason="no_candidates"
        )
    distances = np.full(len(full_points), np.inf)
    for box in boxes:
        distances = np.minimum(distances, distance_to_box(full_points, box))
    solid["origin"] = solid["origin"] * scale + offset
    solid["voxel_size"] *= scale
    for key in ("distance_limit", "full_cloud_median_spacing"):
        solid["approximation"][key] *= scale
    if "maximum_principal_scale" in solid["reconstruction"]:
        solid["reconstruction"]["maximum_principal_scale"] *= scale
        solid["reconstruction"]["full_cloud_median_spacing"] *= scale
    if selection.get("surface_panel_thickness_limit") is not None:
        selection["surface_panel_thickness_limit"] *= scale
    if selection.get("surface_support_bounds") is not None:
        selection["surface_support_bounds"] = dict(selection["surface_support_bounds"])
        selection["surface_support_bounds"]["total_area"] *= scale**2
    world_boxes = [world_box(b) for b in boxes]
    # Preserve the supplied prefix bit for bit, including world-coordinate roundoff.
    world_boxes[: len(reference)] = reference
    for index, box in enumerate(world_boxes):
        if not certify_box(box, solid, strict=True):
            raise RuntimeError(
                f"Export-coordinate containment failed for cuboid {index}; "
                "recenter the source to reduce floating-point roundoff."
            )
    return dict(
        boxes=world_boxes,
        local_boxes=boxes,
        local_candidates=candidates,
        normalization_origin=offset,
        normalization_scale=scale,
        solid=solid,
        frames=frames,
        coverage_curve=curve,
        selection=selection,
        candidate_count=count,
        surface_tolerance=tolerance * scale,
        full_distances=distances * scale,
        seconds=perf_counter() - start,
    )
