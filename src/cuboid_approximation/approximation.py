"""Bounded, penalized fitting support; no object names or anatomical assumptions."""

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from .parallel import current_workers

from .volume import center_spacing, check_grid_memory


def evaluation_tolerance(config, full_spacing, extent):
    """Use acquisition spacing, independently of reconstruction resolution.

    An explicit tolerance remains in the caller's coordinate units. The tiny
    extent-relative floor only avoids zero/underflow for degenerate spacing;
    voxel width is deliberately absent so grid refinement cannot relax a test.
    """
    if config.point_tolerance is not None:
        return float(config.point_tolerance)
    if not np.isfinite(full_spacing) or full_spacing < 0:
        raise ValueError("Evaluation spacing must be finite and nonnegative.")
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("Evaluation extent must be finite and positive.")
    numerical_floor = 64 * np.finfo(float).eps * extent
    return float(config.approximation_detail_factor * max(full_spacing, numerical_floor))


def approximation_solid(solid, points, config, spacing=None):
    """Keep observed thin sheets, then allow a bounded Euclidean error band.

    The reference is the inferred interior UNION observed splat/center cells.
    It is deliberately distinct from the admissible error envelope. A complete
    envelope cell is within radius of a translated, same-size reference cell;
    subsequent whole-OBB SAT certification therefore bounds the entire box.
    No convex hull or global filling of open sheets is performed.
    """
    h = float(solid["voxel_size"])
    spacing = center_spacing(points, h) if spacing is None else spacing
    radius = config.approximation_distance_factor * max(h, spacing)
    pad = int(np.ceil(radius / h)) + 1
    check_grid_memory(
        np.array(solid["safe"].shape) + 2 * pad, config.max_memory_mb, "approximation envelope"
    )
    reference = np.pad(solid["safe"] | solid["observed"], pad)
    conservative = np.pad(solid["safe"], pad)
    distance = ndi.distance_transform_edt(~reference)
    allowed = distance <= radius / h + 1e-7
    del distance
    _, count = ndi.label(allowed)
    return dict(
        solid,
        safe=allowed,
        reference=reference,
        conservative=conservative,
        origin=solid["origin"] - pad * h,
        components=int(count),
        max_memory_mb=config.max_memory_mb,
        approximation=dict(
            mode="approximate",
            distance_limit=float(radius),
            full_cloud_median_spacing=spacing,
            reference_voxels=int(reference.sum()),
            conservative_voxels=int(conservative.sum()),
            envelope_voxels=int(allowed.sum()),
            reference="inferred conservative interior union observed splat/center voxels",
            certificate="every cuboid is contained in the bounded error envelope",
            exterior_penalty=config.approximation_exterior_penalty,
        ),
    )


def quadrature(box, h, max_axis=12):
    """Seeded, jittered strata avoid aliasing against the reference voxel lattice."""
    counts = np.maximum(2, np.minimum(max_axis, np.ceil(box["dimensions"] / h).astype(int)))
    axes = [np.arange(n) for n in counts]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    grid = (grid + np.random.default_rng(0).random(grid.shape)) / counts - 0.5
    return (grid * box["dimensions"]) @ box["rotation"].T + box["center"]


def exterior_fraction(box, solid, max_axis=12):
    """Volume quadrature, NOT a containment certificate or an exact volume."""
    if np.all(np.max(np.abs(box["rotation"]), axis=0) > 1 - 1e-12):
        from .cuboids import CORNER_SIGNS, summed_volume

        if "_reference_prefix" not in solid:
            solid["_reference_prefix"] = summed_volume(solid["reference"])
        prefix = solid["_reference_prefix"]
        half = np.abs(box["rotation"]) @ (box["dimensions"] / 2)
        corners = (box["center"] + CORNER_SIGNS * half - solid["origin"]) / solid["voxel_size"]
        corners = np.clip(corners, 0, np.array(solid["reference"].shape))
        low = np.floor(corners).astype(int)
        fraction = corners - low
        integral = np.zeros(8)
        for bits in ((CORNER_SIGNS + 1) / 2).astype(int):
            idx = np.minimum(low + bits, np.array(prefix.shape) - 1)
            weight = np.prod(np.where(bits, fraction, 1 - fraction), axis=1)
            integral += weight * prefix[tuple(idx.T)]
        inside_volume = np.dot(np.prod(CORNER_SIGNS, axis=1), integral) * solid["voxel_size"] ** 3
        return float(np.clip(1 - inside_volume / np.prod(box["dimensions"]), 0, 1))
    points = quadrature(box, solid["voxel_size"], max_axis)
    ids = np.floor((points - solid["origin"]) / solid["voxel_size"]).astype(int)
    bounded = ((ids >= 0) & (ids < solid["reference"].shape)).all(axis=1)
    inside = np.zeros(len(ids), bool)
    inside[bounded] = solid["reference"][tuple(ids[bounded].T)]
    return float(1 - inside.mean())


def evaluation_cells(points, tolerance, *, origin=None, cell_width_factor=2.0):
    """Quantize the fixed evaluation grid consistently before/after scaling.

    Division roundoff can place an exact boundary immediately below an integer,
    changing both density weights and connectivity after normalization. Snap only
    roundoff-sized differences to that integer before flooring. Process one axis
    at a time to avoid several full-cloud coordinate temporaries.
    """
    points = np.asarray(points)
    cells = np.empty(points.shape, dtype=np.int64)
    if not len(points):
        return cells
    origin = points.min(axis=0) if origin is None else np.asarray(origin)
    for axis in range(points.shape[1]):
        quotient = (points[:, axis] - origin[axis]) / (cell_width_factor * tolerance)
        nearest = np.rint(quotient)
        roundoff = 32 * np.finfo(float).eps * np.maximum(1, np.abs(quotient))
        snap = np.abs(quotient - nearest) <= roundoff
        quotient[snap] = nearest[snap]
        cells[:, axis] = np.floor(quotient).astype(np.int64)
    return cells


def spatial_weights(points, tolerance):
    """Reduce density bias: cap the influence of dense patches and isolated points."""
    if not len(points):
        return np.empty(0)
    cells = evaluation_cells(points, tolerance)
    _, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights / weights.mean()


def spatial_components(points, tolerance):
    """Per-observation connected-component labels at the fixed evaluation scale."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if not len(points):
        return np.empty(0, dtype=np.int64)
    cells, inverse = np.unique(
        evaluation_cells(points, tolerance),
        axis=0,
        return_inverse=True,
    )
    pairs = cKDTree(cells).query_pairs(np.sqrt(3) + 1e-8, output_type="ndarray")
    graph = coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(cells), len(cells))
    )
    _, labels = connected_components(graph, directed=False)
    return labels[inverse]


def component_coverage(covered, labels, weights=None):
    """Spatially weighted coverage per component; labels can be cached by search."""
    if not len(labels):
        return np.empty(0)
    if weights is None:
        weights = np.ones(len(labels))
    totals = np.bincount(labels, weights=weights)
    gains = np.bincount(labels, weights=weights * np.asarray(covered, dtype=bool))
    return np.divide(gains, totals, out=np.zeros_like(totals), where=totals > 0)


def component_target_reached(coverage, target=None):
    """Optional all-components requirement, independent of global population size."""
    return target is None or target == 0 or bool(len(coverage) and np.min(coverage) >= target)


def surface_support_status(report, target):
    """Decide against conservative area bounds, never the quadrature estimate."""
    lower = report.get("surface_supported_area_fraction_lower", 0.0)
    upper = report.get("surface_supported_area_fraction_upper", 1.0)
    if lower >= target:
        return "reached"
    return "not_reached" if upper < target else "inconclusive"


def surface_support_reached(report, target):
    return surface_support_status(report, target) == "reached"


def surface_samples(box, side=4):
    """Equal-area midpoint samples on each cuboid face, with area weights."""
    u, v = np.meshgrid((np.arange(side) + 0.5) / side - 0.5, (np.arange(side) + 0.5) / side - 0.5)
    samples, weights = [], []
    for axis in range(3):
        tangent = [i for i in range(3) if i != axis]
        area = float(np.prod(box["dimensions"][tangent]))
        for sign in (-1, 1):
            local = np.zeros((side * side, 3))
            local[:, axis] = sign / 2
            local[:, tangent] = np.column_stack((u.ravel(), v.ravel()))
            samples.append((local * box["dimensions"]) @ box["rotation"].T + box["center"])
            weights.append(np.full(side * side, area / side**2))
    return np.concatenate(samples), np.concatenate(weights)


def surface_fit_error(box, tree, tolerance, previous=()):
    points, area = surface_samples(box)
    from .cuboids import box_membership

    visible = np.ones(len(points), bool)
    for other in previous:
        visible &= ~box_membership(points, other, tolerance=-tolerance * 1e-8)
    if not visible.any():
        return 0.0
    distances, _ = tree.query(points[visible])
    return float(np.average(np.minimum(distances / tolerance, 4), weights=area[visible]))


def surface_metrics(
    boxes,
    points,
    tolerance,
    *,
    target_support=None,
    support_area_error=0.002,
    support_max_evaluations=65536,
):
    """Exact source distances, reverse quadrature, and conservative support bounds."""
    from .geometry import SurfaceMesh, exposed_triangles

    mesh = SurfaceMesh(exposed_triangles(boxes))
    samples, weights = mesh.samples(max(4096, len(boxes) * 384))
    source_tree = cKDTree(points)
    bounds = mesh.support_bounds(
        source_tree,
        tolerance,
        target=target_support,
        area_fraction_tolerance=support_area_error,
        max_evaluations=support_max_evaluations,
    )
    support_report = dict(
        surface_supported_area_fraction_lower=bounds["lower_fraction"],
        surface_supported_area_fraction_upper=bounds["upper_fraction"],
        surface_support_status=bounds["status"],
        surface_support_bounds=bounds,
    )
    total_area = sum(
        2
        * (
            b["dimensions"][0] * b["dimensions"][1]
            + b["dimensions"][0] * b["dimensions"][2]
            + b["dimensions"][1] * b["dimensions"][2]
        )
        for b in boxes
    )
    if not len(samples):
        return dict(
            exposed_samples=0,
            estimated_exposed_area=0.0,
            source_surface_coverage=0.0,
            surface_supported_area_fraction=0.0,
            component_surface_coverage=[],
            minimum_component_surface_coverage=0.0,
            **support_report,
        )
    to_model = mesh.distances(points)
    to_source, _ = source_tree.query(samples, workers=current_workers())
    weights_source = spatial_weights(points, tolerance)
    components = component_coverage(
        to_model <= tolerance, spatial_components(points, tolerance), weights_source
    )
    return dict(
        method="Clipped union boundary; continuous point-to-triangle distances and area quadrature",
        exposed_samples=len(samples),
        estimated_exposed_area=float(weights.sum()),
        estimated_hidden_area_fraction=float(np.clip(1 - weights.sum() / total_area, 0, 1)),
        source_to_surface_p95=float(np.quantile(to_model, 0.95)),
        source_to_surface_rms=float(np.sqrt(np.mean(to_model**2))),
        source_surface_coverage=float(np.average(to_model <= tolerance, weights=weights_source)),
        component_surface_coverage=components.tolist(),
        minimum_component_surface_coverage=float(min(components, default=0)),
        surface_to_source_rms=float(np.sqrt(np.average(to_source**2, weights=weights))),
        surface_supported_area_fraction=float(np.average(to_source <= tolerance, weights=weights)),
        sampling_note="Source distances use continuous triangles. Reverse RMS and support fraction are area-quadrature estimates; adaptive support bounds are reported separately. No Hausdorff guarantee.",
        **support_report,
    )


def approximation_metrics(
    boxes, solid, points, distances, tolerance, *, target_support=None,
    support_max_evaluations=65536,
):
    """Report fitting error separately from the hard envelope guarantee."""
    fractions = [exterior_fraction(b, solid, max_axis=24) for b in boxes]
    volumes = np.array([b["volume"] for b in boxes])
    # Count connected *observed* residual patches at the evaluation scale.
    residual = points[distances > tolerance]
    largest = 0
    if len(residual):
        cells, counts = np.unique(
            evaluation_cells(residual, tolerance, origin=points.min(0), cell_width_factor=1),
            axis=0, return_counts=True
        )
        tree = cKDTree(cells)
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        pairs = tree.query_pairs(np.sqrt(3) + 1e-8, output_type="ndarray")
        graph = coo_matrix(
            (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(cells), len(cells))
        )
        _, labels = connected_components(graph, directed=False)
        largest = int(np.bincount(labels, weights=counts).max())
    weights = spatial_weights(points, tolerance)
    covered = distances <= tolerance
    components = component_coverage(covered, spatial_components(points, tolerance), weights)
    return dict(
        solid["approximation"],
        estimated_box_exterior_fractions=fractions,
        estimated_exterior_fraction_volume_weighted=float(
            np.dot(volumes, fractions) / volumes.sum()
        ),
        exterior_estimator="exact voxel overlap for axis-aligned boxes; seeded jittered strata otherwise; per-box, not union volume",
        surface=surface_metrics(
            boxes, points, tolerance, target_support=target_support,
            support_max_evaluations=support_max_evaluations,
        ),
        coverage_tolerance=float(tolerance),
        spatial_coverage=float(np.average(covered, weights=weights)),
        component_spatial_coverage=components.tolist(),
        minimum_component_coverage=float(min(components, default=0)),
        largest_uncovered_patch_points=largest,
        point_distance_quantiles=dict(
            zip(
                ("median", "p90", "p95", "p99", "maximum"),
                np.quantile(distances, [0.5, 0.9, 0.95, 0.99, 1]).tolist(),
            )
        ),
    )
