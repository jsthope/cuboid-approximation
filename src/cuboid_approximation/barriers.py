"""Finite, supported edge barriers for region segmentation."""

from dataclasses import asdict, dataclass
from time import perf_counter
import numpy as np
from scipy.spatial import cKDTree

from .parallel import current_workers


@dataclass
class LineConfig:
    distance_factor: float = 1.35
    gap_factor: float = 3.5
    min_length_factor: float = 8.0
    tangent_radius_factor: float = 6.0
    min_linearity: float = 0.60
    angle_degrees: float = 25.0
    merge_angle_degrees: float = 10.0
    min_support: int = 10
    min_linear_fraction: float = 0.5
    max_segments: int = 180
    trials_per_step: int = 24
    max_trials: int = 1800
    seed: int = 42

    def validate(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        for name in ("min_support", "max_segments", "trials_per_step", "max_trials", "seed"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer.")
        if (
            min(
                self.distance_factor,
                self.gap_factor,
                self.min_length_factor,
                self.tangent_radius_factor,
            )
            <= 0
        ):
            raise ValueError("Scale factors must be positive.")
        if not (
            0 < self.min_linearity < 1
            and 0 < self.min_linear_fraction <= 1
            and 0 < self.angle_degrees < 90
            and 0 <= self.merge_angle_degrees < 90
        ):
            raise ValueError("Invalid linearity or angle limits.")
        if (
            self.min_support < 3
            or min(self.max_segments, self.trials_per_step, self.max_trials) < 1
        ):
            raise ValueError(
                "At least three supporting points and a positive search budget are required."
            )


def _tangents(points, radius):
    """PCA on edge candidates, not on the original full surface neighborhoods."""
    distances, ids = cKDTree(points).query(points, k=min(24, len(points)), workers=current_workers())
    keep = distances <= radius
    count = keep.sum(axis=1)
    tangent = np.zeros_like(points)
    linearity = np.zeros(len(points))
    for start in range(0, len(points), 2048):
        part = slice(start, start + 2048)
        delta = points[ids[part]] - points[part, None]
        delta *= keep[part, :, None]
        mean = delta.sum(axis=1) / count[part, None]
        cov = np.einsum("nki,nkj->nij", delta, delta) / count[part, None, None]
        cov -= np.einsum("ni,nj->nij", mean, mean)
        values, vectors = np.linalg.eigh(cov)
        tangent[part] = vectors[:, :, -1]
        linearity[part] = np.divide(
            values[:, 2] - values[:, 1],
            values[:, 2],
            out=np.zeros(len(values)),
            where=values[:, 2] > 1e-20,
        )
    linearity[count < 5] = 0
    return tangent, linearity


def _tls(points, weights):
    weights = weights / weights.sum()
    center = weights @ points
    delta = points - center
    covariance = np.einsum("n,ni,nj->ij", weights, delta, delta)
    _, axes = np.linalg.eigh(covariance)
    return center, axes[:, -1]


def _runs(indices, along, maximum_gap):
    order = indices[np.argsort(along[indices], kind="stable")]
    return np.split(order, np.flatnonzero(np.diff(along[order]) > maximum_gap) + 1)


def _merge_supported(segments, supports, points, scores, tolerance, max_gap, angle):
    """Merge nearly collinear neighbors only after validating their joint support."""
    if angle == 0:
        return segments, supports
    segments, supports = list(segments), list(supports)
    cosine = np.cos(np.deg2rad(angle))
    changed = True
    while changed:
        changed = False
        for i in range(len(segments)):
            first = segments[i][1] - segments[i][0]
            first /= np.linalg.norm(first)
            for j in range(i + 1, len(segments)):
                second = segments[j][1] - segments[j][0]
                second /= np.linalg.norm(second)
                if abs(first @ second) < cosine:
                    continue
                endpoint_gap = np.linalg.norm(
                    segments[i][:, None] - segments[j][None], axis=2
                ).min()
                if endpoint_gap > max_gap + 2 * tolerance:
                    continue
                ids = np.unique(np.r_[supports[i], supports[j]])
                weights = np.maximum(scores[ids], np.max(scores[ids]) * 0.1)
                origin, direction = _tls(points[ids], weights)
                relative = points[ids] - origin
                along = relative @ direction
                perpendicular = np.linalg.norm(relative - along[:, None] * direction, axis=1)
                if (
                    perpendicular.max() > tolerance
                    or np.diff(np.sort(along)).max(initial=0) > max_gap
                ):
                    continue
                segments[i] = origin + np.array([along.min(), along.max()])[:, None] * direction
                supports[i] = ids
                del segments[j], supports[j]
                changed = True
                break
            if changed:
                break
    return segments, supports


def _package(segments, supports, points, scores, metadata, diagnostics=False):
    segments = np.asarray(segments, dtype=float).reshape(-1, 2, 3)
    if not diagnostics:
        return dict(segments=segments)
    lengths, rmse, maximum_gaps, mean_scores = [], [], [], []
    for segment, ids in zip(segments, supports):
        delta = segment[1] - segment[0]
        length = np.linalg.norm(delta)
        direction = delta / length
        relative = points[ids] - segment[0]
        along = relative @ direction
        residual = np.linalg.norm(relative - along[:, None] * direction, axis=1)
        lengths.append(length)
        rmse.append(np.sqrt(np.mean(residual**2)) if len(ids) else 0)
        maximum_gaps.append(np.diff(np.sort(along)).max(initial=0))
        mean_scores.append(np.mean(scores[ids]) if len(ids) else 0)
    counts = np.array([len(ids) for ids in supports], dtype=np.int64)
    return dict(
        segments=segments,
        lengths=np.array(lengths),
        rmse=np.array(rmse),
        max_support_gaps=np.array(maximum_gaps),
        mean_scores=np.array(mean_scores),
        support_counts=counts,
        support_indptr=np.r_[0, np.cumsum(counts)],
        support_point_indices=np.concatenate(supports).astype(np.int64)
        if supports
        else np.empty(0, dtype=np.int64),
        **metadata,
    )


def fit_edge_lines(points, scores, mask, spacing, config=None, voxel_size=0, *, diagnostics=False):
    """Tangent-guided line RANSAC, weighted TLS, and splitting at unsupported gaps.

    Every output is fitted only to this detector's selected points. Tangents
    reject broad surface patches; no world-axis snapping or cross-method fusion.
    Extra support arrays and fit statistics are returned only with diagnostics=True.
    """
    config = config or LineConfig()
    config.validate()
    points, scores, mask = (
        np.asarray(points, float),
        np.asarray(scores, float),
        np.asarray(mask, bool),
    )
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or scores.shape != (len(points),)
        or mask.shape != scores.shape
    ):
        raise ValueError("Expected N-by-3 points and N scores/mask.")
    if (
        not np.isfinite(points).all()
        or not np.isfinite(scores).all()
        or not np.isfinite(spacing)
        or spacing <= 0
    ):
        raise ValueError("Finite points/scores and positive spacing are required.")
    if not np.isfinite(voxel_size) or voxel_size < 0:
        raise ValueError("voxel_size must be finite and nonnegative.")
    started = perf_counter()
    tolerance = max(config.distance_factor * spacing, 0.5 * voxel_size)
    max_gap = max(config.gap_factor * spacing, 1.5 * voxel_size)
    min_length = max(config.min_length_factor * spacing, 2.5 * voxel_size)
    tangent_radius = max(config.tangent_radius_factor * spacing, 2 * voxel_size)
    metadata = dict(
        algorithm="tangent_guided_line_ransac_tls",
        distance_tolerance=tolerance,
        max_gap=max_gap,
        min_length=min_length,
        tangent_radius=tangent_radius,
        input_selected_points=int(mask.sum()),
    )
    candidate_ids = np.flatnonzero(mask & (scores > 1e-10))
    if len(candidate_ids) < config.min_support:
        return _package(
            [], [], points, scores, dict(**metadata, seconds=perf_counter() - started, trials=0),
            diagnostics=diagnostics,
        )
    # Work around a nearby origin so translations cannot destabilize covariance.
    reference = points[candidate_ids].mean(axis=0)
    candidates = points[candidate_ids] - reference
    tangent, linearity = _tangents(candidates, tangent_radius)
    linear = linearity >= config.min_linearity
    ranks = np.argsort(np.argsort(scores[candidate_ids], kind="stable"), kind="stable")
    weights = 0.25 + 0.75 * ranks / max(len(ranks) - 1, 1)
    available = np.ones(len(candidates), dtype=bool)
    proposed = np.zeros(len(candidates), dtype=bool)
    cosine = np.cos(np.deg2rad(config.angle_degrees))
    rng = np.random.default_rng(config.seed)
    segments, supports = [], []
    trials = 0

    def proposal(seed):
        origin, direction = candidates[seed], tangent[seed]
        delta = candidates - origin
        along = delta @ direction
        d2 = np.maximum(np.einsum("ij,ij->i", delta, delta) - along**2, 0)
        aligned = np.abs(tangent @ direction) >= cosine
        nearby = np.flatnonzero(available & (d2 <= tolerance**2) & (~linear | aligned))
        best, quality = None, 0.0
        for run in _runs(nearby, along, max_gap):
            if len(run) < config.min_support or linear[run].mean() < config.min_linear_fraction:
                continue
            length = np.ptp(along[run])
            if length < min_length:
                continue
            # Support and span reward a coherent extended segment, not isolated peaks.
            value = float(weights[run].sum() * np.sqrt(length))
            if value > quality:
                best, quality = run, value
        return best, quality

    while len(segments) < config.max_segments and trials < config.max_trials:
        seeds = np.flatnonzero(available & linear & ~proposed)
        if len(seeds) < 1 or available.sum() < config.min_support:
            break
        count = min(len(seeds), config.trials_per_step, config.max_trials - trials)
        probabilities = weights[seeds] * linearity[seeds]
        probabilities /= probabilities.sum()
        seeds = rng.choice(seeds, size=count, replace=False, p=probabilities)
        proposed[seeds] = True
        trials += count
        best, quality = None, 0.0
        for seed in seeds:
            candidate, value = proposal(seed)
            if value > quality:
                best, quality = candidate, value
        if best is None:
            continue
        # Refine inliers with weighted total least squares. Only shorten/split;
        # do not extrapolate the fit to unrelated candidates outside this run.
        for _ in range(3):
            origin, direction = _tls(candidates[best], weights[best])
            delta = candidates[best] - origin
            along = delta @ direction
            distance = np.linalg.norm(delta - along[:, None] * direction, axis=1)
            keep = (distance <= tolerance) & (
                ~linear[best] | (np.abs(tangent[best] @ direction) >= cosine)
            )
            best = best[keep]
            if len(best) < config.min_support:
                break
        if len(best) < config.min_support:
            continue
        along_all = (candidates - origin) @ direction
        accepted = False
        for run in _runs(best, along_all, max_gap):
            if len(run) < config.min_support or linear[run].mean() < config.min_linear_fraction:
                continue
            # A split run keeps the validated parent line: another TLS can move
            # endpoints away from the already tested support tube.
            low, high = along_all[run].min(), along_all[run].max()
            if high - low < min_length:
                continue
            segment = origin + np.array([low, high])[:, None] * direction + reference
            segments.append(segment)
            supports.append(candidate_ids[run])
            available[run] = False
            accepted = True
            if len(segments) >= config.max_segments:
                break
        if not accepted:
            proposed[best] = True
    raw_segment_count = len(segments)
    segments, supports = _merge_supported(
        segments, supports, points, scores, tolerance, max_gap, config.merge_angle_degrees
    )
    return _package(
        segments,
        supports,
        points,
        scores,
        dict(
            **metadata,
            seconds=perf_counter() - started,
            trials=trials,
            raw_segment_count=raw_segment_count,
            search_budget_reached=bool(
                trials >= config.max_trials or raw_segment_count >= config.max_segments
            ),
            tangent_seed_points=int(linear.sum()),
        ),
        diagnostics=diagnostics,
    )
