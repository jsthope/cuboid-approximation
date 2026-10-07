"""Append-only selection with a cheap surface objective and exact commit checks.

The local signed-distance field is only a lower bound inside overlapping boxes.
It guides refinement, while a clipped union mesh checks every retained source
point before a box becomes immutable. No sampled fit is a burial certificate.
"""

import heapq

import numpy as np
from scipy.spatial import cKDTree

from .approximation import (
    component_coverage,
    component_target_reached,
    exterior_fraction,
    evaluation_cells,
    spatial_components,
    spatial_weights,
)
from .geometry import PointIndex, SurfaceMesh, exposed_triangles
from .support_index import BoxSupportIndex, geometry_key


def signed_box_distance(points, box):
    q = np.abs((points - box["center"]) @ box["rotation"]) - box["dimensions"] / 2
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)


def exterior_volume_cost(box, solid, penalty_weight, reference_volume):
    """Dimensionless cost of absolute exterior volume, including thin panels."""
    return 1 + penalty_weight * exterior_fraction(box, solid) * box["volume"] / reference_volume


def adaptive_face_samples(box, tolerance, budget=768):
    """Area quadrature with aspect-ratio-aware faces and a bounded work budget."""
    dimensions = box["dimensions"]
    face_area = np.array([np.prod(np.delete(dimensions, axis)) for axis in range(3)])
    total = 2 * face_area.sum()
    points, weights = [], []
    for axis in range(3):
        tangent = [a for a in range(3) if a != axis]
        lengths = dimensions[tangent]
        face_budget = max(4, int(budget * face_area[axis] / max(total, 1e-30)))
        counts = np.maximum(
            2, np.minimum(face_budget, np.ceil(lengths / max(tolerance, 1e-30))).astype(int)
        )
        if np.prod(counts) > face_budget:
            counts = np.maximum(
                2, np.floor(counts * np.sqrt(face_budget / np.prod(counts))).astype(int)
            )
            longest = int(np.argmax(counts))
            counts[longest] = min(counts[longest], face_budget // counts[1 - longest])
        u, v = np.meshgrid(*[(np.arange(n) + 0.5) / n - 0.5 for n in counts], indexing="ij")
        for sign in (-1, 1):
            local = np.zeros((u.size, 3))
            local[:, axis] = sign * dimensions[axis] / 2
            local[:, tangent] = np.column_stack((u.ravel(), v.ravel())) * lengths
            points.append(local @ box["rotation"].T + box["center"])
            weights.append(np.full(u.size, face_area[axis] / u.size))
    return np.concatenate(points), np.concatenate(weights)


class UnionSurfaceObjective:
    """Shared bounded-cost trial score; exact full-cloud validation at commit.

    Source samples are deterministic spatial representatives. Reverse samples
    retain the currently exposed boundary, so hiding previously fitted faces
    changes the trial objective as well as adding new faces.
    """

    def __init__(
        self, points, weights, tolerance, config, source_ids, source_weights, component_labels=None
    ):
        self.points, self.weights = points, weights
        self.tolerance, self.config = tolerance, config
        self.tree = cKDTree(points)
        self.source_ids, self.source_weights = source_ids, source_weights
        self.sample = points[source_ids]
        self.component_labels = component_labels
        self.total_weight = float(weights.sum())
        occupied_cells = np.unique(evaluation_cells(points, tolerance), axis=0)
        self.reverse_area_scale = max(2 * len(occupied_cells) * (2 * tolerance) ** 2, tolerance**2)
        self.error_budget = (1 - config.target_coverage) * self.total_weight
        self.boxes = []
        self.inside = np.zeros(len(points), bool)
        self.distances = np.full(len(points), np.inf)
        self.irreparable = np.zeros(len(points), bool)
        self.signed = np.full(len(self.sample), np.inf)
        self.boundary = np.empty((0, 3))
        self.area = self.boundary_source_distance = np.empty(0)
        self.source_coverage = self.reverse_coverage = 0.0
        self.reverse_error = 0.0
        self.mesh = SurfaceMesh(np.empty((0, 3, 3)))
        self.support_bounds = None
        self.distances_exact = True
        self.distance_work = dict(
            full_evaluations=0,
            incremental_evaluations=0,
            points_evaluated=0,
            points_reused=0,
            exact_refreshes=0,
        )
        self.coordinate_scale = float(np.max(np.abs(points), initial=0))

    def evaluate(self, boxes, *, incremental=False):
        """Evaluate exact classifications; optionally reuse unaffected point classes.

        Removing old boundary fragments and adding new ones can only change a
        point's distance<=tau status inside the new box's tau neighborhood. Far
        distance magnitudes can become stale; they are never consumed by trial
        scoring and are explicitly refreshed for the zero-volume-gain decision.
        The default remains a full exact distance evaluation for direct callers.
        """
        append = len(boxes) == len(self.boxes) + 1 and all(
            all(
                np.array_equal(old[name], new[name])
                for name in ("center", "dimensions", "rotation")
            )
            for old, new in zip(self.boxes, boxes)
        )
        old_extent = max((float(np.max(box["dimensions"])) for box in self.boxes), default=0.0)
        new_extent = max((float(np.max(box["dimensions"])) for box in boxes), default=0.0)
        # Clipping uses a scale-dependent epsilon. When that scale grows, old
        # faces away from this box may round differently: preserve the full oracle.
        incremental = incremental and append and (not self.boxes or new_extent <= old_extent)
        mesh = SurfaceMesh(exposed_triangles(boxes))
        inside = self.inside.copy() if append else np.zeros(len(self.points), bool)
        if incremental:
            added = boxes[-1]
            rounding = (
                64
                * np.finfo(float).eps
                * max(1.0, self.coordinate_scale, float(np.max(np.abs(added["center"]))))
                + new_extent * 1e-9
            )
            half = (
                np.abs(added["rotation"]) @ (added["dimensions"] / 2)
                + self.tolerance * (1 + 1e-10)
                + rounding
            )
            affected = np.zeros(len(self.points), bool)
            for start in range(0, len(self.points), 65536):
                affected[start : start + 65536] = np.all(
                    np.abs(self.points[start : start + 65536] - added["center"]) <= half, axis=1
                )
            # Re-triangulation can perturb the last bits on an unchanged face.
            # Globally recheck old threshold ties even when far from the box.
            affected |= np.abs(self.distances - self.tolerance) <= rounding
            affected |= np.abs(self.distances - self.tolerance * (1 + 1e-10)) <= rounding
            distances = self.distances.copy()
            distances[affected] = mesh.distances(self.points[affected])
            inside[affected] |= signed_box_distance(self.points[affected], added) <= 0
            count = int(affected.sum())
            self.distance_work["incremental_evaluations"] += 1
            self.distance_work["points_reused"] += len(self.points) - count
            exact = bool(affected.all())
        else:
            distances = mesh.distances(self.points)
            for box in boxes[len(self.boxes) :] if append else boxes:
                inside |= signed_box_distance(self.points, box) <= 0
            count, exact = len(self.points), True
            self.distance_work["full_evaluations"] += 1
        self.distance_work["points_evaluated"] += count
        boundary, area = mesh.samples(max(4096, len(boxes) * 384))
        reverse = self.tree.query(boundary)[0] if len(boundary) else np.empty(0)
        return dict(
            mesh=mesh,
            distances=distances,
            distances_exact=exact,
            inside=inside,
            irreparable=inside & (distances > self.tolerance * (1 + 1e-10)),
            boundary=boundary,
            area=area,
            boundary_source_distance=reverse,
            source_coverage=float(np.average(distances <= self.tolerance, weights=self.weights)),
            reverse_coverage=float(np.average(reverse <= self.tolerance, weights=area))
            if len(area)
            else 0.0,
        )

    def refresh_exact_distances(self):
        """Restore magnitudes before an operation that needs more than thresholds."""
        if self.distances_exact:
            return
        self.distances = self.mesh.distances(self.points)
        self.distances_exact = True
        self.irreparable = self.inside & (self.distances > self.tolerance * (1 + 1e-10))
        self.source_coverage = float(
            np.average(self.distances <= self.tolerance, weights=self.weights)
        )
        self.distance_work["exact_refreshes"] += 1
        self.distance_work["points_evaluated"] += len(self.points)

    def install(self, boxes, evaluation):
        for name, value in evaluation.items():
            setattr(self, name, value)
        self.boxes = list(boxes)
        self.support_bounds = None
        self.signed = np.full(len(self.sample), np.inf)
        for box in boxes:
            self.signed = np.minimum(self.signed, signed_box_distance(self.sample, box))
        self.reverse_error = self._reverse_error(self.boundary_source_distance, self.area)

    def _reverse_error(self, distances, area):
        if not len(area):
            return 0.0
        # Fixed observed-area normalization forbids diluting an existing error
        # by inventing more faces. Saturation keeps a sparse cloud's small area
        # proxy from overwhelming every positive source-coverage improvement.
        integral = float(np.dot(np.minimum(distances / self.tolerance, 4), area))
        return integral / (self.reverse_area_scale + integral)

    def feasible(self, evaluation):
        if float(self.weights[evaluation["irreparable"]].sum()) > self.error_budget + 1e-9:
            return False
        if self.component_labels is not None:
            return component_target_reached(
                component_coverage(
                    ~evaluation["irreparable"],
                    self.component_labels,
                    self.weights,
                ),
                self.config.target_component_coverage,
            )
        return True

    def complete(self):
        if self.source_coverage < self.config.target_coverage - 1e-12:
            return False
        if self.component_labels is not None and not component_target_reached(
            component_coverage(
                self.distances <= self.tolerance,
                self.component_labels,
                self.weights,
            ),
            self.config.target_component_coverage,
        ):
            return False
        if self.support_bounds is None:
            # Quadrature can underestimate support near the target. It guides
            # ranking but must not veto the conservative completion decision.
            self.support_bounds = self.mesh.support_bounds(
                self.tree,
                self.tolerance,
                target=self.config.target_surface_support,
                max_evaluations=self.config.surface_max_evaluations,
            )
        return self.support_bounds["lower_fraction"] >= self.config.target_surface_support

    def burial_feasible(self, box, ids):
        """Every point deep inside even one box is permanently unreachable."""
        if self.config.reconstruction_mode == "surface" and np.min(
            box["dimensions"]
        ) > 2 * self.tolerance * (1 + 1e-10):
            return False
        if not len(ids):
            return True
        deep = signed_box_distance(self.points[ids], box) < -self.tolerance * (1 + 1e-10)
        new = deep & ~self.irreparable[ids]
        mass = self.weights[self.irreparable].sum() + self.weights[ids[new]].sum()
        if mass > self.error_budget + 1e-9:
            return False
        if self.component_labels is not None:
            irreparable = self.irreparable.copy()
            irreparable[ids[new]] = True
            return self.feasible(dict(irreparable=irreparable))
        return True

    def score(self, box, ids, volume_gain, cost):
        from .cuboids import box_membership

        if not self.burial_feasible(box, ids):
            return 0.0
        signed = np.minimum(self.signed, signed_box_distance(self.sample, box))
        old = np.abs(self.signed) / self.tolerance
        new = np.abs(signed) / self.tolerance
        # Advancing into empty space toward a distant observation is not a gain:
        # only newly supported points or a better fit inside tolerance count.
        source_gain = float(
            np.average(np.minimum(old, 1) - np.minimum(new, 1), weights=self.source_weights)
        )
        coverage_gain = float(
            np.average((new <= 1).astype(float) - (old <= 1), weights=self.source_weights)
        )
        samples, area = adaptive_face_samples(box, self.tolerance)
        visible = np.ones(len(samples), bool)
        for other in self.boxes:
            visible &= ~box_membership(samples, other, tolerance=-self.tolerance * 1e-8)
        reverse = self.tree.query(samples[visible])[0]
        retained = ~box_membership(self.boundary, box, tolerance=-self.tolerance * 1e-8)
        distances = np.r_[self.boundary_source_distance[retained], reverse]
        union_area = np.r_[self.area[retained], area[visible]]
        reverse_gain = self.reverse_error - self._reverse_error(distances, union_area)
        gain = volume_gain / self.total_weight + source_gain + coverage_gain + 0.25 * reverse_gain
        return max(0.0, gain) / cost

    def residual_mask(self, uncovered):
        residual = uncovered.astype(bool) | (self.distances > self.tolerance)
        # Unsupported model faces also identify nearby source patches worth refitting.
        bad = self.boundary_source_distance > self.tolerance
        if bad.any():
            residual[self.tree.query(self.boundary[bad])[1]] = True
        return residual & ~self.irreparable


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
    from .cuboids import certify_box, distance_to_box, grow_certified, refine_box

    h = solid["voxel_size"]

    def key(box):
        return tuple(
            np.round(
                np.r_[
                    (box["center"] - solid["origin"]) / h,
                    box["dimensions"] / h,
                    box["rotation"].ravel(),
                ],
                8,
            )
        )

    unique = {key(box): box for box in candidates}
    pool = sorted(unique.values(), key=lambda box: -box["volume"])
    pool_indices = {key(candidate): index for index, candidate in enumerate(pool)}
    approximate = "approximation" in solid
    surface_enabled = approximate and config.target_surface_support > 0
    source_tree = cKDTree(points) if approximate and not surface_enabled else None
    weights = spatial_weights(points, tolerance) if approximate else np.ones(len(points))
    total_weight = float(weights.sum())
    component_labels = (
        spatial_components(points, tolerance) if config.target_component_coverage > 0 else None
    )
    reference_volume = max(float(solid.get("reference", solid["safe"]).sum()) * h**3, h**3)

    def penalty(box):
        if not approximate:
            return 1.0
        # Fractions alone charge an arbitrary fixed cost to a thin observed
        # plane straddling a voxel boundary and reward inflating that plane.
        return exterior_volume_cost(
            box, solid, config.approximation_exterior_penalty, reference_volume
        )

    boxes = [
        dict(
            b,
            center=b["center"].copy(),
            dimensions=b["dimensions"].copy(),
            rotation=b["rotation"].copy(),
        )
        for b in reference
    ]
    if len(boxes) > config.max_cuboids:
        raise ValueError("The frozen prefix exceeds max_cuboids; increase the budget.")
    for box in boxes:
        if not certify_box(box, solid, strict=True):
            raise ValueError("A frozen cuboid is outside the current admissible solid.")
    uncovered = np.ones(len(points), np.int32)
    for box in boxes:
        uncovered[distance_to_box(points, box) <= tolerance] = 0
    curve = [dict(cuboids=len(boxes), covered_fraction=float(1 - uncovered.mean()), added_points=0)]

    cell_width = max(h, tolerance) / 2
    cells, inverse = np.unique(
        np.floor((points - solid["origin"]) / cell_width).astype(int), axis=0, return_inverse=True
    )
    proxy_points = solid["origin"] + (cells + 0.5) * cell_width
    point_order = np.argsort(inverse, kind="stable")
    point_indptr = np.r_[0, np.cumsum(np.bincount(inverse))]
    proxy_index = PointIndex(proxy_points)
    proxy_tolerance = tolerance + np.sqrt(3) * cell_width / 2
    proxy_weights = np.bincount(inverse, weights=weights * uncovered)

    def raw_proxy_support(box):
        half = np.abs(box["rotation"]) @ (box["dimensions"] / 2) + proxy_tolerance
        ids = proxy_index.query(box["center"], half)
        return ids[distance_to_box(proxy_points[ids], box) <= proxy_tolerance]

    support_index = BoxSupportIndex(
        points,
        tolerance,
        min(64, config.max_memory_mb // 8) * 1024**2,
        proxy_index=proxy_index,
        point_order=point_order,
        point_indptr=point_indptr,
        proxy_radius=proxy_tolerance - tolerance,
        neighborhood_margin=max(h, tolerance),
    )
    support = support_index.query

    def proxy_support(box):
        return support_index.proxy_support(box, raw_proxy_support)

    objective = None
    if surface_enabled:
        chosen_cells = np.linspace(0, len(cells) - 1, min(len(cells), 2048), dtype=int)
        source_ids = point_order[point_indptr[chosen_cells]]
        source_weights = np.bincount(inverse, weights=weights)[chosen_cells]
        objective = UnionSurfaceObjective(
            points, weights, tolerance, config, source_ids, source_weights, component_labels
        )
        if boxes:
            objective.install(boxes, objective.evaluate(boxes))

    def cached_support(index):
        return support(pool[index])

    heap, penalties, proxy_scores = [], [], []
    consumed = set()

    def enqueue(box, index):
        cost = penalties[index] if index < len(penalties) else penalty(box)
        if index == len(penalties):
            penalties.append(cost)
        gain = float(proxy_weights[proxy_support(box)].sum()) / cost
        if index == len(proxy_scores):
            proxy_scores.append(gain)
        else:
            proxy_scores[index] = gain
        heapq.heappush(heap, (-gain, -box["volume"], index, -1))

    for index, box in enumerate(pool):
        enqueue(box, index)
        if (index + 1) % 5000 == 0:
            progress(f"Scored {index + 1}/{len(pool)} candidate coverage bounds")
    required = int(np.ceil(config.target_coverage * len(points)))
    required_spatial = config.target_coverage * total_weight
    proposed_generation = -1
    rejected_burial = 0

    def complete():
        return (
            len(points) - int(uncovered.sum()) >= required
            and weights[uncovered == 0].sum() >= required_spatial - 1e-9
            and (
                component_labels is None
                or component_target_reached(
                    component_coverage(
                        uncovered == 0,
                        component_labels,
                        weights,
                    ),
                    config.target_component_coverage,
                )
            )
            and (objective is None or objective.complete())
        )

    def residual_mask():
        return uncovered.astype(bool) if objective is None else objective.residual_mask(uncovered)

    def add_residuals():
        nonlocal proxy_weights
        residual = residual_mask()
        proxy_weights = np.bincount(inverse, weights=weights * residual, minlength=len(cells))
        added = 0
        if propose is not None and residual.any():
            for candidate in propose(points[residual]):
                identity = key(candidate)
                if identity not in unique:
                    unique[identity] = candidate
                    pool.append(candidate)
                    pool_indices[identity] = len(pool) - 1
                    candidates.append(candidate)
                    enqueue(candidate, len(pool) - 1)
                    added += 1
        return added

    def score(trial):
        ids = support(trial)
        gain = float(np.dot(uncovered[ids], weights[ids]))
        if objective is not None:
            return objective.score(trial, ids, gain, penalty(trial))
        if not len(ids) or gain <= 0:
            return 0.0
        if not approximate:
            return gain
        from .approximation import surface_fit_error
        from .geometry import box_surface_distance

        error = np.average(
            np.minimum(box_surface_distance(points[ids], trial) / tolerance, 4),
            weights=weights[ids],
        )
        return gain / (
            penalty(trial) + surface_fit_error(trial, source_tree, tolerance, boxes) + error
        )

    # Share exact geometric support and objective scores under one byte budget.
    raw_score = score

    def score(trial):
        return support_index.score(trial, raw_score)

    # These are search budgets, not claims that the finite pool was exhausted.
    prescreen_limit, finalist_limit, rescue_limit = 32, 3, 1
    refined_count = prescreened_count = exact_checks = 0
    attempted = set()

    def prescreen(preferred=()):
        """Bound expensive scoring; mix strong proxy candidates with pool diversity."""
        panel_ids = [
            i
            for i, candidate in enumerate(pool)
            if i not in consumed
            and i not in attempted
            and np.min(candidate["dimensions"]) <= 2 * tolerance
        ]
        panel_ids.sort(key=lambda i: (-proxy_scores[i], i))
        preferred = list(dict.fromkeys([*preferred, *panel_ids[:8]]))
        # Residual panels can fill the entire prefix of propose()'s output.
        # Reserve opportunities for local cells/fits instead of truncating that
        # ordered list before its guaranteed small, safe fallback seeds.
        groups = {}
        for index in preferred:
            if index in consumed or index in attempted:
                continue
            kind = pool[index].get("proposal_kind", "local")
            groups.setdefault(kind, []).append(index)
        for group in groups.values():
            group.sort(key=lambda i: (-proxy_scores[i], i))
        ids = []
        for position in range(prescreen_limit):
            for group in groups.values():
                if position < len(group) and len(ids) < prescreen_limit:
                    ids.append(group[position])
            if len(ids) >= prescreen_limit:
                break
        top_limit = max(len(ids), prescreen_limit * 3 // 4)
        while heap and len(ids) < top_limit:
            _, _, index, _ = heapq.heappop(heap)
            if index not in consumed and index not in attempted and index not in ids:
                ids.append(index)
        # Small-volume/local candidates must get a chance even when every large
        # proxy winner buries observed surfaces. This bounded diversity pass does
        # not refine or evaluate the remaining potentially huge candidate pool.
        available = [
            i for i in range(len(pool)) if i not in consumed and i not in attempted and i not in ids
        ]
        count = min(prescreen_limit - len(ids), len(available))
        if count:
            ids.extend(available[j] for j in np.linspace(0, len(available) - 1, count, dtype=int))
        attempted.update(ids)
        return ids

    def certified(trial):
        return certify_box(trial, solid, strict=True) and (
            export_certificate is None or export_certificate(trial)
        )

    def finish_candidate(index):
        seed = pool[index]
        refined = (
            refine_box(seed, points[cached_support(index)], solid, score) if approximate else seed
        )
        grown = grow_certified(refined, solid, score if approximate else None)
        alternatives, seen = [], set()
        for variant, trial in (("grown", grown), ("refined", refined), ("seed", seed)):
            # An optimistic local score can overlook burial by overlapping boxes.
            # Keep the untouched seed and pre-growth fit until exact union checks
            # succeed, rather than losing their admissible geometry permanently.
            trial = dict(trial, dimensions=trial["dimensions"].copy())
            if not certified(trial):
                original = trial["dimensions"].copy()
                for retreat in (2e-9, 2e-8, 2e-7):
                    trial["dimensions"] = original - retreat * h
                    trial["volume"] = float(np.prod(trial["dimensions"]))
                    if certified(trial):
                        break
            if not certified(trial):
                continue
            identity = geometry_key(trial)
            if identity not in seen:
                seen.add(identity)
                alternatives.append((variant, trial))
        if not alternatives:
            raise RuntimeError("Incremental candidate failed its strict containment certificate.")
        return alternatives

    selected_alternatives = dict(seed=0, refined=0, grown=0, raw_seed=0)
    raw_seed_checks = 0

    def fallback_raw_seed():
        """Try raw uncovered cells directly, outside the optimistic finalist beam.

        A cube of side <= tolerance/sqrt(3), contained in the observed source
        voxel and containing its uncovered point, lies in that point's tolerance
        ball. It cannot intersect the previous union when that point is farther
        than tolerance from it. Strict envelope/export and exact union checks
        remain mandatory; there is no refinement that can erase this fallback.
        """
        nonlocal raw_seed_checks, exact_checks, rejected_burial
        raw_ids = np.flatnonzero(uncovered)
        if not len(raw_ids):
            return None
        coordinates, first, groups = np.unique(
            np.floor((points[raw_ids] - solid["origin"]) / h).astype(np.int64),
            axis=0,
            return_index=True,
            return_inverse=True,
        )
        mass = np.bincount(groups, weights=weights[raw_ids])
        finalists = []
        side = min(h, tolerance / np.sqrt(3))
        for cell_index in np.argsort(-mass, kind="stable")[:8]:
            cell = coordinates[cell_index]
            if (
                np.any(cell < 0)
                or np.any(cell >= solid["safe"].shape)
                or not solid["safe"][tuple(cell)]
            ):
                continue
            point = points[raw_ids[first[cell_index]]]
            lower = solid["origin"] + cell * h
            center = np.clip(point, lower + side / 2, lower + h - side / 2)
            trial = dict(
                center=center,
                dimensions=np.full(3, side),
                rotation=np.eye(3),
                volume=side**3,
                proposal_kind="raw_uncovered_cell",
            )
            raw_seed_checks += 1
            if not certified(trial):
                for retreat in (2e-9, 2e-8, 2e-7):
                    trial["dimensions"] = np.full(3, side - retreat * h)
                    trial["volume"] = float(np.prod(trial["dimensions"]))
                    if np.all(trial["dimensions"] > 0) and certified(trial):
                        break
            if not np.all(trial["dimensions"] > 0) or not certified(trial):
                continue
            value = score(trial)
            if value > 1e-12:
                finalists.append((value, trial))
        for _, trial in sorted(finalists, key=lambda item: -item[0]):
            ids = support(trial)
            gain = int(uncovered[ids].sum())
            if gain == 0:
                continue
            evaluation = None
            if objective is not None:
                evaluation = objective.evaluate([*boxes, trial], incremental=True)
                exact_checks += 1
                if not objective.feasible(evaluation):
                    rejected_burial += 1
                    continue
            identity = key(trial)
            index = pool_indices.get(identity)
            if index is None:
                index = len(pool)
                unique[identity] = trial
                pool_indices[identity] = index
                pool.append(trial)
                candidates.append(trial)
                enqueue(trial, index)
            return index, trial, ids, gain, evaluation, "raw_seed"
        return None

    reason = "maximum_box_budget"
    irreparable_prefix = objective is not None and not objective.feasible(
        dict(irreparable=objective.irreparable)
    )
    thick_prefix = (
        surface_enabled
        and config.reconstruction_mode == "surface"
        and any(np.min(box["dimensions"]) > 2 * tolerance * (1 + 1e-10) for box in boxes)
    )
    irreparable_prefix |= thick_prefix
    if irreparable_prefix:
        reason = "irreparable_frozen_prefix"
    while not irreparable_prefix and not complete() and len(boxes) < config.max_cuboids:
        attempted.clear()
        preferred = []
        if (
            len(boxes)
            and len(boxes) % config.residual_interval == 0
            and proposed_generation != len(boxes)
        ):
            first_new = len(pool)
            add_residuals()
            preferred = range(first_new, len(pool))
            proposed_generation = len(boxes)
        accepted = None
        # One coarse-pool round, then one explicit residual round. A bad bulk
        # pool cannot trigger thousands of local refinements before new seeds.
        for round_id in range(2):
            screened = prescreen(preferred)
            prescreened_count += len(screened)
            residual = residual_mask()
            residual_gain = {
                index: float(
                    np.dot(residual[cached_support(index)], weights[cached_support(index)])
                )
                for index in screened
            }
            ranked = sorted(
                ((score(pool[index]), index) for index in screened),
                key=lambda item: (
                    -item[0],
                    -float(
                        np.dot(uncovered[cached_support(item[1])], weights[cached_support(item[1])])
                    ),
                    -residual_gain[item[1]],
                    pool[item[1]]["volume"],
                    item[1],
                ),
            )
            finalists = [index for value, index in ranked if value > 1e-12][:finalist_limit]
            # A coarse zero can become useful after shrinking onto its support;
            # retain a bounded rescue slot instead of rejecting every such seed.
            rescue = [index for value, index in ranked if value <= 1e-12][:rescue_limit]
            finalists.extend(rescue)
            finished, seen_finished = [], set()
            for index in finalists:
                alternatives = finish_candidate(index)
                refined_count += 1
                for variant, trial in alternatives:
                    identity = geometry_key(trial)
                    if identity in seen_finished:
                        continue
                    seen_finished.add(identity)
                    value = score(trial)
                    if value > 1e-12:
                        finished.append((value, index, trial, variant))
            # Compare AFTER refinement and export-coordinate retreat, with the
            # exact same score used during prescreening and local optimization.
            for _, index, trial, variant in sorted(
                finished, key=lambda item: (-item[0], item[2]["volume"], item[1])
            ):
                ids = support(trial)
                gain = int(uncovered[ids].sum())
                evaluation = None
                if objective is not None:
                    if gain == 0:
                        objective.refresh_exact_distances()
                    evaluation = objective.evaluate([*boxes, trial], incremental=gain > 0)
                    exact_checks += 1
                    if not objective.feasible(evaluation):
                        rejected_burial += 1
                        continue
                    if gain == 0:
                        old_error = np.average(
                            np.minimum(objective.distances / tolerance, 4), weights=weights
                        )
                        new_error = np.average(
                            np.minimum(evaluation["distances"] / tolerance, 4), weights=weights
                        )
                        if (
                            new_error >= old_error - 1e-12
                            and evaluation["reverse_coverage"] <= objective.reverse_coverage + 1e-12
                        ):
                            continue
                elif gain == 0:
                    continue
                accepted = (index, trial, ids, gain, evaluation, variant)
                break
            if accepted is not None:
                break
            first_new = len(pool)
            if round_id == 0 and add_residuals():
                preferred = range(first_new, len(pool))
                continue
            break
        if accepted is None:
            accepted = fallback_raw_seed()
        if accepted is None:
            remaining_pool = any(i not in attempted and i not in consumed for i in range(len(pool)))
            reason = (
                "bounded_candidate_search_exhausted"
                if remaining_pool
                else "no_positive_surface_gain"
                if surface_enabled
                else "no_positive_point_gain"
            )
            break
        index, box, ids, gain, evaluation, variant = accepted
        selected_alternatives[variant] += 1
        # Refinement can move a seed onto a different patch. Its original geometry
        # remains eligible next generation until that exact proposal is committed.
        if geometry_key(box) == geometry_key(pool[index]):
            consumed.add(index)
        boxes.append(box)
        uncovered[ids] = 0
        support_index.invalidate_scores()
        if objective is not None:
            objective.install(boxes, evaluation)
            # Surface residuals can grow when an exposed face becomes hidden.
            # Refresh priorities instead of treating their old values as upper bounds.
            proxy_weights = np.bincount(
                inverse, weights=weights * residual_mask(), minlength=len(cells)
            )
        # Every round uses current residual priorities. Surface scores may rise
        # when previous faces become hidden, so lazy monotone bounds do not apply.
        heap.clear()
        for index, candidate in enumerate(pool):
            if index not in consumed:
                enqueue(candidate, index)
        if checkpoint is not None:
            checkpoint(boxes, candidates)
        entry = dict(
            cuboids=len(boxes), covered_fraction=float(1 - uncovered.mean()), added_points=gain
        )
        if objective is not None:
            entry.update(
                source_surface_coverage=objective.source_coverage,
                surface_supported_area_fraction=objective.reverse_coverage,
            )
        curve.append(entry)
        progress(
            f"  Incremental {len(boxes)}: {entry['covered_fraction']:.4%} points; {len(boxes) - 1} earlier boxes frozen"
        )
    if not irreparable_prefix and complete():
        reason = "target_reached"
    stats = dict(
        enabled=True,
        mode="incremental",
        pool_candidates=len(pool),
        frozen_prefix_count=len(reference),
        added_count=len(boxes) - len(reference),
        candidates_prescreened=prescreened_count,
        candidates_refined=refined_count,
        selected_alternatives=selected_alternatives,
        raw_uncovered_seed_checks=raw_seed_checks,
        exact_union_checks=exact_checks,
        support_queries=support_index.stats(),
        candidate_search_budgets=dict(
            prescreen=prescreen_limit,
            finalists=finalist_limit,
            rescue=rescue_limit,
            rounds_per_commit=2,
        ),
        spatial_coverage=float(weights[uncovered == 0].sum() / total_weight),
        stopping_reason=reason,
        finite_pool_minimum_proven=False,
        global_minimum_proven=False,
        policy="append only; committed geometry and order are immutable; textures are rebaked on resume",
    )
    if objective is not None:
        stats.update(
            source_surface_coverage=objective.source_coverage,
            surface_supported_area_fraction=objective.reverse_coverage,
            irreparable_source_points=int(objective.irreparable.sum()),
            irreparable_spatial_fraction=float(weights[objective.irreparable].sum() / total_weight),
            rejected_irreparable_candidates=rejected_burial,
            surface_objective="bounded signed-distance and exposed-area trial score; exact clipped-union full-cloud coverage and burial classification before commit",
            surface_distance_work=dict(objective.distance_work),
            surface_support_bounds=objective.support_bounds,
            surface_panel_thickness_limit=(
                2 * tolerance if config.reconstruction_mode == "surface" else None
            ),
            frozen_prefix_violates_surface_thickness=thick_prefix,
        )
    return boxes, curve, uncovered.astype(bool), len(pool), stats
