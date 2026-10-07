"""Shared box topology and continuous distances to the boundary of a box union."""

from dataclasses import dataclass, field
import itertools
import heapq
from typing import TypedDict

import numpy as np
from scipy.spatial import cKDTree


class Box(TypedDict):
    center: np.ndarray
    dimensions: np.ndarray
    rotation: np.ndarray
    volume: float


CORNER_SIGNS = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
BOX_QUADS = np.array(
    [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
)
BOX_TRIANGLES = np.concatenate([BOX_QUADS[:, [0, 1, 2]], BOX_QUADS[:, [0, 2, 3]]])


def box_surface_distance(points, box: Box):
    q = np.abs((points - box["center"]) @ box["rotation"]) - box["dimensions"] / 2
    return np.abs(np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0))


def _split(poly, normal, center, bound, epsilon):
    """Split a convex polygon by a plane; preserve its winding."""
    distance = (poly - center) @ normal - bound
    distance[np.abs(distance) <= epsilon] = 0
    if np.all(distance <= 0):
        return poly, np.empty((0, 3))
    if np.all(distance >= 0):
        return np.empty((0, 3)), poly
    inside, outside = [], []
    for i, p in enumerate(poly):
        j = (i + 1) % len(poly)
        (inside if distance[i] <= 0 else outside).append(p)
        if distance[i] == 0:
            outside.append(p)
        if distance[i] * distance[j] < 0:
            cross = p + distance[i] / (distance[i] - distance[j]) * (poly[j] - p)
            inside.append(cross)
            outside.append(cross)
    return np.asarray(inside).reshape(-1, 3), np.asarray(outside).reshape(-1, 3)


def exposed_triangles(boxes):
    """Subtract every other box from each face, including internal contact faces.

    Clipping produces convex fragments. Coplanar exterior faces have one owner,
    so duplicate boxes do not duplicate area. Work near the source origin.
    """
    if not boxes:
        return np.empty((0, 3, 3))
    origin = boxes[0]["center"]
    local = [dict(b, center=b["center"] - origin) for b in boxes]
    extent = max(np.max(b["dimensions"]) for b in boxes)
    epsilon = extent * 1e-11
    corners = [(CORNER_SIGNS * b["dimensions"] / 2) @ b["rotation"].T + b["center"] for b in local]
    bounds = [(p.min(0), p.max(0)) for p in corners]
    triangles = []
    for i, vertices in enumerate(corners):
        for quad in BOX_QUADS:
            face = vertices[quad]
            normal = np.cross(face[1] - face[0], face[3] - face[0])
            normal /= np.linalg.norm(normal)
            fragments = [face]
            for j, other in enumerate(local):
                if i == j or not fragments:
                    continue
                if np.any(face.max(0) < bounds[j][0] - epsilon) or np.any(
                    face.min(0) > bounds[j][1] + epsilon
                ):
                    continue
                # Co-oriented coincident exterior faces belong to the first box.
                same_face = any(
                    normal @ (sign * other["rotation"][:, axis]) > 1 - 1e-10
                    and np.max(
                        np.abs(
                            (face - other["center"]) @ (sign * other["rotation"][:, axis])
                            - other["dimensions"][axis] / 2
                        )
                    )
                    <= epsilon
                    for axis in range(3)
                    for sign in (-1, 1)
                )
                if same_face and j > i:
                    continue
                remaining = []
                for fragment in fragments:
                    inside = fragment
                    for axis in range(3):
                        for sign in (-1, 1):
                            if len(inside) < 3:
                                break
                            inside, outside = _split(
                                inside,
                                sign * other["rotation"][:, axis],
                                other["center"],
                                other["dimensions"][axis] / 2,
                                epsilon,
                            )
                            if len(outside) >= 3:
                                remaining.append(outside)
                    # The final inside fragment is covered by the other box.
                fragments = remaining
            for poly in fragments:
                for k in range(1, len(poly) - 1):
                    tri = poly[[0, k, k + 1]]
                    if np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])) > epsilon**2:
                        triangles.append(tri + origin)
    return np.asarray(triangles).reshape(-1, 3, 3)


def triangle_distance_squared(points, triangles):
    """Pairwise point/triangle distance, broadcasting a single triangle if needed."""
    a, b, c = (triangles[..., i, :] for i in range(3))
    ab, ac, ap = b - a, c - a, points - a
    normal = np.cross(ab, ac)
    squared = np.sum(normal * normal, axis=-1)
    projection = np.sum(ap * normal, axis=-1) / squared
    projected = ap - projection[..., None] * normal
    d00, d01, d11 = np.sum(ab * ab, -1), np.sum(ab * ac, -1), np.sum(ac * ac, -1)
    d20, d21 = np.sum(projected * ab, -1), np.sum(projected * ac, -1)
    denom = squared
    u, v = (d11 * d20 - d01 * d21) / denom, (d00 * d21 - d01 * d20) / denom
    inside = (u >= -1e-12) & (v >= -1e-12) & (u + v <= 1 + 1e-12)
    result = np.where(inside, projection**2 * squared, np.inf)
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        t = np.clip(np.sum((points - start) * edge, -1) / np.sum(edge * edge, -1), 0, 1)
        result = np.minimum(result, np.sum((points - start - t[..., None] * edge) ** 2, -1))
    return result


@dataclass
class SurfaceMesh:
    triangles: np.ndarray
    nodes: list = field(default_factory=list, init=False)

    def __post_init__(self):
        self.origin = self.triangles[0, 0].copy() if len(self.triangles) else np.zeros(3)
        self.triangles = self.triangles - self.origin
        self.centers = self.triangles.mean(1)
        self.ab = self.triangles[:, 1] - self.triangles[:, 0]
        self.ac = self.triangles[:, 2] - self.triangles[:, 0]
        self.normal = np.cross(self.ab, self.ac)
        self.ab2 = np.einsum("ij,ij->i", self.ab, self.ab)
        self.ac2 = np.einsum("ij,ij->i", self.ac, self.ac)
        self.abac = np.einsum("ij,ij->i", self.ab, self.ac)
        self.normal2 = np.einsum("ij,ij->i", self.normal, self.normal)
        self.tree = cKDTree(self.centers)
        if len(self.triangles):
            self._build(np.arange(len(self.triangles)))

    def _build(self, ids):
        tri = self.triangles[ids]
        low, high = tri.min((0, 1)), tri.max((0, 1))
        index = len(self.nodes)
        self.nodes.append(None)
        if len(ids) <= 8:
            self.nodes[index] = (low, high, ids, None)
        else:
            axis = np.ptp(self.centers[ids], axis=0).argmax()
            order = ids[np.argsort(self.centers[ids, axis], kind="stable")]
            mid = len(order) // 2
            self.nodes[index] = (
                low,
                high,
                None,
                (self._build(order[:mid]), self._build(order[mid:])),
            )
        return index

    def _distance_triangle(self, points, t):
        delta = points - self.triangles[t, 0]
        x, y, z = delta @ self.ab[t], delta @ self.ac[t], delta @ self.normal[t]
        u = (self.ac2[t] * x - self.abac[t] * y) / self.normal2[t]
        v = (self.ab2[t] * y - self.abac[t] * x) / self.normal2[t]
        inside = (u >= -1e-12) & (v >= -1e-12) & (u + v <= 1 + 1e-12)
        result = z * z / self.normal2[t]
        if np.all(inside):
            return result
        ids = np.flatnonzero(~inside)
        q = delta[ids]
        # Edge distances use actual residual vectors, avoiding cancellation close to edges.
        a = np.clip(x[ids] / self.ab2[t], 0, 1)
        b = np.clip(y[ids] / self.ac2[t], 0, 1)
        da = q - a[:, None] * self.ab[t]
        db = q - b[:, None] * self.ac[t]
        bc = self.ac[t] - self.ab[t]
        dc = q - self.ab[t]
        c = np.clip((dc @ bc) / (bc @ bc), 0, 1)
        dc -= c[:, None] * bc
        result[ids] = np.minimum(
            np.minimum(np.einsum("ij,ij->i", da, da), np.einsum("ij,ij->i", db, db)),
            np.einsum("ij,ij->i", dc, dc),
        )
        return result

    def distances(self, points):
        result = np.full(len(points), np.inf)
        if not self.nodes:
            return result
        for start in range(0, len(points), 4096):
            p = points[start : start + 4096] - self.origin
            _, nearest = self.tree.query(p)
            best = triangle_distance_squared(p, self.triangles[nearest])
            stack = [(0, np.arange(len(p)))]
            while stack:
                node, ids = stack.pop()
                low, high, triangles, children = self.nodes[node]
                lower = np.sum(np.maximum(np.maximum(low - p[ids], p[ids] - high), 0) ** 2, 1)
                ids = ids[lower <= best[ids]]
                if not len(ids):
                    continue
                if triangles is not None:
                    for t in triangles:
                        best[ids] = np.minimum(best[ids], self._distance_triangle(p[ids], t))
                else:
                    stack.extend((child, ids) for child in children)
            result[start : start + len(p)] = np.sqrt(np.maximum(best, 0))
        return result

    @property
    def areas(self):
        return (
            np.linalg.norm(
                np.cross(
                    self.triangles[:, 1] - self.triangles[:, 0],
                    self.triangles[:, 2] - self.triangles[:, 0],
                ),
                axis=1,
            )
            / 2
        )

    def samples(self, budget=4096):
        areas = self.areas
        if not len(areas):
            return np.empty((0, 3)), np.empty(0)
        samples, weights = [], []
        for tri, area in zip(self.triangles, areas):
            count = max(1, int(np.ceil(budget * area / areas.sum())))
            # Deterministic equal-area low-discrepancy barycentric samples.
            u = np.sqrt((np.arange(count) + 0.5) / count)
            v = (np.arange(count) * 0.6180339887498949 + 0.5) % 1
            samples.append(
                (1 - u[:, None]) * tri[0]
                + (u * (1 - v))[:, None] * tri[1]
                + (u * v)[:, None] * tri[2]
                + self.origin
            )
            weights.append(np.full(count, area / count))
        return np.concatenate(samples), np.concatenate(weights)

    def support_bounds(
        self,
        source,
        tolerance,
        *,
        target=None,
        area_fraction_tolerance=0.002,
        max_evaluations=65536,
        max_depth=16,
    ):
        """Bound the area within ``tolerance`` of the discrete source points.

        Nearest-source distance is 1-Lipschitz: its value at a triangle's
        centroid, plus/minus the farthest vertex radius, bounds the complete
        triangle. Ambiguous triangles are split into four equal-area children,
        largest first. Unvisited area always remains in the upper bound.

        These are conservative geometric bounds on the computed mesh, subject
        to floating-point roundoff, not confidence intervals for a quadrature
        estimate or a guarantee about an unobserved continuous source surface.
        Work and pending storage are bounded by the evaluation budget plus the
        input mesh size. A straddled target is explicitly inconclusive.
        """
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Surface support tolerance must be finite and nonnegative.")
        if target is not None and not 0 <= target <= 1:
            raise ValueError("Surface support target must be between zero and one.")
        if not 0 <= area_fraction_tolerance <= 1:
            raise ValueError("Ambiguous area tolerance must be between zero and one.")
        if max_evaluations < 0 or max_depth < 0:
            raise ValueError("Surface support work budgets must be nonnegative.")
        areas = self.areas
        total = float(areas.sum())
        tree = source if isinstance(source, cKDTree) else cKDTree(np.asarray(source))
        supported = unsupported = depth_limited_area = 0.0
        evaluations = deepest = 0
        serial = itertools.count()
        pending = [
            (-float(area), next(serial), 0, triangle)
            for triangle, area in zip(self.triangles, areas)
            if area > 0
        ]
        heapq.heapify(pending)
        reason = "empty_surface" if not total else "work_budget"
        if not tree.n:
            unsupported = total
            pending.clear()
            reason = "empty_source"
        while pending and evaluations < max_evaluations:
            lower = supported / total
            upper = 1 - unsupported / total
            if target is not None and (lower >= target or upper < target):
                reason = "target_decided"
                break
            if upper - lower <= area_fraction_tolerance:
                reason = "area_tolerance"
                break
            batch = [
                heapq.heappop(pending)
                for _ in range(min(1024, len(pending), max_evaluations - evaluations))
            ]
            triangles = np.asarray([entry[3] for entry in batch])
            centers = triangles.mean(axis=1)
            distances, nearest = tree.query(centers + self.origin)
            radii = np.linalg.norm(triangles - centers[:, None, :], axis=2).max(axis=1)
            # Every vertex in one source-point ball also encloses the complete
            # convex triangle. This tightens the Lipschitz upper bound cheaply.
            source_delta = tree.data[nearest] - self.origin
            vertex_upper = np.linalg.norm(
                triangles - source_delta[:, None, :], axis=2
            ).max(axis=1)
            numeric_scale = np.maximum(
                np.max(np.abs(triangles), axis=(1, 2)),
                max(float(np.max(np.abs(self.origin))), float(tolerance)),
            )
            slack = np.finfo(float).eps * 32 * numeric_scale
            inside = np.minimum(distances + radii, vertex_upper) + slack <= tolerance
            outside = distances - radii - slack > tolerance
            evaluations += len(batch)
            for index, (negative_area, _, depth, triangle) in enumerate(batch):
                area = -negative_area
                deepest = max(deepest, depth)
                if inside[index]:
                    supported += area
                elif outside[index]:
                    unsupported += area
                elif depth < max_depth:
                    a, b, c = triangle
                    ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
                    for child in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)):
                        heapq.heappush(
                            pending, (-area / 4, next(serial), depth + 1, np.asarray(child))
                        )
                else:
                    depth_limited_area += area
        if not pending and total and depth_limited_area:
            reason = "depth_budget"
        elif not pending and total and reason != "empty_source":
            reason = "classified"
        lower = float(np.clip(supported / total, 0, 1)) if total else 0.0
        upper = float(np.clip(1 - unsupported / total, lower, 1)) if total else 0.0
        if total and not pending and not depth_limited_area:
            # All area was classified. In particular, a fully supported mesh
            # must reach target=1 despite summation roundoff over many leaves.
            lower = upper = supported / (supported + unsupported)
        status = (
            "bounded"
            if target is None
            else "reached"
            if lower >= target
            else "not_reached"
            if upper < target
            else "inconclusive"
        )
        return dict(
            lower_fraction=lower,
            upper_fraction=upper,
            ambiguous_area_fraction=float(upper - lower),
            total_area=total,
            status=status,
            stopping_reason=reason,
            evaluated_triangles=evaluations,
            maximum_depth=deepest,
            max_evaluations=int(max_evaluations),
            area_fraction_tolerance=float(area_fraction_tolerance),
            method="Adaptive triangle bounds from 1-Lipschitz nearest-source distances",
        )


@dataclass
class PointIndex:
    """Exact AABB broad phase; sorted coordinates avoid enclosing-sphere overfetch."""

    points: np.ndarray

    def __post_init__(self):
        self.orders = [np.argsort(self.points[:, a], kind="stable") for a in range(3)]
        self.sorted = [self.points[o, a] for a, o in enumerate(self.orders)]

    def query(self, center, half):
        ranges = [
            np.searchsorted(s, [center[a] - half[a], center[a] + half[a]], side="left")
            for a, s in enumerate(self.sorted)
        ]
        # Include the upper boundary, including repeated coordinates.
        for a, s in enumerate(self.sorted):
            ranges[a][1] = np.searchsorted(s, center[a] + half[a], side="right")
        axis = int(np.argmin([b - a for a, b in ranges]))
        low, high = ranges[axis]
        ids = self.orders[axis][low:high]
        return ids[np.all(np.abs(self.points[ids] - center) <= half, axis=1)]
