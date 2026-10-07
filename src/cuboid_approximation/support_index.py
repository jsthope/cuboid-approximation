"""Exact cuboid support queries with bounded reusable spatial work.

A neighborhood is an AABB superset, never a sampled support approximation.
Projection reuse uses a fixed nearby origin; points close to the tolerance
boundary are recomputed with the direct expression to preserve its decisions.
"""

from collections import OrderedDict

import numpy as np

from .geometry import PointIndex


def geometry_key(box):
    """Exact immutable key, including numerical retreats smaller than voxel units."""
    return tuple(np.asarray(box[name]).tobytes() for name in ("center", "dimensions", "rotation"))


class BoxSupportIndex:
    def __init__(
        self,
        points,
        tolerance,
        max_cache_bytes=64 * 1024**2,
        *,
        proxy_index=None,
        point_order=None,
        point_indptr=None,
        proxy_radius=0.0,
        neighborhood_margin=None,
    ):
        self.points = np.asarray(points)
        self.tolerance = float(tolerance)
        self.max_cache_bytes = max(0, int(max_cache_bytes))
        self.index = proxy_index if proxy_index is not None else PointIndex(self.points)
        self.point_order, self.point_indptr = point_order, point_indptr
        self.proxy_radius = float(proxy_radius)
        self.neighborhood_margin = (
            2 * self.tolerance if neighborhood_margin is None else float(neighborhood_margin)
        )
        self.cache = OrderedDict()
        self.cache_bytes = self.peak_cache_bytes = 0
        self.next_neighborhood = 0
        self.neighborhood_keys = OrderedDict()
        self._counts = dict(
            queries=0,
            support_cache_hits=0,
            neighborhood_hits=0,
            broadphase_queries=0,
            projection_cache_hits=0,
            projection_computations=0,
            exact_distance_queries=0,
            boundary_rechecks=0,
            evictions=0,
            score_cache_hits=0,
            score_evaluations=0,
            proxy_cache_hits=0,
            proxy_evaluations=0,
        )
        self._coordinate_scale = float(np.max(np.abs(self.points), initial=0))
        self._id_dtype = np.int32 if len(self.points) < 2**31 else np.int64

    def _get(self, key):
        entry = self.cache.get(key)
        if entry is None:
            return None
        self.cache.move_to_end(key)
        return entry[0]

    def _remove(self, key):
        _, size = self.cache.pop(key)
        self.cache_bytes -= size
        self._counts["evictions"] += 1
        if key[0] == "neighborhood":
            self.neighborhood_keys.pop(key, None)
            # A projection without its neighborhood cannot be reused.
            for dependent in list(self.cache):
                if dependent[0] == "projection" and dependent[1] == key[1]:
                    self._remove(dependent)

    def _store(self, key, value, size):
        # Budget Python bookkeeping and immutable geometry keys as well as array
        # payloads. Even cached empty supports consume a nonzero bounded slot.
        def key_bytes(part):
            return (
                sum(key_bytes(v) for v in part)
                if isinstance(part, tuple)
                else len(part)
                if isinstance(part, bytes)
                else 0
            )

        size += 1024 + key_bytes(key)
        if size > self.max_cache_bytes or self.max_cache_bytes == 0:
            return False
        if key in self.cache:
            self._remove(key)
        while self.cache_bytes + size > self.max_cache_bytes:
            self._remove(next(iter(self.cache)))
        if key[0] == "projection" and ("neighborhood", key[1]) not in self.cache:
            return False
        self.cache[key] = (value, size)
        if key[0] == "neighborhood":
            self.neighborhood_keys[key] = None
            while len(self.neighborhood_keys) > 16:
                self._remove(next(iter(self.neighborhood_keys)))
        self.cache_bytes += size
        self.peak_cache_bytes = max(self.peak_cache_bytes, self.cache_bytes)
        return True

    def _broadphase(self, center, half):
        self._counts["broadphase_queries"] += 1
        ids = self.index.query(center, half + self.proxy_radius)
        if self.point_indptr is not None:
            counts = self.point_indptr[ids + 1] - self.point_indptr[ids]
            starts = np.cumsum(counts) - counts
            positions = (
                np.repeat(self.point_indptr[ids], counts)
                + np.arange(int(counts.sum()))
                - np.repeat(starts, counts)
            )
            ids = self.point_order[positions]
        # Make the recorded AABB exact even when proxy cells straddle its edges.
        return ids[np.all(np.abs(self.points[ids] - center) <= half, axis=1)].astype(
            self._id_dtype, copy=False
        )

    def _neighborhood(self, center, half):
        # A tiny cache normally has only a few neighborhoods. Prefer the smallest
        # enclosing one to avoid projecting a whole object for a local face move.
        suitable = []
        for key in self.neighborhood_keys:
            value = self.cache[key][0]
            origin, bounds, ids = value
            if np.all(np.abs(center - origin) + half <= bounds):
                suitable.append((len(ids), key, value))
        if suitable:
            _, key, value = min(suitable, key=lambda item: (item[0], item[1]))
            self.cache.move_to_end(key)
            self.neighborhood_keys.move_to_end(key)
            self._counts["neighborhood_hits"] += 1
            return key[1], value
        # Reserve movement room, but do not change the final point distance test.
        bounds = half + np.maximum(self.neighborhood_margin, half * 0.125)
        origin = np.asarray(center).copy()
        ids = self._broadphase(origin, bounds)
        identity = self.next_neighborhood
        self.next_neighborhood += 1
        value = (origin, bounds, ids)
        self._store(("neighborhood", identity), value, origin.nbytes + bounds.nbytes + ids.nbytes)
        return identity, value

    def query(self, box):
        """Return read-only ascending point IDs at distance <= tolerance."""
        self._counts["queries"] += 1
        identity = geometry_key(box)
        key = ("support", identity)
        cached = self._get(key)
        if cached is not None:
            self._counts["support_cache_hits"] += 1
            return cached
        center, dimensions, frame = (
            np.asarray(box[name]) for name in ("center", "dimensions", "rotation")
        )
        scale = max(
            1.0,
            self._coordinate_scale,
            float(np.max(np.abs(center))),
            float(np.max(dimensions)),
            self.tolerance,
        )
        roundoff = 64 * np.finfo(float).eps * scale
        half = np.abs(frame) @ (dimensions / 2) + self.tolerance + roundoff
        neighborhood_id, (origin, _, ids) = self._neighborhood(center, half)
        projection_key = ("projection", neighborhood_id, frame.tobytes())
        projected = self._get(projection_key)
        if projected is None:
            projected = (self.points[ids] - origin) @ frame
            self._counts["projection_computations"] += 1
            self._store(projection_key, projected, projected.nbytes)
        else:
            self._counts["projection_cache_hits"] += 1
        local = projected + (origin - center) @ frame
        distance = np.linalg.norm(np.maximum(np.abs(local) - dimensions / 2, 0), axis=1)
        self._counts["exact_distance_queries"] += 1
        # Translation of already rounded projections can change the last few
        # bits. Re-evaluate every ambiguous point with the public direct formula.
        near = np.abs(distance - self.tolerance) <= roundoff
        if near.any():
            direct = (self.points[ids[near]] - center) @ frame
            distance[near] = np.linalg.norm(np.maximum(np.abs(direct) - dimensions / 2, 0), axis=1)
            self._counts["boundary_rechecks"] += int(near.sum())
        result = np.sort(ids[distance <= self.tolerance])
        result.setflags(write=False)
        self._store(key, result, result.nbytes)
        return result

    def proxy_support(self, box, compute):
        """Cache immutable proposal-cell support alongside source-point queries."""
        key = ("proxy", geometry_key(box))
        cached = self._get(key)
        if cached is not None:
            self._counts["proxy_cache_hits"] += 1
            return cached
        self._counts["proxy_evaluations"] += 1
        value = np.asarray(compute(box)).astype(self._id_dtype, copy=False)
        value.setflags(write=False)
        self._store(key, value, value.nbytes)
        return value

    def score(self, box, compute):
        """Share this byte budget with scalar scores for the current generation."""
        key = ("score", geometry_key(box))
        cached = self._get(key)
        if cached is not None:
            self._counts["score_cache_hits"] += 1
            return cached
        self._counts["score_evaluations"] += 1
        value = float(compute(box))
        self._store(key, value, 8)
        return value

    def invalidate_scores(self):
        """Coverage changes invalidate scores; geometric supports remain exact."""
        for key in list(self.cache):
            if key[0] == "score":
                _, size = self.cache.pop(key)
                self.cache_bytes -= size

    def stats(self):
        return dict(
            self._counts,
            cache_bytes=self.cache_bytes,
            peak_cache_bytes=self.peak_cache_bytes,
            cache_limit_bytes=self.max_cache_bytes,
            cache_entries=len(self.cache),
            policy="exact support, conservative AABB reuse, bounded LRU projections; boundary points recomputed directly",
        )
