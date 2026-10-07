"""Exactness and bounded-work contracts for repeated cuboid support queries."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from cuboid_approximation.cuboids import distance_to_box
from cuboid_approximation.geometry import PointIndex
from cuboid_approximation.support_index import BoxSupportIndex
from test_surface_selection import box


class SupportIndexTests(unittest.TestCase):
    def assert_support(self, index, candidate):
        expected = np.flatnonzero(distance_to_box(index.points, candidate) <= index.tolerance)
        np.testing.assert_array_equal(index.query(candidate), expected)
        return expected

    def test_rotated_elongated_and_translated_boxes_match_brute_force(self):
        rng = np.random.default_rng(732)
        points = rng.uniform(-2, 2, (8000, 3))
        index = BoxSupportIndex(points, 0.06, max_cache_bytes=2 * 1024**2)
        for i in range(70):
            candidate = box(
                rng.uniform(-0.3, 0.3, 3), [2.8, 0.04, 0.18] if i % 2 else [1.1, 0.7, 0.3]
            )
            candidate["rotation"] = Rotation.random(random_state=rng).as_matrix()
            self.assert_support(index, candidate)
            self.assert_support(
                index, dict(candidate, center=candidate["center"] + [0.003, -0.002, 0.001])
            )
        self.assertLessEqual(index.stats()["peak_cache_bytes"], 2 * 1024**2)

    def test_boundary_duplicates_and_large_coordinates_preserve_direct_decisions(self):
        frame = Rotation.from_euler("xyz", [23, 42, 17], degrees=True).as_matrix()
        center = np.array([1e8, -2e8, 3e8])
        candidate = box(center, [1.8, 0.02, 0.07])
        candidate["rotation"] = frame
        local = np.array([[0.96, 0, 0], [-0.96, 0, 0], [0, 0.07, 0], [0, 0, 0.095], [0, 0, 0]])
        points = np.repeat(local @ frame.T + center, 20, axis=0)
        index = BoxSupportIndex(points, 0.06, max_cache_bytes=256 * 1024)
        self.assert_support(index, candidate)
        self.assert_support(index, dict(candidate, center=center + [1e-6, -1e-6, 0]))
        self.assertGreater(index.stats()["boundary_rechecks"], 0)
        self.assertGreater(index.stats()["projection_cache_hits"], 0)

    def test_proxy_csr_neighborhood_is_a_conservative_full_point_query(self):
        rng = np.random.default_rng(861)
        points = np.vstack(
            (rng.uniform(-1, 1, (2500, 3)), np.repeat([[0.12, 0.3, -0.5]], 100, axis=0))
        )
        width = 0.12
        cells, inverse = np.unique(
            np.floor(points / width).astype(int), axis=0, return_inverse=True
        )
        proxy = (cells + 0.5) * width
        index = BoxSupportIndex(
            points,
            0.04,
            max_cache_bytes=512 * 1024,
            proxy_index=PointIndex(proxy),
            point_order=np.argsort(inverse, kind="stable"),
            point_indptr=np.r_[0, np.cumsum(np.bincount(inverse))],
            proxy_radius=np.sqrt(3) * width / 2,
        )
        for angle in range(0, 90, 9):
            candidate = box([0.1, 0.2, -0.3], [0.15, 0.9, 0.05])
            candidate["rotation"] = Rotation.from_euler("z", angle, degrees=True).as_matrix()
            self.assert_support(index, candidate)

    def test_reuse_reduces_broadphase_and_projection_work_without_changing_results(self):
        rng = np.random.default_rng(68)
        points = rng.uniform(-1, 1, (12000, 3))
        cached = BoxSupportIndex(points, 0.03, max_cache_bytes=2 * 1024**2, neighborhood_margin=0.2)
        uncached = BoxSupportIndex(points, 0.03, max_cache_bytes=0, neighborhood_margin=0.2)
        rotation = Rotation.from_euler("xyz", [20, 10, 30], degrees=True).as_matrix()
        for shift in np.linspace(0, 0.04, 12):
            candidate = box([shift, 0, 0], [0.7, 0.5, 0.1])
            candidate["rotation"] = rotation
            expected = self.assert_support(uncached, candidate)
            np.testing.assert_array_equal(cached.query(candidate), expected)
            np.testing.assert_array_equal(cached.query(candidate), expected)
        stats = cached.stats()
        self.assertEqual(stats["broadphase_queries"], 1)
        self.assertEqual(stats["projection_computations"], 1)
        self.assertEqual(stats["support_cache_hits"], 12)
        self.assertEqual(stats["exact_distance_queries"], 12)
        self.assertEqual(uncached.stats()["projection_computations"], 12)

    def test_empty_supports_and_evictions_stay_within_one_byte_budget(self):
        index = BoxSupportIndex(np.zeros((10, 3)), 0.01, max_cache_bytes=8192)
        for step in range(80):
            candidate = box([10 + step, 0, 0], [0.1, 0.1, 0.1])
            self.assert_support(index, candidate)
            index.score(candidate, lambda trial: float(len(index.query(trial))))
            self.assertLessEqual(index.stats()["cache_bytes"], 8192)
        self.assertLessEqual(index.stats()["peak_cache_bytes"], 8192)
        self.assertLess(index.stats()["cache_entries"], 10)
        self.assertGreater(index.stats()["evictions"], 0)

    def test_generation_invalidation_retains_geometric_support(self):
        index = BoxSupportIndex(np.array([[0.0, 0, 0], [1, 0, 0]]), 0.01, max_cache_bytes=8192)
        candidate = box([0, 0, 0], [0.1, 0.1, 0.1])
        state = [2.0]

        def compute(trial):
            return state[0] * len(index.query(trial))

        self.assertEqual(index.score(candidate, compute), 2.0)
        self.assertEqual(index.score(candidate, compute), 2.0)
        state[0] = 3.0
        index.invalidate_scores()
        self.assertEqual(index.score(candidate, compute), 3.0)
        self.assertEqual(index.stats()["broadphase_queries"], 1)
        self.assertEqual(index.stats()["score_cache_hits"], 1)
        self.assertGreater(index.stats()["support_cache_hits"], 0)


if __name__ == "__main__":
    unittest.main()
