"""Incremental threshold certificates must match full clipped-union evaluation."""

import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from cuboid_approximation.approximation import (
    component_coverage,
    spatial_components,
    spatial_weights,
)
from cuboid_approximation.cuboids import CuboidConfig, select_incremental_boxes
from cuboid_approximation.geometry import SurfaceMesh
from cuboid_approximation.selection import UnionSurfaceObjective
from test_surface_selection import box, solid


class IncrementalSurfaceTests(unittest.TestCase):
    def objective(self, points, tolerance=0.08):
        weights = spatial_weights(points, tolerance)
        return UnionSurfaceObjective(
            points,
            weights,
            tolerance,
            CuboidConfig(),
            np.arange(len(points)),
            weights,
            spatial_components(points, tolerance),
        )

    def assert_classes(self, incremental, full, objective):
        np.testing.assert_array_equal(
            incremental["distances"] <= objective.tolerance,
            full["distances"] <= objective.tolerance,
        )
        np.testing.assert_array_equal(incremental["irreparable"], full["irreparable"])
        np.testing.assert_array_equal(incremental["inside"], full["inside"])
        self.assertEqual(incremental["source_coverage"], full["source_coverage"])
        self.assertEqual(incremental["reverse_coverage"], full["reverse_coverage"])
        np.testing.assert_array_equal(
            component_coverage(
                incremental["distances"] <= objective.tolerance,
                objective.component_labels,
                objective.weights,
            ),
            component_coverage(
                full["distances"] <= objective.tolerance,
                objective.component_labels,
                objective.weights,
            ),
        )

    def test_random_overlapping_rotated_unions_match_full_classifications(self):
        rng = np.random.default_rng(83)
        points = rng.uniform(-2, 2, (2000, 3))
        objective = self.objective(points)
        boxes = []
        for step in range(9):
            candidate = box(rng.uniform(-0.7, 0.7, 3), rng.uniform(0.2, 1.4, 3))
            candidate["rotation"] = Rotation.random(random_state=rng).as_matrix()
            boxes.append(candidate)
            incremental = objective.evaluate(boxes, incremental=True)
            full = objective.evaluate(boxes)
            self.assert_classes(incremental, full, objective)
            objective.install(boxes, incremental)
        self.assertGreater(objective.distance_work["points_reused"], len(points))

    def test_boundary_duplicates_large_coordinates_and_extent_fallback(self):
        offset = np.array([1e8, -2e8, 3e8])
        p = (
            np.array(
                [
                    [0.58, 0, 0],
                    [0.5, 0, 0],
                    [0.500000000001, 0, 0],
                    [0, 0, 0],
                    [1.42, 0, 0],
                    [2.58, 0, 0],
                    [10, 0, 0],
                ]
            )
            + offset
        )
        points = np.repeat(p, 12, axis=0)
        objective = self.objective(points)
        boxes = []
        for center, dimensions in (
            ([0, 0, 0], [1, 1, 1]),
            ([2, 0, 0], [1, 0.8, 0.8]),
            ([0, 2, 0], [2, 1, 1]),
        ):
            boxes.append(box(offset + center, dimensions))
            incremental = objective.evaluate(boxes, incremental=True)
            full = objective.evaluate(boxes)
            self.assert_classes(incremental, full, objective)
            objective.install(boxes, incremental)
        self.assertTrue(incremental["distances_exact"])

    def test_queries_only_near_added_box_and_refreshes_distance_magnitudes(self):
        points = np.vstack(
            (np.zeros((50, 3)), np.tile([10.0, 0, 0], (50, 1)), np.tile([50.0, 0, 0], (500, 1)))
        )
        objective = self.objective(points)
        initial = box([0, 0, 0], [1, 1, 1])
        objective.install([initial], objective.evaluate([initial]))
        added = box([10, 0, 0], [0.5, 0.5, 0.5])
        calls = []
        original = SurfaceMesh.distances

        def distances(mesh, source):
            calls.append(len(source))
            return original(mesh, source)

        with patch.object(SurfaceMesh, "distances", distances):
            evaluation = objective.evaluate([initial, added], incremental=True)
        self.assertEqual(calls, [50])
        self.assertFalse(evaluation["distances_exact"])
        objective.install([initial, added], evaluation)
        objective.refresh_exact_distances()
        np.testing.assert_array_equal(objective.distances, objective.mesh.distances(points))
        self.assertEqual(objective.distance_work["exact_refreshes"], 1)

    def test_old_threshold_ties_are_rechecked_outside_new_box_neighborhood(self):
        points = np.repeat([[-1.75, -1.75, 0.18]], 12, axis=0)
        objective = self.objective(points)
        initial = box([0, 0, 0], [4, 4, 0.2])
        objective.install([initial], objective.evaluate([initial]))
        added = box([1.75, 1.75, 0.15], [0.6, 0.6, 0.2])
        old_count = objective.distance_work["points_evaluated"]
        incremental = objective.evaluate([initial, added], incremental=True)
        self.assertEqual(objective.distance_work["points_evaluated"] - old_count, len(points))
        self.assert_classes(incremental, objective.evaluate([initial, added]), objective)

    def test_selector_is_identical_with_full_and_incremental_distance_evaluation(self):
        u, v = np.meshgrid(np.linspace(-0.3, 0.3, 10), np.linspace(-0.3, 0.3, 10))
        patch_points = np.column_stack((u.ravel(), v.ravel(), np.zeros(u.size)))
        points = np.vstack((patch_points, patch_points + [1, 0, 0]))
        candidates = [box([0, 0, 0], [0.6, 0.6, 0.001]), box([1, 0, 0], [0.6, 0.6, 0.001])]
        config = CuboidConfig(max_cuboids=3, reconstruction_mode="surface")
        envelope = solid()
        envelope["safe"][:] = False
        envelope["safe"][16:24, 16:24, 18:22] = True
        envelope["safe"][26:34, 16:24, 18:22] = True
        envelope["reference"] = envelope["safe"]
        original = UnionSurfaceObjective.evaluate

        def full_only(objective, boxes, **kwargs):
            return original(objective, boxes)

        with patch.object(UnionSurfaceObjective, "evaluate", full_only):
            full = select_incremental_boxes(
                candidates.copy(), [], points, 0.06, config, envelope, lambda _: None
            )
        incremental = select_incremental_boxes(
            candidates.copy(), [], points, 0.06, config, envelope, lambda _: None
        )
        self.assertEqual(len(full[0]), len(incremental[0]))
        for a, b in zip(full[0], incremental[0]):
            for name in ("center", "dimensions", "rotation"):
                np.testing.assert_array_equal(a[name], b[name])
        self.assertEqual(full[1], incremental[1])
        self.assertEqual(full[4]["stopping_reason"], incremental[4]["stopping_reason"])
        self.assertGreater(incremental[4]["surface_distance_work"]["points_reused"], 0)


if __name__ == "__main__":
    unittest.main()
