"""Regressions for irreversible surface errors during append-only selection."""

import unittest
from unittest.mock import patch

import numpy as np

from cuboid_approximation.cuboids import CuboidConfig, select_incremental_boxes
from cuboid_approximation.geometry import box_surface_distance
from cuboid_approximation.selection import UnionSurfaceObjective, adaptive_face_samples


def box(center, dimensions):
    dimensions = np.asarray(dimensions, float)
    return dict(
        center=np.asarray(center, float),
        dimensions=dimensions,
        rotation=np.eye(3),
        volume=float(np.prod(dimensions)),
    )


def solid():
    safe = np.ones((40, 40, 40), bool)
    return dict(
        safe=safe, reference=safe, origin=np.full(3, -2.0), voxel_size=0.1, approximation={}
    )


class SurfaceSelectionTests(unittest.TestCase):
    def test_adaptive_face_budget_stays_bounded_on_extremely_thin_faces(self):
        samples, areas = adaptive_face_samples(
            box([0, 0, 0], [1000, 0.00001, 0.00001]), 1e-6, budget=768
        )
        self.assertLessEqual(len(samples), 792)
        self.assertAlmostEqual(areas.sum(), 0.0400000002)

    def test_full_cloud_burial_is_checked_before_checkpoint(self):
        # The many duplicate boundary observations must not hide one buried patch.
        points = np.vstack((np.repeat([[0.5, 0, 0]], 4000, axis=0), [[0, 0, 0]]))
        committed = []
        chosen, _, _, _, stats = select_incremental_boxes(
            [box([0, 0, 0], [1, 1, 1])],
            [],
            points,
            0.05,
            CuboidConfig(max_cuboids=1, target_coverage=1),
            solid(),
            lambda _: None,
            checkpoint=lambda boxes, _: committed.append(boxes[-1]),
        )
        self.assertEqual(len(chosen), 1)
        self.assertEqual(len(committed), 1)
        self.assertTrue(np.all(box_surface_distance(points, committed[0]) <= 0.05))
        self.assertEqual(stats["irreparable_source_points"], 0)

    def test_frozen_irreparable_prefix_is_preserved_and_reported(self):
        prefix = box([0, 0, 0], [1, 1, 1])
        before = prefix["dimensions"].copy()
        chosen, _, _, _, stats = select_incremental_boxes(
            [box([0, 0, 0], [0.01, 0.01, 0.01])],
            [prefix],
            np.array([[0.0, 0, 0]]),
            0.05,
            CuboidConfig(max_cuboids=8),
            solid(),
            lambda _: None,
            checkpoint=lambda *_: self.fail("An irreparable frozen prefix must not be extended"),
        )
        self.assertEqual(len(chosen), 1)
        np.testing.assert_array_equal(prefix["dimensions"], before)
        self.assertEqual(stats["stopping_reason"], "irreparable_frozen_prefix")
        self.assertEqual(stats["irreparable_source_points"], 1)
        self.assertEqual(stats["irreparable_spatial_fraction"], 1)

    def test_union_burial_is_detected_when_neither_box_is_deep(self):
        # At the cross center each slab is only .08 deep, but the union boundary
        # is sqrt(2) * .08 away: testing individual boxes cannot certify this.
        points = np.array([[0.0, 0, 0], [0, 0.7, 0]])
        horizontal = box([0, 0, 0], [2, 0.16, 2])
        vertical = box([0, 0, 0], [0.16, 2, 2])
        config = CuboidConfig(target_coverage=1, max_cuboids=2)
        objective = UnionSurfaceObjective(points, np.ones(2), 0.1, config, np.arange(2), np.ones(2))
        objective.install([horizontal], objective.evaluate([horizontal]))
        self.assertTrue(objective.burial_feasible(vertical, np.arange(2)))
        combined = objective.evaluate([horizontal, vertical])
        self.assertFalse(objective.feasible(combined))
        self.assertGreater(combined["distances"][0], 0.1)
        saved = []
        with (
            patch("cuboid_approximation.cuboids.refine_box", side_effect=lambda b, *args: b),
            patch("cuboid_approximation.cuboids.grow_certified", side_effect=lambda b, *args: b),
        ):
            chosen, _, _, _, stats = select_incremental_boxes(
                [vertical],
                [horizontal],
                points,
                0.1,
                config,
                solid(),
                lambda _: None,
                checkpoint=lambda boxes, _: saved.append(boxes[-1]),
            )
        self.assertEqual(len(chosen), 2)
        self.assertTrue(saved)
        self.assertTrue(objective.feasible(objective.evaluate(chosen)))
        self.assertEqual(stats["rejected_irreparable_candidates"], 1)

    def test_reverse_surface_residuals_are_proposed_after_volume_is_complete(self):
        u, v = np.meshgrid(np.linspace(-0.4, 0.4, 12), np.linspace(-0.4, 0.4, 12))
        points = np.column_stack((u.ravel(), v.ravel(), np.zeros(u.size)))
        prefix = box([0, 0, 0], [2, 2, 0.02])
        residuals = []

        def propose(residual):
            residuals.append(residual)
            return []

        _, _, remaining, _, stats = select_incremental_boxes(
            [],
            [prefix],
            points,
            0.08,
            CuboidConfig(max_cuboids=4),
            solid(),
            lambda _: None,
            propose=propose,
        )
        self.assertFalse(remaining.any())
        self.assertTrue(residuals)
        self.assertGreater(len(residuals[0]), 0)
        self.assertEqual(stats["source_surface_coverage"], 1)
        self.assertLess(stats["surface_supported_area_fraction"], 0.95)
        self.assertEqual(stats["stopping_reason"], "no_positive_surface_gain")

    def test_volume_only_explicit_opt_out_retains_legacy_completion(self):
        prefix = box([0, 0, 0], [1, 1, 1])
        _, _, remaining, _, stats = select_incremental_boxes(
            [],
            [prefix],
            np.array([[0.0, 0, 0]]),
            0.05,
            CuboidConfig(target_surface_support=0),
            solid(),
            lambda _: None,
        )
        self.assertFalse(remaining.any())
        self.assertEqual(stats["stopping_reason"], "target_reached")
        self.assertNotIn("irreparable_source_points", stats)


if __name__ == "__main__":
    unittest.main()
