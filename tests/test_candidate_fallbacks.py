"""Local proposal refinement must not discard safe progress alternatives."""

import unittest
from unittest.mock import patch

import numpy as np

from cuboid_approximation.cuboids import CuboidConfig, select_incremental_boxes
from test_surface_selection import box, solid


class CandidateFallbackTests(unittest.TestCase):
    def test_rejected_dominant_finalists_cannot_hide_raw_uncovered_cells(self):
        points = np.array([[0.0, 0, 0], [0, 0.7, 0], [0, -0.7, 0], [0, 0.3, 0]])
        prefix = box([0, 0, 0], [2, 0.16, 2])
        # Every high-scoring slab covers all residuals, but their union with
        # the prefix buries the origin. Safe, lower-scoring cells must still
        # get an exact check after this entire optimistic beam is rejected.
        candidates = [box([i * 1e-6, 0, 0], [0.16, 2, 2]) for i in range(40)]
        saved = []
        with (
            patch("cuboid_approximation.cuboids.refine_box", side_effect=lambda b, *args: b),
            patch("cuboid_approximation.cuboids.grow_certified", side_effect=lambda b, *args: b),
        ):
            selected, curve, _, _, stats = select_incremental_boxes(
                candidates,
                [prefix],
                points,
                0.1,
                CuboidConfig(max_cuboids=2, target_coverage=1),
                solid(),
                lambda _: None,
                checkpoint=lambda boxes, _: saved.append(boxes[-1]),
            )
        self.assertEqual(len(selected), 2)
        self.assertEqual(stats["selected_alternatives"]["raw_seed"], 1)
        self.assertGreaterEqual(stats["rejected_irreparable_candidates"], 3)
        self.assertGreater(curve[-1]["covered_fraction"], curve[0]["covered_fraction"])
        self.assertEqual(stats["irreparable_source_points"], 0)
        self.assertLessEqual(stats["raw_uncovered_seed_checks"], 8)
        self.assertEqual(len(saved), 1)
        for name in ("center", "dimensions", "rotation"):
            np.testing.assert_array_equal(selected[0][name], prefix[name])

    def test_many_invalid_panels_cannot_starve_a_valid_local_seed(self):
        panels = [
            dict(box([i * 1e-5, 0, 0], [1, 1, 1]), proposal_kind="surface_panel") for i in range(40)
        ]
        seed = dict(box([0, 0, 0], [0.01, 0.01, 0.01]), proposal_kind="local_cell")
        with (
            patch("cuboid_approximation.cuboids.refine_box", side_effect=lambda b, *args: b),
            patch("cuboid_approximation.cuboids.grow_certified", side_effect=lambda b, *args: b),
        ):
            selected, _, remaining, _, stats = select_incremental_boxes(
                [],
                [],
                np.zeros((1, 3)),
                0.05,
                CuboidConfig(max_cuboids=1),
                solid(),
                lambda _: None,
                propose=lambda residual: [*panels, seed],
            )
        self.assertEqual(len(selected), 1)
        np.testing.assert_array_equal(selected[0]["dimensions"], seed["dimensions"])
        self.assertFalse(remaining.any())
        self.assertEqual(stats["stopping_reason"], "target_reached")
        self.assertLessEqual(stats["candidates_prescreened"], 64)

    def test_rejected_refinement_and_growth_keep_original_safe_seed(self):
        points = np.array([[0.0, 0, 0], [0, 0.7, 0], [0, -0.7, 0], [0, 0.3, 0]])
        prefix = box([0, 0, 0], [2, 0.16, 2])
        seed = box([0, 0.65, 0], [0.16, 0.3, 2])
        # Each crossing slab is individually only .08 deep, but together they
        # bury the origin at sqrt(2)*.08 > tolerance. The original seed is disjoint.
        harmful = box([0, 0, 0], [0.16, 2, 2])
        saved = []
        with (
            patch("cuboid_approximation.cuboids.refine_box", return_value=harmful),
            patch("cuboid_approximation.cuboids.grow_certified", return_value=harmful),
        ):
            selected, _, _, _, stats = select_incremental_boxes(
                [seed],
                [prefix],
                points,
                0.1,
                CuboidConfig(max_cuboids=2, target_coverage=1),
                solid(),
                lambda _: None,
                checkpoint=lambda boxes, _: saved.append(boxes[-1]),
            )
        self.assertEqual(len(selected), 2)
        for name in ("center", "dimensions", "rotation"):
            np.testing.assert_array_equal(selected[-1][name], seed[name])
            np.testing.assert_array_equal(saved[-1][name], seed[name])
        self.assertGreaterEqual(stats["rejected_irreparable_candidates"], 1)
        self.assertEqual(stats["selected_alternatives"]["seed"], 1)


if __name__ == "__main__":
    unittest.main()
