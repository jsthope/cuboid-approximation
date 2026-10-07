"""Search-budget and common-objective regressions for finalist selection."""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from cuboid_approximation.cuboids import CuboidConfig, select_incremental_boxes
from cuboid_approximation.selection import UnionSurfaceObjective
from test_surface_selection import box, solid


class FinalistScoringTests(unittest.TestCase):
    def test_refined_runner_up_is_compared_against_coarse_winner(self):
        u, v = np.meshgrid(np.linspace(-0.5, 0.5, 21), np.linspace(-0.5, 0.5, 21))
        points = np.column_stack((u.ravel(), v.ravel(), np.zeros(u.size)))
        coarse_winner = dict(box([0, 0, 0], [1.2, 1.2, 0.01]), seed="broad")
        runner_up = dict(box([0.2, 0, 0], [0.6, 1, 0.01]), seed="partial")
        visited = []
        coarse_scores = {}

        def refine(candidate, _points, _solid, score):
            visited.append(candidate["seed"])
            coarse_scores[candidate["seed"]] = score(candidate)
            if candidate["seed"] == "partial":
                return dict(box([0, 0, 0], [1, 1, 0.01]), seed="partial")
            return candidate

        with (
            patch("cuboid_approximation.cuboids.refine_box", side_effect=refine),
            patch("cuboid_approximation.cuboids.grow_certified", side_effect=lambda b, *args: b),
        ):
            chosen, _, _, _, stats = select_incremental_boxes(
                [coarse_winner, runner_up],
                [],
                points,
                0.05,
                CuboidConfig(max_cuboids=1),
                solid(),
                lambda _: None,
            )
        self.assertCountEqual(visited, ["broad", "partial"])
        self.assertGreater(coarse_scores["broad"], coarse_scores["partial"])
        self.assertEqual(chosen[0]["seed"], "partial")
        self.assertEqual(stats["candidates_refined"], 2)
        self.assertEqual(stats["exact_union_checks"], 1)

    def test_hiding_old_unsupported_faces_changes_union_score(self):
        points = np.array([[0.0, 0, 0], [1, 0, 0]])
        config = CuboidConfig(target_coverage=0.5)
        objective = UnionSurfaceObjective(points, np.ones(2), 1.0, config, np.arange(2), np.ones(2))
        objective.signed = np.zeros(2)
        objective.boundary = np.array([[0.0, 0, 0], [4, 0, 0]])
        objective.area = np.ones(2)
        objective.boundary_source_distance = np.array([3.0, 0.0])
        objective.reverse_error = objective._reverse_error(
            objective.boundary_source_distance, objective.area
        )
        candidate = box([0, 0, 0], [1, 1, 1])
        with patch(
            "cuboid_approximation.selection.adaptive_face_samples",
            return_value=(np.array([[1.0, 0, 0]]), np.ones(1)),
        ):
            hides_unsupported = objective.score(candidate, np.arange(2), 2.0, 1.0)
            # Keep the old error/area distribution but move its bad witness
            # outside the candidate. Hiding it must improve the union objective.
            objective.boundary[0] = [2, 0, 0]
            preserves_unsupported = objective.score(candidate, np.arange(2), 2.0, 1.0)
        self.assertGreater(hides_unsupported, preserves_unsupported)

    def test_thousands_of_hopeless_candidates_get_bounded_refinement_and_residual_seed(self):
        points = np.array([[0.0, 0, 0]])
        candidates = [box([i * 1e-5, 0, 0], [1, 1, 1]) for i in range(2000)]
        calls = []

        def refine(candidate, *args):
            calls.append(candidate)
            return candidate

        with (
            patch("cuboid_approximation.cuboids.refine_box", side_effect=refine),
            patch("cuboid_approximation.cuboids.grow_certified", side_effect=lambda b, *args: b),
        ):
            chosen, _, _, _, stats = select_incremental_boxes(
                candidates,
                [],
                points,
                0.05,
                CuboidConfig(max_cuboids=1),
                solid(),
                lambda _: None,
                propose=lambda residual: [box([0, 0, 0], [0.01, 0.01, 0.01])],
            )
        self.assertEqual(len(chosen), 1)
        self.assertLessEqual(len(calls), 4)
        self.assertLessEqual(stats["candidates_prescreened"], 64)
        self.assertEqual(stats["stopping_reason"], "target_reached")

    def test_sampled_support_cannot_claim_success_without_lower_bound(self):
        objective = UnionSurfaceObjective(
            np.zeros((1, 3)), np.ones(1), 0.1, CuboidConfig(), np.array([0]), np.ones(1)
        )
        objective.source_coverage = objective.reverse_coverage = 1.0
        objective.mesh = Mock()
        objective.mesh.support_bounds.return_value = dict(
            lower_fraction=0.94, upper_fraction=0.99, status="inconclusive"
        )
        self.assertFalse(objective.complete())
        self.assertFalse(objective.complete())
        objective.mesh.support_bounds.assert_called_once()
        objective.support_bounds = dict(lower_fraction=0.95 - 5e-13, upper_fraction=1.0)
        self.assertFalse(objective.complete())

    def test_bounded_exhaustion_tries_an_uncovered_cell_before_stopping(self):
        candidates = [box([i * 1e-5, 0, 0], [1, 1, 1]) for i in range(100)]
        with (
            patch("cuboid_approximation.cuboids.refine_box", side_effect=lambda b, *args: b),
            patch("cuboid_approximation.cuboids.grow_certified", side_effect=lambda b, *args: b),
        ):
            chosen, _, _, _, stats = select_incremental_boxes(
                candidates,
                [],
                np.zeros((1, 3)),
                0.05,
                CuboidConfig(max_cuboids=1),
                solid(),
                lambda _: None,
            )
        self.assertEqual(len(chosen), 1)
        self.assertEqual(stats["selected_alternatives"]["raw_seed"], 1)
        self.assertEqual(stats["stopping_reason"], "target_reached")
        self.assertFalse(stats["finite_pool_minimum_proven"])


if __name__ == "__main__":
    unittest.main()
