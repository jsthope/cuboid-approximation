"""Observed panels must survive sparse sampling and must not fill sheet gaps."""

import unittest

import numpy as np

from cuboid_approximation.cuboids import (
    CuboidConfig,
    box_membership,
    distance_to_box,
    select_incremental_boxes,
)
from cuboid_approximation.fitting import fit_cuboids
from cuboid_approximation.surface_proposals import SurfacePanelProposer
from cuboid_approximation.selection import UnionSurfaceObjective, exterior_volume_cost
from test_regressions import regions, sheet
from test_surface_selection import box, solid


class SurfaceProposalTests(unittest.TestCase):
    def fit(self, points, tolerance):
        return fit_cuboids(
            points,
            regions(points),
            0.02,
            CuboidConfig(
                resolution=32,
                max_frames=2,
                seeds_per_frame=4,
                max_cuboids=2,
                reconstruction_mode="surface",
                point_tolerance=tolerance,
            ),
            progress=lambda _: None,
        )

    def test_mixed_and_extreme_density_planes_need_two_panels(self):
        for dense in (sheet(24), sheet(20) * 0.001):
            with self.subTest(dense_extent=float(np.ptp(dense, axis=0).max())):
                points = np.vstack((dense, sheet(8) + [5, 0, 0]))
                result = self.fit(points, 0.11)
                self.assertEqual(len(result["boxes"]), 2)
                self.assertEqual(result["selection"]["stopping_reason"], "target_reached")
                self.assertGreaterEqual(result["selection"]["source_surface_coverage"], 0.999)
                self.assertGreaterEqual(
                    result["selection"]["surface_supported_area_fraction"], 0.95
                )
                small = min(result["boxes"], key=lambda b: np.linalg.norm(b["center"]))
                # Sparse points in the other plane must not inflate this patch
                # merely to dilute the union's mean reverse-distance error.
                if np.ptp(dense, axis=0).max() < 0.01:
                    self.assertLess(np.max(small["dimensions"]), 0.002)

    def test_close_layers_preserve_the_unobserved_gap(self):
        points = np.vstack((sheet(41), sheet(41) + [0, 0, 0.08]))
        result = self.fit(points, 0.025)
        self.assertEqual(len(result["boxes"]), 2)
        self.assertEqual(result["selection"]["stopping_reason"], "target_reached")
        middle = sheet(7) * [0.6, 0.6, 1] + [0.2, 0.2, 0.04]
        for candidate in result["boxes"]:
            self.assertLessEqual(np.min(candidate["dimensions"]), 0.05)
            self.assertFalse(box_membership(middle, candidate).any())

    def test_concave_coplanar_patch_is_split_without_filling_its_missing_corner(self):
        u, v = np.meshgrid(np.arange(0.05, 2.0, 0.05), np.arange(0.05, 2.0, 0.05))
        points = np.column_stack((u.ravel(), v.ravel(), np.full(u.size, 0.05)))
        points = points[(points[:, 0] < 1) | (points[:, 1] < 1)]
        # Exercise the production proposal pipeline: planar panels handle area,
        # while its existing local seeds also handle residual lines and points.
        result = fit_cuboids(
            points,
            regions(points),
            0.05,
            CuboidConfig(
                resolution=32,
                max_frames=1,
                seeds_per_frame=4,
                max_cuboids=8,
                reconstruction_mode="surface",
                point_tolerance=0.06,
            ),
            progress=lambda _: None,
        )
        self.assertEqual(result["selection"]["stopping_reason"], "target_reached")
        self.assertTrue(np.all(result["full_distances"] <= 0.06))
        self.assertGreaterEqual(result["selection"]["source_surface_coverage"], 0.999)
        missing = np.array([[1.5, 1.5, 0.05]])
        self.assertFalse(
            any(box_membership(missing, candidate).any() for candidate in result["boxes"])
        )

    def test_hollow_cube_panels_cover_faces_without_covering_the_center(self):
        axis = np.linspace(0.1, 1.1, 21)
        u, v = np.meshgrid(axis, axis)
        faces = []
        for normal in range(3):
            tangent = [a for a in range(3) if a != normal]
            for side in (0.1, 1.1):
                face = np.zeros((u.size, 3))
                face[:, normal] = side
                face[:, tangent] = np.column_stack((u.ravel(), v.ravel()))
                faces.append(face)
        points = np.unique(np.concatenate(faces), axis=0)
        safe = np.ones((24, 24, 24), bool)
        safe[3:21, 3:21, 3:21] = False
        envelope = dict(safe=safe, reference=safe, origin=np.zeros(3), voxel_size=0.05)
        panels = SurfacePanelProposer(
            points, np.eye(3)[None], 0.06, envelope, CuboidConfig(seeds_per_frame=4)
        )(points)
        self.assertTrue(panels)
        self.assertFalse(
            any(
                box_membership(np.array([[0.6, 0.6, 0.6]]), candidate).any() for candidate in panels
            )
        )
        distance = np.min([distance_to_box(points, candidate) for candidate in panels], axis=0)
        self.assertTrue(np.all(distance <= 0.06))

    def test_moving_a_false_floor_toward_distant_points_is_not_source_gain(self):
        points = np.vstack((sheet(21), sheet(21) + [0, 0, 0.08]))
        weights = np.ones(len(points))
        objective = UnionSurfaceObjective(
            points,
            weights,
            0.025,
            CuboidConfig(reconstruction_mode="surface"),
            np.arange(len(points)),
            weights,
        )
        exact = box([0.5, 0.5, 0], [1, 1, 0.00001])
        false_floor = box([0.5, 0.5, 0.02], [1, 1, 0.04])
        scores = []
        for candidate in (exact, false_floor):
            ids = np.flatnonzero(distance_to_box(points, candidate) <= 0.025)
            scores.append(objective.score(candidate, ids, len(ids), 1.0))
        self.assertGreater(scores[0], scores[1])

    def test_reverse_error_cannot_be_diluted_by_adding_area(self):
        points = np.array([[0.0, 0, 0], [1, 1, 0]])
        config = CuboidConfig()
        objective = UnionSurfaceObjective(points, np.ones(2), 0.1, config, np.arange(2), np.ones(2))
        error = objective._reverse_error(np.array([0.1]), np.ones(1))
        self.assertEqual(
            error, objective._reverse_error(np.array([0.1, 0]), np.array([1.0, 100.0]))
        )
        self.assertGreater(
            objective._reverse_error(np.array([0.1, 0.01]), np.array([1.0, 100.0])), error
        )
        repeated = np.repeat(points, 30, axis=0)
        duplicated = UnionSurfaceObjective(
            repeated,
            np.ones(len(repeated)),
            0.1,
            config,
            np.arange(len(repeated)),
            np.ones(len(repeated)),
        )
        self.assertEqual(objective.reverse_area_scale, duplicated.reverse_area_scale)
        scaled = UnionSurfaceObjective(
            points * 7, np.ones(2), 0.7, config, np.arange(2), np.ones(2)
        )
        self.assertAlmostEqual(scaled.reverse_area_scale, 49 * objective.reverse_area_scale)

    def test_thin_plane_cost_is_stable_across_reference_voxel_boundary(self):
        safe = np.ones((4, 4, 4), bool)
        reference = safe.copy()
        reference[:, :, :2] = False
        envelope = dict(safe=safe, reference=reference, origin=np.full(3, -1.0), voxel_size=0.5)
        volume = float(reference.sum()) * 0.5**3
        values = [
            exterior_volume_cost(box([0, 0, z], [1, 1, 1e-5]), envelope, 2.0, volume)
            for z in (-1e-6, 0, 1e-6)
        ]
        self.assertLess(max(values) - min(values), 1e-5)
        self.assertLess(max(values), 1.00001)
        self.assertGreater(
            exterior_volume_cost(box([0, 0, -0.5], [1, 1, 0.5]), envelope, 2.0, volume), 1.2
        )

    def test_surface_mode_rejects_thick_frozen_prefix_without_mutating_it(self):
        prefix = box([0, 0, 0], [1, 1, 1])
        points = np.array([[0, 0, 0.5], [0, 0, -0.5]], float)
        chosen, _, _, _, report = select_incremental_boxes(
            [],
            [prefix],
            points,
            0.025,
            CuboidConfig(reconstruction_mode="surface"),
            solid(),
            lambda _: None,
        )
        self.assertEqual(len(chosen), 1)
        np.testing.assert_array_equal(chosen[0]["dimensions"], prefix["dimensions"])
        self.assertEqual(report["stopping_reason"], "irreparable_frozen_prefix")
        self.assertTrue(report["frozen_prefix_violates_surface_thickness"])
        # The same dimensions remain admissible to the solid-mode scorer.
        objective = UnionSurfaceObjective(
            points, np.ones(2), 0.025, CuboidConfig(), np.arange(2), np.ones(2)
        )
        self.assertTrue(objective.burial_feasible(prefix, np.arange(2)))


if __name__ == "__main__":
    unittest.main()
