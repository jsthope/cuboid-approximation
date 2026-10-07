"""Orientation regressions for symmetric geometry and noisy planar evidence."""

import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from cuboid_approximation.cuboids import CuboidConfig, region_frames
from cuboid_approximation.fitting import fit_cuboids
from cuboid_approximation.orientations import equivalent_frame


def cube_shell(n=17):
    grid = np.linspace(-0.5, 0.5, n)
    u, v = np.meshgrid(grid, grid)
    faces = []
    for axis in range(3):
        for side in (-0.5, 0.5):
            face = np.empty((u.size, 3))
            face[:, axis] = side
            face[:, [other for other in range(3) if other != axis]] = np.c_[u.ravel(), v.ravel()]
            faces.append(face)
    return np.unique(np.vstack(faces), axis=0)


def proposals(points, budget=8):
    return region_frames(
        points,
        np.full(len(points), -1),
        np.zeros(len(points), int),
        0.05,
        CuboidConfig(max_frames=budget),
    )[0]


class OrientationTests(unittest.TestCase):
    def setUp(self):
        self.rotation = Rotation.from_euler("xyz", [13, 29, 51], degrees=True).as_matrix()

    def assertFrameNear(self, actual, expected, degrees):
        self.assertTrue(equivalent_frame(actual, expected, np.cos(np.deg2rad(degrees))))
        np.testing.assert_allclose(actual.T @ actual, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(actual), 1)

    def test_symmetric_cube_orientation_precedes_ambiguous_pca(self):
        for angles in ([13, 29, 51], [45, 45, 45], [0, 0, 31]):
            rotation = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
            frames = proposals(cube_shell() @ rotation.T, budget=2)
            self.assertEqual(len(frames), 2)
            self.assertFrameNear(frames[1], rotation, 0.001)

    def test_noisy_cube_keeps_dominant_axes(self):
        points = cube_shell(21) @ self.rotation.T
        points += np.random.default_rng(24).normal(scale=0.01, size=points.shape)
        self.assertFrameNear(proposals(points, budget=2)[1], self.rotation, 0.5)

    def test_large_density_skewed_cloud_uses_order_independent_spatial_support(self):
        shell = cube_shell(25)
        dense = np.repeat(shell[shell[:, 2] == 0.5], 40, axis=0)
        points = np.vstack((shell, dense)) @ self.rotation.T
        observed = []

        def record_hull(sample):
            observed.append(sample.copy())
            return ConvexHull(sample)

        with patch("cuboid_approximation.orientations.ConvexHull", side_effect=record_hull):
            first = proposals(points, budget=2)[1]
            reordered = proposals(np.random.default_rng(12).permutation(points), budget=2)[1]
        self.assertFrameNear(first, self.rotation, 0.001)
        self.assertFrameNear(reordered, first, 0.001)
        self.assertLessEqual(len(observed[0]), 8006)
        np.testing.assert_array_equal(observed[0], observed[1])

    def test_planar_square_uses_boundary_directions(self):
        grid = np.linspace(-0.5, 0.5, 17)
        u, v = np.meshgrid(grid, grid)
        points = np.c_[u.ravel(), v.ravel(), np.zeros(u.size)] @ self.rotation.T
        self.assertFrameNear(proposals(points, budget=2)[1], self.rotation, 0.001)

    def test_curved_cloud_retains_pca_and_no_duplicate_frames(self):
        points = np.random.default_rng(42).normal(size=(2000, 3))
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        frames = proposals((points * [3, 1, 0.6]) @ self.rotation.T)
        self.assertGreater(len(frames), 1)
        self.assertTrue(
            any(equivalent_frame(f, self.rotation, np.cos(np.deg2rad(4))) for f in frames)
        )
        for i, frame in enumerate(frames):
            self.assertFrameNear(frame, frame, 0.001)
            for old in frames[:i]:
                self.assertFalse(equivalent_frame(frame, old, np.cos(np.deg2rad(2))))

    def test_single_frame_budget_skips_orientation_work(self):
        with patch("numpy.linalg.eigh", side_effect=AssertionError("unnecessary PCA")):
            np.testing.assert_array_equal(proposals(cube_shell(), budget=1), np.eye(3)[None])

    def test_rotated_cube_fits_one_exact_cuboid_in_default_mode(self):
        points = cube_shell(21) @ self.rotation.T
        regions = dict(
            points=points,
            labels=np.full(len(points), -1),
            assignment_kind=np.zeros(len(points), int),
        )
        result = fit_cuboids(
            points,
            regions,
            0.05,
            CuboidConfig(resolution=48, max_frames=8, seeds_per_frame=8, max_cuboids=16),
            progress=lambda _: None,
        )
        self.assertEqual(len(result["boxes"]), 1)
        box = result["boxes"][0]
        self.assertFrameNear(box["rotation"], self.rotation, 0.001)
        np.testing.assert_allclose(box["center"], 0, atol=1e-10)
        np.testing.assert_allclose(box["dimensions"], 1, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
