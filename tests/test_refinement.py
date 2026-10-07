"""Refinement must escape infeasible envelopes with several misplaced faces."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from cuboid_approximation.approximation import spatial_weights
from cuboid_approximation.cuboids import CuboidConfig, certify_box, distance_to_box, refine_box
from cuboid_approximation.selection import UnionSurfaceObjective


def observed_box_with_neighbor():
    """Analytic cuboid shell plus a small adjacent patch, not isolated outliers."""
    grid = np.linspace(0, 1, 11)
    u, v = np.meshgrid(grid, grid)
    faces = []
    for axis in range(3):
        tangent = [a for a in range(3) if a != axis]
        for side in (0, 1):
            face = np.zeros((u.size, 3))
            face[:, axis] = side
            face[:, tangent] = np.column_stack((u.ravel(), v.ravel()))
            faces.append(face * [2, 1, 1])
    shell = np.unique(np.concatenate(faces), axis=0)
    y, z = np.meshgrid(np.linspace(1.11, 1.13, 4), np.linspace(0.2, 0.8, 6))
    neighbor = np.column_stack((np.zeros(y.size), y.ravel(), z.ravel()))
    return np.vstack((shell, neighbor))


class RefinementTests(unittest.TestCase):
    def test_expanded_l_prism_seed_recovers_its_rectangular_arm(self):
        # Ten analytic exterior rectangles of an L prism; its missing quadrant
        # contributes nearby observations that contaminate an expanded arm.
        rectangles = [
            ([0, 0, 0], [0, 2, 1]), ([1, 1, 0], [1, 2, 1]),
            ([2, 0, 0], [2, 1, 1]), ([0, 0, 0], [2, 0, 1]),
            ([1, 1, 0], [2, 1, 1]), ([0, 2, 0], [1, 2, 1]),
            ([0, 0, 0], [2, 1, 0]), ([0, 1, 0], [1, 2, 0]),
            ([0, 0, 1], [2, 1, 1]), ([0, 1, 1], [1, 2, 1]),
        ]
        faces = []
        for low, high in rectangles:
            axes = [np.linspace(a, b, int(np.ceil((b - a) / 0.06)) + 1)
                    for a, b in zip(low, high)]
            faces.append(np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3))
        points = np.unique(np.concatenate(faces), axis=0)
        seed = dict(center=np.array([49, 25, 25]) / 48,
                    dimensions=np.array([53, 27, 27]) / 24,
                    rotation=np.eye(3), volume=float(np.prod(np.array([53, 27, 27]) / 24)))
        solid = dict(safe=np.ones((52, 52, 30), dtype=bool),
                     origin=np.full(3, -0.2), voxel_size=0.05)
        tolerance = 0.06
        weights = spatial_weights(points, tolerance)
        objective = UnionSurfaceObjective(
            points, weights, tolerance, CuboidConfig(target_coverage=1),
            np.arange(len(points)), weights,
        )

        def score(box):
            ids = np.flatnonzero(distance_to_box(points, box) <= tolerance)
            return objective.score(box, ids, weights[ids].sum(), 1)

        support = points[distance_to_box(points, seed) <= tolerance]
        exact_arm = dict(center=np.array([1, 0.5, 0.5]), dimensions=np.array([2, 1, 1]),
                         rotation=np.eye(3), volume=2.0)
        self.assertTrue(certify_box(exact_arm, solid, strict=True))
        self.assertGreater(score(exact_arm), 1)
        self.assertEqual(score(seed), 0)
        result = refine_box(seed, support, solid, score)
        self.assertTrue(certify_box(result, solid, strict=True))
        self.assertGreaterEqual(score(result), score(exact_arm) - 1e-12)
        self.assertFalse(objective.evaluate([result])["irreparable"].any())

    def test_joint_face_trim_recovers_box_despite_small_adjacent_patch(self):
        frame = Rotation.from_euler("xyz", [17, 31, 53], degrees=True).as_matrix()
        for rotation, noise in ((np.eye(3), 0), (frame, 0), (frame, 0.0002)):
            with self.subTest(rotated=not np.array_equal(rotation, np.eye(3)), noise=noise):
                source = observed_box_with_neighbor()
                if noise:
                    source += np.random.default_rng(42).normal(0, noise, source.shape)
                points = source @ rotation.T + [3, 3, 3]
                seed = dict(center=np.array([1, 0.55, 0.5]) @ rotation.T + [3, 3, 3],
                            dimensions=np.array([2.16, 1.3, 1.16]), rotation=rotation,
                            volume=2.16 * 1.3 * 1.16)
                solid = dict(safe=np.ones((64, 64, 64), dtype=bool),
                             origin=np.zeros(3), voxel_size=0.1)
                tolerance = 0.06
                weights = spatial_weights(points, tolerance)
                objective = UnionSurfaceObjective(
                    points, weights, tolerance, CuboidConfig(),
                    np.arange(len(points)), weights,
                )

                def score(box):
                    ids = np.flatnonzero(distance_to_box(points, box) <= tolerance)
                    return objective.score(box, ids, weights[ids].sum(), 1)

                self.assertTrue(certify_box(seed, solid, strict=True))
                self.assertEqual(score(seed), 0)
                result = refine_box(seed, points, solid, score)
                self.assertTrue(certify_box(result, solid, strict=True))
                self.assertGreater(score(result), 1)
                np.testing.assert_allclose(result["dimensions"], [2, 1, 1], atol=0.002)
                self.assertFalse(objective.evaluate([result])["irreparable"].any())


if __name__ == "__main__":
    unittest.main()
