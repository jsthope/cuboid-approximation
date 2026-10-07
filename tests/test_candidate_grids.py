"""Bounded local search must retain the global containment certificate."""

import itertools
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from cuboid_approximation.candidate_grids import generate_candidates
from cuboid_approximation.cuboids import (
    CuboidConfig,
    box_membership,
    certify_box,
    summed_volume,
)
from cuboid_approximation.volume import check_grid_memory


class CandidateGridTests(unittest.TestCase):
    def test_sparse_rotated_scene_fits_budget_that_rejects_the_global_grid(self):
        safe = np.zeros((128, 128, 128), bool)
        for corner in itertools.product((4, 110), repeat=3):
            safe[tuple(slice(p, p + 14) for p in corner)] = True
        solid = dict(safe=safe, origin=np.zeros(3), voxel_size=1.0, max_memory_mb=64)
        frame = Rotation.from_euler("xyz", [23, 37, 19], degrees=True).as_matrix()
        centers = np.array(list(itertools.product((11.0, 117.0), repeat=3)))
        # Analytic size of the former global rotated workspace, without keeping
        # its unused production implementation or allocating all source centers.
        corners = np.array(list(itertools.product((4.5, 123.5), repeat=3))) @ frame
        global_shape = np.floor(corners.max(0)) - np.floor(corners.min(0)) + 3
        global_fixed = safe.size * 12 + int(safe.sum()) * 24
        with self.assertRaisesRegex(ValueError, "global rotated grid"):
            check_grid_memory(global_shape, 64, "global rotated grid", 12, global_fixed)
        observed_sizes = []
        original_argwhere = np.argwhere

        def bounded_argwhere(array):
            observed_sizes.append(array.size)
            return original_argwhere(array)

        with patch("numpy.argwhere", side_effect=bounded_argwhere):
            boxes = generate_candidates(
                solid, np.array([frame]), CuboidConfig(max_memory_mb=64, seeds_per_frame=8),
                np.random.default_rng(3), centers, lambda _: None,
            )
        self.assertTrue(boxes)
        self.assertTrue(all(certify_box(box, solid, strict=True) for box in boxes))
        self.assertTrue(np.any([box_membership(centers, box) for box in boxes], axis=0).all())
        metadata = solid["candidate_search"]
        self.assertEqual(metadata["frames"][0]["searched_tiles"], 8)
        self.assertEqual(metadata["frames"][0]["seed_budget"], 8)
        self.assertLess(metadata["maximum_grid_cells"], 25000)
        self.assertLess(max(observed_sizes), 25000)
        self.assertEqual(solid["voxel_size"], 1.0)

    def test_small_disconnected_detail_receives_a_seed(self):
        safe = np.zeros((120, 60, 60), bool)
        safe[4:34, 4:34, 4:34] = True
        safe[106:110, 48:52, 48:52] = True
        solid = dict(safe=safe, origin=np.zeros(3), voxel_size=1.0)
        points = np.array([[19.0, 19, 19], [108.0, 50, 50]])
        boxes = generate_candidates(
            solid, np.eye(3)[None], CuboidConfig(max_memory_mb=64, seeds_per_frame=2),
            np.random.default_rng(9), points, lambda _: None,
        )
        self.assertTrue(np.any([box_membership(points, box) for box in boxes], axis=0).all())
        self.assertTrue(all(certify_box(box, solid, strict=True) for box in boxes))
        self.assertEqual(solid["candidate_search"]["frames"][0]["seed_budget"], 2)

    def test_oversized_dense_rotated_workspace_splits_without_reducing_resolution(self):
        safe = np.zeros((90, 90, 90), bool)
        safe[2:88, 2:88, 2:88] = True
        solid = dict(safe=safe, origin=np.zeros(3), voxel_size=1.0)
        frame = Rotation.from_euler("xyz", [23, 37, 19], degrees=True).as_matrix()
        boxes = generate_candidates(
            solid, np.array([frame]), CuboidConfig(max_memory_mb=64, seeds_per_frame=8),
            np.random.default_rng(2), np.array([[45.0, 45, 45]]), lambda _: None,
        )
        self.assertTrue(boxes)
        self.assertTrue(all(certify_box(box, solid, strict=True) for box in boxes))
        self.assertGreater(solid["candidate_search"]["frames"][0]["tiles"], 1)
        self.assertLess(solid["candidate_search"]["maximum_grid_cells"], 400000)
        self.assertEqual(solid["voxel_size"], 1.0)

    def test_compact_scene_preserves_whole_object_candidates(self):
        safe = np.zeros((36, 36, 36), bool)
        safe[3:33, 3:33, 3:33] = True
        solid = dict(safe=safe, origin=np.zeros(3), voxel_size=1.0)
        boxes = generate_candidates(
            solid, np.eye(3)[None], CuboidConfig(max_memory_mb=64, seeds_per_frame=4),
            np.random.default_rng(3), np.array([[18.0, 18, 18]]), lambda _: None,
        )
        self.assertEqual(solid["candidate_search"]["frames"][0]["tiles"], 1)
        self.assertTrue(any(np.allclose(box["dimensions"], 30) for box in boxes))

    def test_support_integral_is_guarded_before_allocation(self):
        solid = dict(safe=np.ones((220, 220, 220), bool), origin=np.zeros(3), voxel_size=1.0)
        with patch(
            "cuboid_approximation.candidate_grids.summed_volume",
            side_effect=AssertionError("integral allocated before memory guard"),
        ):
            with self.assertRaisesRegex(ValueError, "support integral"):
                generate_candidates(
                    solid, np.eye(3)[None], CuboidConfig(max_memory_mb=64),
                    np.random.default_rng(3), np.empty((0, 3)), lambda _: None,
                )

    def test_resume_integral_is_reused(self):
        safe = np.zeros((16, 16, 16), bool)
        safe[3:13, 3:13, 3:13] = True
        prefix = summed_volume(safe)
        solid = dict(safe=safe, safe_prefix=prefix, origin=np.zeros(3), voxel_size=1.0)
        with patch(
            "cuboid_approximation.candidate_grids.summed_volume",
            side_effect=AssertionError("unnecessary second global integral"),
        ):
            generate_candidates(
                solid, np.eye(3)[None], CuboidConfig(max_memory_mb=64, seeds_per_frame=1),
                np.random.default_rng(3), np.array([[8.0, 8, 8]]), lambda _: None,
            )
        self.assertIs(solid["safe_prefix"], prefix)


if __name__ == "__main__":
    unittest.main()
