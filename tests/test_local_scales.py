"""Density changes must not erase surfaces or leak normals across nearby sheets."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cuboid_approximation.cloud import load_points, local_geometry, local_point_spacing
from cuboid_approximation.regions import SurfaceConfig, segment_surfaces
from cuboid_approximation.texture import load_cloud, transfer_normals
from test_method import write_cloud
from test_regressions import sheet


def varying_density():
    return np.vstack((sheet(20) * 0.001, sheet(8) + [5, 0, 0]))


class LocalScaleTests(unittest.TestCase):
    def test_dense_component_does_not_invalidate_sparse_plane(self):
        points = varying_density()
        geometry = local_geometry(points)
        self.assertTrue(geometry["valid"].all())
        np.testing.assert_allclose(np.abs(geometry["normals"]), np.tile([0, 0, 1], (len(points), 1)))
        self.assertGreater(geometry["spacing_per_point"][400:].min(),
                           1000 * geometry["spacing_per_point"][:400].max())

    def test_duplicates_do_not_change_scales_or_normal_support(self):
        points = varying_density()
        original = local_geometry(points)
        duplicated = local_geometry(np.repeat(points, 4, axis=0))
        for key in ("valid", "neighbor_counts", "spacing_per_point", "normals"):
            np.testing.assert_allclose(duplicated[key], np.repeat(original[key], 4, axis=0))

    def test_malformed_local_scale_arrays_are_rejected(self):
        points = sheet(10)
        normals = np.tile([0, 0, 1], (len(points), 1))
        for invalid in (np.ones(3), np.zeros(len(points)), np.full(len(points), np.nan)):
            with self.assertRaisesRegex(ValueError, "Local spacing"):
                transfer_normals(points, points, normals, np.ones(len(points), bool),
                                 1 / 9, 1 / 9, spacing_per_point=invalid)

    def test_normal_transfer_and_fallback_both_preserve_sparse_plane(self):
        points = varying_density()
        geometry = local_geometry(points)
        for valid in (geometry["valid"], np.zeros(len(points), dtype=bool)):
            normals, confidence = transfer_normals(
                points, points, geometry["normals"], valid,
                geometry["spacing"], geometry["spacing"],
            )
            self.assertTrue(np.all(confidence > 0.5))
            np.testing.assert_allclose(np.abs(normals), np.tile([0, 0, 1], (len(points), 1)))

    def test_close_sheets_do_not_exchange_normals_or_region_labels(self):
        bottom = sheet(10)
        points = np.vstack((bottom, bottom + [0, 0, 0.04]))
        geometry = local_geometry(points)
        expected = np.tile([0, 0, 1], (len(points), 1))
        normals, confidence = transfer_normals(
            points, bottom, expected[:len(bottom)], np.ones(len(bottom), bool),
            geometry["spacing"], 1 / 9,
        )
        # The upper sheet must use its own local fit, not the lower donor plane.
        np.testing.assert_allclose(confidence[len(bottom):], 0.75)
        np.testing.assert_allclose(np.abs(normals[len(bottom):, 2]), 1, atol=0.01)
        result = segment_surfaces(
            points, expected, np.ones(len(points), bool), np.zeros(len(points), bool),
            np.empty((0, 2, 3)), geometry["spacing"],
            config=SurfaceConfig(min_region_points=20),
            spacing_per_point=geometry["spacing_per_point"],
        )
        self.assertEqual(len(np.unique(result["labels"])), 2)
        self.assertTrue(np.all(result["labels"][:len(bottom)] == result["labels"][0]))
        self.assertTrue(np.all(result["labels"][len(bottom):] == result["labels"][-1]))
        self.assertNotEqual(result["labels"][0], result["labels"][-1])

    def test_cloud_footprints_and_gaussian_limits_use_local_spacing(self):
        points = varying_density()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "cloud.ply"
            write_cloud(source, points)
            loaded = load_points(source)
            geometry = local_geometry(points)
            (folder / "report.json").write_text(json.dumps(dict(
                min_opacity=0.1, median_spacing=geometry["spacing"],
                full_cloud_median_spacing=loaded["spacing"],
            )))
            # Old preparation archives without local scales remain readable.
            np.savez(folder / "common_geometry.npz", points=points,
                     normals=geometry["normals"], normal_valid=geometry["valid"])
            cloud = load_cloud(source, folder, loaded)
            expected = local_point_spacing(points)
            np.testing.assert_allclose(cloud["spacing_per_point"], expected)
            np.testing.assert_allclose(cloud["covariance"][:, 0, 0], (0.6 * expected) ** 2)
            loaded["splats"] = dict(
                log_scales=np.full((len(points), 3), 100.0),
                quaternions_wxyz=np.tile([1, 0, 0, 0], (len(points), 1)),
            )
            gaussian = load_cloud(source, folder, loaded)
            np.testing.assert_allclose(gaussian["covariance"][:, 0, 0], (2 * expected) ** 2)


if __name__ == "__main__":
    unittest.main()
