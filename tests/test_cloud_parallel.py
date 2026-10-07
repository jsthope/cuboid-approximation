"""Input reuse and concurrent normal estimation preserve source geometry."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from cuboid_approximation.cloud import load_points, local_geometry
from test_method import write_cloud


class CloudExecutionTests(unittest.TestCase):
    def test_cached_spacing_preserves_duplicates_invalid_rows_and_source_order(self):
        rng = np.random.default_rng(81)
        source = rng.uniform(size=(100, 3))
        source = np.concatenate((source[23:], source[:23], source[::3], [[np.nan, 0, 0]]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cloud.ply"
            write_cloud(path, source)
            reused = load_points(path)
            fresh = load_points(path, outlier_distance_factor=0)
        self.assertEqual(len(reused["outlier_source_indices"]), 0)
        for key in ("points", "source_indices", "distinct_indices", "spacing_per_point"):
            np.testing.assert_array_equal(reused[key], fresh[key])
        self.assertEqual(reused["spacing"], fresh["spacing"])

    def test_spacing_after_rejection_matches_retained_population(self):
        source = np.random.default_rng(7).uniform(size=(200, 3))
        source = np.concatenate((source, [[10000, 0, 0], [np.nan, 0, 0]]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cloud.ply"
            write_cloud(path, source)
            loaded = load_points(path)
        np.testing.assert_array_equal(loaded["outlier_source_indices"], [200])
        delta = loaded["points"][:, None] - loaded["points"][None, :]
        distances = np.linalg.norm(delta, axis=2)
        np.fill_diagonal(distances, np.inf)
        np.testing.assert_allclose(loaded["spacing_per_point"], distances.min(axis=1), atol=0)

    def test_parallel_normals_match_serial_for_multiple_chunks(self):
        rng = np.random.default_rng(12)
        points = rng.normal(size=(9000, 3))
        points[:, 2] *= 0.002
        points = np.concatenate((points, points[::5]))
        serial = local_geometry(points, workers=1)
        parallel = local_geometry(points, workers=3)
        for key, value in serial.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(parallel[key], value, err_msg=key)
        self.assertEqual(serial["spacing"], parallel["spacing"])


if __name__ == "__main__":
    unittest.main()
