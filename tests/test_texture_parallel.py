"""Parallel projection keeps the serial atlas, confidence, and export ordering."""

from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from cuboid_approximation.geometry import CORNER_SIGNS
from cuboid_approximation.texture import _ordered_face_results, bake_model


class TextureParallelTests(unittest.TestCase):
    def test_scheduler_runs_concurrently_and_returns_input_order(self):
        barrier = threading.Barrier(4, timeout=5)

        def work(index):
            if index < 4:
                barrier.wait()
            return index * index

        self.assertEqual(list(_ordered_face_results(work, range(13), 4)),
                         [i * i for i in range(13)])

    def test_parallel_exports_match_serial_with_slabs_hidden_faces_and_gaps(self):
        rotation = Rotation.from_euler("xyz", [19, 31, -13], degrees=True).as_matrix()
        centers = np.array([[0., 0, 0], [0., 0, 0], [1.2, 0.1, 0.1]])
        dimensions = np.array([[1., 0.8, 0.06], [0.1, 0.1, 0.02], [0.4, 0.4, 0.3]])
        x, y = np.meshgrid(np.linspace(-0.45, 0.45, 9), np.linspace(-0.35, 0.35, 7))
        front = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 0.05)))
        points = np.vstack((front, front * [1, 1, -1])) @ rotation.T
        colors = np.random.default_rng(3).random((len(points), 3))
        source = dict(
            points=points, colors=colors, color_space="srgb", opacity=np.ones(len(points)),
            normals=np.tile([0., 0, 1], (len(points), 1)) @ rotation.T,
            normal_confidence=np.ones(len(points)), spacing=0.025,
            spacing_per_point=np.full(len(points), 0.025),
            covariance=np.tile(np.eye(3) * 0.02**2, (len(points), 1, 1)),
            tree=cKDTree(points), color_source="vertex_rgb", anisotropic_splats=0,
            kernel_normals_recovered=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "parameters.npz", centers=centers @ rotation.T,
                     dimensions=dimensions, rotations=np.tile(rotation, (len(centers), 1, 1)),
                     surface_tolerance=0.04,
                     corners=(centers[:, None] + CORNER_SIGNS * dimensions[:, None] / 2) @ rotation.T)
            np.savez(root / "common_geometry.npz", points=points, colors=colors)
            with patch("cuboid_approximation.texture.load_cloud", return_value=source):
                serial, one = bake_model(root / "unused.ply", root, root, root / "one", 256)
                parallel, four = bake_model(root / "unused.ply", root, root, root / "four", 256,
                                            workers=4)
            self.assertNotIn("_projection_bounds", source)
            self.assertEqual(one["workers"], 1)
            self.assertEqual(four["workers"], 4)
            self.assertGreater(one["hidden_faces_skipped"], 0)
            self.assertTrue(any(f["outward_layer_order"] for f in one["face_details"]))
            for name in ("texture_atlas.png", "projection_confidence.png"):
                np.testing.assert_array_equal(np.asarray(Image.open(serial / name)),
                                              np.asarray(Image.open(parallel / name)))
            for name in ("cuboids_textured.obj", "cuboids_textured.glb"):
                self.assertEqual((serial / name).read_bytes(), (parallel / name).read_bytes())
            with np.load(serial / "texture_parameters.npz") as a, np.load(
                parallel / "texture_parameters.npz"
            ) as b:
                self.assertEqual(a.files, b.files)
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])
            for report in (one, four):
                report.pop("seconds")
                report.pop("workers")
            self.assertEqual(one, four)

    def test_invalid_worker_budget_is_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "output"
            for workers in (0, -1, 1.5, True):
                with self.subTest(workers=workers), self.assertRaisesRegex(ValueError, "workers"):
                    bake_model("unused", "unused", "unused", out, workers=workers)
                self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
