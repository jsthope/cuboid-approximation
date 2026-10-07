"""Geometry contracts and real PLY-to-texture integration tests."""

import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
from plyfile import PlyData, PlyElement

from cuboid_approximation.approximation import approximation_solid
from cuboid_approximation.cloud import local_geometry, normal_variation
from cuboid_approximation.cuboids import (
    CuboidConfig,
    certify_box,
    distance_to_box,
    select_incremental_boxes,
)
from cuboid_approximation.pipeline import main, sha256


def box(center, dimensions):
    return dict(
        center=np.array(center, float),
        dimensions=np.array(dimensions, float),
        rotation=np.eye(3),
        volume=float(np.prod(dimensions)),
    )


def write_cloud(path, points, rgb=True):
    dtype = [(name, "f8") for name in "xyz"]
    if rgb:
        dtype += [(name, "u1") for name in ("red", "green", "blue")]
    data = np.zeros(len(points), dtype=dtype)
    for i, name in enumerate("xyz"):
        data[name] = points[:, i]
    if rgb:
        for name, value in zip(("red", "green", "blue"), (210, 90, 40)):
            data[name] = value
    PlyData([PlyElement.describe(data, "vertex")]).write(str(path))


class GeometryTests(unittest.TestCase):
    def test_distance_is_to_volume_including_interior(self):
        b = box([0, 0, 0], [2, 4, 6])
        distances = distance_to_box(np.array([[0, 0, 0], [1, 2, 3], [2, 3, 4]]), b)
        np.testing.assert_allclose(distances, [0, 0, np.sqrt(3)])

    def test_certificate_detects_internal_void_missed_by_corners(self):
        safe = np.ones((7, 7, 7), bool)
        solid = dict(safe=safe, voxel_size=1.0, origin=np.zeros(3))
        b = box([3.5, 3.5, 3.5], [5.5, 5.5, 5.5])
        self.assertTrue(certify_box(b, solid, strict=True))
        safe[3, 3, 3] = False
        self.assertFalse(certify_box(b, solid, strict=True))

    def test_normal_signs_do_not_change_edges(self):
        x, y = np.meshgrid(np.linspace(-1, 1, 15), np.linspace(-1, 1, 15))
        points = np.column_stack((x.ravel(), y.ravel(), 0.2 * x.ravel() ** 2))
        geometry = local_geometry(points)
        scores, valid, _ = normal_variation(geometry)
        geometry["normals"][::2] *= -1
        changed, changed_valid, _ = normal_variation(geometry)
        np.testing.assert_array_equal(valid, changed_valid)
        np.testing.assert_array_equal(scores, changed)

    def test_observed_thin_sheet_survives_empty_interior(self):
        occupied = np.zeros((5, 5, 5), bool)
        occupied[1:4, 1:4, 2] = True
        solid = dict(
            safe=np.zeros_like(occupied), observed=occupied, voxel_size=1.0, origin=np.zeros(3)
        )
        points = np.argwhere(occupied) + 0.5
        expanded = approximation_solid(solid, points, CuboidConfig())
        self.assertEqual(int(expanded["reference"].sum()), 9)
        self.assertGreater(expanded["safe"].sum(), 9)
        self.assertFalse(expanded["conservative"].any())

    def test_committed_prefix_is_unchanged_when_budget_increases(self):
        safe = np.zeros((12, 5, 5), bool)
        safe[1:4, 1:4, 1:4] = True
        safe[8:11, 1:4, 1:4] = True
        solid = dict(
            safe=safe, reference=safe.copy(), voxel_size=1.0, origin=np.zeros(3), approximation={}
        )
        candidates = [box([2.5, 2.5, 2.5], [2, 2, 2]), box([9.5, 2.5, 2.5], [2, 2, 2])]
        points = np.array([[2, 2, 2], [3, 3, 3], [9, 2, 2], [10, 3, 3]], float)
        short, *_ = select_incremental_boxes(
            candidates, [], points, 0.1, CuboidConfig(max_cuboids=1), solid, lambda _: None
        )
        snapshots = {key: short[0][key].copy() for key in ("center", "dimensions", "rotation")}
        long, *_ = select_incremental_boxes(
            candidates, short, points, 0.1, CuboidConfig(max_cuboids=2), solid, lambda _: None
        )
        self.assertEqual(len(long), 2)
        for key, expected in snapshots.items():
            np.testing.assert_array_equal(short[0][key], expected)
            np.testing.assert_array_equal(long[0][key], expected)


class PipelineTests(unittest.TestCase):
    def run_model(self, points, rgb, target, budget):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "source.ply", root / "result"
            write_cloud(source, points, rgb=rgb)
            original = sha256(source)
            args = [
                "--ply",
                str(source),
                "--output",
                str(output),
                "--resolution",
                "32",
                "--max-points",
                "1000",
                "--max-frames",
                "2",
                "--seeds-per-frame",
                "4",
                "--max-cuboids",
                str(budget),
                "--atlas-size",
                "256",
                "--target-coverage",
                str(target),
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(args)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(sha256(source), original)
            self.assertIn(code, (0, 3))
            self.assertEqual(report["target_reached"], code == 0)
            self.assertEqual(
                report["geometry"]["coverage_population"],
                "all retained source points, including duplicate observations",
            )
            self.assertTrue(report["texture_geometry_unchanged"])
            with (
                np.load(output / "cuboids/parameters.npz") as before,
                np.load(output / "textured/texture_parameters.npz") as after,
            ):
                for key in ("centers", "dimensions", "rotations", "corners"):
                    np.testing.assert_array_equal(before[key], after[key])
            glb = (output / "textured/cuboids_textured.glb").read_bytes()
            self.assertEqual(struct.unpack("<4sII", glb[:12]), (b"glTF", 2, len(glb)))
            length, kind = struct.unpack("<I4s", glb[12:20])
            self.assertEqual(kind, b"JSON")
            document = json.loads(glb[20 : 20 + length])
            self.assertEqual(len(document["meshes"]), report["geometry"]["cuboids"])
            html = (output / "textured/cuboids_textured_3d.html").read_text(encoding="utf-8")
            self.assertNotIn("__PAYLOAD__", html)
            self.assertIn('lang="en"', html)
            self.assertIn('value="0"', html)
            self.assertEqual(report["geometry"]["incremental_steps"][0]["cuboids"], 0)
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(args), 2)  # Existing results must not be overwritten.
            self.assertEqual(report, json.loads((output / "report.json").read_text()))
            return code, report

    def test_plain_xyz_thin_sheet_end_to_end(self):
        x, y = np.meshgrid(np.linspace(0, 1, 18), np.linspace(0, 1, 18))
        points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        code, report = self.run_model(points, False, 0.999, 4)
        self.assertEqual(code, 0)
        self.assertEqual(report["texture"]["color_source"], "neutral_gray_no_source_color")

    def test_budget_exhaustion_keeps_textured_partial_model(self):
        x, y = np.meshgrid(np.linspace(0, 1, 12), np.linspace(0, 1, 12))
        sheet = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        points = np.vstack((sheet, sheet + [3, 0, 0]))
        code, report = self.run_model(points, True, 1.0, 1)
        self.assertEqual(code, 3)
        self.assertEqual(report["status"], "target_not_reached")
        self.assertEqual(report["geometry"]["selection"]["stopping_reason"], "maximum_box_budget")
        self.assertEqual(report["texture"]["color_source"], "vertex_rgb")

    def test_scaled_and_translated_clouds_end_to_end(self):
        x, y = np.meshgrid(np.linspace(0, 1, 18), np.linspace(0, 1, 18))
        points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        for scale, offset in ((1e-10, 0), (1000, 0), (1, 1e8)):
            with self.subTest(scale=scale, offset=offset):
                code, report = self.run_model(points * scale + offset, True, 0.999, 4)
                self.assertEqual(code, 0)
                self.assertGreater(report["texture"]["supported_area_fraction"], 0)


if __name__ == "__main__":
    unittest.main()
