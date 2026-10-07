"""Regressions for units, coverage, memory, projection and exported geometry."""

import base64
import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from cuboid_approximation.approximation import exterior_fraction, surface_metrics
from cuboid_approximation.cloud import load_points, local_geometry, spatial_representatives
from cuboid_approximation.cuboids import (
    CuboidConfig,
    box_corners,
    certify_box,
    region_frames,
    select_incremental_boxes,
)
from cuboid_approximation.fitting import fit_cuboids
from cuboid_approximation.pipeline import main
from cuboid_approximation.texture import (
    bake_face,
    export_glb,
    export_viewer,
    faces_from_corners,
    pack_atlas,
    up_rotation,
)
from cuboid_approximation.volume import (
    center_spacing,
    check_grid_memory,
    conservative_solid,
    splat_occupancy,
)
from test_method import box, write_cloud


def sheet(n=18):
    x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    return np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))


def regions(points):
    return dict(
        points=points, labels=np.zeros(len(points), int), assignment_kind=np.ones(len(points), int)
    )


class InputTests(unittest.TestCase):
    def test_duplicate_centers_do_not_change_spacing(self):
        points = sheet()
        self.assertEqual(
            center_spacing(points, 0.001), center_spacing(np.repeat(points, 3, axis=0), 0.001)
        )

    def test_duplicate_centers_do_not_change_sampling(self):
        points = sheet(12)
        repeated = np.repeat(points, 3, axis=0)
        for budget in (100, 200):
            ids, size = spatial_representatives(points, budget)
            repeated_ids, repeated_size = spatial_representatives(repeated, budget)
            np.testing.assert_array_equal(points[ids], repeated[repeated_ids])
            self.assertEqual(size, repeated_size)

    def test_sampler_refines_initial_grid_and_retains_small_components(self):
        points = np.vstack((sheet(30), [10000, 0, 0]))
        ids, _ = spatial_representatives(points, 100)
        # A square grid plus one remote cell jumps directly from 82 to 101 cells.
        self.assertGreaterEqual(len(ids), 80)
        self.assertIn(900, ids)

    def test_extreme_outlier_is_reported_and_filter_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "points.ply"
            write_cloud(path, np.vstack((sheet(30), [10000, 0, 0])))
            loaded = load_points(path)
            np.testing.assert_array_equal(loaded["outlier_source_indices"], [900])
            self.assertEqual(len(loaded["points"]), 900)
            self.assertEqual(len(load_points(path, outlier_distance_factor=0)["points"]), 901)

    def test_normals_are_invariant_to_scale_and_large_translation(self):
        points = sheet()
        original = local_geometry(points)
        for transformed in (points * 1e-10, points * 1000, points + 1e8):
            actual = local_geometry(transformed)
            np.testing.assert_array_equal(actual["valid"], original["valid"])
            np.testing.assert_allclose(
                np.abs(actual["normals"]), np.abs(original["normals"]), atol=1e-10
            )

    def test_volumetric_noise_is_not_declared_a_reliable_surface(self):
        points = np.random.default_rng(3).uniform(-1, 1, (5000, 3))
        self.assertLess(local_geometry(points)["valid"].mean(), 0.15)

    def test_missing_vertex_returns_documented_error_code(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "invalid.ply"
            PlyData([PlyElement.describe(np.zeros(1, dtype=[("dummy", "i4")]), "face")]).write(
                str(path)
            )
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                code = main(["--ply", str(path), "--output", str(Path(folder) / "result")])
            self.assertEqual(code, 2)


class VolumeTests(unittest.TestCase):
    def test_sparse_tiny_splats_use_actual_bounds(self):
        points = np.stack(np.meshgrid(*[np.linspace(0, 1, 5)] * 3, indexing="ij"), axis=-1).reshape(
            -1, 3
        )
        shapes = dict(
            log_scales=np.full((len(points), 3), np.log(1e-5)),
            quaternions_wxyz=np.tile([1.0, 0, 0, 0], (len(points), 1)),
        )

        # Capture the allocation dimensions without constructing a 320^3 volume.
        class Captured(Exception):
            pass

        def capture(points, shapes, origin, h, grid_shape, *args):
            self.assertLess(max(grid_shape), 340)
            raise Captured

        with patch("cuboid_approximation.volume.splat_occupancy", side_effect=capture):
            with self.assertRaises(Captured):
                conservative_solid(points, CuboidConfig(max_memory_mb=4096), shapes)

    def test_memory_budget_rejects_before_allocation(self):
        with self.assertRaisesRegex(ValueError, "Lower --resolution"):
            check_grid_memory((1929, 1929, 1929), 2048, "test")

    def test_invalid_splats_preserve_centers(self):
        points = np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]])
        shapes = dict(log_scales=np.full((2, 3), np.nan), quaternions_wxyz=np.zeros((2, 4)))
        occupied, report = splat_occupancy(points, shapes, np.zeros(3), 1.0, (3, 3, 3))
        self.assertEqual(occupied.sum(), 2)
        self.assertEqual(report["center_only_splats"], 2)

    def test_surface_mode_does_not_fill_a_closed_shell(self):
        points = sheet(40)
        shell = []
        for axis in range(3):
            for sign in (0, 1):
                face = np.roll(points, axis, axis=1)
                face[:, (2 + axis) % 3] = sign
                shell.append(face)
        points = np.unique(np.vstack(shell), axis=0)
        for mode, expected in (("solid", True), ("surface", False)):
            solid = conservative_solid(
                points, CuboidConfig(resolution=32, reconstruction_mode=mode)
            )
            idx = np.floor((np.full(3, 0.5) - solid["origin"]) / solid["voxel_size"]).astype(int)
            self.assertEqual(bool(solid["safe"][tuple(idx)]), expected)

    def test_axis_aligned_exterior_is_exact_for_stripes_and_partial_cells(self):
        reference = np.zeros((96, 2, 2), bool)
        reference[::2] = True
        solid = dict(reference=reference, voxel_size=1.0, origin=np.zeros(3))
        self.assertAlmostEqual(exterior_fraction(box([48, 1, 1], [96, 2, 2]), solid), 0.5)
        self.assertAlmostEqual(exterior_fraction(box([1, 1, 1], [1, 1, 1]), solid), 0.5)
        self.assertAlmostEqual(exterior_fraction(box([0.25, 1, 1], [0.5, 1, 1]), solid), 0)

    def test_rotated_exterior_does_not_alias_periodic_support(self):
        reference = np.zeros((100, 8, 8), bool)
        reference[::2] = True
        solid = dict(reference=reference, voxel_size=1.0, origin=np.zeros(3))
        b = box([50, 4, 4], [90, 1, 1])
        b["rotation"] = Rotation.from_euler("x", 17, degrees=True).as_matrix()
        self.assertAlmostEqual(exterior_fraction(b, solid, max_axis=24), 0.5, delta=0.06)


class SearchTests(unittest.TestCase):
    def test_certificate_is_valid_in_export_coordinates(self):
        points = sheet() + 1e8
        result = fit_cuboids(
            points,
            regions(points),
            1 / 17,
            CuboidConfig(resolution=32, max_frames=1, seeds_per_frame=1, max_cuboids=2),
            progress=lambda _: None,
        )
        self.assertTrue(all(certify_box(b, result["solid"], strict=True) for b in result["boxes"]))

    def test_curved_cloud_has_non_world_orientation(self):
        rng = np.random.default_rng(42)
        points = rng.normal(size=(2000, 3))
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        points = (points * [3, 1, 0.6]) @ Rotation.from_euler(
            "xyz", [25, 37, 19], degrees=True
        ).as_matrix().T
        frames, _, faces = region_frames(
            points, np.zeros(len(points), int), np.ones(len(points), int), 0.03, CuboidConfig()
        )
        self.assertFalse(faces)
        self.assertGreater(len(frames), 1)
        for frame in frames:
            np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(frame), 1)

    def test_full_cloud_target_includes_points_missing_from_sample(self):
        sampled = sheet(12)
        full = np.vstack((sampled, sampled + [3, 0, 0]))
        result = fit_cuboids(
            full,
            regions(sampled),
            1 / 11,
            CuboidConfig(
                resolution=32, max_frames=1, seeds_per_frame=1, max_cuboids=8, target_coverage=1
            ),
            progress=lambda _: None,
        )
        self.assertEqual(result["selection"]["stopping_reason"], "target_reached")
        self.assertTrue(np.all(result["full_distances"] <= result["surface_tolerance"]))

    def test_tolerance_is_independent_of_normal_sample_budget(self):
        points = sheet(18)
        tolerance = []
        for sample in (points, points[::3]):
            result = fit_cuboids(
                points,
                regions(sample),
                local_geometry(sample)["spacing"],
                CuboidConfig(resolution=32, max_frames=1, seeds_per_frame=1, max_cuboids=2),
                progress=lambda _: None,
            )
            tolerance.append(result["surface_tolerance"])
        self.assertEqual(*tolerance)

    def test_residual_proposals_recover_empty_initial_pool(self):
        safe = np.ones((5, 5, 5), bool)
        solid = dict(safe=safe, voxel_size=1.0, origin=np.zeros(3))
        chosen, _, remaining, _, _ = select_incremental_boxes(
            [],
            [],
            np.array([[2.5, 2.5, 2.5]]),
            0.1,
            CuboidConfig(max_cuboids=1),
            solid,
            lambda _: None,
            propose=lambda _: [box([2.5, 2.5, 2.5], [1, 1, 1])],
        )
        self.assertEqual(len(chosen), 1)
        self.assertFalse(remaining.any())
        self.assertTrue(certify_box(chosen[0], solid, strict=True))

    def test_surface_metrics_detect_points_buried_inside_volume(self):
        report = surface_metrics([box([0, 0, 0], [2, 2, 2])], np.zeros((1, 3)), 0.01)
        self.assertGreater(report["source_to_surface_rms"], 0.9)
        self.assertGreater(report["surface_to_source_rms"], 0.9)


class TextureTests(unittest.TestCase):
    def test_raster_uv_orientation_and_ambiguous_layers(self):
        points = np.array(
            [[0.25, 0.75, 0.01], [0.75, 0.75, 0.01], [0.25, 0.25, 0.01], [0.75, 0.25, 0.01]]
        )
        colors = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 1.0]])
        face = dict(
            origin=np.zeros(3),
            u=np.array([1.0, 0, 0]),
            v=np.array([0.0, 1, 0]),
            normal=np.array([0.0, 0, 1]),
            width=1.0,
            height=1.0,
        )
        cloud = dict(
            points=points,
            colors=colors,
            normals=np.tile([0.0, 0, 1], (4, 1)),
            opacity=np.ones(4),
            covariance=np.tile(np.eye(3) * 0.1**2, (4, 1, 1)),
            spacing=0.1,
            tree=cKDTree(points),
        )
        rgb, support, _ = bake_face(face, 2, 2, cloud, 0.2, 0.2)
        np.testing.assert_allclose(rgb.reshape(-1, 3), colors)
        self.assertTrue(support.all())
        points = np.array([[0.5, 0.5, -0.06], [0.5, 0.5, 0.06]])
        cloud.update(
            points=points,
            colors=colors[:2],
            normals=cloud["normals"][:2],
            opacity=np.ones(2),
            covariance=cloud["covariance"][:2],
            tree=cKDTree(points),
        )
        _, support, _ = bake_face(face, 1, 1, cloud, 0.2, 0.2)
        self.assertFalse(support.any())

    def test_atlas_is_invariant_to_world_units(self):
        faces = [dict(width=1.0, height=0.5)] * 6
        a, density = pack_atlas(faces, 256)
        for scale in (1e-10, 1000):
            b, scaled_density = pack_atlas(
                [dict(width=f["width"] * scale, height=f["height"] * scale) for f in faces], 256
            )
            self.assertEqual(a, b)
            self.assertAlmostEqual(density, scaled_density * scale, places=8)

    def test_sparse_near_layer_wins_over_dense_far_layer(self):
        face = dict(
            origin=np.zeros(3),
            u=np.array([1.0, 0, 0]),
            v=np.array([0.0, 1, 0]),
            normal=np.array([0.0, 0, 1]),
            width=1.0,
            height=1.0,
        )
        red = np.array([[0.515, 0.5, 0.01]])
        blue = np.column_stack(
            (0.5 + np.linspace(-0.001, 0.001, 64), np.full(64, 0.5), np.full(64, 0.1))
        )
        points = np.vstack((red, blue))
        cloud = dict(
            points=points,
            colors=np.vstack(([1.0, 0, 0], np.tile([0.0, 0, 1.0], (64, 1)))),
            normals=np.tile([0.0, 0, 1.0], (65, 1)),
            spacing=0.03,
            opacity=np.ones(65),
            covariance=np.tile(np.eye(3) * 0.03**2, (65, 1, 1)),
            tree=cKDTree(points),
        )
        rgb, supported, depth = bake_face(face, 1, 1, cloud, 0.2, 0.2)
        np.testing.assert_allclose(rgb[0, 0], [1, 0, 0])
        self.assertTrue(supported[0, 0])
        self.assertAlmostEqual(depth[0], 0.01)
        cloud["opacity"][:] = 0
        rgb, supported, _ = bake_face(face, 1, 1, cloud, 0.2, 0.2)
        self.assertTrue(np.isfinite(rgb).all())
        self.assertFalse(supported.any())

    def test_fallback_does_not_borrow_distant_colors(self):
        points = np.array([[100.0, 100, 100]])
        face = dict(
            origin=np.zeros(3),
            u=np.array([1.0, 0, 0]),
            v=np.array([0.0, 1, 0]),
            normal=np.array([0.0, 0, 1]),
            width=1.0,
            height=1.0,
        )
        cloud = dict(
            points=points,
            colors=np.array([[1.0, 0, 0]]),
            normals=np.zeros((1, 3)),
            spacing=0.01,
            opacity=np.ones(1),
            covariance=np.eye(3)[None],
            tree=cKDTree(points),
        )
        rgb, supported, _ = bake_face(face, 1, 1, cloud, 0.1, 0.1)
        np.testing.assert_allclose(rgb, 0.65)
        self.assertFalse(supported.any())

    def test_exports_preserve_geometry_and_viewer_precision(self):
        b = box([1e8, 1e8, 1e8], [1000, 2000, 3000])
        b["rotation"] = Rotation.from_euler("xyz", [13, 29, 51], degrees=True).as_matrix()
        params = dict(
            centers=b["center"][None],
            dimensions=b["dimensions"][None],
            rotations=b["rotation"][None],
            corners=box_corners(b)[None],
        )
        faces = faces_from_corners(params["corners"])
        rects, _ = pack_atlas(faces, 256)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.new("RGB", (256, 256)).save(root / "texture_atlas.png")
            Image.new("RGB", (256, 256)).save(root / "projection_confidence.png")
            export_glb(root, params, faces, rects, 256, source_up="z", units_per_meter=1000)
            raw = (root / "cuboids_textured.glb").read_bytes()
            length = struct.unpack("<I", raw[12:16])[0]
            doc = json.loads(raw[20 : 20 + length])
            accessor = doc["accessors"][doc["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
            view = doc["bufferViews"][accessor["bufferView"]]
            local = np.frombuffer(
                raw,
                dtype="<f4",
                count=accessor["count"] * 3,
                offset=28 + length + view["byteOffset"],
            ).reshape(-1, 3)
            node = np.array(doc["nodes"][0]["matrix"]).reshape(4, 4).T
            root_matrix = np.array(doc["nodes"][-1]["matrix"]).reshape(4, 4).T
            source = local @ node[:3, :3].T + node[:3, 3]
            world = source @ root_matrix[:3, :3].T
            restored = world @ up_rotation("z") * 1000 + doc["extras"]["source_origin"]
            expected = np.concatenate([params["corners"][0][f["indices"]] for f in faces])
            np.testing.assert_allclose(restored, expected, rtol=0, atol=2e-8)
            common = root / "common.npz"
            points = sheet() + 1e8
            np.savez(common, points=points, colors=np.zeros_like(points))
            report = dict(cuboids=1, faces=6, atlas_size=[256, 256], supported_area_fraction=1.0)
            export_viewer(root, params, faces, rects, 256, report, common)
            html = (root / "cuboids_textured_3d.html").read_text()
            payload = json.loads(html.split("const data=", 1)[1].split(";\n", 1)[0])
            displayed = np.frombuffer(base64.b64decode(payload["points"]), dtype="<f4").reshape(
                -1, 3
            )
            self.assertGreater(np.ptp(displayed, axis=0).max(), 0.9)


class ResumeTests(unittest.TestCase):
    def test_resume_preserves_prefix_and_appends_to_reach_full_cloud(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.ply"
            write_cloud(source, np.vstack((sheet(12), sheet(12) + [3, 0, 0])))
            base = [
                "--ply",
                str(source),
                "--resolution",
                "32",
                "--max-points",
                "1000",
                "--max-frames",
                "2",
                "--seeds-per-frame",
                "2",
                "--atlas-size",
                "256",
                "--target-coverage",
                "1",
            ]
            first, second = root / "first", root / "second"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(base + ["--output", str(first), "--max-cuboids", "1"]), 3)
                self.assertEqual(
                    main(
                        base
                        + ["--output", str(second), "--max-cuboids", "8", "--resume", str(first)]
                    ),
                    0,
                )
            with (
                np.load(first / "cuboids/parameters.npz") as a,
                np.load(second / "cuboids/parameters.npz") as b,
            ):
                for key in ("centers", "dimensions", "rotations", "corners", "colors"):
                    np.testing.assert_array_equal(a[key], b[key][: len(a[key])])
            report = json.loads((second / "report.json").read_text())
            self.assertEqual(report["geometry"]["full_cloud_proximity"], 1)
            self.assertEqual(report["geometry"]["selection"]["frozen_prefix_count"], 1)
            self.assertIn("dependencies", report["environment"])
            # A damaged checkpoint must fail before creating a new run directory.
            self.assertFalse((first / "cuboids/search_state.npz").exists())
            self.assertNotIn("search_state_sha256", report["geometry"])
            manifest = json.loads((first / "checkpoint.json").read_text())
            checkpoint = first / manifest["state_file"]
            checkpoint.write_bytes(checkpoint.read_bytes() + b"corrupted")
            third = root / "corrupted"
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(main(base + ["--output", str(third), "--resume", str(first)]), 2)
            self.assertFalse(third.exists())


if __name__ == "__main__":
    unittest.main()
