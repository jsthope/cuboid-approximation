"""Known geometry, density, color and interrupted-run regressions from the review."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from cuboid_approximation.approximation import spatial_weights, surface_metrics
from cuboid_approximation.checkpoint import SearchCheckpoint
from cuboid_approximation.cloud import load_points
from cuboid_approximation.cuboids import (
    CuboidConfig,
    certify_box,
    region_frames,
    select_incremental_boxes,
)
from cuboid_approximation.fitting import fit_cuboids
from cuboid_approximation.geometry import (
    PointIndex,
    SurfaceMesh,
    box_surface_distance,
    exposed_triangles,
)
from cuboid_approximation.pipeline import main
from cuboid_approximation.texture import bake_face, transfer_normals, pack_atlas
from cuboid_approximation.volume import conservative_solid, preflight_grid, check_grid_memory
from test_method import box, write_cloud
from test_regressions import sheet, regions


def shell(n=21):
    points = []
    for axis in range(3):
        for side in (0, 1):
            q = sheet(n)
            q = np.roll(q, axis, axis=1)
            q[:, (axis + 2) % 3] = side
            points.append(q)
    return np.unique(np.vstack(points), axis=0)


class BoundaryTests(unittest.TestCase):
    def test_cube_exact_source_distance_has_no_sampling_bias(self):
        report = surface_metrics([box([0.5] * 3, [1] * 3)], shell(), 0.001)
        self.assertLess(report["source_to_surface_rms"], 1e-14)
        self.assertAlmostEqual(report["estimated_exposed_area"], 6)

    def test_contacts_overlaps_and_duplicate_faces_are_removed(self):
        for shift, area in ((1.0, 10.0), (0.5, 8.0), (0.0, 6.0)):
            boxes = [box([0.5] * 3, [1] * 3), box([0.5 + shift, 0.5, 0.5], [1] * 3)]
            mesh = SurfaceMesh(exposed_triangles(boxes))
            self.assertAlmostEqual(mesh.areas.sum(), area)
            # Internal interfaces must not masquerade as exposed surfaces.
            self.assertAlmostEqual(
                mesh.distances(np.array([[0.75, 0.5, 0.5]]))[0], 0.25 if shift == 0 else 0.5
            )

    def test_rotated_surface_distance_matches_analytic_box(self):
        b = box([2, 3, 4], [1, 2, 0.3])
        b["rotation"] = Rotation.from_euler("xyz", [23, 47, 12], degrees=True).as_matrix()
        p = np.random.default_rng(7).uniform(1, 5, (1000, 3))
        np.testing.assert_allclose(
            SurfaceMesh(exposed_triangles([b])).distances(p), box_surface_distance(p, b), atol=1e-12
        )

    def test_boundary_is_invariant_to_scale(self):
        for scale in (1e-10, 1.0, 1000.0):
            boxes = [
                box(np.array([0.5] * 3) * scale, np.ones(3) * scale),
                box(np.array([1.5, 0.5, 0.5]) * scale, np.ones(3) * scale),
            ]
            mesh = SurfaceMesh(exposed_triangles(boxes))
            self.assertAlmostEqual(mesh.areas.sum() / scale**2, 10)

    def test_rotated_union_outside_distance_matches_distance_to_volumes(self):
        from cuboid_approximation.cuboids import box_membership, distance_to_box

        boxes = [box([0, 0, 0], [2, 1, 1]), box([0.6, 0.2, 0], [1, 2, 1])]
        boxes[1]["rotation"] = Rotation.from_euler("xyz", [12, 23, 37], degrees=True).as_matrix()
        points = np.random.default_rng(12).uniform(-2, 2, (1000, 3))
        outside = ~np.any([box_membership(points, b) for b in boxes], axis=0)
        expected = np.min([distance_to_box(points[outside], b) for b in boxes], axis=0)
        mesh = SurfaceMesh(exposed_triangles(boxes))
        np.testing.assert_allclose(mesh.distances(points[outside]), expected, atol=1e-11)
        samples, _ = mesh.samples(2000)
        self.assertFalse(any(box_membership(samples, b, tolerance=-1e-8).any() for b in boxes))


class ReconstructionTests(unittest.TestCase):
    def test_closed_cube_survives_grid_refinement(self):
        p = shell()
        for resolution in (32, 64, 128):
            solid = conservative_solid(p, CuboidConfig(resolution=resolution))
            idx = np.floor((np.full(3, 0.5) - solid["origin"]) / solid["voxel_size"]).astype(int)
            self.assertTrue(solid["safe"][tuple(idx)])

    def test_open_cube_does_not_become_solid(self):
        p = shell()
        p = p[p[:, 2] < 1]
        for resolution in (32, 64):
            solid = conservative_solid(p, CuboidConfig(resolution=resolution))
            idx = np.floor((np.full(3, 0.5) - solid["origin"]) / solid["voxel_size"]).astype(int)
            self.assertFalse(solid["safe"][tuple(idx)])

    def test_default_cube_support_and_candidate_workspace_fit_budget(self):
        config = CuboidConfig()
        shape = preflight_grid(shell(100), config, 1 / 99)
        pad = int(np.ceil(2 * max(1 / config.resolution, 1 / 99) * config.resolution)) + 1
        padded = shape + 2 * pad
        check_grid_memory(padded, config.max_memory_mb, "candidate worst case", 128)


class FittingQualityTests(unittest.TestCase):
    def test_export_roundoff_is_corrected_before_checkpoint_commit(self):
        safe = np.zeros((10, 10, 10), bool)
        safe[2:8, 2:8, 2:8] = True
        solid = dict(safe=safe, origin=np.zeros(3), voxel_size=1.0)
        exported = dict(safe=safe, origin=np.full(3, 0.123), voxel_size=0.1)

        def export_certificate(b):
            world = dict(b, center=b["center"] * 0.1 + 0.123, dimensions=b["dimensions"] * 0.1)
            return certify_box(world, exported, strict=True)

        candidate = box([5] * 3, [6] * 3)
        self.assertTrue(certify_box(candidate, solid, strict=True))
        self.assertFalse(export_certificate(candidate))
        saved = []
        boxes, *_ = select_incremental_boxes(
            [candidate],
            [],
            np.array([[5.0, 5, 5]]),
            0.1,
            CuboidConfig(max_cuboids=1),
            solid,
            lambda _: None,
            export_certificate=export_certificate,
            checkpoint=lambda boxes, _: saved.append(export_certificate(boxes[-1])),
        )
        self.assertTrue(export_certificate(boxes[0]))
        self.assertEqual(saved, [True])

    def test_perfect_cube_is_recovered_in_one_box(self):
        p = shell(33)
        with patch(
            "cuboid_approximation.fitting.generate_candidates",
            side_effect=AssertionError("unnecessary grids for a certified exact box"),
        ):
            result = fit_cuboids(
                p,
                regions(p),
                1 / 32,
                CuboidConfig(resolution=32, max_frames=2, seeds_per_frame=4, max_cuboids=8),
                progress=lambda _: None,
            )
        self.assertEqual(len(result["boxes"]), 1)
        np.testing.assert_allclose(result["boxes"][0]["dimensions"], 1, atol=1e-7)
        report = surface_metrics(result["boxes"], p, result["surface_tolerance"])
        self.assertGreater(report["source_surface_coverage"], 0.999)

    def test_density_weights_give_each_cell_equal_mass(self):
        points = np.vstack((np.repeat([[0.0, 0, 0]], 10000, axis=0), [[10.0, 0, 0]]))
        weights = spatial_weights(points, 0.01)
        self.assertAlmostEqual(weights[:-1].sum(), weights[-1])

    def test_dense_component_cannot_hide_uncovered_sparse_component(self):
        points = np.vstack((np.repeat([[2.0, 2, 2]], 10000, axis=0), [[8.0, 2, 2]]))
        safe = np.zeros((12, 5, 5), bool)
        safe[1:4, 1:4, 1:4] = True
        safe[7:10, 1:4, 1:4] = True
        solid = dict(
            safe=safe, reference=safe, voxel_size=1.0, origin=np.zeros(3), approximation={}
        )
        boxes, _, remaining, _, _ = select_incremental_boxes(
            [box([2.5] * 3, [2] * 3), box([8.5, 2.5, 2.5], [2] * 3)],
            [],
            points,
            0.1,
            CuboidConfig(max_cuboids=2),
            solid,
            lambda _: None,
        )
        self.assertEqual(len(boxes), 2)
        self.assertFalse(remaining.any())

    def test_sparse_legitimate_component_survives_local_outlier_filter(self):
        p = np.vstack((sheet(20) * 0.001, sheet(8) + [5, 0, 0], [10000, 0, 0]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.ply"
            write_cloud(path, p)
            result = load_points(path)
        np.testing.assert_array_equal(result["outlier_source_indices"], [len(p) - 1])

    def test_single_frame_skips_region_covariances(self):
        with patch("numpy.linalg.eigh", side_effect=AssertionError("unnecessary PCA")):
            frames, _, _ = region_frames(
                sheet(), np.zeros(324), np.ones(324), 0.1, CuboidConfig(max_frames=1)
            )
        np.testing.assert_array_equal(frames, np.eye(3)[None])

    def test_aabb_index_matches_brute_force_with_boundary_duplicates(self):
        p = np.vstack((np.random.default_rng(7).normal(size=(2000, 3)), [[1, 1, 1]] * 5))
        index = PointIndex(p)
        for half in ([1, 1, 1], [0.01, 5, 0.01], [0, 0, 0]):
            half = np.array(half)
            expected = np.flatnonzero(np.all(np.abs(p) <= half, axis=1))
            np.testing.assert_array_equal(np.sort(index.query(np.zeros(3), half)), expected)


class ProjectionQualityTests(unittest.TestCase):
    def test_srgb_blending_occurs_in_linear_light(self):
        points = np.array([[0.5, 0.5, 0.01]] * 2)
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
            colors=np.array([[0.0, 0, 0], [1.0, 1, 1]]),
            normals=np.tile([0.0, 0, 1], (2, 1)),
            opacity=np.ones(2),
            covariance=np.tile(np.eye(3) * 0.01, (2, 1, 1)),
            spacing=0.1,
            tree=cKDTree(points),
        )
        rgb, supported, _ = bake_face(face, 1, 1, cloud, 0.2, 0.2)
        np.testing.assert_allclose(rgb, 0.735356983, atol=1e-8)
        self.assertTrue(supported.all())

    def test_invalid_nearest_normal_does_not_block_valid_neighbors(self):
        points = sheet(10)
        normals = np.tile([0.0, 0, 1], (len(points), 1))
        valid = np.ones(len(points), bool)
        valid[55] = False
        output, confidence = transfer_normals(points, points, normals, valid, 1 / 9, 1 / 9)
        np.testing.assert_allclose(np.abs(output[55]), [0, 0, 1])
        self.assertGreater(confidence[55], 0.5)

    def test_atlas_rectangles_and_gutters_do_not_overlap(self):
        faces = [
            dict(width=float(w), height=float(h))
            for w, h in np.random.default_rng(5).uniform(0.1, 10, (80, 2))
        ]
        rects, _ = pack_atlas(faces, 512)
        for i, (x, y, w, h) in enumerate(rects):
            self.assertGreaterEqual(min(x, y), 4)
            self.assertLessEqual(max(x + w + 4, y + h + 4), 512)
            for a, b, c, d in rects[i + 1 :]:
                self.assertTrue(
                    x + w + 4 <= a - 4
                    or a + c + 4 <= x - 4
                    or y + h + 4 <= b - 4
                    or b + d + 4 <= y - 4
                )


class CheckpointQualityTests(unittest.TestCase):
    def test_volume_coverage_cannot_hide_an_unrepresented_internal_surface(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.ply"
            interior = sheet(12) * 0.4 + [0.3, 0.3, 0.5]
            write_cloud(source, np.vstack((shell(17), interior)))
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "--ply",
                        str(source),
                        "--output",
                        str(root / "run"),
                        "--resolution",
                        "32",
                        "--max-frames",
                        "1",
                        "--seeds-per-frame",
                        "4",
                        "--max-cuboids",
                        "1",
                        "--atlas-size",
                        "256",
                    ]
                )
            report = json.loads((root / "run/report.json").read_text())["geometry"]
            self.assertEqual(code, 3)
            # A single solid cube would bury the internal sheet. The search must
            # now reject it before commitment, even though it covers the volume.
            self.assertFalse(report["volume_target_reached"])
            self.assertFalse(report["surface_target_reached"])
            self.assertLessEqual(report["selection"]["irreparable_spatial_fraction"], 0.001)
            self.assertEqual(report["selection"]["stopping_reason"], "maximum_box_budget")

    def test_interruption_preserves_search_and_resume_reuses_preparation(self):
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
                "--max-cuboids",
                "8",
                "--atlas-size",
                "256",
                "--target-coverage",
                "1",
            ]
            original = SearchCheckpoint.__call__

            def interrupt(self, boxes, *args):
                original(self, boxes, *args)
                if len(boxes) == 1:
                    raise RuntimeError("simulated interruption")

            first, second = root / "first", root / "second"
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                with patch.object(SearchCheckpoint, "__call__", interrupt):
                    self.assertEqual(main(base + ["--output", str(first)]), 2)
                self.assertEqual(
                    json.loads((first / "checkpoint.json").read_text())["committed_count"], 1
                )
                with patch(
                    "cuboid_approximation.pipeline.local_geometry",
                    side_effect=AssertionError("cache not reused"),
                ):
                    self.assertEqual(
                        main(base + ["--output", str(second), "--resume", str(first)]), 0
                    )
            report = json.loads((second / "report.json").read_text())
            self.assertTrue(report["preparation"]["reused_preparation"])
            self.assertEqual(report["geometry"]["selection"]["frozen_prefix_count"], 1)
            # Recorded code provenance is part of compatibility, not merely informational.
            report["implementation_sha256"]["cuboids.py"] = "0" * 64
            (second / "report.json").write_text(json.dumps(report))
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(
                    main(base + ["--output", str(root / "third"), "--resume", str(second)]), 2
                )
            self.assertFalse((root / "third").exists())


if __name__ == "__main__":
    unittest.main()
