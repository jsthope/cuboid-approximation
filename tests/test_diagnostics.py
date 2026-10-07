"""Preparation diagnostics are opt-in and do not change fitting inputs."""

from dataclasses import fields
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cuboid_approximation.barriers import LineConfig, _package, fit_edge_lines
from cuboid_approximation.cloud import normal_variation
from cuboid_approximation.cuboids import CuboidConfig
from cuboid_approximation.pipeline import parser, prepare
from cuboid_approximation.regions import SurfaceConfig, segment_surfaces
from test_method import write_cloud
from test_regressions import sheet


class PreparationDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.source = self.root / "source.ply"
        write_cloud(self.source, sheet(16))

    def run_prepare(self, name, *options, previous=None):
        run = self.root / name
        output = run / "preparation"
        output.mkdir(parents=True)
        args = parser().parse_args([
            "--ply", str(self.source), "--output", str(run), "--resolution", "32",
            "--max-points", "500", "--max-frames", "2", *options,
        ])
        result = prepare(self.source, output, args, CuboidConfig(resolution=32), previous)
        return output, result

    def test_default_pca_keeps_normals_without_edge_or_visualization_work(self):
        with (
            patch("cuboid_approximation.pipeline.normal_variation", side_effect=AssertionError),
            patch("cuboid_approximation.pipeline.fit_edge_lines", side_effect=AssertionError),
            patch("cuboid_approximation.pipeline.write_points", side_effect=AssertionError),
        ):
            output, (_, _, report) = self.run_prepare("minimal")
        self.assertEqual(
            {path.name for path in output.iterdir()},
            {"common_geometry.npz", "regions.npz", "report.json"},
        )
        with np.load(output / "common_geometry.npz") as common:
            self.assertTrue(common["normal_valid"].all())
            self.assertEqual(common["normals"].shape, (256, 3))
        self.assertFalse(report["edge_detection_computed"])
        self.assertIsNone(report["edge_points"])

    def test_diagnostics_enable_artifacts_without_changing_fitting_geometry(self):
        _, (_, minimal, _) = self.run_prepare("minimal")
        output, (_, detailed, report) = self.run_prepare("detailed", "--diagnostics")
        for name in ("source_sample.ply", "regions.ply", "edges.ply", "edges.npz"):
            self.assertTrue((output / name).is_file())
        self.assertTrue(report["diagnostics_enabled"])
        for key in ("points", "normals", "labels", "assignment_kind", "segments"):
            np.testing.assert_array_equal(minimal[key], detailed[key])

    def test_region_mode_keeps_edge_work_but_skips_optional_packaging(self):
        with (
            patch("cuboid_approximation.pipeline.normal_variation", wraps=normal_variation) as edge,
            patch("cuboid_approximation.pipeline.fit_edge_lines", wraps=fit_edge_lines) as barrier,
        ):
            output, (_, regions, report) = self.run_prepare("regions", "--orientation-mode", "regions")
        edge.assert_called_once()
        self.assertFalse(barrier.call_args.kwargs["diagnostics"])
        self.assertTrue(report["edge_detection_computed"])
        self.assertTrue(np.any(regions["assignment_kind"] == 1))
        self.assertFalse((output / "edges.npz").exists())
        self.assertFalse((output / "barriers.npz").exists())

    def test_resume_can_add_or_omit_diagnostics_while_reusing_essential_arrays(self):
        first, _ = self.run_prepare("first")
        second, (_, _, report) = self.run_prepare(
            "second", "--diagnostics", previous=first.parent
        )
        self.assertTrue(report["reused_preparation"])
        self.assertTrue((second / "edges.npz").is_file())
        self.assertEqual(
            (first / "common_geometry.npz").read_bytes(),
            (second / "common_geometry.npz").read_bytes(),
        )
        with patch("cuboid_approximation.pipeline.normal_variation", side_effect=AssertionError):
            third, _ = self.run_prepare("third", previous=second.parent)
        self.assertFalse((third / "edges.npz").exists())
        self.assertFalse(list(third.glob("*.ply")))


class BarrierDiagnosticsTests(unittest.TestCase):
    def test_minimal_package_does_not_traverse_supports(self):
        class ExpensiveSupports:
            def __iter__(self):
                raise AssertionError("Unrequested diagnostics traversed every support")

        segment = np.array([[[0.0, 0, 0], [1.0, 0, 0]]])
        result = _package(segment, ExpensiveSupports(), None, None, {})
        self.assertEqual(set(result), {"segments"})
        np.testing.assert_array_equal(result["segments"], segment)

    def test_detailed_and_minimal_barriers_have_identical_segments(self):
        points = np.c_[np.linspace(0, 1, 80), np.zeros((80, 2))]
        scores, mask = np.full(80, 30.0), np.ones(80, bool)
        config = LineConfig(max_trials=30, trials_per_step=4)
        minimal = fit_edge_lines(points, scores, mask, 1 / 79, config)
        detailed = fit_edge_lines(points, scores, mask, 1 / 79, config, diagnostics=True)
        self.assertTrue(len(minimal["segments"]))
        self.assertEqual(set(minimal), {"segments"})
        np.testing.assert_array_equal(minimal["segments"], detailed["segments"])
        self.assertIn("support_point_indices", detailed)
        self.assertIn("rmse", detailed)

    def test_removed_border_attachment_is_explicit(self):
        self.assertFalse(any(field.name.startswith("attachment_") for field in fields(SurfaceConfig)))
        with self.assertRaisesRegex(ValueError, "attach_borders=True is no longer supported"):
            segment_surfaces(None, None, None, None, None, None, attach_borders=True)


if __name__ == "__main__":
    unittest.main()
