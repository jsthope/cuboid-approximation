"""Fitting compatibility, immutable rebakes and restored certification caches."""

import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cuboid_approximation import pipeline
from cuboid_approximation.checkpoint import atomic_npz, read_checkpoint, sha256, write_json
from cuboid_approximation.cloud import load_points
from cuboid_approximation.cuboids import summed_volume
from cuboid_approximation.fitting import fit_cuboids
from cuboid_approximation.provenance import compatibility_signature, implementation_provenance
from test_method import write_cloud
from test_orientations import cube_shell


class StageSignatureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.package = Path(self.directory.name) / "package"
        shutil.copytree(
            Path(pipeline.__file__).parent,
            self.package,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        self.args = pipeline.parser().parse_args(["--ply", "source.ply", "--output", "result"])

    def signature(self, provenance=None):
        return compatibility_signature(
            self.args,
            "source-digest",
            provenance or implementation_provenance(self.package),
            self.package,
        )

    def test_viewer_texture_and_pillow_changes_preserve_fitting_compatibility(self):
        original = self.signature()
        for name in ("texture.py", "viewer_template.html"):
            path = self.package / name
            path.write_text(path.read_text() + "\n# Export implementation changed\n")
        # Export-only pipeline helpers are outside the explicit fitting boundary.
        path = self.package / "pipeline.py"
        path.write_text(
            path.read_text().replace(
                "def save_geometry(output, result, loaded, regions, config):",
                "def save_geometry(output, result, loaded, regions, config):\n    changed_export = True",
            )
        )
        provenance = implementation_provenance(self.package)
        provenance["environment"]["dependencies"]["Pillow"] = "next-release"
        self.assertEqual(original, self.signature(provenance))

    def test_core_dependencies_and_new_pipeline_helpers_are_covered(self):
        original = self.signature()
        provenance = implementation_provenance(self.package)
        provenance["environment"]["dependencies"]["numpy"] = "next-release"
        self.assertNotEqual(original, self.signature(provenance))
        path = self.package / "cuboids.py"
        path.write_text(path.read_text() + "\n# Geometry implementation changed\n")
        self.assertNotEqual(original, self.signature())
        path = self.package / "pipeline.py"
        text = path.read_text().replace(
            "def prepare(source, output, args, config, previous=None):",
            "def prepare(source, output, args, config, previous=None):\n    new_helper()",
        )
        path.write_text(text + "\ndef new_helper():\n    return 1\n")
        first_helper = self.signature()
        path.write_text(text + "\ndef new_helper():\n    return 2\n")
        self.assertNotEqual(first_helper["preparation"], self.signature()["preparation"])

    def test_export_budgets_quality_targets_and_diagnostics_are_runtime_controls(self):
        original = self.signature()
        for key, value in dict(
            diagnostics=True,
            max_cuboids=500,
            max_memory_mb=8192,
            target_coverage=0.7,
            target_surface_support=0.4,
            target_component_coverage=0.8,
            surface_max_evaluations=131072,
            atlas_size=512,
            color_space="linear",
            source_up="z",
            units_per_meter=1000,
        ).items():
            setattr(self.args, key, value)
        self.assertEqual(original, self.signature())
        self.args.point_tolerance = 0.003
        self.assertNotEqual(original, self.signature())


class ResumeStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.source = cls.root / "source.ply"
        write_cloud(cls.source, cube_shell(9))
        cls.original = cls.root / "original"
        cls.base = [
            "--ply",
            str(cls.source),
            "--resolution",
            "32",
            "--max-frames",
            "2",
            "--seeds-per-frame",
            "2",
            "--max-points",
            "1000",
            "--max-cuboids",
            "2",
            "--atlas-size",
            "256",
            "--outlier-distance-factor",
            "0",
        ]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = pipeline.main(cls.base + ["--output", str(cls.original)])
        if code != 0:
            cls.directory.cleanup()
            raise AssertionError(f"Cube fixture must reach its targets; exit={code}")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def clone(self, name):
        target = self.root / name
        shutil.copytree(self.original, target)
        return target

    def call(self, arguments):
        self.stderr = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(self.stderr):
            return pipeline.main(arguments)

    def test_rebake_never_fits_and_preserves_parameters_across_old_fit_code(self):
        previous = self.clone("old-implementation")
        manifest = json.loads((previous / "checkpoint.json").read_text())
        report = json.loads((previous / "report.json").read_text())
        # A valid completed model may come from an unavailable older fitting
        # implementation. Its source provenance is retained, not impersonated.
        manifest["signature"]["fitting"]["implementation"]["cuboids.py"] = "old-fitting-code"
        manifest["provenance"]["implementation_sha256"]["cuboids.py"] = "old-fitting-code"
        report["implementation_sha256"]["cuboids.py"] = "old-fitting-code"
        report["compatibility"] = manifest["signature"]
        write_json(previous / "checkpoint.json", manifest)
        write_json(previous / "report.json", report)
        output = self.root / "rebaked"
        with (
            patch.object(
                pipeline, "fit_cuboids", side_effect=AssertionError("rebake fitted geometry")
            ),
            patch.object(
                pipeline, "prepare", side_effect=AssertionError("rebake prepared geometry")
            ),
        ):
            code = self.call(
                [
                    "--ply",
                    str(self.source),
                    "--output",
                    str(output),
                    "--rebake",
                    str(previous),
                    "--atlas-size",
                    "512",
                    "--color-space",
                    "linear",
                    "--source-up",
                    "z",
                    "--units-per-meter",
                    "1000",
                ]
            )
        self.assertEqual(code, 0, self.stderr.getvalue())
        self.assertEqual(
            (previous / "cuboids/parameters.npz").read_bytes(),
            (output / "cuboids/parameters.npz").read_bytes(),
        )
        result = json.loads((output / "report.json").read_text())
        self.assertEqual(result["geometry_provenance"], manifest["provenance"])
        self.assertEqual(result["config"]["outlier_distance_factor"], 0)
        self.assertEqual(result["texture"]["atlas_size"], [512, 512])
        self.assertEqual(result["texture"]["source_color_space"], "linear")
        self.assertEqual(result["texture"]["source_up"], "z")
        # The rebaked output itself remains a valid immutable source.
        read_checkpoint(output, None, source_hash=sha256(self.source), load_caches=False)
        with (
            np.load(output / "cuboids/parameters.npz") as before,
            np.load(output / "textured/texture_parameters.npz") as after,
        ):
            for key in ("centers", "dimensions", "rotations", "corners"):
                np.testing.assert_array_equal(before[key], after[key])

    def test_rebake_refuses_artifact_and_input_tampering_before_creating_output(self):
        for index, name in enumerate(
            ("cuboids/parameters.npz", "preparation/common_geometry.npz", "checkpoint")
        ):
            previous = self.clone(f"damaged-{index}")
            if name == "checkpoint":
                name = json.loads((previous / "checkpoint.json").read_text())["state_file"]
            path = previous / name
            path.write_bytes(path.read_bytes() + b"corrupted")
            output = self.root / f"refused-{index}"
            self.assertEqual(
                self.call(
                    [
                        "--ply",
                        str(self.source),
                        "--output",
                        str(output),
                        "--rebake",
                        str(previous),
                        "--atlas-size",
                        "256",
                    ]
                ),
                2,
            )
            self.assertFalse(output.exists())
        changed = self.root / "different.ply"
        changed.write_bytes(self.source.read_bytes() + b"changed")
        output = self.root / "refused-input"
        self.assertEqual(
            self.call(
                [
                    "--ply",
                    str(changed),
                    "--output",
                    str(output),
                    "--rebake",
                    str(self.original),
                ]
            ),
            2,
        )
        self.assertFalse(output.exists())

    def test_rebake_checks_normalization_against_retained_source(self):
        previous = self.clone("wrong-normalization")
        manifest = json.loads((previous / "checkpoint.json").read_text())
        path = previous / manifest["state_file"]
        with np.load(path) as data:
            arrays = dict(data)
        # Preserve world geometry while moving the normalized coordinate origin.
        arrays["normalization_origin"] += 8
        arrays["committed_center"] -= 8 / arrays["normalization_scale"]
        atomic_npz(path, **arrays)
        manifest["state_sha256"] = sha256(path)
        write_json(previous / "checkpoint.json", manifest)
        output = self.root / "refused-normalization"
        self.assertEqual(
            self.call(
                [
                    "--ply",
                    str(self.source),
                    "--output",
                    str(output),
                    "--rebake",
                    str(previous),
                ]
            ),
            2,
        )
        self.assertIn("normalization", self.stderr.getvalue())
        self.assertFalse(output.exists())

    def test_resume_accepts_export_environment_changes_and_runtime_targets(self):
        provenance = implementation_provenance()
        provenance["implementation_sha256"]["texture.py"] = "new-texture-code"
        provenance["implementation_sha256"]["viewer_template.html"] = "new-viewer-code"
        provenance["environment"]["dependencies"]["Pillow"] = "new-pillow"
        output = self.root / "resumed-new-export"
        with patch.object(pipeline, "implementation_provenance", return_value=provenance):
            code = self.call(
                self.base
                + [
                    "--output",
                    str(output),
                    "--resume",
                    str(self.original),
                    "--target-component-coverage",
                    "0.9",
                    "--surface-max-evaluations",
                    "131072",
                    "--diagnostics",
                ]
            )
        self.assertEqual(code, 0, self.stderr.getvalue())
        report = json.loads((output / "report.json").read_text())
        self.assertTrue(report["preparation"]["reused_preparation"])
        self.assertEqual(report["implementation_sha256"]["texture.py"], "new-texture-code")
        changed = copy.deepcopy(provenance)
        changed["implementation_sha256"]["cuboids.py"] = "new-core-code"
        rejected = self.root / "refused-core"
        with patch.object(pipeline, "implementation_provenance", return_value=changed):
            self.assertEqual(
                self.call(
                    self.base
                    + [
                        "--output",
                        str(rejected),
                        "--resume",
                        str(self.original),
                    ]
                ),
                2,
            )
        self.assertFalse(rejected.exists())

    def test_rebake_rejects_changed_retained_population_with_same_bounds(self):
        original = load_points(self.source, 0.1, 0)
        for name in ("count", "indices"):
            altered = dict(original)
            if name == "count":
                altered["points"] = original["points"][1:]
                altered["source_indices"] = original["source_indices"][1:]
            else:
                altered["source_indices"] = original["source_indices"].copy()
                altered["source_indices"][[1, 2]] = altered["source_indices"][[2, 1]]
            output = self.root / f"refused-population-{name}"
            with patch.object(pipeline, "load_points", return_value=altered):
                self.assertEqual(
                    self.call(
                        [
                            "--ply",
                            str(self.source),
                            "--output",
                            str(output),
                            "--rebake",
                            str(self.original),
                        ]
                    ),
                    2,
                )
            self.assertIn("Retained source population", self.stderr.getvalue())
            self.assertFalse(output.exists())

    def test_completed_geometry_can_resume_after_rebake(self):
        rendered, resumed = self.root / "render-for-resume", self.root / "resume-after-render"
        self.assertEqual(
            self.call(
                self.base
                + [
                    "--output",
                    str(rendered),
                    "--rebake",
                    str(self.original),
                    "--color-space",
                    "linear",
                ]
            ),
            0,
            self.stderr.getvalue(),
        )
        with patch.object(
            pipeline, "local_geometry", side_effect=AssertionError("cache not reused")
        ):
            self.assertEqual(
                self.call(
                    self.base
                    + [
                        "--output",
                        str(resumed),
                        "--resume",
                        str(rendered),
                    ]
                ),
                0,
                self.stderr.getvalue(),
            )
        report = json.loads((resumed / "report.json").read_text())
        self.assertEqual(report["geometry"]["selection"]["frozen_prefix_count"], 1)
        with (
            np.load(self.original / "cuboids/parameters.npz") as before,
            np.load(resumed / "cuboids/parameters.npz") as after,
        ):
            for key in ("centers", "dimensions", "rotations", "corners"):
                np.testing.assert_array_equal(before[key], after[key])

    def test_cached_candidate_resume_rebuilds_support_integral_once(self):
        manifest = json.loads((self.original / "checkpoint.json").read_text())
        resumed, _, _ = read_checkpoint(self.original, manifest["signature"])
        self.assertNotIn("safe_prefix", resumed["cached_solid"])
        loaded = load_points(self.source, 0.1, 0)
        with np.load(self.original / "preparation/regions.npz") as data:
            regions = dict(data)
        report = json.loads((self.original / "preparation/report.json").read_text())
        args = pipeline.parser().parse_args(self.base + ["--output", str(self.root / "unused")])
        observed = []
        with (
            patch(
                "cuboid_approximation.fitting.generate_candidates",
                side_effect=AssertionError("cached candidates regenerated"),
            ),
            patch(
                "cuboid_approximation.candidate_grids.summed_volume", wraps=summed_volume
            ) as summed,
        ):
            result = fit_cuboids(
                loaded["points"],
                regions,
                report["median_spacing"],
                pipeline.fitting_config(args),
                full_spacing=loaded["spacing"],
                progress=lambda _: None,
                checkpoint=lambda _boxes, _pool, solid, *_: observed.append(solid["safe_prefix"]),
                **resumed,
            )
        summed.assert_called_once()
        self.assertTrue(observed)
        self.assertTrue(all(prefix is result["solid"]["safe_prefix"] for prefix in observed))


if __name__ == "__main__":
    unittest.main()
