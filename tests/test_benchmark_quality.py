"""Quality corpus contracts must fail CI when a partial export misses its target."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image


path = Path(__file__).resolve().parents[1] / "tools" / "benchmark.py"
spec = importlib.util.spec_from_file_location("quality_benchmark", path)
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)


class BenchmarkQualityTests(unittest.TestCase):
    def test_quality_targets_cannot_pass_with_a_partial_export(self):
        case = next(benchmark.corpus())
        report = dict(
            target_reached=False,
            geometry=dict(cuboids=1),
            texture=dict(supported_area_fraction=1.0),
        )
        failures = benchmark.quality_failures(case, 3, report, {})
        self.assertTrue(any("exit 3" in failure for failure in failures))
        self.assertTrue(any("target_reached" in failure for failure in failures))

    def test_strict_default_fails_and_exploratory_explicitly_records_failures(self):
        def partial_pipeline(command):
            output = Path(command[command.index("--output") + 1])
            output.mkdir()
            report = dict(
                target_reached=False,
                config={},
                environment={},
                implementation_sha256="test",
                geometry=dict(cuboids=0),
            )
            (output / "report.json").write_text(json.dumps(report))
            return 3

        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(benchmark, "run_pipeline", side_effect=partial_pipeline) as run,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            for name, extra, expected in (
                ("strict", ["--resolution", "32"], 1),
                ("explore", ["--exploratory", "--resolution", "160"], 0),
            ):
                output = Path(folder) / name
                code = benchmark.main(["--output", str(output), "--case", "cube", *extra])
                self.assertEqual(code, expected)
                summary = json.loads((output / "summary.json").read_text())
                self.assertFalse(summary[0]["quality_passed"])
                self.assertTrue(summary[0]["failures"])
            commands = [call.args[0] for call in run.call_args_list]
            self.assertTrue(all("--point-tolerance" in command for command in commands))
            self.assertEqual(
                [command[command.index("--point-tolerance") + 1] for command in commands],
                ["0.06", "0.06"],
            )

    def test_analytic_union_reference_has_correct_exposed_area(self):
        rectangles = benchmark.union_rectangles(
            [
                [[0, 0, 0], [1, 2, 1]],
                [[1, 0, 0], [2, 1, 1]],
            ]
        )
        area = sum(np.prod((hi - lo)[hi > lo]) for lo, hi in rectangles)
        self.assertAlmostEqual(area, 14.0)
        points = benchmark.sample_rectangles(rectangles, 0.1)
        # The shared contact face is internal; it must not appear in the reference.
        self.assertFalse(np.any(np.all(np.isclose(points, [1, 0.5, 0.5]), axis=1)))

    def test_corpus_exercises_symmetry_density_details_and_separated_layers(self):
        cases = {case.name: case for case in benchmark.corpus()}
        self.assertIn("rotated_cube", cases)
        self.assertIn("noisy_cube", cases)
        self.assertIn("extreme_density", cases)
        self.assertIn("partial_corner", cases)
        self.assertGreater(len(cases["thin_appendage"].detail_probes), 0)
        self.assertLess(cases["close_layers"].tolerance, 0.08 / 2)
        np.testing.assert_array_equal(
            np.unique(cases["close_layers"].colors, axis=0), [[0, 0, 1], [1, 0, 0]]
        )

    def test_independent_checks_detect_atlas_color_swap_and_filled_gap(self):
        case = next(case for case in benchmark.corpus() if case.name == "close_layers")
        centers = np.array([[0.5, 0.5, 0], [0.5, 0.5, 0.08]])
        dimensions = np.tile([1, 1, 0.002], (2, 1))
        signs = np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ]
        )
        corners = centers[:, None] + signs[None] * dimensions[:, None] / 2
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "cuboids").mkdir()
            (output / "textured").mkdir()
            np.savez(
                output / "cuboids" / "parameters.npz",
                centers=centers,
                dimensions=dimensions,
                rotations=np.tile(np.eye(3), (2, 1, 1)),
                corners=corners,
            )
            faces = [
                dict(
                    box=i + 1,
                    hidden_in_all_prefixes=False,
                    corner_indices=[0, 1, 2, 3],
                    rect=[8 * i, 0, 8, 8],
                )
                for i in range(2)
            ]
            (output / "textured" / "texture_report.json").write_text(
                json.dumps(dict(face_details=faces))
            )
            atlas = np.zeros((8, 16, 3), dtype=np.uint8)
            atlas[:, :8, 0], atlas[:, 8:, 2] = 255, 255
            Image.fromarray(atlas).save(output / "textured" / "texture_atlas.png")
            metrics = benchmark.analytic_checks(case, output)
            self.assertEqual(metrics["occupied_empty_probes"], 0)
            self.assertEqual(metrics["layer_color_p95"], [0.0, 0.0])
            Image.fromarray(atlas[:, ::-1]).save(output / "textured" / "texture_atlas.png")
            swapped = benchmark.analytic_checks(case, output)
            self.assertGreater(min(swapped["layer_color_p95"]), 1.0)
            report = dict(
                target_reached=True,
                geometry=dict(cuboids=2),
                texture=dict(supported_area_fraction=1),
            )
            self.assertTrue(
                any(
                    "red/blue" in value
                    for value in benchmark.quality_failures(case, 0, report, swapped)
                )
            )
            _, occupied = benchmark.box_distances(
                case.empty_probes,
                dict(
                    centers=np.array([[0.5, 0.5, 0.04]]),
                    dimensions=np.array([[1, 1, 0.1]]),
                    rotations=np.eye(3)[None],
                ),
            )
            self.assertTrue(occupied.all())


if __name__ == "__main__":
    unittest.main()
