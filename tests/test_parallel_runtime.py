"""Worker budgets must be bounded, reversible and independent of cached geometry."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_info

from cuboid_approximation import parallel, pipeline
from cuboid_approximation.provenance import compatibility_signature, implementation_provenance


class ParallelRuntimeTests(unittest.TestCase):
    def test_auto_explicit_and_serial_counts_respect_affinity(self):
        with patch.object(parallel, "available_cpus", return_value=6):
            self.assertEqual(parallel.worker_count(), 4)
            self.assertEqual(parallel.worker_count(1), 1)
            self.assertEqual(parallel.worker_count(6), 6)
            self.assertEqual(parallel.worker_count(100), 6)
        with patch.object(parallel, "available_cpus", return_value=1):
            self.assertEqual(parallel.worker_count(), 1)
        with self.assertRaises(ValueError):
            parallel.worker_count(-1)
        for invalid in (1.5, "4", None):
            with self.assertRaises(ValueError):
                parallel.worker_count(invalid)

    def test_affinity_failure_and_missing_cpu_count_fall_back_to_one(self):
        with (
            patch.object(parallel.os, "sched_getaffinity", side_effect=OSError, create=True),
            patch.object(parallel.os, "cpu_count", return_value=None),
        ):
            self.assertEqual(parallel.available_cpus(), 1)

    def test_nested_failure_restores_context_and_native_thread_limits(self):
        np.eye(3) @ np.eye(3)  # Ensure the native BLAS library is loaded.
        before = threadpool_info()
        with patch.object(parallel, "available_cpus", return_value=8):
            with parallel.execution_workers(4):
                self.assertEqual(parallel.current_workers(), 4)
                for library in threadpool_info():
                    if library["user_api"] == "blas":
                        self.assertEqual(library["num_threads"], 1)
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    with parallel.execution_workers(2):
                        self.assertEqual(parallel.current_workers(), 2)
                        raise RuntimeError("failed")
                self.assertEqual(parallel.current_workers(), 4)
        self.assertEqual(parallel.current_workers(), 1)
        self.assertEqual(threadpool_info(), before)

    def test_pipeline_scopes_budget_and_rejects_negative_before_creating_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "result"
            args = pipeline.parser().parse_args([
                "--ply", "unused.ply", "--output", str(output), "--workers", "-1",
            ])
            with self.assertRaises(ValueError):
                pipeline.run(args)
            self.assertFalse(output.exists())
            args.workers = 1
            with patch.object(pipeline, "_run", side_effect=lambda _: parallel.current_workers()):
                self.assertEqual(pipeline.run(args), 1)

    def test_worker_count_does_not_invalidate_resume_geometry(self):
        args = pipeline.parser().parse_args(["--ply", "unused.ply", "--output", "unused"])
        provenance = implementation_provenance()
        before = compatibility_signature(args, "input", provenance)
        args.workers = 6
        self.assertEqual(compatibility_signature(args, "input", provenance), before)


if __name__ == "__main__":
    unittest.main()
