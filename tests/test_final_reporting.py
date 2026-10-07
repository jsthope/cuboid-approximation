"""Final reports preserve hard constraints already rejected by selection."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from cuboid_approximation.cuboids import CuboidConfig, select_incremental_boxes
from cuboid_approximation.pipeline import save_geometry
from test_quality import shell
from test_surface_selection import box, solid


class FinalReportingTests(unittest.TestCase):
    def test_thick_frozen_surface_prefix_cannot_pass_on_distance_metrics_alone(self):
        points = shell() - 0.5
        prefix = box([0, 0, 0], [1, 1, 1])
        envelope = solid()
        envelope.update(conservative=envelope["safe"], reconstruction={})
        config = CuboidConfig(reconstruction_mode="surface", point_tolerance=0.1)
        boxes, curve, _, count, selection = select_incremental_boxes(
            [], [prefix], points, 0.1, config, envelope, lambda _: None
        )
        self.assertTrue(selection["frozen_prefix_violates_surface_thickness"])
        result = dict(
            boxes=boxes, solid=envelope, full_distances=np.zeros(len(points)),
            coverage_curve=curve, surface_tolerance=0.1, frames=np.eye(3)[None],
            candidate_count=count, selection=selection, seconds=0,
        )
        with tempfile.TemporaryDirectory() as folder:
            report = save_geometry(
                Path(folder), result, dict(points=points), dict(points=points), config
            )
        self.assertTrue(report["volume_target_reached"])
        self.assertEqual(report["approximation"]["surface"]["source_surface_coverage"], 1)
        self.assertEqual(report["approximation"]["surface"]["surface_support_status"], "reached")
        self.assertFalse(report["surface_target_reached"])
        self.assertFalse(report["target_reached"])
        self.assertEqual(report["selection"]["stopping_reason"], "irreparable_frozen_prefix")
        np.testing.assert_array_equal(boxes[0]["dimensions"], prefix["dimensions"])


if __name__ == "__main__":
    unittest.main()
