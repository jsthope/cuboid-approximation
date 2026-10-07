"""Independent geometric references for tolerance and area-support decisions."""

from dataclasses import replace
import unittest

import numpy as np

from cuboid_approximation.approximation import (
    component_coverage,
    component_target_reached,
    evaluation_tolerance,
    spatial_components,
    spatial_weights,
    surface_metrics,
    surface_support_reached,
    surface_support_status,
)
from cuboid_approximation.cuboids import CuboidConfig
from cuboid_approximation.geometry import SurfaceMesh, exposed_triangles


class EvaluationToleranceTests(unittest.TestCase):
    def test_default_physical_tolerance_is_fixed_across_grid_resolutions(self):
        tolerances = [
            evaluation_tolerance(CuboidConfig(resolution=resolution), 0.003, 12.0)
            for resolution in (32, 160, 768)
        ]
        np.testing.assert_array_equal(tolerances, [0.006] * 3)

    def test_explicit_physical_tolerance_survives_spacing_and_resolution_changes(self):
        config = CuboidConfig(point_tolerance=0.015)
        for resolution, spacing in ((32, 0.001), (768, 0.1)):
            self.assertEqual(
                evaluation_tolerance(replace(config, resolution=resolution), spacing, 8),
                0.015,
            )

    def test_numerical_spacing_floor_scales_with_geometry(self):
        config = CuboidConfig()
        tiny = evaluation_tolerance(config, 0, 1)
        self.assertGreater(tiny, 0)
        self.assertLess(tiny, 1e-12)
        self.assertEqual(evaluation_tolerance(config, 0, 100), 100 * tiny)

    def test_fitting_reports_area_and_surface_thickness_in_source_units(self):
        from cuboid_approximation.fitting import fit_cuboids

        x, y = np.meshgrid(np.linspace(0, 10, 17), np.linspace(0, 10, 17))
        points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))) + [100, 200, 300]
        regions = dict(points=points, labels=np.zeros(len(points), dtype=int),
                       assignment_kind=np.ones(len(points), dtype=int))
        config = CuboidConfig(resolution=32, max_frames=1, seeds_per_frame=2,
                              max_cuboids=1, reconstruction_mode="surface", point_tolerance=0.7)
        result = fit_cuboids(points, regions, 0.625, config, progress=lambda _: None)
        self.assertEqual(len(result["boxes"]), 1)
        self.assertAlmostEqual(result["surface_tolerance"], 0.7)
        self.assertAlmostEqual(result["selection"]["surface_panel_thickness_limit"], 1.4)
        dimensions = result["boxes"][0]["dimensions"]
        expected_area = 2 * sum(np.prod(np.delete(dimensions, axis)) for axis in range(3))
        self.assertAlmostEqual(
            result["selection"]["surface_support_bounds"]["total_area"], expected_area
        )


class SurfaceSupportBoundTests(unittest.TestCase):
    def setUp(self):
        # Triangle area is two. A radius-1/2 ball around the origin cuts out
        # exactly a quarter-disc of area pi/16, hence fraction pi/32.
        self.triangles = np.array([[[0.0, 0, 0], [2, 0, 0], [0, 2, 0]]])
        self.mesh = SurfaceMesh(self.triangles)
        self.source = np.zeros((1, 3))
        self.fraction = np.pi / 32

    def test_adaptive_bounds_enclose_independently_known_disc_area(self):
        report = self.mesh.support_bounds(
            self.source, 0.5, area_fraction_tolerance=0.001, max_evaluations=10000
        )
        self.assertLess(report["lower_fraction"], self.fraction)
        self.assertGreater(report["upper_fraction"], self.fraction)
        self.assertLessEqual(report["ambiguous_area_fraction"], 0.001)
        self.assertLess(report["evaluated_triangles"], 10000)

    def test_decisive_targets_stop_without_exhausting_budget(self):
        for target, status in ((0.05, "reached"), (0.15, "not_reached")):
            report = self.mesh.support_bounds(self.source, 0.5, target=target)
            self.assertEqual(report["status"], status)
            self.assertEqual(report["stopping_reason"], "target_decided")
            self.assertLess(report["evaluated_triangles"], 65536)

    def test_near_threshold_remains_inconclusive_and_deterministic(self):
        first = self.mesh.support_bounds(
            self.source, 0.5, target=self.fraction, max_evaluations=1000
        )
        second = self.mesh.support_bounds(
            self.source, 0.5, target=self.fraction, max_evaluations=1000
        )
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "inconclusive")
        self.assertEqual(first["evaluated_triangles"], 1000)
        self.assertLess(first["lower_fraction"], self.fraction)
        self.assertGreater(first["upper_fraction"], self.fraction)

    def test_unvisited_area_remains_unknown_when_work_or_depth_expires(self):
        for kwargs, reason in ((dict(max_evaluations=0), "work_budget"),
                               (dict(max_depth=0), "depth_budget")):
            report = self.mesh.support_bounds(self.source, 0.5, target=0.5, **kwargs)
            self.assertEqual(report["lower_fraction"], 0)
            self.assertEqual(report["upper_fraction"], 1)
            self.assertEqual(report["status"], "inconclusive")
            self.assertEqual(report["stopping_reason"], reason)

    def test_fully_supported_and_empty_source_are_exact(self):
        supported = self.mesh.support_bounds(self.source, 4, target=1)
        unsupported = self.mesh.support_bounds(np.empty((0, 3)), 4, target=0.1)
        self.assertEqual(supported["lower_fraction"], 1)
        self.assertEqual(supported["status"], "reached")
        self.assertEqual(unsupported["upper_fraction"], 0)
        self.assertEqual(unsupported["status"], "not_reached")

    def test_full_support_target_one_survives_area_summation_roundoff(self):
        box = dict(center=np.zeros(3), dimensions=np.array([1.1, 1.7, 2.3]),
                   rotation=np.eye(3), volume=1.1 * 1.7 * 2.3)
        report = SurfaceMesh(exposed_triangles([box])).support_bounds(
            self.source, 10, target=1
        )
        self.assertEqual(report["lower_fraction"], 1)
        self.assertEqual(report["upper_fraction"], 1)
        self.assertEqual(report["status"], "reached")

    def test_bounds_hold_after_geometry_scaling_and_translation(self):
        for scale in (1e-6, 1e5):
            offset = np.array([3, -7, 9]) * scale
            mesh = SurfaceMesh(self.triangles * scale + offset)
            report = mesh.support_bounds(
                self.source * scale + offset, 0.5 * scale, max_evaluations=2000
            )
            self.assertLess(report["lower_fraction"], self.fraction)
            self.assertGreater(report["upper_fraction"], self.fraction)

    def test_quadrature_estimate_cannot_certify_a_straddled_target(self):
        report = dict(
            surface_supported_area_fraction=0.98,
            surface_supported_area_fraction_lower=0.94,
            surface_supported_area_fraction_upper=0.96,
        )
        self.assertFalse(surface_support_reached(report, 0.95))
        self.assertEqual(surface_support_status(report, 0.95), "inconclusive")
        self.assertTrue(surface_support_reached(report, 0.94))
        self.assertEqual(surface_support_status(report, 0.97), "not_reached")
        self.assertFalse(surface_support_reached({"surface_supported_area_fraction": 1}, 0.95))

    def test_quadrature_underestimate_cannot_veto_certified_completion(self):
        from cuboid_approximation.selection import UnionSurfaceObjective

        # Disjoint quarter-discs on the six faces have exact supported fraction
        # pi * .35**2 / 4 = .096211..., but quadrature estimates only .095029...
        source = np.array([[0.0, 0, 0], [1, 1, 1]])
        config = CuboidConfig(target_surface_support=0.0951)
        objective = UnionSurfaceObjective(
            source, np.ones(2), 0.35, config, np.arange(2), np.ones(2)
        )
        boxes = [dict(center=np.full(3, 0.5), dimensions=np.ones(3),
                      rotation=np.eye(3), volume=1.0)]
        objective.install(boxes, objective.evaluate(boxes))
        self.assertLess(objective.reverse_coverage, config.target_surface_support)
        self.assertTrue(objective.complete())
        self.assertGreaterEqual(
            objective.support_bounds["lower_fraction"], config.target_surface_support
        )
        self.assertEqual(objective.support_bounds["status"], "reached")

    def test_surface_report_preserves_exact_source_distances_and_reverse_estimate(self):
        box = dict(center=np.zeros(3), dimensions=np.ones(3) * 2,
                   rotation=np.eye(3), volume=8.0)
        source = np.array([[1.0, 0, 0], [-1, 0, 0]])
        report = surface_metrics([box], source, 0.5, target_support=0.95)
        self.assertEqual(report["source_to_surface_rms"], 0)
        self.assertEqual(report["source_surface_coverage"], 1)
        self.assertIn("surface_supported_area_fraction", report)
        self.assertEqual(report["surface_support_status"], "not_reached")
        self.assertEqual(report["component_surface_coverage"], [1.0, 1.0])


class ComponentEvaluationTests(unittest.TestCase):
    def test_normalization_cannot_change_density_weights_or_split_a_component(self):
        # These decimal cell boundaries previously floored differently after
        # normalization: coverage jumped from .90 to .944 and one component
        # became two, despite identical physical points and tolerance.
        points = np.vstack((np.column_stack((np.arange(10) * 0.3, np.zeros((10, 2)))),
                            [[1.0, 0, 0]]))
        tolerance = 0.15
        scale = np.ptp(points, axis=0).max()
        normalized = (points - points.min(axis=0)) / scale
        weights = spatial_weights(points, tolerance)
        local_weights = spatial_weights(normalized, tolerance / scale)
        labels = spatial_components(points, tolerance)
        local_labels = spatial_components(normalized, tolerance / scale)
        np.testing.assert_array_equal(weights, local_weights)
        np.testing.assert_array_equal(labels, local_labels)
        self.assertEqual(len(np.unique(labels)), 1)
        covered = np.arange(len(points)) != 2
        self.assertEqual(
            np.average(covered, weights=weights), np.average(covered, weights=local_weights)
        )

    def test_small_uncovered_component_can_fail_an_optional_target(self):
        points = np.vstack((np.repeat([[0.0, 0, 0]], 1000, axis=0), [[10.0, 0, 0]]))
        covered = np.arange(len(points)) < 1000
        labels = spatial_components(points, 0.1)
        coverage = component_coverage(covered, labels, spatial_weights(points, 0.1))
        np.testing.assert_array_equal(coverage, [1, 0])
        self.assertFalse(component_target_reached(coverage, 0.99))
        self.assertTrue(component_target_reached(coverage, 0))
        self.assertTrue(component_target_reached(coverage, None))
        self.assertFalse(component_target_reached([], 0.99))


if __name__ == "__main__":
    unittest.main()
