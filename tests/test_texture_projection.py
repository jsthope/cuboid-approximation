"""Regression checks for complete splat footprints and compatible gap donors."""

import unittest

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from cuboid_approximation.texture import bake_face


def face():
    return dict(
        origin=np.zeros(3),
        u=np.array([1.0, 0, 0]),
        v=np.array([0.0, 1, 0]),
        normal=np.array([0.0, 0, 1]),
        width=1.0,
        height=1.0,
    )


def cloud(points, spacing=0.1, covariance=None, colors=None, local_spacing=None):
    points = np.asarray(points, dtype=float)
    n = len(points)
    result = dict(
        points=points,
        colors=np.tile([1.0, 0, 0], (n, 1)) if colors is None else np.asarray(colors),
        normals=np.tile([0.0, 0, 1], (n, 1)),
        opacity=np.ones(n),
        covariance=(
            np.tile(np.eye(3) * 0.001**2, (n, 1, 1))
            if covariance is None
            else np.broadcast_to(covariance, (n, 3, 3)).copy()
        ),
        spacing=spacing,
        tree=cKDTree(points),
    )
    if local_spacing is not None:
        result["spacing_per_point"] = np.asarray(local_spacing, dtype=float)
    return result


class TextureProjectionTests(unittest.TestCase):
    def test_full_anisotropic_footprint_is_projected(self):
        source = cloud([[0.05, 0.5, 0]], covariance=np.diag([0.2**2, 0.035**2, 0.035**2]))
        rgb, supported, depth = bake_face(face(), 1, 1, source, 0.1, 0.1)
        np.testing.assert_allclose(rgb[0, 0], [1, 0, 0])
        self.assertTrue(supported[0, 0])
        self.assertEqual(depth[0], 0)

    def test_broad_phase_includes_kernels_centered_outside_spacing_margin(self):
        target = face()
        target.update(width=0.1, height=0.1)
        source = cloud([[-0.45, 0.05, 0]], covariance=np.diag([0.2**2, 0.035**2, 0.035**2]))
        rgb, supported, _ = bake_face(target, 1, 1, source, 0.1, 0.1)
        np.testing.assert_allclose(rgb[0, 0], [1, 0, 0])
        self.assertTrue(supported[0, 0])

    def test_rotated_covariance_and_rigid_transforms_preserve_ellipse(self):
        angle = np.pi / 4
        major = np.array([np.cos(angle), np.sin(angle), 0])
        minor = np.array([-np.sin(angle), np.cos(angle), 0])
        axes = np.column_stack((major, minor, [0, 0, 1]))
        covariance = axes @ np.diag([0.2**2, 0.035**2, 0.035**2]) @ axes.T
        rotation = Rotation.from_euler("xyz", [23, 51, -17], degrees=True).as_matrix()
        translation = np.array([9.0, -4, 12])
        for direction, projected in ((major, True), (minor, False)):
            source = cloud([np.array([0.5, 0.5, 0]) - 0.45 * direction], covariance=covariance)
            target = face()
            expected, expected_support, _ = bake_face(target, 1, 1, source, 0.1, 0.1)
            np.testing.assert_allclose(expected[0, 0], [1, 0, 0] if projected else [0.65] * 3)
            self.assertEqual(bool(expected_support[0, 0]), projected)
            for key in ("u", "v", "normal"):
                target[key] = rotation @ target[key]
            target["origin"] = translation
            source["points"] = source["points"] @ rotation.T + translation
            source["normals"] = source["normals"] @ rotation.T
            source["covariance"] = rotation @ source["covariance"] @ rotation.T
            source["tree"] = cKDTree(source["points"])
            actual, actual_support, _ = bake_face(target, 1, 1, source, 0.1, 0.1)
            np.testing.assert_allclose(actual, expected, atol=1e-12)
            np.testing.assert_array_equal(actual_support, expected_support)

    def test_fallback_cannot_reintroduce_ineligible_sources(self):
        for depth in (-0.3, 0.3):
            source = cloud([[0.5, 0.5, depth]])
            rgb, supported, projected_depth = bake_face(face(), 1, 1, source, 0.02, 0.02)
            np.testing.assert_allclose(rgb, 0.65)
            self.assertFalse(supported.any())
            self.assertTrue(np.isnan(projected_depth).all())
        for field, value in (
            ("normals", np.array([[1.0, 0, 0]])),
            ("opacity", np.array([0.0])),
            ("normal_confidence", np.array([0.0])),
        ):
            source = cloud([[0.25, 0.5, 0.01]])
            source[field] = value
            rgb, supported, _ = bake_face(face(), 1, 1, source, 0.02, 0.02)
            np.testing.assert_allclose(rgb, 0.65)
            self.assertFalse(supported.any())

    def test_local_spacing_controls_projection_and_fallback_ranges(self):
        for offset, projected in ((0.2, True), (0.65, False)):
            source = cloud([[0.5 - offset, 0.5, 0]], spacing=0.005, local_spacing=[0.2])
            rgb, supported, projected_depth = bake_face(face(), 1, 1, source, 0.02, 0.02)
            np.testing.assert_allclose(rgb[0, 0], [1, 0, 0])
            self.assertEqual(bool(supported[0, 0]), projected)
            self.assertEqual(bool(np.isfinite(projected_depth[0])), projected)
            del source["spacing_per_point"]
            rgb, supported, _ = bake_face(face(), 1, 1, source, 0.02, 0.02)
            np.testing.assert_allclose(rgb, 0.65)
            self.assertFalse(supported.any())

    def test_layer_separation_uses_finer_donor_scale(self):
        for local_scales in ([0.2, 0.005], [0.005, 0.2]):
            source = cloud(
                [[0.5, 0.5, 0.01], [0.5, 0.5, 0.1]],
                spacing=0.2,
                covariance=np.eye(3) * 0.1**2,
                colors=[[1.0, 0, 0], [0, 0, 1.0]],
                local_spacing=local_scales,
            )
            rgb, supported, depth = bake_face(face(), 1, 1, source, 0.2, 0.2)
            np.testing.assert_allclose(rgb[0, 0], [1, 0, 0])
            self.assertTrue(supported[0, 0])
            self.assertAlmostEqual(depth[0], 0.01)

    def test_sparse_near_interpolation_donor_wins_over_dense_far_layer(self):
        red = [[0.15, 0.5, 0.01]]
        blue = np.column_stack(
            (0.3 + np.linspace(-0.001, 0.001, 64), np.full(64, 0.5), np.full(64, 0.1))
        )
        source = cloud(
            np.vstack((red, blue)),
            spacing=0.06,
            colors=np.vstack(([1.0, 0, 0], np.tile([0, 0, 1.0], (64, 1)))),
            local_spacing=np.r_[0.1, np.full(64, 0.06)],
        )
        rgb, supported, depth = bake_face(face(), 1, 1, source, 0.2, 0.2)
        np.testing.assert_allclose(rgb[0, 0], [1, 0, 0])
        self.assertFalse(supported[0, 0])
        self.assertTrue(np.isnan(depth[0]))

    def test_near_layer_survives_source_order_and_source_batches(self):
        points = np.vstack((np.tile([0.5, 0.5, 0.1], (130, 1)), [0.6, 0.5, 0.01]))
        colors = np.vstack((np.tile([0, 0, 1.0], (130, 1)), [1.0, 0, 0]))
        for permutation in (np.arange(131), np.arange(130, -1, -1)):
            source = cloud(
                points[permutation],
                spacing=0.03,
                covariance=np.eye(3) * 0.05**2,
                colors=colors[permutation],
            )
            rgb, supported, depth = bake_face(face(), 1, 1, source, 0.2, 0.2)
            np.testing.assert_allclose(rgb[0, 0], [1, 0, 0])
            self.assertTrue(supported[0, 0])
            self.assertAlmostEqual(depth[0], 0.01)

    def test_large_kernel_crosses_raster_batches_without_seams(self):
        source = cloud([[0.5, 0.5, 0.01]], covariance=np.eye(3) * 2**2)
        rgb, supported, depth = bake_face(face(), 257, 257, source, 0.2, 0.2)
        np.testing.assert_allclose(rgb, np.broadcast_to([1, 0, 0], rgb.shape), atol=1e-12)
        self.assertTrue(supported.all())
        np.testing.assert_allclose(depth, 0.01)


if __name__ == "__main__":
    unittest.main()
