"""Regression checks for complete splat footprints and compatible gap donors."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from PIL import Image

from cuboid_approximation.geometry import CORNER_SIGNS
from cuboid_approximation.texture import bake_face, bake_model, linear_to_srgb


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
    def test_translucent_front_layer_does_not_hide_opaque_back_layer(self):
        source = cloud([[0.5, 0.5, 0.005], [0.5, 0.5, 0.03]], spacing=0.02,
                       colors=[[0, 0, 0], [1, 1, 1]])
        source["opacity"] = np.array([0.1, 1.0])
        rgb, supported, depth = bake_face(face(), 1, 1, source, 0.1, 0.1)
        np.testing.assert_allclose(rgb, linear_to_srgb(0.9))
        self.assertTrue(supported.all())
        self.assertAlmostEqual(depth[0], 0.005)
        source["opacity"][0] = 1
        opaque, _, _ = bake_face(face(), 1, 1, source, 0.1, 0.1)
        np.testing.assert_allclose(opaque, 0)

    def test_layer_compositing_is_invariant_to_duplicates_order_and_tiles(self):
        for duplicates in (1, 130):
            points = np.vstack(([[0.5, 0.5, 0.005]],
                                np.tile([0.5, 0.5, 0.03], (duplicates, 1)),
                                [[0.5, 0.5, 0.06]]))
            colors = np.vstack(([1, 0, 0], np.tile([0, 1, 0], (duplicates, 1)), [0, 0, 1]))
            opacity = np.r_[0.1, np.full(duplicates, 0.5), 1.0]
            for order in (np.arange(len(points)), np.arange(len(points))[::-1]):
                source = cloud(points[order], spacing=0.02, colors=colors[order],
                               covariance=np.eye(3) * 2**2)
                source["opacity"] = opacity[order]
                rgb, supported, depth = bake_face(face(), 23, 29, source, 0.1, 0.1)
                expected = np.broadcast_to(linear_to_srgb([0.1, 0.45, 0.45]), rgb.shape)
                np.testing.assert_allclose(rgb, expected, atol=1e-12)
                self.assertTrue(supported.all())
                np.testing.assert_allclose(depth, 0.005)

    def test_single_translucent_layer_is_not_darkened_by_an_invented_background(self):
        source = cloud([[0.5, 0.5, 0.005]], spacing=0.02, colors=[[1, 1, 1]])
        source["opacity"] = np.array([0.1])
        rgb, _, _ = bake_face(face(), 1, 1, source, 0.1, 0.1)
        np.testing.assert_allclose(rgb, 1)

    def test_single_texel_keeps_more_than_one_raster_batch_of_donors(self):
        source = cloud(np.vstack((np.tile([0.5, 0.5, 0.005], (65540, 1)),
                                  [0.5, 0.5, 0.03])), spacing=0.02,
                       colors=np.vstack((np.zeros((65540, 3)), [1, 1, 1])))
        source["opacity"] = np.r_[np.full(65540, 0.1), 1.0]
        rgb, supported, _ = bake_face(face(), 1, 1, source, 0.1, 0.1)
        np.testing.assert_allclose(rgb, linear_to_srgb(0.9))
        self.assertTrue(supported.all())

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

    def test_accepted_normal_offset_does_not_shrink_tangent_gap_radius(self):
        rotation = Rotation.from_euler("xyz", [23, 51, -17], degrees=True).as_matrix()
        for transform in (np.eye(3), rotation):
            for offset in (0.0, -0.04, 0.04):
                target = face()
                target.update(width=0.1, height=0.05)
                for key in ("u", "v", "normal"):
                    target[key] = transform @ target[key]
                source = cloud(np.array([[0.003, 0.025, offset]]) @ transform.T,
                               spacing=0.006)
                source["normals"] = source["normals"] @ transform.T
                rgb, supported, depth = bake_face(target, 2, 1, source, 0.05, 0.05)
                # A nearby gap at the left edge fills; the distant texel stays gray.
                np.testing.assert_allclose(rgb[0], [[1, 0, 0], [0.65, 0.65, 0.65]],
                                           atol=1e-12)
                self.assertFalse(supported.any())
                self.assertTrue(np.isnan(depth).all())

    def test_offset_surface_fills_sampling_gaps_at_all_edges_and_corners(self):
        x, y = np.meshgrid(np.linspace(0.025, 0.975, 20), np.linspace(0.025, 0.975, 20))
        source = cloud(np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 0.04))),
                       spacing=0.01)
        rgb, supported, depth = bake_face(face(), 40, 40, source, 0.05, 0.05)
        np.testing.assert_allclose(rgb, np.broadcast_to([1, 0, 0], rgb.shape), atol=1e-12)
        self.assertFalse(supported.all())
        self.assertTrue(np.isnan(depth).any())

    def test_adjacent_plane_colors_only_the_nearby_edge_after_rotation(self):
        rotation = Rotation.from_euler("xyz", [23, 51, -17], degrees=True).as_matrix()
        expected = None
        for transform in (np.eye(3), rotation):
            target = face()
            for key in ("u", "v", "normal"):
                target[key] = transform @ target[key]
            source = cloud(np.array([[0, 0.5, 0.005]]) @ transform.T,
                           spacing=0.001, local_spacing=[0.03])
            source["normals"] = np.array([[1., 0, 0]]) @ transform.T
            rgb, supported, depth = bake_face(target, 100, 1, source, 0.02, 0.02)
            np.testing.assert_allclose(rgb[0, :12], np.tile([1, 0, 0], (12, 1)), atol=1e-12)
            np.testing.assert_allclose(rgb[0, 12:], 0.65, atol=1e-12)
            self.assertFalse(supported.any())
            self.assertTrue(np.isnan(depth).all())
            if expected is not None:
                np.testing.assert_allclose(rgb, expected, atol=1e-12)
            expected = rgb

    def test_edge_fallback_rejects_planes_that_do_not_meet_the_edge(self):
        source = cloud([[0.15, 0.5, 0.005]], spacing=0.04)
        source["normals"] = np.array([[1., 0, 0]])
        rgb, supported, depth = bake_face(face(), 100, 1, source, 0.02, 0.02)
        np.testing.assert_allclose(rgb, 0.65)
        self.assertFalse(supported.any())
        self.assertTrue(np.isnan(depth).all())

    def test_edge_fallback_preserves_depth_opacity_and_confidence_rejections(self):
        for field, value in (
            ("points", np.array([[0, 0.5, 0.03]])),
            ("points", np.array([[0, 0.5, -0.03]])),
            ("opacity", np.zeros(1)),
            ("normal_confidence", np.zeros(1)),
        ):
            source = cloud([[0, 0.5, 0.005]], spacing=0.03)
            source["normals"] = np.array([[1., 0, 0]])
            source[field] = value
            source["tree"] = cKDTree(source["points"])
            rgb, supported, depth = bake_face(face(), 100, 1, source, 0.02, 0.02)
            np.testing.assert_allclose(rgb, 0.65)
            self.assertFalse(supported.any())
            self.assertTrue(np.isnan(depth).all())

    def test_edge_fallback_preserves_existing_projection_and_gap_colors(self):
        source = cloud([[0.035, 0.5, 0.015], [0, 0.5, 0.005]], spacing=0.01,
                       colors=[[0, 0, 1], [1, 0, 0]])
        source["normals"][1] = [1, 0, 0]
        rgb, supported, depth = bake_face(face(), 100, 1, source, 0.02, 0.02)
        np.testing.assert_allclose(rgb[0, :7], np.tile([0, 0, 1], (7, 1)), atol=1e-12)
        self.assertTrue(supported[0, 3])
        self.assertFalse(supported[0, 0])
        self.assertTrue(np.isnan(depth[0]))

    def test_adjacent_plane_fallback_reaches_all_four_edges_and_corners(self):
        for axis in (0, 1):
            for boundary in (0., 1.):
                point = np.array([0.5, 0.5, 0.005])
                point[axis] = boundary
                normal = np.zeros((1, 3))
                normal[0, axis] = 1
                source = cloud([point], spacing=0.03)
                source["normals"] = normal
                rgb, supported, depth = bake_face(face(), 20, 20, source, 0.02, 0.02)
                row, column = (10, 0 if boundary == 0 else 19) if axis == 0 else (
                    19 if boundary == 0 else 0, 10
                )
                np.testing.assert_allclose(rgb[row, column], [1, 0, 0], atol=1e-12)
                np.testing.assert_allclose(rgb[10, 10], 0.65)
                self.assertFalse(supported.any())
                self.assertTrue(np.isnan(depth).all())
        for x in (0., 1.):
            for y in (0., 1.):
                source = cloud([[x, y, 0.005]], spacing=0.03)
                source["normals"] = np.array([[1., 0, 0]])
                rgb, _, _ = bake_face(face(), 20, 20, source, 0.02, 0.02)
                np.testing.assert_allclose(rgb[19 if y == 0 else 0, 0 if x == 0 else 19],
                                           [1, 0, 0], atol=1e-12)

    def test_edge_layers_preserve_opacity_order_and_duplicate_invariance(self):
        for duplicates in (1, 130):
            points = np.vstack(([[0, 0.5, 0.005]],
                                np.tile([0, 0.5, 0.02], (duplicates, 1))))
            colors = np.vstack(([1, 0, 0], np.tile([0, 0, 1], (duplicates, 1))))
            opacity = np.r_[0.1, np.ones(duplicates)]
            for order in (np.arange(len(points)), np.arange(len(points))[::-1]):
                source = cloud(points[order], spacing=0.01, colors=colors[order])
                source["opacity"] = opacity[order]
                source["normals"] = np.tile([1., 0, 0], (len(points), 1))
                rgb, supported, depth = bake_face(face(), 100, 1, source, 0.03, 0.03)
                np.testing.assert_allclose(rgb[0, 0], linear_to_srgb([0.1, 0, 0.9]),
                                           atol=1e-12)
                self.assertFalse(supported.any())
                self.assertTrue(np.isnan(depth).all())

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


class TextureVisibilityTests(unittest.TestCase):
    def test_outer_checkerboard_is_not_hidden_by_nearer_reverse_skin(self):
        size = 17
        x, y = np.meshgrid((np.arange(size) + 0.5) / size, (np.arange(size) + 0.5) / size)
        pattern = (np.indices((size, size)).sum(0) % 2).ravel()
        front = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 0.03)))
        back = front.copy()
        back[:, 2] = -0.005
        front_colors = np.column_stack((pattern, 1 - pattern, np.zeros(x.size)))
        source = cloud(np.vstack((front, back)), spacing=0.01,
                       colors=np.vstack((front_colors, np.tile([0, 0, 1], (x.size, 1)))))
        rgb, supported, depth = bake_face(face(), size, size, source, 0.1, 0.1, outward=True)
        np.testing.assert_allclose(rgb, front_colors.reshape(size, size, 3)[::-1], atol=1e-12)
        self.assertTrue(supported.all())
        np.testing.assert_allclose(depth, 0.03)
        reverse = face()
        reverse.update(origin=np.array([1., 0, 0]), u=np.array([-1., 0, 0]),
                       normal=np.array([0., 0, -1]))
        rgb, supported, depth = bake_face(reverse, size, size, source, 0.1, 0.1, outward=True)
        np.testing.assert_allclose(rgb, np.broadcast_to([0, 0, 1], rgb.shape), atol=1e-12)
        self.assertTrue(supported.all())
        np.testing.assert_allclose(depth, 0.005)

    def test_gaussian_tail_does_not_occlude_the_local_surface(self):
        for duplicates in (1, 130):
            points = np.vstack((np.tile([0.645, 0.5, 0.04], (duplicates, 1)), [0.5, 0.5, 0.01]))
            colors = np.vstack((np.zeros((duplicates, 3)), [1, 1, 1]))
            for order in (np.arange(len(points)), np.arange(len(points))[::-1]):
                source = cloud(points[order], spacing=0.001, colors=colors[order],
                               covariance=np.eye(3) * 0.05**2)
                rgb, supported, depth = bake_face(face(), 1, 1, source, 0.1, 0.1, outward=True)
                tail_alpha = np.exp(-0.5 * 0.145**2 / (0.05**2 + (0.45 * 0.001)**2))
                np.testing.assert_allclose(rgb, linear_to_srgb(1 - tail_alpha), atol=1e-12)
                self.assertTrue(supported.all())
                np.testing.assert_allclose(depth, 0.04)

    def bake_slabs(self, neighbor, rotation, only_neighbor=False):
        centers = np.array([[0., 0, 0]] + ([[0., 0, 0.08]] if neighbor else []))
        dimensions = np.tile([1., 1, 0.02], (len(centers), 1))
        points = np.array([[0., 0, 0.04], [0., 0, 0.005]] +
                          ([[0., 0, 0.08]] if neighbor else []))
        colors = np.array([[1., 0, 0], [0., 1, 0]] + ([[0., 0, 1]] if neighbor else []))
        if only_neighbor:
            points = np.array([[-0.1, 0, 0.08], [0, 0, 0.08], [0.1, 0, 0.08]])
            colors = np.tile([0., 0, 1], (3, 1))
        source = cloud(points @ rotation.T, spacing=0.001, colors=colors,
                       covariance=np.eye(3) * 2**2)
        source.update(normals=source["normals"] @ rotation.T,
                      spacing_per_point=np.full(len(points), 0.001),
                      color_source="vertex_rgb", anisotropic_splats=0, kernel_normals_recovered=0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "parameters.npz", centers=centers @ rotation.T, dimensions=dimensions,
                     rotations=np.tile(rotation, (len(centers), 1, 1)), surface_tolerance=0.04,
                     corners=(centers[:, None] + CORNER_SIGNS[None] * dimensions[:, None] / 2) @ rotation.T)
            np.savez(root / "common_geometry.npz", points=source["points"], colors=colors)
            with patch("cuboid_approximation.texture.load_cloud", return_value=source):
                out, report = bake_model(root / "unused.ply", root, root, root / "textured", 256)
            atlas = np.asarray(Image.open(out / "texture_atlas.png"))
            samples = {}
            for detail in report["face_details"]:
                x, y, w, h = detail["rect"]
                samples[detail["box"], detail["side"]] = atlas[y + h // 2, x + w // 2].copy() / 255
            self.assertTrue(report["geometry_unchanged"])
            self.assertEqual(sum(f["outward_layer_order"] for f in report["face_details"]),
                             2 * len(centers))
            # Verify that the exported report describes the same projection policy.
            self.assertEqual(json.loads((out / "texture_report.json").read_text())["face_details"],
                             report["face_details"])
        return samples, report

    def test_model_bake_recovers_both_sides_of_a_displaced_thin_slab(self):
        for rotation in (np.eye(3), Rotation.from_euler("xyz", [23, 51, -17], degrees=True).as_matrix()):
            samples, _ = self.bake_slabs(False, rotation)
            np.testing.assert_allclose(samples[1, 4], [0, 1, 0], atol=0.02)
            np.testing.assert_allclose(samples[1, 5], [1, 0, 0], atol=0.02)

    def test_model_bake_does_not_steal_a_neighbor_slabs_color(self):
        for rotation in (np.eye(3), Rotation.from_euler("xyz", [23, 51, -17], degrees=True).as_matrix()):
            samples, _ = self.bake_slabs(True, rotation)
            np.testing.assert_allclose(samples[1, 5], [1, 0, 0], atol=0.02)
            np.testing.assert_allclose(samples[2, 5], [0, 0, 1], atol=0.02)

    def test_unobserved_slab_retains_nearest_color_as_unsupported_fallback(self):
        samples, report = self.bake_slabs(True, np.eye(3), only_neighbor=True)
        np.testing.assert_allclose(samples[1, 5], [0, 0, 1], atol=0.02)
        self.assertEqual(report["face_details"][5]["projected_fraction"], 0)
        self.assertIsNone(report["face_details"][5]["median_projection_depth"])


if __name__ == "__main__":
    unittest.main()
