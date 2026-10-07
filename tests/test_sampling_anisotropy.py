"""Rectangular scans must retain tangent support under rigid rotations."""

import unittest

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from cuboid_approximation.cloud import local_geometry, local_point_spacing
from cuboid_approximation.texture import bake_face, sampling_covariance, transfer_normals


class SamplingAnisotropyTests(unittest.TestCase):
    def test_six_to_one_grid_keeps_transverse_neighbors_after_rotation(self):
        x, y = np.meshgrid(np.linspace(0, 1.5, 21), np.linspace(0, 0.25, 21))
        original = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        interior = (x.ravel() > 0) & (x.ravel() < 1.5) & (y.ravel() > 0) & (y.ravel() < 0.25)
        rotations = [np.eye(3), Rotation.from_euler("xyz", [13, 29, 51], degrees=True).as_matrix()]
        valid_masks = []
        for rotation in rotations:
            points = original @ rotation.T
            geometry = local_geometry(points)
            valid_masks.append(geometry["valid"])
            self.assertTrue(geometry["valid"][interior].all())
            np.testing.assert_allclose(np.abs(geometry["normals"][interior] @ rotation[:, 2]), 1,
                                       atol=1e-10)
            # Exercise both the donor path and the local fallback PCA.
            for valid in (geometry["valid"], np.zeros(len(points), bool)):
                normals, confidence = transfer_normals(points, points, geometry["normals"], valid,
                                                       geometry["spacing"], geometry["spacing"])
                self.assertTrue(np.all(confidence[interior] > 0.5))
                np.testing.assert_allclose(np.abs(normals[interior] @ rotation[:, 2]), 1, atol=1e-10)
            scales, tree = geometry["spacing_per_point"], cKDTree(points)
            covariance = sampling_covariance(points, normals, confidence, scales, tree,
                                             np.arange(len(points)), np.ones(len(points), bool))
            # Wider transverse steps are supported without inflating depth variance.
            depth_variance = np.einsum("i,nij,j->n", rotation[:, 2], covariance, rotation[:, 2])
            np.testing.assert_allclose(depth_variance, (0.6 * scales) ** 2)
            cloud = dict(points=points, normals=normals, normal_confidence=confidence,
                         spacing=geometry["spacing"], spacing_per_point=scales,
                         covariance=covariance, colors=np.tile([1., 0, 0], (len(points), 1)),
                         opacity=np.ones(len(points)), tree=tree)
            face = dict(origin=np.zeros(3), u=rotation[:, 0], v=rotation[:, 1],
                        normal=rotation[:, 2], width=1.5, height=0.25)
            rgb, supported, _ = bake_face(face, 240, 40, cloud, max_depth=0.01)
            self.assertGreater(supported.mean(), 0.98)
            np.testing.assert_allclose(rgb[supported], np.tile([1., 0, 0], (supported.sum(), 1)),
                                       atol=1e-10)
        np.testing.assert_array_equal(valid_masks[0], valid_masks[1])

    def test_off_plane_neighbors_do_not_widen_sampling_footprints(self):
        x = np.linspace(0, 0.2, 21)
        line = np.column_stack((x, np.zeros((len(x), 2))))
        points = np.vstack((line, line + [0, 0.05, 0.01]))
        normals = np.tile([0., 0, 1], (len(points), 1))
        scales = local_point_spacing(points)
        covariance = sampling_covariance(points, normals, np.ones(len(points)), scales,
                                         cKDTree(points), np.arange(len(points)),
                                         np.ones(len(points), bool))
        np.testing.assert_allclose(covariance, np.eye(3)[None] * (0.6 * scales[:, None, None]) ** 2)


if __name__ == "__main__":
    unittest.main()
