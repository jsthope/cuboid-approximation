"""Containment acceleration must preserve exact separating-axis decisions."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from cuboid_approximation.cuboids import _sat_overlaps_voxels


def legacy_sat(centers, box, voxel_width, epsilon):
    """Frozen scalar-axis implementation used before batching cross products."""
    frame, half = box["rotation"], box["dimensions"] / 2
    delta = centers - box["center"]
    axes = [*np.eye(3), *frame.T]
    axes += [np.cross(a, b) for a in np.eye(3) for b in frame.T]
    overlapping = np.ones(len(centers), bool)
    for axis in axes:
        length = np.linalg.norm(axis)
        if length < 1e-10:
            continue
        axis = axis / length
        radius = np.sum(np.abs(frame.T @ axis) * half)
        radius += 0.5 * voxel_width * np.abs(axis).sum()
        overlapping &= np.abs(delta @ axis) < radius - epsilon
        if not overlapping.any():
            break
    return overlapping


class SeparatingAxisExecutionTests(unittest.TestCase):
    def test_rotated_and_boundary_masks_match_legacy(self):
        rng = np.random.default_rng(92)
        for index in range(64):
            frame = Rotation.random(random_state=rng).as_matrix() if index % 2 else np.eye(3)
            dimensions = rng.uniform(0.01, 2, 3)
            center = rng.normal(size=3)
            box = dict(rotation=frame, dimensions=dimensions, center=center)
            centers = center + rng.uniform(-1, 1, (100, 3)) * dimensions
            # Probe both sides of OBB-face/voxel contact, including roundoff
            # neighbors where changing an arithmetic ordering is significant.
            for axis in range(3):
                normal = frame[:, axis]
                radius = dimensions[axis] / 2 + 0.025 * np.abs(normal).sum()
                offsets = np.array([np.nextafter(radius, 0), radius, np.nextafter(radius, np.inf)])
                centers = np.concatenate((centers, center + offsets[:, None] * normal))
            for epsilon in (0.0, 1e-10):
                np.testing.assert_array_equal(
                    _sat_overlaps_voxels(centers, box, 0.05, epsilon),
                    legacy_sat(centers, box, 0.05, epsilon),
                )

    def test_face_contact_is_allowed_but_one_ulp_overlap_is_detected(self):
        box = dict(rotation=np.eye(3), dimensions=np.ones(3), center=np.zeros(3))
        centers = np.array([
            [1, 0, 0],
            [np.nextafter(1.0, 0.0), 0, 0],
            [np.nextafter(1.0, np.inf), 0, 0],
            [-1, 0, 0],
            [-np.nextafter(1.0, 0.0), 0, 0],
        ])
        np.testing.assert_array_equal(
            _sat_overlaps_voxels(centers, box, 1), [False, True, False, False, True]
        )


if __name__ == "__main__":
    unittest.main()
