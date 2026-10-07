"""Appearance completion follows cuboid surfaces and keeps measured colors intact."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from cuboid_approximation.geometry import CORNER_SIGNS
from cuboid_approximation.texture import complete_chart_edges, faces_from_corners


class TextureEdgeCompletionTests(unittest.TestCase):
    def charts(self, dimensions=(1, 1, 1), size=20, rotation=None, copies=1):
        corners = CORNER_SIGNS * np.asarray(dimensions) / 2
        if rotation is not None:
            corners = corners @ rotation.T + [2, -3, 7]
        faces = faces_from_corners(np.tile(corners, (copies, 1, 1)))
        tiles = [np.full((size, size, 3), 166, np.uint8) for _ in faces]
        masks = [np.zeros((size, size), bool) for _ in faces]
        return faces, tiles, masks

    def test_nearby_gaps_fill_but_observed_gray_and_interior_stay_intact(self):
        faces, tiles, masks = self.charts()
        masks[0][2:4, :] = True
        tiles[0][2:4, :] = [255, 0, 0]
        masks[0][0, 0] = True  # Real gray must not be mistaken for a gap.
        before = tiles[0].copy()
        original_masks = [m.copy() for m in masks]
        completed = complete_chart_edges(faces, tiles, masks, 0.16)
        np.testing.assert_array_equal(tiles[0][0, 5], [255, 0, 0])
        np.testing.assert_array_equal(tiles[0][masks[0]], before[masks[0]])
        np.testing.assert_array_equal(tiles[0][10, 10], [166, 166, 166])
        self.assertTrue(completed[0][0, 5])
        self.assertFalse(completed[0][10, 10])
        for before_mask, mask in zip(original_masks, masks):
            np.testing.assert_array_equal(before_mask, mask)

    def test_adjacent_faces_fill_without_other_box_or_opposite_skin_shortcuts(self):
        faces, tiles, masks = self.charts(dimensions=(0.02, 1, 1), copies=2)
        source = next(i for i, f in enumerate(faces[:6]) if abs(f['normal'][0]) > 0.9)
        reverse = next(i for i, f in enumerate(faces[:6]) if f['normal'] @ faces[source]['normal'] < -0.9)
        masks[source][:] = True
        tiles[source][:] = [12, 34, 56]
        completed = complete_chart_edges(faces, tiles, masks, 0.12)
        self.assertFalse(completed[reverse].any())
        self.assertTrue(any(c.any() for i, c in enumerate(completed[:6]) if i != source))
        self.assertFalse(any(c.any() for c in completed[6:]))
        for tile, mask in zip(tiles, completed):
            if mask.any():
                np.testing.assert_array_equal(tile[mask], np.tile([12, 34, 56], (mask.sum(), 1)))

    def test_adjacent_distance_is_unfolded_and_not_a_diagonal_shortcut(self):
        faces, tiles, masks = self.charts()
        masks[0][:] = True
        tiles[0][:] = [255, 0, 0]
        completed = complete_chart_edges(faces, tiles, masks, 0.09)
        # Centers are 0.025 from each edge: first row travels 0.05, second
        # travels 0.10. A 3D diagonal would incorrectly color the second row.
        for i, f in enumerate(faces):
            if abs(f['normal'] @ faces[0]['normal']) < 0.1:
                self.assertEqual(completed[i].sum(), 20)

    def test_hidden_sources_and_empty_charts_cannot_donate(self):
        faces, tiles, masks = self.charts()
        masks[0][:] = True
        tiles[0][:] = [255, 0, 0]
        faces[0]['hidden'] = True
        before = [t.copy() for t in tiles]
        completed = complete_chart_edges(faces, tiles, masks, 1)
        self.assertFalse(any(c.any() for c in completed))
        for a, b in zip(before, tiles):
            np.testing.assert_array_equal(a, b)

    def test_completion_does_not_cascade_and_is_rotation_invariant(self):
        results = []
        for rotation in (None, Rotation.from_euler('xyz', [13, 27, 49], degrees=True).as_matrix()):
            faces, tiles, masks = self.charts(dimensions=(0.02, 0.8, 1), rotation=rotation)
            source = next(i for i, f in enumerate(faces) if f['width'] * f['height'] > 0.7)
            masks[source][:] = True
            tiles[source][:] = [255, 0, 0]
            results.append(complete_chart_edges(faces, tiles, masks, 0.12))
        for a, b in zip(*results):
            np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
