"""Coplanar texture patches have one owner without changing cuboid geometry."""

import base64
import json
from pathlib import Path
import re
import struct
import tempfile
import unittest

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from cuboid_approximation.geometry import CORNER_SIGNS
from cuboid_approximation.texture import (
    export_glb, export_obj, export_viewer, face_pixel_mask, face_uv,
    faces_from_corners, polygon_area, polygon_uv, render_face_polygons,
)


def model(centers):
    centers = np.asarray(centers, dtype=float)
    dimensions = np.full((len(centers), 3), 2.)
    rotations = np.tile(np.eye(3), (len(centers), 1, 1))
    return dict(centers=centers, dimensions=dimensions, rotations=rotations,
                corners=CORNER_SIGNS[None] + centers[:, None])


class CoplanarExportTests(unittest.TestCase):
    def test_identical_boxes_have_one_owner_and_preserve_first_prefix(self):
        params = model([[0, 0, 0], [0, 0, 0]])
        saved = params["corners"].copy()
        faces = faces_from_corners(params["corners"])
        polygons = render_face_polygons(params["corners"], faces)
        self.assertEqual(sum(polygon_area(p) for ps in polygons for p in ps), 24.)
        self.assertTrue(all(not ps for ps in polygons[6:]))
        for face, ps in zip(faces[:6], polygons[:6]):
            np.testing.assert_array_equal(ps[0], saved[0, face["indices"]])
        np.testing.assert_array_equal(params["corners"], saved)

    def test_partial_overlap_is_clipped_with_original_uv_chart(self):
        params = model([[0, 0, 0], [1, 0, 0]])
        faces = faces_from_corners(params["corners"])
        polygons = render_face_polygons(params["corners"], faces)
        top = next(i for i, f in enumerate(faces) if f["box"] == 1 and f["normal"][2] > .9)
        self.assertAlmostEqual(sum(polygon_area(p) for p in polygons[top]), 2.)
        rect = (8, 16, 80, 60)
        face = faces[top]
        q = params["corners"][1, face["indices"]]
        np.testing.assert_allclose(polygon_uv(q, face, rect, 256), face_uv(rect, 256))
        mask = face_pixel_mask(face, polygons[top], 8, 8)
        self.assertEqual(mask.sum(), 32)
        for polygon in polygons[top]:
            self.assertTrue(np.all(polygon[:, 0] >= 1 - 1e-12))
            uv = polygon_uv(polygon, face, rect, 256)
            decoded = (face["origin"] + ((uv[:, 0] * 256 - rect[0]) / rect[2])[:, None]
                       * face["width"] * face["u"]
                       + (1 - (uv[:, 1] * 256 - rect[1]) / rect[3])[:, None]
                       * face["height"] * face["v"])
            np.testing.assert_allclose(decoded, polygon, atol=1e-12)

    def test_clipping_is_invariant_to_rotation_scale_and_large_translation(self):
        original = model([[0, 0, 0], [1, 0, 0]])["corners"]
        rotation = Rotation.from_euler("xyz", [15, 33, -27], degrees=True).as_matrix()
        for scale, offset in ((1e-4, np.zeros(3)), (3., np.array([1e8, -2e8, 3e8]))):
            corners = original * scale @ rotation.T + offset
            polygons = render_face_polygons(corners, faces_from_corners(corners))
            area = sum(polygon_area(p) for ps in polygons for p in ps)
            self.assertAlmostEqual(area / scale**2, 40., places=5)

    def test_future_large_box_cannot_change_rendered_prefix(self):
        prefix = model([[0, 0, 0], [1, 0, 0]])["corners"]
        large = (CORNER_SIGNS * 1e12 + 1e14)[None]
        corners = np.concatenate((prefix, large))
        before = render_face_polygons(prefix, faces_from_corners(prefix))
        after = render_face_polygons(corners, faces_from_corners(corners))
        for expected, actual in zip(before, after):
            self.assertEqual(len(expected), len(actual))
            for a, b in zip(expected, actual):
                np.testing.assert_array_equal(a, b)

    def test_distant_first_box_does_not_erase_later_small_faces(self):
        small = model([[0, 0, 0], [.0001, 0, 0]])["corners"] * 1e-4
        large = (CORNER_SIGNS + 1e12)[None]
        corners = np.concatenate((large, small))
        before = render_face_polygons(small, faces_from_corners(small))
        after = render_face_polygons(corners, faces_from_corners(corners))[6:]
        for expected, actual in zip(before, after):
            self.assertEqual(len(expected), len(actual))
            for a, b in zip(expected, actual):
                np.testing.assert_array_equal(a, b)

    def test_separated_and_oppositely_oriented_faces_are_not_clipped(self):
        for displacement in ([2, 0, 0], [1, 0, 0.01]):
            params = model([[0, 0, 0], displacement])
            faces = faces_from_corners(params["corners"])
            polygons = render_face_polygons(params["corners"], faces)
            # The contacting +/-X faces remain separate: both orientations
            # are intentional, not co-oriented texture duplicates.
            for i in (0, 1, 6, 7):
                np.testing.assert_array_equal(polygons[i][0],
                                              params["corners"][faces[i]["box"], faces[i]["indices"]])

    def test_exports_allow_a_box_without_render_triangles(self):
        params = model([[0, 0, 0], [0, 0, 0]])
        faces = faces_from_corners(params["corners"])
        rects = [(4, 4, 8, 8)] * 12
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            for name in ("texture_atlas.png", "projection_confidence.png"):
                Image.new("RGB", (32, 32), "white").save(out / name)
            np.savez(out / "common.npz", points=CORNER_SIGNS, colors=np.ones((8, 3)))
            report = dict(cuboids=2, faces=12, atlas_size=[32, 32], supported_area_fraction=1.)
            export_glb(out, params, faces, rects, 32)
            export_obj(out, params, faces, rects, 32)
            export_viewer(out, params, faces, rects, 32, report, out / "common.npz")
            data = (out / "cuboids_textured.glb").read_bytes()
            length = struct.unpack_from("<I", data, 12)[0]
            gltf = json.loads(data[20:20 + length])
            self.assertEqual(len(gltf["meshes"]), 1)
            self.assertNotIn("mesh", gltf["nodes"][1])
            self.assertEqual(gltf["nodes"][-1]["children"], [0, 1])
            html = (out / "cuboids_textured_3d.html").read_text()
            payload = json.loads(re.search(r"const data=(.*?);", html).group(1))
            self.assertEqual(payload["box_vertex_ends"], [0, 36, 36])
            self.assertEqual(len(base64.b64decode(payload["vertices"])), 36 * 3 * 4)
            obj = (out / "cuboids_textured.obj").read_text().splitlines()
            self.assertEqual(sum(line.startswith("f ") for line in obj), 6)


if __name__ == "__main__":
    unittest.main()
