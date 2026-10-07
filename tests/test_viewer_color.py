"""Viewer point colors share the atlas's sRGB display convention."""

import base64
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from cuboid_approximation.geometry import CORNER_SIGNS
from cuboid_approximation.texture import export_viewer, faces_from_corners, pack_atlas


class ViewerColorTests(unittest.TestCase):
    def export_variants(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            points = np.array([[10.0, 20, 30], [11.0, 22, 34]])
            colors = np.array([[0.5, 0.003, 1.0], [0.25, 0, 0.75]])
            common = root / "common.npz"
            np.savez(common, points=points, colors=colors)
            original = common.read_bytes()
            params = dict(corners=CORNER_SIGNS[None] + points.mean(axis=0))
            faces = faces_from_corners(params["corners"])
            rects, _ = pack_atlas(faces, 256)
            for filename in ("texture_atlas.png", "projection_confidence.png"):
                Image.new("RGB", (256, 256)).save(root / filename)
            payloads = {}
            for color_space in ("linear", "srgb", None):
                report = dict(
                    cuboids=1, faces=6, atlas_size=[256, 256], supported_area_fraction=1.0
                )
                if color_space is not None:
                    report["source_color_space"] = color_space
                export_viewer(root, params, faces, rects, 256, report, common)
                html = (root / "cuboids_textured_3d.html").read_text()
                payloads[color_space] = json.loads(
                    html.split("const data=", 1)[1].split(";\n", 1)[0]
                )
            self.assertEqual(common.read_bytes(), original)
        return payloads, colors

    def test_linear_colors_are_encoded_for_display(self):
        payloads, _ = self.export_variants()
        actual = np.frombuffer(base64.b64decode(payloads["linear"]["colors"]), dtype="<f4")
        np.testing.assert_allclose(
            actual,
            [0.735356983, 0.03876, 1.0, 0.537098730, 0, 0.880825021],
            atol=3e-8,
            rtol=0,
        )

    def test_srgb_and_legacy_calls_preserve_colors_and_positions(self):
        payloads, colors = self.export_variants()
        actual = np.frombuffer(base64.b64decode(payloads["srgb"]["colors"]), dtype="<f4")
        np.testing.assert_array_equal(actual.reshape(-1, 3), colors.astype("<f4"))
        self.assertEqual(payloads["srgb"]["colors"], payloads[None]["colors"])
        for field in ("vertices", "points", "wires", "uv", "source_origin", "source_scale"):
            self.assertEqual(payloads["linear"][field], payloads["srgb"][field])
            self.assertEqual(payloads["linear"][field], payloads[None][field])


if __name__ == "__main__":
    unittest.main()
