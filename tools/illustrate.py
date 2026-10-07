"""Render README figures from a completed run (no browser or external assets)."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
from PIL import Image

from cuboid_approximation.texture import faces_from_corners, polygon_uv, render_face_polygons


BG = "#f5f5f4"
BLUE = "#2078b4"
ORANGE = "#e79535"
RED = "#e24250"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def edge_diagnostics(root, count):
    path = root / "preparation/edges.npz"
    if path.is_file():
        with np.load(path) as data:
            return data["mask"], True
    print("Edge diagnostics unavailable: add --diagnostics to a pipeline run or resume.")
    return np.zeros(count, bool), False


def camera(points):
    az, el = np.radians([40, 12])
    eye = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.cross([0, 0, 1], eye)
    right /= np.linalg.norm(right)
    up = np.cross(eye, right)
    axes = np.column_stack((right, up, eye))
    display = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
    transform = display @ axes
    projected = points @ transform
    center = (projected[:, :2].min(0) + projected[:, :2].max(0)) / 2
    extent = np.ptp(projected[:, :2], axis=0).max() * 1.12
    return transform, center, extent


def point_panel(ax, points, colors, view, title, normals=None):
    transform, center, extent = view
    p = points @ transform
    order = np.argsort(p[:, 2], kind="stable")
    ax.scatter(p[order, 0], p[order, 1], c=np.asarray(colors)[order], s=1.2, linewidths=0)
    if normals is not None:
        ids = np.flatnonzero(normals["valid"])[::150]
        starts = p[ids, :2]
        directions = normals["normals"][ids] @ transform[:, :2] * extent * 0.025
        ax.quiver(
            *starts.T,
            *directions.T,
            color=BLUE,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.003,
            headwidth=3,
        )
    ax.set_xlim(center[0] - extent / 2, center[0] + extent / 2)
    ax.set_ylim(center[1] - extent / 2, center[1] + extent / 2)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=11, pad=5)


def render_cuboids(params, view, count=None, atlas=None, rectangles=None, size=700):
    """Orthographic triangle rasterization with a depth buffer and actual UVs."""
    transform, center, extent = view
    count = len(params["corners"]) if count is None else count
    rgb = np.broadcast_to(np.array([245, 245, 244], np.uint8), (size, size, 3)).copy()
    depth = np.full((size, size), -np.inf)
    faces = faces_from_corners(params["corners"][:count])
    polygons = render_face_polygons(params["corners"][:count], faces) if count else []
    light = np.array([0.4, -0.6, 1.0])
    light /= np.linalg.norm(light)
    fragments = ((i, face, q) for i, face in enumerate(faces) for q in polygons[i])
    for i, face, q in fragments:
        projected = q @ transform
        projected[:, :2] = (projected[:, :2] - center) / extent * (size - 1) + (size - 1) / 2
        projected[:, 1] = size - 1 - projected[:, 1]
        if atlas is None:
            normal = face["normal"][[0, 2, 1]] * [1, 1, -1]
            color = np.rint(
                255 * params["colors"][face["box"]] * (0.76 + 0.24 * abs(normal @ light))
            ).astype(np.uint8)
        else:
            uv = polygon_uv(q, face, rectangles[i], atlas.shape[0])
        for triangle in ([0, k, k + 1] for k in range(1, len(q) - 1)):
            a, b, c = projected[triangle]
            low = np.maximum(np.floor(np.min([a[:2], b[:2], c[:2]], axis=0)).astype(int), 0)
            high = np.minimum(np.ceil(np.max([a[:2], b[:2], c[:2]], axis=0)).astype(int), size - 1)
            if np.any(high < low):
                continue
            y, x = np.mgrid[low[1] : high[1] + 1, low[0] : high[0] + 1]
            denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(denominator) < 1e-9:
                continue
            w0 = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / denominator
            w1 = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / denominator
            w2 = 1 - w0 - w1
            z = w0 * a[2] + w1 * b[2] + w2 * c[2]
            active = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9) & (z > depth[y, x])
            yy, xx = y[active], x[active]
            depth[yy, xx] = z[active]
            if atlas is None:
                rgb[yy, xx] = color
            else:
                ta, tb, tc = uv[triangle]
                tuv = w0[active, None] * ta + w1[active, None] * tb + w2[active, None] * tc
                indices = np.clip(np.floor(tuv * atlas.shape[0]).astype(int), 0, atlas.shape[0] - 1)
                rgb[yy, xx] = atlas[indices[:, 1], indices[:, 0]]
    return rgb


def image_panel(ax, raster, title):
    ax.imshow(raster)
    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=11, pad=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/images"))
    parser.add_argument("--texture-dir", type=Path, help="Use a separately rebaked texture export")
    parser.add_argument("--pipeline-only", action="store_true", help="Update only pipeline.png")
    parser.add_argument("--caption", help="Explicit geometry/texture provenance label")
    args = parser.parse_args()
    root, out = args.run, args.output
    texture_dir = args.texture_dir or root / "textured"
    out.mkdir(parents=True, exist_ok=True)
    common = dict(np.load(root / "preparation/common_geometry.npz"))
    regions = dict(np.load(root / "preparation/regions.npz"))
    edge_mask, has_edges = edge_diagnostics(root, len(common["points"]))
    params = dict(np.load(root / "cuboids/parameters.npz"))
    texture = dict(np.load(texture_dir / "texture_parameters.npz"))
    if not np.array_equal(params["corners"], texture["corners"]):
        raise ValueError("Texture export geometry does not match the illustrated run.")
    report = read(root / "report.json")
    geometry = report["geometry"]
    view = camera(common["points"])
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": "#292929",
            "axes.labelcolor": "#444444",
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    def save(fig, name):
        fig.supxlabel(args.caption or f"Cuboid Approximation {report['version']}",
                      fontsize=8, color="#666666")
        fig.savefig(out / name, dpi=165, bbox_inches="tight", pad_inches=0.12)
        plt.close(fig)
        print(out / name)

    atlas = np.asarray(Image.open(texture_dir / "texture_atlas.png"))
    colored = render_cuboids(params, view)
    textured = render_cuboids(params, view, atlas=atlas, rectangles=texture["atlas_rectangles"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    point_panel(axes[0], common["points"], common["colors"], view, "Input point cloud")
    image_panel(axes[1], colored, f"{geometry['cuboids']} incremental cuboids")
    image_panel(axes[2], textured, "Projected PLY colors")
    save(fig, "pipeline.png")
    if args.pipeline_only:
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    point_panel(
        axes[0],
        common["points"],
        common["colors"],
        view,
        "1. PCA normals",
        normals=dict(valid=common["normal_valid"], normals=common["normals"]),
    )
    edge_colors = np.where(edge_mask[:, None], [0.89, 0.15, 0.22], [0.72, 0.74, 0.75])
    point_panel(
        axes[1],
        common["points"],
        edge_colors,
        view,
        f"2. Normal variation >= {report['config']['normal_angle']:g} degrees"
        if has_edges else "2. Edge diagnostics not requested (--diagnostics)",
    )
    point_panel(
        axes[2],
        regions["points"],
        regions["colors"],
        view,
        "3. Connected regions for orientations"
        if report["preparation"].get("orientation_mode", "regions") == "regions"
        else "3. PCA orientations (region segmentation skipped)",
    )
    save(fig, "regions.png")

    volume = np.load(root / "cuboids/envelope.npz")
    reference, allowed = volume["reference"], volume["safe"]
    origin, h = volume["origin"], float(volume["voxel_size"])
    occupied = np.flatnonzero(reference.any(axis=(0, 1)))
    index = (np.median(common["points"][:, 2]) - origin[2]) / h
    index = occupied[np.argmin(abs(occupied - index))]
    classes = np.zeros(reference.shape[:2], np.uint8)
    classes[allowed[:, :, index]] = 1
    classes[reference[:, :, index]] = 2
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    ax.imshow(
        classes.T,
        origin="lower",
        cmap=ListedColormap([BG, ORANGE, BLUE]),
        vmin=0,
        vmax=2,
        interpolation="nearest",
        extent=[
            origin[0],
            origin[0] + classes.shape[0] * h,
            origin[1],
            origin[1] + classes.shape[1] * h,
        ],
    )
    ax.set(
        xlabel="Source X",
        ylabel="Source Y",
        title=f"Approximation envelope / section at Z = {origin[2] + (index + 0.5) * h:.4f}",
    )
    ax.legend(
        handles=[
            Patch(color=BLUE, label="Observed + inferred support"),
            Patch(color=ORANGE, label="Bounded exterior band"),
        ],
        frameon=False,
        loc="upper left",
    )
    save(fig, "envelope.png")

    curve = geometry["incremental_steps"]
    counts = sorted(
        set(
            [
                0,
                min(1, geometry["cuboids"]),
                min(16, geometry["cuboids"]),
                min(64, geometry["cuboids"]),
                geometry["cuboids"],
            ]
        )
    )
    fig, axes = plt.subplots(
        1, len(counts), figsize=(3.1 * len(counts), 3.2), squeeze=False, layout="constrained"
    )
    for ax, k in zip(axes.ravel(), counts):
        image_panel(
            ax,
            render_cuboids(params, view, count=k),
            f"K = {k} / {curve[k]['sampled_proximity']:.3%}",
        )
    save(fig, "increments.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained")
    x = np.array([row["cuboids"] for row in curve])
    for ax in axes:
        ax.plot(
            x, [100 * r["sampled_proximity"] for r in curve], color=BLUE, label="Sampled points"
        )
        ax.plot(
            x,
            [100 * r["full_cloud_proximity"] for r in curve],
            color="#169b82",
            ls="--",
            label="All retained points",
        )
        ax.axhline(100 * geometry["target_coverage"], color=RED, ls=":", label="Requested target")
        ax.set(xlabel="Cuboid count K", ylabel="Points within tolerance (%)")
        ax.grid(alpha=0.12)
    axes[0].set_ylim(0, 101)
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].set_xlim(max(0, geometry["cuboids"] - 40), geometry["cuboids"] + 1)
    axes[1].set_ylim(98.5, 100)
    axes[1].set_title("Final increments", loc="left", fontsize=11)
    save(fig, "coverage.png")


if __name__ == "__main__":
    main()
