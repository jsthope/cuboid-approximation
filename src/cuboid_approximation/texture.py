"""Project PLY colors onto a freshly computed cuboid model, without moving it."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .geometry import BOX_QUADS as QUADS, CORNER_SIGNS as SIGNS

ROOT = Path(__file__).resolve().parent


def up_rotation(source_up):
    """Proper rotation from a declared source up axis to glTF +Y."""
    axis = np.eye(3)["xyz".index(source_up[-1])] * (-1 if source_up.startswith("-") else 1)
    target = np.array([0.0, 1.0, 0.0])
    cross = np.cross(axis, target)
    if np.linalg.norm(cross) == 0:
        return np.eye(3) if axis @ target > 0 else np.diag([1.0, -1.0, -1.0])
    return Rotation.from_rotvec(cross * (np.pi / 2)).as_matrix()


def faces_from_corners(corners):
    faces = []
    for box, vertices in enumerate(corners):
        for side, indices in enumerate(QUADS):
            indices = indices.copy()
            q = vertices[indices]
            if np.linalg.norm(q[1] - q[0]) < np.linalg.norm(q[3] - q[0]):
                indices = np.roll(indices, -1)
                q = vertices[indices]
            du, dv = q[1] - q[0], q[3] - q[0]
            width, height = np.linalg.norm(du), np.linalg.norm(dv)
            u, v = du / width, dv / height
            faces.append(
                dict(
                    box=box,
                    side=side,
                    indices=indices,
                    origin=q[0],
                    u=u,
                    v=v,
                    normal=np.cross(u, v),
                    width=width,
                    height=height,
                )
            )
    return faces


def pack_atlas(faces, size=2048, pad=4):
    def attempt(density):
        dims = [
            (
                max(4, int(np.ceil(f["width"] * density))) if not f.get("hidden", False) else 4,
                max(4, int(np.ceil(f["height"] * density))) if not f.get("hidden", False) else 4,
            )
            for f in faces
        ]
        order = sorted(range(len(faces)), key=lambda i: (-dims[i][1], -dims[i][0], i))
        free = [(0, 0, size, size)]
        rects = [None] * len(faces)
        for i in order:
            w, h = dims[i]
            rw, rh = w + 2 * pad, h + 2 * pad
            fits = [
                (min(fw - rw, fh - rh), fw * fh - rw * rh, k)
                for k, (_, _, fw, fh) in enumerate(free)
                if fw >= rw and fh >= rh
            ]
            if not fits:
                return None
            _, _, chosen = min(fits)
            x, y, fw, fh = free.pop(chosen)
            rects[i] = (x + pad, y + pad, w, h)
            # Split without overlapping free rectangles; choose the larger remainder.
            if fw - rw > fh - rh:
                pieces = [(x + rw, y, fw - rw, fh), (x, y + rh, rw, fh - rh)]
            else:
                pieces = [(x + rw, y, fw - rw, rh), (x, y + rh, fw, fh - rh)]
            free.extend(r for r in pieces if r[2] > 0 and r[3] > 0)
        return rects

    if not faces or any(
        not np.isfinite([f["width"], f["height"]]).all() or min(f["width"], f["height"]) <= 0
        for f in faces
    ):
        raise ValueError("Atlas faces must have finite, positive dimensions.")
    lo, hi = 0.0, size / max(max(f["width"], f["height"]) for f in faces)
    if attempt(lo) is None:
        raise ValueError("Atlas too small for the face count and padding.")
    for _ in range(45):
        mid = (lo + hi) / 2
        if attempt(mid) is None:
            hi = mid
        else:
            lo = mid
    return attempt(lo), lo


def srgb_to_linear(rgb):
    rgb = np.asarray(rgb)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb):
    rgb = np.maximum(rgb, 0)
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)


def transfer_normals(
    points, sample, normals, valid, spacing, sample_spacing,
    spacing_per_point=None, sample_spacing_per_point=None,
):
    """Transfer only nearby, tangent-compatible valid normals; fit unresolved points locally."""
    from .cloud import local_point_spacing

    point_scale = (
        local_point_spacing(points, spacing) if spacing_per_point is None
        else np.asarray(spacing_per_point, dtype=float)
    )
    sample_scale = (
        local_point_spacing(sample, sample_spacing) if sample_spacing_per_point is None
        else np.asarray(sample_spacing_per_point, dtype=float)
    )
    for scales, count in ((point_scale, len(points)), (sample_scale, len(sample))):
        if scales.shape != (count,) or not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError("Local spacing must be finite, positive and aligned with points.")
    output = np.zeros_like(points)
    confidence = np.zeros(len(points))
    good = np.flatnonzero(valid)
    if len(good):
        tree = cKDTree(sample[good])
        for start in range(0, len(points), 4096):
            p = points[start : start + 4096]
            scale = point_scale[start : start + len(p), None]
            d, ids = tree.query(p, k=list(range(1, min(8, len(good)) + 1)))
            exists = np.isfinite(d)
            ids = good[np.minimum(ids, len(good) - 1)]
            n = normals[ids]
            delta = p[:, None] - sample[ids]
            residual = np.abs(np.einsum("nki,nki->nk", delta, n))
            # Use the donor's sampling radius, but the finer of the two local
            # scales for tangent compatibility to avoid borrowing across sheets.
            compatible = exists & (d <= 4 * sample_scale[ids] * (1 + 1e-10))
            compatible &= residual <= 0.5 * np.minimum(scale, sample_scale[ids])
            cost = np.where(compatible, d + 2 * residual, np.inf)
            anchor = n[np.arange(len(p)), cost.argmin(1)]
            dots = np.einsum("nki,ni->nk", n, anchor)
            compatible &= np.abs(dots) >= np.cos(np.deg2rad(25))
            weights = np.where(compatible, 1 / np.maximum(d, scale * 0.25) ** 2, 0)
            mean = np.sum(n * np.sign(dots)[..., None] * weights[..., None], axis=1)
            length = np.linalg.norm(mean, axis=1)
            usable = length > 0
            output[start : start + len(p)][usable] = mean[usable] / length[usable, None]
            confidence[start : start + len(p)][usable] = length[usable] / weights.sum(1)[usable]
    unresolved = np.flatnonzero(confidence == 0)
    if len(unresolved) and len(points) >= 9:
        distinct = np.unique(points, axis=0)
        tree = cKDTree(distinct)
        for start in range(0, len(unresolved), 2048):
            target = unresolved[start : start + 2048]
            scale = point_scale[target]
            d, ids = tree.query(points[target], k=list(range(1, min(24, len(distinct)) + 1)))
            keep = d <= 6 * scale[:, None] * (1 + 1e-10)
            delta = (distinct[ids] - points[target, None]) / scale[:, None, None]
            delta *= keep[..., None]
            count = keep.sum(1)
            mean = delta.sum(1) / count[:, None]
            covariance = np.einsum("nki,nkj->nij", delta, delta) / count[:, None, None]
            covariance -= np.einsum("ni,nj->nij", mean, mean)
            values, axes = np.linalg.eigh(covariance)
            usable = (count >= 9) & (values[:, 1] > values.sum(1) * 1e-5)
            usable &= values[:, 0] <= 0.3 * values[:, 1]
            output[target[usable]] = axes[usable, :, 0]
            confidence[target[usable]] = 0.75
    return output, confidence


def sampling_covariance(
    points, normals, confidence, scales, tree, distinct_indices, eligible, covariance=None,
):
    """Infer tangent footprints for XYZ samples with directional sampling gaps.

    A rectangular scan can have a transverse step six times its nearest-point
    distance. Infer that step from nearby, coplanar, normal-compatible samples;
    never enlarge the normal variance or bridge an unsupported neighborhood.
    Explicit Gaussian covariances are supplied separately and remain unchanged.
    """
    if covariance is None:
        covariance = np.eye(3)[None] * (0.6 * scales[:, None, None]) ** 2
    targets = np.flatnonzero(eligible & (confidence > 0))
    k = list(range(1, min(31, len(distinct_indices)) + 1))
    for start in range(0, len(targets), 2048):
        target = targets[start : start + 2048]
        distance, near = tree.query(points[target], k=k)
        ids = distinct_indices[near]
        delta = points[ids] - points[target, None]
        normal = normals[target]
        residual = np.einsum("nki,ni->nk", delta, normal)
        tangent = delta - residual[..., None] * normal[:, None]
        alignment = np.abs(np.einsum("nki,ni->nk", normals[ids], normal))
        local_scale = scales[target, None]
        compatible = (distance > 0) & (distance <= 6 * local_scale * (1 + 1e-10))
        compatible &= np.abs(residual) <= 0.5 * np.minimum(local_scale, scales[ids])
        compatible &= (alignment >= np.cos(np.deg2rad(25))) & (confidence[ids] > 0)
        nearest = np.argmin(np.where(compatible, distance, np.inf), axis=1)
        direction = tangent[np.arange(len(target)), nearest]
        length = np.linalg.norm(direction, axis=1)
        direction = np.divide(direction, length[:, None], out=np.zeros_like(direction),
                              where=length[:, None] > 0)
        transverse = np.cross(normal, direction)
        transverse_distance = np.abs(np.einsum("nki,ni->nk", tangent, transverse))
        # A distinctly different tangent direction is needed: nearly collinear
        # neighbors cannot establish the missing transverse sampling interval.
        compatible &= transverse_distance >= 0.5 * np.linalg.norm(tangent, axis=2)
        step = np.min(np.where(compatible, transverse_distance, np.inf), axis=1)
        widen = np.isfinite(step) & (step > 1.5 * scales[target])
        amount = 0.6 ** 2 * (step[widen] ** 2 - scales[target[widen]] ** 2)
        covariance[target[widen]] += np.einsum(
            "n,ni,nj->nij", amount, transverse[widen], transverse[widen]
        )
    return covariance


def load_cloud(ply_path, preparation_dir, loaded=None, color_space="srgb"):
    from .cloud import load_points, local_point_spacing

    report = json.loads((Path(preparation_dir) / "report.json").read_text(encoding="utf-8"))
    if loaded is None:
        loaded = load_points(
            ply_path, report["min_opacity"], report.get("outlier_distance_factor", 50)
        )
    points, colors = loaded["points"], loaded["colors"]
    if not len(points):
        raise ValueError("No finite points remain after the opacity filter.")
    spacing = report.get("full_cloud_median_spacing", report["median_spacing"])
    local_spacing = loaded.get("spacing_per_point")
    if local_spacing is None:
        local_spacing = local_point_spacing(points, spacing)
    opacity = loaded["opacity"]
    covariance = np.eye(3)[None] * (0.6 * local_spacing[:, None, None]) ** 2
    anisotropic = np.zeros(len(points), dtype=bool)
    if loaded["splats"] is not None:
        logs = loaded["splats"]["log_scales"]
        quats = loaded["splats"]["quaternions_wxyz"]
        anisotropic = np.isfinite(logs).all(axis=1) & np.isfinite(quats).all(axis=1)
        quaternion_scale = np.max(np.abs(quats), axis=1)
        anisotropic &= quaternion_scale > 0
        scales = np.exp(np.clip(
            logs[anisotropic],
            np.log(0.35 * local_spacing[anisotropic, None]),
            np.log(2 * local_spacing[anisotropic, None]),
        ))
        rotations = (
            Rotation.from_quat(
                quats[anisotropic][:, [1, 2, 3, 0]] / quaternion_scale[anisotropic, None]
            ).as_matrix()
            if anisotropic.any()
            else np.empty((0, 3, 3))
        )
        covariance[anisotropic] = np.einsum("nik,nk,njk->nij", rotations, scales**2, rotations)
    with np.load(Path(preparation_dir) / "common_geometry.npz") as common:
        normals, normal_confidence = transfer_normals(
            points,
            common["points"],
            common["normals"],
            common["normal_valid"],
            spacing,
            report["median_spacing"],
            spacing_per_point=local_spacing,
            sample_spacing_per_point=common.get("spacing_per_point"),
        )
    tree = cKDTree(points)
    if not anisotropic.all():
        distinct_indices = loaded.get("distinct_indices")
        if distinct_indices is None:
            _, distinct_indices = np.unique(points, axis=0, return_index=True)
        distinct_tree = tree if len(distinct_indices) == len(points) else cKDTree(points[distinct_indices])
        # The original tree indexes input order, which need not match np.unique.
        if distinct_tree is tree:
            distinct_indices = np.arange(len(points))
        sampling_covariance(
            points, normals, normal_confidence, local_spacing,
            distinct_tree, distinct_indices, ~anisotropic, covariance=covariance,
        )
    return dict(
        points=points,
        colors=colors,
        opacity=opacity,
        normals=normals,
        normal_confidence=normal_confidence,
        color_space=color_space,
        linear_colors=srgb_to_linear(colors) if color_space == "srgb" else colors,
        covariance=covariance,
        tree=tree,
        spacing=spacing,
        spacing_per_point=local_spacing,
        color_source=loaded["color_source"],
        anisotropic_splats=int(anisotropic.sum()),
    )


def bake_face(face, width, height, cloud, max_depth, backward_depth=0.0):
    points = cloud["points"]
    scales = np.asarray(cloud.get("spacing_per_point", cloud["spacing"]), dtype=float)
    scales = np.broadcast_to(scales, (len(points),))
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Local spacing must be finite, positive and aligned with points.")
    colors = cloud.get("linear_colors")
    if colors is None:
        colors = (
            srgb_to_linear(cloud["colors"])
            if cloud.get("color_space", "srgb") == "srgb"
            else cloud["colors"]
        )
    gray = float(srgb_to_linear(0.65))
    origin, u, v, normal = [face[k] for k in ("origin", "u", "v", "normal")]
    center = origin + 0.5 * face["width"] * u + 0.5 * face["height"] * v
    # A covariance can be much wider than the local point spacing. The trace
    # bounds every directional variance, including rotated anisotropic splats.
    # Reduce in blocks to keep the broad phase's temporary memory bounded.
    margin = largest_scale = 0.0
    for start in range(0, len(points), 65536):
        local_scale = scales[start : start + 65536]
        variance = np.trace(cloud["covariance"][start : start + 65536], axis1=1, axis2=2)
        footprint = 3 * np.sqrt(np.maximum(variance, 0) + (0.45 * local_scale) ** 2)
        margin = max(margin, float(np.max(np.maximum(footprint, 4 * local_scale))))
        largest_scale = max(largest_scale, float(local_scale.max()))
    radius = np.sqrt(
        (face["width"] / 2 + margin) ** 2
        + (face["height"] / 2 + margin) ** 2
        + max(max_depth, backward_depth, 0.75 * largest_scale) ** 2
    )
    ids = np.array(cloud["tree"].query_ball_point(center, radius), dtype=int)
    delta = points[ids] - origin
    uv = np.column_stack((delta @ u, delta @ v))
    depth = delta @ normal
    alignment = np.abs(cloud["normals"][ids] @ normal)
    confidence = cloud.get("normal_confidence")
    confidence = np.ones(len(ids)) if confidence is None else confidence[ids]
    # These eligibility rules apply to both projection and interpolation. In
    # particular, fallback must never resurrect a source rejected by depth.
    use = (
        (depth >= -np.maximum(0.75 * scales[ids], backward_depth))
        & (depth <= max_depth)
        & (alignment >= 0.55)
        & (cloud["opacity"][ids] > 0)
        & (confidence > 0)
    )
    ids, uv, depth, alignment, confidence = [
        a[use] for a in (ids, uv, depth, alignment, confidence)
    ]
    local_scale = scales[ids]
    basis = np.column_stack((u, v))
    cov2 = np.einsum("ia,nij,jb->nab", basis, cloud["covariance"][ids], basis)
    cov2 += np.eye(2)[None] * (0.45 * local_scale[:, None, None]) ** 2
    # Coordinate extrema of the ellipse d.T @ inv(cov2) @ d <= 9 are exactly
    # +/-3*sqrt(diag(cov2)); off-diagonal covariance still enters the final test.
    extents = 3 * np.sqrt(np.diagonal(cov2, axis1=1, axis2=2))
    bounds = np.maximum(extents, 4 * local_scale[:, None])
    use = np.all((uv + bounds >= 0) & (uv - bounds <= [face["width"], face["height"]]), axis=1)
    ids, uv, depth, alignment, confidence, local_scale, cov2, extents = [
        a[use] for a in (ids, uv, depth, alignment, confidence, local_scale, cov2, extents)
    ]
    # Visit nearer layers first so an accepted texel anchor can never be
    # displaced by a later source. This makes layer selection a single pass.
    order = np.lexsort((depth, np.abs(depth)))
    ids, uv, depth, alignment, confidence, local_scale, cov2, extents = [
        a[order] for a in (ids, uv, depth, alignment, confidence, local_scale, cov2, extents)
    ]
    inv_cov = np.linalg.inv(cov2)
    source_weight = cloud["opacity"][ids] * alignment**4 * confidence
    rgb = np.full((height * width, 3), gray, dtype=np.float64)
    supported = np.zeros(height * width, dtype=bool)
    projected_depth = np.full(height * width, np.nan, dtype=np.float32)

    # Rasterize every compatible footprint, including interpolation donors.
    # Fixed neighbor quotas can hide a sparse near layer behind a dense far one.
    # At most 65,536 source/texel pairs are materialized at once, even for a
    # footprint that covers the entire atlas.
    def contributions(interpolate=False):
        pixel_density = np.array([width / face["width"], height / face["height"]])
        if interpolate:
            # Skip donors whose entire pixel rectangle was already projected.
            # The summed-area table avoids rasterizing large covered regions
            # just to fill a few isolated holes.
            missing = np.zeros((height + 1, width + 1), dtype=np.uint32)
            missing[1:, 1:] = ~np.isfinite(projected_depth.reshape(height, width))
            np.cumsum(missing, axis=0, out=missing)
            np.cumsum(missing, axis=1, out=missing)
        for start in range(0, len(ids), 128):
            stop = min(start + 128, len(ids))
            pixel_center = np.column_stack(
                (
                    uv[start:stop, 0] * pixel_density[0] - 0.5,
                    (face["height"] - uv[start:stop, 1]) * pixel_density[1] - 0.5,
                )
            )
            radius = (
                4 * local_scale[start:stop, None] if interpolate else extents[start:stop]
            ) * pixel_density
            # Clip before integer conversion; large physical kernels must not
            # overflow the pixel bounds or allocate a complete source grid.
            low = np.ceil(np.clip(pixel_center - radius, 0, [width, height])).astype(int)
            high = np.floor(np.clip(pixel_center + radius, -1, [width - 1, height - 1])).astype(int)
            spans = np.maximum(high - low + 1, 0)
            counts = np.prod(spans, axis=1)
            if interpolate:
                x0, y0 = low.T
                x1, y1 = (low + spans).T
                holes = missing[y1, x1] + missing[y0, x0] - missing[y1, x0] - missing[y0, x1]
                counts[holes == 0] = 0
            ends = np.cumsum(counts)
            for begin in range(0, int(ends[-1]), 65536):
                flat = np.arange(begin, min(begin + 65536, int(ends[-1])))
                local = np.searchsorted(ends, flat, side="right")
                offset = flat - (ends - counts)[local]
                xy = low[local] + np.column_stack(
                    (offset // spans[local, 1], offset % spans[local, 1])
                )
                owner = start + local
                pixels = xy[:, 1] * width + xy[:, 0]
                if interpolate:
                    unobserved = ~np.isfinite(projected_depth[pixels])
                    pixels, xy, owner = pixels[unobserved], xy[unobserved], owner[unobserved]
                target = np.column_stack(
                    (
                        (xy[:, 0] + 0.5) / pixel_density[0],
                        face["height"] - (xy[:, 1] + 0.5) / pixel_density[1],
                    )
                )
                delta = target - uv[owner]
                if interpolate:
                    distance2 = np.sum(delta**2, axis=1) + depth[owner] ** 2
                    weight = source_weight[owner] / np.maximum(
                        distance2, (0.25 * local_scale[owner]) ** 2
                    )
                    keep = distance2 <= (4 * local_scale[owner]) ** 2
                else:
                    mahal = np.einsum("ni,nij,nj->n", delta, inv_cov[owner], delta)
                    weight = np.exp(-0.5 * np.minimum(mahal, 150)) * source_weight[owner]
                    keep = mahal <= 9
                keep &= weight > 1e-10
                yield pixels[keep], owner[keep], weight[keep]

    def accumulate(interpolate=False):
        first = np.full(height * width, len(ids), dtype=int)
        total = np.zeros(height * width)
        color = np.zeros((height * width, 3))
        depths = np.zeros(height * width)
        ambiguous = np.zeros(height * width, bool)
        for pixels, owner, weight in contributions(interpolate):
            np.minimum.at(first, pixels, owner)
            anchor = first[pixels]
            separation = np.abs(depth[owner] - depth[anchor])
            # The finer scale prevents a sparse donor from blending across a
            # separate, densely sampled sheet (and works in either direction).
            layer_scale = np.minimum(local_scale[owner], local_scale[anchor])
            uncertain = (separation > 0.75 * layer_scale) & (
                np.abs(depth[owner]) - np.abs(depth[anchor]) < 0.25 * layer_scale
            )
            ambiguous[pixels[uncertain]] = True
            layer = separation <= 0.75 * layer_scale
            pixels, owner, weight = pixels[layer], owner[layer], weight[layer]
            np.add.at(total, pixels, weight)
            np.add.at(color, pixels, weight[:, None] * colors[ids[owner]])
            np.add.at(depths, pixels, weight * depth[owner])
        good = total > 1e-10
        rgb[good] = color[good] / total[good, None]
        if not interpolate:
            projected_depth[good] = depths[good] / total[good]
            supported[:] = good & ~ambiguous

    if len(ids):
        accumulate()
        if np.any(~np.isfinite(projected_depth)):
            # Gap interpolation remains explicitly unsupported and depthless.
            accumulate(interpolate=True)
    return (
        np.clip(linear_to_srgb(rgb), 0, 1).reshape(height, width, 3),
        supported.reshape(height, width),
        projected_depth,
    )


def face_uv(rect, size):
    x, y, w, h = rect
    return np.array([[x, y + h], [x + w, y + h], [x + w, y], [x, y]], dtype=float) / size


def export_obj(out, params, faces, rects, size):
    lines = [
        "# Cloned cuboids: original double-precision positions, projected UV texture",
        "mtllib cuboids_textured.mtl",
    ]
    for point in params["corners"].reshape(-1, 3):
        lines.append("v " + " ".join(format(float(x), ".17g") for x in point))
    for rect in rects:
        for u, v in face_uv(rect, size):
            lines.append(f"vt {u:.17g} {1 - v:.17g}")
    for f in faces:
        lines.append("vn " + " ".join(format(float(x), ".17g") for x in f["normal"]))
    lines += ["usemtl projected_cloud", "s off"]
    for i, f in enumerate(faces):
        if i % 6 == 0:
            lines.append(f"o cuboid_{f['box'] + 1:02d}")
        lines.append(
            "f "
            + " ".join(
                f"{f['box'] * 8 + int(idx) + 1}/{i * 4 + j + 1}/{i + 1}"
                for j, idx in enumerate(f["indices"])
            )
        )
    (out / "cuboids_textured.obj").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "cuboids_textured.mtl").write_text(
        "newmtl projected_cloud\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\nmap_Kd texture_atlas.png\n",
        encoding="utf-8",
    )


def export_glb(out, params, faces, rects, size, source_up="-y", units_per_meter=1.0):
    binary = bytearray()
    views, accessors, meshes, nodes = [], [], [], []

    def buffer_view(data, target=None):
        binary.extend(b"\0" * ((-len(binary)) % 4))
        item = dict(buffer=0, byteOffset=len(binary), byteLength=len(data))
        if target is not None:
            item["target"] = target
        binary.extend(data)
        views.append(item)
        return len(views) - 1

    def accessor(array, component, typ, target, bounds=False):
        item = dict(
            bufferView=buffer_view(array.tobytes(), target),
            componentType=component,
            count=len(array),
            type=typ,
        )
        if bounds:
            item.update(min=array.min(axis=0).tolist(), max=array.max(axis=0).tolist())
        accessors.append(item)
        return len(accessors) - 1

    source_origin = np.mean(params["centers"], axis=0)
    for box in range(len(params["centers"])):
        positions, normals, uv = [], [], []
        for i in range(box * 6, box * 6 + 6):
            local = SIGNS[faces[i]["indices"]]
            n = np.cross(local[1] - local[0], local[3] - local[0])
            n /= np.linalg.norm(n)
            positions.extend(local)
            normals.extend([n] * 4)
            uv.extend(face_uv(rects[i], size))
        indices = (
            (np.arange(6)[:, None] * 4 + np.array([0, 1, 2, 0, 2, 3])[None])
            .reshape(-1)
            .astype("<u2")
        )
        pa = accessor(np.asarray(positions, dtype="<f4"), 5126, "VEC3", 34962, True)
        na = accessor(np.asarray(normals, dtype="<f4"), 5126, "VEC3", 34962)
        ta = accessor(np.asarray(uv, dtype="<f4"), 5126, "VEC2", 34962)
        ia = accessor(indices, 5123, "SCALAR", 34963)
        meshes.append(
            dict(
                name=f"cuboid_{box + 1:02d}",
                primitives=[
                    dict(
                        attributes=dict(POSITION=pa, NORMAL=na, TEXCOORD_0=ta),
                        indices=ia,
                        material=0,
                    )
                ],
            )
        )
        matrix = np.eye(4)
        matrix[:3, :3] = params["rotations"][box] @ np.diag(params["dimensions"][box] / 2)
        matrix[:3, 3] = params["centers"][box] - source_origin
        nodes.append(
            dict(name=f"cuboid_{box + 1:02d}", mesh=box, matrix=matrix.T.reshape(-1).tolist())
        )
    image_view = buffer_view((out / "texture_atlas.png").read_bytes())
    root = np.eye(4)
    root[:3, :3] = up_rotation(source_up) / units_per_meter
    root_index = len(nodes)
    nodes.append(
        dict(
            name="source_to_gltf",
            matrix=root.T.reshape(-1).tolist(),
            children=list(range(root_index)),
        )
    )
    doc = dict(
        asset=dict(version="2.0", generator="Cloned cuboids / Gaussian color projection"),
        extensionsUsed=["KHR_materials_unlit"],
        extensionsRequired=["KHR_materials_unlit"],
        scene=0,
        scenes=[dict(nodes=[root_index])],
        nodes=nodes,
        meshes=meshes,
        materials=[
            dict(
                name="Projected PLY colors",
                doubleSided=False,
                pbrMetallicRoughness=dict(
                    baseColorTexture=dict(index=0), metallicFactor=0, roughnessFactor=1
                ),
                extensions={"KHR_materials_unlit": {}},
            )
        ],
        textures=[dict(sampler=0, source=0)],
        samplers=[dict(magFilter=9729, minFilter=9729, wrapS=33071, wrapT=33071)],
        images=[dict(bufferView=image_view, mimeType="image/png")],
        buffers=[dict(byteLength=len(binary))],
        bufferViews=views,
        accessors=accessors,
        extras=dict(
            coordinates="Centered, +Y up, meters; invert the root transform and add source_origin to recover PLY coordinates",
            source_origin=source_origin.tolist(),
            source_up=source_up,
            source_units_per_meter=units_per_meter,
            geometry="Unit local corners and double-precision node transforms",
        ),
    )
    encoded = json.dumps(doc, separators=(",", ":"), ensure_ascii=True).encode()
    encoded += b" " * ((-len(encoded)) % 4)
    binary.extend(b"\0" * ((-len(binary)) % 4))
    total = 12 + 8 + len(encoded) + 8 + len(binary)
    result = struct.pack("<4sII", b"glTF", 2, total)
    result += struct.pack("<I4s", len(encoded), b"JSON") + encoded
    result += struct.pack("<I4s", len(binary), b"BIN\0") + binary
    (out / "cuboids_textured.glb").write_bytes(result)


def export_viewer(out, params, faces, rects, size, report, common_path, source_up="-y"):
    def encoded(a):
        return base64.b64encode(np.asarray(a, dtype="<f4").tobytes()).decode()

    with np.load(common_path) as archive:
        common = dict(archive)
    vertices, uv, wires = [], [], []
    for f, rect in zip(faces, rects):
        q = params["corners"][f["box"]][f["indices"]]
        vertices.extend(q[[0, 1, 2, 0, 2, 3]])
        uv.extend(face_uv(rect, size)[[0, 1, 2, 0, 2, 3]])
        wires.extend(q[[0, 1, 1, 2, 2, 3, 3, 0]])
    origin = common["points"].min(0)
    scale = float(np.ptp(common["points"], axis=0).max())
    rotation = up_rotation(source_up)

    def display(points):
        return encoded(((np.asarray(points) - origin) / scale) @ rotation.T)

    # The point shader writes display RGB directly, matching the atlas shader's
    # sRGB output. Preparation keeps source colors in their declared space.
    display_colors = common["colors"]
    if report.get("source_color_space", "srgb") == "linear":
        display_colors = np.clip(linear_to_srgb(display_colors), 0, 1)
    payload = dict(
        vertices=display(vertices),
        uv=encoded(uv),
        wires=display(wires),
        points=display(common["points"]),
        source_origin=origin.tolist(),
        source_scale=scale,
        source_up=source_up,
        colors=encoded(display_colors),
        texture=base64.b64encode((out / "texture_atlas.png").read_bytes()).decode(),
        confidence=base64.b64encode((out / "projection_confidence.png").read_bytes()).decode(),
        report={
            k: report[k] for k in ("cuboids", "faces", "atlas_size", "supported_area_fraction")
        },
    )
    template = (ROOT / "viewer_template.html").read_text(encoding="utf-8")
    template = template.replace("__CUBOID_COUNT__", str(report["cuboids"]))
    (out / "cuboids_textured_3d.html").write_text(
        template.replace("__PAYLOAD__", json.dumps(payload)), encoding="utf-8"
    )


def bake_model(
    ply_path,
    preparation_dir,
    cuboids_dir,
    output_dir,
    atlas_size=2048,
    loaded=None,
    source_up="-y",
    units_per_meter=1.0,
    color_space="srgb",
):
    if not 256 <= atlas_size <= 8192:
        raise ValueError("atlas_size must be between 256 and 8192")
    started = time.perf_counter()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    parameters_path = Path(cuboids_dir) / "parameters.npz"
    original_hash = hashlib.sha256(parameters_path.read_bytes()).hexdigest()
    with np.load(parameters_path) as archive:
        params = {
            key: archive[key]
            for key in ("centers", "dimensions", "rotations", "corners", "surface_tolerance")
        }
    if not len(params["corners"]):
        raise ValueError("Cannot texture an empty cuboid model.")
    faces = faces_from_corners(params["corners"])
    from .cuboids import box_membership

    boxes = [
        dict(center=c, dimensions=d, rotation=r)
        for c, d, r in zip(params["centers"], params["dimensions"], params["rotations"])
    ]
    for face in faces:
        corners = params["corners"][face["box"]][face["indices"]]
        # A convex face wholly inside an earlier box is invisible in every prefix.
        face["hidden"] = any(
            box_membership(corners, box, tolerance=-min(face["width"], face["height"]) * 1e-9).all()
            for box in boxes[: face["box"]]
        )
    rects, density = pack_atlas(faces, atlas_size)
    print("Loading PLY colors and projection attributes...", flush=True)
    cloud = load_cloud(ply_path, preparation_dir, loaded=loaded, color_space=color_space)
    max_depth = 2.5 * float(params["surface_tolerance"])
    geometry_report_path = Path(cuboids_dir) / "cuboids_report.json"
    geometry_report = (
        json.loads(geometry_report_path.read_text(encoding="utf-8"))
        if geometry_report_path.exists()
        else {"approximation": {"distance_limit": 0}}
    )
    approximation = geometry_report["approximation"]
    backward_depth = max(max_depth, approximation["distance_limit"] + 2.5 * cloud["spacing"])
    atlas = np.zeros((atlas_size, atlas_size, 3), dtype=np.uint8)
    confidence = np.zeros_like(atlas)
    metadata = []
    pad = 4
    for i, (face, rect) in enumerate(zip(faces, rects)):
        x, y, w, h = rect
        if face["hidden"]:
            rgb = np.full((h, w, 3), 0.65)
            supported, depths = np.zeros((h, w), bool), np.full(h * w, np.nan)
        else:
            rgb, supported, depths = bake_face(face, w, h, cloud, max_depth, backward_depth)
        tile = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        quality = np.where(
            supported[:, :, None], np.array([51, 188, 147]), np.array([240, 145, 62])
        ).astype(np.uint8)
        atlas[y - pad : y + h + pad, x - pad : x + w + pad] = np.pad(
            tile, ((pad, pad), (pad, pad), (0, 0)), mode="edge"
        )
        confidence[y - pad : y + h + pad, x - pad : x + w + pad] = np.pad(
            quality, ((pad, pad), (pad, pad), (0, 0)), mode="edge"
        )
        metadata.append(
            dict(
                box=face["box"] + 1,
                side=face["side"],
                rect=list(rect),
                corner_indices=face["indices"].tolist(),
                area=face["width"] * face["height"],
                hidden_in_all_prefixes=face["hidden"],
                projected_fraction=float(supported.mean()),
                median_projection_depth=float(np.nanmedian(depths)) if supported.any() else None,
            )
        )
        if (i + 1) % 12 == 0:
            print(f"Baked {i + 1}/{len(faces)} faces", flush=True)
    Image.fromarray(atlas).save(out / "texture_atlas.png")
    Image.fromarray(confidence).save(out / "projection_confidence.png")
    export_obj(out, params, faces, rects, atlas_size)
    export_glb(out, params, faces, rects, atlas_size, source_up, units_per_meter)
    if hashlib.sha256(parameters_path.read_bytes()).hexdigest() != original_hash:
        raise RuntimeError("Cuboid parameters changed during texture baking.")
    report = dict(
        cuboids=len(params["centers"]),
        faces=len(faces),
        triangles=12 * len(params["centers"]),
        atlas_size=[atlas_size] * 2,
        pixels_per_world_unit=density,
        retained_points=len(cloud["points"]),
        median_spacing=cloud["spacing"],
        color_source=cloud["color_source"],
        source_color_space=color_space,
        blending_color_space="linear",
        atlas_color_space="srgb",
        anisotropic_splats=cloud["anisotropic_splats"],
        max_projection_depth=max_depth,
        max_backward_projection_depth=backward_depth,
        supported_area_fraction=sum(
            f["area"] * f["projected_fraction"] for f in metadata if not f["hidden_in_all_prefixes"]
        )
        / sum(f["area"] for f in metadata if not f["hidden_in_all_prefixes"]),
        hidden_faces_skipped=sum(f["hidden_in_all_prefixes"] for f in metadata),
        support_definition="Unambiguous local projection; excludes faces proven hidden in every construction prefix",
        source_up=source_up,
        source_units_per_meter=units_per_meter,
        method="Nearest supported layer on either side of the cuboid surface"
        + "; anisotropic Gaussian RGB splats weighted by opacity and normal alignment",
        fallback="Distance-bounded local RGB interpolation, then neutral gray; orange also marks ambiguous layers",
        geometry="Original cuboids unchanged. No new fit, vertex displacement, added boxes, or topology change.",
        cuboid_parameters_sha256=original_hash,
        geometry_unchanged=True,
        face_details=metadata,
        seconds=time.perf_counter() - started,
    )
    (out / "texture_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        out / "texture_parameters.npz",
        **{k: params[k] for k in ("centers", "dimensions", "rotations", "corners")},
        face_corner_indices=np.array([f["indices"] for f in faces]),
        atlas_rectangles=np.array(rects),
        atlas_size=atlas_size,
        face_supported_fraction=np.array([f["projected_fraction"] for f in metadata]),
    )
    export_viewer(
        out,
        params,
        faces,
        rects,
        atlas_size,
        report,
        Path(preparation_dir) / "common_geometry.npz",
        source_up,
    )
    print(
        f"Done: {len(faces)} faces; {report['supported_area_fraction']:.1%} of face area has projection support; {report['seconds']:.1f}s",
        flush=True,
    )
    return out, report
