"""Run the final method from a PLY file to textured cuboids."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import shutil
from time import perf_counter
import traceback

import numpy as np

from .approximation import (
    approximation_metrics,
    component_target_reached,
    surface_support_reached,
)
from .barriers import LineConfig, fit_edge_lines
from .cloud import (
    load_points,
    local_geometry,
    normal_variation,
    spatial_representatives,
    write_points,
)
from .cuboids import CuboidConfig, box_corners, distance_to_box, write_cuboid_mesh
from .fitting import fit_cuboids
from .regions import SurfaceConfig, region_colors, segment_surfaces
from .texture import bake_model
from .volume import CoverageNotReachedError, preflight_grid
from .checkpoint import (
    SearchCheckpoint, read_checkpoint, validate_final_geometry, sha256, write_json,
    retained_population, validate_retained_population,
)
from .provenance import compatibility_signature, implementation_provenance


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--ply", type=Path, required=True, help="XYZ, XYZ RGB or Gaussian-splat PLY.")
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory; existing directories are refused.",
    )
    p.add_argument(
        "--target-coverage",
        type=float,
        default=0.999,
        help="Required fraction of all retained source points within the proximity tolerance.",
    )
    p.add_argument(
        "--max-cuboids", type=int, default=128,
        help="Maximum total cuboid count, including the frozen prefix on resume.",
    )
    p.add_argument(
        "--max-points",
        type=int,
        default=60_000,
        help="Spatial sample budget for normals and fitting.",
    )
    p.add_argument(
        "--resolution", type=int, default=160, help="Voxel resolution on the longest source axis."
    )
    p.add_argument(
        "--normal-angle",
        type=float,
        default=30.0,
        help="Normal-variation edge threshold in degrees.",
    )
    p.add_argument(
        "--min-opacity",
        type=float,
        default=0.1,
        help="Minimum sigmoid opacity for Gaussian splats.",
    )
    p.add_argument(
        "--approximation-distance-factor",
        type=float,
        default=2.0,
        help="Envelope radius in max(voxel size, source spacing) units.",
    )
    p.add_argument(
        "--point-tolerance-factor",
        type=float,
        default=2.0,
        help="Point tolerance in full-cloud distinct-point spacing units, independent of grid resolution.",
    )
    p.add_argument(
        "--exterior-penalty",
        type=float,
        default=2.0,
        help="Penalty for estimated cuboid volume outside observed/inferred support.",
    )
    p.add_argument("--max-frames", type=int, default=32, help="Orientation search budget.")
    p.add_argument(
        "--seeds-per-frame", type=int, default=96, help="Spatial seed budget per orientation."
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for barrier and cuboid proposals."
    )
    p.add_argument(
        "--atlas-size", type=int, default=2048, help="Square texture atlas width in pixels."
    )
    p.add_argument(
        "--point-tolerance", type=float, help="Override proximity tolerance in source units."
    )
    p.add_argument(
        "--reconstruction-mode",
        choices=("solid", "surface"),
        default="solid",
        help="Fill enclosed interiors, or preserve surface/shell topology without filling.",
    )
    p.add_argument(
        "--max-memory-mb",
        type=int,
        default=2048,
        help="Estimated dense-grid workspace budget in MiB; checked before allocation.",
    )
    p.add_argument(
        "--outlier-distance-factor",
        type=float,
        default=50.0,
        help="Reject extreme third-neighbor distances relative to the median of nearby points; 0 disables.",
    )
    p.add_argument(
        "--source-up",
        choices=("x", "y", "z", "-x", "-y", "-z"),
        default="-y",
        help="Source up axis for the viewer and glTF conversion (e.g. --source-up=-y).",
    )
    p.add_argument(
        "--units-per-meter",
        type=float,
        default=1.0,
        help="Source units per meter, used only for the glTF export.",
    )
    p.add_argument(
        "--resume",
        type=Path,
        help="Previous compatible output directory; append cuboids into a NEW output.",
    )
    p.add_argument(
        "--rebake",
        type=Path,
        help="Previous output with complete geometry; recompute textures/exports only into a NEW output.",
    )
    p.add_argument(
        "--target-surface-support",
        type=float,
        default=0.95,
        help="Required supported exposed area; 0 explicitly disables the surface quality gate.",
    )
    p.add_argument(
        "--color-space",
        choices=("srgb", "linear"),
        default="srgb",
        help="Encoding of input RGB/SH-DC colors; blending is always linear, atlas is sRGB.",
    )
    p.add_argument(
        "--surface-max-evaluations",
        type=int,
        default=65536,
        help="Adaptive exposed-area support budget; exhausted uncertain bounds do not pass.",
    )
    p.add_argument(
        "--target-component-coverage",
        type=float,
        default=0.0,
        help="Minimum coverage within every spatial component; 0 disables this extra gate.",
    )
    p.add_argument(
        "--orientation-mode",
        choices=("regions", "pca"),
        default="pca",
        help="Support-plane/PCA proposals, with additional region proposals in regions mode.",
    )
    p.add_argument(
        "--edge-barriers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fitted edge segments when proposing region orientations.",
    )
    p.add_argument(
        "--diagnostics",
        action="store_true",
        help="Compute edge diagnostics and export preparation PLYs and detailed barrier data.",
    )
    return p


def prepare(source, output, args, config, previous=None):
    diagnostics = getattr(args, "diagnostics", False)

    def world_barriers(barriers, scale, offset):
        barriers["segments"] = barriers["segments"] * scale + offset
        for key in (
            "lengths", "rmse", "max_support_gaps", "distance_tolerance", "max_gap",
            "min_length", "tangent_radius",
        ):
            if key in barriers:
                barriers[key] = barriers[key] * scale
        return barriers

    def export_diagnostics(common, regions, scores, valid, mask, barriers=None):
        source_ids, points = common["source_indices"], common["points"]
        np.savez_compressed(
            output / "edges.npz", scores=scores, valid=valid, mask=mask, source_indices=source_ids
        )
        write_points(output / "source_sample.ply", points, common["colors"], source_ids)
        write_points(output / "regions.ply", points, regions["colors"], source_ids)
        write_points(
            output / "edges.ply", points[mask],
            np.tile([1.0, 0.15, 0.2], (mask.sum(), 1)), source_ids[mask],
        )
        if barriers is not None and "support_point_indices" in barriers:
            np.savez_compressed(output / "barriers.npz", **barriers)

    loaded = load_points(source, args.min_opacity, args.outlier_distance_factor)
    if not len(loaded["points"]):
        raise ValueError("No retained points.")
    preflight_grid(loaded["points"], config, loaded["spacing"], loaded["splats"])
    if previous is not None:
        for name in ("common_geometry.npz", "regions.npz", "report.json"):
            shutil.copy2(previous / "preparation" / name, output / name)
        with np.load(output / "regions.npz") as data:
            regions = dict(data)
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        report["reused_preparation"] = True
        report["diagnostics_enabled"] = diagnostics
        if diagnostics:
            with np.load(output / "common_geometry.npz") as data:
                common = dict(data)
            edge_path = previous / "preparation/edges.npz"
            if edge_path.is_file():
                with np.load(edge_path) as edges:
                    scores, valid, mask = edges["scores"], edges["valid"], edges["mask"]
            else:
                scale = float(np.ptp(loaded["points"], axis=0).max())
                local = (common["points"] - loaded["points"].min(0)) / scale
                scores, valid, _ = normal_variation(local_geometry(local))
                mask = valid & (scores >= args.normal_angle)
            export_diagnostics(common, regions, scores, valid, mask)
            barrier_path = previous / "preparation/barriers.npz"
            if barrier_path.is_file():
                shutil.copy2(barrier_path, output / barrier_path.name)
            elif args.max_frames > 1 and args.orientation_mode == "regions" and args.edge_barriers:
                scale = float(np.ptp(loaded["points"], axis=0).max())
                offset = loaded["points"].min(0)
                barriers = fit_edge_lines(
                    (common["points"] - offset) / scale, scores, mask,
                    report["median_spacing"] / scale, LineConfig(seed=args.seed), diagnostics=True,
                )
                np.savez_compressed(
                    output / "barriers.npz", **world_barriers(barriers, scale, offset)
                )
            report["edge_detection_computed"] = True
            report["edge_points"] = int(mask.sum())
        write_json(output / "report.json", report)
        return loaded, regions, report
    selected, sample_voxel = spatial_representatives(
        loaded["points"], args.max_points, loaded["distinct_indices"]
    )
    points = loaded["points"][selected]
    offset = loaded["points"].min(0)
    scale = float(np.ptp(loaded["points"], axis=0).max())
    local_points = (points - offset) / scale
    geometry = local_geometry(local_points)
    use_regions = args.max_frames > 1 and args.orientation_mode == "regions"
    if use_regions or diagnostics:
        scores, valid, _ = normal_variation(geometry)
        mask = valid & (scores >= args.normal_angle)
    else:
        scores, valid = None, None
        mask = np.zeros(len(points), bool)
    source_ids = loaded["source_indices"][selected]
    common = dict(
        points=points,
        colors=loaded["colors"][selected],
        normals=geometry["normals"],
        normal_valid=geometry["valid"],
        spacing_per_point=geometry["spacing_per_point"] * scale,
        source_indices=source_ids,
    )
    np.savez_compressed(output / "common_geometry.npz", **common)
    local_spacing = geometry["spacing"]
    spacing = local_spacing * scale
    barriers = dict(segments=np.empty((0, 2, 3)))
    if use_regions and args.edge_barriers:
        barriers = fit_edge_lines(
            local_points, scores, mask, local_spacing, LineConfig(seed=args.seed),
            diagnostics=diagnostics,
        )
    if use_regions:
        result = segment_surfaces(
            local_points,
            geometry["normals"],
            geometry["valid"],
            mask,
            barriers["segments"],
            local_spacing,
            spacing_per_point=geometry["spacing_per_point"],
        )
    else:
        result = dict(
            labels=np.full(len(points), -1, dtype=int),
            assignment_kind=np.zeros(len(points), dtype=np.uint8),
        )
    barriers = world_barriers(barriers, scale, offset)
    colors, palette = region_colors(result["labels"])
    regions = dict(
        points=points,
        normals=geometry["normals"],
        source_indices=source_ids,
        colors=colors,
        labels=result["labels"],
        assignment_kind=result["assignment_kind"],
        edge_mask=mask,
        segments=barriers["segments"],
    )
    np.savez_compressed(output / "regions.npz", **regions)
    if diagnostics:
        export_diagnostics(common, regions, scores, valid, mask, barriers)
    report = dict(
        source_points=loaded["source_count"],
        retained_points=len(loaded["points"]),
        sampled_points=len(points),
        median_spacing=spacing,
        full_cloud_median_spacing=loaded["spacing"],
        outlier_distance_factor=args.outlier_distance_factor,
        rejected_outlier_count=len(loaded["outlier_source_indices"]),
        rejected_outlier_source_indices=loaded["outlier_source_indices"].tolist(),
        invalid_sample_normals=int((~geometry["valid"]).sum()),
        sample_voxel_size=sample_voxel,
        gaussian_splats=loaded["gaussian_splats"],
        orientation_mode=args.orientation_mode,
        edge_barriers_enabled=use_regions and args.edge_barriers,
        edge_detection_computed=use_regions or diagnostics,
        diagnostics_enabled=diagnostics,
        min_opacity=args.min_opacity,
        edge_points=int(mask.sum()) if use_regions or diagnostics else None,
        supported_barriers=len(barriers["segments"]),
        regions=len(palette),
        normal_neighbors=30,
        normal_radius_factor=6.0,
        normal_angle_degrees=args.normal_angle,
        line_config=asdict(LineConfig(seed=args.seed)),
        region_config=asdict(SurfaceConfig()),
    )
    write_json(output / "report.json", report)
    return loaded, regions, report


def save_geometry(output, result, loaded, regions, config):
    boxes, solid = result["boxes"], result["solid"]
    colors, _ = region_colors(np.arange(len(boxes)))
    full_distances = result["full_distances"]
    full_curve = {row["cuboids"]: row["covered_fraction"] for row in result["coverage_curve"]}
    prefix_distances = np.full(len(loaded["points"]), np.inf)
    sample_distances = np.full(len(regions["points"]), np.inf)
    curve = [dict(cuboids=0, sampled_proximity=0.0, full_cloud_proximity=0.0)]
    for i, box in enumerate(boxes, 1):
        if i not in full_curve:
            prefix_distances = np.minimum(prefix_distances, distance_to_box(loaded["points"], box))
            full_curve[i] = float(np.mean(prefix_distances <= result["surface_tolerance"]))
        sample_distances = np.minimum(sample_distances, distance_to_box(regions["points"], box))
        curve.append(
            dict(
                cuboids=i,
                sampled_proximity=float(np.mean(sample_distances <= result["surface_tolerance"])),
                full_cloud_proximity=full_curve[i],
            )
        )
    arrays = dict(
        centers=np.array([b["center"] for b in boxes]),
        dimensions=np.array([b["dimensions"] for b in boxes]),
        rotations=np.array([b["rotation"] for b in boxes]),
        corners=np.array([box_corners(b) for b in boxes]),
        colors=colors,
    )
    np.savez_compressed(
        output / "parameters.npz",
        **arrays,
        surface_tolerance=result["surface_tolerance"],
        sampled_distances=sample_distances,
        full_distances=full_distances,
        frames=result["frames"],
        cuboid_ids=np.arange(len(boxes)),
    )

    np.savez_compressed(
        output / "envelope.npz",
        **{k: solid[k] for k in ("safe", "reference", "conservative", "origin", "voxel_size")},
    )
    write_cuboid_mesh(output, boxes, colors)
    metrics = approximation_metrics(
        boxes, solid, loaded["points"], full_distances, result["surface_tolerance"],
        target_support=config.target_surface_support,
        support_max_evaluations=config.surface_max_evaluations,
    )
    volume_reached = int(np.count_nonzero(full_distances <= result["surface_tolerance"])) >= int(
        np.ceil(config.target_coverage * len(full_distances))
    )
    spatial_reached = metrics["spatial_coverage"] >= config.target_coverage - 1e-12
    surface_reached = config.target_surface_support == 0 or (
        metrics["surface"]["source_surface_coverage"] >= config.target_coverage - 1e-12
        and surface_support_reached(metrics["surface"], config.target_surface_support)
        and not result["selection"].get("frozen_prefix_violates_surface_thickness", False)
    )
    component_reached = component_target_reached(
        metrics["component_spatial_coverage"], config.target_component_coverage
    ) and (
        config.target_surface_support == 0
        or component_target_reached(
            metrics["surface"]["component_surface_coverage"], config.target_component_coverage
        )
    )
    target_reached = bool(volume_reached and spatial_reached and surface_reached and component_reached)
    if (
        volume_reached
        and spatial_reached
        and not surface_reached
        and result["selection"]["stopping_reason"] != "irreparable_frozen_prefix"
    ):
        result["selection"]["stopping_reason"] = "surface_quality_not_reached"
    elif volume_reached and spatial_reached and surface_reached and not component_reached:
        result["selection"]["stopping_reason"] = "component_quality_not_reached"
    report = dict(
        cuboids=len(boxes),
        target_coverage=config.target_coverage,
        target_reached=target_reached,
        volume_target_reached=bool(volume_reached),
        spatial_target_reached=bool(spatial_reached),
        surface_target_reached=bool(surface_reached),
        component_target_reached=bool(component_reached),
        target_component_coverage=config.target_component_coverage,
        sampled_proximity=curve[-1]["sampled_proximity"],
        full_cloud_proximity=curve[-1]["full_cloud_proximity"],
        surface_tolerance=result["surface_tolerance"],
        coverage_population="all retained source points, including duplicate observations",
        approximation=metrics,
        excluded_envelope_voxel_intersections=0,
        candidate_count=result["candidate_count"],
        candidate_search=solid.get("candidate_search", {"method": "reused or single-box candidates"}),
        orientations=len(result["frames"]),
        selection=result["selection"],
        incremental_steps=curve,
        reconstruction=solid["reconstruction"],
        seconds=result["seconds"],
        parameters_sha256=sha256(output / "parameters.npz"),
    )
    write_json(output / "cuboids_report.json", report)
    return report


def fitting_config(args):
    """All CLI-to-fitting settings live inside the construction signature."""
    config = CuboidConfig(
        resolution=args.resolution,
        max_cuboids=args.max_cuboids,
        target_coverage=args.target_coverage,
        target_surface_support=args.target_surface_support,
        target_component_coverage=args.target_component_coverage,
        surface_max_evaluations=args.surface_max_evaluations,
        max_frames=args.max_frames,
        seeds_per_frame=args.seeds_per_frame,
        seed=args.seed,
        approximation_distance_factor=args.approximation_distance_factor,
        approximation_detail_factor=args.point_tolerance_factor,
        approximation_exterior_penalty=args.exterior_penalty,
        point_tolerance=args.point_tolerance,
        reconstruction_mode=args.reconstruction_mode,
        max_memory_mb=args.max_memory_mb,
    )
    config.validate()
    return config


def fit_prepared(loaded, regions, preparation_report, config, checkpoint, resumed=None):
    """Explicit geometry-stage entrypoint, independent of mesh/texture exports."""
    return fit_cuboids(
        loaded["points"],
        regions,
        preparation_report["median_spacing"],
        config,
        splats=loaded["splats"],
        full_spacing=loaded["spacing"],
        checkpoint=checkpoint,
        **(resumed or {}),
    )


def run(args):
    source, output = args.ply.expanduser().resolve(), args.output.expanduser().resolve()
    config = fitting_config(args)
    if not source.is_file() or source.suffix.lower() != ".ply":
        raise ValueError(f"PLY file not found: {source}")
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    if args.resume is not None and args.rebake is not None:
        raise ValueError("Choose either --resume or --rebake.")
    if not 0 <= args.min_opacity <= 1 or not 0 < args.normal_angle < 90:
        raise ValueError("Opacity must be in [0, 1] and the normal angle in (0, 90).")
    if args.max_points < 20 or not 256 <= args.atlas_size <= 8192:
        raise ValueError("At least 20 sampled points and an atlas size in [256, 8192] are required.")
    if not np.isfinite(args.units_per_meter) or args.units_per_meter <= 0:
        raise ValueError("units-per-meter must be finite and positive.")
    if not np.isfinite(args.outlier_distance_factor) or args.outlier_distance_factor < 0:
        raise ValueError("outlier-distance-factor must be finite and nonnegative.")
    source_hash = sha256(source)
    provenance = implementation_provenance()
    signature = compatibility_signature(args, source_hash, provenance)
    resumed, previous, previous_report, rebake_loaded = None, None, None, None
    previous_option = args.rebake if args.rebake is not None else args.resume
    if previous_option is not None:
        previous = previous_option.expanduser().resolve()
        previous_report = json.loads((previous / "report.json").read_text(encoding="utf-8"))
        if previous_report.get("input_sha256") != source_hash:
            raise ValueError("Resume/rebake requires the identical source PLY input.")
        resumed, recorded_origin, recorded_scale = read_checkpoint(
            previous, None if args.rebake else signature, config.max_memory_mb,
            source_hash=source_hash, load_caches=args.rebake is None,
        )
        previous_geometry = validate_final_geometry(
            previous, previous_report, resumed, required=args.rebake is not None
        )
        if args.rebake is None and len(resumed["reference"]) > config.max_cuboids:
            raise ValueError("The resumed prefix exceeds --max-cuboids.")
        if args.rebake is not None:
            prepared_report = json.loads((previous / "preparation/report.json").read_text(encoding="utf-8"))
            # Use the recorded retained population; export settings may change,
            # but parser defaults must not alter the original filtering choices.
            rebake_loaded = load_points(
                source, prepared_report["min_opacity"],
                prepared_report.get("outlier_distance_factor", 50.0),
            )
            validate_retained_population(previous, rebake_loaded)
            if (
                not len(rebake_loaded["points"])
                or not np.array_equal(rebake_loaded["points"].min(0), recorded_origin)
                or float(np.ptp(rebake_loaded["points"], axis=0).max()) != recorded_scale
            ):
                raise ValueError("Checkpoint normalization does not match the retained source.")
    configuration = {
        k: v for k, v in vars(args).items() if k not in ("ply", "output", "resume", "rebake")
    }
    if args.rebake is not None:
        configuration = dict(previous_report["config"])
        for key in ("atlas_size", "source_up", "units_per_meter", "color_space"):
            configuration[key] = getattr(args, key)
    output.mkdir(parents=True)
    state = dict(
        provenance,
        status="running",
        input=source.name,
        input_sha256=source_hash,
        started_utc=datetime.now(timezone.utc).isoformat(),
        config=configuration,
        resumed_from=str(args.resume) if args.resume else None,
        rebaked_from=str(args.rebake) if args.rebake else None,
        cuboid_config=(previous_report["cuboid_config"] if args.rebake else asdict(config)),
        compatibility=signature,
        stages=[],
    )
    if args.rebake is not None:
        manifest = json.loads((previous / "checkpoint.json").read_text(encoding="utf-8"))
        state["compatibility"] = manifest["signature"]
        state["geometry_provenance"] = previous_report.get(
            "geometry_provenance", {k: previous_report[k] for k in provenance}
        )
    start = perf_counter()

    def save():
        write_json(output / "report.json", state)

    def stage(name, action):
        print(f"[{len(state['stages']) + 1}/3] {name}", flush=True)
        record = dict(name=name, status="running")
        state["stages"].append(record)
        save()
        began = perf_counter()
        value = action()
        record.update(status="complete", seconds=perf_counter() - began)
        save()
        return value

    with (
        (output / "run.log").open("w", encoding="utf-8", buffering=1) as log,
        redirect_stdout(Tee(sys.stdout, log)),
    ):
        try:
            preparation, cuboids = output / "preparation", output / "cuboids"
            preparation.mkdir()
            cuboids.mkdir()
            if args.rebake is not None:
                def reuse_preparation():
                    shutil.copytree(previous / "preparation", preparation, dirs_exist_ok=True)
                    return rebake_loaded, None, dict(prepared_report, reused_preparation=True)

                def reuse_geometry():
                    shutil.copytree(previous / "cuboids", cuboids, dirs_exist_ok=True)
                    shutil.copy2(previous / "checkpoint.json", output / "checkpoint.json")
                    return previous_geometry

                loaded, regions, preparation_report = stage(
                    "Reuse verified source preparation", reuse_preparation
                )
                state["preparation"] = preparation_report
                geometry_report = stage("Reuse immutable cuboid geometry", reuse_geometry)
            else:
                loaded, regions, preparation_report = stage(
                    "Load PLY, estimate normals and segment regions",
                    lambda: prepare(source, preparation, args, config, previous),
                )
                state["preparation"] = preparation_report
                if resumed is not None:
                    validate_retained_population(previous, loaded)
                    if (
                        not np.array_equal(loaded["points"].min(0), recorded_origin)
                        or float(np.ptp(loaded["points"], axis=0).max()) != recorded_scale
                    ):
                        raise ValueError("Checkpoint normalization does not match the retained source.")
                checkpoint = SearchCheckpoint(
                    output, signature, provenance=provenance, population=retained_population(loaded)
                )

                def construct():
                    result = fit_prepared(
                        loaded, regions, preparation_report, config, checkpoint, resumed
                    )
                    return save_geometry(cuboids, result, loaded, regions, config)

                geometry_report = stage("Append and certify cuboids", construct)
            state["geometry"] = geometry_report
            state["retained_population"] = retained_population(loaded)
            _, texture_report = stage(
                "Project colors and export the model",
                lambda: bake_model(
                    source,
                    preparation,
                    cuboids,
                    output / "textured",
                    args.atlas_size,
                    loaded=loaded,
                    source_up=args.source_up,
                    units_per_meter=args.units_per_meter,
                    color_space=args.color_space,
                ),
            )
            if sha256(source) != source_hash:
                raise RuntimeError("The source PLY changed during processing.")
            with (
                np.load(cuboids / "parameters.npz") as before,
                np.load(output / "textured" / "texture_parameters.npz") as after,
            ):
                if any(
                    not np.array_equal(before[k], after[k])
                    for k in ("centers", "dimensions", "rotations", "corners")
                ):
                    raise RuntimeError("Texture export changed the cuboid geometry.")
            if sha256(cuboids / "parameters.npz") != geometry_report["parameters_sha256"]:
                raise RuntimeError("Final cuboid parameters changed during export.")
            state.update(
                status="complete" if geometry_report["target_reached"] else "target_not_reached",
                target_reached=geometry_report["target_reached"],
                texture=texture_report,
                source_unchanged=True,
                texture_geometry_unchanged=True,
                seconds=perf_counter() - start,
            )
            save()
            print(
                f"Result: {geometry_report['cuboids']} cuboids; {geometry_report['full_cloud_proximity']:.4%} full-cloud proximity",
                flush=True,
            )
            print(f"Viewer: {output / 'textured' / 'cuboids_textured_3d.html'}", flush=True)
            if not geometry_report["target_reached"]:
                print(
                    f"Target {geometry_report['target_coverage']:.3%} not reached ({geometry_report['selection']['stopping_reason']}); partial model and textures exported.",
                    flush=True,
                )
            return 0 if geometry_report["target_reached"] else 3
        except (Exception, KeyboardInterrupt) as exc:
            state.update(
                status="target_not_reached"
                if isinstance(exc, CoverageNotReachedError)
                else "failed",
                target_reached=False,
                error=str(exc),
                seconds=perf_counter() - start,
            )
            if state["stages"]:
                state["stages"][-1].update(status=state["status"])
            if isinstance(exc, CoverageNotReachedError):
                state["error_details"] = exc.details
            save()
            traceback.print_exc(file=log)
            raise


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return run(args)
    except CoverageNotReachedError as exc:
        print(f"Target not reached: {exc}", file=sys.stderr)
        return 3
    except (OSError, ValueError, RuntimeError, KeyError, IndexError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
