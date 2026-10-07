"""Quality regression corpus through the CLI; --exploratory records unmet contracts."""

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from cuboid_approximation.cloud import write_points
from cuboid_approximation.pipeline import main as run_pipeline


@dataclass
class Case:
    name: str
    points: np.ndarray | None
    mode: str = "solid"
    tolerance: float = 0.06
    max_boxes: int = 16
    rectangles: list = field(default_factory=list)
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    empty_probes: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    detail_probes: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    colors: np.ndarray | None = None
    minimum_texture_support: float = 0.9
    expectation: str = "All source and exposed-surface quality targets must be met."


def shell(n=21):
    """Analytic cube surface: unrelated to production triangulation or sampling."""
    faces = []
    u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    for axis in range(3):
        for side in (0, 1):
            p = np.zeros((n * n, 3))
            p[:, axis] = side
            p[:, [i for i in range(3) if i != axis]] = np.column_stack((u.ravel(), v.ravel()))
            faces.append(p)
    return np.unique(np.vstack(faces), axis=0)


def plane(n=24, size=1.0, offset=(0, 0, 0)):
    u, v = np.meshgrid(np.linspace(0, size, n), np.linspace(0, size, n))
    return np.column_stack((u.ravel(), v.ravel(), np.zeros(u.size))) + offset


def union_rectangles(bounds):
    """Exact exposed rectangles of an axis-aligned analytic reference union.

    Split each face at all box boundary coordinates and keep only patches whose
    outward side is empty. This does not call the production mesh implementation.
    """
    bounds = np.asarray(bounds, dtype=float)
    rectangles = []
    for lower, upper in bounds:
        for axis in range(3):
            tangents = [d for d in range(3) if d != axis]
            cuts = [np.unique(np.clip(bounds[:, :, d], lower[d], upper[d])) for d in tangents]
            for side in (0, 1):
                coordinate = (lower, upper)[side][axis]
                for a, b in zip(cuts[0][:-1], cuts[0][1:]):
                    for c, d in zip(cuts[1][:-1], cuts[1][1:]):
                        lo, hi = lower.copy(), upper.copy()
                        lo[axis] = hi[axis] = coordinate
                        lo[tangents], hi[tangents] = [a, c], [b, d]
                        probe = (lo + hi) / 2
                        probe[axis] += (2 * side - 1) * 1e-7
                        inside = np.all((probe > bounds[:, 0]) & (probe < bounds[:, 1]), axis=1)
                        if not inside.any():
                            rectangles.append((lo, hi))
    return rectangles


def sample_rectangles(rectangles, step):
    points = []
    for lo, hi in rectangles:
        axes = [np.linspace(a, b, max(1, int(np.ceil((b - a) / step)) + 1)) for a, b in zip(lo, hi)]
        points.append(np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3))
    return np.unique(np.concatenate(points), axis=0)


def corpus():
    cube_rectangles = union_rectangles([[[0, 0, 0], [1, 1, 1]]])
    cube = shell()
    yield Case("cube", cube, max_boxes=1, rectangles=cube_rectangles)
    yield Case(
        "noisy_cube",
        cube + np.random.default_rng(42).normal(0, 0.003, cube.shape),
        max_boxes=1,
        rectangles=cube_rectangles,
    )
    rotation = Rotation.from_euler("xyz", [13, 29, 51], degrees=True).as_matrix()
    yield Case(
        "rotated_cube",
        cube @ rotation.T,
        max_boxes=1,
        rectangles=cube_rectangles,
        rotation=rotation,
    )
    box_rectangles = union_rectangles([[[0, 0, 0], [1.5, 0.5, 0.25]]])
    yield Case(
        "rotated_box",
        (cube * [1.5, 0.5, 0.25]) @ rotation.T,
        max_boxes=1,
        rectangles=box_rectangles,
        rotation=rotation,
    )
    for name, bounds, tolerance, details in (
        ("l_shape", [[[0, 0, 0], [1, 2, 1]], [[1, 0, 0], [2, 1, 1]]], 0.06, [[1.9, 0.5, 1]]),
        (
            "thin_appendage",
            [[[0, 0, 0], [1, 1, 1]], [[1, 0.45, 0.45], [1.8, 0.55, 0.55]]],
            0.02,
            [[1.8, 0.5, 0.5], [1.6, 0.45, 0.5]],
        ),
    ):
        rectangles = union_rectangles(bounds)
        yield Case(
            name,
            sample_rectangles(rectangles, tolerance),
            tolerance=tolerance,
            rectangles=rectangles,
            detail_probes=np.array(details),
        )
    yield Case(
        "hollow_shell",
        cube,
        mode="surface",
        rectangles=cube_rectangles,
        empty_probes=np.array([[0.5, 0.5, 0.5]]),
    )
    layers = np.vstack((plane(41), plane(41, offset=(0, 0, 0.08))))
    yield Case(
        "close_layers",
        layers,
        mode="surface",
        tolerance=0.025,
        rectangles=[(np.array([0, 0, z]), np.array([1, 1, z])) for z in (0, 0.08)],
        empty_probes=plane(5, size=0.6, offset=(0.2, 0.2, 0.04)),
        colors=np.where((layers[:, 2] > 0.04)[:, None], [0.0, 0, 1], [1.0, 0, 0]),
    )
    yield Case(
        "mixed_density",
        np.vstack((plane(), plane(8, offset=(3, 0, 0)))),
        mode="surface",
        tolerance=0.11,
        rectangles=[(np.array([x, 0, 0]), np.array([x + 1, 1, 0])) for x in (0, 3)],
        detail_probes=np.array([[3.5, 0.5, 0]]),
    )
    yield Case(
        "extreme_density",
        np.vstack((plane(20, size=0.001), plane(8, offset=(5, 0, 0)))),
        mode="surface",
        tolerance=0.11,
        rectangles=[
            (np.zeros(3), np.array([0.001, 0.001, 0])),
            (np.array([5, 0, 0]), np.array([6, 1, 0])),
        ],
        detail_probes=np.array([[0.0005, 0.0005, 0], [5.5, 0.5, 0]]),
    )
    # A partial scan only constrains observed faces: no hidden volume is asserted.
    partial = np.vstack((plane(), plane()[:, [0, 2, 1]]))
    yield Case(
        "partial_corner",
        np.unique(partial, axis=0),
        mode="surface",
        rectangles=[(np.zeros(3), np.array([1, 1, 0])), (np.zeros(3), np.array([1, 0, 1]))],
        expectation="Match observed faces only; hidden object completion is not a reference.",
    )


def box_distances(points, parameters):
    """Independent point-to-volume formula; used only for semantic probes."""
    distances = np.full(len(points), np.inf)
    inside = np.zeros(len(points), bool)
    for center, dimensions, rotation in zip(
        parameters["centers"], parameters["dimensions"], parameters["rotations"]
    ):
        q = np.abs((points - center) @ rotation) - dimensions / 2
        distances = np.minimum(distances, np.linalg.norm(np.maximum(q, 0), axis=1))
        inside |= np.all(q < -1e-8, axis=1)
    return distances, inside


def analytic_checks(case, output):
    """Independent reference-distance, semantic-void, detail and atlas checks."""
    with np.load(output / "cuboids" / "parameters.npz") as archive:
        parameters = dict(archive)
    metrics = {}
    if len(case.empty_probes):
        _, inside = box_distances(case.empty_probes, parameters)
        metrics["occupied_empty_probes"] = int(inside.sum())
    if len(case.detail_probes):
        distances, _ = box_distances(case.detail_probes, parameters)
        metrics["maximum_detail_distance"] = float(distances.max())
    if case.rectangles:
        # Midpoint quadrature on every output face, with weights in world units.
        # Discard hidden samples using independent volume membership tests.
        reference, weights = [], []
        for index, (center, dimensions, rotation) in enumerate(
            zip(parameters["centers"], parameters["dimensions"], parameters["rotations"])
        ):
            for axis in range(3):
                tangent = [d for d in range(3) if d != axis]
                side = max(8, min(64, int(np.ceil(max(dimensions[tangent]) / case.tolerance))))
                uv = (np.arange(side) + 0.5) / side - 0.5
                u, v = np.meshgrid(uv, uv)
                for sign in (-1, 1):
                    local = np.zeros((side * side, 3))
                    local[:, axis] = sign * dimensions[axis] / 2
                    local[:, tangent] = (
                        np.column_stack((u.ravel(), v.ravel())) * dimensions[tangent]
                    )
                    points = local @ rotation.T + center
                    visible = np.ones(len(points), bool)
                    for other in range(len(parameters["centers"])):
                        if other != index:
                            q = np.abs(
                                (points - parameters["centers"][other])
                                @ parameters["rotations"][other]
                            )
                            visible &= ~np.all(
                                q < parameters["dimensions"][other] / 2 - 1e-8, axis=1
                            )
                    reference.append(points[visible] @ case.rotation)
                    weights.append(np.full(visible.sum(), np.prod(dimensions[tangent]) / side**2))
        points, weights = np.concatenate(reference), np.concatenate(weights)
        distances = np.full(len(points), np.inf)
        for lo, hi in case.rectangles:
            delta = np.maximum(np.maximum(lo - points, points - hi), 0)
            distances = np.minimum(distances, np.linalg.norm(delta, axis=1))
        metrics["analytic_supported_area_fraction"] = float(
            np.sum(weights[distances <= case.tolerance]) / weights.sum()
        )
        metrics["analytic_surface_rms"] = float(np.sqrt(np.average(distances**2, weights=weights)))
    if case.name == "close_layers":
        report = json.loads((output / "textured" / "texture_report.json").read_text())
        atlas = np.asarray(Image.open(output / "textured" / "texture_atlas.png"), dtype=float) / 255
        layer_errors = [[], []]
        for face in report["face_details"]:
            if face["hidden_in_all_prefixes"]:
                continue
            corners = parameters["corners"][face["box"] - 1][face["corner_indices"]]
            normal = np.cross(corners[1] - corners[0], corners[3] - corners[0])
            if abs(normal[2]) < 0.95 * np.linalg.norm(normal):
                continue
            layer = int(corners[:, 2].mean() > 0.04)
            x, y, w, h = face["rect"]
            # Central texels avoid expected interpolation uncertainty at sheet edges.
            tile = atlas[
                y + h // 4 : y + max(h // 4 + 1, 3 * h // 4),
                x + w // 4 : x + max(w // 4 + 1, 3 * w // 4),
            ]
            expected = np.array([1.0, 0, 0]) if layer == 0 else np.array([0.0, 0, 1])
            layer_errors[layer].extend(np.linalg.norm(tile - expected, axis=2).ravel().tolist())
        metrics["layer_color_p95"] = [
            float(np.quantile(values, 0.95)) if values else None for values in layer_errors
        ]
    return metrics


def quality_failures(case, code, report, metrics):
    """A partial export is evidence of failure, never a successful regression."""
    failures = []
    geometry = report.get("geometry", {})
    if code != 0:
        failures.append(f"pipeline exit {code}; expected all quality targets to be met")
    if not report.get("target_reached", geometry.get("target_reached", False)):
        failures.append("pipeline target_reached is false")
    if geometry.get("cuboids", 0) > case.max_boxes:
        failures.append(f"cuboid count exceeds analytic-case budget {case.max_boxes}")
    support = report.get("texture", {}).get("supported_area_fraction", 0)
    if not np.isfinite(support) or support < case.minimum_texture_support:
        failures.append(f"texture support {support:.4f} below {case.minimum_texture_support}")
    if metrics.get("occupied_empty_probes", 0):
        failures.append("model fills a required empty gap or hollow interior")
    if metrics.get("maximum_detail_distance", 0) > case.tolerance:
        failures.append("required thin feature or component is missing")
    if "analytic_supported_area_fraction" in metrics:
        value = metrics["analytic_supported_area_fraction"]
        if not np.isfinite(value) or value < 0.95:
            failures.append(f"independent analytic surface support {value:.4f} below 0.95")
    if "layer_color_p95" in metrics and any(
        v is None or not np.isfinite(v) or v > 0.15 for v in metrics["layer_color_p95"]
    ):
        failures.append(
            "close-layer red/blue atlas color error exceeds 0.15, or a layer is missing"
        )
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", help="Only these corpus cases; repeatable.")
    parser.add_argument(
        "--ply", type=Path, help="Benchmark an existing input instead of synthetic cases."
    )
    parser.add_argument("--resolution", type=int, default=48)
    parser.add_argument("--max-cuboids", type=int, default=16)
    parser.add_argument(
        "--surface-max-evaluations", type=int, default=262144,
        help="Adaptive area certificate work budget for the regression corpus.",
    )
    parser.add_argument(
        "--point-tolerance",
        type=float,
        help="Fixed source-unit tolerance, overriding the corpus contract; required with --ply.",
    )
    parser.add_argument(
        "--ablation", action="store_true", help="Compare regions, no barriers, and PCA."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--strict", action="store_true", help="Fail on every violated contract (default)."
    )
    mode.add_argument(
        "--exploratory",
        action="store_true",
        help="Record unmet quality contracts without failing; runtime errors still fail.",
    )
    args = parser.parse_args(argv)
    if args.ply and args.point_tolerance is None:
        parser.error("--ply requires --point-tolerance in physical source units")
    if args.point_tolerance is not None and (
        not np.isfinite(args.point_tolerance) or args.point_tolerance <= 0
    ):
        parser.error("--point-tolerance must be finite and positive")
    cases = (
        [Case("input", None, tolerance=args.point_tolerance, max_boxes=args.max_cuboids)]
        if args.ply
        else list(corpus())
    )
    known = {case.name for case in cases}
    if args.case and set(args.case) - known:
        parser.error(f"Unknown cases: {', '.join(sorted(set(args.case) - known))}")
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for case in cases:
        if args.case and case.name not in args.case:
            continue
        if args.point_tolerance is not None:
            case.tolerance = args.point_tolerance
        source = args.ply
        if source is None:
            source = args.output / (case.name + ".ply")
            colors = case.colors if case.colors is not None else np.full_like(case.points, 0.65)
            write_points(source, case.points, colors, np.arange(len(case.points)))
        variants = [("regions", ["--orientation-mode", "regions"])]
        if args.ablation:
            variants += [
                ("no_barriers", ["--orientation-mode", "regions", "--no-edge-barriers"]),
                ("pca", ["--orientation-mode", "pca"]),
            ]
        for variant, extra in variants:
            output = args.output / f"{case.name}-{variant}"
            command = [
                "--ply",
                str(source),
                "--output",
                str(output),
                "--resolution",
                str(args.resolution),
                "--max-cuboids",
                str(args.max_cuboids),
                "--max-points",
                "6000",
                "--max-frames",
                "8",
                "--seeds-per-frame",
                "8",
                "--atlas-size",
                "512",
                "--surface-max-evaluations",
                str(args.surface_max_evaluations),
                "--point-tolerance",
                str(case.tolerance),
                "--target-component-coverage",
                "0.999",
                "--reconstruction-mode",
                case.mode,
                *extra,
            ]
            code = run_pipeline(command)
            if code not in (0, 3):
                raise RuntimeError(
                    f"Benchmark execution failed: {case.name}/{variant}, exit {code}"
                )
            report = json.loads((output / "report.json").read_text())
            geometry = report.get("geometry", {})
            metrics = analytic_checks(case, output) if geometry.get("cuboids", 0) else {}
            failures = quality_failures(case, code, report, metrics)
            records.append(
                dict(
                    case=case.name,
                    variant=variant,
                    exit_code=code,
                    quality_passed=not failures,
                    failures=failures,
                    expectation=case.expectation,
                    physical_tolerance=case.tolerance,
                    analytic=metrics,
                    config=report["config"],
                    environment=report["environment"],
                    implementation_sha256=report["implementation_sha256"],
                    cuboids=geometry.get("cuboids", 0),
                    raw_coverage=geometry.get("full_cloud_proximity"),
                    spatial_coverage=geometry.get("approximation", {}).get("spatial_coverage"),
                    surface=geometry.get("approximation", {}).get("surface"),
                    texture_support=report.get("texture", {}).get("supported_area_fraction"),
                    seconds=report.get("seconds"),
                    stages=report.get("stages"),
                )
            )
            (args.output / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
            for failure in failures:
                print(f"QUALITY FAILURE {case.name}/{variant}: {failure}", flush=True)
    failed = sum(not record["quality_passed"] for record in records)
    print(
        f"Quality contracts: {len(records) - failed}/{len(records)} passed"
        + (" (exploratory; failures are recorded)" if args.exploratory else " (strict)")
    )
    return 1 if failed and not args.exploratory else 0


if __name__ == "__main__":
    raise SystemExit(main())
