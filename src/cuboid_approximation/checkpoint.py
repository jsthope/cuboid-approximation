"""Atomic, versioned search checkpoints, including preparation and envelope caches."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def pack_boxes(boxes, prefix):
    return {
        prefix + k: np.asarray([b[k] for b in boxes], dtype=float).reshape((-1, *shape))
        for k, shape in (("center", (3,)), ("dimensions", (3,)), ("rotation", (3, 3)))
    }


def unpack_boxes(data, prefix):
    centers, dimensions, rotations = (
        np.asarray(data[prefix + k]) for k in ("center", "dimensions", "rotation")
    )
    if (
        centers.ndim != 2
        or centers.shape[1:] != (3,)
        or dimensions.shape != centers.shape
        or rotations.shape != (len(centers), 3, 3)
        or not all(np.isfinite(a).all() for a in (centers, dimensions, rotations))
        or np.any(dimensions <= 0)
        or not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-8, rtol=0)
        or not np.allclose(np.linalg.det(rotations), 1, atol=1e-8, rtol=0)
    ):
        raise ValueError("Invalid cuboid arrays in resume checkpoint.")
    return [
        dict(center=c.copy(), dimensions=d.copy(), rotation=r.copy(), volume=float(np.prod(d)))
        for c, d, r in zip(centers, dimensions, rotations)
    ]


class SearchCheckpoint:
    def __init__(self, root, signature, provenance=None, population=None):
        self.root = Path(root)
        self.signature = signature
        self.provenance = provenance
        self.population = population
        self.base = None
        self.sequence = 0

    def __call__(self, boxes, candidates, solid, frames, origin, scale):
        if self.base is None:
            path = self.root / "cuboids/envelope-local.npz"
            atomic_npz(
                path,
                **{
                    k: solid[k]
                    for k in ("safe", "reference", "conservative", "origin", "voxel_size")
                },
            )
            preparation = {
                str(p.relative_to(self.root)): sha256(p)
                for p in (self.root / "preparation").iterdir()
                if p.name in {"common_geometry.npz", "regions.npz", "report.json"}
            }
            self.base = dict(
                format_version=2,
                signature=self.signature,
                provenance=self.provenance,
                retained_population=self.population,
                preparation=preparation,
                envelope_sha256=sha256(path),
                approximation=solid["approximation"].copy(),
                reconstruction=solid["reconstruction"].copy(),
                components=solid["components"],
                grid_shape=list(solid["safe"].shape),
            )
        # Two slots keep the last committed state intact until the manifest switches.
        filename = f"cuboids/checkpoint-{self.sequence % 2}.npz"
        path = self.root / filename
        atomic_npz(
            path,
            **pack_boxes(boxes, "committed_"),
            **pack_boxes(candidates, "candidate_"),
            normalization_origin=origin,
            normalization_scale=scale,
            frames=frames,
        )
        write_json(
            self.root / "checkpoint.json",
            dict(
                self.base,
                state_file=filename,
                state_sha256=sha256(path),
                committed_count=len(boxes),
            ),
        )
        self.sequence += 1


def read_checkpoint(root, signature, max_memory_mb=2048, *, source_hash=None, load_caches=True):
    """Validate a canonical checkpoint, optionally without enforcing old fit code.

    A render-only rebake passes signature=None and an explicit source digest.
    It still verifies all artifacts and normalization, but need not load dense
    fitting caches or install the implementation that originally fitted them.
    """
    root = Path(root)
    manifest = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 2:
        raise ValueError("Unsupported checkpoint format; regenerate with version 0.4 or newer.")
    if signature is None:
        if source_hash is None or manifest["signature"].get("input_sha256") != source_hash:
            raise ValueError("Rebake requires the identical source PLY input.")
    elif manifest["signature"] != signature:
        raise ValueError(
            "Resume requires identical input, preparation/fitting code, dependencies and fitting settings."
        )
    # Full provenance remains auditable, even when current export code differs.
    # A rebake retains the original geometry provenance separately from its own.
    if manifest.get("provenance") is not None:
        report = json.loads((root / "report.json").read_text(encoding="utf-8"))
        recorded = report.get("geometry_provenance", report)
        if any(recorded.get(k) != v for k, v in manifest["provenance"].items()):
            raise ValueError("Recorded geometry provenance disagrees with the checkpoint.")
        if report.get("compatibility") != manifest["signature"]:
            raise ValueError("Recorded fitting compatibility disagrees with the checkpoint.")
    filename = manifest["state_file"]
    if filename not in ("cuboids/checkpoint-0.npz", "cuboids/checkpoint-1.npz"):
        raise ValueError("Invalid checkpoint state filename.")
    required = {
        "preparation/common_geometry.npz",
        "preparation/regions.npz",
        "preparation/report.json",
    }
    if not required.issubset(manifest["preparation"]):
        raise ValueError("Incomplete checkpoint preparation manifest.")
    expected = {
        filename: manifest["state_sha256"],
        "cuboids/envelope-local.npz": manifest["envelope_sha256"],
        **manifest["preparation"],
    }
    for name, digest in expected.items():
        path = (root / name).resolve()
        if root.resolve() not in path.parents or sha256(path) != digest:
            raise ValueError(f"Resume checkpoint integrity check failed: {name}")
    if load_caches:
        from .volume import check_grid_memory

        check_grid_memory(manifest["grid_shape"], max_memory_mb, "resumed envelope")
    with np.load(root / filename) as data:
        boxes = unpack_boxes(data, "committed_")
        origin, scale = data["normalization_origin"].copy(), float(data["normalization_scale"])
        if load_caches:
            candidates = unpack_boxes(data, "candidate_")
            frames = data["frames"].copy()
    if (
        origin.shape != (3,)
        or not np.isfinite(origin).all()
        or not np.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("Invalid checkpoint normalization.")
    world = [
        dict(
            b,
            center=b["center"] * scale + origin,
            dimensions=b["dimensions"] * scale,
            volume=b["volume"] * scale**3,
        )
        for b in boxes
    ]
    result = dict(reference=world, reference_local=boxes)
    if load_caches:
        with np.load(root / "cuboids/envelope-local.npz") as data:
            solid = dict(data)
        solid.update(
            approximation=manifest["approximation"],
            reconstruction=manifest["reconstruction"],
            components=manifest["components"],
        )
        result.update(cached_candidates=candidates, cached_solid=solid, cached_frames=frames)
    return result, origin, scale


def validate_final_geometry(root, report, resumed, *, required=False):
    """Tie immutable final parameters to the canonical normalized checkpoint."""
    root = Path(root)
    geometry = report.get("geometry", {})
    if "parameters_sha256" not in geometry:
        if required:
            raise ValueError("Rebake requires a completed geometric stage with final parameters.")
        return None
    path = root / "cuboids/parameters.npz"
    if sha256(path) != geometry["parameters_sha256"]:
        raise ValueError("Resume checkpoint integrity check failed: parameters.npz")
    recorded = json.loads((root / "cuboids/cuboids_report.json").read_text(encoding="utf-8"))
    if recorded != geometry:
        raise ValueError("Final geometry report disagrees with the recorded run.")
    from .geometry import CORNER_SIGNS

    with np.load(path) as data:
        for plural, singular in (
            ("centers", "center"),
            ("dimensions", "dimensions"),
            ("rotations", "rotation"),
        ):
            expected = np.array([b[singular] for b in resumed["reference"]])
            if not np.array_equal(data[plural], expected):
                raise ValueError("Final geometry disagrees with the normalized checkpoint.")
        corners = np.array(
            [
                (CORNER_SIGNS * (b["dimensions"] / 2)) @ b["rotation"].T + b["center"]
                for b in resumed["reference"]
            ]
        )
        if not np.array_equal(data["corners"], corners):
            raise ValueError("Final geometry corners disagree with the normalized checkpoint.")
        tolerance = float(data["surface_tolerance"])
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Invalid surface tolerance in final geometry.")
    return geometry


def retained_population(loaded):
    """Stable identity of the retained source rows, independent of machine int width."""
    indices = np.asarray(loaded["source_indices"], dtype="<i8")
    return dict(
        count=len(loaded["points"]),
        source_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
    )


def validate_retained_population(root, loaded):
    """Reject a changed filter population before reusing its prepared geometry."""
    root = Path(root)
    manifest = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
    report = json.loads((root / "preparation/report.json").read_text(encoding="utf-8"))
    population = retained_population(loaded)
    if population["count"] != report["retained_points"] or (
        manifest.get("retained_population") is not None
        and population != manifest["retained_population"]
    ):
        raise ValueError("Retained source population differs from the geometric checkpoint.")
    with np.load(root / "preparation/common_geometry.npz") as common:
        retained = loaded["source_indices"]
        ids = np.searchsorted(retained, common["source_indices"])
        if (
            np.any(ids >= len(retained))
            or not np.array_equal(retained[ids], common["source_indices"])
            or not np.array_equal(loaded["points"][ids], common["points"])
        ):
            raise ValueError("Prepared sample no longer matches the retained source rows.")
    return population
