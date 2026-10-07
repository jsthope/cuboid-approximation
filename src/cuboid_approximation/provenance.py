"""Full run provenance and stage-specific compatibility for cached fitting."""

import ast
import hashlib
from importlib.metadata import version
from pathlib import Path
import sys

from . import __version__
from .checkpoint import sha256


# These controls do not change already prepared data or frozen geometric state.
# Unknown future options conservatively belong to fitting until classified here.
RUNTIME_SETTINGS = {
    "ply",
    "output",
    "resume",
    "rebake",
    "diagnostics",
    "max_cuboids",
    "target_coverage",
    "target_surface_support",
    "target_component_coverage",
    "surface_max_evaluations",
    "atlas_size",
    "max_memory_mb",
    "source_up",
    "units_per_meter",
    "color_space",
    "workers",
}
PREPARATION_SETTINGS = {
    "max_points",
    "min_opacity",
    "normal_angle",
    "outlier_distance_factor",
    "orientation_mode",
    "edge_barriers",
    "seed",
}
EXPORT_MODULES = {"texture.py", "viewer_template.html", "__main__.py", "__init__.py"}


def implementation_provenance(package=None):
    package = Path(package) if package is not None else Path(__file__).parent
    return dict(
        version=__version__,
        implementation_sha256={
            p.name: sha256(p) for p in sorted(package.iterdir()) if p.suffix in (".py", ".html")
        },
        environment=dict(
            python=sys.version,
            dependencies={
                name: version(name)
                for name in ("numpy", "scipy", "plyfile", "Pillow", "threadpoolctl")
            },
        ),
    )


def pipeline_fingerprint(path, roots):
    """Hash stage entrypoints plus referenced module globals, including helpers.

    The orchestrator also contains export/UI code. Traversing referenced globals
    captures new helpers and constants without coupling fitting to those exports.
    Imported implementation bodies are covered by the module hashes below.
    """
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    definitions = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name):
                        definitions[name.id] = node
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                name = alias.asname or (
                    alias.name.split(".")[0] if isinstance(node, ast.Import) else alias.name
                )
                definitions[name] = (
                    ast.Import(names=[alias])
                    if isinstance(node, ast.Import)
                    else ast.ImportFrom(module=node.module, names=[alias], level=node.level)
                )
    selected, pending = {}, list(roots)
    for root in roots:
        if root not in definitions:
            raise ValueError(f"Missing pipeline stage entrypoint: {root}")
    while pending:
        name = pending.pop()
        if name in selected or name not in definitions:
            continue
        node = definitions[name]
        selected[name] = ast.dump(node, annotate_fields=True, include_attributes=False)
        pending.extend(
            item.id
            for item in ast.walk(node)
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
        )
    content = "\n".join(f"{name}:{selected[name]}" for name in sorted(selected))
    return hashlib.sha256(content.encode()).hexdigest()


def compatibility_signature(args, source_hash, provenance, package=None):
    package = Path(package) if package is not None else Path(__file__).parent
    implementation = provenance["implementation_sha256"]
    # All non-export modules are included conservatively: a future imported
    # geometry helper cannot silently escape the cache compatibility boundary.
    fitting_code = {
        name: digest
        for name, digest in implementation.items()
        if name.endswith(".py") and name not in EXPORT_MODULES | {"pipeline.py"}
    }
    fitting_code["pipeline:construction"] = pipeline_fingerprint(
        package / "pipeline.py", ("fitting_config", "fit_prepared")
    )
    preparation_code = {
        name: implementation[name]
        for name in ("cloud.py", "regions.py", "barriers.py", "volume.py", "geometry.py")
    }
    preparation_code["pipeline:preparation"] = pipeline_fingerprint(
        package / "pipeline.py", ("prepare",)
    )
    options = {k: v for k, v in vars(args).items() if k not in RUNTIME_SETTINGS}
    dependencies = provenance["environment"]["dependencies"]
    return dict(
        format_version=2,
        input_sha256=source_hash,
        python=list(sys.version_info[:2]),
        preparation=dict(
            implementation=preparation_code,
            dependencies={k: dependencies[k] for k in ("numpy", "scipy", "plyfile")},
            settings={k: v for k, v in options.items() if k in PREPARATION_SETTINGS},
        ),
        fitting=dict(
            implementation=fitting_code,
            dependencies={k: dependencies[k] for k in ("numpy", "scipy")},
            settings={k: v for k, v in options.items() if k not in PREPARATION_SETTINGS},
        ),
    )
