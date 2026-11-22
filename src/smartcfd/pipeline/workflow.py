"""Generate, evaluate, and export the two-dimensional NACA demonstration grid."""

import json
from contextlib import ExitStack
from importlib.resources import files
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory

import numpy as np

from smartcfd.core.io import write_plot3d
from smartcfd.evaluation.mesh_quality import quality
from smartcfd.geometry.airfoil import naca
from smartcfd.meshing.structured import PARAM_DIM, PARAM_NAMES, generate
from smartcfd.rl.agents.sac import SAC
from smartcfd.rl.env import OBSERVATION_DIM, SEED, run_case
from smartcfd.rl.trainer import dataset


def load_sac_policy(checkpoint: Path | None = None) -> SAC:
    """Load a local actor or the small benchmark actor included in the package."""
    actor = SAC(OBSERVATION_DIM, PARAM_DIM, np.random.default_rng(SEED))
    if checkpoint is None:
        resource = files("smartcfd.rl.agents").joinpath("data", "naca_2d_sac_actor.npz")
        with resource.open("rb") as stream:
            actor.load_actor(stream)
    else:
        actor.load_actor(checkpoint)
    return actor


def generate_naca_mesh(code: str, method: str = "sac", *, checkpoint: Path | None = None,
                       fine: bool = False) -> tuple[np.ndarray, dict]:
    """Return physical mesh coordinates and the chosen candidate's diagnostics."""
    case = naca(code)
    if method == "sac":
        actor = load_sac_policy(checkpoint)
    elif method == "fixed" and checkpoint is None:
        actor = None
    else:
        raise ValueError("Use method 'sac' or 'fixed'; a checkpoint requires 'sac'")
    selected = run_case(case, method, actor)
    mesh = generate(case, selected["parameters"], fine=fine)
    report = {
        "naca": case.name,
        "method": method,
        "selected_step": selected["best_step"],
        "quality_evaluations": selected["evaluations"],
        "normalized_parameters": dict(zip(PARAM_NAMES, selected["parameters"])),
        "metrics": quality(mesh),
        "coarse_selection_metrics": selected["metrics"],
    }
    return mesh, report


def export_naca_mesh(code: str, destination: Path, method: str = "sac", *,
                     checkpoint: Path | None = None, fine: bool = False) -> dict:
    """Write four positive-orientation PLOT3D blocks and their interface metadata."""
    if destination.suffix.lower() != ".x":
        raise ValueError("The PLOT3D output path must use the .x suffix")
    sidecar = destination.with_suffix(".interfaces.json")
    for output in (destination, sidecar):
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}")
    mesh, report = generate_naca_mesh(code, method, checkpoint=checkpoint, fine=fine)
    if report["metrics"]["invalid_cells"]:
        raise ValueError(f"NACA {code} grid has {report['metrics']['invalid_cells']} invalid cells")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as staging:
        staged_grid = Path(staging) / destination.name
        ni, nj, nk = write_plot3d(staged_grid, mesh)
        blocks = ("upper aft", "upper forward", "lower forward", "lower aft")
        report.update({
            "format": "PLOT3D ASCII multizone XYZ (no IBLANK)",
            "grid_file": destination.name,
            "index_base": 1,
            "blocks": [{"id": i, "section": section, "dimensions": [ni, nj, nk]}
                       for i, section in enumerate(blocks, 1)],
            "boundary_faces": {"farfield_j": 1, "airfoil_j": nj},
            "interfaces": [{"from_block": i, "from_i": ni, "to_block": i % 4 + 1, "to_i": 1,
                            "j_order": "same", "k": 1} for i in range(1, 5)],
        })
        staged_sidecar = Path(staging) / sidecar.name
        staged_sidecar.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        # Use exclusive creation so a concurrent operation cannot replace a run.
        with ExitStack() as rollback:
            for staged, output in ((staged_grid, destination), (staged_sidecar, sidecar)):
                with staged.open("rb") as source, output.open("xb") as target:
                    rollback.callback(output.unlink)
                    copyfileobj(source, target)
            rollback.pop_all()
    return report


def evaluate_naca_holdout(checkpoint: Path | None = None) -> dict:
    """Evaluate fixed and learned policies on the frozen analytic NACA split."""
    actor = load_sac_policy(checkpoint)
    methods = ("fixed", "random", "greedy", "sac")
    cases = []
    for case in dataset()[2]:
        measures = {}
        for method in methods:
            result = run_case(case, method, actor if method == "sac" else None)
            result["fine_metrics"] = quality(generate(case, result["parameters"], fine=True))
            measures[method] = result
        cases.append({"naca": case.name, "methods": measures})
    summary = {}
    for method in methods:
        rows = [case["methods"][method] for case in cases]
        summary[method] = {
            "score": float(np.mean([row["metrics"]["score"] for row in rows])),
            "valid_grids": sum(row["metrics"]["invalid_cells"] == 0 for row in rows),
            "fine_score": float(np.mean([row["fine_metrics"]["score"] for row in rows])),
            "fine_valid_grids": sum(row["fine_metrics"]["invalid_cells"] == 0 for row in rows),
            "evaluations_per_case": rows[0]["evaluations"],
        }
    return {"benchmark": "analytic_naca_2d_v1", "test_cases": len(cases), "summary": summary, "cases": cases}
