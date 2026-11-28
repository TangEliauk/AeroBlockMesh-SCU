"""SU2 validation on the immutable evaluated topology, with joint residual/force checks."""
from __future__ import annotations

import csv
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np

from smartcfd.airfoil_decomposition.config import AirfoilDecompositionConfig
from smartcfd.airfoil_decomposition.experiment import file_sha, write_json
from smartcfd.airfoil_decomposition.geometry import CurveGeometry
from smartcfd.cfd.convergence import check_history
from smartcfd.cfd.solver import export_su2
from smartcfd.core.types import BlockGraph
from smartcfd.meshing.generator import save_mesh

from .evaluate import case_key
from .mesh import BUDGETS, evaluate_mesh_budget
from .protocol import ROOT, published_bytes, read_json
from .rollout import selection_key


def solver_path():
    name = "SU2_CFD.exe" if os.name == "nt" else "SU2_CFD"
    local = ROOT/".runtime/su2_8.6.0/bin"/name
    solver = os.environ.get("AEROBLOCK_SU2") or (str(local) if local.is_file() else shutil.which("SU2_CFD"))
    if not solver or not Path(solver).is_file():
        raise FileNotFoundError("Set AEROBLOCK_SU2 to the SU2_CFD executable, or add it to PATH.")
    return Path(solver).resolve()


def read_history(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [{key.strip().strip('"'): value.strip() for key, value in row.items()}
                for row in csv.DictReader(stream, skipinitialspace=True)]


def _solve(directory, stage):
    cfg_name = f"{stage}.cfg"
    original = published_bytes(f"cfd_verification/naca0012_v2_5120/{cfg_name}")
    config = original.decode().replace("airfoil", "wall")
    config_path = directory/cfg_name
    config_path.write_text(config, encoding="ascii")
    start = time.perf_counter()
    state = {"config_sha256": file_sha(config_path)}
    with (directory/f"{stage}.stdout.log").open("wb") as errout, (directory/f"{stage}.stderr.log").open("wb") as err:
        try:
            child = subprocess.run([str(solver_path()), cfg_name], cwd=directory, stdout=errout, stderr=err,
                                   timeout=1800, env={**os.environ, "OMP_NUM_THREADS": "2"}, check=False)
            state["exit_code"] = child.returncode
        except subprocess.TimeoutExpired:
            state["exit_code"] = None
            state["reason"] = "solver_timeout"
        except OSError as exc:
            state.update(exit_code=None, reason=f"solver_launch_failed: {exc}")
    state["seconds"] = time.perf_counter()-start
    history = directory/f"{stage}_history.csv"
    if history.is_file():
        rows = read_history(history)
        state["iterations"] = len(rows)
        if rows:
            state["last"] = rows[-1]
            state["forces"] = {name: float(rows[-1][name])
                               for name in ("CL", "CD") if name in rows[-1] and np.isfinite(float(rows[-1][name]))}
        state["convergence"] = check_history(rows, residual_log10_target=-9., minimum_rows=100)
        state["history_sha256"] = file_sha(history)
    else:
        state["reason"] = state.get("reason", "missing_solver_history")
    state["converged"] = bool(state["exit_code"] == 0 and state.get("convergence", {}).get("converged"))
    return state


def _simulate(directory, sample, graph, budget, config):
    geom = CurveGeometry(sample, AirfoilDecompositionConfig(**config))
    if not geom.validate(graph, refine=True)["valid"]:
        raise ValueError("Sealed evaluated graph failed CFD source verification")
    metrics, mesh = evaluate_mesh_budget(geom.domain, graph, budget)
    record = {"mesh": metrics, "phases": {}, "converged": False}
    if mesh is None or not metrics["budget_match"]:
        return {**record, "status": "invalid_mesh" if mesh is None else "budget_mismatch"}
    save_mesh(mesh, directory)
    path = directory/"mesh.su2"
    exported = export_su2(mesh, path)
    text = path.read_text(encoding="ascii")
    n_cells = re.search(r"(?m)^NELEM= (\d+)$", text)
    n_markers = set(re.findall(r"(?m)^MARKER_TAG= (\w+)$", text))
    if not n_cells or int(n_cells[1]) != metrics["quads"] or n_markers != {"wall", "farfield"}:
        raise ValueError("SU2 file differs from the validated TFI mesh")
    record.update(export=exported, mesh_sha256=file_sha(path))
    warm = _solve(directory, "warm")
    record["phases"]["warm"] = warm
    if warm["exit_code"] != 0 or not (directory/"warm.dat").is_file():
        return {**record, "status": "warmup_failed"}
    rans = _solve(directory, "rans")
    record["phases"]["rans"] = rans
    record["converged"] = rans["converged"]
    record["status"] = "converged" if rans["converged"] else "nonconverged" if rans["exit_code"] == 0 else "solver_failed"
    return record


def run(output, cfg):
    directory = output/"cfd"
    directory.mkdir(exist_ok=True)
    lock = read_json(output/"evaluation_lock.json")
    rows = read_json(output/"evaluation_cases.json")["cases"]
    sample = next((row for row in rows if row["sample_id"] == lock["cfd_sample_id"]), None)
    summary = {"scope": "preliminary solver check at Mach .15, Re 6e6, alpha 10 degrees",
               "solver_sha256": file_sha(solver_path()), "sample_id": lock["cfd_sample_id"], "cases": []}
    if sample is None:
        summary["status"] = "naca0012_not_eligible_in_frozen_holdout"
        write_json(directory/"summary.json", summary, replace=True)
        return summary
    for arm in ("E1", "E7", "E8") if cfg["profile"] == "formal" else ("E1",):
        seed, tag = 42, "rule"
        if arm != "E1":
            candidates = []
            for seed in cfg["seeds"]:
                history = read_json(output/"arms"/f"{arm}_seed{seed}"/"completion.json")["validation"]
                candidates.append((selection_key(max(history, key=selection_key)), seed))
            seed, tag = sorted(candidates)[len(candidates)//2][1], "best"
        result = read_json(output/"evaluation"/f"{case_key(arm, seed, tag)}.json")
        row = next(r for r in result["rows"] if r["sample_id"] == sample["sample_id"])
        for budget in BUDGETS if cfg["profile"] == "formal" else BUDGETS[:1]:
            identity = {"arm": arm, "seed": seed, "tag": tag, "budget": budget, "sample_id": sample["sample_id"]}
            case = directory/f"{arm}_seed{seed}_{budget}"
            case.mkdir(exist_ok=True)
            receipt = case/"result.json"
            if receipt.exists():
                summary["cases"].append(read_json(receipt))
                continue
            if row["success"]:
                attempt = case/f"attempt_{len(list(case.glob('attempt_*')))+1:03d}"
                attempt.mkdir()
                graph = BlockGraph.from_dict(row["graph"])
                result = {**_simulate(attempt, sample, graph, budget, cfg["geometry"]),
                          "directory": attempt.relative_to(output).as_posix()}
            else:
                result = {"status": "partition_failed", "reason": row["reason"], "converged": False}
            result.update(identity)
            write_json(receipt, result, replace=True)
            summary["cases"].append(result)
            print(f"CFD {arm} seed={seed} cells={budget} {result['status']}", flush=True)
    write_json(directory/"summary.json", summary, replace=True)
    return summary
