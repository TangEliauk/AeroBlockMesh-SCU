"""Utilities shared by the archived SU2 evaluation and its figure script."""

import csv
import json
from pathlib import Path
import shutil
import subprocess
from time import perf_counter

import numpy as np

from run_naca_su2 import read_history

OUT = Path(__file__).resolve().parent
DATA = OUT.parents[2] / "data/references/public_v1/experiments"
CONFIG_DIR = OUT / "reference/su2_configs"
CFLS = {"warm": 6, "rans": 15, "l2roe": 15}


def write_json(path, payload):
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def solve(folder, solver, threads, stages):
    for stage in stages:
        config = folder / f"{stage}.cfg"
        shutil.copyfile(CONFIG_DIR / config.name, config)
        start = perf_counter()
        print(f"{folder.name} {stage} starting", flush=True)
        with (folder / f"{stage}.log").open("x", encoding="utf-8") as log:
            process = subprocess.run(
                [str(solver), "-t", str(threads), config.name],
                cwd=folder, stdout=log, stderr=subprocess.STDOUT, check=False,
            )
        status_path = folder / f"{stage}_status.json"
        if process.returncode != 0:
            write_json(status_path, {"exit_code": process.returncode})
            return False
        history = read_history(folder / f"{stage}_history.csv", target=-9.0)
        write_json(status_path, {
            **history, "exit_code": 0, "seconds": perf_counter() - start, "cfl": CFLS[stage],
        })
    return True


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream, skipinitialspace=True)
        rows.fieldnames = [field.strip() for field in rows.fieldnames]
        return list(rows)


def wind_tunnel():
    records = read_rows(DATA / "naca0012_ladson_force.csv")
    samples = sorted((float(r["alpha_deg"]), float(r["cl"]), float(r["cd"]))
                     for r in records if r["grit_number"] == "80")
    a, b = max(row for row in samples if row[0] < 10), min(row for row in samples if row[0] > 10)
    fraction = (10 - a[0]) / (b[0] - a[0])
    return {"cl": a[1] + fraction * (b[1] - a[1]), "cd": a[2] + fraction * (b[2] - a[2]),
            "source": "Ladson NASA TM 4074, grit 80, linear interpolation to alpha=10 deg",
            "bracketing_points_alpha_cl_cd": [a, b]}


def pressure(path):
    rows = read_rows(path)
    xy = np.array([[float(r["x"]), float(r["y"])] for r in rows])
    values = np.array([[float(r[f]) for f in ("Density", "Momentum_x", "Momentum_y", "Energy")] for r in rows])
    density = values[:, 0]
    kinetic = .5 * np.sum(values[:, 1:3] ** 2, axis=1) / density
    p = (1.4 - 1) * (values[:, 3] - kinetic)
    # FREESTREAM_VEL_EQ_MACH in the run's log gives rho_inf=1 and p_inf=1/gamma.
    cp = (p - 1 / 1.4) / (.5 * .15**2)
    if not np.isfinite(cp).all() or np.any(density <= 0):
        raise ValueError(f"Invalid surface solution {path}")
    upper = xy[:, 1] >= -1e-12
    order = np.argsort(xy[upper, 0])
    # For this symmetric section, lower LE→TE then upper TE→LE is solid CCW.
    loop = np.argsort(np.where(xy[:, 1] > 1e-12, 2 - xy[:, 0], xy[:, 0]))
    wall = xy[loop]
    steps = np.roll(wall, -1, axis=0) - wall
    edge_cp = .5 * (cp[loop] + np.roll(cp[loop], -1))
    force = np.array([-edge_cp @ steps[:, 1], edge_cp @ steps[:, 0]])
    alpha = np.deg2rad(10)
    pressure_drag = float(force @ [np.cos(alpha), np.sin(alpha)])
    return (xy[upper, 0][order], cp[upper][order]), pressure_drag


def su2_extra_metrics(points, cells, markers):
    """Compute equiangle skewness, aspect ratio, and farfield bounds on SU2 quads."""
    xy = points[cells]
    edges = np.roll(xy, -1, axis=1) - xy
    lengths = np.linalg.norm(edges, axis=-1)
    u = edges / np.maximum(lengths[..., None], 1e-30)
    v = -np.roll(u, 1, axis=1)
    cos_theta = np.clip((u * v).sum(axis=-1), -1.0, 1.0)
    theta_deg = np.degrees(np.arccos(cos_theta))
    cell_skew = (np.abs(theta_deg - 90.0) / 90.0).max(axis=1)
    len_a = 0.5 * (lengths[:, 0] + lengths[:, 2])
    len_b = 0.5 * (lengths[:, 1] + lengths[:, 3])
    ar = np.maximum(len_a, len_b) / np.maximum(np.minimum(len_a, len_b), 1e-30)
    far_pts = points[np.unique(markers["farfield"])]
    return {
        "max_equiangle_skewness": float(cell_skew.max()),
        "q95_equiangle_skewness": float(np.quantile(cell_skew, 0.95)),
        "mean_equiangle_skewness": float(cell_skew.mean()),
        "max_aspect_ratio": float(ar.max()),
        "q95_aspect_ratio": float(np.quantile(ar, 0.95)),
        "median_aspect_ratio": float(np.median(ar)),
        "farfield_bounds": {
            "x_min": float(far_pts[:, 0].min()),
            "x_max": float(far_pts[:, 0].max()),
            "y_min": float(far_pts[:, 1].min()),
            "y_max": float(far_pts[:, 1].max()),
        },
    }
