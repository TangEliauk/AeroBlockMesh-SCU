"""Step 5: 5-seed held-out evaluation (UIUC-64 + NACA-24) and SU2 8.6 RANS-SA (ROE + L2ROE) verification."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import sys
from time import perf_counter

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SCRIPTS = ROOT / "scripts"

for p in (OUT, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from audit_gmsh_airfoil import inspect, read_su2
from ch_env import (
    BUDGETS,
    DEFAULT_U,
    MAX_STEPS,
    STEP_SCALE,
    apply_7controls,
    build_obs,
    eval_graph,
    load_cached_split,
    u_to_params,
)
from ch_sac_v2_model import SquashedGaussianActor
from cfd_support import (
    CONFIG_DIR,
    DATA as EXP_DATA,
    pressure,
    read_rows,
    solve,
    su2_extra_metrics,
    wind_tunnel,
    write_json,
)
from smartcfd.meshing.gmsh_graph import verify_connectivity, write_mesh


SEEDS = (42, 43, 44, 45, 46)
TOL = 1e-9

def farfield_ar_metrics(points, cells):
    xy = points[cells]
    edges = np.roll(xy, -1, axis=1) - xy
    lens = np.linalg.norm(edges, axis=-1)
    la = 0.5 * (lens[:, 0] + lens[:, 2])
    lb = 0.5 * (lens[:, 1] + lens[:, 3])
    ar = np.maximum(la, lb) / np.maximum(np.minimum(la, lb), 1e-30)
    ctr = xy.mean(axis=1)
    far_mask = (np.abs(ctr[:, 1]) > 0.25) | (ctr[:, 0] > 1.5) | (ctr[:, 0] < -0.5)
    far_ar = ar[far_mask]
    return {
        "farfield_max_aspect_ratio": float(far_ar.max()),
        "farfield_q95_aspect_ratio": float(np.quantile(far_ar, 0.95)),
        "farfield_median_aspect_ratio": float(np.median(far_ar)),
    }


def _rollout_seed_on_split(items, ckpt_path):
    actor = SquashedGaussianActor()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    n = len(items)
    u_cur = np.repeat(DEFAULT_U[None, :], n, axis=0).astype(np.float32)
    m_cur = [dict(it["default_metrics"]) for it in items]
    best_u = u_cur.copy()
    best_m = [dict(m) for m in m_cur]

    with torch.no_grad():
        for step_idx in range(MAX_STEPS):
            obs = np.stack(
                [
                    build_obs(items[k]["geom_features"], u_cur[k], m_cur[k], step_idx, items[k]["budget"], MAX_STEPS)
                    for k in range(n)
                ],
                axis=0,
            )
            act, _ = actor.sample(torch.as_tensor(obs, dtype=torch.float32), deterministic=True)
            u_try = np.clip(u_cur + STEP_SCALE * act.numpy(), -1.0, 1.0).astype(np.float32)
            for k, it in enumerate(items):
                try:
                    g = apply_7controls(it["g0"], u_to_params(u_try[k]))
                    m_try = eval_graph(g)
                    if m_try["quads"] != it["quads"]:
                        raise ValueError("quad mismatch")
                    u_cur[k] = u_try[k]
                    m_cur[k] = m_try
                    if m_try["score"] > best_m[k]["score"]:
                        best_u[k] = u_try[k].copy()
                        best_m[k] = m_try
                except Exception:
                    pass

    recs = []
    for k, it in enumerate(items):
        recs.append(
            {
                "id": it["id"],
                "name": it["name"],
                "budget": it["budget"],
                "blunt": bool(it["blunt"]),
                "valid": True,
                "u_final": u_cur[k].tolist(),
                "metrics": m_cur[k],
                "u_best": best_u[k].tolist(),
                "best_metrics": best_m[k],
            }
        )
    return recs


def _agg_budget_stats(recs, use_best=False):
    out = {}
    key = "best_metrics" if use_best else "metrics"
    for budget in BUDGETS:
        sub = [r for r in recs if r["budget"] == budget]
        ms = [r[key] for r in sub if r["valid"]]
        out[str(budget)] = {
            "total": len(sub),
            "valid": len(ms),
            "score_mean": float(np.mean([m["score"] for m in ms])),
            "jmin_mean": float(np.mean([m["min_scaled_jacobian"] for m in ms])),
            "jmin_min": float(np.min([m["min_scaled_jacobian"] for m in ms])),
            "j05_mean": float(np.mean([m["q05_scaled_jacobian"] for m in ms])),
            "wall_q10_mean": float(np.mean([m["wall_angle_q10_deg"] for m in ms])),
            "far_ar_mean": float(np.mean([m["farfield_max_aspect_ratio"] for m in ms])),
            "far_ar_max": float(np.max([m["farfield_max_aspect_ratio"] for m in ms])),
            "max_ar_mean": float(np.mean([m["max_aspect_ratio"] for m in ms])),
        }
    return out


def _paired_vs(recs_a, recs_b, metrics=("score", "min_scaled_jacobian", "q05_scaled_jacobian", "wall_angle_q10_deg", "farfield_max_aspect_ratio")):
    map_b = {r["id"]: r["metrics"] for r in recs_b if r["valid"] and r["metrics"] is not None}
    out = {}
    for m in metrics:
        diffs = []
        better = same = worse = 0
        higher_better = m != "farfield_max_aspect_ratio"
        for ra in recs_a:
            mb = map_b.get(ra["id"])
            if not ra["valid"] or mb is None:
                continue
            da = float(ra["metrics"][m])
            db = float(mb[m])
            d = da - db
            diffs.append(d)
            eff = d if higher_better else -d
            if eff > TOL:
                better += 1
            elif eff < -TOL:
                worse += 1
            else:
                same += 1
        arr = np.asarray(diffs, dtype=float)
        out[m] = {
            "n": len(diffs),
            "mean_delta": float(arr.mean()) if len(arr) else None,
            "median_delta": float(np.median(arr)) if len(arr) else None,
            "win": better,
            "tie": same,
            "loss": worse,
        }
    return out


def _load_legacy_ch_naca24():
    """Load archived geometry metrics for the legacy C-H actor."""
    return json.loads((OUT / "reference/legacy_ch_metrics.json").read_text(encoding="utf-8"))


def run_heldout_eval(result_dir=OUT):
    t0 = perf_counter()
    cache_dir = OUT / "cache"
    test_uiuc = load_cached_split(cache_dir / "test_uiuc_g0.pkl.gz")
    test_naca = load_cached_split(cache_dir / "test_naca24_g0.pkl.gz")
    baselines = json.loads((OUT / "baselines_per_case.json").read_text(encoding="utf-8"))
    train_sum = json.loads((OUT / "training_summary.json").read_text(encoding="utf-8"))

    seed_evals = {"test_uiuc": {}, "test_naca24": {}}
    seed_stats = {"test_uiuc": {}, "test_naca24": {}}

    for s in SEEDS:
        ckpt = OUT / "seeds" / f"seed_{s}" / "best.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(f"Missing checkpoint {ckpt}")
        r_uiuc = _rollout_seed_on_split(test_uiuc, ckpt)
        r_naca = _rollout_seed_on_split(test_naca, ckpt)
        seed_evals["test_uiuc"][str(s)] = r_uiuc
        seed_evals["test_naca24"][str(s)] = r_naca
        seed_stats["test_uiuc"][str(s)] = _agg_budget_stats(r_uiuc, use_best=False)
        seed_stats["test_naca24"][str(s)] = _agg_budget_stats(r_naca, use_best=False)

    # 5-seed mean +/- SD summary and per-instance 5-seed mean
    multi_seed_summary = {}
    ensemble_mean_recs = {}
    median_seed = int(train_sum["median_val_seed"])

    for sname, items in (("test_uiuc", test_uiuc), ("test_naca24", test_naca)):
        multi_seed_summary[sname] = {}
        for budget in BUDGETS:
            bkey = str(budget)
            metric_keys = ("score_mean", "jmin_mean", "j05_mean", "wall_q10_mean", "far_ar_mean", "far_ar_max", "max_ar_mean")
            agg = {"valid_min": min(seed_stats[sname][str(s)][bkey]["valid"] for s in SEEDS),
                   "total": seed_stats[sname][str(SEEDS[0])][bkey]["total"]}
            for mk in metric_keys:
                vals = [seed_stats[sname][str(s)][bkey][mk] for s in SEEDS]
                agg[mk] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1))}
            multi_seed_summary[sname][bkey] = agg

        # Build 5-seed mean record per instance for paired comparison
        mean_recs = []
        for idx, it in enumerate(items):
            m_keys = ("score", "min_scaled_jacobian", "q05_scaled_jacobian", "wall_angle_q10_deg", "farfield_max_aspect_ratio", "max_aspect_ratio")
            avg_m = {
                mk: float(np.mean([seed_evals[sname][str(s)][idx]["metrics"][mk] for s in SEEDS]))
                for mk in m_keys
            }
            mean_recs.append({"id": it["id"], "name": it["name"], "budget": it["budget"], "valid": True, "metrics": avg_m})
        ensemble_mean_recs[sname] = mean_recs

    # Paired comparisons (5-seed mean of CH-SAC v2 and median seed vs all baselines)
    legacy_ch_naca = _load_legacy_ch_naca24()
    paired_tables = {}
    for sname in ("test_uiuc", "test_naca24"):
        paired_tables[sname] = {}
        med_recs = seed_evals[sname][str(median_seed)]
        ens_recs = ensemble_mean_recs[sname]
        for bname in ("B0_Default", "B1_GlobalOpt", "B2_LocalSearch", "B3_SupervisedMLP"):
            for budget in BUDGETS:
                key = f"vs_{bname}_B{budget}"
                paired_tables[sname][key] = {
                    "5seed_mean": _paired_vs(
                        [r for r in ens_recs if r["budget"] == budget],
                        [r for r in baselines[bname][sname] if r["budget"] == budget],
                    ),
                    "median_seed": _paired_vs(
                        [r for r in med_recs if r["budget"] == budget],
                        [r for r in baselines[bname][sname] if r["budget"] == budget],
                    ),
                }
        if sname == "test_naca24":
            for budget in BUDGETS:
                key = f"vs_Legacy_CH_Actor_B{budget}"
                paired_tables[sname][key] = {
                    "5seed_mean": _paired_vs(
                        [r for r in ens_recs if r["budget"] == budget],
                        [r for r in legacy_ch_naca if r["budget"] == budget],
                        metrics=("min_scaled_jacobian", "q05_scaled_jacobian", "wall_angle_q10_deg", "farfield_max_aspect_ratio"),
                    ),
                    "median_seed": _paired_vs(
                        [r for r in med_recs if r["budget"] == budget],
                        [r for r in legacy_ch_naca if r["budget"] == budget],
                        metrics=("min_scaled_jacobian", "q05_scaled_jacobian", "wall_angle_q10_deg", "farfield_max_aspect_ratio"),
                    ),
                }

    summary = {
        "seeds": list(SEEDS),
        "median_val_seed": median_seed,
        "ch_sac_v2_multi_seed": multi_seed_summary,
        "ch_sac_v2_per_seed": seed_stats,
        "legacy_ch_naca24": _agg_budget_stats(legacy_ch_naca),
        "baselines": json.loads((OUT / "baselines_summary.json").read_text(encoding="utf-8"))["methods"],
        "paired_comparisons": paired_tables,
        "elapsed_sec": round(perf_counter() - t0, 2),
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    write_json(result_dir / "eval_summary.json", summary)
    write_json(
        result_dir / "eval_per_case.json",
        {"seed_evals": seed_evals, "ensemble_mean": ensemble_mean_recs, "legacy_ch_naca24": legacy_ch_naca},
    )
    print("Held-out 5-seed eval summary written.", flush=True)
    return summary, seed_evals


def _export_su2_case(folder, item, u_vec):
    folder.mkdir(parents=True, exist_ok=True)
    g = apply_7controls(item["g0"], u_to_params(u_vec))
    metrics, mesh = eval_graph(g, return_mesh=True)
    write_json(folder / "block_graph.json", g.to_dict())
    write_mesh(g, mesh, folder)
    pts, cells, marks = read_su2(folder / "mesh.su2", expected_markers=("airfoil", "farfield"))
    verify_connectivity(g, mesh, pts, cells, marks)
    base = inspect(pts, cells, marks)
    for k in ("quads", "min_scaled_jacobian", "q05_scaled_jacobian"):
        if abs(base[k] - mesh.quality[k]) > TOL:
            raise ValueError(f"SU2 audit mismatch on {folder.name}/{k}")
    audit = {
        **base,
        **su2_extra_metrics(pts, cells, marks),
        **farfield_ar_metrics(pts, cells),
        "score": metrics["score"],
        "mesh_sha256": sha256((folder / "mesh.su2").read_bytes()).hexdigest(),
    }
    write_json(folder / "mesh_audit.json", audit)
    return audit


def run_su2_verification(seed_evals, median_seed, solver, cfd_root):

    naca_items = {it["id"]: it for it in load_cached_split(OUT / "cache/test_naca24_g0.pkl.gz")}
    med_recs = {r["id"]: r for r in seed_evals["test_naca24"][str(median_seed)]}

    cfd_root.mkdir(parents=True, exist_ok=False)

    case_specs = [
        ("naca0012_v2_1536", "naca0012_1536", med_recs["naca0012_1536"]["u_final"], ("warm", "rans"), True),
        ("naca0012_v2_5120", "naca0012_5120", med_recs["naca0012_5120"]["u_final"], ("warm", "rans", "l2roe"), True),
        ("naca6423_def_5120", "naca6423_5120", DEFAULT_U, ("warm", "rans", "l2roe"), False),
        ("naca6423_v2_5120", "naca6423_5120", med_recs["naca6423_5120"]["u_final"], ("warm", "rans", "l2roe"), False),
        ("naca4415_def_5120", "naca4415_5120", DEFAULT_U, ("warm", "rans", "l2roe"), False),
        ("naca4415_v2_5120", "naca4415_5120", med_recs["naca4415_5120"]["u_final"], ("warm", "rans", "l2roe"), False),
    ]

    folders = {}
    audits = {}
    is_n0012_map = {}
    stages_map = {}
    for cname, item_id, u_vec, stages, is_0012 in case_specs:
        d = cfd_root / cname
        audits[cname] = _export_su2_case(d, naca_items[item_id], u_vec)
        folders[cname] = d
        is_n0012_map[cname] = is_0012
        stages_map[cname] = [stage for stage in stages if stage != "warm"]

    jobs = [(folders[cname], stages) for cname, _, _, stages, _ in case_specs]
    with ThreadPoolExecutor(max_workers=2) as pool:
        ok = list(pool.map(lambda job: solve(job[0], solver, 2, job[1]), jobs))
    if not all(ok):
        raise SystemExit("One or more SU2 verification runs failed")

    # Verify that every reproduced stage used the archived solver configuration.
    ref_cfg_sha = {config.stem: sha256(config.read_bytes()).hexdigest() for config in CONFIG_DIR.glob("*.cfg")}
    for folder, stages in jobs:
        for stage in stages:
            digest = sha256((folder / f"{stage}.cfg").read_bytes()).hexdigest()
            if digest != ref_cfg_sha[stage]:
                raise ValueError(f"Config SHA mismatch on {folder.name}/{stage}")

    experimental = wind_tunnel()
    numerical = {"cl": 1.09102722, "cd": 0.0122724643, "source": "TMR FUN3D Family II"}
    observed = np.array(
        [(float(r["x_over_c"]), float(r["cp"])) for r in read_rows(EXP_DATA / "naca0012_gregory_cp.csv") if r["alpha_deg"] == "10"]
    )
    observed = observed[observed[:, 0].argsort()]

    cfd_cases = {}
    for cname, d in folders.items():
        st_out = {}
        for st in stages_map[cname]:
            status = json.loads((d / f"{st}_status.json").read_text(encoding="utf-8"))
            log = (d / f"{st}.log").read_text(encoding="utf-8", errors="replace")
            status.update(
                {
                    "joint_converged": "All convergence criteria satisfied." in log,
                    "force_cauchy": {q: float(v) for q, v in re.findall(r"Cauchy\[(CD|CL)\]\|\s*([0-9.eE+-]+)", log)},
                }
            )
            if is_n0012_map[cname]:
                (x, cp), _ = pressure(d / f"{st}_surface.csv")
                use = (observed[:, 0] >= x[0]) & (observed[:, 0] <= x[-1])
                delta = np.interp(observed[use, 0], x, cp) - observed[use, 1]
                status.update(
                    {
                        "cl_error_pct_ladson": 100.0 * (status["cl"] / experimental["cl"] - 1.0),
                        "cd_error_pct_ladson": 100.0 * (status["cd"] / experimental["cd"] - 1.0),
                        "cl_error_pct_tmr": 100.0 * (status["cl"] / numerical["cl"] - 1.0),
                        "cd_error_pct_tmr": 100.0 * (status["cd"] / numerical["cd"] - 1.0),
                        "upper_cp_rmse_gregory": float(np.sqrt(np.mean(delta**2))),
                    }
                )
            st_out[st] = status
        cfd_cases[cname] = {"mesh": audits[cname], "stages": st_out}

    out = {
        "median_val_seed": median_seed,
        "configuration_sha256": ref_cfg_sha,
        "ladson": experimental,
        "tmr_fun3d": numerical,
        "cases": cfd_cases,
    }
    write_json(cfd_root / "cfd_verification_summary.json", out)
    print("\n=== CH-SAC v2 SU2 VERIFICATION SUMMARY ===", flush=True)
    for cname, row in cfd_cases.items():
        for st, d in row["stages"].items():
            extra = f" | Cp_RMSE={d['upper_cp_rmse_gregory']:.4f}" if "upper_cp_rmse_gregory" in d else ""
            print(
                f"{cname:18s} {st:5s} | q={row['mesh']['quads']} Jmin={row['mesh']['min_scaled_jacobian']:.4f} "
                f"farAR={row['mesh']['farfield_max_aspect_ratio']:.1f} conv={d['joint_converged']} "
                f"rms={d['final_log10_rms_density']:.2f} CL={d['cl']:.5f} CD={d['cd']:.5f}{extra}",
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--skip-cfd", action="store_true")
    mode.add_argument("--cfd-only", action="store_true")
    parser.add_argument("--eval-out", type=Path, default=OUT)
    parser.add_argument("--su2", type=Path, help="Path to an SU2 8.6 SU2_CFD executable")
    parser.add_argument("--cfd-out", type=Path, default=OUT / "cfd_reproduction")
    args = parser.parse_args()

    solver = None
    if not args.skip_cfd:
        solver_arg = args.su2 or shutil.which("SU2_CFD")
        if solver_arg is None or not Path(solver_arg).is_file():
            parser.error("provide the SU2_CFD executable with --su2 or use --skip-cfd")
        solver = Path(solver_arg).resolve()

    if args.cfd_only:
        ev_sum = json.loads((args.eval_out / "eval_summary.json").read_text(encoding="utf-8"))
        ev_pc = json.loads((args.eval_out / "eval_per_case.json").read_text(encoding="utf-8"))
        seed_evals = ev_pc["seed_evals"]
        median_seed = ev_sum["median_val_seed"]
    else:
        ev_sum, seed_evals = run_heldout_eval(args.eval_out)
        median_seed = ev_sum["median_val_seed"]

    if not args.skip_cfd:
        run_su2_verification(seed_evals, median_seed, solver, args.cfd_out.resolve())


if __name__ == "__main__":
    main()
