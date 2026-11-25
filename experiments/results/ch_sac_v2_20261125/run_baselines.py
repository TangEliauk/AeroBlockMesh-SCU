"""Step 3: Strong non-RL baselines (Default, GlobalOpt, LocalSearch-6eval, SupervisedMLP)."""

from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter

import numpy as np
from scipy.optimize import minimize
import torch
import torch.nn as nn

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))

from ch_env import ACTION_DIM, BUDGETS, DEFAULT_U, GEOM_DIM, MAX_STEPS, evaluate_u, load_cached_split

_WORKER_ITEMS = {}


def _init_worker(cache_paths):
    global _WORKER_ITEMS
    for name, path_str in cache_paths.items():
        _WORKER_ITEMS[name] = load_cached_split(path_str)


def _eval_batch_job(job):
    split_name, idx_list, u = job
    items = _WORKER_ITEMS[split_name]
    out = []
    for idx in idx_list:
        it = items[idx]
        try:
            m = evaluate_u(it, u)
            out.append((idx, True, m))
        except Exception:
            out.append((idx, False, None))
    return out


def _eval_fixed_u_parallel(pool, split_name, indices, u, n_workers=8):
    chunks = [indices[i::n_workers] for i in range(n_workers) if len(indices[i::n_workers]) > 0]
    res = {}
    for part in pool.map(_eval_batch_job, [(split_name, ch, u) for ch in chunks]):
        for idx, ok, m in part:
            res[idx] = (ok, m)
    return res


def _local_search_job(job):
    """Matched 6-eval budget: start at u_init (eval 1) + 5 deterministic coordinate/axis probes."""
    split_name, idx, u_init = job
    it = _WORKER_ITEMS[split_name][idx]
    u_cur = np.clip(np.asarray(u_init, dtype=np.float32), -1.0, 1.0)
    try:
        m_cur = evaluate_u(it, u_cur)
        valid = True
    except Exception:
        u_cur = DEFAULT_U.copy()
        m_cur = dict(it["default_metrics"])
        valid = False

    # 5 deterministic coordinate probes prioritizing high-leverage parameters
    # Order of PARAM_SPECS: 0=split_frac, 1=exit_first_dy, 2=wake_top_dx, 3=aft_top_bump, 4=dcut_u, 5=dcut_l, 6=wake_deg
    probes = [
        (0, +0.35),
        (1, +0.35),
        (3, -0.35),
        (4, +0.35),
        (5, -0.35),
    ]
    for dim, delta in probes:
        # Try sign aligned with improving direction if first try doesn't improve on remaining budget?
        # Strictly 5 probes total => 6 total evaluations (1 init + 5 steps), matched to RL's 6 evals.
        u_try = u_cur.copy()
        u_try[dim] = float(np.clip(u_try[dim] + delta, -1.0, 1.0))
        try:
            m_try = evaluate_u(it, u_try)
            if m_try["score"] > m_cur["score"]:
                u_cur = u_try
                m_cur = m_try
        except Exception:
            pass
    return {"id": it["id"], "name": it["name"], "budget": it["budget"], "valid": True, "u": u_cur.tolist(), "metrics": m_cur}


def _expert_target_job(job):
    """Find per-instance target u* on Train via bounded Powell search (max 28 evals)."""
    idx, u_global = job
    it = _WORKER_ITEMS["train"][idx]
    best_u = DEFAULT_U.copy()
    best_m = dict(it["default_metrics"])
    best_s = best_m["score"]

    # Check global start first
    u_g = np.clip(np.asarray(u_global, dtype=np.float32), -1.0, 1.0)
    try:
        m_g = evaluate_u(it, u_g)
        if m_g["score"] > best_s:
            best_u, best_m, best_s = u_g, m_g, m_g["score"]
    except Exception:
        pass

    start_u = best_u.copy()
    cache = {}

    def obj(x):
        xc = np.clip(np.asarray(x, dtype=float), -1.0, 1.0)
        key = tuple(np.round(xc, 4))
        if key in cache:
            return cache[key]
        try:
            m = evaluate_u(it, xc)
            val = -float(m["score"])
            nonlocal best_u, best_m, best_s
            if m["score"] > best_s:
                best_u, best_m, best_s = xc.astype(np.float32), m, m["score"]
        except Exception:
            val = 5.0
        cache[key] = val
        return val

    minimize(obj, start_u.astype(float), method="Powell", options={"maxfev": 26, "ftol": 1e-3, "xtol": 1e-2})
    return {"idx": idx, "id": it["id"], "budget": it["budget"], "u_star": best_u.tolist(), "score_star": best_s}


class SupervisedMLP(nn.Module):
    def __init__(self, in_dim=GEOM_DIM + 2, out_dim=ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, out_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


def _features_for_mlp(items):
    xs = []
    for it in items:
        bvec = [1.0 if it["budget"] == 1536 else 0.0, 1.0 if it["budget"] == 5120 else 0.0]
        xs.append(np.concatenate([np.asarray(it["geom_features"], dtype=np.float32), np.asarray(bvec, dtype=np.float32)]))
    return np.stack(xs, axis=0)


def summarize_records(records):
    out = {}
    for budget in BUDGETS:
        sub = [r for r in records if r["budget"] == budget]
        valid = [r for r in sub if r["valid"]]
        ms = [r["metrics"] for r in valid]
        out[str(budget)] = {
            "total": len(sub),
            "valid": len(valid),
            "score_mean": float(np.mean([m["score"] for m in ms])) if ms else None,
            "score_sd": float(np.std([m["score"] for m in ms], ddof=1)) if len(ms) > 1 else 0.0,
            "jmin_mean": float(np.mean([m["min_scaled_jacobian"] for m in ms])) if ms else None,
            "jmin_min": float(np.min([m["min_scaled_jacobian"] for m in ms])) if ms else None,
            "j05_mean": float(np.mean([m["q05_scaled_jacobian"] for m in ms])) if ms else None,
            "wall_q10_mean": float(np.mean([m["wall_angle_q10_deg"] for m in ms])) if ms else None,
            "far_ar_mean": float(np.mean([m["farfield_max_aspect_ratio"] for m in ms])) if ms else None,
            "far_ar_max": float(np.max([m["farfield_max_aspect_ratio"] for m in ms])) if ms else None,
        }
    return out


def main():
    t0 = perf_counter()
    cache_dir = OUT / "cache"
    cache_paths = {
        "train": str(cache_dir / "train_g0.pkl.gz"),
        "val": str(cache_dir / "val_g0.pkl.gz"),
        "test_uiuc": str(cache_dir / "test_uiuc_g0.pkl.gz"),
        "test_naca24": str(cache_dir / "test_naca24_g0.pkl.gz"),
    }
    splits = {name: load_cached_split(p) for name, p in cache_paths.items()}

    per_case = {}
    summary = {"global_u": {}, "methods": {}}

    with ProcessPoolExecutor(max_workers=8, initializer=_init_worker, initargs=(cache_paths,)) as pool:
        # 1. B0_Default
        b0 = {}
        for sname in ("val", "test_uiuc", "test_naca24"):
            recs = [
                {"id": it["id"], "name": it["name"], "budget": it["budget"], "valid": True, "u": DEFAULT_U.tolist(), "metrics": it["default_metrics"]}
                for it in splits[sname]
            ]
            b0[sname] = recs
        per_case["B0_Default"] = b0
        summary["methods"]["B0_Default"] = {k: summarize_records(v) for k, v in b0.items()}
        print("B0_Default done.", flush=True)

        # 2. B1_GlobalOpt: Powell search on 64 stratified train indices per budget
        train_items = splits["train"]
        u_global = {}
        for budget in BUDGETS:
            all_b_idx = [i for i, it in enumerate(train_items) if it["budget"] == budget]
            sub_idx = all_b_idx[::3][:64]

            def global_obj(u):
                uc = np.clip(np.asarray(u, dtype=float), -1.0, 1.0)
                res = _eval_fixed_u_parallel(pool, "train", sub_idx, uc)
                vals = []
                for idx in sub_idx:
                    ok, m = res[idx]
                    vals.append(m["score"] if ok else -2.0)
                return -float(np.mean(vals))

            opt = minimize(
                global_obj,
                DEFAULT_U.astype(float),
                method="Powell",
                options={"maxfev": 75, "ftol": 2e-3, "xtol": 2e-2},
            )
            ug = np.clip(opt.x, -1.0, 1.0).astype(np.float32)
            # Safety verify on all 192 train airfoils; shrink toward DEFAULT_U if any fail
            for alpha in (1.0, 0.75, 0.5, 0.25, 0.0):
                u_cand = (alpha * ug + (1.0 - alpha) * DEFAULT_U).astype(np.float32)
                full_res = _eval_fixed_u_parallel(pool, "train", all_b_idx, u_cand)
                if all(ok for ok, _ in full_res.values()):
                    ug = u_cand
                    break
            u_global[str(budget)] = ug.tolist()
            print(f"B1_GlobalOpt budget={budget} u={np.round(ug, 3).tolist()}", flush=True)

        summary["global_u"] = u_global
        b1 = {}
        for sname in ("val", "test_uiuc", "test_naca24"):
            recs = []
            items = splits[sname]
            for budget in BUDGETS:
                idxs = [i for i, it in enumerate(items) if it["budget"] == budget]
                res = _eval_fixed_u_parallel(pool, sname, idxs, u_global[str(budget)])
                for idx in idxs:
                    ok, m = res[idx]
                    it = items[idx]
                    recs.append({"id": it["id"], "name": it["name"], "budget": it["budget"], "valid": ok, "u": u_global[str(budget)], "metrics": m})
            b1[sname] = recs
        per_case["B1_GlobalOpt"] = b1
        summary["methods"]["B1_GlobalOpt"] = {k: summarize_records(v) for k, v in b1.items()}
        print("B1_GlobalOpt eval done.", flush=True)

        # 3. B2_LocalSearch (6 evals total)
        b2 = {}
        for sname in ("val", "test_uiuc", "test_naca24"):
            jobs = [(sname, i, u_global[str(it["budget"])]) for i, it in enumerate(splits[sname])]
            b2[sname] = list(pool.map(_local_search_job, jobs, chunksize=4))
        per_case["B2_LocalSearch"] = b2
        summary["methods"]["B2_LocalSearch"] = {k: summarize_records(v) for k, v in b2.items()}
        print("B2_LocalSearch done.", flush=True)

        # 4. B3_SupervisedMLP: generate expert targets on Train, fit MLP on CPU, eval
        expert_jobs = [(i, u_global[str(it["budget"])]) for i, it in enumerate(train_items)]
        expert_rows = list(pool.map(_expert_target_job, expert_jobs, chunksize=4))
        (OUT / "train_expert_targets.json").write_text(json.dumps(expert_rows, indent=2), encoding="utf-8")

        torch.manual_seed(42)
        X_tr = torch.tensor(_features_for_mlp(train_items), dtype=torch.float32)
        Y_tr = torch.tensor(np.array([r["u_star"] for r in expert_rows], dtype=np.float32), dtype=torch.float32)

        model = SupervisedMLP()
        opt_mlp = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        loss_fn = nn.SmoothL1Loss()
        perm_rng = np.random.default_rng(42)
        for _epoch in range(180):
            perm = perm_rng.permutation(len(X_tr))
            for start in range(0, len(X_tr), 64):
                bi = perm[start : start + 64]
                pred = model(X_tr[bi])
                loss = loss_fn(pred, Y_tr[bi])
                opt_mlp.zero_grad()
                loss.backward()
                opt_mlp.step()
        model.eval()
        torch.save(model.state_dict(), OUT / "supervised_mlp.pt")

        b3 = {}
        for sname in ("val", "test_uiuc", "test_naca24"):
            items = splits[sname]
            with torch.no_grad():
                preds = model(torch.tensor(_features_for_mlp(items), dtype=torch.float32)).numpy()
            recs = []
            for it, u_pred in zip(items, preds):
                try:
                    m = evaluate_u(it, u_pred)
                    recs.append({"id": it["id"], "name": it["name"], "budget": it["budget"], "valid": True, "u": u_pred.tolist(), "metrics": m})
                except Exception:
                    recs.append({"id": it["id"], "name": it["name"], "budget": it["budget"], "valid": False, "u": u_pred.tolist(), "metrics": None})
            b3[sname] = recs
        per_case["B3_SupervisedMLP"] = b3
        summary["methods"]["B3_SupervisedMLP"] = {k: summarize_records(v) for k, v in b3.items()}
        print("B3_SupervisedMLP done.", flush=True)

    summary["elapsed_sec"] = round(perf_counter() - t0, 2)
    (OUT / "baselines_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "baselines_per_case.json").write_text(json.dumps(per_case, indent=2), encoding="utf-8")
    print(json.dumps(summary["methods"], indent=2), flush=True)


if __name__ == "__main__":
    main()
