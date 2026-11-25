"""Step 2: Stratified UIUC + NACA-24 dataset curation and dual-budget g0 caching."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(OUT))
sys.path.insert(0, str(ROOT / "scripts"))

from ch_env import (
    BUDGETS,
    DEFAULT_U,
    apply_7controls,
    build_g0,
    eval_graph,
    is_blunt_branches,
)
import generate_wake_gmsh as wake

SPLITS_DIR = ROOT / "data/splits/uiuc_v1"
NACA_DIR = OUT / "reference/naca24"

TARGET_COUNTS = {"train": 192, "val": 32, "test_uiuc": 64}


def _scan_profile(name):
    try:
        _, polygon, branches = wake.read_profile(name, wake.DATA)
        poly = np.asarray(polygon, dtype=float)
        lower = poly[branches[0]]
        upper = poly[branches[1]]
        x_dense = np.linspace(0.0, 1.0, 101)
        order_u = np.argsort(upper[:, 0])
        xu, yu = upper[order_u, 0], upper[order_u, 1]
        _, iu = np.unique(xu, return_index=True)
        order_l = np.argsort(lower[:, 0])
        xl, yl = lower[order_l, 0], lower[order_l, 1]
        _, il = np.unique(xl, return_index=True)
        yui = np.interp(x_dense, xu[iu], yu[iu])
        yli = np.interp(x_dense, xl[il], yl[il])
        max_t = float(np.max(yui - yli))
        max_c = float(np.max(np.abs(0.5 * (yui + yli))))
        blunt = is_blunt_branches(branches)
        return {"name": name, "blunt": blunt, "max_t": max_t, "max_c": max_c, "ok": True}
    except Exception as exc:
        return {"name": name, "ok": False, "error": str(exc)}


def _stratified_order(records, seed=2026):
    """Order split records round-robin across (blunt, thickness_tertile, camber_tertile) strata."""
    rng = np.random.default_rng(seed)
    ts = np.array([r["max_t"] for r in records])
    cs = np.array([r["max_c"] for r in records])
    t_q = np.quantile(ts, [1.0 / 3.0, 2.0 / 3.0])
    c_q = np.quantile(cs, [1.0 / 3.0, 2.0 / 3.0])

    strata = {}
    for r in records:
        tb = int(np.searchsorted(t_q, r["max_t"]))
        cb = int(np.searchsorted(c_q, r["max_c"]))
        key = (int(r["blunt"]), tb, cb)
        strata.setdefault(key, []).append(r)

    keys = sorted(strata.keys())
    for k in keys:
        idx = rng.permutation(len(strata[k]))
        strata[k] = [strata[k][i] for i in idx]

    ordered = []
    pos = {k: 0 for k in keys}
    remaining = len(records)
    while remaining > 0:
        progressed = False
        for k in keys:
            i = pos[k]
            if i < len(strata[k]):
                ordered.append(strata[k][i])
                pos[k] = i + 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return ordered


def _build_airfoil_pair(job):
    """Build both budget instances (1536 and 5120) for one airfoil."""
    name, data_dir_str, display_id = job
    data_dir = Path(data_dir_str)
    items = []
    try:
        for budget in BUDGETS:
            g0, blunt, quads, geom = build_g0(name, data_dir, budget)
            g_def = apply_7controls(g0, u_to_params_default())
            m_def = eval_graph(g_def)
            if m_def["quads"] != quads:
                raise ValueError(f"Default quad mismatch {m_def['quads']} != {quads}")
            items.append(
                {
                    "id": f"{display_id}_{budget}",
                    "name": display_id,
                    "raw_name": name,
                    "budget": int(budget),
                    "blunt": bool(blunt),
                    "quads": int(quads),
                    "geom_features": geom,
                    "g0_dict": g0.to_dict(),
                    "default_metrics": m_def,
                }
            )
        return {"ok": True, "name": display_id, "items": items}
    except Exception as exc:
        return {"ok": False, "name": display_id, "error": str(exc)}


def u_to_params_default():
    from ch_env import DEFAULT_U, u_to_params

    return u_to_params(DEFAULT_U)


def _summarize_items(items):
    out = {}
    for budget in BUDGETS:
        sub = [it["default_metrics"] for it in items if it["budget"] == budget]
        out[str(budget)] = {
            "count": len(sub),
            "blunt_count": sum(1 for it in items if it["budget"] == budget and it["blunt"]),
            "score_mean": float(np.mean([m["score"] for m in sub])),
            "jmin_mean": float(np.mean([m["min_scaled_jacobian"] for m in sub])),
            "jmin_min": float(np.min([m["min_scaled_jacobian"] for m in sub])),
            "j05_mean": float(np.mean([m["q05_scaled_jacobian"] for m in sub])),
            "wall_q10_mean": float(np.mean([m["wall_angle_q10_deg"] for m in sub])),
            "far_ar_mean": float(np.mean([m["farfield_max_aspect_ratio"] for m in sub])),
            "far_ar_max": float(np.max([m["farfield_max_aspect_ratio"] for m in sub])),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out.resolve()

    t0 = perf_counter()
    cache_dir = out / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    raw_splits = {
        "train": json.loads((SPLITS_DIR / "train.json").read_text(encoding="utf-8")),
        "val": json.loads((SPLITS_DIR / "val.json").read_text(encoding="utf-8")),
        "test_uiuc": json.loads((SPLITS_DIR / "test.json").read_text(encoding="utf-8")),
    }
    # Verify strict disjointness of source splits
    sets = {k: set(v) for k, v in raw_splits.items()}
    assert not (sets["train"] & sets["val"])
    assert not (sets["train"] & sets["test_uiuc"])
    assert not (sets["val"] & sets["test_uiuc"])

    manifest = {"splits": {}, "excluded": {}, "timings_sec": {}}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for split_name, names in raw_splits.items():
            target_n = TARGET_COUNTS[split_name]
            scanned = list(pool.map(_scan_profile, names, chunksize=16))
            valid_recs = [r for r in scanned if r["ok"]]
            bad_scan = [r for r in scanned if not r["ok"]]
            ordered = _stratified_order(valid_recs, seed=2026 + len(split_name))

            # Build candidates with small surplus buffer in case any airfoil fails g0 meshing
            accepted_items = []
            accepted_names = []
            excluded = [{"name": b["name"], "stage": "scan", "error": b["error"]} for b in bad_scan]

            batch_start = 0
            while len(accepted_names) < target_n and batch_start < len(ordered):
                need = target_n - len(accepted_names)
                take = min(len(ordered) - batch_start, need + max(4, need // 8))
                cand = ordered[batch_start : batch_start + take]
                batch_start += take
                jobs = [(r["name"], str(wake.DATA), r["name"]) for r in cand]
                for res in pool.map(_build_airfoil_pair, jobs, chunksize=4):
                    if res["ok"]:
                        if len(accepted_names) < target_n:
                            accepted_names.append(res["name"])
                            accepted_items.extend(res["items"])
                    else:
                        excluded.append({"name": res["name"], "stage": "build_g0", "error": res["error"]})

            if len(accepted_names) != target_n:
                raise RuntimeError(f"Split {split_name} only yielded {len(accepted_names)}/{target_n} airfoils")

            cache_path = cache_dir / f"{split_name}_g0.pkl.gz"
            with gzip.open(cache_path, "wb") as fh:
                pickle.dump(accepted_items, fh, protocol=pickle.HIGHEST_PROTOCOL)

            manifest["splits"][split_name] = {
                "airfoils": len(accepted_names),
                "instances": len(accepted_items),
                "cache_file": cache_path.relative_to(out).as_posix(),
                "names": accepted_names,
                "default_metrics": _summarize_items(accepted_items),
            }
            manifest["excluded"][split_name] = excluded
            print(
                f"[{split_name}] cached {len(accepted_names)} airfoils ({len(accepted_items)} instances), "
                f"excluded={len(excluded)}",
                flush=True,
            )

        # NACA-24 benchmark in the archived sample order.
        with (NACA_DIR / "samples.jsonl").open(encoding="utf-8") as stream:
            naca_codes = [json.loads(line)["sample_id"].removeprefix("analytic_naca_") for line in stream]
        naca_jobs = [(f"naca{code}", str(NACA_DIR), f"naca{code}") for code in naca_codes]
        naca_items = []
        naca_names = []
        for res in pool.map(_build_airfoil_pair, naca_jobs, chunksize=2):
            if not res["ok"]:
                raise RuntimeError(f"NACA benchmark failure on {res['name']}: {res['error']}")
            naca_names.append(res["name"])
            naca_items.extend(res["items"])
        naca_cache = cache_dir / "test_naca24_g0.pkl.gz"
        with gzip.open(naca_cache, "wb") as fh:
            pickle.dump(naca_items, fh, protocol=pickle.HIGHEST_PROTOCOL)
        manifest["splits"]["test_naca24"] = {
            "airfoils": len(naca_names),
            "instances": len(naca_items),
            "cache_file": naca_cache.relative_to(out).as_posix(),
            "names": naca_names,
            "default_metrics": _summarize_items(naca_items),
        }
        print(f"[test_naca24] cached {len(naca_names)} airfoils ({len(naca_items)} instances)", flush=True)

    manifest["timings_sec"]["total"] = round(perf_counter() - t0, 2)
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Dataset preparation done in {manifest['timings_sec']['total']}s", flush=True)


if __name__ == "__main__":
    main()
