"""Sealed test evaluation, paired geometry metrics and the pinned CH-SAC reference."""
from __future__ import annotations

from collections import Counter
import json
import re

import numpy as np
import torch

from smartcfd.airfoil_decomposition.config import AirfoilDecompositionConfig
from smartcfd.airfoil_decomposition.data import preflight
from smartcfd.airfoil_decomposition.experiment import file_sha, write_json

from .mesh import BUDGETS
from .protocol import BASELINES, B_COMMIT, SAC_ARMS, official_cases, published_bytes, read_json
from .rollout import Environments, episode, summarize
from .train import get_inference_agent


def hypotheses():
    return (("E4a", "E4"), ("E6", "E4a"), ("E7", "E6"), ("E7", "E5"), ("E8", "E7"), ("E7", "E1"))


def case_key(arm, seed, tag):
    return f"{arm}_{seed}_{tag}"


def lock_testing(output, cfg, cases, checks):
    """Read test coordinates only once all model and budget choices are durable."""
    ready = read_json(output/"training_matrix.json")
    path = output/"evaluation_lock.json"
    for name, sha in ready["checkpoints"].items():
        if file_sha(output/name) != sha:
            raise ValueError(f"Training identity changed before evaluation: {name}")
    if path.exists():
        lock = read_json(path)
        if lock["training_sha256"] != file_sha(output/"training_matrix.json") or lock["cases_sha256"] != file_sha(output/"evaluation_cases.json"):
            raise ValueError("Evaluation lock and content differ")
        return lock
    if cfg["profile"] == "formal":
        sample_rows, manifest = official_cases()
        env_cfg = AirfoilDecompositionConfig(**cfg["geometry"])
        init_rows = preflight({"train": [], "val": sample_rows}, env_cfg)["val"]
        sample = next((r for r in sample_rows if any(
            re.sub("[^a-z0-9]", "", alias.lower()) == "naca0012" for alias in r["aliases"]
        )), None)
    else:
        sample_rows, init_rows = cases["val"], checks["val"]
        manifest = {"scope": "development validation only", "eligible": len(sample_rows)}
        sample = sample_rows[0]
    if not sample_rows:
        raise RuntimeError("No eligible evaluation cases")
    write_json(output/"evaluation_cases.json", {"cases": sample_rows, "preflight": init_rows, "manifest": manifest}, replace=True)
    lock = {"training_sha256": file_sha(output/"training_matrix.json"), "cases_sha256": file_sha(output/"evaluation_cases.json"),
            "profile": cfg["profile"], "cfd_sample_id": sample["sample_id"] if sample else None}
    write_json(path, lock)
    return lock


def jobs(cfg):
    keys = [("E1", 42, "rule"), ("E3", 42, "rule")]
    keys.extend(("E2", seed, "rule") for seed in cfg["seeds"])
    keys.extend((arm, seed, tag) for arm in (*SAC_ARMS, "E5") for seed in cfg["seeds"] for tag in ("best", "final"))
    return keys


def evaluate(output, cfg, arm, seed, tag):
    torch.set_num_threads(cfg["runtime"]["torch_threads"])
    lock = read_json(output/"evaluation_lock.json")
    case_path = output/"evaluation_cases.json"
    if file_sha(case_path) != lock["cases_sha256"] or file_sha(output/"training_matrix.json") != lock["training_sha256"]:
        raise ValueError("Evaluation input changed after lock")
    test = read_json(case_path)
    directory = output/"evaluation"
    directory.mkdir(exist_ok=True)
    path = directory/f"{case_key(arm, seed, tag)}.json"
    if path.exists():
        report = read_json(path)
        if report["lock_sha256"] != file_sha(output/"evaluation_lock.json"):
            raise ValueError(f"Evaluation artifact belongs to another lock: {path.name}")
        return report["summary"]
    agent, step, method, sha = None, None, BASELINES.get(arm, "learned"), None
    if method == "learned":
        checkpoint = output/"arms"/f"{arm}_seed{seed}"/f"{tag}.pt"
        sha = file_sha(checkpoint)
        if read_json(output/"training_matrix.json")["checkpoints"][checkpoint.relative_to(output).as_posix()] != sha:
            raise ValueError("Evaluation expected the sealed trained weights")
        agent, step = get_inference_agent(checkpoint, arm, seed, file_sha(output/"prepared.json"))
    pool = Environments(cfg["geometry"], arm)
    prechecked = {r["sample_id"]: r for r in test["preflight"]}
    rows = []
    for sample in test["cases"]:
        row, _ = episode(pool, sample, method, agent=agent, seed=seed, prechecked=prechecked[sample["sample_id"]], trace=True)
        rows.append(row)
        if len(rows) % 8 == 0 or len(rows) == len(test["cases"]):
            print(f"eval {arm} seed={seed} {tag} {len(rows)}/{len(test['cases'])}", flush=True)
    payload = {"arm": arm, "seed": seed, "tag": tag, "step": step, "checkpoint_sha256": sha,
               "lock_sha256": file_sha(output/"evaluation_lock.json"), "summary": summarize(rows), "rows": rows}
    write_json(path, payload, replace=True)
    return payload["summary"]


def paired(left, right, budget):
    paired_labels = set(left) & set(right)
    both_topology = [sid for sid in paired_labels if left[sid]["success"] and right[sid]["success"]]
    fields = ("score", "min_scaled_jacobian", "q05_scaled_jacobian", "wall_angle_q10_deg",
              "farfield_max_aspect_ratio", "max_aspect_ratio")
    rows = []
    for sid in sorted(both_topology):
        a, b = (table[sid].get("mesh", {}).get(str(budget), {}) for table in (left, right))
        if a.get("budget_match") and b.get("budget_match") and abs(a["quads"]-b["quads"]) <= .02*budget:
            rows.append({"sample_id": sid, "left_cells": a["quads"], "right_cells": b["quads"],
                         **{field: a[field]-b[field] for field in fields}})
    return {"common_cases": len(paired_labels), "both_partitioned": len(both_topology),
            "matched_meshes": len(rows), "rows": rows,
            "success_delta": sum(int(left[s]["success"])-int(right[s]["success"]) for s in paired_labels)/len(paired_labels),
            "mean_delta": {key: float(np.mean([r[key] for r in rows])) for key in fields} if rows else {}}


def benchmark_reference(output):
    stats = json.loads(published_bytes("eval_summary.json"))
    training = json.loads(published_bytes("training_summary.json"))
    cfd = json.loads(published_bytes("cfd_verification_summary.json"))
    result = {"arm": "E9", "commit": B_COMMIT, "evidence": "published five-seed deterministic template optimization",
              "train": training, "mesh_quality": stats["ch_sac_v2_multi_seed"], "baselines": stats["baselines"], "cfd": cfd}
    write_json(output/"evaluation"/"E9_ch_sac_reference.json", result, replace=True)
    return result


def aggregate(output, cfg):
    expected = jobs(cfg)
    test = read_json(output/"evaluation_cases.json")
    identities = {sample["sample_id"] for sample in test["cases"]}
    runs, detailed = [], {}
    for arm, seed, tag in expected:
        key = case_key(arm, seed, tag)
        report = read_json(output/"evaluation"/f"{key}.json")
        if (report["arm"], report["seed"], report["tag"]) != (arm, seed, tag):
            raise ValueError(f"Evaluation file mislabeled: {key}")
        id_list = [row["sample_id"] for row in report["rows"]]
        if len(id_list) != len(identities) or set(id_list) != identities:
            raise ValueError(f"Missing or duplicate evaluation cases: {key}")
        details = {row["sample_id"]: {k: row[k] for k in
                   ("success", "reason", "mesh", "topology") if k in row} for row in report["rows"]}
        detailed[key] = details
        runs.append({"arm": arm, "seed": seed, "tag": tag, "step": report["step"],
                     **report["summary"], "failures": dict(Counter(
                         row["reason"] for row in details.values() if not row["success"])),
                     "distinct_topologies": len({json.dumps(row["topology"]) for row in details.values() if row["success"]})})
    comparisons = []
    for left, right in hypotheses():
        for seed in cfg["seeds"]:
            for budget in BUDGETS:
                l = detailed[case_key(left, seed, "best")]
                r = detailed[case_key(right, seed if right != "E1" else 42, "best" if right != "E1" else "rule")]
                comparisons.append({"left": left, "right": right, "seed": seed, "budget": budget, **paired(l, r, budget)})
    result = {"profile": cfg["profile"], "data": test["manifest"], "runs": runs, "paired": comparisons}
    write_json(output/"evaluation"/"summary.json", result, replace=True)
    benchmark_reference(output)
    return result
