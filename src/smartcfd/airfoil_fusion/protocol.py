"""Freeze data, source and engineering reachability before starting a study."""
from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import json
import subprocess

import numpy as np
import yaml

from smartcfd.airfoil_decomposition.data import digest, preflight, select_cases
from smartcfd.airfoil_decomposition.config import AirfoilDecompositionConfig
from smartcfd.airfoil_decomposition.experiment import file_sha, normalize_config, source_fingerprint, write_json
from smartcfd.partition.topology import fingerprint
from smartcfd.pipeline.data import load_samples

from .rollout import Environments, episode, replay

ROOT = Path(__file__).resolve().parents[3]
A_ARCHIVE = ROOT/"data/references/airfoil_fusion_a_v1"
B_COMMIT = "cbcad235a4dbb8b8d433a86b93f5daea0cda059c"
B_REL = "experiments/results/ch_sac_v2_20261125"
SAC_ARMS = ("E4", "E4a", "E6", "E7", "E8")
BASELINES = {"E1": "fixed", "E2": "random", "E3": "greedy"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def local_path(path):
    runs = (ROOT/"experiments/runs").resolve()
    path = (ROOT/path).resolve()
    if path == runs or path.parent != runs:
        raise ValueError("Use one fresh named directory immediately inside experiments/runs")
    return path


def current_source():
    source = source_fingerprint()
    source["files"].pop("src/smartcfd/airfoil_fusion/report.py", None)
    entry = ROOT/"scripts/run_airfoil_fusion.py"
    source["files"][entry.relative_to(ROOT).as_posix()] = file_sha(entry)
    source["source_sha256"] = digest(source["files"])
    return source


def published_bytes(name):
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "cat-file", "blob", f"{B_COMMIT}:{B_REL}/{name}"],
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Missing CH-SAC archive {name} at {B_COMMIT}. Use a full Git clone.") from exc


def reference_identities():
    a_paths = ("source.json", "frozen.json", "audit.json", "teachers/manifest.json", "report/summary.json")
    a = {"source_sha256": read_json(A_ARCHIVE/"source.json")["source_sha256"],
         "artifacts": {name: file_sha(A_ARCHIVE/name) for name in a_paths}}
    names = ["README.md", "ch_env.py", "ch_sac_v2_model.py", "prepare_dataset.py", "train_ch_sac_v2.py",
             "eval_and_cfd.py", "cfd_support.py", "dataset_manifest.json", "training_summary.json",
             "eval_summary.json", "baselines_summary.json", "cfd_verification_summary.json",
             "cfd_verification/naca0012_v2_5120/warm.cfg",
             "cfd_verification/naca0012_v2_5120/rans.cfg"]
    names.extend(f"seeds/seed_{seed}/{tag}.pt" for seed in range(42, 47) for tag in ("best", "last"))
    published = {}
    for name in names:
        blob = published_bytes(name)
        pinned = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", f"{B_COMMIT}:{B_REL}/{name}"], text=True
        ).strip()
        published[name] = {"git": pinned, "sha256": hashlib.sha256(blob).hexdigest()}
    return {"local": a, "github": {"commit": B_COMMIT, "artifacts": published}}


def load_config(path, profile):
    recipe = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    legacy = normalize_config(yaml.safe_load((ROOT/recipe["geometry_config"]).read_text(encoding="utf-8")))
    return {"profile": profile, **recipe[profile], "runtime": recipe["runtime"],
            "geometry": legacy["environment"], "max_workers": recipe["runtime"]["workers"]}


def verify_reachability(cases, checks, config):
    by_id = {r["sample_id"]: r for r in checks}
    legacy, fused, guarded = (Environments(config, arm, capacity=2) for arm in ("E4", "E5", "E8"))
    examples = []
    for sample in cases:
        if not by_id[sample["sample_id"]]["valid"]:
            continue
        traces, identities, choice_states = [], set(), []
        for method in ("fixed", "refined"):
            result, _ = episode(legacy, sample, method, mesh=False)
            if not result["success"]:
                continue
            descriptors = [row["candidate"] for row in result["actions"]]
            env, _, steps = replay(fused, sample, descriptors)
            tested = guarded.get(sample)
            tested.reset(seed=42)
            proofs = []
            from .rollout import resolve, stop_action

            for descriptor in [*descriptors, {"kind": "stop", "face_id": None}]:
                index = resolve(tested, descriptor)
                targets = {item["face_id"] for row, item in enumerate(tested.candidates)
                           if tested.action_mask[row] and item["face_id"] is not None}
                if len(targets.intersection(tested.queue)) >= 2:
                    choice_states.append({"step": tested.steps, "faces": sorted(targets),
                                          "unfinished_faces": list(tested.queue)})
                proofs.append({"step": tested.steps, "selected": index, "reason": tested._guard_report["actions"][index]["reason"],
                               "legal_stop": stop_action(tested) is not None})
                tested.step(index)
            if not tested.success or tested.graph.to_dict() != env.graph.to_dict():
                raise ValueError("Guard changed a demonstrated feasible completion")
            identity = tuple((protected, tuple((round(x, 8), round(y, 8)) for x, y in points))
                             for protected, points in fingerprint(env.graph))
            identities.add(identity)
            traces.append({"method": method, "edits": len(steps)-1, "outer_blocks": len(env.graph.faces)-4,
                           "graph": env.graph.to_dict(), "actions": descriptors, "guard": proofs})
        if len(identities) >= 2 and choice_states:
            examples.append({"sample_id": sample["sample_id"], "group_id": sample["group_id"],
                             "traces": traces, "choice_states": choice_states})
            print(f"reachability witness {len(examples)}/2 {sample['sample_id']}", flush=True)
        if len(examples) >= 2:
            return {"passed": True, "examples": examples}
    raise RuntimeError("Failed to witness two distinct reachable partitions on two training cases")


def prepare(output, recipe, profile):
    output = local_path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"prepared.json").exists():
        return open_study(output)
    cfg = load_config(recipe, profile)
    cases = select_cases(42, cfg["train_count"], cfg["val_count"], root=ROOT)
    identity = {"source": current_source(), "references": reference_identities()}
    checks = preflight(cases, AirfoilDecompositionConfig(**cfg["geometry"]))
    reachability = verify_reachability(cases["train"], checks["train"], cfg["geometry"])
    documents = {"config.json": cfg, "source.json": identity, "cases.json": cases,
                 "preflight.json": checks, "reachability.json": reachability}
    for filename, body in documents.items():
        write_json(output/filename, body, replace=(output/filename).exists())
    write_json(output/"prepared.json", {"profile": profile, "files": {name: file_sha(output/name) for name in documents}})
    return open_study(output)


def open_study(output):
    output = local_path(output)
    receipt = read_json(output/"prepared.json")
    for filename, sha in receipt["files"].items():
        if file_sha(output/filename) != sha:
            raise ValueError(f"Frozen study artifact modified: {filename}")
    source = read_json(output/"source.json")
    if source["source"]["source_sha256"] != current_source()["source_sha256"]:
        raise ValueError("Current source differs from the prepared study")
    return (output, read_json(output/"config.json"), read_json(output/"cases.json"), read_json(output/"preflight.json"))


def official_cases():
    ids = set(read_json(ROOT/"data/splits/uiuc_v1/test.json"))
    groups = read_json(ROOT/"data/splits/uiuc_v1/groups.json")
    membership = {sample: group for group in groups for sample in group["sample_ids"] if sample in ids}
    if len(ids) != len(membership) or any(group["split"] != "test" for group in membership.values()):
        raise ValueError("Invalid official family holdout")
    rows = [{**row, "split": "test", "group_id": membership[row["sample_id"]]["group_id"]}
            for row in load_samples("test", root=ROOT, sharp_only=True)]
    return rows, {"official_total": len(ids), "eligible": len(rows), "ineligible": len(ids)-len(rows),
                  "by_family": dict(Counter(row["group_id"] for row in rows)),
                  "excluded_ineligible_ids": sorted(ids-{row["sample_id"] for row in rows}),
                  "official_hash": file_sha(ROOT/"data/splits/uiuc_v1/test.json")}
