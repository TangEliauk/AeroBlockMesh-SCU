"""Training-side demonstrations and audit of historical rule trajectories."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from smartcfd.airfoil_decomposition.experiment import file_sha, write_json
from smartcfd.airfoil_decomposition.study import TEACHER_SCHEMA
from smartcfd.airfoil_decomposition.data import digest

from .rollout import Environments, episode, replay


def audit_historical(archive, output, config):
    archive = Path(archive)
    from .protocol import read_json

    samples = {row["sample_id"]: row for row in read_json(archive/"cases.json")["train"]}
    manifest_path = archive/"teachers/manifest.json"
    manifest = read_json(manifest_path)
    content = {key: value for key, value in manifest.items() if key != "dataset_sha256"}
    if manifest["schema"] != TEACHER_SCHEMA or digest(content) != manifest["dataset_sha256"]:
        raise ValueError("Historical teacher manifest identity mismatch")
    teacher_dir = manifest_path.parent.resolve()
    pool = Environments(config, "E5")
    rows = []
    for entry in manifest["selected"]:
        evidence_path = (teacher_dir/entry["evidence"]).resolve()
        if not evidence_path.is_relative_to(teacher_dir) or file_sha(evidence_path) != entry["evidence_sha256"]:
            raise ValueError("Historical teacher evidence path/hash mismatch")
        evidence = read_json(evidence_path)
        sample = samples[entry["sample_id"]]
        if sample["group_id"] != entry["group_id"]:
            raise ValueError("Historical teacher family differs from the frozen source")
        descriptors = [step["candidate"] for step in evidence["actions"]]
        env, _, proposed = replay(pool, sample, descriptors)
        rows.append({"sample_id": sample["sample_id"], "group_id": sample["group_id"],
                     "source_sha256": entry["evidence_sha256"], "old_actions": len(descriptors),
                     "new_decisions": len(proposed), "graph_sha256": digest(env.graph.to_dict()),
                     "matches_historical_topology": env.graph.to_dict() == evidence["partition"]["graph"]})
    result = {"count": len(rows), "teacher_manifest_sha256": file_sha(manifest_path),
              "families": len({row["group_id"] for row in rows}), "rows": rows}
    if not rows or not all(row["matches_historical_topology"] for row in rows):
        raise ValueError("Historical rule action audit did not preserve all terminal graphs")
    write_json(output, result)
    return result


def _case_teacher(sample, check, config):
    fused, legacy = Environments(config, "E5"), Environments(config, "E4")
    candidates, attempts = [], []
    for method, pool in (("fixed", fused), ("greedy", fused), ("refined", legacy)):
        outcome, _ = episode(pool, sample, method, prechecked=check, mesh=False)
        row = {"method": method, "reason": outcome["reason"], "success": outcome["success"],
               "seconds": outcome["seconds"],
               "actions": [transition["candidate"] for transition in outcome.get("actions", [])]}
        if outcome["success"]:
            env, _, steps = replay(fused, sample, row["actions"])
            if row["actions"][-1]["kind"] != "stop":
                row["actions"].append({"kind": "stop", "face_id": None})
            row.update(decisions=len(steps), graph_sha256=digest(env.graph.to_dict()),
                       score=outcome["quality"]["q"]-.02*outcome["outer_blocks"])
            candidates.append(row)
        attempts.append(row)
    selected = max(candidates, key=lambda r: (r["score"], -r["decisions"]), default=None)
    return {"sample_id": sample["sample_id"], "group_id": sample["group_id"],
            "selected": selected, "attempts": attempts}


def build(cases, preflight, config, output, *, workers=3):
    from .protocol import read_json

    output = Path(output)
    output.mkdir(exist_ok=True)
    archive = output/"manifest.json"
    if archive.exists():
        result = read_json(archive)
        if result["manifest_sha256"] != digest(result["rows"]):
            raise ValueError("Teacher manifest content changed")
        return result
    prechecked = {row["sample_id"]: row for row in preflight}
    if any(sample["split"] != "train" for sample in cases):
        raise ValueError("Teacher generation is restricted to the training split")
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future = {}
        for sample in cases:
            sid = sample["sample_id"]
            if not (output/f"{sid}.json").exists():
                future[sid] = executor.submit(_case_teacher, sample, prechecked[sid], config)
        for number, sample in enumerate(cases, 1):
            sid = sample["sample_id"]
            evidence = output/f"{sid}.json"
            if evidence.exists():
                row = read_json(evidence)
            else:
                row = future[sid].result()
                write_json(evidence, row)
            rows.append(row)
            selected = row["selected"]
            print(f"teachers {number}/{len(cases)} {sid} {selected['method'] if selected else 'unavailable'}", flush=True)
    result = {"rows": rows, "total": len(rows), "available": sum(row["selected"] is not None for row in rows),
              "manifest_sha256": digest(rows)}
    if not result["available"]:
        raise RuntimeError("No complete successful training demonstration")
    write_json(archive, result)
    return result


def teacher_groups(manifest, samples, pool):
    samples = {sample["sample_id"]: sample for sample in samples}
    groups = {}
    for row in manifest["rows"]:
        if row["selected"] is not None:
            sample = samples[row["sample_id"]]
            if sample["split"] != "train" or row["group_id"] != sample["group_id"]:
                raise ValueError("Demonstration does not match the frozen training split")
            env, _, sequence = replay(pool, sample, row["selected"]["actions"])
            if digest(env.graph.to_dict()) != row["selected"]["graph_sha256"]:
                raise ValueError(f"Teacher topology changed for {sample['sample_id']}")
            groups.setdefault(row["group_id"], []).append(sequence)
    return groups
