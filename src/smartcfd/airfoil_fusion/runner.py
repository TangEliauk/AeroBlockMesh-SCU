"""Durable experiment stages and bounded local worker supervision."""
from __future__ import annotations

from collections import deque
from pathlib import Path
import os
import subprocess
import sys
import time

import torch

from smartcfd.airfoil_decomposition.experiment import file_sha, write_json

from .protocol import A_ARCHIVE, ROOT, SAC_ARMS, local_path, open_study, prepare, read_json
from .rollout import Environments
from .teachers import audit_historical, build, teacher_groups
from .train import pretrain, save_torch, train


def teachers(output, cfg, cases, checks):
    output = Path(output)
    directory = output/"teachers"
    directory.mkdir(exist_ok=True)
    history = directory/"historical_audit.json"
    if not history.exists():
        result = audit_historical(A_ARCHIVE, history, cfg["geometry"])
        print(f"Verified {result['count']} historical rule demonstrations", flush=True)
    manifest = build(cases["train"], checks["train"], cfg["geometry"], directory, workers=cfg["max_workers"])
    for name, arm in (("plain", "E5"), ("guarded", "E8")):
        path = directory/f"{name}.pt"
        receipt = directory/f"{name}.json"
        if receipt.exists():
            _verify_teacher_cache(directory, name)
            continue
        groups = teacher_groups(manifest, cases["train"], Environments(cfg["geometry"], arm))
        save_torch(path, groups)
        write_json(receipt, {"teachers_sha256": file_sha(directory/"manifest.json"), "sha256": file_sha(path),
                             "families": len(groups), "transitions": sum(len(t) for seqs in groups.values() for t in seqs)})
        print(f"Replayed all {name} demonstrations", flush=True)
    return manifest


def _verify_teacher_cache(directory, name):
    path = directory/f"{name}.pt"
    receipt = read_json(directory/f"{name}.json")
    if receipt["sha256"] != file_sha(path) or receipt["teachers_sha256"] != file_sha(directory/"manifest.json"):
        raise ValueError(f"Teacher replay cache does not match evidence: {name}")
    return path


def worker(output, stage, arm, seed, value):
    output, cfg, cases, checks = open_study(output)
    if seed not in cfg["seeds"]:
        raise ValueError(f"Seed {seed} was not frozen")
    if stage in ("bc", "train") and (output/"evaluation_lock.json").exists():
        raise ValueError("Training updates are closed for this sealed test run")
    if stage == "bc":
        groups = torch.load(_verify_teacher_cache(output/"teachers", "plain"), weights_only=False, map_location="cpu")
        return pretrain(output, cfg, cases, checks, groups, seed)
    if stage == "train":
        if arm not in SAC_ARMS:
            raise ValueError(f"Unknown SAC training arm: {arm}")
        groups = None
        if arm in ("E7", "E8"):
            name = "guarded" if arm == "E8" else "plain"
            groups = torch.load(_verify_teacher_cache(output/"teachers", name), weights_only=False, map_location="cpu")
        return train(output, cfg, cases, checks, arm, seed, int(value), groups)
    if stage == "evaluate":
        from .evaluate import evaluate, jobs

        if (arm, seed, value) not in jobs(cfg):
            raise ValueError("Evaluation job was not frozen")
        return evaluate(output, cfg, arm, seed, value)
    raise ValueError(f"Unknown worker stage: {stage}")


def _drive(output, name, tasks, workers):
    logdir = output/"queue_logs"
    logdir.mkdir(exist_ok=True)
    pending = deque((task, 1) for task in tasks)
    active, statuses, failures = {}, {}, []
    interrupted = False
    entry = ROOT/"scripts/run_airfoil_fusion.py"
    state_path = output/f"queue_{name}.json"
    while pending or active:
        interrupted = interrupted or (output/"STOP_REQUESTED").exists()
        while pending and len(active) < workers and not interrupted:
            task, attempt = pending.popleft()
            stage, arm, seed, value = task
            key = f"{arm}_{seed}_{value}"
            log = logdir/f"{name}_{key}_a{attempt}.log"
            with log.open("ab") as stream:
                process = subprocess.Popen(
                    [sys.executable, "-B", "-u", str(entry), "--output", str(output), "--stage", stage,
                     "--arm", arm, "--seed", str(seed), "--value", str(value)],
                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            active[key] = process, task, attempt, log
            statuses[key] = {"status": "running", "pid": process.pid, "log": log.name, "attempt": attempt}
            print(f"{name} started {key} PID={process.pid}", flush=True)
        for key, (process, task, attempt, log) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            del active[key]
            statuses[key].update(exit_code=code, status="completed" if code == 0 else "failed")
            print(f"{name} {key} exit={code}", flush=True)
            if code:
                if attempt < 2 and not interrupted:
                    pending.append((task, attempt+1))
                else:
                    failures.append(log.name)
        state = {"stage": name, "status": "stopping" if interrupted else "running",
                 "jobs": statuses, "pending": len(pending), "live_children": {k: t[0].pid for k, t in active.items()}}
        write_json(state_path, state, replace=True)
        if interrupted and not active:
            break
        if pending or active:
            time.sleep(2)
    state.update(status="stopped" if interrupted else "failed" if failures else "completed", failures=failures)
    write_json(state_path, state, replace=True)
    if interrupted:
        raise InterruptedError(f"Local queue stopped during {name}")
    if failures:
        raise RuntimeError(f"Worker failures in {name}: {failures}")


def _matrix(output, cfg):
    from .train import _verify_outputs

    def collect(arm, seed):
        path = output/"arms"/f"{arm}_seed{seed}"/"completion.json"
        receipt = read_json(path)
        _verify_outputs(path.parent, receipt)
        return receipt

    path = output/"training_matrix.json"
    if path.exists():
        return read_json(path)
    workers, seeds = cfg["max_workers"], cfg["seeds"]
    _drive(output, "bc", [("bc", "E5", seed, 0) for seed in seeds], workers)
    first, extended = cfg["first_steps"], cfg["extended_steps"]
    tasks = [(arm, seed) for seed in seeds for arm in ("E8", "E4", "E4a", "E6", "E7")]
    _drive(output, "first", [("train", arm, seed, first) for arm, seed in tasks], workers)
    decisions = {}
    for arm in ("E6", "E7", "E8"):
        decisions[arm] = {}
        for seed in seeds:
            history = {row["step"]: row["successful"] for row in collect(arm, seed)["validation"]}
            decisions[arm][str(seed)] = history[first]-history[cfg["extension_compare_step"]]
    longer = first < extended and any(sum(v > 0 for v in row.values()) >= 2 for row in decisions.values())
    final = extended if longer else first
    if longer:
        _drive(output, "extended", [("train", arm, seed, final) for arm, seed in tasks], workers)
    checkpoints, receipts = {}, {}
    for arm, seed in [*tasks, *(("E5", seed) for seed in seeds)]:
        result = collect(arm, seed)
        if result["online_steps"] != (0 if arm == "E5" else final):
            raise RuntimeError(f"Training horizon differs across the frozen matrix: {arm} {seed}")
        root = f"arms/{arm}_seed{seed}"
        receipts[f"{arm}_{seed}"] = file_sha(output/root/"completion.json")
        checkpoints.update({f"{root}/{name}": sha for name, sha in result["checkpoints"].items()})
    payload = {"profile": cfg["profile"], "extended": longer, "online_steps": final, "extension_deltas": decisions,
               "checkpoints": checkpoints, "receipt_hashes": receipts}
    write_json(path, payload, replace=True)
    return payload


def pipeline(output, recipe, profile, smoke_evidence):
    from .cfd import run as run_cfd
    from .evaluate import aggregate, jobs, lock_testing

    output = local_path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output/".queue.lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt

            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        if profile == "formal":
            smoke, smoke_cfg, _, _ = open_study(smoke_evidence)
            audit = read_json(smoke/"audit.json")
            if smoke_cfg["profile"] != "smoke" or not audit["passed"] or audit.get("solver_runs", 0) < 1:
                raise ValueError("Formal run requires the verified smoke matrix and a real solver run")
        output, cfg, cases, checks = prepare(output, recipe, profile)
        if cfg["profile"] != profile:
            raise ValueError("Requested profile differs from the existing run")
        if not (output/"training_matrix.json").exists():
            teachers(output, cfg, cases, checks)
            _matrix(output, cfg)
        lock_testing(output, cfg, cases, checks)
        _drive(output, "evaluation", [("evaluate", arm, seed, tag) for arm, seed, tag in jobs(cfg)], cfg["max_workers"])
        aggregate(output, cfg)
        run_cfd(output, cfg)
        from .report import build_report, audit_run

        build_report(output, cfg)
        return audit_run(output, cfg)
