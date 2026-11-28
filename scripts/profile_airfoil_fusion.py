"""Measure frozen auto-partitioning code without writing to an active run."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import cProfile
import json
from pathlib import Path
import pstats
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from smartcfd.airfoil_decomposition.experiment import _json_default, file_sha, write_json
from smartcfd.airfoil_fusion.learning import FusionSAC
from smartcfd.airfoil_fusion.protocol import open_study, read_json
from smartcfd.airfoil_fusion.rollout import Environments, episode, resolve, selection_key, summarize
from smartcfd.airfoil_fusion.train import get_inference_agent, make_agent, state_hash


def timed(operation):
    wall, cpu = time.perf_counter(), time.process_time()
    value = operation()
    return value, {"wall": time.perf_counter() - wall, "cpu": time.process_time() - cpu}


def update_times(run, cfg, device="cpu"):
    path = run / "teachers" / "plain.pt"
    groups = torch.load(path, map_location="cpu", weights_only=False)
    rows = [step for group in groups.values() for trajectory in group for step in trajectory]
    agent = make_agent("E7", 42) if device == "cpu" else FusionSAC(seed=42, device=device)
    rng = random.Random(42)
    batches = [rng.sample(rows, cfg["runtime"]["batch_size"]) for _ in range(44)]
    def update(batch):
        result = agent.update(batch)
        if device != "cpu":
            torch.cuda.synchronize()
        return result

    for batch in batches[:4]:
        update(batch)
    results = [timed(lambda: update(batch))[1] for batch in batches[4:]]
    parameter_count = sum(parameter.numel() for parameter in agent.actor.parameters())
    return {"scope": "40 SAC updates on fixed batches of real training demonstration transitions after 4 warmups",
            "device": str(agent.device), "torch_version": torch.__version__, "cuda_available": torch.cuda.is_available(),
            "actor_parameters": parameter_count, "batch_size": cfg["runtime"]["batch_size"],
            "median": {field: statistics.median(r[field] for r in results) for field in ("wall", "cpu")}}


def deterministic_rollouts(run, cfg, cases, checks):
    pool = Environments(cfg["geometry"], "E8")
    sample = cases["val"][0]
    prechecked = next(row for row in checks["val"] if row["sample_id"] == sample["sample_id"])
    checkpoint = run / "arms" / "E8_seed42" / "validation_000000" / "weights.pt"
    agent, step = get_inference_agent(checkpoint, "E8", 42, file_sha(run / "prepared.json"))
    archived = read_json(checkpoint.parent / "report.json")["per_case"][0]
    assert archived["sample_id"] == sample["sample_id"] and step == 0

    def run_case():
        outcome, transitions = episode(pool, sample, "learned", agent=agent, seed=42, prechecked=prechecked, capture=True)
        signature = [(state_hash(t.observation), t.action, t.reward, state_hash(t.next_observation),
                      t.terminated, t.truncated) for t in transitions]
        return outcome, signature

    (cold, left), cold_times = timed(run_case)
    (warm, right), warm_times = timed(run_case)
    assert cold["graph"] == warm["graph"] == archived["graph"]
    assert cold["success"] == warm["success"] and cold["actions"] == warm["actions"] == archived["actions"]
    assert cold["mesh"]["1536"]["score"] == archived["mesh"]["1536"]["score"]
    assert cold["mesh"]["5120"]["score"] == archived["mesh"]["5120"]["score"] and left == right
    return {"sample_id": sample["sample_id"], "checkpoint": checkpoint.relative_to(run).as_posix(),
            "checkpoint_sha256": file_sha(checkpoint), "case_matched": True,
            "cold": cold_times | {"work": cold["work"], "guard": cold["guard_work"]},
            "warm": warm_times | {"work": warm["work"], "guard": warm["guard_work"]}}


def profile_primitives(cfg, cases, checks):
    sample = next(sample for sample, check in zip(cases["train"], checks["train"], strict=True) if check["valid"])
    pool = Environments(cfg["geometry"], "E8")
    env = pool.get(sample)
    profiler = cProfile.Profile()
    profiler.enable()
    env.reset(seed=42)
    first = next(index for index, row in enumerate(env.candidates)
                 if row["kind"] == "insert_edge_point" and env.action_mask[index])
    description = env.candidates[first]
    env.step(resolve(env, description))
    profiler.disable()
    stats = pstats.Stats(profiler)
    ranked = sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)
    return {"sample_id": sample["sample_id"], "edits": 1, "profiled_seconds": stats.total_tt,
            "top_cumulative": [{"file": Path(file).name, "line": line, "function": func,
                                "calls": item[1], "self_seconds": item[2], "cumulative_seconds": item[3]}
                               for (file, line, func), item in ranked[:40]]}


def checkpoint_validation(run, cfg, samples, preflight, snapshots, reuse):
    """Evaluate one shard in frozen order, optionally retaining its per-geometry caches."""
    torch.set_num_threads(cfg["runtime"]["torch_threads"])
    cached = Environments(cfg["geometry"], "E8", capacity=len(samples)) if reuse else None
    rows = []
    for seed, target in snapshots:
        folder = run / "arms" / f"E8_seed{seed}" / f"validation_{target:06d}"
        agent, step = get_inference_agent(folder / "weights.pt", "E8", seed, file_sha(run / "prepared.json"))
        assert step == target
        pool = cached if cached is not None else Environments(cfg["geometry"], "E8")
        outcomes = [episode(pool, sample, "learned", agent=agent, seed=seed, prechecked=preflight[sample["sample_id"]])[0]
                    for sample in samples]
        rows.append(outcomes)
    return rows


def stable_rows(outcomes):
    transient = {"seconds", "work", "guard_work", "trajectory_reused"}
    stable = [{key: value for key, value in row.items() if key not in transient} for row in outcomes]
    return json.loads(json.dumps(stable, default=_json_default, allow_nan=False))


def validate_parallelism(run, cfg, cases, checks):
    indices = [0, 3, 8, 12, 18, 23, 27, 31]
    samples = [cases["val"][i] for i in indices]
    preflight = {row["sample_id"]: row for row in checks["val"]}
    snapshots = [(42, 0), (42, 500), (43, 1000)]
    cold, cold_time = timed(lambda: checkpoint_validation(run, cfg, samples, preflight, snapshots, False))
    cached, cached_time = timed(lambda: checkpoint_validation(run, cfg, samples, preflight, snapshots, True))
    with ProcessPoolExecutor(max_workers=2) as pool:
        def validate_shards():
            jobs = [pool.submit(checkpoint_validation, run, cfg, samples[index:index + 4], preflight, snapshots, True)
                    for index in (0, 4)]
            shards = [job.result() for job in jobs]
            return [first + second for first, second in zip(*shards, strict=True)]

        parallel, parallel_time = timed(validate_shards)
    per_checkpoint = []
    for (seed, checkpoint), a, b, c in zip(snapshots, cold, cached, parallel, strict=True):
        path = run / "arms" / f"E8_seed{seed}" / f"validation_{checkpoint:06d}" / "report.json"
        original = [read_json(path)["per_case"][i] for i in indices]
        assert stable_rows(a) == stable_rows(b) == stable_rows(c) == stable_rows(original)
        assert selection_key({"step": checkpoint, **summarize(a)}) == selection_key({"step": checkpoint, **summarize(c)})
        per_checkpoint.append({"seed": seed, "checkpoint": checkpoint, "report_sha256": file_sha(path),
                               "cases": len(a), "successful": sum(row["success"] for row in a),
                               "cold_cache_hits": sum(row["guard_work"].get("guard_cache_hit", 0) for row in a),
                               "reused_cache_hits": sum(row["guard_work"].get("guard_cache_hit", 0) for row in b)})
    return {"scope": "eight development validation cases across three archived E8 checkpoints; two ordered CPU shards",
            "exact_equality": "success, actions, rewards, terminal graph, quality, both mesh budgets and checkpoint selection metrics",
            "serial_cold": cold_time, "serial_cached": cached_time,
            "parallel_cached": {"wall": parallel_time["wall"], "parent_cpu_only": parallel_time["cpu"]},
            "per_checkpoint": per_checkpoint}


def verify_accelerated_runner(run, cfg, cases, checks):
    from run_airfoil_fusion_accelerated import ParallelValidation

    results = []
    groups = (("E8", 42, (0, 500), list(range(32))), ("E8", 43, (0, 500, 1000), [0, 31]),
              ("E4", 42, (0, 2000), [3, 18]))
    for arm, seed, steps, indices in groups:
        samples = [cases["val"][i] for i in indices]
        ids = {sample["sample_id"] for sample in samples}
        checked = [row for row in checks["val"] if row["sample_id"] in ids]
        with ParallelValidation(2) as validation:
            for step in steps:
                path = run / "arms" / f"{arm}_seed{seed}" / f"validation_{step:06d}"
                archived = read_json(path / "report.json")["per_case"]
                weights = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
                assert weights["step"] == step and weights["binding"]["arm"] == arm
                rows, elapsed = timed(lambda: validation.rows(cfg, samples, checked, arm, seed, weights))
                expected = [archived[i] for i in indices]
                assert stable_rows(rows) == stable_rows(expected)
                assert selection_key({"step": step, **summarize(rows)}) == selection_key({"step": step, **summarize(expected)})
                results.append({"arm": arm, "seed": seed, "step": step, "cases": len(rows),
                                "successes": sum(row["success"] for row in rows), "wall_seconds": elapsed["wall"],
                                "archive_case_seconds": sum(row["seconds"] for row in expected),
                                "weights_sha256": file_sha(path / "weights.pt"), "report_sha256": file_sha(path / "report.json"),
                                "guard_cache_hits": sum(row.get("guard_work", {}).get("guard_cache_hit", 0) for row in rows)})
                print(json.dumps(results[-1]), flush=True)
    return {"passed": True, "launcher_sha256": file_sha(ROOT / "scripts" / "run_airfoil_fusion_accelerated.py"),
            "records": results}


def verify_compact_cache(run, cfg, cases, checks):
    from run_airfoil_fusion_accelerated import ParallelValidation

    records = []
    for arm, seed, steps, indices in (("E8", 42, (0, 500), [0, 9, 22, 31]),
                                     ("E8", 43, (0, 500, 1000), [0]), ("E4", 42, (0, 2000, 2500), [0, 22])):
        samples = [cases["val"][i] for i in indices]
        ids = {sample["sample_id"] for sample in samples}
        checked = [row for row in checks["val"] if row["sample_id"] in ids]
        with ParallelValidation(1) as validation:
            for step in steps:
                path = run / "arms" / f"{arm}_seed{seed}" / f"validation_{step:06d}"
                agent, actual_step = get_inference_agent(path / "weights.pt", arm, seed, file_sha(run / "prepared.json"))
                assert actual_step == step
                rows, elapsed = timed(lambda: validation.rows(cfg, samples, checked, arm, seed, None, agent))
                archived = read_json(path / "report.json")["per_case"]
                assert stable_rows(rows) == stable_rows([archived[i] for i in indices])
                cache = validation.local_cache
                assert cache.size <= cache.byte_limit
                records.append({"arm": arm, "seed": seed, "step": step, "cases": len(rows),
                                "successes": sum(row["success"] for row in rows),
                                "retained_observation_bytes": cache.size, "wall_seconds": elapsed["wall"],
                                "reused": sum(row.get("trajectory_reused", False) for row in rows)})
                print(json.dumps(records[-1]), flush=True)
    assert any(row["reused"] for row in records)
    return {"passed": True, "scope": "distinct and repeated deterministic trajectories, including failures",
            "launcher_sha256": file_sha(ROOT / "scripts" / "run_airfoil_fusion_accelerated.py"),
            "cache_sha256": file_sha(ROOT / "scripts" / "fusion_validation_cache.py"), "records": records}


def verify_online_memo(run, cfg, cases):
    from fusion_geometry_cache import CurveGeometry, install_geometry_cache

    inputs = []
    by_sample = {row["sample_id"]: row for row in cases["train"]}
    for seed in (42, 43):
        path = run / "arms" / f"E8_seed{seed}" / "interactions.jsonl"
        with path.open(encoding="utf-8") as file:
            rows = [json.loads(line) for _, line in zip(range(110), file)]
        for episode_number in (1, 2):
            episode_rows = [row for row in rows if row["episode"] == episode_number][:8]
            assert episode_rows and len({row["sample_id"] for row in episode_rows}) == 1
            inputs.append((seed, episode_number, by_sample[episode_rows[0]["sample_id"]], episode_rows))

    def transitions(seed, episode_number, sample, rows):
        env = Environments(cfg["geometry"], "E8").get(sample)
        obs, _ = env.reset(seed=seed + episode_number)
        for row in rows:
            assert state_hash(obs) == row["observation"]
            assert [i for i, value in enumerate(obs["mask"]) if value] == row["effective_actions"]
            guards = [{**data, "kept": bool(env.action_mask[data["action"]])} for data in env._guard_report["actions"]]
            assert guards == row["guard"]["actions"]
            obs, reward, done, truncated, info = env.step(resolve(env, row["candidate"]))
            assert state_hash(obs) == row["next_observation"] and reward == row["reward"]
            assert (done, truncated, info["reason"]) == (row["terminated"], row["truncated"], row["reason"])
        return getattr(env.geometry, "_exact_validation_work", {})

    results = []
    for args in inputs:
        _, original = timed(lambda: transitions(*args))
        base = install_geometry_cache()
        try:
            work, accelerated = timed(lambda: transitions(*args))
        finally:
            CurveGeometry.validate = base
        seed, episode_number, sample, rows = args
        record = {"seed": seed, "episode": episode_number, "case": sample["sample_id"], "steps": len(rows),
                  "original": original, "accelerated": accelerated, **work}
        results.append(record)
        print(json.dumps(record), flush=True)
    return {"passed": True, "scope": "same logged action, state, reward, mask and guard proof",
            "cache_sha256": file_sha(ROOT / "scripts" / "fusion_geometry_cache.py"), "records": results}


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--output", required=True, type=Path)
    mode = args.add_mutually_exclusive_group()
    mode.add_argument("--parallel-validation", action="store_true")
    mode.add_argument("--production-gate", action="store_true")
    mode.add_argument("--compact-cache", action="store_true")
    mode.add_argument("--online-memo", action="store_true")
    options = args.parse_args()
    run, cfg, cases, checks = open_study("experiments/runs/airfoil_fusion_formal_v2")
    torch.set_num_threads(cfg["runtime"]["torch_threads"])
    result = {"source_manifest_sha256": file_sha(run / "source.json")}
    operations = (("sac_updates", lambda: update_times(run, cfg)),
                  ("repeated_validation_case", lambda: deterministic_rollouts(run, cfg, cases, checks)),
                  ("guarded_primitive_profile", lambda: profile_primitives(cfg, cases, checks)))
    if options.parallel_validation:
        operations = (("validation_prototype", lambda: validate_parallelism(run, cfg, cases, checks)),)
        if torch.cuda.is_available():
            operations = (("cuda_update_comparison", lambda: update_times(run, cfg, "cuda")), *operations)
    if options.production_gate:
        operations = (("production_gate", lambda: verify_accelerated_runner(run, cfg, cases, checks)),)
    if options.compact_cache:
        operations = (("compact_cache", lambda: verify_compact_cache(run, cfg, cases, checks)),)
    if options.online_memo:
        operations = (("online_memo", lambda: verify_online_memo(run, cfg, cases)),)
    for name, execute in operations:
        result[name] = execute()
        write_json(options.output, result, replace=True)
        print(json.dumps({name: result[name]}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
