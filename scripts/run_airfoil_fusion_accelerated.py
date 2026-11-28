"""Run the frozen airfoil study with persistent, ordered validation workers."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import io
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from smartcfd.airfoil_decomposition.experiment import network_hashes, write_json
from smartcfd.airfoil_fusion import train
from smartcfd.airfoil_fusion.rollout import summarize

from fusion_validation_cache import ValidationTrajectories


_validation_shard = None


def _initialize_shard(geometry, arm, seed, samples, checks, threads):
    global _validation_shard
    torch.set_num_threads(threads)
    _validation_shard = (ValidationTrajectories(geometry, arm), train.make_agent(arm, seed), samples, checks)


def _evaluate_shard(weights, seed):
    trajectories, policy, samples, checks = _validation_shard
    train._restore_weights(policy, torch.load(io.BytesIO(weights), map_location="cpu", weights_only=True))
    return [trajectories.evaluate(sample, checks[sample["sample_id"]], policy, seed)
            for sample in samples]


class ParallelValidation:
    """One long-lived executor per shard keeps cases on the same workers."""

    def __init__(self, workers):
        self.workers = workers
        self.identity = None
        self.shards = []
        self.local_cache = None
        self.checks = None

    def close(self):
        for shard in self.shards:
            shard.shutdown(cancel_futures=True)
        self.shards.clear()
        self.identity = None
        self.local_cache = None
        self.checks = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def rows(self, cfg, cases, checked, arm, seed, weights, agent=None, progress=None):
        ids = [sample["sample_id"] for sample in cases]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Validation requires an ordered manifest of unique cases")
        identity = (cfg["geometry"], cfg["runtime"]["torch_threads"], cases, checked, arm, seed)
        if self.identity != identity:
            self.close()
            check_by_id = {row["sample_id"]: row for row in checked}
            if set(check_by_id) != set(ids):
                raise ValueError("Validation preflight does not match the case manifest")
            count = min(self.workers, len(cases))
            if count == 1:
                self.local_cache = ValidationTrajectories(cfg["geometry"], arm)
                self.checks = check_by_id
            else:
                for index in range(count):
                    samples = cases[len(cases) * index // count:len(cases) * (index + 1) // count]
                    self.shards.append(ProcessPoolExecutor(
                        max_workers=1, mp_context=multiprocessing.get_context("spawn"), initializer=_initialize_shard,
                        initargs=(cfg["geometry"], arm, seed, samples, check_by_id, cfg["runtime"]["torch_threads"]),
                    ))
            self.identity = identity
        if self.local_cache is not None:
            if agent is None:
                raise ValueError("In-process validation requires the training agent")
            def evaluate(sample):
                return self.local_cache.evaluate(sample, self.checks[sample["sample_id"]], agent, seed)

            return progress.rows(cases, evaluate) if progress is not None else [evaluate(sample) for sample in cases]
        payload = io.BytesIO()
        torch.save(weights, payload)
        payload = payload.getvalue()
        futures = [shard.submit(_evaluate_shard, payload, seed) for shard in self.shards]
        rows = [row for future in futures for row in future.result()]
        if [row["sample_id"] for row in rows] != ids:
            raise ValueError("Parallel validation changed case order")
        return rows

    def validate(self, directory, cfg, cases, checked, arm, seed, agent, binding, step):
        directory.mkdir(exist_ok=True)
        start = time.perf_counter()
        hashes = network_hashes(agent)
        weights = train._weights(agent, binding, step)
        progress = None
        if self.workers == 1:
            from fusion_validation_progress import ValidationCaseReceipts

            progress = ValidationCaseReceipts(directory, binding, hashes, step)
        rows = self.rows(cfg, cases, checked, arm, seed, weights, agent, progress)
        if any(row["decisions"] for row in rows):
            agent.actor.eval()
        if hashes != network_hashes(agent):
            raise ValueError("Validation changed policy tensors")
        report = {"step": step, **summarize(rows), "network_sha256": hashes}
        runtime = {"workers": len(self.shards) or 1, "validation_elapsed_seconds": time.perf_counter() - start,
                   "reused_trajectories": sum(row.get("trajectory_reused", False) for row in rows),
                   "resumed_case_receipts": progress.reused if progress is not None else 0}
        write_json(directory / "report.json", {**report, "per_case": rows, "validation_runtime": runtime}, replace=True)
        train.save_torch(directory / "weights.pt", weights)
        print(f"{arm} seed={seed} validation step={step} success={report['successful']}/{report['total']} "
              f"workers={runtime['workers']} wall={runtime['validation_elapsed_seconds']:.2f}s "
              f"reused_trajectories={runtime['reused_trajectories']}", flush=True)
        return report


def _publish_queue_state(path, value, *, replace=False):
    if os.name != "nt" or not replace or not path.name.startswith("queue_"):
        write_json(path, value, replace=replace)
        return
    temporary = path.with_suffix(".json.tmp")
    write_json(temporary, value)
    deadline = time.monotonic() + 30
    delay = 0.05
    while True:
        try:
            temporary.replace(path)
            return
        except PermissionError as exc:
            if exc.winerror not in (5, 32, 33) or time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(2 * delay, 0.5)


def _configure_supervisor(workers, cache_geometry):
    from smartcfd.airfoil_fusion import runner

    frozen_entry = ROOT / "scripts" / "run_airfoil_fusion.py"
    accelerated_entry = Path(__file__).resolve()

    def launch(command, **kwargs):
        if len(command) > 3 and Path(command[3]) == frozen_entry and "--stage" in command:
            stage = command[command.index("--stage") + 1]
            if stage in ("bc", "train") or (stage == "evaluate" and cache_geometry):
                command = [*command[:3], str(accelerated_entry), *command[4:], "--validation-workers", str(workers)]
                if cache_geometry:
                    command.append("--cache-geometry")
        return subprocess.Popen(command, **kwargs)

    # Adapt only the supervisor's launch and queue state publication. The frozen
    # training, evaluation, matrix extension and CFD entry points remain in use.
    runner.subprocess = SimpleNamespace(Popen=launch, STDOUT=subprocess.STDOUT,
                                        CREATE_NO_WINDOW=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    runner.write_json = _publish_queue_state


def main():
    import run_airfoil_fusion

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--validation-workers", default=1, type=int)
    parser.add_argument("--cache-geometry", action="store_true")
    parser.add_argument("--stage", default="all")
    parser.add_argument("--arm", default="E5")
    options, remaining = parser.parse_known_args()
    if not 1 <= options.validation_workers <= 4:
        parser.error("--validation-workers must be between 1 and 4")
    sys.argv[1:] = [*remaining, "--stage", options.stage, "--arm", options.arm]
    if options.stage == "all":
        _configure_supervisor(options.validation_workers, options.cache_geometry)
        run_airfoil_fusion.main()
        return
    if options.stage == "evaluate":
        from fusion_evaluation_progress import install_evaluation_progress

        install_evaluation_progress()
    if options.cache_geometry and options.stage in ("train", "evaluate") and options.arm == "E8":
        from fusion_geometry_cache import install_geometry_cache

        install_geometry_cache()
    with ParallelValidation(options.validation_workers) as validation:
        train._validate = validation.validate
        run_airfoil_fusion.main()


if __name__ == "__main__":
    main()
