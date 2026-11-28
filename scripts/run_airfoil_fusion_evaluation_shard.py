"""Evaluate a range of sealed E8 cases and publish the usual case receipts."""

import argparse
from copy import deepcopy
import os
from pathlib import Path

import torch

from fusion_evaluation_progress import _digest, _save_receipt
from fusion_geometry_cache import install_geometry_cache
from smartcfd.airfoil_decomposition.experiment import file_sha
from smartcfd.airfoil_fusion.evaluate import case_key, jobs
from smartcfd.airfoil_fusion.protocol import open_study, read_json
from smartcfd.airfoil_fusion.rollout import Environments, episode
from smartcfd.airfoil_fusion.train import get_inference_agent


def _same_measurement(left, right):
    def without_wall_time(row):
        value = deepcopy(row)
        value.pop("seconds")
        value.get("guard_work", {}).pop("elapsed_seconds", None)
        return value

    return without_wall_time(left) == without_wall_time(right)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tag", choices=("best", "final"), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify", type=int, metavar="POSITION", help="replay an existing receipt without publishing a case")
    mode.add_argument("--range", type=int, nargs=2, metavar=("FIRST", "LAST"), help="inclusive one-based positions")
    args = parser.parse_args()

    output, cfg, _, _ = open_study(args.output)
    arm, seed, tag = "E8", args.seed, args.tag
    if cfg["profile"] != "formal" or (arm, seed, tag) not in jobs(cfg):
        parser.error("the job must belong to the formal E8 evaluation matrix")
    lock_path = output/"evaluation_lock.json"
    cases_path = output/"evaluation_cases.json"
    training_path = output/"training_matrix.json"
    lock = read_json(lock_path)
    if (file_sha(cases_path) != lock["cases_sha256"] or file_sha(training_path) != lock["training_sha256"]
            or lock["profile"] != cfg["profile"]):
        raise ValueError("Evaluation input changed after the lock")
    cohort = read_json(cases_path)
    cases = cohort["cases"]
    preflight = {row["sample_id"]: row for row in cohort["preflight"]}
    if len(cases) != len(preflight) or {row["sample_id"] for row in cases} != set(preflight):
        raise ValueError("A shard expects each sealed geometry once")
    first, last = (args.verify, args.verify) if args.verify is not None else args.range
    if not 1 <= first <= last <= len(cases):
        parser.error(f"positions must satisfy 1 <= FIRST <= LAST <= {len(cases)}")

    checkpoint_name = f"arms/{arm}_seed{seed}/{tag}.pt"
    checkpoint = output/checkpoint_name
    checkpoint_sha = file_sha(checkpoint)
    if checkpoint_sha != read_json(training_path)["checkpoints"][checkpoint_name]:
        raise ValueError("The shard expected the sealed checkpoint")
    binding = {"version": 1, "lock_sha256": file_sha(lock_path), "arm": arm, "seed": seed, "tag": tag}
    job = case_key(arm, seed, tag)
    ledger = output/"evaluation"/"case_receipts"/job
    evidence = output/"evaluation"/"shard_evidence"/job
    evidence.mkdir(parents=True, exist_ok=True)
    gate_path = evidence/"replay_gate.json"
    script = Path(__file__)
    gate_binding = {**binding, "checkpoint_sha256": checkpoint_sha, "shard_sha256": file_sha(script),
                    "geometry_cache_sha256": file_sha(script.with_name("fusion_geometry_cache.py")),
                    "ledger_sha256": file_sha(script.with_name("fusion_evaluation_progress.py"))}

    def case_identity(position):
        return {**binding, "position": position, "sample_id": cases[position-1]["sample_id"]}

    def read_receipt(path, identity):
        saved = read_json(path)
        row = saved["row"]
        if (saved["identity"] != identity or row["sample_id"] != identity["sample_id"]
                or row["seed"] != seed or _digest(row) != saved["row_sha256"]):
            raise ValueError(f"Unexpected evaluation receipt at {path.name}")
        return saved

    if args.verify is None:
        gate = read_json(gate_path)
        reference = read_receipt(ledger/f"{gate['position']:05d}.json", case_identity(gate["position"]))
        replayed = gate["replayed_row"]
        if (gate["binding"] != gate_binding or gate["reference_row_sha256"] != reference["row_sha256"]
                or gate["replayed_row_sha256"] != _digest(replayed) or not _same_measurement(reference["row"], replayed)):
            raise ValueError("The replay gate must match the sealed evaluator and reference receipt")

    torch.set_num_threads(cfg["runtime"]["torch_threads"])
    install_geometry_cache()
    agent, _ = get_inference_agent(checkpoint, arm, seed, file_sha(output/"prepared.json"))
    pool = Environments(cfg["geometry"], arm, capacity=1)

    def evaluate_case(position):
        sample = cases[position-1]
        row, transitions = episode(pool, sample, "learned", agent=agent, seed=seed,
                                   prechecked=preflight[sample["sample_id"]], trace=True)
        if transitions or (row["sample_id"], row["seed"]) != (sample["sample_id"], seed):
            raise ValueError(f"Frozen evaluator returned an unexpected row at position {position}")
        return row

    if args.verify is not None:
        path = ledger/f"{first:05d}.json"
        reference = read_receipt(path, case_identity(first))
        replayed = evaluate_case(first)
        if not _same_measurement(reference["row"], replayed):
            raise ValueError("Shard replay differs from the existing sealed evaluation")
        _save_receipt(gate_path, {"binding": gate_binding, "position": first,
                                  "reference_row_sha256": reference["row_sha256"],
                                  "replayed_row_sha256": _digest(replayed), "replayed_row": replayed})
        print(f"{job} replay gate passed for case {first}", flush=True)
        return

    staged = evidence/f"positions_{first:05d}_{last:05d}"
    staged.mkdir(exist_ok=True)
    ledger.mkdir(parents=True, exist_ok=True)
    for position in range(last, first-1, -1):
        identity = case_identity(position)
        name = f"{position:05d}.json"
        destination = ledger/name
        if destination.exists():
            read_receipt(destination, identity)
            continue
        candidate = staged/name
        if candidate.exists():
            row = read_receipt(candidate, identity)["row"]
        else:
            row = evaluate_case(position)
            _save_receipt(candidate, {"identity": identity, "row_sha256": _digest(row), "row": row})
        try:
            if os.name == "nt":
                candidate.rename(destination)
            else:
                os.link(candidate, destination)
        except FileExistsError:
            if not _same_measurement(row, read_receipt(destination, identity)["row"]):
                raise ValueError(f"Concurrent result differs at position {position}")
        else:
            print(f"{job} published case {position}", flush=True)


if __name__ == "__main__":
    main()
