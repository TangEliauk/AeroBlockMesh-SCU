"""Verify the published fusion archive and optionally replay one sealed test case."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from smartcfd.airfoil_decomposition.experiment import file_sha
from smartcfd.airfoil_fusion.evaluate import jobs
from smartcfd.airfoil_fusion.protocol import ROOT, read_json, reference_identities
from smartcfd.airfoil_fusion.rollout import Environments, episode
from smartcfd.airfoil_fusion.train import get_inference_agent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_digest(archive, name, expected):
    path = (archive / name).resolve()
    require(path.is_relative_to(archive), f"Archive path escapes its root: {name}")
    require(file_sha(path) == expected, f"SHA-256 mismatch: {name}")


def verify_archive(archive):
    prepared = read_json(archive / "prepared.json")
    matrix = read_json(archive / "training_matrix.json")
    lock = read_json(archive / "evaluation_lock.json")
    audit = read_json(archive / "audit.json")
    cfg = read_json(archive / "config.json")
    require(all(record["profile"] == "formal" for record in (prepared, matrix, lock, audit, cfg)),
            "Expected a formal archive")
    for name, expected in prepared["files"].items():
        verify_digest(archive, name, expected)
    require(read_json(archive / "source.json")["references"] == reference_identities(),
            "Pinned A or B reference identity differs")
    verify_digest(archive, "training_matrix.json", lock["training_sha256"])
    verify_digest(archive, "evaluation_cases.json", lock["cases_sha256"])

    expected_checkpoints = {f"arms/{arm}_seed{seed}/{tag}.pt" for arm, seed, tag in jobs(cfg) if tag != "rule"}
    require(len(expected_checkpoints) == 60 and matrix["checkpoints"].keys() == expected_checkpoints,
            "Unexpected formal checkpoints")
    for name, expected in matrix["checkpoints"].items():
        verify_digest(archive, name, expected)
    expected_receipts = {f"{arm}_{seed}" for arm, seed, tag in jobs(cfg) if tag == "best"}
    require(matrix["receipt_hashes"].keys() == expected_receipts, "Unexpected completion receipts")
    for name, expected in matrix["receipt_hashes"].items():
        arm, seed = name.split("_")
        verify_digest(archive, f"arms/{arm}_seed{seed}/completion.json", expected)

    test = read_json(archive / "evaluation_cases.json")
    require(len(test["cases"]) == 90 and len(jobs(cfg)) == 67, "Unexpected test scope")
    require(audit["passed"] and not audit["problems"] and audit["cases_checked"] == 6030
            and audit["evaluation_case_receipts_checked"] == 6030, "Archived formal audit did not pass")
    require(len(audit["artifacts"]) >= 20, "Missing audited report artifact digests")
    for name, expected in audit["artifacts"].items():
        relative = f"report/{name}" if (archive / "report" / name).is_file() else name
        verify_digest(archive, relative, expected)

    cfd = read_json(archive / "cfd/summary.json")
    require(len(cfd["cases"]) == 6 and audit["solver_runs"] == 12, "Unexpected CFD scope")
    for case in cfd["cases"]:
        directory = case["directory"]
        verify_digest(archive, f"{directory}/mesh.su2", case["mesh_sha256"])
        require(case["phases"].keys() == {"warm", "rans"}, "Missing solver phase")
        for stage, phase in case["phases"].items():
            verify_digest(archive, f"{directory}/{stage}.cfg", phase["config_sha256"])
            verify_digest(archive, f"{directory}/{stage}_history.csv", phase["history_sha256"])
    return cfg, test, matrix


def replay_case(archive, cfg, test, matrix):
    witness = read_json(archive / "evaluation/replay_witness.json")
    arm, seed, tag = (witness[field] for field in ("arm", "seed", "tag"))
    verify_digest(archive, "evaluation_lock.json", witness["lock_sha256"])
    weight = f"arms/{arm}_seed{seed}/{tag}.pt"
    require(matrix["checkpoints"][weight] == witness["checkpoint_sha256"], "Witness weight changed")
    expected = witness["case"]
    sample = next(row for row in test["cases"] if row["sample_id"] == expected["sample_id"])
    preflight = next(row for row in test["preflight"] if row["sample_id"] == expected["sample_id"])
    torch.set_num_threads(cfg["runtime"]["torch_threads"])
    agent, step = get_inference_agent(archive / weight, arm, seed, file_sha(archive / "prepared.json"))
    require(step == witness["step"], "Unexpected checkpoint step")
    row, _ = episode(Environments(cfg["geometry"], arm), sample, "learned",
                     agent=agent, seed=seed, prechecked=preflight, trace=True)
    for field in ("success", "reason", "decisions", "edits", "topology"):
        require(json.loads(json.dumps(row[field])) == expected[field], f"Replayed {field} differs")
    require([action["candidate"] for action in row["actions"]] == expected["actions"],
            "Replayed actions differ")
    for budget, metrics in expected["mesh"].items():
        mesh = row["mesh"][budget]
        for field, value in metrics.items():
            same = (math.isclose(mesh[field], value, rel_tol=1e-6, abs_tol=1e-9)
                    if isinstance(value, float) else mesh[field] == value)
            require(same, f"Replayed mesh {budget} {field} differs")
    print(f"Replayed {arm}/{seed}/{tag} on {sample['sample_id']}: "
          f"{row['decisions']} decisions, 1536- and 5120-cell meshes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path,
                        default=ROOT / "experiments/results/airfoil_fusion_20261128")
    parser.add_argument("--replay", action="store_true", help="Recompute the archived example on CPU")
    args = parser.parse_args()
    archive = args.archive.resolve()
    cfg, test, matrix = verify_archive(archive)
    print("Archive verified: 60 checkpoints, 30 completion receipts, 67 x 90 evaluations, 12 CFD histories")
    if args.replay:
        replay_case(archive, cfg, test, matrix)


if __name__ == "__main__":
    main()
