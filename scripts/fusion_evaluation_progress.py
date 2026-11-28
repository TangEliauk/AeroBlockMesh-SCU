"""Add durable case receipts to the frozen evaluation worker."""

from functools import wraps
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from smartcfd.airfoil_decomposition.experiment import _json_default, file_sha
from smartcfd.airfoil_fusion import evaluate as evaluation
from smartcfd.airfoil_fusion.protocol import read_json


def _digest(row):
    encoded = json.dumps(row, separators=(",", ":"), ensure_ascii=False,
                         allow_nan=False, default=_json_default).encode("utf-8")
    return sha256(encoded).hexdigest()


def _save_receipt(path, receipt):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(receipt, stream, ensure_ascii=False, allow_nan=False, default=_json_default)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def install_evaluation_progress():
    evaluate = evaluation.evaluate

    @wraps(evaluate)
    def resume(output, cfg, arm, seed, tag):
        key = evaluation.case_key(arm, seed, tag)
        ledger = output / "evaluation" / "case_receipts" / key
        episode = evaluation.episode
        position = 0
        binding = None
        reused = 0

        def record(pool, sample, method, **options):
            nonlocal binding, position, reused
            if binding is None:
                # The original evaluator reaches this hook after checking all seals
                # and loading the exact frozen weights for this worker.
                binding = {"version": 1, "lock_sha256": file_sha(output / "evaluation_lock.json"),
                           "arm": arm, "seed": seed, "tag": tag}
            position += 1
            sample_id = sample["sample_id"]
            identity = {**binding, "position": position, "sample_id": sample_id}
            path = ledger / f"{position:05d}.json"
            if path.is_file():
                saved = read_json(path)
                row = saved["row"]
                if (saved["identity"] != identity or row["sample_id"] != sample_id or row["seed"] != seed
                        or _digest(row) != saved["row_sha256"]):
                    raise ValueError(f"Evaluation receipt identity or digest changed: {key} case {position}")
                reused += 1
                return row, []
            row, transitions = episode(pool, sample, method, **options)
            if transitions or (row["sample_id"], row["seed"]) != (sample_id, seed):
                raise ValueError("Frozen evaluation returned an unexpected case receipt")
            _save_receipt(path, {"identity": identity, "row_sha256": _digest(row), "row": row})
            return row, transitions

        evaluation.episode = record
        try:
            result = evaluate(output, cfg, arm, seed, tag)
        finally:
            evaluation.episode = episode
        if reused:
            print(f"eval {key} resumed {reused}/{position} sealed case receipts", flush=True)
        return result

    evaluation.evaluate = resume
    return evaluate
