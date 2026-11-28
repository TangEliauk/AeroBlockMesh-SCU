"""Retain finished development cases for an exact checkpoint retry."""

from smartcfd.airfoil_fusion.protocol import read_json

from fusion_evaluation_progress import _digest, _save_receipt


class ValidationCaseReceipts:
    def __init__(self, directory, binding, hashes, step):
        self.directory = directory / "case_receipts"
        self.identity = {"version": 1, "binding": binding, "step": step, "network_sha256": hashes}
        self.reused = 0

    def rows(self, cases, evaluate):
        binding = self.identity["binding"]
        seed = binding["seed"]
        rows = []
        for position, sample in enumerate(cases, 1):
            identity = {**self.identity, "position": position, "sample_id": sample["sample_id"]}
            path = self.directory / f"{position:05d}.json"
            if path.is_file():
                record = read_json(path)
                row = record["row"]
                if (record["identity"] != identity or _digest(row) != record["row_sha256"]
                        or (row["sample_id"], row["seed"]) != (sample["sample_id"], seed)):
                    raise ValueError(f"Development validation receipt differs at case {position} for {self.directory.parent}")
                self.reused += 1
            else:
                row = evaluate(sample)
                if (row["sample_id"], row["seed"]) != (sample["sample_id"], seed):
                    raise ValueError("Development validation returned an unexpected case")
                _save_receipt(path, {"identity": identity, "row": row, "row_sha256": _digest(row)})
            rows.append(row)
            if position % 8 == 0 or position == len(cases):
                print(f"{binding['arm']} seed={seed} validation step={self.identity['step']} "
                      f"cases={position}/{len(cases)} resumed={self.reused}", flush=True)
        return rows
