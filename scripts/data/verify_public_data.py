"""Verify published data bytes, geometry representation, and frozen split membership."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify(root: Path) -> dict:
    manifest = json.loads((root / "data/public_v1_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = (root / item["path"]).resolve()
        require(path.is_relative_to(root.resolve()), "Manifest path outside repository")
        require(path.is_file(), f"Missing published file: {item['path']}")
        require(path.stat().st_size == item["bytes"], f"Size mismatch: {item['path']}")
        require(digest(path) == item["sha256"], f"Hash mismatch: {item['path']}")
    directory = root / "data/processed/uiuc_v1"
    samples = [json.loads(line) for line in (directory / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    ids = set()
    for row in samples:
        sid, points = row["sample_id"], row["polygon"]
        require(sid not in ids, f"Repeated sample ID: {sid}")
        ids.add(sid)
        require(len(points) >= 3 and points[0] != points[-1], f"Invalid ring representation: {sid}")
        require(all(len(p) == 2 and all(math.isfinite(x) for x in p) for p in points), f"Invalid coordinates: {sid}")
        require(row["closure"] == "implicit_last_to_first" and row["orientation"] == "CCW", f"Unexpected geometry convention: {sid}")
        area2 = sum(points[i][0] * points[(i + 1) % len(points)][1] - points[(i + 1) % len(points)][0] * points[i][1] for i in range(len(points)))
        require(area2 > 0, f"Nonpositive area: {sid}")
    splitdir = root / "data/splits/uiuc_v1"
    splits = {name: json.loads((splitdir / (name + ".json")).read_text(encoding="utf-8")) for name in ("train", "val", "test")}
    sets = {name: set(values) for name, values in splits.items()}
    require(all(len(sets[name]) == len(values) for name, values in splits.items()), "Duplicate IDs in split")
    require(not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]), "Overlapping splits")
    require(set.union(*sets.values()) == ids, "Split IDs do not cover samples exactly")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    split_report = json.loads((splitdir / "split_report.json").read_text(encoding="utf-8"))
    require(summary["unique_samples"] == len(samples), "Stale cleaning summary")
    require(split_report["input"]["sha256"] == digest(directory / "samples.jsonl"), "Split report targets different geometry bytes")
    require(split_report["actual_counts"] == {k: len(v) for k, v in splits.items()}, "Stale split counts")
    references = json.loads((root / "data/references/public_v1/cleaning_report.json").read_text(encoding="utf-8"))
    require(references["total_rows"] == sum(item["rows"] for item in references["datasets"]), "Stale reference count")
    return {"published_files_checked": len(manifest["files"]), "geometry_samples": len(samples),
            "splits": {k: len(v) for k, v in splits.items()}, "experimental_rows": references["total_rows"],
            "all_hashes_match": True, "split_input_hash_matches": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(verify(args.repo_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
