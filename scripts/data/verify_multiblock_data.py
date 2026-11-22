"""Verify bundled multiblock originals and cleaned outputs against release hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


ASSET_IDS = {
    "cgns_channel_12", "cgns_multielement_airfoil_4", "dpw3_f6_icem_rev2",
    "onera_m6_four_block_grid", "onera_m6_gman_boundaries",
}
CASE_IDS = {"cgns_channel12", "cgns_airfoil4", "dlr_f6_icem", "onera_m6"}
RAW_PREFIX = "data/references/multiblock_v1/"
CLEAN_PREFIX = "data/processed/block_topology_v1/"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def resolve_file(root: Path, name: str, prefix: str) -> Path:
    require(isinstance(name, str) and bool(name), "Missing file path")
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and ".." not in relative.parts
            and "\\" not in name and ":" not in name, "Nonportable or escaping file path")
    require(name.startswith(prefix), f"File outside expected dataset directory: {name}")
    path = (root / name).resolve()
    require(path.is_relative_to(root), "File resolves outside repository")
    require(path.is_file(), f"Missing file: {name}")
    return path


def check_file(root: Path, item: dict, prefix: str) -> None:
    path = resolve_file(root, item["path"], prefix)
    require(path.stat().st_size == item["bytes"], f"Size mismatch: {item['path']}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    require(digest == item["sha256"], f"Hash mismatch: {item['path']}")


def verify(root: Path) -> dict:
    root = root.resolve()
    manifest = json.loads((root / RAW_PREFIX / "manifest.json").read_text(encoding="utf-8"))
    require(manifest["schema_version"] == 1 and manifest["version"] == "multiblock_v1",
            "Unsupported multiblock manifest")
    require(manifest["source_manifest"] == "data/references/public_v1/reference_manifest.json",
            "Unexpected frozen source manifest")
    source = json.loads((root / manifest["source_manifest"]).read_text(encoding="utf-8"))
    originals = {item["asset_id"]: item for item in source["assets"]}
    raw, cleaned = manifest["raw_files"], manifest["cleaned_files"]
    require(len(raw) == 5 and {item["asset_id"] for item in raw} == ASSET_IDS,
            "Missing or duplicate source assets")
    require(len(cleaned) == 4 and {item["case_id"] for item in cleaned} == CASE_IDS,
            "Missing or duplicate cleaned cases")
    require(len({item["path"] for item in raw + cleaned}) == 9, "Duplicate release file path")
    require(manifest["raw_file_count"] == len(raw) and manifest["case_count"] == len(cleaned),
            "Stale release counts")
    require(manifest["raw_total_bytes"] == sum(item["bytes"] for item in raw),
            "Stale source byte count")
    for item in raw:
        original = originals[item["asset_id"]]
        require((item["bytes"], item["sha256"]) == (original["bytes"], original["sha256"]),
                f"Original differs from frozen public_v1 source: {item['asset_id']}")
        check_file(root, item, RAW_PREFIX)
    for item in cleaned:
        require(item["path"] == CLEAN_PREFIX + item["case_id"] + ".json",
                "Unexpected cleaned case path")
        check_file(root, item, CLEAN_PREFIX)
        document = json.loads((root / item["path"]).read_text(encoding="utf-8"))
        require(document["case_id"] == item["case_id"], "Cleaned case identity mismatch")
    return {
        "cases": len(cleaned), "original_files": len(raw),
        "original_bytes": manifest["raw_total_bytes"], "cleaned_files": len(cleaned),
        "source_hashes_match_frozen_manifest": True, "all_release_hashes_match": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(verify(args.repo_root), indent=2))


if __name__ == "__main__":
    main()
