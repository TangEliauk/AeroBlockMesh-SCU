"""Verify the independent 2D reference release, source links and seven-case roles."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

RAW_PREFIX = "data/references/multiblock_2d_v1/"
CLEAN_PREFIX = "data/processed/block_topology_2d_v1/"
CASE_IDS = {"backstep4", "ejector3", "hump4"}
CATALOG_IDS = CASE_IDS | {"cgns_channel12", "cgns_airfoil4", "dlr_f6_icem", "onera_m6"}
OLD_CLEAN_PREFIX = "data/processed/block_topology_v1/"
ASSET_IDS = {
    "backstep4_grid", "backstep4_nmf", "ejector3_grid", "ejector3_input",
    "ejector_splitter_input", "hump4_grid", "hump4_input_steady",
    "hump4_input_oscillatory", "hump4_input_average", "hump4_upstream_readme",
    "hump_splitter_input",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def resolve_file(root: Path, name: str, prefix: str) -> Path:
    require(isinstance(name, str) and bool(name), "Missing portable path")
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and ".." not in relative.parts
            and "\\" not in name and ":" not in name, "Unsafe or nonportable path")
    require(name.startswith(prefix), "Unexpected release directory")
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), "Path escapes repository")
    require(path.is_file(), f"Missing release file: {name}")
    return path


def check_file(root: Path, item: dict, prefix: str) -> None:
    path = resolve_file(root, item["path"], prefix)
    require(path.stat().st_size == item["bytes"], f"Size mismatch: {item['path']}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    require(actual == item["sha256"], f"Hash mismatch: {item['path']}")


def verify(root: Path) -> dict:
    root = root.resolve()
    manifest = json.loads((root / RAW_PREFIX / "manifest.json").read_text(encoding="utf-8"))
    require(manifest["schema_version"] == 1 and manifest["version"] == "multiblock_2d_v1",
            "Unsupported 2D reference manifest")
    raw, cases, cleaned = manifest["raw_files"], manifest["cases"], manifest["cleaned_files"]
    require(len(raw) == len(ASSET_IDS) and {a["asset_id"] for a in raw} == ASSET_IDS,
            "Missing or duplicate source assets")
    require(len(cases) == 3 and {c["case_id"] for c in cases} == CASE_IDS,
            "Missing or duplicate reference cases")
    require(len(cleaned) == 3 and {c["case_id"] for c in cleaned} == CASE_IDS,
            "Missing or duplicate cleaned cases")
    require(len({item["path"] for item in raw + cleaned}) == len(raw) + len(cleaned),
            "Duplicate release path")
    require(manifest["case_count"] == 3 and manifest["raw_file_count"] == len(raw),
            "Stale release counts")
    require(manifest["raw_total_bytes"] == sum(item["bytes"] for item in raw),
            "Stale source byte count")
    for item in raw:
        check_file(root, item, RAW_PREFIX)
        require(item["source_url"].startswith("https://"), "Missing HTTPS source URL")
        require(item["source_page"].startswith("https://"), "Missing source documentation")
        require(bool(item["license_evidence"]["scope"]), "Missing source license scope")
    raw_paths = {item["path"]: item for item in raw}
    output_paths = {item["case_id"]: item["path"] for item in cleaned}
    for case in cases:
        require(case["source_grid"] in raw_paths, "Untracked source grid")
        require(case["cleaned_path"] == output_paths[case["case_id"]],
                "Case output path mismatch")
        require(case["has_original_expert_action_history"] is False,
                "Unsupported claim of original expert actions")
        require(case["data_role"] == "auxiliary_terminal_block_reference",
                "Supplement cannot be relabeled as expert training data")
        require(case["split_assignment"] is None, "Reference case assigned to training split")
        for field in ("reference_information", "uses", "positioning", "limitations"):
            require(bool(case[field]), f"Missing reference role field: {field}")
    for item in cleaned:
        require(item["path"] == CLEAN_PREFIX + item["case_id"] + ".json",
                "Unexpected cleaned case path")
        check_file(root, item, CLEAN_PREFIX)
        document = json.loads((root / item["path"]).read_text(encoding="utf-8"))
        require(document["case_id"] == item["case_id"], "Cleaned identity mismatch")
        case = next(c for c in cases if c["case_id"] == item["case_id"])
        require(len(document["blocks"]) == case["source_declared_blocks"],
                "Parsed block count disagrees with source description")
        source = document["source"]
        require(source["path"] == case["source_grid"], "Cleaned source path mismatch")
        require(source["sha256"] == raw_paths[source["path"]]["sha256"],
                "Cleaned source hash mismatch")
    catalog = json.loads((root / "data/references/multiblock_case_catalog.json").read_text(encoding="utf-8"))
    require(catalog["case_count"] == 7 and len(catalog["cases"]) == 7
            and {c["case_id"] for c in catalog["cases"]} == CATALOG_IDS,
            "Incomplete seven-case reference catalog")
    for entry in catalog["cases"]:
        is_new = entry["case_id"] in CASE_IDS
        expected_prefix = CLEAN_PREFIX if is_new else OLD_CLEAN_PREFIX
        expected_path = expected_prefix + entry["case_id"] + ".json"
        expected_manifest = RAW_PREFIX + "manifest.json" if is_new else "data/references/multiblock_v1/manifest.json"
        require(entry["cleaned_path"] == expected_path, "Catalog case points to unrelated output")
        require(entry["source_manifest"] == expected_manifest, "Catalog source manifest mismatch")
        output = resolve_file(root, entry["cleaned_path"], expected_prefix)
        require(json.loads(output.read_text(encoding="utf-8"))["case_id"] == entry["case_id"],
                "Catalog output identity mismatch")
        resolve_file(root, entry["source_manifest"], "data/references/")
        require(entry["has_original_expert_action_history"] is False,
                "Catalog falsely claims expert action history")
        require(entry["split_assignment"] is None, "Catalog assigns reference to training")
        require(entry["included_in_new_2d_release"] == (entry["case_id"] in CASE_IDS),
                "Catalog new-release membership mismatch")
        for field in ("reference_information", "uses", "positioning", "limitations"):
            require(bool(entry[field]), f"Missing catalog field: {field}")
        if is_new:
            case = next(c for c in cases if c["case_id"] == entry["case_id"])
            for field in ("reference_information", "uses", "positioning"):
                require(entry[field] == case[field], f"Catalog role differs from release manifest: {field}")
            require(set(case["limitations"]).issubset(entry["limitations"]),
                    "Catalog omits release limitations")
    return {"cases": 3, "raw_files": len(raw), "raw_bytes": manifest["raw_total_bytes"],
            "cleaned_files": len(cleaned), "catalog_cases": 7,
            "all_release_hashes_match": True, "expert_actions_generated": False,
            "training_split_assignment": None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(verify(args.repo_root), indent=2))


if __name__ == "__main__":
    main()
