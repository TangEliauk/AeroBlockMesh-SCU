"""Release checks must reject corrupted bytes and manifest path manipulation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "verify_multiblock_data", ROOT / "scripts/data/verify_multiblock_data.py")
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


class ReleaseIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest_path = self.root / verify.RAW_PREFIX / "manifest.json"
        sources, raw, cleaned = [], [], []
        for number, asset_id in enumerate(sorted(verify.ASSET_IDS)):
            path = verify.RAW_PREFIX + str(number) + ".bin"
            payload = (asset_id + "\n").encode()
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            entry = {"asset_id": asset_id, "path": path, "bytes": len(payload),
                     "sha256": hashlib.sha256(payload).hexdigest()}
            raw.append(entry)
            sources.append(entry.copy())
        for case_id in sorted(verify.CASE_IDS):
            path = verify.CLEAN_PREFIX + case_id + ".json"
            payload = json.dumps({"case_id": case_id}).encode()
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            cleaned.append({"case_id": case_id, "path": path, "bytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest()})
        source = self.root / "data/references/public_v1/reference_manifest.json"
        source.parent.mkdir(parents=True)
        source.write_text(json.dumps({"assets": sources}), encoding="utf-8")
        self.manifest = {
            "schema_version": 1, "version": "multiblock_v1", "case_count": 4,
            "source_manifest": "data/references/public_v1/reference_manifest.json",
            "raw_files": raw, "raw_file_count": 5,
            "raw_total_bytes": sum(item["bytes"] for item in raw), "cleaned_files": cleaned,
        }
        self.write_manifest()

    def write_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_complete_release_is_accepted(self):
        result = verify.verify(self.root)
        self.assertTrue(result["all_release_hashes_match"])
        self.assertEqual(result["cases"], 4)

    def test_corrupt_source_is_rejected_without_modification(self):
        path = self.root / self.manifest["raw_files"][0]["path"]
        payload = path.read_bytes()
        corrupt = bytes([payload[0] ^ 1]) + payload[1:]
        path.write_bytes(corrupt)
        with self.assertRaisesRegex(ValueError, "Hash mismatch"):
            verify.verify(self.root)
        self.assertEqual(path.read_bytes(), corrupt)

    def test_rewriting_hash_in_release_manifest_cannot_change_frozen_source(self):
        item = self.manifest["raw_files"][0]
        payload = b"replacement"
        (self.root / item["path"]).write_bytes(payload)
        item.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        self.manifest["raw_total_bytes"] = sum(row["bytes"] for row in self.manifest["raw_files"])
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "differs from frozen"):
            verify.verify(self.root)

    def test_traversal_and_absolute_paths_are_rejected(self):
        for path in ["../outside", "/outside", "C:/outside", verify.RAW_PREFIX + "../outside"]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                verify.resolve_file(self.root, path, verify.RAW_PREFIX)

    def test_missing_case_cannot_be_hidden_by_editing_count(self):
        self.manifest["cleaned_files"].pop()
        self.manifest["case_count"] = 3
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Missing or duplicate cleaned cases"):
            verify.verify(self.root)


if __name__ == "__main__":
    unittest.main()
