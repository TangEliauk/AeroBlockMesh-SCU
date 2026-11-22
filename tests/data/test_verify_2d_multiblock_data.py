"""Release verification rejects broken provenance and misleading reference labels."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "verify_2d_multiblock_data", ROOT / "scripts/data/verify_2d_multiblock_data.py")
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


class ReferenceReleaseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest_path = self.root / verify.RAW_PREFIX / "manifest.json"
        raw = []
        for asset_id in sorted(verify.ASSET_IDS):
            path = verify.RAW_PREFIX + asset_id + ".bin"
            payload = asset_id.encode()
            self.write(path, payload)
            raw.append(dict(asset_id=asset_id, path=path, bytes=len(payload),
                            sha256=hashlib.sha256(payload).hexdigest(),
                            source_url="https://example.invalid/source",
                            source_page="https://example.invalid/docs",
                            license_evidence={"scope": "upstream source"}))
        cases, cleaned = [], []
        for case_id in sorted(verify.CASE_IDS):
            source = next(item for item in raw if item["asset_id"] == case_id + "_grid")
            path = verify.CLEAN_PREFIX + case_id + ".json"
            document = {"case_id": case_id, "blocks": [{}, {}, {}],
                        "source": {"path": source["path"], "sha256": source["sha256"]}}
            payload = json.dumps(document).encode()
            self.write(path, payload)
            cleaned.append(dict(case_id=case_id, path=path, bytes=len(payload),
                                sha256=hashlib.sha256(payload).hexdigest()))
            cases.append(dict(case_id=case_id, source_grid=source["path"], cleaned_path=path,
                              source_declared_blocks=3, has_original_expert_action_history=False,
                              data_role="auxiliary_terminal_block_reference", split_assignment=None,
                              reference_information=["boundaries"], uses=["interface checks"],
                              positioning="auxiliary", limitations=["no original actions"]))
        self.manifest = dict(schema_version=1, version="multiblock_2d_v1", case_count=3,
                             raw_file_count=len(raw), raw_total_bytes=sum(r["bytes"] for r in raw),
                             raw_files=raw, cases=cases, cleaned_files=cleaned)
        catalog = []
        for case_id in sorted(verify.CATALOG_IDS):
            is_new = case_id in verify.CASE_IDS
            path = (verify.CLEAN_PREFIX if is_new else verify.OLD_CLEAN_PREFIX) + case_id + ".json"
            if not is_new:
                self.write(path, json.dumps({"case_id": case_id}).encode())
            case = next((item for item in cases if item["case_id"] == case_id), None)
            catalog.append(dict(case_id=case_id, cleaned_path=path,
                                source_manifest=verify.RAW_PREFIX + "manifest.json" if is_new else "data/references/multiblock_v1/manifest.json",
                                included_in_new_2d_release=case_id in verify.CASE_IDS,
                                has_original_expert_action_history=False, split_assignment=None,
                                reference_information=case["reference_information"] if case else ["blocks"],
                                uses=case["uses"] if case else ["reference"],
                                positioning="auxiliary", limitations=["no original actions"]))
        self.catalog = dict(case_count=7, cases=catalog)
        self.write("data/references/multiblock_v1/manifest.json", b"{}")
        self.save()

    def write(self, relative, payload):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    def save(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.write("data/references/multiblock_case_catalog.json", json.dumps(self.catalog).encode())

    def test_complete_release_is_accepted(self):
        self.assertTrue(verify.verify(self.root)["all_release_hashes_match"])

    def test_corrupt_source_is_rejected_without_overwriting(self):
        item = self.manifest["raw_files"][0]
        path = self.root / item["path"]
        corrupt = b"x" * item["bytes"]
        path.write_bytes(corrupt)
        with self.assertRaisesRegex(ValueError, "Hash mismatch"):
            verify.verify(self.root)
        self.assertEqual(path.read_bytes(), corrupt)

    def test_missing_case_cannot_be_hidden_by_editing_count(self):
        self.manifest["cases"].pop()
        self.manifest["case_count"] = 2
        self.save()
        with self.assertRaisesRegex(ValueError, "Missing or duplicate reference cases"):
            verify.verify(self.root)

    def test_escaping_and_absolute_paths_are_rejected(self):
        for value in ["../outside", "/outside", "C:/outside", verify.RAW_PREFIX + "../outside"]:
            with self.subTest(path=value), self.assertRaises(ValueError):
                verify.resolve_file(self.root, value, verify.RAW_PREFIX)

    def test_cleaned_source_cannot_point_to_unrelated_asset(self):
        item = self.manifest["cleaned_files"][0]
        path = self.root / item["path"]
        document = json.loads(path.read_text())
        document["source"]["sha256"] = "0" * 64
        payload = json.dumps(document).encode()
        path.write_bytes(payload)
        item.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        self.save()
        with self.assertRaisesRegex(ValueError, "Cleaned source hash mismatch"):
            verify.verify(self.root)

    def test_reference_cannot_be_relabeled_as_original_expert_actions(self):
        self.manifest["cases"][0]["has_original_expert_action_history"] = True
        self.save()
        with self.assertRaisesRegex(ValueError, "Unsupported claim of original expert actions"):
            verify.verify(self.root)

    def test_catalog_requires_clear_positioning(self):
        self.catalog["cases"][0]["positioning"] = ""
        self.save()
        with self.assertRaisesRegex(ValueError, "Missing catalog field: positioning"):
            verify.verify(self.root)

    def test_catalog_cannot_point_a_case_at_another_existing_output(self):
        entry = next(item for item in self.catalog["cases"] if item["case_id"] == "cgns_airfoil4")
        entry["cleaned_path"] = verify.OLD_CLEAN_PREFIX + "cgns_channel12.json"
        self.save()
        with self.assertRaisesRegex(ValueError, "Catalog case points to unrelated output"):
            verify.verify(self.root)

    def test_catalog_source_manifest_and_roles_cannot_drift(self):
        entry = next(item for item in self.catalog["cases"] if item["case_id"] == "backstep4")
        original = entry["source_manifest"]
        entry["source_manifest"] = "data/references/unrelated.json"
        self.save()
        with self.assertRaisesRegex(ValueError, "Catalog source manifest mismatch"):
            verify.verify(self.root)
        entry["source_manifest"] = original
        entry["positioning"] = "core expert training"
        self.save()
        with self.assertRaisesRegex(ValueError, "Catalog role differs from release manifest"):
            verify.verify(self.root)


if __name__ == "__main__":
    unittest.main()
