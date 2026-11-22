"""Acquisition tests: preserve existing inputs and reject corrupt/escaping writes."""
import hashlib
import importlib.util
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = [ROOT / "scripts/data/download_public_sources.py", ROOT / "scripts/download_public_sources.py"]
if not any(p.exists() for p in CANDIDATES):
    ROOT = Path(__file__).resolve().parents[2]
    CANDIDATES = [ROOT / "scripts/data/download_public_sources.py"]
spec = importlib.util.spec_from_file_location("download_public_sources", next(p for p in CANDIDATES if p.exists()))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AcquisitionTests(unittest.TestCase):
    def entry(self, payload=b"source"):
        return {"id": "fixture", "url": "https://example.invalid/fixture", "local_path": "fixture.dat",
                "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}

    def test_valid_existing_source_does_not_use_network(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(module.urllib.request, "urlopen") as request:
            path = Path(tmp) / "fixture.dat"
            path.write_bytes(b"source")
            self.assertEqual(module.download(self.entry(), Path(tmp)), path)
            request.assert_not_called()

    def test_existing_mismatch_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.dat"
            path.write_bytes(b"user-local-input")
            with self.assertRaises(ValueError):
                module.download(self.entry(), Path(tmp))
            self.assertEqual(path.read_bytes(), b"user-local-input")

    def test_changed_upstream_is_not_promoted_to_final_file(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(module.urllib.request, "urlopen", return_value=io.BytesIO(b"changed")):
            with self.assertRaises(ValueError):
                module.download(self.entry(), Path(tmp))
            self.assertFalse((Path(tmp) / "fixture.dat").exists())

    def test_registry_path_cannot_escape_output(self):
        entry = self.entry()
        entry["local_path"] = "../outside.dat"
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            module.download(entry, Path(tmp))

    def test_zip_path_cannot_escape_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "fixture.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("../outside.dat", b"fixture")
            with self.assertRaises(ValueError):
                module.extract_checked(archive, Path(tmp) / "extracted")
            self.assertFalse((Path(tmp) / "outside.dat").exists())


if __name__ == "__main__":
    unittest.main()
