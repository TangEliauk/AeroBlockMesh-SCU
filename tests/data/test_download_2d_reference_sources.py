"""Offline acquisition regressions: preserve inputs and reject changed archives."""
from __future__ import annotations

import contextlib
import copy
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import zlib


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "download_2d_reference_sources", ROOT / "scripts/data/download_2d_reference_sources.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
PREFIX = "data/references/multiblock_2d_v1/fixture/"


class Response(io.BytesIO):
    def __init__(self, data, *, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}
        self.read_calls = 0

    def read(self, size=-1):
        self.read_calls += 1
        return super().read(size)


def asset(payload=b"source", *, name="fixture.dat", asset_id="fixture"):
    return {"asset_id": asset_id, "path": PREFIX + name, "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "source_url": "https://example.invalid/source", "retrieval": {"kind": "direct"}}


def manifest(*entries):
    return {"schema_version": 1, "version": "multiblock_2d_v1", "raw_files": list(entries)}


def tar_payload(members):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, payload, kind in members:
            item = tarfile.TarInfo(name)
            if kind == "file":
                item.size = len(payload)
                archive.addfile(item, io.BytesIO(payload))
            else:
                item.type = tarfile.SYMTYPE
                item.linkname = "../../outside"
                archive.addfile(item)
    return stream.getvalue()


def tar_asset(payload, archive, member="case/source.dat", **kwargs):
    entry = asset(payload, **kwargs)
    entry["retrieval"] = {"kind": "tar_member", "archive_member": member,
                          "archive_bytes": len(archive),
                          "archive_sha256": hashlib.sha256(archive).hexdigest()}
    return entry


class AcquisitionTests(unittest.TestCase):
    def test_matching_existing_file_is_verified_without_network(self):
        entry = asset()
        with tempfile.TemporaryDirectory() as tmp, patch.object(module, "get") as request:
            path = Path(tmp) / entry["path"]
            path.parent.mkdir(parents=True)
            path.write_bytes(b"source")
            report = module.download_selected(manifest(entry), ["fixture"], Path(tmp))
            self.assertEqual(report[0]["status"], "reused_verified_existing")
            request.assert_not_called()

    def test_existing_conflict_is_preserved_without_network(self):
        entry = asset()
        with tempfile.TemporaryDirectory() as tmp, patch.object(module, "get") as request:
            path = Path(tmp) / entry["path"]
            path.parent.mkdir(parents=True)
            path.write_bytes(b"local user data")
            with self.assertRaises(ValueError):
                module.download_selected(manifest(entry), ["fixture"], Path(tmp))
            self.assertEqual(path.read_bytes(), b"local user data")
            request.assert_not_called()

    def test_changed_source_is_never_published(self):
        entry = asset()
        with tempfile.TemporaryDirectory() as tmp, patch.object(module, "get", return_value=b"change"):
            with self.assertRaises(ValueError):
                module.download_selected(manifest(entry), ["fixture"], Path(tmp))
            self.assertFalse((Path(tmp) / entry["path"]).exists())
            self.assertEqual(list(Path(tmp).rglob("*.part-*")), [])

    def test_bad_gzip_crc_is_rejected_even_with_matching_compressed_hash(self):
        compressed = bytearray(gzip.compress(b"scientific data", mtime=0))
        compressed[-8] ^= 1
        entry = asset(bytes(compressed), name="fixture.gz")
        with tempfile.TemporaryDirectory() as tmp, patch.object(module, "get", return_value=bytes(compressed)):
            with self.assertRaises((ValueError, gzip.BadGzipFile)):
                module.download_selected(manifest(entry), ["fixture"], Path(tmp))
            self.assertFalse((Path(tmp) / entry["path"]).exists())

    def test_alternate_output_root_preserves_manifest_path_and_leaves_stale_part_alone(self):
        entry = asset()
        with tempfile.TemporaryDirectory() as tmp, patch.object(module, "get", return_value=b"source"):
            target = Path(tmp) / entry["path"]
            target.parent.mkdir(parents=True)
            part = target.with_name(target.name + ".part-old")
            part.write_bytes(b"incomplete older attempt")
            module.download_selected(manifest(entry), ["fixture"], Path(tmp))
            self.assertEqual(target.read_bytes(), b"source")
            self.assertEqual(part.read_bytes(), b"incomplete older attempt")

    def test_racing_destination_is_not_overwritten_and_temporary_is_removed(self):
        entry = asset()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / entry["path"]

            def fetch(*args):
                target.parent.mkdir(parents=True)
                target.write_bytes(b"concurrent user file")
                return b"source"

            with patch.object(module, "get", side_effect=fetch), self.assertRaises(FileExistsError):
                module.download_selected(manifest(entry), ["fixture"], Path(tmp))
            self.assertEqual(target.read_bytes(), b"concurrent user file")
            self.assertEqual(list(target.parent.glob("*.part-*")), [])

    def test_escaping_or_nonportable_output_paths_fail_before_network(self):
        names = ["../outside.dat", PREFIX + "../outside.dat", "/outside.dat",
                 "C:/outside.dat", PREFIX + "..\\outside.dat", PREFIX + "a//b.dat"]
        for name in names:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                entry = asset()
                entry["path"] = name
                with patch.object(module, "get") as request, self.assertRaises(ValueError):
                    module.download_selected(manifest(entry), ["fixture"], Path(tmp))
                request.assert_not_called()

    def test_duplicate_ids_paths_and_unknown_selection_fail_before_network(self):
        one = asset()
        two = asset(asset_id="second")
        bad_manifests = [manifest(one, copy.deepcopy(one)), manifest(one, two)]
        for value in bad_manifests:
            with tempfile.TemporaryDirectory() as tmp, patch.object(module, "get") as request:
                with self.assertRaises(ValueError):
                    module.download_selected(value, ["fixture"], Path(tmp))
                request.assert_not_called()
        with tempfile.TemporaryDirectory() as tmp, patch.object(module, "get") as request:
            with self.assertRaises(ValueError):
                module.download_selected(manifest(one), ["not_present"], Path(tmp))
            request.assert_not_called()

    def test_list_is_offline_and_does_not_create_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(module, "get") as request:
            root = Path(tmp)
            (root / "manifest.json").write_text(json.dumps(manifest(asset())), encoding="utf-8")
            output = root / "not_created"
            arguments = ["download", "--repo-root", str(root), "--manifest", "manifest.json",
                         "--output-root", str(output), "--list"]
            with patch("sys.argv", arguments), contextlib.redirect_stdout(io.StringIO()) as captured:
                module.main()
            self.assertIn("fixture 6", captured.getvalue())
            self.assertFalse(output.exists())
            request.assert_not_called()

    def test_ignored_range_is_rejected_before_reading_full_archive(self):
        response = Response(b"whole archive", status=200)
        with patch.object(module.urllib.request, "urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "full archive refused"):
                module.get("https://example.invalid/large.zip", 30, (10, 39))
        self.assertEqual(response.read_calls, 0)

    def test_shifted_truncated_and_oversized_range_responses_are_rejected(self):
        cases = [
            Response(b"1234", status=206, headers={"Content-Range": "bytes 1-4/100"}),
            Response(b"123", status=206, headers={"Content-Range": "bytes 0-3/100"}),
            Response(b"12345", status=206, headers={"Content-Range": "bytes 0-3/100"}),
        ]
        for response in cases:
            with self.subTest(headers=response.headers):
                with patch.object(module.urllib.request, "urlopen", return_value=response):
                    with self.assertRaises(ValueError):
                        module.get("https://example.invalid/large.zip", 4, (0, 3))

    def test_valid_range_checks_headers_and_exact_payload(self):
        response = Response(b"1234", status=206,
                            headers={"Content-Range": "bytes 10-13/100", "Content-Length": "4"})
        with patch.object(module.urllib.request, "urlopen", return_value=response) as request:
            self.assertEqual(module.get("https://example.invalid/large.zip", 4, (10, 13)), b"1234")
        self.assertEqual(request.call_args.args[0].get_header("Range"), "bytes=10-13")

    def test_zip_download_fetches_only_selected_member_ranges(self):
        payload = gzip.compress(b"grid coordinates", mtime=0)
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("case/grid.gz", payload)
            archive.writestr("other/large.dat", b"unselected" * 1000)
            info = archive.getinfo("case/grid.gz")
        archive_bytes = stream.getvalue()
        entry = asset(payload, name="grid.gz")
        entry["retrieval"] = {"kind": "zip_member_http_range", "archive_member": info.filename,
                              "zip_local_header_offset": info.header_offset,
                              "compressed_bytes": info.compress_size,
                              "zip_compression_method": info.compress_type,
                              "zip_crc32": f"{info.CRC:08x}"}

        def fetch(url, limit, byte_range):
            start, end = byte_range
            data = archive_bytes[start:end + 1]
            self.assertLessEqual(len(data), limit)
            return data

        with tempfile.TemporaryDirectory() as tmp, patch.object(module, "get", side_effect=fetch) as request:
            module.download_selected(manifest(entry), ["fixture"], Path(tmp))
            self.assertEqual((Path(tmp) / entry["path"]).read_bytes(), payload)
            self.assertEqual(request.call_count, 3)
            self.assertTrue(all(call.args[2] is not None for call in request.call_args_list))

    def test_zip_local_member_name_and_crc_must_match_pins(self):
        payload = b"grid"
        name = b"case/grid.dat"
        crc = zlib.crc32(payload) & 0xffffffff
        header = struct.pack("<4s5H3L2H", b"PK\x03\x04", 20, 0, 0, 0, 0,
                             crc, len(payload), len(payload), len(name), 0)
        entry = asset(payload)
        entry["retrieval"] = {"kind": "zip_member_http_range", "archive_member": "case/other.dat",
                              "zip_local_header_offset": 0, "compressed_bytes": len(payload),
                              "zip_compression_method": 0, "zip_crc32": f"{crc:08x}"}
        with patch.object(module, "get", side_effect=[header, name]), self.assertRaises(ValueError):
            module.zip_member(entry, 1024)
        entry["retrieval"]["archive_member"] = name.decode()
        entry["retrieval"]["zip_crc32"] = "00000000"
        with patch.object(module, "get", side_effect=[header, name]), self.assertRaises(ValueError):
            module.zip_member(entry, 1024)

    def test_tar_archive_is_verified_once_for_multiple_selected_members(self):
        first, second = b"first grid", b"second input"
        archive = tar_payload([("case/one.dat", first, "file"), ("case/two.dat", second, "file")])
        one = tar_asset(first, archive, "case/one.dat", name="one.dat", asset_id="one")
        two = tar_asset(second, archive, "case/two.dat", name="two.dat", asset_id="two")
        with tempfile.TemporaryDirectory() as tmp, patch.object(module, "get", return_value=archive) as request:
            module.download_selected(manifest(one, two), ["one", "two"], Path(tmp))
            self.assertEqual(request.call_count, 1)
            self.assertEqual((Path(tmp) / one["path"]).read_bytes(), first)
            self.assertEqual((Path(tmp) / two["path"]).read_bytes(), second)

    def test_changed_tar_archive_is_rejected_without_publishing(self):
        archive = tar_payload([("case/source.dat", b"source", "file")])
        entry = tar_asset(b"source", archive)
        changed = bytearray(archive)
        changed[-1] ^= 1
        with tempfile.TemporaryDirectory() as tmp, patch.object(module, "get", return_value=bytes(changed)):
            with self.assertRaisesRegex(ValueError, "archive size or SHA-256"):
                module.download_selected(manifest(entry), ["fixture"], Path(tmp))
            self.assertFalse((Path(tmp) / entry["path"]).exists())

    def test_tar_selected_symlink_or_duplicate_member_is_rejected(self):
        cases = [[("case/source.dat", b"", "symlink")],
                 [("case/source.dat", b"source", "file"), ("case/source.dat", b"source", "file")]]
        for members in cases:
            archive = tar_payload(members)
            entry = tar_asset(b"source", archive)
            with self.subTest(members=members), tempfile.TemporaryDirectory() as tmp:
                with patch.object(module, "get", return_value=archive), self.assertRaises(ValueError):
                    module.download_selected(manifest(entry), ["fixture"], Path(tmp))
                self.assertFalse((Path(tmp) / entry["path"]).exists())

    def test_tar_traversal_member_is_rejected_before_download(self):
        archive = tar_payload([("../source.dat", b"source", "file")])
        entry = tar_asset(b"source", archive, "../source.dat")
        with tempfile.TemporaryDirectory() as tmp, patch.object(module, "get") as request:
            with self.assertRaises(ValueError):
                module.download_selected(manifest(entry), ["fixture"], Path(tmp))
            request.assert_not_called()

    def test_explicit_size_budget_is_enforced_before_any_fetch(self):
        entry = asset(b"longer than budget")
        with tempfile.TemporaryDirectory() as tmp, patch.object(module, "get") as request:
            with self.assertRaises(ValueError):
                module.download_selected(manifest(entry), ["fixture"], Path(tmp), max_bytes=3)
            request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
