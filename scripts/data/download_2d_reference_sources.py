"""Fetch explicitly selected, hash-pinned originals for multiblock_2d_v1.

Only Python's standard library is required. --output-root is an alternate
repository root: every manifest path remains relative to that root. Existing
matching files are verified without network access; conflicting files are never
overwritten. ZIP retrieval uses bounded HTTP Range, with no whole-archive fallback.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
import tarfile
import tempfile
from urllib.parse import urlsplit
import urllib.request
import zlib


RAW_PREFIX = "data/references/multiblock_2d_v1/"
DEFAULT_MANIFEST = RAW_PREFIX + "manifest.json"
DEFAULT_MAX_BYTES = 64 * 1024 * 1024


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def positive_integer(value: object, field: str) -> int:
    require(type(value) is int and value > 0, f"Invalid {field}")
    return value


def safe_member_name(name: str) -> str:
    require(isinstance(name, str) and bool(name), "Missing archive member name")
    path = PurePosixPath(name)
    require(not path.is_absolute() and path.as_posix() == name
            and ".." not in path.parts and "\\" not in name and ":" not in name,
            "Unsafe archive member name")
    return name


def safe_relative_path(name: str) -> str:
    safe_member_name(name)
    require(name.startswith(RAW_PREFIX), "Asset path outside multiblock_2d_v1")
    require(len(PurePosixPath(name).parts) >= 5, "Asset path must include a case directory")
    return name


def target_path(output_root: Path, relative: str) -> Path:
    root = output_root.resolve()
    target = root / safe_relative_path(relative)
    require(target.resolve().is_relative_to(root), "Asset path escapes output root")
    current = target
    while current != root:
        require(not current.is_symlink(), "Symlink in asset output path")
        current = current.parent
    return target


def get(url: str, limit: int, byte_range: tuple[int, int] | None = None) -> bytes:
    """Read a bounded response; reject ignored, shifted or truncated ranges."""
    positive_integer(limit, "response limit")
    parsed = urlsplit(url)
    require(parsed.scheme == "https" and bool(parsed.netloc), "Expected HTTPS source URL")
    headers = {"User-Agent": "AeroBlockMesh-reference-data/1.0"}
    if byte_range is not None:
        start, end = byte_range
        require(type(start) is int and type(end) is int and 0 <= start <= end,
                "Invalid HTTP byte range")
        require(end - start + 1 <= limit, "HTTP Range exceeds response limit")
        headers["Range"] = f"bytes={start}-{end}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        require(response.status == (206 if byte_range is not None else 200),
                "Server did not honor bounded HTTP Range; full archive refused"
                if byte_range is not None else "Unexpected source HTTP status")
        announced = response.headers.get("Content-Length")
        if announced is not None:
            require(announced.isdigit() and int(announced) <= limit,
                    "Response exceeds size limit or has invalid Content-Length")
        if byte_range is not None:
            start, end = byte_range
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",
                                 response.headers.get("Content-Range", ""))
            require(match is not None and tuple(map(int, match.groups()[:2])) == byte_range
                    and int(match.group(3)) > end, "HTTP Content-Range differs from request")
        payload = response.read(limit + 1)
        require(len(payload) <= limit, "Response exceeds size limit")
        if byte_range is not None:
            require(len(payload) == end - start + 1, "Truncated HTTP Range response")
        if announced is not None:
            require(len(payload) == int(announced), "Response length differs from Content-Length")
        return payload


def validate_manifest(manifest: dict) -> list[dict]:
    require(manifest.get("schema_version") == 1
            and manifest.get("version") == "multiblock_2d_v1", "Unsupported source manifest")
    entries = manifest.get("raw_files")
    require(isinstance(entries, list) and bool(entries), "Missing raw_files")
    ids, paths = set(), set()
    for entry in entries:
        require(isinstance(entry, dict), "Invalid source asset")
        asset_id = entry.get("asset_id")
        require(isinstance(asset_id, str) and re.fullmatch(r"[a-z0-9_]+", asset_id),
                "Invalid asset_id")
        require(asset_id not in ids, "Duplicate source asset ID")
        ids.add(asset_id)
        relative = safe_relative_path(entry.get("path"))
        require(relative.casefold() not in paths, "Duplicate source asset path")
        paths.add(relative.casefold())
        positive_integer(entry.get("bytes"), "asset byte size")
        require(isinstance(entry.get("sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), "Invalid source SHA-256")
        parsed = urlsplit(entry.get("source_url", ""))
        require(parsed.scheme == "https" and bool(parsed.netloc), "Expected HTTPS source URL")
        retrieval = entry.get("retrieval", {})
        require(retrieval.get("kind") in {"direct", "zip_member_http_range", "tar_member"},
                "Unsupported retrieval kind")
        if retrieval["kind"] != "direct":
            safe_member_name(retrieval.get("archive_member"))
        if retrieval["kind"] == "tar_member":
            positive_integer(retrieval.get("archive_bytes"), "tar archive size")
            require(isinstance(retrieval.get("archive_sha256"), str)
                    and re.fullmatch(r"[0-9a-f]{64}", retrieval["archive_sha256"]),
                    "Invalid archive SHA-256")
        if retrieval["kind"] == "zip_member_http_range":
            require(type(retrieval.get("zip_local_header_offset")) is int
                    and retrieval["zip_local_header_offset"] >= 0, "Invalid ZIP offset")
            positive_integer(retrieval.get("compressed_bytes"), "ZIP compressed byte size")
            require(retrieval.get("zip_compression_method") in {0, 8}, "Unsupported ZIP method")
            require(isinstance(retrieval.get("zip_crc32"), str)
                    and re.fullmatch(r"[0-9a-f]{8}", retrieval["zip_crc32"]), "Invalid ZIP CRC")
    return entries


def zip_member(entry: dict, max_bytes: int) -> bytes:
    retrieval, url = entry["retrieval"], entry["source_url"]
    offset = retrieval["zip_local_header_offset"]
    header = get(url, 30, (offset, offset + 29))
    fields = struct.unpack("<4s5H3L2H", header)
    require(fields[0] == b"PK\x03\x04" and not fields[2] & 1,
            "Invalid or encrypted ZIP local header")
    require(fields[3] == retrieval["zip_compression_method"], "ZIP method differs from manifest")
    name_bytes, extra_bytes = fields[-2:]
    require(name_bytes > 0, "Missing ZIP member filename")
    metadata_size = name_bytes + extra_bytes
    metadata = get(url, metadata_size, (offset + 30, offset + 29 + metadata_size))
    encoding = "utf-8" if fields[2] & (1 << 11) else "cp437"
    name = metadata[:name_bytes].decode(encoding)
    require(safe_member_name(name) == retrieval["archive_member"],
            "ZIP local member name differs from manifest")
    if not fields[2] & 8:  # Without a data descriptor the local values must also match.
        require((fields[6], fields[7], fields[8]) ==
                (int(retrieval["zip_crc32"], 16), retrieval["compressed_bytes"], entry["bytes"]),
                "ZIP local size or CRC differs from manifest")
    length = retrieval["compressed_bytes"]
    require(length <= max_bytes, "ZIP compressed member exceeds --max-bytes")
    start = offset + 30 + metadata_size
    compressed = get(url, length, (start, start + length - 1))
    if retrieval["zip_compression_method"] == 0:
        payload = compressed
    else:
        decoder = zlib.decompressobj(-15)
        payload = decoder.decompress(compressed, entry["bytes"] + 1)
        require(len(payload) <= entry["bytes"] and decoder.eof
                and not decoder.unused_data and not decoder.unconsumed_tail,
                "ZIP member decompression differs from expected size or stream")
    require(len(payload) == entry["bytes"]
            and f"{zlib.crc32(payload) & 0xffffffff:08x}" == retrieval["zip_crc32"],
            "ZIP member size or CRC differs from manifest")
    return payload


def tar_member(entry: dict, max_bytes: int, archive_cache: dict) -> bytes:
    retrieval, url = entry["retrieval"], entry["source_url"]
    size = retrieval["archive_bytes"]
    require(size <= max_bytes, "Tar archive exceeds --max-bytes")
    cache_key = (url, size, retrieval["archive_sha256"])
    if cache_key not in archive_cache:
        payload = get(url, size)
        require(len(payload) == size and hashlib.sha256(payload).hexdigest() == cache_key[2],
                "Tar archive size or SHA-256 differs from manifest")
        archive_cache[cache_key] = payload
    with tarfile.open(fileobj=io.BytesIO(archive_cache[cache_key]), mode="r:gz") as archive:
        name = safe_member_name(retrieval["archive_member"])
        matches = [member for member in archive.getmembers() if member.name == name]
        require(len(matches) == 1, "Missing or duplicate selected tar member")
        member = matches[0]
        require(member.isfile() and not member.issparse(), "Selected tar member is not a regular file")
        require(member.size == entry["bytes"] and member.size <= max_bytes,
                "Tar member size differs from manifest")
        stream = archive.extractfile(member)
        require(stream is not None, "Tar member has no data")
        with stream:
            payload = stream.read(entry["bytes"] + 1)
        require(len(payload) == entry["bytes"], "Truncated tar member")
        return payload


def check_payload(entry: dict, payload: bytes, max_bytes: int) -> None:
    require(len(payload) == entry["bytes"]
            and hashlib.sha256(payload).hexdigest() == entry["sha256"],
            "Source size or SHA-256 differs from manifest: " + entry["asset_id"])
    if entry["path"].endswith(".gz"):
        # Exhaust the bounded decoded stream to validate every gzip member's CRC.
        with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as stream:
            decoded = stream.read(max_bytes + 1)
        require(len(decoded) <= max_bytes, "Decoded gzip exceeds --max-bytes")


def publish_without_overwrite(target: Path, payload: bytes) -> None:
    """Publish an exclusively created sibling temporary file, without replacement."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=target.name + ".part-",
                                         dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Windows rename refuses an existing destination. POSIX rename may
        # overwrite, so use exclusive hard-link creation on that platform.
        if os.name == "nt":
            os.rename(temporary, target)
        else:
            os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download(entry: dict, output_root: Path, *, max_bytes: int = DEFAULT_MAX_BYTES,
             archive_cache: dict | None = None) -> dict:
    positive_integer(max_bytes, "maximum byte size")
    require(entry["bytes"] <= max_bytes, "Selected source exceeds --max-bytes")
    target = target_path(output_root, entry["path"])
    if target.exists():
        require(target.is_file(), "Source target is not a regular file")
        require(target.stat().st_size == entry["bytes"],
                "Existing source size differs from manifest: " + entry["asset_id"])
        payload = target.read_bytes()
        status = "reused_verified_existing"
    else:
        retrieval = entry["retrieval"]["kind"]
        if retrieval == "direct":
            payload = get(entry["source_url"], entry["bytes"])
        elif retrieval == "zip_member_http_range":
            payload = zip_member(entry, max_bytes)
        elif retrieval == "tar_member":
            payload = tar_member(entry, max_bytes, archive_cache if archive_cache is not None else {})
        else:
            raise ValueError("Unsupported retrieval kind")
        status = "downloaded_verified"
    check_payload(entry, payload, max_bytes)
    if status == "downloaded_verified":
        # Recheck containment after network I/O and before creating any directories.
        require(target == target_path(output_root, entry["path"]), "Output path changed")
        publish_without_overwrite(target, payload)
    return {"asset_id": entry["asset_id"], "path": entry["path"],
            "bytes": entry["bytes"], "sha256": entry["sha256"], "status": status}


def download_selected(manifest: dict, ids: list[str], output_root: Path, *,
                      max_bytes: int = DEFAULT_MAX_BYTES) -> list[dict]:
    entries = validate_manifest(manifest)
    positive_integer(max_bytes, "maximum byte size")
    assets = {item["asset_id"]: item for item in entries}
    require(bool(ids) and len(set(ids)) == len(ids), "Select unique, explicit asset IDs")
    require(all(asset_id in assets for asset_id in ids), "Unknown source asset ID")
    require(all(assets[asset_id]["bytes"] <= max_bytes for asset_id in ids),
            "Selected source exceeds --max-bytes")
    # Validate every destination before any network requests or writes.
    for asset_id in ids:
        target_path(output_root, assets[asset_id]["path"])
    archive_cache = {}
    return [download(assets[asset_id], output_root, max_bytes=max_bytes,
                     archive_cache=archive_cache) for asset_id in ids]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--manifest", type=Path, help="Source manifest, relative to --repo-root if not absolute")
    parser.add_argument("--output-root", type=Path,
                        help="Alternate repository root; preserves data/references/... relative paths")
    parser.add_argument("--ids", nargs="+", help="Explicit raw asset IDs to fetch")
    parser.add_argument("--list", action="store_true", help="List all source IDs without network access")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES,
                        help="Maximum source, archive and decoded-gzip size (default 64 MiB)")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    path = args.manifest if args.manifest is not None else Path(DEFAULT_MANIFEST)
    if not path.is_absolute():
        path = root / path
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries = validate_manifest(manifest)
    if args.list:
        for entry in entries:
            print(entry["asset_id"], entry["bytes"], entry["path"])
        return
    if not args.ids:
        parser.error("--ids is required unless --list is used")
    report = download_selected(manifest, args.ids,
                               args.output_root.resolve() if args.output_root else root,
                               max_bytes=args.max_bytes)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
