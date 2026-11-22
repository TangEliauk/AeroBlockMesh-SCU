"""Fetch explicitly selected, hash-pinned reference assets; no assets are implicit.

Supports direct URLs, an official-page download link, and bounded HTTP Range
extraction of a ZIP member. Large reference payloads stay outside the Git release.
"""
import argparse
import gzip
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import struct
import urllib.parse
import urllib.request
import zipfile
import zlib


def get(url, limit, byte_range=None):
    headers = {"User-Agent": "AeroBlockMesh-reference-data/1.0"}
    if byte_range:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as response:
        if byte_range and response.status != 206:
            raise ValueError("Server did not honor bounded HTTP Range; full archive download refused")
        announced = response.headers.get("Content-Length")
        if announced and int(announced) > limit:
            raise ValueError("Response exceeds size limit")
        payload = response.read(limit + 1)
        if len(payload) > limit:
            raise ValueError("Response exceeds size limit")
        if byte_range:
            expected = byte_range[1] - byte_range[0] + 1
            content_range = response.headers.get("Content-Range", "")
            if len(payload) != expected or not content_range.startswith(f"bytes {byte_range[0]}-{byte_range[1]}/"):
                raise ValueError("HTTP Range payload or Content-Range does not match request")
        return payload


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []
    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            self.hrefs.extend(value for key, value in attrs if key.lower() == "href" and value)


def fetch(asset, max_bytes):
    if asset["bytes"] > max_bytes:
        raise ValueError("Selected artifact exceeds --max-bytes")
    retrieval = asset["retrieval"]
    url = asset["download_url"]
    if retrieval["kind"] == "official_page_link":
        page = get(url, 2 * 1024 * 1024).decode("utf-8")
        parser = Links()
        parser.feed(page)
        matches = [urllib.parse.urljoin(url, href) for href in parser.hrefs
                   if urllib.parse.unquote(urllib.parse.urlsplit(href).path).endswith("/" + retrieval["link_file_name"])]
        matches = list(dict.fromkeys(matches))
        if len(matches) != 1:
            raise ValueError("Could not unambiguously resolve the official CGNS download link")
        url = matches[0]
    if retrieval["kind"] == "zip_member_http_range":
        offset = retrieval["zip_local_header_offset"]
        header = get(url, 30, (offset, offset + 29))
        fields = struct.unpack("<4s5H3L2H", header)
        if fields[0] != b"PK\x03\x04" or fields[2] & 1:
            raise ValueError("Invalid or encrypted ZIP local header")
        name_bytes, extra_bytes = fields[-2:]
        start = offset + 30 + name_bytes + extra_bytes
        length = retrieval["compressed_bytes"]
        compressed = get(url, length, (start, start + length - 1))
        method = retrieval["zip_compression_method"]
        if method == 0:
            payload = compressed
        elif method == 8:
            payload = zlib.decompress(compressed, -15)
        else:
            raise ValueError("Unsupported ZIP compression method")
        if f"{zlib.crc32(payload) & 0xffffffff:08x}" != retrieval["zip_crc32"]:
            raise ValueError("ZIP member CRC mismatch")
        return payload
    return get(url, min(max_bytes, asset["bytes"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True, help="Local cache directory, excluded from the public Git payload")
    parser.add_argument("--ids", nargs="+", help="Explicit asset IDs to fetch")
    parser.add_argument("--list", action="store_true", help="List asset IDs and byte sizes without network requests")
    parser.add_argument("--max-bytes", type=int, default=160000000, help="Maximum individual artifact size (default 160 MB)")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    assets = {item["asset_id"]: item for item in manifest["assets"]}
    if args.list:
        for asset in assets.values():
            print(asset["asset_id"], asset["bytes"], asset["file_name"])
        return
    if not args.ids:
        parser.error("--ids is required unless --list is used")
    if any(asset_id not in assets for asset_id in args.ids):
        parser.error("Unknown asset ID")
    args.output_root.mkdir(parents=True, exist_ok=True)
    report = []
    for asset_id in args.ids:
        asset = assets[asset_id]
        filename = asset["file_name"]
        if Path(filename).name != filename or filename in ("", ".", ".."):
            raise ValueError("Unsafe artifact filename")
        folder = args.output_root / asset_id
        folder.mkdir(exist_ok=True)
        target = folder / filename
        if target.is_file():
            payload = target.read_bytes()
            status = "reused_verified_existing"
        else:
            payload = fetch(asset, args.max_bytes)
            status = "downloaded_verified"
        digest = hashlib.sha256(payload).hexdigest()
        if len(payload) != asset["bytes"] or digest != asset["sha256"]:
            raise ValueError("Artifact length or SHA-256 differs from release manifest: " + asset_id)
        if filename.endswith(".gz"):
            # Full decompression validates the gzip CRC; resulting bytes are not saved.
            gzip.decompress(payload)
        if not target.exists():
            part = target.with_name(target.name + ".part")
            with part.open("xb") as stream:
                stream.write(payload)
            part.rename(target)
        report.append(dict(asset_id=asset_id, file_name=filename, bytes=len(payload), sha256=digest, status=status))
        print(asset_id, status, len(payload), flush=True)
    (args.output_root / "download_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
