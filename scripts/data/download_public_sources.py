"""Download checksum-pinned small public inputs for the published data pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(entry: dict, output: Path) -> Path:
    output = output.resolve()
    path = (output / entry["local_path"]).resolve()
    if not path.is_relative_to(output):
        raise ValueError("Source registry path escapes output directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and sha256(path) == entry["sha256"]:
        print(f"already present: {entry['id']}")
        return path
    if path.exists():
        raise ValueError(f"Existing file differs from pinned source; preserved: {path}")
    part = path.with_name(path.name + ".part")
    request = urllib.request.Request(entry["url"], headers={"User-Agent": "AeroBlockMesh-SCU-public-data/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    if part.stat().st_size != entry["bytes"] or sha256(part) != entry["sha256"]:
        raise ValueError(f"Upstream bytes/hash changed; file left as .part for inspection: {entry['id']}")
    part.rename(path)
    print(f"downloaded: {entry['id']}")
    return path


def extract_checked(archive: Path, output: Path) -> None:
    destination = output.resolve()
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("UIUC ZIP CRC failed")
        for member in z.infolist():
            path = (destination / member.filename).resolve()
            if not path.is_relative_to(destination):
                raise ValueError("Archive member escapes output directory")
            if member.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            payload = z.read(member)
            if path.exists():
                if path.read_bytes() != payload:
                    raise ValueError(f"Existing extracted source differs; preserved: {path}")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)


def main() -> None:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=repository / "data/sources/public_v1/raw_sources.json")
    parser.add_argument("--output", type=Path, default=repository / "data/raw/public_v1")
    parser.add_argument("--group", choices=("all", "geometry", "experiments"), default="all")
    parser.add_argument("--extract-uiuc", action="store_true")
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    for entry in registry["sources"]:
        if args.group != "all" and entry["group"] != args.group:
            continue
        path = download(entry, args.output)
        if args.extract_uiuc and entry["id"] == "uiuc_selig_archive":
            extract_checked(path, args.output / "uiuc/coordinates")
    print("Pinned input acquisition complete. Raw inputs retain their upstream attribution.")


if __name__ == "__main__":
    main()
