"""Read immutable geometry and family splits without creating new partitions."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def repository_root() -> Path:
    for candidate in (*Path(__file__).resolve().parents, Path.cwd(), *Path.cwd().parents):
        if (candidate / 'configs/rl.yaml').is_file() and (candidate / 'data/splits/uiuc_v1/train.json').is_file():
            return candidate
    raise FileNotFoundError('Run this package from an AeroBlockMesh checkout containing configs and frozen data')


def sharp_eligible(sample: dict) -> bool:
    """Compatibility bool wrapper around the single domain eligibility rule."""
    from smartcfd.geometry.domain import sharp_eligibility
    return sharp_eligibility(sample)[0]


def load_samples(split: str = "train", *, root: Path | None = None,
                 sharp_only: bool = True, limit: int | None = None) -> list[dict]:
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train, val or test")
    root = Path(root) if root else repository_root()
    ids = set(json.loads((root / f"data/splits/uiuc_v1/{split}.json").read_text(encoding="utf-8")))
    path = root / "data/processed/uiuc_v1/samples.jsonl"
    samples = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            sample = json.loads(line)
            if sample["sample_id"] in ids and (not sharp_only or sharp_eligible(sample)):
                samples.append(sample)
    samples.sort(key=lambda row: row["sample_id"])
    return samples if limit is None else samples[:limit]


def data_version(root: Path | None = None) -> dict:
    root = Path(root) if root else repository_root()
    files = ["data/processed/uiuc_v1/samples.jsonl", "data/splits/uiuc_v1/split_config.json",
             "data/splits/uiuc_v1/train.json", "data/splits/uiuc_v1/val.json", "data/splits/uiuc_v1/test.json"]
    return {"version": "public_v1_2026-11-22", "files": {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}}


def load_domains(split: str = "train", *, limit: int | None = None, root: Path | None = None,
                 farfield: float = 500., stretch: float = 18.) -> list:
    from smartcfd.geometry.domain import make_domain
    domains = []
    for sample in load_samples(split, limit=limit, root=root):
        domain = make_domain(sample)
        domain.farfield, domain.stretch = farfield, stretch
        domains.append(domain)
    return domains
