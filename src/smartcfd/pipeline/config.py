"""Versioned configuration and run receipts for reproducible local experiments."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import subprocess
from dataclasses import asdict
from smartcfd.core.types import ENVIRONMENT_VERSION, EnvConfig, MeshSpec
from smartcfd.pipeline.data import repository_root, data_version


def read_config(path: str | Path | None = None) -> dict:
    import yaml
    if path is None:
        path = repository_root() / "configs/rl.yaml"
    else:
        path = Path(path)
        if not path.is_absolute():
            path = repository_root() / path
    result = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    allowed = {"version", "domain", "env", "model", "training", "data", "output", "mesh", "cfd"}
    unknown = set(result) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    environment_config(result)
    return result


def environment_config(config: dict) -> EnvConfig:
    """The exported MeshSpec is also the MDP's terminal acceptance contract.

    Explicit conflicting settings are errors rather than silent overrides.
    Legacy dictionaries may still be inspected, but cannot execute as v2.
    """
    settings = dict(config.get('env', {}))
    spec = MeshSpec(**config.get('mesh', {}))
    for name, value in asdict(spec).items():
        if name in settings and settings[name] != value:
            raise ValueError(f'env.{name} conflicts with final mesh.{name}; terminal acceptance must match export')
        settings[name] = value
    if settings.get('environment_version', ENVIRONMENT_VERSION) != ENVIRONMENT_VERSION:
        raise ValueError('Legacy environment configuration requires explicit trajectory migration')
    settings['environment_version'] = ENVIRONMENT_VERSION
    result = EnvConfig(**settings)
    if (min(spec.intervals, spec.protected_intervals) < 2
            or spec.max_intervals < max(spec.intervals, spec.protected_intervals)
            or spec.max_cells < 1 or spec.smoothing_steps < 0):
        raise ValueError('Invalid final mesh budget')
    return result


def environment_context(config: dict) -> dict:
    env = environment_config(config)
    domain = {'farfield': 500., 'stretch': 18., **config.get('domain', {})}
    return {'environment_version': ENVIRONMENT_VERSION, 'env_config': asdict(env),
            'mesh_spec': asdict(env.mesh_spec), 'domain_config': {k: float(v) for k, v in domain.items()}}


def new_run(kind: str, config: dict, output: str | Path | None = None) -> Path:
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        output = repository_root() / "experiments/runs" / f"{kind}_{stamp}"
    else:
        output = Path(output)
        if not output.is_absolute():
            output = repository_root() / output
    output = output.resolve()
    if not output.is_relative_to(repository_root().resolve()):
        raise ValueError('Output must stay inside the local implementation repository')
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"kind": kind, "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "config": config, "data": data_version(), "status": "started", "scope": "local implementation copy"}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository_root(), capture_output=True, text=True)
    receipt["commit"] = revision.stdout.strip() if revision.returncode == 0 else None
    receipt['implementation'] = implementation_version()
    receipt['environment_context'] = environment_context(config)
    write_json(output / "run.json", receipt)
    return output


def implementation_version() -> dict:
    root = repository_root()
    files = sorted([*root.glob('src/**/*.py'), *root.glob('configs/*'), root / 'pyproject.toml'])
    digest = hashlib.sha256()
    for path in files:
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b'\0')
            digest.update(path.read_bytes())
    packages = {}
    for name in ('numpy', 'torch', 'gymnasium', 'PyYAML', 'matplotlib'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {'source_sha256': digest.hexdigest(), 'packages': packages}


def write_json(path: str | Path, value) -> None:
    import numpy as np
    def encode(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.generic): return obj.item()
        if isinstance(obj, Path): return obj.as_posix()
        raise TypeError(type(obj).__name__)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=encode, allow_nan=False) + "\n", encoding="utf-8")


def complete_run(path: Path, result: dict, status: str = "completed") -> None:
    receipt = json.loads((path / "run.json").read_text(encoding="utf-8"))
    receipt.update(status=status, finished_at_utc=datetime.now(timezone.utc).isoformat(), result=result)
    write_json(path / "run.json", receipt)
