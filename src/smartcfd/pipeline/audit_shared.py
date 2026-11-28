"""Canonical identities shared by demonstration audit, review and training."""
from __future__ import annotations
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path

EPISODE_SCHEMA = 'audited_episode_v2'
PACKAGE_SCHEMA = 'demo_package_v2'
LEGACY_EPISODE_SCHEMA = 'audited_episode_v1'


def _encode(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError(type(value).__name__)


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False, default=_encode)


def canonical_hash(value):
    return hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def episode_id_for(record):
    """The terminal mesh contract is part of the MDP, unlike seeds/reviews."""
    keys = ('geometry_sha256', 'actions', 'env_config', 'domain_config',
            'mapping_version', 'observation_version', 'action_version',
            'environment_version', 'mesh_spec')
    return canonical_hash({key: record[key] for key in keys})


def package_id_for(manifest):
    return canonical_hash({key: value for key, value in manifest.items() if key != 'package_id'})


def load_jsonl(path):
    records = []
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    return records
