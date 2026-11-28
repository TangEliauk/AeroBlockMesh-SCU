"""Freeze a small group-aware development manifest from existing train/val."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import random

from smartcfd.pipeline.data import load_samples

ROOT = Path(__file__).resolve().parents[3]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def select_cases(seed=42, train_count=32, val_count=8, *, root=None):
    """Selection precedes physical preflight; failures must never be replaced."""
    root = Path(root) if root else ROOT
    if any(type(n) is not int or n < 1 for n in (train_count, val_count)):
        raise ValueError('Positive train/val counts required')
    group_path = root/'data/splits/uiuc_v1/groups.json'
    groups = json.loads(group_path.read_text(encoding='utf-8'))
    rng = random.Random(seed)
    result = {'train': [], 'val': []}
    membership = {}
    for group in groups:
        for sample_id in group['sample_ids']:
            if sample_id in membership:
                raise ValueError('Duplicate geometry group membership')
            membership[sample_id] = group
    for split, count in (('train', train_count), ('val', val_count)):
        rows = load_samples(split, root=root, sharp_only=True)
        by_group = {}
        for sample in rows:
            group = membership[sample['sample_id']]
            if group['split'] != split or group.get('forced_test'):
                raise ValueError('Frozen development split includes an incompatible/held-out group')
            by_group.setdefault(group['group_id'], []).append(sample)
        ordered = sorted(by_group)
        rng.shuffle(ordered)
        for group_id in ordered:
            by_group[group_id].sort(key=lambda row: row['sample_id'])
            rng.shuffle(by_group[group_id])
        if sum(map(len, by_group.values())) < count:
            raise ValueError(f'Not enough eligible {split} samples; refusing silent count changes')
        while len(result[split]) < count:
            for group_id in ordered:
                if not by_group[group_id]:
                    continue
                sample = by_group[group_id].pop(0)
                group = membership[sample['sample_id']]
                result[split].append({**sample, 'split': split, 'group_id': group_id,
                    'family_keys': group['family_keys'], 'source': 'frozen_uiuc_v1_development',
                    'expert_demonstration': False})
                if len(result[split]) == count:
                    break
    group_sets = [{row['group_id'] for row in result[s]} for s in ('train', 'val')]
    if group_sets[0] & group_sets[1]:
        raise ValueError('Train/val family overlap')
    file_names = ['data/processed/uiuc_v1/samples.jsonl', 'data/splits/uiuc_v1/groups.json',
                  'data/splits/uiuc_v1/train.json', 'data/splits/uiuc_v1/val.json',
                  'data/splits/uiuc_v1/split_config.json']
    result['manifest'] = {'seed': seed, 'selection': 'seeded group round-robin before physical preflight',
        'counts': {s: len(result[s]) for s in ('train', 'val')},
        'group_counts': {s: len({r['group_id'] for r in result[s]}) for s in ('train', 'val')},
        'source_files': {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in file_names},
        'scope': 'Fixed train/val development only; no test/NACA evaluation or expert actions'}
    return result


def preflight(cases, config):
    from .geometry import CurveGeometry
    from smartcfd.partition.topology import initial_graph
    reports = {'train': [], 'val': []}
    for split in reports:
        for sample in cases[split]:
            try:
                geometry = CurveGeometry(sample, config)
                result = geometry.validate(initial_graph(geometry.domain), refine=True)
                record = {'sample_id': sample['sample_id'], 'valid': bool(result['valid']),
                          'reason': result.get('reason'), 'quality': result.get('quality', {})}
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                record = {'sample_id': sample['sample_id'], 'valid': False,
                          'reason': f'{type(exc).__name__}: {exc}', 'quality': {}}
            reports[split].append(record)
    return reports
