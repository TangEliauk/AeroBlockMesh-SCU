"""Load human-reviewed train episodes and separately audited validation data."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path

from smartcfd.core.types import ENVIRONMENT_VERSION, Failure, MeshSpec
from smartcfd.geometry.domain import make_domain
from smartcfd.pipeline.data import data_version, load_samples, repository_root
from smartcfd.pipeline.config import environment_config


def load_demo_package(value, config):
    """Validate review lineage before constructing any learning agent.

    All transitions are recreated in the current environment. Validation
    transitions are returned separately and never inserted in training replay.
    """
    from smartcfd.pipeline.review import validate_training_package
    from smartcfd.rl.env import BlockEnv
    from smartcfd.rl.imitation.dataset import replay_audited_episode
    from smartcfd.meshing.generator import generate_mesh
    from smartcfd.cfd.solver import validate_su2

    path = Path(value)
    if not path.is_absolute():
        path = repository_root() / path
    manifest, train_records, val_records = validate_training_package(path, config=config)
    if manifest.get('training_ready') is not True or not train_records or not val_records:
        raise ValueError('Demonstration package is not training-ready: human-approved train and audited val episodes are both required')
    samples = {split: {s['sample_id']: s for s in load_samples(split)} for split in ('train', 'val')}
    frozen = data_version()
    spec = MeshSpec(**config.get('mesh', {}))
    transitions = {'train': [], 'val': []}
    for split, records in (('train', train_records), ('val', val_records)):
        for record in records:
            if record.get('split') != split or record.get('sample_id') not in samples[split]:
                raise ValueError('Reviewed package split conflicts with frozen eligible membership')
            domain = make_domain(samples[split][record['sample_id']], **config.get('domain', {}))
            env = BlockEnv([domain], environment_config(config))
            try:
                replayed = replay_audited_episode(record, env, {'data': frozen, 'mesh_spec': asdict(spec)})
                mesh = generate_mesh(domain, env.graph, spec)
                if isinstance(mesh, Failure) or not mesh.quality.get('valid', False):
                    raise ValueError(f"Reviewed episode {record['episode_id']} no longer generates a valid fine mesh")
                validate_su2(mesh)
                transitions[split].extend(replayed)
            finally:
                env.close()
    report = {'package_path': str(path.resolve()), 'source_package_id': manifest.get('source_package_id'),
        'train_episodes': len(train_records), 'val_episodes': len(val_records),
        'train_unique_geometries': len({r['sample_id'] for r in train_records}),
        'val_unique_geometries': len({r['sample_id'] for r in val_records}),
        'train_transitions': len(transitions['train']), 'val_transitions': len(transitions['val']),
        'train_source': 'human-reviewed audited demonstrations; source labels preserved',
        'environment_version': ENVIRONMENT_VERSION, 'mesh_spec': asdict(spec),
        'test_used': False, 'validation_enters_replay': False}
    return transitions['train'], transitions['val'], report
