"""Environment migrations never reuse old states, rewards or human approvals."""
from argparse import Namespace
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile

import pytest

from smartcfd.core.types import ENVIRONMENT_VERSION, EnvConfig, Failure, MeshSpec
from smartcfd.baselines.rule_partition import solve
from smartcfd.geometry.domain import make_domain
from smartcfd.pipeline import demo_audit, review
from smartcfd.pipeline.audit_shared import EPISODE_SCHEMA, PACKAGE_SCHEMA, canonical_hash, episode_id_for
from smartcfd.pipeline.config import environment_config, read_config
from smartcfd.pipeline.data import data_version, load_samples, repository_root
from smartcfd.rl.env import BlockEnv
from smartcfd.rl.imitation.dataset import validate_audited_episode


@pytest.fixture(scope='module')
def examples():
    config = read_config()
    all_samples = load_samples('train')
    by_alias = {name: sample for sample in all_samples for name in sample.get('aliases', [])}
    samples = {s['sample_id']: s for s in (all_samples[0], by_alias['goe15'], by_alias['goe481'])}
    membership = {sid: 'train' for sid in samples}
    domain = make_domain(all_samples[0], **config['domain'])
    env = BlockEnv([domain], environment_config(config))
    try:
        env.reset(seed=42, options={'sample_id': domain.sample_id})
        raw = {'sample_id': domain.sample_id, 'seed': 42, 'actions': solve(env, seed=42),
            'options': {'sample_id': domain.sample_id}, 'source': 'rule_synthetic',
            'env_config': asdict(env.config), 'environment_version': ENVIRONMENT_VERSION}
    finally:
        env.close()
    return config, samples, membership, raw, by_alias


def old_raw(raw):
    result = deepcopy(raw)
    result.pop('environment_version')
    for key in ('environment_version', 'max_intervals', 'smoothing_steps', 'max_cells'):
        result['env_config'].pop(key)
    result['env_config'].update(intervals=8, protected_intervals=16)
    return result


def audit(record, examples, migrate=False):
    config, samples, membership, _, _ = examples
    return demo_audit._audit_episode(record, config, samples, membership, data_version(),
                                    migrate_environment=migrate)


def test_old_environment_is_refused_without_explicit_migration(examples):
    old = old_raw(examples[3])
    with pytest.raises(demo_audit.AuditRejection, match='migrate-environment'):
        audit(old, examples)
    migrated = audit(old, examples, migrate=True)
    assert migrated['schema'] == EPISODE_SCHEMA
    assert migrated['environment_version'] == ENVIRONMENT_VERSION
    assert migrated['env_config']['intervals'] == 16 and migrated['env_config']['protected_intervals'] == 32
    assert migrated['terminal_mesh_spec'] == migrated['mesh_spec'] == asdict(environment_config(examples[0]).mesh_spec)
    assert migrated['terminal_quality']['cells'] == migrated['fine_quality']['cells'] == 3072
    origin = migrated['origin']['environment_migration']
    assert origin['source_env_config'] == old['env_config'] and origin['source_record_sha256'] == canonical_hash(old)
    assert migrated['human_reviewed'] is False and migrated['origin']['status'] == 'legacy_unknown'


def test_v1_identity_and_approval_are_preserved_only_as_provenance(examples):
    current = audit(examples[3], examples)
    old = deepcopy(current)
    old['schema'] = 'audited_episode_v1'
    old.pop('environment_version')
    old['env_config'] = old_raw(examples[3])['env_config']
    old['human_reviewed'] = True
    old['review'] = {'status': 'approved', 'reviewer': 'synthetic old verdict, not a user approval'}
    keys = ('geometry_sha256', 'actions', 'env_config', 'domain_config',
            'mapping_version', 'observation_version', 'action_version')
    old['episode_id'] = canonical_hash({key: old[key] for key in keys})
    migrated = audit(old, examples, migrate=True)
    assert migrated['episode_id'] != old['episode_id'] and migrated['episode_id'] == current['episode_id']
    assert migrated['human_reviewed'] is False and 'review' not in migrated
    migration = migrated['origin']['environment_migration']
    assert migration['source_episode_id'] == old['episode_id'] and migration['discarded_human_review'] is True
    again = audit(migrated, examples)
    assert again['origin']['environment_migration'] == migration
    assert again['origin']['status'] == 'legacy_unknown'
    old['episode_id'] = 'forged-old-identity'
    with pytest.raises(demo_audit.AuditRejection, match='Legacy identity'):
        audit(old, examples, migrate=True)


@pytest.mark.parametrize('change,reason', [
    ({'env_config': {'gamma': .5}}, 'env_config_mismatch'),
    ({'env_config': {'success_reward': 99}}, 'env_config_mismatch'),
    ({'env_config': {'intervals': 7}}, 'env_config_mismatch'),
    ({'domain_config': {'farfield': 600., 'stretch': 18.}}, 'domain_config_mismatch'),
    ({'data': {'version': 'other'}}, 'data_mismatch'),
    ({'geometry_sha256': 'other'}, 'geometry_sha256_mismatch'),
    ({'mapping_version': 'other'}, 'mapping_version_mismatch'),
    ({'action_version': 'other'}, 'action_version_mismatch'),
    ({'observation_version': 'other'}, 'observation_version_mismatch'),
    ({'environment_version': 'final_mesh_v99'}, 'environment_version_mismatch'),
    ({'mesh_spec': {'intervals': 32, 'protected_intervals': 64}}, 'mesh_spec_mismatch'),
])
def test_migration_cannot_bypass_unrelated_context_conflicts(examples, change, reason):
    old = old_raw(examples[3])
    if 'env_config' in change:
        old['env_config'].update(change['env_config'])
    else:
        old.update(change)
    with pytest.raises(demo_audit.AuditRejection) as error:
        audit(old, examples, migrate=True)
    assert error.value.reason == reason


def test_goe15_fine_valid_actions_are_not_rejected_by_old_coarse_mesh(examples):
    config, samples, membership, _, aliases = examples
    sample = aliases['goe15']
    env = BlockEnv([make_domain(sample)], environment_config(config))
    try:
        env.reset(seed=42)
        actions = solve(env, seed=42)
        assert isinstance(actions, list)
        raw = {'sample_id': sample['sample_id'], 'actions': actions, 'seed': 42,
            'source': 'rule_synthetic', 'env_config': asdict(env.config), 'environment_version': ENVIRONMENT_VERSION}
        record = audit(raw, examples)
        assert record['terminal_quality']['valid'] and record['fine_quality']['valid']
        assert record['terminal_quality']['cells'] == 3072
    finally:
        env.close()


def test_goe481_coarse_valid_actions_are_rejected_when_final_mesh_fails(examples):
    sample = examples[4]['goe481']
    raw = deepcopy(examples[3])
    raw['sample_id'] = sample['sample_id']
    raw['options']['sample_id'] = sample['sample_id']
    with pytest.raises(demo_audit.AuditRejection) as error:
        audit(raw, examples)
    assert error.value.reason == 'replay_failed'
    assert error.value.diagnostics['reason'] == 'invalid_generated_mesh'


def test_strict_identity_binds_both_environment_version_and_every_mesh_field(examples):
    record = audit(examples[3], examples)
    env = BlockEnv([make_domain(examples[1][record['sample_id']])], environment_config(examples[0]))
    try:
        validate_audited_episode(record, env)
        for key in asdict(MeshSpec()):
            changed = deepcopy(record)
            changed['mesh_spec'][key] += 1
            assert episode_id_for(changed) != record['episode_id']
        old = deepcopy(record); old['environment_version'] = 'coarse_mesh_v1';old['episode_id'] = episode_id_for(old)
        with pytest.raises(ValueError, match='environment version'):
            validate_audited_episode(old, env)
    finally:
        env.close()
