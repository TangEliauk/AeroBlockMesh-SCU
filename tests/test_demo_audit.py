"""Real replay auditing, immutable identities and strict-loader boundaries."""
from argparse import Namespace
from dataclasses import asdict
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType

import pytest

from smartcfd.baselines.rule_partition import solve
from smartcfd.core.types import ENVIRONMENT_VERSION, EnvConfig, Failure
from smartcfd.pipeline import demo_audit
from smartcfd.pipeline.audit_shared import canonical_hash, episode_id_for, load_jsonl, package_id_for
from smartcfd.pipeline.config import environment_config, read_config
from smartcfd.pipeline.data import load_domains, repository_root
from smartcfd.rl.env import BlockEnv
from smartcfd.rl.imitation.dataset import load_replayed_demonstrations, validate_audited_episode


@pytest.fixture
def audit_workspace(monkeypatch):
    root = repository_root()
    base = root / '.runtime/demo_audit_tests'
    base.mkdir(parents=True, exist_ok=True)
    review_stub = ModuleType('smartcfd.pipeline.review')
    review_stub.build_review_package = lambda run, episodes, config: {'selected_count': 0}
    monkeypatch.setitem(sys.modules, 'smartcfd.pipeline.review', review_stub)
    with tempfile.TemporaryDirectory(prefix='case_', dir=base) as temporary:
        path = Path(temporary).resolve()
        assert path.is_relative_to(root.resolve())
        yield path
        assert path.is_relative_to(root.resolve())


@pytest.fixture(scope='module')
def raw_examples():
    config = read_config('configs/smoke.yaml')
    records, domains = {}, {}
    for split in ('train', 'val'):
        domain = load_domains(split, limit=1)[0]
        domains[split] = domain
        env = BlockEnv([domain], environment_config(config))
        env.reset(seed=42, options={'sample_id': domain.sample_id})
        actions = solve(env, seed=42)
        assert isinstance(actions, list)
        records[split] = {'sample_id': domain.sample_id, 'actions': actions, 'seed': 42,
                         'options': {'sample_id': domain.sample_id}, 'source': 'rule_synthetic',
                         'env_config': asdict(env.config), 'environment_version': ENVIRONMENT_VERSION, 'human_reviewed': True}
        env.close()
    return config, records, domains


def run_audit(workspace, config, records, name='run'):
    source = workspace / f'{name}.jsonl'
    with source.open('wb') as output:
        for record in records:
            output.write((record if isinstance(record, bytes) else json.dumps(record).encode('utf-8')) + b'\n')
    return demo_audit.audit_demos(config, Namespace(input=[str(source)], output=str(workspace / name)))


def test_mixed_audit_replays_real_meshes_deduplicates_seed_and_preserves_all_failures(audit_workspace, raw_examples):
    config, raw, domains = raw_examples
    first = copy.deepcopy(raw['train'])
    duplicate = {**copy.deepcopy(first), 'seed': 44}
    bad_env = {**copy.deepcopy(first), 'env_config': {**first['env_config'], 'gamma': .5}}
    bad_options = {**copy.deepcopy(first), 'options': {'sample_id': raw['val']['sample_id']}}
    bad_version = {**copy.deepcopy(first), 'action_version': 'future_action_schema'}
    held_out_id = json.loads((repository_root() / 'data/splits/uiuc_v1/test.json').read_text(encoding='utf-8'))[0]
    held_out = {**copy.deepcopy(first), 'sample_id': held_out_id, 'options': {'sample_id': held_out_id}}
    forged_reward = {**copy.deepcopy(first), 'reward': 999999, 'observations': 'forged'}
    run, report = run_audit(audit_workspace, config,
        [first, duplicate, raw['val'], bad_env, bad_options, bad_version, held_out, forged_reward, b'{invalid'])
    assert report['input_lines'] == 9
    assert report['accepted_unique'] == 2 and report['duplicates'] == 2 and report['failed'] == 5
    assert report['human_approved_count'] == 0
    assert report['unique_geometry_counts'] == {'train': 1, 'val': 1}
    assert report['verified_topology_category_count'] == 1
    assert report['variant_counts'] == {'train': {'0': 1}, 'val': {'0': 1}}
    records = load_jsonl(run / 'audit_records.jsonl')
    assert len(records) == 9
    assert {r.get('reason') for r in records} >= {'env_config_mismatch', 'sample_options_conflict',
        'action_version_mismatch', 'test_split_forbidden', 'invalid_record'}
    episodes = load_jsonl(run / 'episodes.jsonl')
    assert {e['split'] for e in episodes} == {'train', 'val'}
    for episode in episodes:
        assert episode['auto_passed'] is True and episode['human_reviewed'] is False
        assert episode['source'] == 'rule_synthetic' and episode['origin']['status'] == 'legacy_unknown'
        assert episode['coarse_quality']['valid'] and episode['fine_quality']['valid']
        assert episode['su2_stats']['cells'] == episode['fine_quality']['cells']
        assert episode['episode_id'] == episode_id_for(episode)
        assert episode['variant'] == 0 and episode['variant_verified'] is True
        assert episode['replayed_return'] < 999999
    package = json.loads((run / 'package.json').read_text(encoding='utf-8'))
    assert package['package_id'] == package_id_for(package)
    assert package['split_counts'] == {'train': 1, 'val': 1}


def test_final_mesh_failure_is_a_terminal_failure(audit_workspace, raw_examples, monkeypatch):
    from smartcfd.meshing import generator
    config, raw, _ = raw_examples
    original = generator.generate_mesh
    def fail_fine(domain, graph, spec):
        if spec.intervals == config['mesh']['intervals']:
            return Failure('fixture_fine_mesh_failure', diagnostics={'cell': [0, 1], 'q10': float('nan')})
        return original(domain, graph, spec)
    monkeypatch.setattr(generator, 'generate_mesh', fail_fine)
    run, report = run_audit(audit_workspace, config, [raw['train']])
    assert report['accepted_unique'] == 0 and report['failed'] == 1
    failure = load_jsonl(run / 'audit_records.jsonl')[0]
    assert failure['reason'] == 'replay_failed'
    assert failure['diagnostics']['diagnostics']['cell'] == [0, 1]
    assert failure['diagnostics']['diagnostics']['q10'] == 'nonfinite:nan'
    assert (run / 'episodes.jsonl').read_text(encoding='utf-8') == ''


def test_strict_loader_checks_provenance_and_cannot_disable_terminal_success(audit_workspace, raw_examples):
    config, raw, domains = raw_examples
    run, _ = run_audit(audit_workspace, config, [raw['train']])
    record = load_jsonl(run / 'episodes.jsonl')[0]
    factory = lambda: BlockEnv([domains['train']], environment_config(config))
    records = load_replayed_demonstrations(run / 'episodes.jsonl', factory, strict=True,
                                         expected_context={'mesh_spec': record['mesh_spec']})
    assert len(records) == len(record['actions']) and records[-1].terminated
    for field, wrong in [('episode_id', 'forged'), ('action_version', 'wrong'),
                         ('data', {'version': 'wrong'}), ('split', 'val')]:
        altered = {**copy.deepcopy(record), field: wrong}
        with pytest.raises(ValueError):
            validate_audited_episode(altered, factory())
    altered = copy.deepcopy(record)
    altered['actions'] = altered['actions'][:-1]
    altered['episode_id'] = episode_id_for(altered)
    broken = audit_workspace / 'incomplete.jsonl'
    broken.write_text(json.dumps(altered) + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='successful terminal'):
        load_replayed_demonstrations(broken, factory, strict=True, require_success=False)


def test_declared_metadata_variant_and_source_are_verified(audit_workspace, raw_examples):
    config, raw, _ = raw_examples
    first, _ = run_audit(audit_workspace, config, [raw['train']], 'first')
    canonical = load_jsonl(first / 'episodes.jsonl')[0]
    changed_data = {**copy.deepcopy(canonical), 'data': {'version': 'forged'}}
    changed_variant = {**copy.deepcopy(raw['train']), 'variant': 1}
    unverified_expert = {**copy.deepcopy(raw['train']), 'source': 'human_expert'}
    run, report = run_audit(audit_workspace, config, [canonical, changed_data, changed_variant, unverified_expert], 'second')
    assert report['accepted_unique'] == 1 and report['duplicates'] == 1 and report['failed'] == 2
    episode = load_jsonl(run / 'episodes.jsonl')[0]
    # Re-auditing cannot turn originally unknown historical versions into known ones.
    assert episode['origin']['status'] == 'legacy_unknown'
    assert episode['source'] == 'rule_synthetic' and episode['human_reviewed'] is False
    failed = [r['reason'] for r in load_jsonl(run / 'audit_records.jsonl') if r['status'] == 'failed']
    assert set(failed) == {'data_mismatch', 'variant_mismatch'}


def test_invalid_utf8_or_boolean_actions_are_per_line_failures(audit_workspace, raw_examples):
    config, raw, _ = raw_examples
    invalid_action = {**copy.deepcopy(raw['train']), 'actions': [True]}
    run, result = run_audit(audit_workspace, config, [b'\xff', invalid_action, raw['train']])
    assert result['failed'] == 2 and result['accepted_unique'] == 1
    assert len(load_jsonl(run / 'audit_records.jsonl')) == 3


def test_episode_identity_ignores_seed_but_binds_terminal_mesh_contract(audit_workspace, raw_examples):
    config, raw, _ = raw_examples
    run, _ = run_audit(audit_workspace, config, [raw['train']])
    record = load_jsonl(run / 'episodes.jsonl')[0]
    another = {**record, 'seed': 999, 'human_reviewed': True}
    assert episode_id_for(record) == episode_id_for(another)
    another['mesh_spec'] = {**record['mesh_spec'], 'intervals': 64}
    assert episode_id_for(record) != episode_id_for(another)
    another['actions'] = record['actions'] + [1]
    assert episode_id_for(record) != episode_id_for(another)


def test_legacy_loader_checks_declared_context_and_sample_options(audit_workspace, raw_examples):
    config, raw, domains = raw_examples
    factory = lambda: BlockEnv([domains['train']], environment_config(config))
    for record in ({**raw['train'], 'env_config': {**raw['train']['env_config'], 'gamma': .5}},
                   {**raw['train'], 'options': {'sample_id': raw['val']['sample_id']}}):
        path = audit_workspace / 'legacy_bad.jsonl'
        path.write_text(json.dumps(record) + '\n', encoding='utf-8')
        with pytest.raises(ValueError, match='env_config|conflict'):
            load_replayed_demonstrations(path, factory)
    without_context = {key: value for key, value in raw['train'].items() if key != 'env_config'}
    path.write_text(json.dumps(without_context) + '\n', encoding='utf-8')
    assert len(load_replayed_demonstrations(path, factory)) == len(without_context['actions'])


def test_unexpected_per_episode_error_is_distinct_and_batch_continues(audit_workspace, raw_examples, monkeypatch):
    config, raw, _ = raw_examples
    original = demo_audit._audit_episode
    calls = 0
    def first_error(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError('fixture unexpected error')
        return original(*args, **kwargs)
    monkeypatch.setattr(demo_audit, '_audit_episode', first_error)
    run, report = run_audit(audit_workspace, config, [raw['train'], raw['val']])
    assert report['failed'] == 1 and report['accepted_unique'] == 1
    assert report['failure_reasons'] == {'audit_internal_error': 1}
    result = load_jsonl(run / 'audit_records.jsonl')[0]
    assert 'RuntimeError' in result['detail']


def test_explicit_unknown_schema_or_data_versions_are_not_treated_as_legacy(audit_workspace, raw_examples):
    config, raw, _ = raw_examples
    records = [{**raw['train'], key: value} for key, value in
               (('schema', 'audited_episode_v99'), ('schema_version', 2), ('data_version', 'future_data'))]
    run, report = run_audit(audit_workspace, config, records)
    assert report['failed'] == 3 and report['accepted_unique'] == 0
    assert set(report['failure_reasons']) == {'episode_schema_mismatch', 'episode_schema_version_mismatch',
                                              'data_version_mismatch'}
