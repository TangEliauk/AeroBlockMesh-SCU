"""Offline review uses real train/val replay; human verdicts here are fixtures only."""
from argparse import Namespace
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile

import pytest

from smartcfd.core.types import ENVIRONMENT_VERSION, EnvConfig
from smartcfd.pipeline.audit_shared import canonical_hash, canonical_json, file_sha256, package_id_for
from smartcfd.pipeline.config import environment_config, read_config, write_json
from smartcfd.pipeline.data import data_version, load_samples, repository_root
from smartcfd.pipeline.demo_audit import _audit_episode
from smartcfd.pipeline import review


@pytest.fixture(scope='module')
def workspace():
    base = repository_root() / '.runtime/offline_review_test_fixtures'
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base, prefix='review_') as value:
        folder = Path(value).resolve()
        assert folder.is_relative_to(repository_root())
        yield folder
        assert folder.is_relative_to(repository_root())


@pytest.fixture(scope='module')
def audited_source(workspace):
    from smartcfd.baselines.rule_partition import solve
    from smartcfd.geometry.domain import make_domain
    from smartcfd.rl.env import BlockEnv
    config = read_config()
    samples, membership, episodes = {}, {}, []
    for split in ('train', 'val'):
        sample = load_samples(split, limit=1)[0]
        samples[sample['sample_id']] = sample
        membership[sample['sample_id']] = split
        for variant in ((0, 1) if split == 'train' else (0,)):
            env = BlockEnv([make_domain(sample)], environment_config(config))
            env.reset(seed=42+variant)
            actions = solve(env, seed=42+variant)
            raw = {'sample_id': sample['sample_id'], 'actions': actions, 'seed': 42+variant,
                   'options': {'sample_id': sample['sample_id']}, 'source': 'rule_synthetic',
                   'env_config': asdict(env.config), 'environment_version': ENVIRONMENT_VERSION, 'variant': variant}
            episodes.append(_audit_episode(raw, config, samples, membership, data_version()))
    episodes.sort(key=lambda e: (e['split'], e['sample_id'], e['episode_id']))
    source = workspace / 'source'
    source.mkdir()
    review._write_jsonl(source / 'episodes.jsonl', episodes)
    manifest = {'schema': 'demo_package_v2', 'schema_version': 2, 'environment_version': ENVIRONMENT_VERSION, 'episodes_file': 'episodes.jsonl',
        'episodes_sha256': file_sha256(source / 'episodes.jsonl'), 'episode_count': len(episodes),
        'split_counts': {'train': 2, 'val': 1}, 'data': data_version(), 'config': config,
        'implementation': {'source_sha256': 'synthetic-test-provenance'}, 'inputs': []}
    manifest['package_id'] = package_id_for(manifest)
    write_json(source / 'package.json', manifest)
    result = review.build_review_package(source, episodes, config)
    return source, config, episodes, result


def _decision_document(source, statuses=None):
    manifest = json.loads((source / 'human_review/review_manifest.json').read_text(encoding='utf-8'))
    decisions = [{'episode_id': e['episode_id'], 'episode_digest': e['episode_digest'],
                  'status': (statuses or {}).get(i, 'pending'), 'reviewer': 'synthetic test reviewer',
                  'comment': 'Synthetic test verdict; not a user approval.'} for i, e in enumerate(manifest['episodes'])]
    return {'schema': 'review_decisions_v2', 'package_id': manifest['package_id'],
            'episodes_sha256': manifest['episodes_sha256'], 'review_id': manifest['review_id'], 'decisions': decisions}


def _import(source, config, workspace, name, statuses=None, document=None):
    path = workspace / f'{name}.json'
    write_json(path, document or _decision_document(source, statuses))
    return review.import_reviews(config, Namespace(package=str(source), decisions=str(path), output=str(workspace / name)))


def test_real_pending_page_and_physical_assets(audited_source):
    source, config, episodes, result = audited_source
    assert result['selected'] == 2 and result['human_approved_count'] == 0
    manifest, records = review.validate_demo_package(source, config)
    assert len(records) == 3
    page = (source / 'human_review/index.html').read_text(encoding='utf-8')
    assert 'fetch(' not in page and 'https://' not in page and 'http://' not in page
    assert 'application/json' in page and 'review_decisions_v2' in page
    assert '环境终态验收（最终规格）' in page and '环境粗网格' not in page
    for record in episodes:
        if record['split'] == 'train':
            png = source / 'human_review' / record['episode_id'] / 'mesh_near_wall.png'
            assert png.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    bundle, train, val = review.validate_training_package(source, config)
    assert not bundle['training_ready'] and train == [] and len(val) == 1
    assert manifest['package_id'] == bundle['source_package_id']


def test_quality_rank_sampling_balanced_and_not_seed_padding():
    examples = []
    for variant in (0, 1):
        for i in range(25):
            examples.append({'episode_id': f'{variant}_{i:02}', 'sample_id': f'geometry_{i}',
                'split': 'train', 'source': 'rule_synthetic', 'variant': variant, 'variant_verified': True,
                'auto_passed': True, 'fine_quality': {'q10_scaled_jacobian': i/25}})
    examples += deepcopy(examples[:5])
    selected, availability = review.select_review_episodes(examples)
    assert len(selected) == len({e['episode_id'] for e in selected}) == 20
    assert [sum(e['variant'] == v for e in selected) for v in (0, 1)] == [10, 10]
    assert selected[0]['fine_quality']['q10_scaled_jacobian'] == 0
    assert selected[9]['fine_quality']['q10_scaled_jacobian'] == 24/25
    assert availability['0']['available'] == 25
    unknown = deepcopy(examples[0]); unknown['source'] = 'external_unverified'; unknown['episode_id'] = 'unknown'
    assert review.select_review_episodes([unknown])[0] == []


def test_pending_and_rejected_are_not_training_data(audited_source, workspace):
    source, config, _, _ = audited_source
    output, report = _import(source, config, workspace, 'pending_reject', {0: 'rejected'})
    bundle, train, val = review.validate_training_package(output, config)
    assert train == [] and len(val) == 1 and not report['training_ready']
    assert bundle['counts']['review_statuses'] == {'rejected': 1, 'pending': 1}


def test_fixture_approval_retains_rule_identity_and_full_provenance(audited_source, workspace):
    source, config, episodes, _ = audited_source
    original_hash = file_sha256(source / 'episodes.jsonl')
    output, report = _import(source, config, workspace, 'fixture_approval', {0: 'approved', 1: 'needs_revision'})
    bundle, train, val = review.validate_training_package(output, config)
    assert len(train) == 1 and len(val) == 1 and report['training_ready']
    assert train[0]['source'] == 'rule_synthetic' and train[0]['human_reviewed'] is True
    assert train[0]['review']['reviewer'] == 'synthetic test reviewer'
    assert train[0]['review']['package_id'] == bundle['source_package_id']
    assert file_sha256(source / 'episodes.jsonl') == original_hash
    assert (output / 'imported_decisions.json').is_file()
    assert all(e['split'] == 'val' for e in val)


@pytest.mark.parametrize('field', ['package_id', 'episodes_sha256', 'review_id'])
def test_stale_decision_binding_is_rejected_before_allocating_output(audited_source, workspace, field):
    source, config, _, _ = audited_source
    document = _decision_document(source, {0: 'approved'}); document[field] = 'stale'
    with pytest.raises(ValueError, match='stale or different'):
        _import(source, config, workspace, 'stale_' + field, document=document)
    assert not (workspace / ('stale_' + field)).exists()


@pytest.mark.parametrize('problem', ['reviewer', 'digest', 'unknown', 'duplicate', 'status'])
def test_bad_human_decisions_are_rejected(audited_source, workspace, problem):
    source, config, _, _ = audited_source
    doc = _decision_document(source, {0: 'approved'})
    if problem == 'reviewer': doc['decisions'][0]['reviewer'] = ' '
    elif problem == 'digest': doc['decisions'][0]['episode_digest'] = 'changed'
    elif problem == 'unknown': doc['decisions'][0]['episode_id'] = 'not-an-episode'
    elif problem == 'duplicate': doc['decisions'].append(deepcopy(doc['decisions'][0]))
    elif problem == 'status': doc['decisions'][0]['status'] = 'auto_approved'
    with pytest.raises(ValueError):
        _import(source, config, workspace, 'bad_' + problem, document=doc)


def test_modified_episode_bytes_are_rejected(audited_source, workspace):
    source, config, _, _ = audited_source
    modified = workspace / 'modified_source'; shutil.copytree(source, modified)
    with (modified / 'episodes.jsonl').open('a', encoding='utf-8') as stream: stream.write('\n')
    with pytest.raises(ValueError, match='episodes hash'):
        review.validate_demo_package(modified, config)


def test_training_file_tamper_is_rejected(audited_source, workspace):
    source, config, _, _ = audited_source
    modified = workspace / 'modified_training'; shutil.copytree(source, modified)
    (modified / 'approved_train.jsonl').write_text('injected\n', encoding='utf-8')
    with pytest.raises(ValueError, match='file hash'):
        review.validate_training_package(modified, config)


def test_changed_environment_rejected_but_model_and_seed_may_change(audited_source):
    source, config, _, _ = audited_source
    changed = deepcopy(config); changed['training']['seed'] = 777; changed['model']['hidden_dim'] = 16
    assert review.validate_demo_package(source, changed)[0]['package_id']
    changed['env']['max_steps'] -= 1
    with pytest.raises(ValueError, match='configuration mismatch'):
        review.validate_demo_package(source, changed)


def test_resumed_import_preserves_previous_history(audited_source, workspace):
    source, config, _, _ = audited_source
    first, _ = _import(source, config, workspace, 'history_first', {0: 'approved'})
    document = _decision_document(source, {1: 'rejected'})
    document['decisions'] = document['decisions'][1:]
    path = workspace / 'history_next.json'; write_json(path, document)
    second, _ = review.import_reviews(config, Namespace(package=str(first), decisions=str(path), output=str(workspace / 'history_second')))
    bundle, train, _ = review.validate_training_package(second, config)
    history = json.loads((second / 'review_history.json').read_text(encoding='utf-8'))
    assert len(history['events']) == 3 and len(history['imports']) == 2
    assert len(train) == 1 and bundle['counts']['review_statuses'] == {'approved': 1, 'rejected': 1}


def test_empty_batch_produces_explicit_not_ready_page(workspace):
    source = workspace / 'empty_source'; source.mkdir()
    config = read_config(); review._write_jsonl(source / 'episodes.jsonl', [])
    manifest = {'schema': 'demo_package_v2', 'environment_version': ENVIRONMENT_VERSION, 'episodes_file': 'episodes.jsonl',
        'episodes_sha256': file_sha256(source / 'episodes.jsonl'), 'episode_count': 0,
        'split_counts': {}, 'data': data_version(), 'config': config, 'implementation': {}, 'inputs': []}
    manifest['package_id'] = package_id_for(manifest); write_json(source / 'package.json', manifest)
    result = review.build_review_package(source, [], config)
    bundle, train, val = review.validate_training_package(source, config)
    assert result['selected'] == 0 and not bundle['training_ready'] and train == val == []


def test_embedded_json_cannot_close_script_tag():
    html = review._render_html({'episodes': [], 'test': '</script><script>alert(1)</script>'})
    payload = html.split('<script id="review-data" type="application/json">', 1)[1].split('</script>', 1)[0]
    assert '<script>' not in payload and json.loads(payload)['test'].startswith('</script>')


def test_v1_verdicts_are_not_accepted_by_current_review_package(audited_source, workspace):
    source, config, _, _ = audited_source
    document = _decision_document(source, {0: 'approved'})
    document['schema'] = 'review_decisions_v1'
    with pytest.raises(ValueError, match='stale or different'):
        _import(source, config, workspace, 'old_environment_verdict', document=document)
    assert not (workspace / 'old_environment_verdict').exists()


def test_v1_package_schema_is_refused_even_with_recomputed_hash(audited_source, workspace):
    source, config, _, _ = audited_source
    modified = workspace / 'old_package_schema'
    shutil.copytree(source, modified)
    manifest = json.loads((modified/'package.json').read_text(encoding='utf-8'))
    manifest['schema'] = 'demo_package_v1'
    manifest['package_id'] = package_id_for(manifest)
    write_json(modified/'package.json', manifest)
    with pytest.raises(ValueError, match='schema or package_id'):
        review.validate_demo_package(modified, config)
