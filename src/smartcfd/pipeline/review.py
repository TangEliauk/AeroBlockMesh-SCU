"""Offline human review, immutable decisions and strictly verified training bundles."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np

from smartcfd.core.types import ENVIRONMENT_VERSION, EnvConfig, Failure, MeshSpec
from smartcfd.geometry.domain import make_domain
from smartcfd.partition.actions import ACTION_SCHEMA
from smartcfd.rl.state import OBSERVATION_SCHEMA
from smartcfd.pipeline.audit_shared import (
    EPISODE_SCHEMA, PACKAGE_SCHEMA, canonical_hash, canonical_json,
    episode_id_for, file_sha256, load_jsonl, package_id_for,
)
from smartcfd.pipeline.config import complete_run, environment_context, new_run, write_json
from smartcfd.pipeline.data import data_version, load_samples, repository_root

REVIEW_SCHEMA = 'offline_review_v2'
DECISION_SCHEMA = 'review_decisions_v2'
TRAINING_SCHEMA = 'reviewed_demo_package_v2'
STATUSES = ('pending', 'approved', 'rejected', 'needs_revision')


def _path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repository_root() / path).resolve()


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'Nonfinite JSON: {value}')))


def _domain_settings(config):
    return {key: float(value) for key, value in {'farfield': 500., 'stretch': 18., **config.get('domain', {})}.items()}


def _context(config):
    return environment_context(config)


def _check_package(manifest, episodes, actual_sha256, config=None):
    if manifest.get('schema') != PACKAGE_SCHEMA or manifest.get('package_id') != package_id_for(manifest):
        raise ValueError('Demo package schema or package_id mismatch')
    if manifest.get('environment_version') != ENVIRONMENT_VERSION:
        raise ValueError('Demo package environment version mismatch; re-audit legacy actions explicitly')
    if manifest.get('episodes_sha256') != actual_sha256:
        raise ValueError('Demo package episodes hash mismatch')
    if manifest.get('data') != data_version():
        raise ValueError('Demo package frozen data version mismatch')
    if len(episodes) != manifest.get('episode_count'):
        raise ValueError('Demo package episode count mismatch')
    expected = _context(manifest['config'])
    if config is not None and canonical_hash(_context(config)) != canonical_hash(expected):
        raise ValueError('Demo package environment/domain/mesh configuration mismatch')
    samples = {s['sample_id']: (split, s) for split in ('train', 'val') for s in load_samples(split)}
    identities = set()
    for record in episodes:
        if not isinstance(record, dict):
            raise ValueError('Demo package episode must be an object')
        canonical_hash(record)  # Reject nonfinite values anywhere, including quality metadata.
        if record.get('schema') != EPISODE_SCHEMA or record.get('auto_passed') is not True:
            raise ValueError('Demo package contains an unaudited episode')
        if record.get('human_reviewed') is not False:
            raise ValueError('Canonical source episodes must not contain human approval')
        sample_id = record.get('sample_id')
        if sample_id not in samples or record.get('split') != samples[sample_id][0]:
            raise ValueError('Demo package contains a forbidden or inconsistent split')
        sample = samples[sample_id][1]
        domain = make_domain(sample, **expected['domain_config'])
        if record.get('geometry_sha256') != canonical_hash(domain.polygon):
            raise ValueError('Demo package geometry hash mismatch')
        if record.get('data') != manifest['data']:
            raise ValueError('Episode data version mismatch')
        for key, value in expected.items():
            if canonical_hash(record.get(key)) != canonical_hash(value):
                raise ValueError(f'Episode {key} mismatch')
        if (record.get('observation_version') != OBSERVATION_SCHEMA
                or record.get('action_version') != ACTION_SCHEMA
                or record.get('mapping_version') != domain.metadata['mapping_version']):
            raise ValueError('Episode observation/action/mapping version mismatch')
        actions = record.get('actions')
        if not isinstance(actions, list) or not actions or any(
                isinstance(a, bool) or not isinstance(a, int) or not 0 <= a < 1024 for a in actions):
            raise ValueError('Episode actions must be nonempty legal integer IDs')
        if record.get('options') != {'sample_id': sample_id}:
            raise ValueError('Episode reset options mismatch')
        eid = record.get('episode_id')
        if eid != episode_id_for(record) or eid in identities:
            raise ValueError('Episode identity mismatch or duplicate')
        identities.add(eid)
        if record.get('source') not in ('rule_synthetic', 'external_unverified'):
            raise ValueError('Episode source cannot be relabeled as an expert')
        if record.get('terminal_mesh_spec') != expected['mesh_spec']:
            raise ValueError('Episode terminal mesh spec mismatch')
        for key in ('terminal_quality', 'fine_quality'):
            quality = record.get(key, {})
            if quality.get('valid') is not True or not np.isfinite(quality.get('min_scaled_jacobian', np.nan)) or quality['min_scaled_jacobian'] <= 1e-10:
                raise ValueError('Automatically passed episode contains invalid mesh quality')
    if dict(Counter(e['split'] for e in episodes)) != manifest.get('split_counts'):
        raise ValueError('Demo package split counts mismatch')


def validate_demo_package(path, config=None):
    """Validate frozen bytes, identities and current context; never train here."""
    folder = _path(path)
    if folder.is_file():
        folder = folder.parent
    manifest = _json(folder / 'package.json')
    name = manifest.get('episodes_file')
    if name != 'episodes.jsonl':
        raise ValueError('Unsupported demo package episodes filename')
    episodes = load_jsonl(folder / name)
    _check_package(manifest, episodes, file_sha256(folder / name), config)
    return manifest, episodes


def select_review_episodes(episodes, quota=10):
    """Equal-rank quality sampling of verified rule variants, without padding."""
    selected, availability = [], {}
    for variant in (0, 1):
        pool = {e['episode_id']: e for e in episodes if e.get('split') == 'train'
                and e.get('auto_passed') is True and e.get('source') == 'rule_synthetic'
                and e.get('variant_verified') is True
                and type(e.get('variant')) is int and e['variant'] == variant}
        ordered = sorted(pool.values(), key=lambda e: (e['fine_quality']['q10_scaled_jacobian'], e['sample_id'], e['episode_id']))
        count = min(quota, len(ordered))
        indices = np.linspace(0, len(ordered)-1, count).round().astype(int).tolist() if count else []
        selected.extend(ordered[i] for i in indices)
        availability[str(variant)] = {'available': len(ordered), 'selected': count, 'requested': quota}
    return selected, availability


def _episode_assets(record, folder):
    from smartcfd.pipeline.workflow import draw_mesh, draw_topology
    from smartcfd.rl.env import BlockEnv
    from smartcfd.meshing.generator import generate_mesh
    sample = next(s for s in load_samples(record['split']) if s['sample_id'] == record['sample_id'])
    domain = make_domain(sample, **record['domain_config'])
    env = BlockEnv([domain], EnvConfig(**record['env_config']))
    env.reset(seed=record['seed'], options=record['options'])
    folder.mkdir()
    frames = []
    try:
        draw_topology(env.graph, folder / 'step_000.png', 'Initial C-parameter topology')
        frames.append({'step': 0, 'image': 'step_000.png', 'description': '初始拓扑；蓝色区域为保护带。'})
        for step, action in enumerate(record['actions'], 1):
            if env.terminated or not env.action_mask[action]:
                raise ValueError('Reviewed episode no longer replays legally')
            description = env.describe_action(action)
            env.step(action)
            name = f'step_{step:03d}.png'
            draw_topology(env.graph, folder / name, f'C-parameter topology after action {step}')
            label = '插入连弦' if description['kind'] == 'insert_chord' else '插入边中点'
            frames.append({'step': step, 'image': name, 'action': action,
                'description': f"{label}：{description['source_uv']} 至 {description['target_uv']}"})
        if not env.success:
            raise ValueError('Reviewed episode no longer reaches a successful terminal')
        mesh = generate_mesh(domain, env.graph, MeshSpec(**record['mesh_spec']))
        if isinstance(mesh, Failure) or not mesh.quality.get('valid', False):
            raise ValueError('Reviewed episode fine mesh is invalid')
        draw_mesh(mesh, folder)
        return frames
    finally:
        env.close()


def _review_id(manifest):
    return canonical_hash({k: v for k, v in manifest.items() if k != 'review_id'})


def _validate_review_manifest(review, manifest, episodes, assets_root=None):
    if (review.get('schema') != REVIEW_SCHEMA or review.get('review_id') != _review_id(review)
            or review.get('environment_version') != ENVIRONMENT_VERSION
            or review.get('package_id') != manifest['package_id']
            or review.get('episodes_sha256') != manifest['episodes_sha256']):
        raise ValueError('Review manifest is stale or has been modified')
    records = {e['episode_id']: e for e in episodes}
    seen = set()
    for item in review['episodes']:
        eid = item['episode_id']
        if eid in seen or eid not in records or item['episode_digest'] != canonical_hash(records[eid]):
            raise ValueError('Review episode digest mismatch')
        if records[eid]['split'] != 'train' or records[eid]['auto_passed'] is not True:
            raise ValueError('Only automatically passed training episodes are reviewable')
        seen.add(eid)
    if assets_root is not None:
        base = Path(assets_root).resolve()
        for name, digest in review['assets_sha256'].items():
            path = (base / name).resolve()
            if not path.is_relative_to(base) or not path.is_file() or file_sha256(path) != digest:
                raise ValueError('Review image asset missing or modified')


def _write_jsonl(path, records):
    with Path(path).open('w', encoding='utf-8', newline='\n') as stream:
        for record in records:
            stream.write(canonical_json(record) + '\n')


def _write_training_bundle(output, manifest, episodes, review, history, decisions, source_path, source_bytes=None):
    statuses = {item['episode_id']: item for item in decisions}
    approved = []
    for episode in episodes:
        decision = statuses.get(episode['episode_id'])
        if episode['split'] == 'train' and decision and decision['status'] == 'approved':
            reviewed = deepcopy(episode)
            reviewed['human_reviewed'] = True
            reviewed['review'] = deepcopy(decision)
            approved.append(reviewed)
    val = [e for e in episodes if e['split'] == 'val']
    write_json(output / 'source_package.json', manifest)
    if source_bytes is None:
        _write_jsonl(output / 'source_episodes.jsonl', episodes)
    else:
        (output / 'source_episodes.jsonl').write_bytes(source_bytes)
    if file_sha256(output / 'source_episodes.jsonl') != manifest['episodes_sha256']:
        raise ValueError('Copied source episodes do not preserve original audited bytes')
    write_json(output / 'source_review_manifest.json', review)
    write_json(output / 'review_history.json', history)
    write_json(output / 'review_decisions.json', decisions)
    _write_jsonl(output / 'approved_train.jsonl', approved)
    _write_jsonl(output / 'auto_passed_val.jsonl', val)
    files = ('source_package.json', 'source_episodes.jsonl', 'source_review_manifest.json',
             'review_history.json', 'review_decisions.json', 'approved_train.jsonl', 'auto_passed_val.jsonl')
    result = {'schema': TRAINING_SCHEMA, 'schema_version': 2, 'environment_version': ENVIRONMENT_VERSION,
        'source_package_id': manifest['package_id'],
        'source_package_path': str(Path(source_path).resolve()), 'source_episodes_sha256': manifest['episodes_sha256'],
        'review_id': review['review_id'], 'data': manifest['data'], 'config': manifest['config'],
        'files_sha256': {name: file_sha256(output / name) for name in files},
        'counts': {'approved_train': len(approved), 'auto_passed_val': len(val),
                   'selected_for_review': len(review['episodes']),
                   'review_statuses': dict(Counter(d['status'] for d in decisions))},
        'training_ready': bool(approved and val),
        'source_policy': 'Human review does not change rule_synthetic into engineer-original expert data.'}
    result['training_package_id'] = canonical_hash(result)
    write_json(output / 'training_package.json', result)
    return result


def build_review_package(run, episodes, config):
    """Create a pending offline page and an explicitly not-ready training bundle."""
    run = _path(run)
    manifest, canonical_episodes = validate_demo_package(run, config)
    if canonical_hash(episodes) != canonical_hash(canonical_episodes):
        raise ValueError('Review records disagree with saved source package')
    selected, availability = select_review_episodes(episodes)
    folder = run / 'human_review'
    folder.mkdir(exist_ok=False)
    items = []
    aliases = {s['sample_id']: s.get('aliases', []) for s in load_samples('train')}
    for episode in selected:
        eid = episode['episode_id']
        frames = _episode_assets(episode, folder / eid)
        items.append({'episode_id': eid, 'episode_digest': canonical_hash(episode),
            'sample_id': episode['sample_id'], 'aliases': aliases.get(episode['sample_id'], []), 'variant': episode['variant'],
            'source': episode['source'], 'review_status': 'pending', 'frames': frames,
            'near_wall_image': f'{eid}/mesh_near_wall.png', 'overview_image': f'{eid}/mesh_overview.png',
            'terminal_quality': episode['terminal_quality'], 'fine_quality': episode['fine_quality'],
            'environment_version': ENVIRONMENT_VERSION,
            'env_config': episode['env_config'], 'mesh_spec': episode['mesh_spec']})
    review = {'schema': REVIEW_SCHEMA, 'environment_version': ENVIRONMENT_VERSION, 'package_id': manifest['package_id'],
              'episodes_sha256': manifest['episodes_sha256'], 'selection': availability, 'episodes': items,
              'assets_sha256': {p.relative_to(folder).as_posix(): file_sha256(p)
                               for p in sorted(folder.rglob('*.png'))}}
    review['review_id'] = _review_id(review)
    write_json(folder / 'review_manifest.json', review)
    (folder / 'index.html').write_text(_render_html(review), encoding='utf-8')
    pending = [{'episode_id': e['episode_id'], 'episode_digest': e['episode_digest'],
                'status': 'pending', 'reviewer': '', 'comment': ''} for e in items]
    history = {'schema': 'review_history_v2', 'source_package_id': manifest['package_id'], 'events': []}
    training = _write_training_bundle(run, manifest, episodes, review, history, pending, run,
                                      (run / 'episodes.jsonl').read_bytes())
    return {'selected': len(items), 'availability': availability, 'human_approved_count': 0,
            'review_page': str(folder / 'index.html'), 'training_ready': training['training_ready']}


def validate_training_package(path, config=None):
    """Return checked records, including empty pending bundles; caller gates readiness."""
    folder = _path(path)
    if folder.is_file():
        folder = folder.parent
    bundle = _json(folder / 'training_package.json')
    if bundle.get('schema') != TRAINING_SCHEMA:
        raise ValueError('Training package schema mismatch')
    if bundle.get('environment_version') != ENVIRONMENT_VERSION:
        raise ValueError('Training package environment version mismatch')
    identity = {k: v for k, v in bundle.items() if k != 'training_package_id'}
    if bundle.get('training_package_id') != canonical_hash(identity):
        raise ValueError('Training package identity mismatch')
    required = {'source_package.json', 'source_episodes.jsonl', 'source_review_manifest.json',
                'review_history.json', 'review_decisions.json', 'approved_train.jsonl', 'auto_passed_val.jsonl'}
    if set(bundle.get('files_sha256', {})) != required:
        raise ValueError('Training package file list mismatch')
    for name, digest in bundle['files_sha256'].items():
        if not (folder / name).is_file() or file_sha256(folder / name) != digest:
            raise ValueError(f'Training package file hash mismatch: {name}')
    source = _json(folder / 'source_package.json')
    episodes = load_jsonl(folder / 'source_episodes.jsonl')
    _check_package(source, episodes, file_sha256(folder / 'source_episodes.jsonl'), config)
    if (bundle['source_package_id'] != source['package_id'] or bundle['data'] != source['data']
            or bundle['config'] != source['config'] or bundle['source_episodes_sha256'] != source['episodes_sha256']):
        raise ValueError('Training package source provenance mismatch')
    review = _json(folder / 'source_review_manifest.json')
    _validate_review_manifest(review, source, episodes)
    if bundle['review_id'] != review['review_id']:
        raise ValueError('Training package review identity mismatch')
    decisions = _json(folder / 'review_decisions.json')
    _validate_decisions(decisions, review, source, imported=True)
    records = {e['episode_id']: e for e in episodes}
    expected_approved = {d['episode_id']: d for d in decisions if d['status'] == 'approved'}
    train, val = load_jsonl(folder / 'approved_train.jsonl'), load_jsonl(folder / 'auto_passed_val.jsonl')
    if len(train) != len(expected_approved) or {e['episode_id'] for e in train} != set(expected_approved):
        raise ValueError('Approved training list disagrees with review decisions')
    for episode in train:
        original = records[episode['episode_id']]
        stripped = {k: v for k, v in episode.items() if k not in ('review', 'human_reviewed')}
        stripped['human_reviewed'] = False
        if (canonical_hash(stripped) != canonical_hash(original) or episode['human_reviewed'] is not True
                or episode.get('review') != expected_approved[episode['episode_id']] or original['split'] != 'train'):
            raise ValueError('Approved training episode changed from audited source')
    if canonical_hash(val) != canonical_hash([e for e in episodes if e['split'] == 'val']):
        raise ValueError('Validation episodes changed or leaked across splits')
    counts = {'approved_train': len(train), 'auto_passed_val': len(val), 'selected_for_review': len(review['episodes']),
              'review_statuses': dict(Counter(d['status'] for d in decisions))}
    if bundle['counts'] != counts or bundle['training_ready'] is not bool(train and val):
        raise ValueError('Training readiness or counts mismatch')
    history = _json(folder / 'review_history.json')
    if history.get('schema') != 'review_history_v2' or history.get('source_package_id') != source['package_id']:
        raise ValueError('Review history source mismatch')
    for decision in decisions:
        if decision['status'] != 'pending' and decision not in history['events']:
            raise ValueError('Human review decision missing from preserved history')
    return bundle, train, val


def _validate_decisions(decisions, review, manifest, imported=False):
    if not isinstance(decisions, list):
        raise ValueError('Review decisions must be a list')
    allowed = {e['episode_id']: e for e in review['episodes']}
    seen = set()
    for decision in decisions:
        eid = decision.get('episode_id')
        if eid not in allowed or eid in seen or decision.get('episode_digest') != allowed[eid]['episode_digest']:
            raise ValueError('Review decision is stale, duplicated or unknown')
        if decision.get('status') not in STATUSES:
            raise ValueError('Unknown human review status')
        if not isinstance(decision.get('reviewer'), str) or not isinstance(decision.get('comment'), str):
            raise ValueError('Review reviewer and comment must be strings')
        if decision['status'] != 'pending' and not decision['reviewer'].strip():
            raise ValueError('A human review verdict requires a reviewer name')
        if imported and decision['status'] != 'pending':
            if (decision.get('package_id') != manifest['package_id'] or decision.get('review_id') != review['review_id']
                    or not decision.get('imported_at_utc')):
                raise ValueError('Human decision import provenance missing')
        seen.add(eid)


def import_reviews(config, args):
    """Import user-provided verdicts into a new run; never alter the audit source."""
    source_folder = _path(args.package)
    if source_folder.is_file():
        source_folder = source_folder.parent
    previous = []
    if (source_folder / 'package.json').is_file():
        manifest, episodes = validate_demo_package(source_folder, config)
        review = _json(source_folder / 'human_review/review_manifest.json')
        _validate_review_manifest(review, manifest, episodes, source_folder / 'human_review')
        history = {'schema': 'review_history_v2', 'source_package_id': manifest['package_id'], 'events': []}
        provenance = source_folder
        source_bytes = (source_folder / 'episodes.jsonl').read_bytes()
    else:
        bundle, _, _ = validate_training_package(source_folder, config)
        manifest, episodes = _json(source_folder / 'source_package.json'), load_jsonl(source_folder / 'source_episodes.jsonl')
        review, history = _json(source_folder / 'source_review_manifest.json'), _json(source_folder / 'review_history.json')
        previous = _json(source_folder / 'review_decisions.json')
        provenance = Path(bundle['source_package_path'])
        source_bytes = (source_folder / 'source_episodes.jsonl').read_bytes()
    incoming_path = _path(args.decisions)
    incoming = _json(incoming_path)
    if (incoming.get('schema') != DECISION_SCHEMA or incoming.get('package_id') != manifest['package_id']
            or incoming.get('episodes_sha256') != manifest['episodes_sha256']
            or incoming.get('review_id') != review['review_id']):
        raise ValueError('Imported decisions target a stale or different review package')
    incoming_decisions = incoming.get('decisions')
    _validate_decisions(incoming_decisions, review, manifest)
    current = {e['episode_id']: {'episode_id': e['episode_id'], 'episode_digest': e['episode_digest'],
               'status': 'pending', 'reviewer': '', 'comment': ''} for e in review['episodes']}
    current.update({d['episode_id']: d for d in previous})
    now = datetime.now(timezone.utc).isoformat()
    for entry in incoming_decisions:
        decision = {key: entry[key] for key in ('episode_id', 'episode_digest', 'status', 'reviewer', 'comment')}
        decision['reviewer'] = decision['reviewer'].strip()
        decision.update(package_id=manifest['package_id'], review_id=review['review_id'], imported_at_utc=now)
        current[decision['episode_id']] = decision
        history['events'].append(deepcopy(decision))
    output = new_run('import_reviews', config, args.output)
    try:
        shutil.copyfile(incoming_path, output / 'imported_decisions.json')
        history.setdefault('imports', []).append({'path': str(incoming_path), 'sha256': file_sha256(incoming_path),
            'imported_at_utc': now, 'source_package_id': manifest['package_id']})
        result = _write_training_bundle(output, manifest, episodes, review, history, list(current.values()), provenance, source_bytes)
        validate_training_package(output, config)
        report = {'source_package_id': manifest['package_id'], 'training_package_id': result['training_package_id'],
            'counts': result['counts'], 'training_ready': result['training_ready'],
            'human_approved_count': result['counts']['approved_train'], 'test_data_used': False}
        write_json(output / 'metrics.json', report)
        complete_run(output, report)
        return output, report
    except BaseException as error:
        complete_run(output, {'error': f'{type(error).__name__}: {error}'}, status='failed')
        raise


def _render_html(review):
    payload = json.dumps(review, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    return _HTML.replace('__REVIEW_PAYLOAD__', payload)


_HTML = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>规则示范 · 离线审阅</title>
<style>
:root{font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;color:#233647;background:#f5f8fb;font-size:15px}*{box-sizing:border-box}body{margin:0}header{padding:22px 30px;background:#163d61;color:white}h1{font-size:25px;margin:0 0 8px;font-weight:600}header p{margin:0;max-width:72ch;line-height:1.7}.toolbar{display:flex;gap:16px;flex-wrap:wrap;align-items:end;padding:18px 30px;background:white;border-bottom:1px solid #c6d4e0}label{display:grid;gap:6px;font-size:13px}input,select,textarea,button{font:inherit}input,select,textarea{padding:9px;border:1px solid #9fb2c5;border-radius:3px;background:white;color:#233647}button{cursor:pointer;padding:10px 15px;border:1px solid #136fc2;border-radius:3px;background:#136fc2;color:white}button.secondary{background:white;color:#163d61}button:focus-visible,input:focus-visible,select:focus-visible,textarea:focus-visible{outline:3px solid #e4ad42;outline-offset:2px}.workspace{display:grid;grid-template-columns:285px minmax(0,1fr);max-width:1600px;margin:auto}aside{padding:24px 18px}aside select,aside input{width:100%;margin:0 0 14px}#list{display:grid;gap:6px}.episode-button{border:1px solid transparent;text-align:left;color:#233647;background:transparent;padding:12px 10px;font-size:13px}.episode-button.current{border-color:#136fc2;background:#e7f0f8}.episode-button small{display:block;margin-top:6px;color:#526c81}main{background:white;padding:26px 32px;min-width:0;min-height:80vh}h2{margin:0 0 10px;font-size:22px}h3{font-size:17px;margin-top:28px}.muted{color:#526c81;line-height:1.65}.notice{padding:14px 18px;background:#f1f6fb;border-left:4px solid #136fc2;line-height:1.7}.warning{background:#fff8e7;border-color:#a46916}.verdict{display:grid;grid-template-columns:170px 1fr;gap:16px;padding:20px 0;border-bottom:1px solid #c6d4e0}.verdict textarea{width:100%;min-height:80px}.quality{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px 4px;border-bottom:1px solid #dfE7ee;text-align:left;overflow-wrap:anywhere}th{color:#526c81;font-weight:500}figure{margin:12px 0 0}figure img{display:block;width:100%;height:auto;background:white}figcaption{font-size:13px;color:#526c81;margin:8px 0;line-height:1.6}.image-tabs{display:flex;gap:10px}.frames{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}details{margin:18px 0}summary{cursor:pointer;color:#163d61;padding:8px 0}pre{overflow:auto;max-height:250px;background:#f5f8fb;padding:14px;font-size:12px}#message{margin-left:auto;color:#526c81;font-size:13px;max-width:50ch;white-space:pre-line}.status-pending{color:#526c81}.status-approved{color:#267044}.status-rejected{color:#ba3b47}.status-needs_revision{color:#a46916}footer{padding:18px 30px;font-size:12px;color:#526c81}#stats{font-size:13px;margin:0 0 16px;line-height:1.7}@media(max-width:900px){.workspace{grid-template-columns:1fr}aside{padding:16px}#list{grid-template-columns:repeat(2,minmax(0,1fr));max-height:230px;overflow:auto}main{padding:22px 18px}.quality,.frames{grid-template-columns:1fr}.toolbar{padding:16px;gap:12px}.verdict{grid-template-columns:1fr}}@media print{.toolbar,aside,.verdict{display:none}.workspace{display:block}main{padding:0}header{color:#233647;background:white}.frames{grid-template-columns:1fr 1fr}}
</style>
<header><h1>规则示范 · 离线审阅</h1><p>逐步核对分块动作与实际网格。所有示范均来自规则构造，人工批准只表示已复核；不会改变数据来源。</p></header>
<div class="toolbar"><label>审阅人<input id="reviewer" placeholder="填写实际审阅人姓名" autocomplete="name"></label><button id="export">导出审阅结论</button><label>载入之前的结论<input id="restore" type="file" accept=".json,application/json"></label><span id="message" role="status">当前结论仅在本页内存中；关闭前请导出 JSON。</span></div>
<div class="workspace"><aside><div id="stats"></div><input id="search" aria-label="按样本编号查找" placeholder="查找样本编号"><select id="filter" aria-label="按结论筛选"><option value="all">全部结论</option><option value="pending">待审阅</option><option value="approved">批准</option><option value="rejected">拒绝</option><option value="needs_revision">需修订</option></select><div id="list"></div></aside><main id="main"></main></div>
<footer>离线文件，无需服务或网络。机器检查尚未证明全局单元不重叠、近壁分辨率充分或 CFD 精度达标。</footer>
<script id="review-data" type="application/json">__REVIEW_PAYLOAD__</script>
<script>
'use strict';
const review=JSON.parse(document.getElementById('review-data').textContent),labels={pending:'待审阅',approved:'批准',rejected:'拒绝',needs_revision:'需修订'};
let current=review.episodes.length?review.episodes[0].episode_id:null,dirty=false;
const decisions=Object.fromEntries(review.episodes.map(e=>[e.episode_id,{episode_id:e.episode_id,episode_digest:e.episode_digest,status:'pending',reviewer:'',comment:''}]));
const $=id=>document.getElementById(id),escapeHtml=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function note(text){$('message').textContent=text}function changed(){dirty=true;note('有未导出的修改。关闭前请导出 JSON。');renderList()}
function renderList(){const counts={pending:0,approved:0,rejected:0,needs_revision:0};Object.values(decisions).forEach(d=>counts[d.status]++);$('stats').textContent=`本批 ${review.episodes.length} 条；待审阅 ${counts.pending}，批准 ${counts.approved}，拒绝 ${counts.rejected}，需修订 ${counts.needs_revision}。两类拓扑最多各 10 条，数量不足不重复补齐。`;$('list').replaceChildren();for(const e of review.episodes){const d=decisions[e.episode_id];if(($('filter').value!=='all'&&$('filter').value!==d.status)||!e.sample_id.includes($('search').value))continue;const b=document.createElement('button');b.className='episode-button'+(current===e.episode_id?' current':'');b.innerHTML=`${escapeHtml(e.sample_id)}<small>规则 ${e.variant} · <span class="status-${d.status}">${labels[d.status]}</span></small>`;b.onclick=()=>{current=e.episode_id;renderMain();renderList()};$('list').append(b)}}
const metricNames={valid:'机器有效性',cells:'单元数',min_scaled_jacobian:'最小缩放 Jacobian',q10_scaled_jacobian:'10% 分位 Jacobian',wall_normal_deviation_deg_max:'壁面面中心法向偏差 / °',wall_node_normal_deviation_deg_max:'壁面节点法向偏差 / °',first_height_min:'最小首层高度 / c',first_height_max:'最大首层高度 / c',protected_layer_growth_max:'保护带最大层距增长比'};
function table(q){return '<table><tbody>'+Object.entries(metricNames).filter(([k])=>k in q).map(([k,n])=>`<tr><th>${n}</th><td>${typeof q[k]==='number'?q[k].toPrecision(6):escapeHtml(q[k])}</td></tr>`).join('')+'</tbody></table>'}
function renderMain(){const e=review.episodes.find(e=>e.episode_id===current);if(!e){$('main').innerHTML='<h2>当前没有可审阅轨迹</h2><p class="muted">请先提供通过自动审计、且拓扑类别已核实的训练轨迹。不会自动批准数据。</p>';return}const d=decisions[current];$('main').innerHTML=`<h2>${escapeHtml(e.sample_id)}</h2><p class="muted">规则 ${e.variant}；机器检查通过；数据来源 ${escapeHtml(e.source)}。</p><div class="notice">检查动作是否合理、保护带是否保持、块接口是否一致，以及前缘、尾缘与近壁网格是否可接受。下方数值是几何筛查结果，不能替代工程判断或 CFD 验证。</div><div class="verdict"><label>人工结论<select id="verdict">${Object.entries(labels).map(([v,n])=>`<option value="${v}">${n}</option>`).join('')}</select></label><label>审阅意见<textarea id="comment" placeholder="记录依据、问题位置或需要修改的动作"></textarea><span class="muted" id="saved-reviewer"></span></label></div><h3>物理网格</h3><div class="image-tabs"><button class="secondary" id="near">近壁</button><button class="secondary" id="full">全场</button></div><figure><img id="mesh-image" src="${escapeHtml(e.near_wall_image)}" alt="实际终态网格的近壁视图"><figcaption>物理坐标 x/c、y/c。全场与近壁图片展示的是同一张最终验收网格。</figcaption></figure><div class="quality"><section><h3>环境终态验收（最终规格）</h3>${table(e.terminal_quality)}</section><section><h3>独立复验（相同最终规格）</h3>${table(e.fine_quality)}</section></div><details><summary>展开网格参数与轨迹标识</summary><pre id="provenance"></pre></details><h3>逐步动作与参数域拓扑</h3><p class="muted">以下为 u/v 参数坐标，不是翼型物理坐标。蓝色为保护带，橙色为可编辑区域。</p><div class="frames">${e.frames.map(f=>`<figure><img src="${escapeHtml(e.episode_id+'/'+f.image)}" alt="第 ${f.step} 步的参数域拓扑" loading="lazy"><figcaption>第 ${f.step} 步${f.action===undefined?'':'，动作 '+f.action}：${escapeHtml(f.description)}</figcaption></figure>`).join('')}</div>`;$('verdict').value=d.status;$('comment').value=d.comment;$('saved-reviewer').textContent=d.reviewer?'此结论记录的审阅人：'+d.reviewer:'尚未记录人工审阅人';$('provenance').textContent=JSON.stringify({episode_id:e.episode_id,episode_digest:e.episode_digest,env_config:e.env_config,mesh_spec:e.mesh_spec,environment_version:review.environment_version,package_id:review.package_id},null,2);$('verdict').onchange=()=>{d.status=$('verdict').value;d.reviewer=$('reviewer').value.trim();changed()};$('comment').oninput=()=>{d.comment=$('comment').value;if($('reviewer').value.trim())d.reviewer=$('reviewer').value.trim();changed()};$('near').onclick=()=>{$('mesh-image').src=e.near_wall_image;$('mesh-image').alt='实际终态网格的近壁视图'};$('full').onclick=()=>{$('mesh-image').src=e.overview_image;$('mesh-image').alt='实际终态网格的全场视图'}}
$('export').onclick=()=>{const rows=Object.values(decisions).map(d=>({...d}));for(const d of rows){if(d.status!=='pending'&&!d.reviewer)d.reviewer=$('reviewer').value.trim();if(d.status!=='pending'&&!d.reviewer){note('导出失败：每条非待审阅结论都需要填写实际审阅人姓名。');return}}const data={schema:'review_decisions_v2',package_id:review.package_id,episodes_sha256:review.episodes_sha256,review_id:review.review_id,decisions:rows};const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)+'\n'],{type:'application/json;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='review_decisions_'+review.package_id.slice(0,12)+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);dirty=false;note('已导出 JSON。使用 import-reviews 命令校验并生成新的训练清单。')};
$('restore').onchange=async()=>{const file=$('restore').files[0];if(!file)return;try{const incoming=JSON.parse(await file.text());if(incoming.schema!=='review_decisions_v2'||incoming.package_id!==review.package_id||incoming.episodes_sha256!==review.episodes_sha256||incoming.review_id!==review.review_id)throw Error('此文件属于不同或过期的审阅包');if(!Array.isArray(incoming.decisions))throw Error('缺少结论列表');const seen=new Set();for(const d of incoming.decisions){if(!decisions[d.episode_id]||seen.has(d.episode_id)||d.episode_digest!==decisions[d.episode_id].episode_digest||!Object.hasOwn(labels,d.status)||typeof d.reviewer!=='string'||typeof d.comment!=='string')throw Error('轨迹或结论字段不匹配');seen.add(d.episode_id)}for(const d of incoming.decisions)decisions[d.episode_id]={episode_id:d.episode_id,episode_digest:d.episode_digest,status:d.status,reviewer:d.reviewer,comment:d.comment};dirty=false;renderList();renderMain();note('已载入之前的结论，可继续审阅。')}catch(error){note('载入失败：'+error.message)}};
$('search').oninput=renderList;$('filter').onchange=renderList;window.addEventListener('beforeunload',event=>{if(dirty){event.preventDefault();event.returnValue=''}});renderList();renderMain();
</script></html>'''
