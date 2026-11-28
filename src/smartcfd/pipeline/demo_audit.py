"""Replay raw demonstration lines and build an immutable, deduplicated package."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

from smartcfd.core.types import ENVIRONMENT_VERSION, EnvConfig, Failure, MeshSpec
from smartcfd.geometry.domain import make_domain
from smartcfd.partition.actions import ACTION_SCHEMA
from smartcfd.rl.state import OBSERVATION_SCHEMA
from smartcfd.pipeline.audit_shared import (EPISODE_SCHEMA, PACKAGE_SCHEMA, LEGACY_EPISODE_SCHEMA, canonical_hash,
    canonical_json, episode_id_for, file_sha256, package_id_for)
from smartcfd.pipeline.config import complete_run, environment_config, implementation_version, new_run, write_json
from smartcfd.pipeline.data import data_version, load_samples, repository_root


class AuditRejection(ValueError):
    def __init__(self, reason, detail='', diagnostics=None):
        self.reason, self.detail, self.diagnostics = reason, detail, diagnostics or {}
        super().__init__(detail or reason)


def _report_value(value):
    """Keep failed numerical diagnostics writable without calling NaN valid."""
    if hasattr(value, 'tolist'):
        return _report_value(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return f'nonfinite:{value}'
    if isinstance(value, dict):
        return {key: _report_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_report_value(item) for item in value]
    return value


def _check_declared(record, key, expected, constructor=None):
    declared = record.get(key)
    if declared is None:
        return False
    try:
        normalized = asdict(constructor(**declared)) if constructor else declared
    except (TypeError, ValueError) as error:
        raise AuditRejection(f'invalid_{key}', str(error)) from error
    if canonical_hash(normalized) != canonical_hash(expected):
        raise AuditRejection(f'{key}_mismatch', diagnostics={'declared': declared, 'expected': expected})
    return True


def _domain_config(config):
    settings = {'farfield': 500.0, 'stretch': 18.0, **config.get('domain', {})}
    if set(settings) != {'farfield', 'stretch'}:
        raise ValueError('Only farfield and stretch are supported domain configuration fields')
    return {key: float(value) for key, value in settings.items()}


def _legacy_environment(record, expected_env, expected_mesh, migrate_environment):
    """Only explicitly replay old 8/16 mesh semantics into the current MDP.

    Reward, geometry, domain and data conflicts are never migration exceptions.
    Existing review flags are provenance only and cannot become new approvals.
    """
    schema = record.get('schema')
    declared_env = record.get('env_config')
    if declared_env is not None and not isinstance(declared_env, dict):
        raise AuditRejection('invalid_env_config')
    nested_version = (declared_env or {}).get('environment_version')
    version = record.get('environment_version', nested_version)
    if (record.get('environment_version') is not None and nested_version is not None
            and record['environment_version'] != nested_version):
        raise AuditRejection('environment_version_conflict')
    if schema == EPISODE_SCHEMA:
        if version != ENVIRONMENT_VERSION:
            raise AuditRejection('environment_version_mismatch')
        return False
    if version == ENVIRONMENT_VERSION:
        if schema == LEGACY_EPISODE_SCHEMA:
            raise AuditRejection('episode_schema_environment_conflict')
        return False
    if version not in (None, 'coarse_mesh_v1'):
        raise AuditRejection('environment_version_mismatch')
    if not migrate_environment:
        raise AuditRejection('environment_migration_required',
            'Replay legacy actions with explicit --migrate-environment; old reviews do not transfer')
    if schema not in (None, LEGACY_EPISODE_SCHEMA):
        raise AuditRejection('episode_schema_mismatch')
    mesh_fields = set(asdict(MeshSpec()))
    for key, value in (declared_env or {}).items():
        if key not in expected_env:
            raise AuditRejection('invalid_env_config', f'Unknown environment field: {key}')
        if key not in mesh_fields | {'environment_version'} and canonical_hash(value) != canonical_hash(expected_env[key]):
            raise AuditRejection('env_config_mismatch', f'Migration cannot change {key}')
    old = asdict(MeshSpec(intervals=8, protected_intervals=16))
    declared_mesh = {key: value for key, value in (declared_env or {}).items() if key in mesh_fields}
    for candidate in (old, expected_mesh):
        if all(canonical_hash(value) == canonical_hash(candidate[key]) for key, value in declared_mesh.items()):
            break
    else:
        raise AuditRejection('env_config_mismatch', 'Only the known legacy 8/16 or current mesh contract can be migrated')
    if record.get('mesh_spec') is not None:
        try:
            mesh = asdict(MeshSpec(**record['mesh_spec']))
        except (TypeError, ValueError) as error:
            raise AuditRejection('invalid_mesh_spec', str(error)) from error
        if canonical_hash(mesh) not in (canonical_hash(old), canonical_hash(expected_mesh)):
            raise AuditRejection('mesh_spec_mismatch', 'Unknown mesh settings cannot be migrated')
    if schema == LEGACY_EPISODE_SCHEMA and record.get('episode_id') is not None:
        keys = ('geometry_sha256', 'actions', 'env_config', 'domain_config',
                'mapping_version', 'observation_version', 'action_version')
        if any(key not in record for key in keys) or record['episode_id'] != canonical_hash({key: record[key] for key in keys}):
            raise AuditRejection('episode_id_mismatch', 'Legacy identity must verify before migration')
    return True


def _audit_episode(record, config, samples, membership, frozen_data, *, migrate_environment=False):
    from smartcfd.rl.env import BlockEnv
    from smartcfd.rl.imitation.dataset import rollout_episode
    from smartcfd.meshing.generator import generate_mesh
    from smartcfd.cfd.solver import validate_su2
    from smartcfd.baselines.rule_partition import rule_graph
    from smartcfd.partition.topology import fingerprint
    if not isinstance(record, dict):
        raise AuditRejection('record_not_an_object')
    if record.get('schema') not in (None, EPISODE_SCHEMA, LEGACY_EPISODE_SCHEMA):
        raise AuditRejection('episode_schema_mismatch')
    if record.get('schema_version') not in (None, 2 if record.get('schema') == EPISODE_SCHEMA else 1):
        raise AuditRejection('episode_schema_version_mismatch')
    if record.get('data_version') not in (None, 'uiuc_v1', frozen_data['version']):
        raise AuditRejection('data_version_mismatch')
    sample_id = record.get('sample_id')
    if not isinstance(sample_id, str) or sample_id not in membership:
        raise AuditRejection('unknown_sample_id', str(sample_id))
    split = membership[sample_id]
    if split == 'test':
        raise AuditRejection('test_split_forbidden', 'Test data cannot become training or validation demonstrations')
    if record.get('split') not in (None, split):
        raise AuditRejection('split_mismatch', diagnostics={'declared': record['split'], 'frozen': split})
    if sample_id not in samples:
        raise AuditRejection('unsupported_geometry')
    if not isinstance(record.get('actions'), list):
        raise AuditRejection('actions_must_be_a_list')
    options = record.get('options') or {}
    if not isinstance(options, dict):
        raise AuditRejection('invalid_options')
    if options.get('sample_id', sample_id) != sample_id:
        raise AuditRejection('sample_options_conflict')
    if 'domain_index' in options:
        raise AuditRejection('domain_index_not_portable', 'Demonstrations must select an explicit sample_id')
    if set(options) - {'sample_id'}:
        raise AuditRejection('unknown_reset_options')
    options = {'sample_id': sample_id}
    seed = record.get('seed')
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        raise AuditRejection('invalid_seed')
    env_config = asdict(environment_config(config))
    domain_config = _domain_config(config)
    mesh_spec = asdict(MeshSpec(**config.get('mesh', {})))
    domain = make_domain(samples[sample_id], **domain_config)
    geometry_hash = canonical_hash(domain.polygon)
    migrated = _legacy_environment(record, env_config, mesh_spec, migrate_environment)
    expected = {'env_config': env_config, 'domain_config': domain_config, 'mesh_spec': mesh_spec,
                'environment_version': ENVIRONMENT_VERSION,
                'observation_version': OBSERVATION_SCHEMA, 'action_version': ACTION_SCHEMA,
                'mapping_version': domain.metadata['mapping_version'], 'data': frozen_data,
                'geometry_sha256': geometry_hash}
    present = []
    for key, value in expected.items():
        if migrated and key in ('env_config', 'mesh_spec', 'environment_version'):
            present.append(False)
            continue
        present.append(_check_declared(record, key, value,
            EnvConfig if key == 'env_config' else (MeshSpec if key == 'mesh_spec' else None)))
    if record.get('schema') == EPISODE_SCHEMA and not all(present):
        raise AuditRejection('audited_context_missing', 'A v2 audited record must declare its full replay contract')
    if record.get('observation_schema') not in (None, OBSERVATION_SCHEMA):
        raise AuditRejection('observation_schema_mismatch')
    if record.get('action_schema') not in (None, ACTION_SCHEMA):
        raise AuditRejection('action_schema_mismatch')
    env = BlockEnv([domain], EnvConfig(**env_config))
    try:
        try:
            transitions = rollout_episode(env, record['actions'], seed, options)
        except (ValueError, RuntimeError, IndexError, KeyError) as error:
            diagnostic = asdict(env.last_failure) if env.last_failure else {}
            raise AuditRejection('replay_failed', str(error), diagnostic) from error
        fine = generate_mesh(domain, env.graph, MeshSpec(**mesh_spec))
        if isinstance(fine, Failure):
            raise AuditRejection('fine_mesh_failed', fine.reason, asdict(fine))
        if not fine.quality.get('valid', False):
            raise AuditRejection('fine_mesh_invalid', diagnostics=fine.quality)
        try:
            su2_stats = validate_su2(fine)
        except (ValueError, IndexError, KeyError) as error:
            raise AuditRejection('su2_validation_failed', str(error)) from error
        declared_source = record.get('source')
        prior_origin = record.get('origin') if isinstance(record.get('origin'), dict) else {}
        missing_origin = [key for key, exists in zip(expected, present) if not exists]
        if prior_origin.get('status') == 'legacy_unknown':
            missing_origin = sorted(set(missing_origin) | set(prior_origin.get('missing_provenance_fields', [])))
        variant = None
        if declared_source == 'rule_synthetic':
            for candidate in (0, 1):
                reference = rule_graph(domain, candidate)
                if not isinstance(reference, Failure) and fingerprint(env.graph) == fingerprint(reference):
                    variant = candidate
                    break
        if record.get('variant') is not None:
            declared_variant = record['variant']
            if (not isinstance(declared_variant, int) or isinstance(declared_variant, bool)
                    or declared_variant not in (0, 1) or declared_variant != variant):
                raise AuditRejection('variant_mismatch', 'Declared variant must match the replayed topology')
        audited = {'schema': EPISODE_SCHEMA, 'split': split, 'sample_id': sample_id,
            'actions': [int(action) for action in record['actions']], 'seed': seed, 'options': options,
            **expected, 'auto_passed': True, 'human_reviewed': False,
            'source': 'rule_synthetic' if declared_source == 'rule_synthetic' else 'external_unverified',
            'origin': {'status': 'declared_and_verified' if all(present) and prior_origin.get('status') != 'legacy_unknown' else 'legacy_unknown',
                       'declared_source': prior_origin.get('declared_source', declared_source),
                       'missing_provenance_fields': missing_origin,
                       'current_replay_verified': True},
            # Keep coarse_quality as a read-only compatibility alias; this is
            # now the actual final-spec terminal check, never a coarse veto.
            'terminal_quality': dict(env.mesh.quality), 'terminal_mesh_spec': mesh_spec,
            'coarse_quality': dict(env.mesh.quality), 'fine_quality': dict(fine.quality),
            'su2_stats': su2_stats, 'transition_count': len(transitions),
            'replayed_return': sum(transition.reward for transition in transitions)}
        audited['variant'] = variant
        audited['variant_verified'] = variant is not None
        if prior_origin.get('environment_migration'):
            audited['origin']['environment_migration'] = deepcopy(prior_origin['environment_migration'])
        if migrated:
            audited['origin']['environment_migration'] = {
                'source_record_sha256': canonical_hash(record), 'source_episode_id': record.get('episode_id'),
                'source_schema': record.get('schema'), 'source_environment_version': record.get('environment_version',
                    (record.get('env_config') or {}).get('environment_version')),
                'source_env_config': record.get('env_config'), 'source_mesh_spec': record.get('mesh_spec'),
                'source_origin': record.get('origin'), 'target_environment_version': ENVIRONMENT_VERSION,
                'discarded_human_review': bool(record.get('human_reviewed') or record.get('review')),
                'policy': 'Actions replayed from scratch; no old states, rewards, replay or approvals reused'}
        audited['episode_id'] = episode_id_for(audited)
        if not migrated and record.get('episode_id') is not None and record['episode_id'] != audited['episode_id']:
            raise AuditRejection('episode_id_mismatch')
        return audited
    finally:
        env.close()


def audit_demos(config, args):
    """Audit all lines independently; data errors remain records, never disappear."""
    root = repository_root()
    paths = [Path(value) if Path(value).is_absolute() else root / value for value in args.input]
    if not paths:
        raise ValueError('At least one --input JSONL path is required')
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    frozen_data = data_version()
    samples = {sample['sample_id']: sample for split in ('train', 'val') for sample in load_samples(split)}
    membership = {}
    for split in ('train', 'val', 'test'):
        for sample_id in json.loads((root / f'data/splits/uiuc_v1/{split}.json').read_text(encoding='utf-8')):
            if sample_id in membership:
                raise ValueError('Frozen split memberships overlap')
            membership[sample_id] = split
    run = new_run('audit_demonstrations', config, args.output)
    episodes, seen, counts, failure_reasons = [], set(), Counter(), Counter()
    inputs = [{'path': str(path.resolve()), 'sha256': file_sha256(path)} for path in paths]
    expected_lines = 0
    for path in paths:
        with path.open('rb') as stream:
            expected_lines += sum(bool(line.strip()) for line in stream)
    batch_started = time.perf_counter()
    try:
        with (run / 'audit_records.jsonl').open('w', encoding='utf-8', newline='\n') as results:
            for path in paths:
                with path.open('rb') as stream:
                    for number, line in enumerate(stream, 1):
                        if not line.strip():
                            continue
                        started = time.perf_counter()
                        outcome = {'input_file': str(path.resolve()), 'line': number}
                        counts['input_lines'] += 1
                        try:
                            record = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'Nonfinite JSON constant {value}')))
                            outcome['sample_id'] = record.get('sample_id') if isinstance(record, dict) else None
                            audited = _audit_episode(record, config, samples, membership, frozen_data,
                                migrate_environment=bool(getattr(args, 'migrate_environment', False)))
                            eid = audited['episode_id']
                            outcome.update(episode_id=eid, split=audited['split'], auto_passed=True)
                            if eid in seen:
                                counts['duplicates'] += 1
                                outcome['status'] = 'duplicate'
                            else:
                                seen.add(eid); episodes.append(audited)
                                counts['accepted_unique'] += 1
                                outcome['status'] = 'passed'
                        except AuditRejection as error:
                            counts['failed'] += 1; failure_reasons[error.reason] += 1
                            outcome.update(status='failed', auto_passed=False, reason=error.reason,
                                           detail=error.detail, diagnostics=error.diagnostics)
                        except (ValueError, TypeError, KeyError, IndexError) as error:
                            counts['failed'] += 1; failure_reasons['invalid_record'] += 1
                            outcome.update(status='failed', auto_passed=False, reason='invalid_record', detail=str(error))
                        except Exception as error:
                            # Unexpected per-episode failures remain visible and do not
                            # erase the remainder of a batch. Interrupts still propagate.
                            counts['failed'] += 1; failure_reasons['audit_internal_error'] += 1
                            outcome.update(status='failed', auto_passed=False, reason='audit_internal_error',
                                           detail=f'{type(error).__name__}: {error}')
                        outcome['elapsed_seconds'] = time.perf_counter() - started
                        results.write(canonical_json(_report_value(outcome)) + '\n')
                        if counts['input_lines'] % 50 == 0 or counts['input_lines'] == expected_lines:
                            results.flush()
                            progress = {'input_lines': counts['input_lines'], 'expected_lines': expected_lines,
                                'accepted_unique': counts['accepted_unique'], 'duplicates': counts['duplicates'],
                                'failed': counts['failed'], 'elapsed_seconds': time.perf_counter() - batch_started}
                            write_json(run / 'progress.json', progress)
                            print(f"demo audit {counts['input_lines']}/{expected_lines} lines; {counts['accepted_unique']} unique, {counts['failed']} failed", flush=True)
        if any(file_sha256(item['path']) != item['sha256'] for item in inputs):
            raise ValueError('A demonstration input changed during audit; no package was published')
        episodes.sort(key=lambda episode: (episode['split'], episode['sample_id'], episode['episode_id']))
        with (run / 'episodes.jsonl').open('w', encoding='utf-8', newline='\n') as output:
            for episode in episodes:
                output.write(canonical_json(episode) + '\n')
        manifest = {'schema': PACKAGE_SCHEMA, 'schema_version': 2, 'environment_version': ENVIRONMENT_VERSION,
            'episodes_file': 'episodes.jsonl',
            'episodes_sha256': file_sha256(run / 'episodes.jsonl'), 'episode_count': len(episodes),
            'split_counts': dict(Counter(e['split'] for e in episodes)), 'data': frozen_data,
            'config': config, 'implementation': implementation_version(), 'inputs': inputs}
        manifest['package_id'] = package_id_for(manifest)
        write_json(run / 'package.json', manifest)
        from smartcfd.pipeline.review import build_review_package
        review_report = build_review_package(run, episodes, config)
        report = {key: counts[key] for key in ('input_lines', 'accepted_unique', 'duplicates', 'failed')}
        report.update(package_id=manifest['package_id'], split_counts=manifest['split_counts'],
            unique_geometry_counts={split: len({e['sample_id'] for e in episodes if e['split'] == split})
                                    for split in ('train', 'val')},
            verified_topology_category_count=len({e['variant'] for e in episodes if e.get('variant_verified')}),
            variant_counts={split: dict(Counter(str(e['variant']) if e.get('variant_verified') else 'unverified'
                                for e in episodes if e['split'] == split)) for split in ('train', 'val')},
            failure_reasons=dict(failure_reasons), human_approved_count=0,
            source_counts=dict(Counter(e['source'] for e in episodes)),
            provenance_counts=dict(Counter(e['origin']['status'] for e in episodes)),
            environment_version=ENVIRONMENT_VERSION,
            migrated_episodes=sum('environment_migration' in e['origin'] for e in episodes),
            review_status='pending_human_review', training_ready=review_report.get('training_ready', False),
            test_data_used=False, review_package=review_report, elapsed_seconds=time.perf_counter() - batch_started,
            failed_diagnostic_nonfinite_encoding='nonfinite:<value>; never a valid metric')
        write_json(run / 'metrics.json', report)
        complete_run(run, report)
        return run, report
    except BaseException as error:
        complete_run(run, {'error': f'{type(error).__name__}: {error}'}, status='failed')
        raise
