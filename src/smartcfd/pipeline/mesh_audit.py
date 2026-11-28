"""Failure-preserving paired-resolution audit of frozen training/validation data."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from smartcfd.core.types import ENVIRONMENT_VERSION, EnvConfig, Failure, MeshSpec
from smartcfd.pipeline.config import complete_run, environment_config, new_run, write_json
from smartcfd.pipeline.data import data_version, load_samples
from smartcfd.geometry.domain import make_domain
from smartcfd.baselines.rule_partition import solve, rule_graph
from smartcfd.partition.topology import fingerprint
from smartcfd.rl.env import BlockEnv
from smartcfd.rl.state import OBSERVATION_SCHEMA
from smartcfd.partition.actions import ACTION_SCHEMA


def _safe(value):
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return value


def _line(stream, value):
    stream.write(json.dumps(_safe(value), ensure_ascii=False, allow_nan=False) + '\n')
    stream.flush()


def _failure(error):
    if isinstance(error, Failure):
        return _safe(asdict(error))
    return {'reason': 'exception', 'detail': str(error), 'exception_type': type(error).__name__}


def _distribution(values):
    finite = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not len(finite):
        return {'count': 0, 'min': None, 'q10': None, 'median': None, 'q90': None, 'max': None}
    return {'count': len(finite), 'min': float(finite.min()), 'q10': float(np.quantile(finite, .1)),
            'median': float(np.median(finite)), 'q90': float(np.quantile(finite, .9)), 'max': float(finite.max())}


def summarize_slots(slots):
    groups = defaultdict(list)
    for row in slots:
        groups[(row['split'], row['variant'], row['resolution'])].append(row)
    summaries = []
    for key, rows in sorted(groups.items()):
        passed = [r for r in rows if r['status'] == 'passed']
        metrics = ('min_scaled_jacobian', 'q10_scaled_jacobian', 'wall_normal_deviation_deg_max',
                   'wall_node_normal_deviation_deg_max', 'first_height_min', 'first_height_max',
                   'protected_layer_growth_max', 'cells')
        summaries.append({'split': key[0], 'variant': key[1], 'resolution': key[2],
            'total': len(rows), 'statuses': dict(Counter(r['status'] for r in rows)),
            'success_rate_all_slots': len(passed) / len(rows),
            'failure_reasons': dict(Counter(r.get('failure', {}).get('reason', 'unknown')
                                            for r in rows if r['status'] != 'passed')),
            'quality_successful_only': {name: _distribution([r.get('quality', {}).get(name) for r in passed]) for name in metrics},
            'elapsed_seconds_all_slots': _distribution([r['elapsed_seconds'] for r in rows])})
    return summaries


def audit_meshes(config, args):
    """Each case has teacher replay plus independent coarse/fine graph checks.

    A graph constructed for diagnosis never becomes a demonstration unless the
    actual teacher also replays successfully and its terminal graph agrees.
    """
    from smartcfd.meshing.generator import generate_mesh
    from smartcfd.cfd.solver import validate_su2

    splits = list(getattr(args, 'splits', None) or ('train', 'val'))
    variants = list(getattr(args, 'variants', None) or (0, 1))
    if not splits or len(set(splits)) != len(splits) or set(splits) - {'train', 'val'}:
        raise ValueError('Mesh audit supports unique train/val splits only; held-out test access is forbidden')
    if not variants or len(set(variants)) != len(variants) or set(variants) - {0, 1}:
        raise ValueError('Audit variants must be unique 0 and/or 1')
    limit = getattr(args, 'limit', None)
    if limit is not None and limit < 1:
        raise ValueError('Audit limit must be positive')
    samples = {split: load_samples(split, limit=limit) for split in splits}
    env_config = environment_config(config)
    # The historical 8/16 grid is an auxiliary diagnostic, never an MDP veto.
    coarse_spec = MeshSpec(intervals=8, protected_intervals=16)
    specs = {'coarse': coarse_spec, 'fine': env_config.mesh_spec}
    run = new_run('mesh_audit', config, args.output)
    frozen = data_version()
    slots, cases, episode_count = [], [], 0
    expected_cases = sum(map(len, samples.values())) * len(variants)
    write_json(run / 'manifest.json', {'schema': 'mesh_audit_v2', 'environment_version': ENVIRONMENT_VERSION,
        'coarse_diagnostic_only': True, 'terminal_acceptance': 'fine mesh plus interface/exterior validation',
        'splits': {s: [r['sample_id'] for r in rows] for s, rows in samples.items()},
        'variants': variants, 'mesh_specs': {k: asdict(v) for k, v in specs.items()},
        'expected_cases': expected_cases, 'expected_slots': 2 * expected_cases,
        'mapping': 'c_normal_envelope_wakegraded_v2', 'data': frozen, 'test_used': False})
    started = time.perf_counter()
    try:
        with (run / 'slots.jsonl').open('w', encoding='utf-8') as slot_stream, \
             (run / 'cases.jsonl').open('w', encoding='utf-8') as case_stream, \
             (run / 'episodes.jsonl').open('w', encoding='utf-8') as episode_stream:
            for split, rows in samples.items():
                for sample in rows:
                    domain, domain_error = None, None
                    try:
                        domain = make_domain(sample, **config.get('domain', {}))
                    except Exception as exc:
                        domain_error = _failure(exc)
                    for variant in variants:
                        case_started = time.perf_counter()
                        case_id = f"{split}_{sample['sample_id']}_v{variant}"
                        case = {'case_id': case_id, 'split': split, 'sample_id': sample['sample_id'],
                                'aliases': sample.get('aliases', []), 'variant': variant}
                        seed_base = int(getattr(args, 'seed', 42))
                        seed = seed_base + ((variant - seed_base) % 2)
                        graph, graph_error, actions, teacher_graph = None, domain_error, None, None
                        case['teacher_status'] = 'not_run'
                        teacher_started = time.perf_counter()
                        if domain is not None:
                            try:
                                graph = rule_graph(domain, variant=variant)
                                if isinstance(graph, Failure):
                                    graph_error, graph = _failure(graph), None
                            except Exception as exc:
                                graph_error = _failure(exc)
                            try:
                                env = BlockEnv([domain], env_config)
                                env.reset(seed=seed, options={'sample_id': domain.sample_id})
                                actions = solve(env, seed=seed)
                                if isinstance(actions, Failure):
                                    case.update(teacher_status='failed', teacher_failure=_failure(actions))
                                    actions = None
                                else:
                                    for action in actions:
                                        _, _, terminated, truncated, info = env.step(action)
                                    if not actions or not env.success or not env.terminated:
                                        raise ValueError('Teacher actions did not replay to a successful terminal state')
                                    teacher_graph = env.graph
                                    case.update(teacher_status='passed', steps=len(actions))
                            except Exception as exc:
                                case.update(teacher_status='failed', teacher_failure=_failure(exc))
                                actions = None
                        else:
                            case['teacher_failure'] = domain_error
                        case['teacher_elapsed_seconds'] = time.perf_counter() - teacher_started
                        case_slots = []
                        for resolution, spec in specs.items():
                            tick = time.perf_counter()
                            slot = {**{key: case[key] for key in ('case_id', 'split', 'sample_id', 'variant')},
                                    'resolution': resolution, 'mesh_spec': asdict(spec),
                                    'teacher_status': case['teacher_status'], 'mesh_valid': None}
                            if graph is None:
                                slot.update(status='not_run', stage='graph' if domain is not None else 'domain',
                                            failure=graph_error or {'reason': 'graph_unavailable'})
                            else:
                                diagnostics_dir = run / 'diagnostics' / case_id / resolution
                                try:
                                    result = generate_mesh(domain, graph, spec, diagnostics_dir=diagnostics_dir)
                                    if isinstance(result, Failure):
                                        slot.update(status='failed', stage='mesh', mesh_valid=False, failure=_failure(result))
                                    elif not result.quality.get('valid', False):
                                        slot.update(status='failed', stage='mesh', mesh_valid=False,
                                                    failure={'reason': 'invalid_mesh_contract_violation'}, quality=_safe(result.quality))
                                    else:
                                        slot.update(mesh_valid=True, quality=_safe(result.quality), blocks=len(result.blocks),
                                                    mapped_tfi_fallback_blocks=result.metadata.get('mapped_tfi_fallback_blocks', []))
                                        try:
                                            slot.update(su2_stats=validate_su2(result), status='passed', stage='completed')
                                        except Exception as exc:
                                            slot.update(status='failed', stage='export', failure=_failure(exc))
                                except Exception as exc:
                                    slot.update(status='failed', stage='mesh', failure=_failure(exc))
                                if slot['status'] != 'passed':
                                    diagnostics_dir.mkdir(parents=True, exist_ok=True)
                                    write_json(diagnostics_dir / 'graph.json', graph.to_dict())
                                    write_json(diagnostics_dir / 'failure.json', _safe(slot))
                                    slot['diagnostics_directory'] = diagnostics_dir.relative_to(run).as_posix()
                            slot['elapsed_seconds'] = time.perf_counter() - tick
                            slots.append(slot)
                            case_slots.append(slot)
                            _line(slot_stream, slot)
                        coarse_ok, fine_ok = [r['status'] == 'passed' for r in case_slots]
                        graph_matches = bool(graph is not None and teacher_graph is not None and fingerprint(graph) == fingerprint(teacher_graph))
                        case.update(coarse_passed=coarse_ok, fine_passed=fine_ok, terminal_graph_matches=graph_matches,
                                    teacher_matches_final=bool((case['teacher_status'] == 'passed') == fine_ok),
                                    pipeline_passed=bool(fine_ok and graph_matches and case['teacher_status'] == 'passed'))
                        if case['pipeline_passed']:
                            record = {'sample_id': domain.sample_id, 'split': split, 'seed': seed, 'variant': variant,
                                'actions': actions, 'options': {'sample_id': domain.sample_id}, 'source': 'rule_synthetic',
                                'human_reviewed': False, 'env_config': asdict(env_config),
                                'environment_version': ENVIRONMENT_VERSION,
                                'domain_config': {'farfield': domain.farfield, 'stretch': domain.stretch},
                                'mesh_spec': asdict(specs['fine']), 'data': frozen,
                                'geometry_sha256': hashlib.sha256(json.dumps(sample['polygon'], sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest(),
                                'mapping_version': domain.metadata['mapping_version'],
                                'observation_version': OBSERVATION_SCHEMA, 'action_version': ACTION_SCHEMA,
                                'origin': {'source_run': run.name, 'teacher_variant': variant}, 'success': True}
                            _line(episode_stream, record)
                            episode_count += 1
                        cases.append(case)
                        case['elapsed_seconds'] = time.perf_counter() - case_started
                        _line(case_stream, case)
                        if len(cases) % 25 == 0 or len(cases) == expected_cases:
                            progress = {'completed_cases': len(cases), 'expected_cases': expected_cases,
                                'recorded_slots': len(slots), 'passed_slots': sum(r['status'] == 'passed' for r in slots),
                                'elapsed_seconds': time.perf_counter() - started}
                            write_json(run / 'progress.json', progress)
                            print(f"mesh audit {len(cases)}/{expected_cases} cases; {progress['passed_slots']}/{len(slots)} slots passed", flush=True)
        statuses = dict(Counter(r['status'] for r in slots))
        result = {'schema': 'mesh_audit_v2', 'environment_version': ENVIRONMENT_VERSION,
            'coarse_diagnostic_only': True, 'geometry_counts': {s: len(rows) for s, rows in samples.items()},
            'expected_cases': expected_cases, 'recorded_cases': len(cases), 'expected_slots': 2 * expected_cases,
            'recorded_slots': len(slots), 'unique_slots': len({(r['case_id'], r['resolution']) for r in slots}),
            'slot_statuses': statuses, 'success_rate_all_slots': statuses.get('passed', 0) / max(1, len(slots)),
            'teacher_statuses': dict(Counter(c['teacher_status'] for c in cases)),
            'pipeline_passed_cases': sum(c['pipeline_passed'] for c in cases),
            'teacher_final_disagreements': [c['case_id'] for c in cases if not c['teacher_matches_final']],
            'teacher_final_disagreement_count': sum(not c['teacher_matches_final'] for c in cases),
            'coarse_failed_fine_passed': [c['case_id'] for c in cases if not c['coarse_passed'] and c['fine_passed']],
            'coarse_passed_fine_failed': [c['case_id'] for c in cases if c['coarse_passed'] and not c['fine_passed']],
            'coarse_failed_fine_passed_count': sum(not c['coarse_passed'] and c['fine_passed'] for c in cases),
            'coarse_passed_fine_failed_count': sum(c['coarse_passed'] and not c['fine_passed'] for c in cases),
            'replayable_demonstrations': episode_count, 'human_approved': 0,
            'groups': summarize_slots(slots), 'elapsed_seconds': time.perf_counter() - started,
            'records_file': 'slots.jsonl', 'cases_file': 'cases.jsonl', 'demonstrations_file': 'episodes.jsonl',
            'test_used': False, 'frozen_data_unchanged': data_version() == frozen,
            'unverified': ['global physical cell overlap', 'engineering near-wall adequacy', 'CFD accuracy'],
            'validity_rule': 'finite coordinates, corner scaled Jacobian > 1e-10, exact interfaces and exterior boundary coverage'}
        if result['recorded_slots'] != result['expected_slots'] or result['unique_slots'] != result['expected_slots']:
            raise RuntimeError('Incomplete or duplicate audit slots')
        if not result['frozen_data_unchanged']:
            raise RuntimeError('Frozen input data changed during the mesh audit')
        write_json(run / 'metrics.json', result)
        report = ['# Training/validation mesh baseline', '',
                  f"Geometries: {result['geometry_counts']}; slots: {len(slots)}; statuses: {statuses}.",
                  f"Pipeline-passed cases: {result['pipeline_passed_cases']}/{len(cases)}. Human approvals: 0.", '',
                  '| Split | Variant | Resolution | Passed / All | Success rate |', '|---|---:|---|---:|---:|']
        report += [f"| {g['split']} | {g['variant']} | {g['resolution']} | {g['statuses'].get('passed', 0)} / {g['total']} | {g['success_rate_all_slots']:.4f} |" for g in result['groups']]
        report += ['', 'Quality distributions include successful meshes only. Failures and not-run slots remain in the denominator.',
                   'MDP terminal acceptance uses the final fine MeshSpec and boundary validation. Coarse checks are diagnostic only.',
                   'The numerical validity threshold is not an engineering mesh standard. Global overlap and CFD accuracy are unverified.']
        (run / 'report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
        complete_run(run, result)
        return run, result
    except BaseException as exc:
        complete_run(run, {'error': f'{type(exc).__name__}: {exc}', 'recorded_slots': len(slots),
                           'recorded_cases': len(cases), 'partial_records_retained': True}, status='failed')
        raise
