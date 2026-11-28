"""Frozen-input orchestration for the nine-arm development study.

Every phase has a separate receipt. Completed phases may be reused only after
their receipt and frozen inputs verify; incomplete training is never resumed
from an inference-only checkpoint.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shutil
import traceback

import torch
import yaml

from .data import ROOT, select_cases, preflight, digest
from .config import AirfoilDecompositionConfig, OBSERVATION_VERSION
from .experiment import (normalize_config, source_fingerprint, file_sha, write_json,
                         utc_now, reachability_gate, mesh_diagnostics)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_study_config(path):
    raw = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    expected = {'version', 'experiment_config', 'seeds', 'guard', 'bc', 'execution'}
    if not isinstance(raw, dict) or set(raw) != expected or raw['version'] != 1:
        raise ValueError('Expected version1 airfoil guard study configuration')
    if raw['seeds'] != [42, 43, 44]:
        raise ValueError('The registered study fixes seeds42/43/44')
    if raw['guard'] != {'depth': 2, 'second_layer_budget': 128}:
        raise ValueError('The registered study fixes depth2 and 128 second-layer transitions')
    if raw['bc'] != {'updates': 1000, 'batch_size': 32, 'learning_rate': .0003}:
        raise ValueError('The registered study fixes the BC update budget')
    if raw['execution'] != {'device': 'cpu', 'torch_threads': 1}:
        raise ValueError('The registered study uses one CPU torch thread')
    base = (ROOT/raw['experiment_config']).resolve()
    if not base.is_relative_to(ROOT):
        raise ValueError('Experiment configuration must be inside the implementation copy')
    experiment = normalize_config(yaml.safe_load(base.read_text(encoding='utf-8')))
    if experiment['data'] != {'seed': 42, 'train_count': 32, 'val_count': 8}:
        raise ValueError('Refusing to change the frozen development manifest')
    t = experiment['training']
    if any(t[k] != v for k, v in {'steps': 2000, 'warmup_steps': 128, 'batch_size': 32,
                                  'evaluation_interval': 500, 'alpha': .01, 'gamma': .99,
                                  'tau': .005, 'learning_rate': .0003}.items()):
        raise ValueError('Base training settings differ from the registered study')
    return {**raw, 'experiment': experiment}


def _run_directory(output):
    output = Path(output).resolve()
    allowed = (ROOT/'experiments/runs').resolve()
    if output == allowed or not output.is_relative_to(allowed):
        raise ValueError('Study output must be a new child of this copy experiments/runs')
    return output


def prepare_study(config_path, output):
    from .guard import GUARD_VERSION
    cfg = load_study_config(config_path)
    output = _run_directory(output)
    if output.exists():
        raise FileExistsError(f'Refusing existing study: {output}')
    torch.set_num_threads(1)
    cases = select_cases(**cfg['experiment']['data'])
    source = source_fingerprint()
    extra = ['scripts/run_airfoil_guard_study.py', 'configs/airfoil_guard_study.yaml',
             'configs/airfoil_decomposition.yaml']
    source['files'].update({name: file_sha(ROOT/name) for name in extra})
    source['source_sha256'] = digest(source['files'])
    source['scope'] = 'All smartcfd Python sources, both entrypoints and the registered configurations'
    protocol = {'schema': 'airfoil_guard_study_protocol_v1', 'created_at_utc': utc_now(),
        'config': cfg, 'seeds': cfg['seeds'], 'primary_checkpoint': 'final', 'best_checkpoint': 'development_diagnostic_only',
        'groups': ['fixed_c_off', 'random_off', 'random_on', 'greedy_off', 'greedy_on',
                   'sac_off', 'sac_on', 'bc_off', 'bc_on'],
        'scope': 'Frozen train/val development; no test/NACA or CFD',
        'teacher_tie_order': ['fixed_c', 'greedy', 'refined_c'],
        'teacher_selection': 'valid completion, highest terminal_quality_score, fewer steps, method order',
        'primary_metric': 'partition success / complete eight-case validation manifest',
        'secondary_metrics': ['failure_penalized_steps_32', 'paired_common_success_terminal_quality_score'],
        'mesh_scope': 'posthoc fixed mesh specification; actual cells reported; not equal cell budget',
        'timing_scope': 'logical root/second-layer previews, actual geometry calls and wall time; not equal compute',
        'old_run_scope': 'airfoil_c_mvp_20261123_seed42 is incomplete historical development, not a completed baseline',
        'observation_version': OBSERVATION_VERSION,
        'guard_version': GUARD_VERSION,
        'guard_enumeration': 'Lazy ordered child enumeration; exact root previews and unchanged depth2/128 logical budget. Unknown child totals are null until exhaustion; generated lower bounds are diagnostic only.',
        'superseded_eager_attempt': 'airfoil_guard_study_20261124 retained as development evidence. Exact enumeration optimization validated independently before restart; all phases rerun without checkpoint reuse.',
        'observation_numerics': 'Preserve original finite differences on resolvable edges; adapt only unrepresentable short-edge tangent probes. Physical legality tolerances unchanged.',
        'previous_attempt': 'airfoil_guard_study_20261123 aborted during train[9] greedy teacher generation because the fixed tangent probe rounded to zero. No sample replacement or completed model result; evidence retained separately.'}
    context = {'study_schema': 'airfoil_guard_study_v1', 'dataset_sha256': digest(cases),
               'source': source, 'protocol_sha256': digest(protocol)}
    output.mkdir(parents=True, exist_ok=False)
    for name, value in [('config.json', cfg), ('cases.json', cases), ('source.json', source),
                        ('protocol.json', protocol), ('context.json', context)]:
        write_json(output/name, value)
    for relative in source['files']:
        target = output/'source_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/relative, target)
    receipt = {'status': 'preparing', 'created_at_utc': utc_now(), 'context': context}
    write_json(output/'run.json', receipt)
    try:
        checks = preflight(cases, AirfoilDecompositionConfig(**cfg['experiment']['environment']))
        write_json(output/'preflight.json', checks)
        valid = {r['sample_id'] for r in checks['train'] if r['valid']}
        sample = next((s for s in cases['train'] if s['sample_id'] in valid), None)
        if sample is None:
            raise ValueError('No initializable training geometry')
        gate = reachability_gate(sample, AirfoilDecompositionConfig(**cfg['experiment']['environment']))
        write_json(output/'reachability_gate.json', gate)
        if not gate['passed']:
            raise ValueError('Two distinct valid partitions are required before study training')
        frozen = {name: file_sha(output/name) for name in
                  ('config.json', 'cases.json', 'source.json', 'protocol.json', 'context.json',
                   'preflight.json', 'reachability_gate.json')}
        write_json(output/'frozen.json', frozen)
        receipt.update(status='prepared', prepared_at_utc=utc_now(), frozen_sha256=file_sha(output/'frozen.json'))
        write_json(output/'run.json', receipt, replace=True)
    except BaseException as exc:
        receipt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(output/'run.json', receipt, replace=True)
        raise
    return output


def verify_study(output):
    output = _run_directory(output)
    if file_sha(output/'frozen.json') != _read(output/'run.json').get('frozen_sha256'):
        raise ValueError('Frozen input manifest identity changed')
    for name, expected in _read(output/'frozen.json').items():
        if file_sha(output/name) != expected:
            raise ValueError(f'Frozen study input changed: {name}')
    cases = _read(output/'cases.json')
    source = _read(output/'source.json')
    for name, expected in {**source['files'], **cases['manifest']['source_files']}.items():
        if file_sha(ROOT/name) != expected:
            raise ValueError(f'Source or frozen public input changed: {name}')
    for name, expected in source['files'].items():
        if file_sha(output/'source_snapshot'/name) != expected:
            raise ValueError(f'Source snapshot changed: {name}')
    context = _read(output/'context.json')
    if context['dataset_sha256'] != digest(cases) or context['protocol_sha256'] != digest(_read(output/'protocol.json')):
        raise ValueError('Study context identity mismatch')
    if context['source'] != source:
        raise ValueError('Study source context identity mismatch')
    return (_read(output/'config.json'), cases, _read(output/'preflight.json'), context)


def _phase(output, name, callback):
    directory = output/'phases'
    directory.mkdir(exist_ok=True)
    path = directory/f'{name}.json'
    if path.exists():
        receipt = _read(path)
        if receipt.get('status') != 'completed':
            raise ValueError(f'Phase {name} is incomplete; refusing an implicit training restart')
        for relative, expected in receipt['artifacts'].items():
            if file_sha(output/relative) != expected:
                raise ValueError(f'Completed phase artifact changed: {relative}')
        return receipt
    receipt = {'phase': name, 'status': 'running', 'started_at_utc': utc_now()}
    write_json(path, receipt)
    try:
        paths = callback()
        receipt.update(status='completed', finished_at_utc=utc_now(),
                       artifacts={Path(p).relative_to(output).as_posix(): file_sha(p) for p in paths})
        write_json(path, receipt, replace=True)
        return receipt
    except BaseException as exc:
        receipt.update(status='failed', error=f'{type(exc).__name__}: {exc}', finished_at_utc=utc_now())
        write_json(path, receipt, replace=True)
        (directory/f'{name}_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


def _mesh_evaluation(study_root, evaluation_dir, cases, cfg):
    from .study import load_evaluation_rows
    from .visualize import plot_partition
    evaluation_dir = Path(evaluation_dir)
    report = _read(evaluation_dir/'summary.json')
    full = {**report, 'episodes': load_evaluation_rows(evaluation_dir/'summary.json')}
    diagnostics = mesh_diagnostics({'evaluation': full}, cases, AirfoilDecompositionConfig(**cfg['environment']), cfg['diagnostics'])['evaluation']
    figures = evaluation_dir/'figures'
    figures.mkdir(exist_ok=False)
    samples = {s['sample_id']: s for s in cases}
    outcomes = {r['sample_id']: r for r in full['episodes']}
    for row in diagnostics:
        row.update(method=report['method'], seed=report['seed'],
                   guard_enabled=bool(report.get('guard_enabled', report.get('guard'))))
        figure = figures/f"{row['sample_id']}.png"
        outcome = outcomes[row['sample_id']]
        label = f"{report['method']} | guard={'on' if row['guard_enabled'] else 'off'} | seed={report['seed']}"
        plot_partition(samples[row['sample_id']], {**outcome, 'method': label}, figure)
        row['partition_figure'] = {'path': figure.relative_to(study_root).as_posix(), 'sha256': file_sha(figure)}
    path = evaluation_dir/'mesh_diagnostics.json'
    write_json(path, diagnostics)
    return path


def run_phase(output, stage, *, seed=42, guarded=False):
    from .study import build_teachers, train_sac_arm, train_bc_arm, evaluate_cases, load_study_checkpoint
    output = _run_directory(output)
    cfg, cases, checks, context = verify_study(output)
    torch.set_num_threads(cfg['execution']['torch_threads'])
    experiment = cfg['experiment']
    if seed not in cfg['seeds']:
        raise ValueError('Seed is outside the registered study')
    protection = cfg['guard'] if guarded else None
    label = 'on' if guarded else 'off'
    if stage == 'teachers':
        def work():
            build_teachers(cases['train'], experiment['environment'], output/'teachers', preflight_rows=checks['train'])
            return list((output/'teachers').glob('*'))
        return _phase(output, 'teachers', work)
    if stage == 'sac':
        arm = output/'arms'/f'sac_{label}_seed{seed}'
        def work():
            train_sac_arm(cases, experiment, arm, seed=seed, guard_config=protection, context=context, preflight_rows=checks)
            evaluation = arm/f"validation_{experiment['training']['steps']:06d}"
            mesh = _mesh_evaluation(output, evaluation, cases['val'], experiment)
            return [arm/'run.json', arm/'summary.json', arm/'final.pt', arm/'best.pt', evaluation/'summary.json', mesh]
        return _phase(output, f'sac_{label}_seed{seed}', work)
    if stage == 'bc':
        arm = output/'arms'/f'bc_seed{seed}'
        def work():
            if not (output/'phases/teachers.json').is_file() or _read(output/'phases/teachers.json')['status'] != 'completed':
                raise ValueError('Complete the teacher phase before BC')
            result = train_bc_arm(output/'teachers/manifest.json', experiment, arm, seed=seed,
                                  context=context, updates=cfg['bc']['updates'])
            agent, _ = load_study_checkpoint(arm/'final.pt', expected_profile=result['profile'])
            paths = [arm/'run.json', arm/'summary.json', arm/'final.pt']
            for enabled in (False, True):
                name = 'on' if enabled else 'off'
                evaluation = output/'evaluations'/f'bc_{name}_seed{seed}'
                evaluate_cases(cases['val'], experiment['environment'], evaluation, method='bc', seed=seed,
                               guard_config=cfg['guard'] if enabled else None, agent=agent, preflight_rows=checks['val'],
                               checkpoint_binding={'checkpoint': str(arm/'final.pt'),
                                   'checkpoint_sha256': result['checkpoint_sha256'], 'profile': result['profile']})
                paths.extend([evaluation/'summary.json', _mesh_evaluation(output, evaluation, cases['val'], experiment)])
            if file_sha(arm/'final.pt') != result['checkpoint_sha256']:
                raise ValueError('BC checkpoint changed during inference')
            return paths
        return _phase(output, f'bc_seed{seed}', work)
    if stage == 'baselines':
        def work():
            paths = []
            arms = [('fixed_c', False, 42), ('greedy', False, 42), ('greedy', True, 42)]
            arms += [('random', enabled, s) for enabled in (False, True) for s in cfg['seeds']]
            for method, enabled, s in arms:
                name = 'on' if enabled else 'off'
                evaluation = output/'evaluations'/f'{method}_{name}_seed{s}'
                evaluate_cases(cases['val'], experiment['environment'], evaluation, method=method, seed=s,
                               guard_config=cfg['guard'] if enabled else None, preflight_rows=checks['val'])
                paths.extend([evaluation/'summary.json', _mesh_evaluation(output, evaluation, cases['val'], experiment)])
            return paths
        return _phase(output, 'baselines', work)
    raise ValueError(f'Unknown phase: {stage}')


def finalize_study(output):
    from .study import load_evaluation_rows
    from .study_report import write_study_report
    output = _run_directory(output)
    cfg, cases, checks, context = verify_study(output)
    phases = ['teachers', 'baselines'] + [f'bc_seed{s}' for s in cfg['seeds']]
    phases += [f'sac_{label}_seed{s}' for label in ('off', 'on') for s in cfg['seeds']]
    for name in phases:
        receipt = _read(output/'phases'/f'{name}.json')
        if receipt['status'] != 'completed':
            raise ValueError(f'Required study phase incomplete: {name}')
        for relative, expected in receipt['artifacts'].items():
            if file_sha(output/relative) != expected:
                raise ValueError(f'Phase artifact changed: {relative}')
    def work():
        reports, diagnostics = [], {}
        names = [('fixed_c', False, 42), ('greedy', False, 42), ('greedy', True, 42)]
        names += [(m, g, s) for m in ('random', 'sac', 'bc') for g in (False, True) for s in cfg['seeds']]
        for method, enabled, seed in names:
            label = f"{method}_{'on' if enabled else 'off'}_seed{seed}"
            directory = (output/'arms'/label/f"validation_{cfg['experiment']['training']['steps']:06d}"
                         if method == 'sac' else output/'evaluations'/label)
            report = _read(directory/'summary.json')
            # Verify all per-case artifacts even though aggregate tables need
            # only compact rows. Do not retain all dense partitions in memory.
            rows = load_evaluation_rows(directory/'summary.json')
            if {r['sample_id'] for r in rows} != {r['sample_id'] for r in cases['val']}:
                raise ValueError(f'Final evaluation sample set changed: {label}')
            del rows
            report.update(selection='final', guard_enabled=enabled)
            reports.append(report)
            diagnostics[label] = _read(directory/'mesh_diagnostics.json')
            for row in diagnostics[label]:
                figure = row['partition_figure']
                if file_sha(output/figure['path']) != figure['sha256']:
                    raise ValueError(f'Partition figure changed: {label}/{row["sample_id"]}')
        paths = write_study_report(output/'report', reports, mesh_diagnostics=diagnostics,
                                   protocol=_read(output/'protocol.json'), require_complete=True)
        return [Path(p) for p in paths.values()]
    receipt = _phase(output, 'finalize', work)
    run = _read(output/'run.json')
    run.update(status='completed', finished_at_utc=utc_now(), final_phase_sha256=file_sha(output/'phases/finalize.json'),
               scope='Complete development study; no test/NACA/CFD or equal-compute claim')
    write_json(output/'run.json', run, replace=True)
    return {'output': str(output), 'status': 'completed', 'report': str(output/'report/report.md'),
            'phase': receipt}
