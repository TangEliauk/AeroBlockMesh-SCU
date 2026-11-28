"""Reproducible train/validation-only airfoil decomposition experiment."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import statistics
import time
import traceback

import numpy as np
import torch
import yaml

from smartcfd.core.types import Transition, MeshSpec, Failure
from smartcfd.rl.agents.sac import SACConfig
from smartcfd.rl.replay_buffer import ReplayBuffer, snapshot
from .config import AirfoilDecompositionConfig, ENVIRONMENT_VERSION, ACTION_VERSION, OBSERVATION_VERSION
from .data import select_cases, preflight, digest, ROOT


DEFAULTS = {'version': 1, 'environment': {}, 'data': {'seed': 42, 'train_count': 32, 'val_count': 8},
    'model': {'hidden_dim': 64, 'layers': 2},
    'training': {'seed': 42, 'steps': 2000, 'warmup_steps': 128, 'batch_size': 32,
        'evaluation_interval': 500, 'replay_capacity': 2048, 'replay_max_bytes': 128*1024**2,
        'alpha': .01, 'gamma': .99, 'tau': .005, 'learning_rate': .0003, 'grad_clip': 1.},
    'diagnostics': asdict(MeshSpec())}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value, *, replace=False):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp') if replace else path
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default)
        stream.write('\n')
    if replace:
        temporary.replace(path)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Unsupported report value: {type(value).__name__}')


def normalize_config(value):
    if not isinstance(value, dict) or set(value)-set(DEFAULTS):
        raise ValueError('Unknown experiment configuration sections')
    result = json.loads(json.dumps(DEFAULTS))
    for key, section in value.items():
        if key == 'environment':
            result[key] = dict(section)
        elif isinstance(result[key], dict):
            if not isinstance(section, dict) or set(section)-set(result[key]):
                raise ValueError(f'Unknown configuration keys: {key}')
            result[key].update(section)
        else:
            result[key] = section
    if result['version'] != 1 or result['model'] != {'hidden_dim': 64, 'layers': 2}:
        raise ValueError('This MVP requires version1 and width64/layers2')
    result['environment'] = asdict(AirfoilDecompositionConfig(**result['environment']))
    for key in ('steps', 'batch_size', 'evaluation_interval', 'replay_capacity', 'replay_max_bytes'):
        if type(result['training'][key]) is not int or result['training'][key] < 1:
            raise ValueError(f'training.{key} must be a positive integer')
    if type(result['training']['warmup_steps']) is not int or result['training']['warmup_steps'] < 0:
        raise ValueError('warmup_steps must be a nonnegative integer')
    if result['environment']['gamma'] != result['training']['gamma']:
        raise ValueError('Reward shaping and SAC discounts must match')
    if any(type(result['data'][k]) is not int or result['data'][k] < 1 for k in ('train_count', 'val_count')):
        raise ValueError('Positive manifest counts required')
    MeshSpec(**result['diagnostics'])
    return result


def source_fingerprint():
    paths = sorted((ROOT/'src/smartcfd').rglob('*.py')) + [ROOT/'scripts/run_airfoil_decomposition.py']
    files = {p.relative_to(ROOT).as_posix(): file_sha(p) for p in paths}
    packages = {p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'gymnasium', 'shapely', 'matplotlib', 'PyYAML')}
    return {'files': files, 'source_sha256': digest(files), 'packages': packages,
            'scope': 'All current smartcfd Python source and the independent airfoil entrypoint'}


def network_hashes(agent):
    result = {}
    for name in ('actor', 'q1', 'q2', 'target_q1', 'target_q2'):
        h = hashlib.sha256()
        for key, tensor in getattr(agent, name).state_dict().items():
            array = tensor.detach().cpu().contiguous().numpy()
            h.update(key.encode()); h.update(str(array.dtype).encode())
            h.update(str(array.shape).encode()); h.update(array.tobytes())
        result[name] = h.hexdigest()
    return result


def fixed_action(env):
    """Translate the existing four-outer-strip rule into the current candidates."""
    descriptors = env.candidates
    for u in (.25, .5, .75):
        a, b = np.array([u, .5]), np.array([u, 1.])
        for index in np.flatnonzero(env.action_mask):
            candidate = descriptors[int(index)]
            if candidate['kind'] != 'insert_chord':
                continue
            p, q = np.array(candidate['source_uv']), np.array(candidate['target_uv'])
            if (np.allclose(p, a, atol=1e-12, rtol=0) and np.allclose(q, b, atol=1e-12, rtol=0)
                or np.allclose(q, a, atol=1e-12, rtol=0) and np.allclose(p, b, atol=1e-12, rtol=0)):
                return int(index)
    return None


def rule_action(env):
    legal = np.flatnonzero(env.action_mask).tolist()
    def score(action):
        p = env.preview(action)
        return (float(p['reward']), float(p['quality'].get('q', 0.)), -action)
    return max(legal, key=score) if legal else None


def evaluate(cases, config, *, method, seed=42, agent=None, preflight_rows=None, pool=None, progress=False):
    from .env import AirfoilBlockEnv
    pool = {} if pool is None else pool
    prechecked = {r['sample_id']: r for r in (preflight_rows or [])}
    modes = {k: getattr(agent, k).training for k in ('actor', 'q1', 'q2')} if agent else {}
    outcomes = []
    try:
        for index, sample in enumerate(cases):
            sample_id = sample['sample_id']
            record = {'sample_id': sample_id, 'group_id': sample['group_id'], 'method': method,
                'success': False, 'reason': None, 'steps': 0, 'return': 0., 'base_return': 0.,
                'actions': [], 'step_rewards': [], 'partition': None, 'elapsed_seconds': 0.}
            if sample_id in prechecked and not prechecked[sample_id]['valid']:
                record.update(reason=prechecked[sample_id]['reason'], phase='preflight', initializable=False)
                outcomes.append(record)
                continue
            started = time.perf_counter()
            if sample_id not in pool:
                pool[sample_id] = AirfoilBlockEnv(sample, config=config)
            env = pool[sample_id]
            obs, info = env.reset(seed=seed+index)
            rng = random.Random(seed+index)
            record['initializable'] = not (env.terminated and not env.success)
            for _ in range(config.max_steps):
                if env.terminated:
                    break
                legal = np.flatnonzero(obs['mask']).tolist()
                if not legal:
                    record['reason'] = 'no_legal_actions'
                    break
                if method == 'random':
                    action = rng.choice(legal)
                elif method == 'fixed_c':
                    action = fixed_action(env)
                elif method == 'rule':
                    action = rule_action(env)
                else:
                    if agent is None:
                        raise ValueError('A learned evaluation requires an agent')
                    action = agent.act(obs, deterministic=True)
                if action is None:
                    record['reason'] = 'fixed_rule_has_no_legal_next_action'
                    break
                descriptor = dict(env.candidates[action])
                obs, reward, terminated, truncated, info = env.step(action)
                record['actions'].append({'index': int(action), 'candidate': descriptor})
                base = float(info['base_reward'])
                record['step_rewards'].append({'total': float(reward), 'base': base,
                                               'shaping': float(info['shaping'])})
                record['return'] += float(reward); record['base_return'] += base
                if terminated or truncated:
                    break
            record.update(success=bool(env.success), steps=len(record['actions']), partition=env.partition(),
                          elapsed_seconds=time.perf_counter()-started)
            record['reason'] = record['reason'] or info.get('reason') or 'evaluation_budget'
            q = record['partition'].get('quality', {})
            record['terminal_quality_score'] = (float(q.get('q', 0.))-config.outer_block_cost*q.get('outer_blocks', 0)
                                                 if record['success'] else -1.)
            outcomes.append(record)
            if progress:
                print(json.dumps({'evaluation': method, 'completed': index+1, 'total': len(cases),
                                  'success': record['success'], 'steps': record['steps']}), flush=True)
    finally:
        for name, training in modes.items():
            getattr(agent, name).train(training)
    passed = [r for r in outcomes if r['success']]
    ready = [r for r in outcomes if r.get('initializable')]
    quality = {}
    if passed:
        keys = set.intersection(*(set(row['partition']['quality']) for row in passed))
        for key in sorted(keys):
            values = [row['partition']['quality'][key] for row in passed]
            if all(type(v) in (float, int) and np.isfinite(v) for v in values):
                quality[key] = {'mean': statistics.mean(values), 'min': min(values), 'max': max(values)}
    return {'method': method, 'total': len(cases), 'success_count': len(passed), 'success_rate': len(passed)/len(cases),
        'initializable_count': len(ready), 'success_rate_initializable': len(passed)/len(ready) if ready else None,
        'mean_return': statistics.mean(r['return'] for r in outcomes),
        'mean_base_return': statistics.mean(r['base_return'] for r in outcomes),
        'mean_steps': statistics.mean(r['steps'] for r in outcomes),
        'mean_terminal_quality_score': statistics.mean(r.get('terminal_quality_score', -1.) for r in outcomes),
        'mean_elapsed_seconds': statistics.mean(r['elapsed_seconds'] for r in outcomes),
        'quality_successful_only': quality, 'episodes': outcomes,
        'case_ids': [r['sample_id'] for r in cases],
        'timing_scope': 'Environment construction/reset, selection, geometric updates and export; pure geometry caches may be warm. Rule additionally previews candidates; not equal compute work.'}


def reachability_gate(sample, config):
    """Find two distinct valid partitions before learning, without using val."""
    from .env import AirfoilBlockEnv
    from .baselines import refined_c_action
    from smartcfd.partition.topology import fingerprint
    env = AirfoilBlockEnv(sample, config=config)
    trajectories, seen = [], set()
    for attempt in range(2):
        attempt_started = time.perf_counter()
        print(json.dumps({'reachability_attempt': attempt, 'phase': 'starting'}), flush=True)
        obs, info = env.reset(seed=42+attempt)
        actions = []
        for step in range(config.max_steps):
            if env.terminated:
                break
            if attempt == 0:
                action = fixed_action(env)
            else:
                action = refined_c_action(env)
            if action is None:
                break
            descriptor = dict(env.candidates[action])
            obs, reward, done, truncated, info = env.step(action)
            actions.append({'index': int(action), 'candidate': descriptor})
            if done or truncated:
                break
        part = env.partition()
        identity = repr(tuple(sorted((protected, tuple((round(x, 8), round(y, 8)) for x, y in points))
                                     for protected, points in fingerprint(env.graph))))
        trajectories.append({'attempt': attempt, 'success': bool(env.success), 'reason': info.get('reason'),
                             'actions': actions, 'partition': part})
        print(json.dumps({'reachability_attempt': attempt, 'success': bool(env.success), 'steps': len(actions),
                          'reason': info.get('reason'), 'seconds': time.perf_counter()-attempt_started}), flush=True)
        if env.success:
            seen.add(identity)
        if len(seen) >= 2:
            return {'passed': True, 'sample_id': sample['sample_id'], 'distinct_partitions': len(seen),
                    'attempts': trajectories, 'scope': 'Engineering reachability proof on one training geometry; not demonstrations or performance evidence'}
    return {'passed': False, 'sample_id': sample['sample_id'], 'distinct_partitions': len(seen), 'attempts': trajectories}


def train_online(agent, cases, config, training, replay, callback, output):
    from .env import AirfoilBlockEnv
    rng = random.Random(training['seed'])
    order, cursor, episode_id = list(range(len(cases))), 0, 0
    pool, episodes, updates = {}, [], []
    episode_log = None
    def append_episode(record):
        episodes.append(record)
        if episode_log is not None:
            episode_log.write(json.dumps(record, allow_nan=False)+'\n'); episode_log.flush()
    def reset():
        nonlocal cursor, episode_id
        for _ in range(len(cases)):
            if cursor % len(order) == 0:
                rng.shuffle(order)
            sample = cases[order[cursor % len(order)]]; cursor += 1
            key = sample['sample_id']
            if key not in pool:
                pool[key] = AirfoilBlockEnv(sample, config=config)
            env = pool[key]
            obs, info = env.reset(seed=training['seed']+episode_id); episode_id += 1
            if env.terminated:
                append_episode({'sample_id': key, 'steps': 0, 'return': 0., 'base_return': 0.,
                                'success': bool(env.success), 'reason': info.get('reason'), 'initial_terminal': True})
                continue
            return env, obs, sample
        raise ValueError('No training case can collect a legal interaction')
    env, obs, sample = reset()
    total, base_total, count = 0., 0., 0
    with (output/'updates.jsonl').open('x', encoding='utf-8') as update_log, (output/'episodes.jsonl').open('x', encoding='utf-8') as episode_log:
        for record in episodes:
            episode_log.write(json.dumps(record, allow_nan=False)+'\n')
        episode_log.flush()
        for step in range(1, training['steps']+1):
            legal = np.flatnonzero(obs['mask']).tolist()
            action = rng.choice(legal) if step <= training['warmup_steps'] else agent.act(obs)
            before = snapshot(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            replay.add(Transition(before, action, float(reward), next_obs, bool(terminated), bool(truncated)))
            total += float(reward); base_total += float(info['base_reward']); count += 1
            if step >= training['warmup_steps'] and len(replay) >= training['batch_size']:
                update = {**agent.update(replay.sample(training['batch_size'])), 'online_step': step}
                updates.append(update)
                update_log.write(json.dumps(update, allow_nan=False)+'\n'); update_log.flush()
            obs = next_obs
            if terminated or truncated:
                record = {'sample_id': sample['sample_id'], 'steps': count, 'return': total,
                    'base_return': base_total, 'success': bool(info['success']), 'reason': info.get('reason'),
                    'terminated': bool(terminated), 'truncated': bool(truncated), 'initial_terminal': False}
                append_episode(record)
                total, base_total, count = 0., 0., 0
                if step < training['steps']:
                    env, obs, sample = reset()
            if step % 100 == 0:
                print(json.dumps({'online_step': step, 'updates': agent.updates, 'episodes': len(episodes)}), flush=True)
            if step % training['evaluation_interval'] == 0 or step == training['steps']:
                callback(agent, step)
    return {'steps': training['steps'], 'updates': updates, 'episodes': episodes, 'partial_episode_steps': count,
        'partial_episode': {'sample_id': sample['sample_id'], 'steps': count, 'return': total, 'base_return': base_total} if count else None,
        'replay_size': len(replay), 'replay_retained_bytes': replay.allocated_bytes, 'replay_max_bytes': replay.max_bytes,
        'replay_total_added': replay.total_added, 'replay_evicted': replay.total_evicted,
        'scope': 'Only real airfoil environment interactions; no BC, demonstrations, old replay or holdout'}


def mesh_diagnostics(outcomes, cases, config, mesh_config):
    """Explicit adapter for this exact UV/CMapping contract; never a reward."""
    from smartcfd.core.types import BlockGraph
    from smartcfd.meshing.generator import generate_mesh
    from .geometry import CurveGeometry
    samples = {s['sample_id']: s for s in cases}
    reports = {}
    for method, report in outcomes.items():
        rows = []
        for outcome in report['episodes']:
            record = {'sample_id': outcome['sample_id'], 'attempted': bool(outcome['success']),
                      'scope': 'Post-evaluation fixed traditional mesh diagnostic; not partition success or checkpoint selection'}
            if outcome['success']:
                try:
                    graph = BlockGraph.from_dict(outcome['partition']['graph'])
                    geometry = CurveGeometry(samples[outcome['sample_id']], config)
                    validation = geometry.validate(graph, refine=True)
                    if not validation['valid']:
                        raise ValueError('Partition lost validity before mesh adapter')
                    mesh = generate_mesh(geometry.domain, graph, MeshSpec(**mesh_config))
                    if isinstance(mesh, Failure):
                        record.update(valid=False, reason=mesh.reason, detail=mesh.detail, diagnostics=mesh.diagnostics)
                    else:
                        record.update(valid=True, quality=mesh.quality,
                                      block_count=len(mesh.blocks), interface_count=len(mesh.interfaces))
                except (ValueError, RuntimeError, FloatingPointError) as exc:
                    record.update(valid=False, reason=f'{type(exc).__name__}: {exc}')
            rows.append(record)
        reports[method] = rows
    return reports


def run_experiment(config_path, output, *, steps=None, seed=None, argv=None):
    from .learning import AirfoilSACAgent, save_inference_checkpoint, load_inference_checkpoint
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT/'experiments/runs') or output == ROOT/'experiments/runs':
        raise ValueError('Output must be a new child of this checkout experiments/runs')
    if output.exists():
        raise FileExistsError(f'Refusing existing run: {output}')
    config_path = Path(config_path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if steps is not None:
        raw.setdefault('training', {})['steps'] = steps
    if seed is not None:
        raw.setdefault('training', {})['seed'] = seed
    cfg = normalize_config(raw)
    output.mkdir(parents=True)
    receipt = {'schema': 'airfoil_decomposition_run_v1', 'status': 'running', 'created_at_utc': utc_now(),
        'environment_version': ENVIRONMENT_VERSION, 'observation_version': OBSERVATION_VERSION, 'action_version': ACTION_VERSION,
        'config': cfg, 'config_input': str(config_path), 'config_input_sha256': file_sha(config_path), 'argv': argv,
        'scope': 'C-mapping-constrained single sharp-airfoil decomposition; no free topology, BC or CFD'}
    write_json(output/'run.json', receipt)
    try:
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        source = source_fingerprint(); receipt['source'] = source
        cases = select_cases(**cfg['data'])
        dataset_sha = digest(cases); receipt['dataset_sha256'] = dataset_sha
        write_json(output/'cases.json', cases); write_json(output/'config.json', cfg); write_json(output/'source.json', source)
        write_json(output/'run.json', receipt, replace=True)
        e = AirfoilDecompositionConfig(**cfg['environment']); t = cfg['training']
        checks = preflight(cases, e); write_json(output/'preflight.json', checks)
        accepted = {r['sample_id'] for r in checks['train'] if r['valid']}
        training_cases = [s for s in cases['train'] if s['sample_id'] in accepted]
        if not training_cases:
            raise ValueError('No physically initializable training sample; retain frozen manifest and failures')
        gate = reachability_gate(training_cases[0], e)
        write_json(output/'reachability_gate.json', gate)
        if not gate['passed']:
            raise ValueError('Two distinct valid reachable partitions not demonstrated; training not started')
        sac = SACConfig(alpha=t['alpha'], gamma=t['gamma'], tau=t['tau'], lr=t['learning_rate'], grad_clip=t['grad_clip'])
        agent = AirfoilSACAgent(sac, device='cpu', seed=t['seed'])
        initial_hashes = network_hashes(agent)
        pool = {}
        outcomes = {m: evaluate(cases['val'], e, method=m, seed=t['seed'], preflight_rows=checks['val'], pool=pool,
                               agent=agent if m == 'untrained' else None, progress=True) for m in ('fixed_c', 'random', 'rule', 'untrained')}
        history, best_score = [], (-float('inf'),)*3
        def validation(current, step, report=None):
            nonlocal best_score
            report = report or evaluate(cases['val'], e, method='learned_selection', seed=t['seed'], agent=current,
                                         preflight_rows=checks['val'], pool=pool)
            history.append({'online_step': step, **report})
            score = (report['success_rate'], report['mean_terminal_quality_score'], -report['mean_steps'])
            if score > best_score:
                best_score = score
                save_inference_checkpoint(current, output/'best.pt', environment_config=cfg['environment'], dataset_sha256=dataset_sha,
                    source=source, online_step=step, selection='val_success_then_terminal_quality_then_fewer_steps_earliest_tie', overwrite=step != 0)
            write_json(output/'validation_history.json', history, replace=step != 0)
            print(json.dumps({'validation_step': step, 'success': report['success_count'], 'total': report['total'],
                              'terminal_quality_score': report['mean_terminal_quality_score']}), flush=True)
        validation(agent, 0, outcomes['untrained'])
        replay = ReplayBuffer(t['replay_capacity'], t['replay_max_bytes'], t['seed'])
        start = time.perf_counter()
        online = train_online(agent, training_cases, e, t, replay, validation, output)
        training_seconds = time.perf_counter()-start
        final_hashes = network_hashes(agent)
        save_inference_checkpoint(agent, output/'final.pt', environment_config=cfg['environment'], dataset_sha256=dataset_sha,
                                  source=source, online_step=t['steps'], selection='final')
        checkpoints = {}
        for selection in ('best', 'final'):
            path = output/f'{selection}.pt'; before = file_sha(path)
            restored, state = load_inference_checkpoint(path, environment_config=cfg['environment'], dataset_sha256=dataset_sha,
                                                         source=source, sac_config=sac, device='cpu')
            outcomes[selection] = evaluate(cases['val'], e, method=f'learned_{selection}', seed=t['seed'], agent=restored,
                                           preflight_rows=checks['val'], pool=pool, progress=True)
            if file_sha(path) != before:
                raise ValueError('Checkpoint changed during read-only reload evaluation')
            checkpoints[selection] = {'sha256_before': before, 'sha256_after': before,
                                     'online_step': state['online_step'], 'inference_only': True}
        write_json(output/'outcomes.json', outcomes)
        diagnostics = mesh_diagnostics(outcomes, cases['val'], e, cfg['diagnostics'])
        write_json(output/'mesh_diagnostics.json', diagnostics)
        from .visualize import plot_partition, plot_learning
        figures = output/'figures'; figures.mkdir()
        lookup = {s['sample_id']: s for s in cases['val']}
        for method, report in outcomes.items():
            for outcome in report['episodes']:
                plot_partition(lookup[outcome['sample_id']], outcome, figures/f'{method}_{outcome["sample_id"]}.png')
        plot_learning(history, online['updates'], output/'learning_curves.png')
        source_data_unchanged = all(file_sha(ROOT/name) == expected for name, expected in cases['manifest']['source_files'].items())
        if (source_fingerprint() != source or file_sha(config_path) != receipt['config_input_sha256']
                or digest(cases) != dataset_sha or not source_data_unchanged):
            raise ValueError('Source, dependency, config or frozen manifest changed during execution')
        metrics = {'schema': 'airfoil_decomposition_metrics_v1', 'seed': t['seed'], 'online': online,
            'training_seconds_including_validation': training_seconds, 'training_manifest_count': len(cases['train']),
            'training_initializable_count': len(training_cases), 'validation_history': history,
            'evaluations': {m: {k: v for k, v in report.items() if k != 'episodes'} for m, report in outcomes.items()},
            'initial_network_hashes': initial_hashes, 'final_network_hashes': final_hashes,
            'network_changed': {k: initial_hashes[k] != final_hashes[k] for k in initial_hashes}, 'checkpoints': checkpoints,
            'best_online_step': checkpoints['best']['online_step'], 'source': source, 'dataset_sha256': dataset_sha,
            'limitations': ['Single seed development; same validation set selects and reports best, no held-out generalization claim.',
                'C mapping, wake connection and protected band are prescribed; topology is not free.',
                'Geometry sampling/refinement is numerical verification, not an analytic CAD proof.',
                'Partition validity, traditional mesh diagnostics and CFD accuracy are distinct; no CFD was executed.',
                'No BC/expert data; checkpoints are inference-only and cannot resume full training.',
                'Method timing includes unequal geometry preview work and warm caches, not equal-compute comparison.']}
        write_json(output/'metrics.json', metrics)
        receipt.update(status='completed', finished_at_utc=utc_now(), summary={'steps': t['steps'], 'updates': len(online['updates']),
            'best_online_step': metrics['best_online_step'], 'success_counts': {m: r['success_count'] for m, r in outcomes.items()},
            'validation_total': len(cases['val']), 'metrics_sha256': file_sha(output/'metrics.json')})
        write_json(output/'run.json', receipt, replace=True)
        return output, metrics
    except BaseException as exc:
        (output/'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        receipt.update(status='failed', finished_at_utc=utc_now(), error=f'{type(exc).__name__}: {exc}')
        write_json(output/'run.json', receipt, replace=True)
        raise
