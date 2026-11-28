"""Isolated orthogonal-polygon SAC experiment; no airfoil data or BC workflow."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

import numpy as np
import torch
import yaml

from smartcfd.core.types import Transition
from smartcfd.rl.agents.sac import SACConfig
from smartcfd.rl.replay_buffer import ReplayBuffer, snapshot
from . import DecompositionConfig, OrthogonalBlockEnv, generate_cases, ENVIRONMENT_VERSION, ACTION_VERSION, OBSERVATION_VERSION
from .learning import BoundarySACAgent, NETWORK, save_inference_checkpoint, load_inference_checkpoint

ROOT = Path(__file__).resolve().parents[3]
DEFAULTS = {'version': 1, 'environment': {}, 'data': {'seed': 42, 'train_count': 40, 'val_count': 12},
    'model': {'hidden_dim': 64, 'layers': 2},
    'training': {'seed': 42, 'steps': 2000, 'warmup_steps': 128, 'batch_size': 32,
        'evaluation_interval': 500, 'replay_capacity': 2048, 'replay_max_bytes': 64*1024**2,
        'alpha': .01, 'gamma': .99, 'tau': .005, 'learning_rate': .0003, 'grad_clip': 1.}}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value, *, replace=False):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp') if replace else path
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    if replace:
        temporary.replace(path)


def normalize_config(config, *, steps=None, seed=None):
    if not isinstance(config, dict) or set(config)-set(DEFAULTS):
        raise ValueError('Expected only version/environment/data/model/training configuration sections')
    result = json.loads(json.dumps(DEFAULTS))
    for key, value in config.items():
        if key == 'environment':
            result[key] = dict(value)
        elif isinstance(result[key], dict):
            if not isinstance(value, dict) or set(value)-set(result[key]):
                raise ValueError(f'Unknown configuration keys in {key}')
            result[key].update(value)
        else:
            result[key] = value
    if result['version'] != 1 or result['model'] != {'hidden_dim': 64, 'layers': 2}:
        raise ValueError('MVP requires version1 and the fixed64x2 boundary network')
    if steps is not None:
        result['training']['steps'] = steps
    if seed is not None:
        result['training']['seed'] = seed
    result['environment'] = asdict(DecompositionConfig(**result['environment']))
    t = result['training']
    for key in ('steps', 'batch_size', 'evaluation_interval', 'replay_capacity', 'replay_max_bytes'):
        if type(t[key]) is not int or t[key] < 1:
            raise ValueError(f'training.{key} must be a positive integer')
    if type(t['warmup_steps']) is not int or t['warmup_steps'] < 0:
        raise ValueError('warmup_steps must be a nonnegative integer')
    for key in ('train_count', 'val_count'):
        if type(result['data'][key]) is not int or result['data'][key] < 1:
            raise ValueError(f'data.{key} must be a positive integer')
    if t['gamma'] != result['environment']['gamma']:
        raise ValueError('SAC and environment gamma must match')
    return result


def source_fingerprint():
    paths = [*sorted((ROOT/'src/smartcfd/decomposition').glob('*.py')),
        *[ROOT/name for name in ('src/smartcfd/rl/agents/sac.py', 'src/smartcfd/rl/agents/policy.py',
            'src/smartcfd/rl/replay_buffer.py', 'src/smartcfd/core/types.py', 'scripts/run_decomposition.py')]]
    files = {path.relative_to(ROOT).as_posix(): file_sha(path) for path in paths}
    packages = {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'gymnasium', 'PyYAML', 'matplotlib', 'shapely')}
    return {'source_sha256': digest(files), 'files': files, 'packages': packages,
            'scope': 'Only decomposition source, runner and directly reused SAC/policy/replay/Transition files; no old run fingerprints'}


def network_hashes(agent):
    result = {}
    for name in ('actor', 'q1', 'q2', 'target_q1', 'target_q2'):
        value = hashlib.sha256()
        for key, tensor in getattr(agent, name).state_dict().items():
            array = tensor.detach().cpu().contiguous().numpy()
            value.update(key.encode()); value.update(str(array.dtype).encode())
            value.update(canonical(list(array.shape))); value.update(array.tobytes())
        result[name] = value.hexdigest()
    return result


def _preview_reward(env, action):
    preview = env.peek(action)
    if 'reward' not in preview or not np.isfinite(preview['reward']):
        raise ValueError('Geometry preview must expose its finite immediate reward')
    return float(preview['reward'])


def evaluate(cases, environment_config, *, method, seed, agent=None):
    """All methods use the same cases/seeds; no action/model RNG used by greedy evaluation."""
    modes = {name: getattr(agent, name).training for name in ('actor', 'q1', 'q2')} if agent else {}
    episodes = []
    try:
        for index, case in enumerate(cases):
            env = OrthogonalBlockEnv(case, config=DecompositionConfig(**environment_config))
            observation, info = env.reset(seed=seed+index)
            rng = random.Random(seed+index)
            actions, rewards = [], []
            total, ended = 0., bool(info.get('success'))
            started = time.perf_counter()
            for _ in range(environment_config['max_steps']):
                if ended:
                    break
                legal = np.flatnonzero(observation['mask']).tolist()
                if not legal:
                    info = {**info, 'success': False, 'reason': 'no_legal_actions'}
                    ended = True
                    break
                if method == 'random':
                    action = rng.choice(legal)
                elif method == 'rule':
                    action = max(legal, key=lambda a: (_preview_reward(env, a), -a))
                else:
                    if agent is None:
                        raise ValueError('Learned evaluation requires an orthogonal agent')
                    action = agent.act(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                actions.append(int(action)); rewards.append(float(reward)); total += float(reward)
                ended = bool(terminated or truncated)
            partition = env.partition()
            success = bool(info.get('success', False)) and ended
            episodes.append({'case_id': case['case_id'], 'family': case.get('family'), 'seed': seed+index,
                'success': success, 'reason': info.get('reason') if ended else 'evaluation_limit_reached',
                'steps': len(actions), 'return': total, 'actions': actions, 'step_rewards': rewards,
                'elapsed_seconds': time.perf_counter()-started, 'partition': partition,
                'initial_terminal': len(actions) == 0 and success})
    finally:
        for name, training in modes.items():
            getattr(agent, name).train(training)
    passed = [row for row in episodes if row['success']]
    quality = {}
    if passed:
        common_keys = set.intersection(*(set(row['partition']['quality']) for row in passed))
        for key in sorted(common_keys):
            values = [row['partition']['quality'][key] for row in passed]
            if all(type(value) in (int, float) and np.isfinite(value) for value in values):
                quality[key] = {'mean': statistics.mean(values), 'median': statistics.median(values),
                                'min': min(values), 'max': max(values)}
    return {'method': method, 'total': len(cases), 'success_count': len(passed), 'success_rate': len(passed)/len(cases),
        'mean_return': statistics.mean(row['return'] for row in episodes),
        'mean_steps': statistics.mean(row['steps'] for row in episodes),
        'mean_elapsed_seconds': statistics.mean(row['elapsed_seconds'] for row in episodes),
        'quality_successful_only': quality or None, 'episodes': episodes,
        'timing_scope': 'Per-case action selection, geometry steps and partition report; excludes agent/model loading and initial environment reset. Rule additionally previews every legal cut.',
        'case_ids': [case['case_id'] for case in cases]}


def train_online(agent, cases, environment_config, training, replay, callback):
    rng = random.Random(training['seed'])
    pool, cursor = list(range(len(cases))), 0
    episodes, updates, episode_number = [], [], 0

    def reset():
        nonlocal cursor, episode_number
        for _ in range(len(cases)*2):
            if cursor % len(pool) == 0:
                rng.shuffle(pool)
            case = cases[pool[cursor % len(pool)]]; cursor += 1
            env = OrthogonalBlockEnv(case, config=DecompositionConfig(**environment_config))
            obs, info = env.reset(seed=training['seed']+episode_number)
            episode_number += 1
            if info.get('success'):
                episodes.append({'case_id': case['case_id'], 'steps': 0, 'return': 0., 'success': True,
                                 'initial_terminal': True, 'reason': info.get('reason')})
                continue
            if not bool(np.asarray(obs['mask']).any()):
                raise ValueError(f'Nonterminal training reset has no legal action: {case["case_id"]}')
            return env, obs, case
        raise ValueError('Training pool has no nonterminal case; cannot collect the requested interactions')

    env, obs, case = reset()
    episode_return, episode_steps = 0., 0
    for step in range(1, training['steps']+1):
        legal = np.flatnonzero(obs['mask']).tolist()
        action = rng.choice(legal) if step <= training['warmup_steps'] else agent.act(obs)
        before = snapshot(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        replay.add(Transition(before, action, float(reward), next_obs, bool(terminated), bool(truncated)))
        episode_return += float(reward); episode_steps += 1
        if step >= training['warmup_steps'] and len(replay) >= training['batch_size']:
            result = agent.update(replay.sample(training['batch_size']))
            updates.append({**result, 'online_step': step})
        obs = next_obs
        if terminated or truncated:
            episodes.append({'case_id': case['case_id'], 'steps': episode_steps, 'return': episode_return,
                'success': bool(info.get('success')), 'reason': info.get('reason'),
                'terminated': bool(terminated), 'truncated': bool(truncated), 'initial_terminal': False})
            episode_return, episode_steps = 0., 0
            if step < training['steps']:
                env, obs, case = reset()
        if step % training['evaluation_interval'] == 0 or step == training['steps']:
            callback(agent, step)
    return {'steps': training['steps'], 'updates': updates, 'episodes': episodes,
        'partial_episode_steps': episode_steps, 'replay_size': len(replay), 'replay_retained_bytes': replay.allocated_bytes,
        'replay_max_bytes': replay.max_bytes, 'replay_total_added': replay.total_added, 'replay_evicted': replay.total_evicted,
        'source': 'Real current-environment transitions only; no BC, expert demonstrations or old replay'}


def run_experiment(config_path, output, *, steps=None, seed=None, argv=None):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT/'experiments/runs') or output == ROOT/'experiments/runs':
        raise ValueError('Output must be a new child of this copy\'s experiments/runs')
    if output.exists():
        raise FileExistsError(f'Refusing existing run: {output}')
    config_path = Path(config_path).resolve()
    output.mkdir(parents=True)
    receipt = {'schema': 'orthogonal_decomposition_run_v1', 'status': 'running', 'created_at_utc': utc_now(),
        'environment_version': ENVIRONMENT_VERSION, 'observation_version': OBSERVATION_VERSION,
        'action_version': ACTION_VERSION, 'config_input': str(config_path),
        'argv': argv if argv is not None else list(sys.argv),
        'scope': 'Orthogonal-polygon decomposition development MVP; no CFD/grid generation, BC, airfoil data, held-out test or resume'}
    write_json(output/'run.json', receipt)
    try:
        receipt['config_input_sha256'] = file_sha(config_path)
        config = normalize_config(yaml.safe_load(config_path.read_text(encoding='utf-8')), steps=steps, seed=seed)
        receipt['config'] = config
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise
        receipt['threads'] = {'intra': torch.get_num_threads(), 'interop': torch.get_num_interop_threads(), 'device': 'cpu'}
        source = source_fingerprint(); receipt['source'] = source
        cases = generate_cases(**config['data'])
        if set(cases) != {'train', 'val'}:
            raise ValueError('Case generator must expose train/val only')
        ids = {split: [case['case_id'] for case in rows] for split, rows in cases.items()}
        if len(set(ids['train'])) != len(ids['train']) or len(set(ids['val'])) != len(ids['val']) or set(ids['train']) & set(ids['val']):
            raise ValueError('Case IDs must be unique with disjoint train/val membership')
        dataset_sha = digest(cases); receipt['dataset_sha256'] = dataset_sha
        write_json(output/'config.json', config); write_json(output/'cases.json', cases)
        write_json(output/'source.json', source); write_json(output/'run.json', receipt, replace=True)
        t = config['training']; e = config['environment']
        sac = SACConfig(alpha=t['alpha'], gamma=t['gamma'], tau=t['tau'], lr=t['learning_rate'], grad_clip=t['grad_clip'])
        agent = BoundarySACAgent(sac, device='cpu', seed=t['seed'])
        initial_hashes = network_hashes(agent)
        outcomes = {method: evaluate(cases['val'], e, method=method, seed=t['seed'], agent=agent if method == 'untrained' else None)
                    for method in ('random', 'rule', 'untrained')}
        history, best_score = [], (-float('inf'), -float('inf'))

        def validation(current, step, report=None):
            nonlocal best_score
            report = report or evaluate(cases['val'], e, method='learned_selection', seed=t['seed'], agent=current)
            record = {'online_step': step, **report}; history.append(record)
            score = (report['success_rate'], report['mean_return'])
            if score > best_score:
                best_score = score
                save_inference_checkpoint(current, output/'best.pt', environment_config=e, dataset_sha256=dataset_sha,
                    source=source, online_step=step, selection='validation_success_then_mean_return_earliest_tie', overwrite=step != 0)
            write_json(output/'validation_history.json', history, replace=step != 0)
            print(json.dumps({'step': step, 'val_success': report['success_count'], 'val_total': report['total'],
                              'val_return': report['mean_return']}), flush=True)

        validation(agent, 0, outcomes['untrained'])
        replay = ReplayBuffer(t['replay_capacity'], t['replay_max_bytes'], t['seed'])
        train_started = time.perf_counter()
        online = train_online(agent, cases['train'], e, t, replay, validation)
        training_seconds = time.perf_counter()-train_started
        final_hashes = network_hashes(agent)
        save_inference_checkpoint(agent, output/'final.pt', environment_config=e, dataset_sha256=dataset_sha,
                                  source=source, online_step=t['steps'], selection='final')
        checkpoint_inputs = {}
        for selection in ('best', 'final'):
            path = output/f'{selection}.pt'; before = file_sha(path)
            restored, state = load_inference_checkpoint(path, environment_config=e, dataset_sha256=dataset_sha,
                source=source, sac_config=sac, device='cpu')
            outcomes[selection] = evaluate(cases['val'], e, method=f'learned_{selection}', seed=t['seed'], agent=restored)
            if file_sha(path) != before:
                raise ValueError('Checkpoint changed during read-only post-evaluation')
            checkpoint_inputs[selection] = {'path': str(path), 'sha256_before': before, 'sha256_after': before,
                                            'selected_at_online_step': state['online_step'], 'inference_only': True}
        if source_fingerprint() != source or file_sha(config_path) != receipt['config_input_sha256']:
            raise ValueError('Source/dependencies/config changed during execution; preserve run as failed')
        if digest(cases) != dataset_sha:
            raise ValueError('Generated dataset mutated during execution')
        metrics = {'schema': 'orthogonal_decomposition_metrics_v1', 'seed': t['seed'], 'online': online,
            'training_seconds_including_validation': training_seconds, 'validation_history': history,
            'evaluations': {name: {key: value for key, value in report.items() if key != 'episodes'} for name, report in outcomes.items()},
            'best_selected_at_online_step': checkpoint_inputs['best']['selected_at_online_step'],
            'initial_network_hashes': initial_hashes, 'final_network_hashes': final_hashes,
            'network_changed': {name: initial_hashes[name] != final_hashes[name] for name in initial_hashes},
            'checkpoints': checkpoint_inputs, 'dataset_sha256': dataset_sha, 'source': source,
            'limitations': ['Boundary message passing with global queue pooling, not paper SplineCNN.',
                'Random/legal policies may already succeed; compare block counts, aspect ratios and return, not success alone.',
                'One-step rule previews all legal geometry cuts; timing is not equal inference work.',
                'Single-seed synthetic train/val development only; same val set selects and reports best; no generalization claim.',
                'Best at step0 is untrained control preservation, not SAC learning.',
                'Inference-only checkpoints omit optimizer/replay/RNG continuation; training resume is unsupported.',
                'Results are polygon partitions and cuts, not CFD multiblock grids.']}
        write_json(output/'outcomes.json', outcomes)
        write_json(output/'metrics.json', metrics)
        from .visualize import plot_learning, plot_partition
        figure_dir = output/'figures'; figure_dir.mkdir()
        for name, report in outcomes.items():
            for case, outcome in zip(cases['val'], report['episodes']):
                plot_partition(case, outcome, figure_dir/f'{name}_{case["case_id"]}.png', title=name)
        plot_learning(history, online['updates'], output/'learning_curves.png')
        receipt.update(status='completed', finished_at_utc=utc_now(), summary={
            'steps': t['steps'], 'dataset_sha256': dataset_sha, 'best_online_step': metrics['best_selected_at_online_step'],
            'val_success_counts': {name: report['success_count'] for name, report in outcomes.items()},
            'val_total': len(cases['val']), 'metrics_sha256': file_sha(output/'metrics.json')})
        write_json(output/'run.json', receipt, replace=True)
        return output, metrics
    except BaseException as exc:
        (output/'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        receipt.update(status='failed', finished_at_utc=utc_now(), error=f'{type(exc).__name__}: {exc}')
        write_json(output/'run.json', receipt, replace=True)
        raise
