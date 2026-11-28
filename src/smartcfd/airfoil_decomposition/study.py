"""Independent, bounded experiments for candidate protection and learning.

The environment/reward/network are unchanged. Protection is an explicit policy
profile; baseline and learned actions use the same effective candidate mask.
Full per-case evidence is written incrementally rather than retained in RAM.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import asdict, is_dataclass
import copy
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
import traceback

import numpy as np
import torch

from smartcfd.core.types import Transition
from smartcfd.rl.agents.sac import SACConfig
from smartcfd.rl.replay_buffer import ReplayBuffer, snapshot, retained_bytes
from .config import AirfoilDecompositionConfig, ENVIRONMENT_VERSION, ACTION_VERSION, OBSERVATION_VERSION
from .data import digest
from .env import AirfoilBlockEnv
from .experiment import fixed_action, write_json, file_sha, utc_now, network_hashes
from .baselines import refined_c_action
from .learning import AirfoilSACAgent, NETWORK, FIELDS, batch_candidate_observations

STUDY_VERSION = 'airfoil_candidate_study_v1'
CHECKPOINT_SCHEMA = 'airfoil_candidate_study_inference_v1'
TEACHER_SCHEMA = 'airfoil_candidate_rule_teachers_v1'
METHODS = ('fixed_c', 'random', 'greedy', 'refined_c', 'sac', 'bc')
NETWORK_NAMES = ('actor', 'q1', 'q2', 'target_q1', 'target_q2')
GUARD_CACHE_BYTES = 256 * 1024**2


def _config(value):
    return value if isinstance(value, AirfoilDecompositionConfig) else AirfoilDecompositionConfig(**value)


def _guard_dict(value):
    if value is None:
        return {'enabled': False}
    values = asdict(value) if is_dataclass(value) else dict(value)
    enabled = values.pop('enabled', True)
    if not enabled:
        return {'enabled': False}
    from .guard import GuardConfig
    return {'enabled': True, **asdict(GuardConfig(**values))}


def _guard_enabled(value):
    return value is not None and _guard_dict(value).get('enabled', True)


def _new_directory(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_line(stream, row):
    def numeric(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f'Unsupported log value {type(value).__name__}')
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, default=numeric)+'\n')
    stream.flush()


def _progress(**values):
    print(json.dumps(values, ensure_ascii=True, allow_nan=False), flush=True)


def observation_hash(observation):
    h = hashlib.sha256()
    for key in FIELDS:
        value = np.ascontiguousarray(observation[key])
        h.update(key.encode()); h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode()); h.update(value.tobytes())
    return h.hexdigest()


def _checkpoint_network_hashes(path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    result = {}
    for name in NETWORK_NAMES:
        h = hashlib.sha256()
        for key, tensor in state['networks'][name].items():
            array = tensor.detach().cpu().contiguous().numpy()
            h.update(key.encode()); h.update(str(array.dtype).encode())
            h.update(str(array.shape).encode()); h.update(array.tobytes())
        result[name] = h.hexdigest()
    return result


def make_profile(config, guard_config, context, *, method):
    """Freeze all compatibility inputs, including the guard used in training."""
    from .guard import GUARD_VERSION
    if method not in ('sac', 'bc'):
        raise ValueError('Checkpoint method must be sac or bc')
    if not context.get('dataset_sha256') or not context.get('source'):
        raise ValueError('Study checkpoints require dataset_sha256 and source')
    return {'study_version': STUDY_VERSION, 'environment_version': ENVIRONMENT_VERSION,
            'action_version': ACTION_VERSION, 'observation_version': OBSERVATION_VERSION,
            'network_config': asdict(NETWORK), 'environment': copy.deepcopy(config['environment']),
            'training': copy.deepcopy(config['training']), 'guard_version': GUARD_VERSION,
            'guard': _guard_dict(guard_config), 'method': method, 'context': copy.deepcopy(context)}


def save_study_checkpoint(agent, path, *, profile, online_step, selection, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    state = {'schema': CHECKPOINT_SCHEMA, 'inference_only': True, 'training_resume_supported': False,
             'profile': copy.deepcopy(profile), 'seed': agent.seed, 'online_step': int(online_step),
             'selection': selection, 'sac_config': asdict(agent.config),
             'updates': agent.updates, 'actor_updates': agent.actor_updates,
             'networks': {name: {k: v.detach().cpu().clone() for k, v in getattr(agent, name).state_dict().items()}
                          for name in NETWORK_NAMES}}
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('xb') as stream:
        torch.save(state, stream)
    temporary.replace(path)


def load_study_checkpoint(path, *, expected_profile, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if (state.get('schema') != CHECKPOINT_SCHEMA or state.get('inference_only') is not True
            or state.get('training_resume_supported') is not False):
        raise ValueError('Study checkpoint schema mismatch; no legacy conversion is implicit')
    if state.get('profile') != expected_profile:
        raise ValueError('Study checkpoint profile mismatch (source/data/guard/config/method)')
    if state.get('seed') != expected_profile['training']['seed']:
        raise ValueError('Study checkpoint seed mismatch')
    if set(state.get('networks', {})) != set(NETWORK_NAMES):
        raise ValueError('Study checkpoint requires five network states')
    t = expected_profile['training']
    expected_sac = asdict(SACConfig(alpha=t['alpha'], gamma=t['gamma'], tau=t['tau'],
                                   lr=t['learning_rate'], grad_clip=t['grad_clip']))
    if state.get('sac_config') != expected_sac:
        raise ValueError('Study checkpoint SAC configuration mismatch')
    agent = AirfoilSACAgent(expected_sac, device=device, seed=state['seed'])
    for name, weights in state['networks'].items():
        getattr(agent, name).load_state_dict(weights, strict=True)
        getattr(agent, name).eval()
    agent.updates, agent.actor_updates = state['updates'], state['actor_updates']
    return agent, state


class GuardCachePool:
    """Retain only pure guard results across fresh training environments.

    Per-guard entry limits remain enforced by ConservativeGuard. The global
    limit additionally accounts for Python headers/owned arrays with a 10%
    cushion. It bounds retained results after each decision, not peak RSS while
    geometry validation or one new guard analysis is in progress.
    """
    def __init__(self, guard_config, max_bytes=GUARD_CACHE_BYTES):
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError('Guard cache memory cap must be a positive integer')
        self.config = _guard_dict(guard_config)
        if not self.config['enabled']:
            raise ValueError('A guard cache pool requires an enabled guard')
        self.max_bytes = max_bytes
        self.guards, self.sizes = OrderedDict(), {}
        self.peak_retained_bytes = self.evicted_guards = self.evicted_entries = 0

    def acquire(self, sample_id):
        from .guard import ConservativeGuard
        if sample_id not in self.guards:
            settings = {k: v for k, v in self.config.items() if k != 'enabled'}
            self.guards[sample_id] = ConservativeGuard(settings)
            self.sizes[sample_id] = 0
        self.guards.move_to_end(sample_id)
        return self.guards[sample_id]

    def trim(self, sample_id):
        guard = self.guards[sample_id]
        def size():
            return int(retained_bytes(guard._cache)*1.1)+512 if guard._cache else 0
        self.sizes[sample_id] = size()
        while sum(self.sizes.values()) > self.max_bytes and len(self.guards) > 1:
            victim = next(key for key in self.guards if key != sample_id)
            dropped = self.guards.pop(victim)
            self.evicted_entries += len(dropped._cache)
            self.evicted_guards += 1
            dropped._cache.clear()
            del self.sizes[victim]
        while sum(self.sizes.values()) > self.max_bytes and guard._cache:
            guard._cache.popitem(last=False)
            self.evicted_entries += 1
            self.sizes[sample_id] = size()
        self.peak_retained_bytes = max(self.peak_retained_bytes, sum(self.sizes.values()))

    def statistics(self):
        return {'max_bytes': self.max_bytes, 'retained_bytes': sum(self.sizes.values()),
                'peak_retained_bytes': self.peak_retained_bytes, 'retained_sample_guards': len(self.guards),
                'retained_entries': sum(len(guard._cache) for guard in self.guards.values()),
                'evicted_guards': self.evicted_guards, 'evicted_entries': self.evicted_entries,
                'accounting': 'Python headers and owned buffers plus 10 percent cushion; excludes transient geometry/RSS'}


def _environment(sample, config, guard_config, guard_pool=None):
    if not _guard_enabled(guard_config):
        return AirfoilBlockEnv(sample, config)
    from .guard import GuardedAirfoilEnv, GuardConfig
    values = _guard_dict(guard_config); values.pop('enabled', None)
    env = GuardedAirfoilEnv(sample, config=config, guard_config=GuardConfig(**values))
    if guard_pool is not None:
        env.guard = guard_pool.acquire(sample['sample_id'])
    return env


def _guard_report(info):
    return copy.deepcopy(info.get('guard', {'enabled': False, 'metrics': {}}))


def _effective_step(env, action, guard_config):
    observation, reward, terminated, truncated, info = env.step(action)
    return observation, float(reward), terminated, truncated, info, _guard_report(info)


def _select(env, observation, method, rng, agent=None):
    legal = np.flatnonzero(observation['mask']).tolist()
    if not legal:
        return None, 0
    if method == 'random':
        return rng.choice(legal), 0
    if method == 'greedy':
        def score(action):
            # Same primitive one-step information for both arms. The guard
            # has already supplied its separate candidate feasibility mask.
            item = AirfoilBlockEnv.preview(env, action)
            return float(item['reward']), float(item['quality'].get('q', 0.)), -action
        return max(legal, key=score), len(legal)
    if method in ('fixed_c', 'refined_c'):
        action = (fixed_action if method == 'fixed_c' else refined_c_action)(env)
        return (action if action in legal else None), 0
    if method in ('sac', 'bc') and agent is not None:
        return agent.act(observation, deterministic=True), 0
    raise ValueError('Unknown method or missing learned agent')


def _record_base(sample, method, seed):
    return {'sample_id': sample['sample_id'], 'group_id': sample.get('group_id', sample['sample_id']),
            'method': method, 'seed': seed, 'success': False, 'initializable': False, 'reason': None,
            'steps': 0, 'return': 0., 'base_return': 0., 'actions': [], 'partition': None,
            'elapsed_seconds': 0., 'terminal_quality_score': -1.}


def _rollout_case(sample, env_config, *, method, seed=42, guard_config=None, agent=None,
                  preflight_row=None, capture_observations=False, trace=None):
    """One fresh environment; optional teacher observations never enter JSON."""
    trace = {} if trace is None else trace
    trace['phase'] = 'configuration'
    if method not in METHODS:
        raise ValueError(f'Unknown study method: {method}')
    config = _config(env_config)
    row = _record_base(sample, method, seed)
    trace['row'] = row
    observations = []
    started = time.perf_counter()
    if preflight_row is not None and not preflight_row['valid']:
        row.update(reason=preflight_row['reason'], phase='preflight', penalized_steps=config.max_steps)
        return row, observations
    trace['phase'] = 'environment_construction'
    env = _environment(sample, config, guard_config)
    trace.update(env=env, phase='reset')
    observation, info = env.reset(seed=seed)
    trace['observation'] = observation
    row['initializable'] = (bool(preflight_row['valid']) if preflight_row is not None else
                            bool(getattr(env, '_validation', {}).get('valid', False)))
    rng = random.Random(seed)
    guard_report = _guard_report(info)
    while not env.terminated and len(row['actions']) < config.max_steps:
        if not observation['mask'].any():
            row['reason'] = 'proven_unfinishable'
            row['terminal_guard'] = guard_report
            break
        trace['phase'] = 'action_selection'
        action, previews = _select(env, observation, method, rng, agent)
        if action is None:
            row['reason'] = 'fixed_rule_has_no_legal_next_action'
            break
        if not observation['mask'][action]:
            raise ValueError('Policy selected an action outside effective mask')
        if capture_observations:
            observations.append(snapshot(observation))
        event = {'index': int(action), 'candidate': env.candidates[action],
                 'observation_sha256': observation_hash(observation),
                 'effective_actions': np.flatnonzero(observation['mask']).tolist(),
                 'guard': guard_report, 'selection_previews': previews}
        trace.update(pending_action=event, phase='environment_step')
        observation, reward, terminated, truncated, info, next_guard = _effective_step(env, action, guard_config)
        trace['observation'] = observation
        event.update(reward=float(reward), base_reward=float(info['base_reward']),
                     shaping=float(info['shaping']), reason=info.get('reason'),
                     terminated=bool(terminated), truncated=bool(truncated),
                     next_observation_sha256=observation_hash(observation), next_guard=next_guard)
        row['actions'].append(event)
        trace['pending_action'] = None
        row['return'] += float(reward); row['base_return'] += float(info['base_reward'])
        if terminated or truncated:
            break
        guard_report = next_guard
    trace['phase'] = 'partition_export'
    row.update(success=bool(env.success), steps=len(row['actions']), partition=env.partition(),
               reason=row['reason'] or info.get('reason') or 'evaluation_budget',
               elapsed_seconds=time.perf_counter()-started)
    if row['success']:
        quality = row['partition']['quality']
        row['terminal_quality_score'] = float(quality.get('q', 0.))-config.outer_block_cost*quality.get('outer_blocks', 0)
    row['penalized_steps'] = row['steps'] if row['success'] else config.max_steps
    row['guard_cost'] = copy.deepcopy(getattr(env, 'guard_statistics', {}))
    row['guard_cost']['selection_previews'] = sum(item['selection_previews'] for item in row['actions'])
    row['compute_cost'] = copy.deepcopy(getattr(env, 'work_statistics', {}))
    row['cost'] = {**row['compute_cost'], **{f'guard_{k}': v for k, v in row['guard_cost'].items()},
                   'selection_previews': row['guard_cost']['selection_previews']}
    return row, observations


def rollout_case(sample, env_config, *, method, seed=42, guard_config=None, agent=None,
                 preflight_row=None, capture_observations=False, diagnostic_path=None):
    """Capture a runtime exception without converting it into an ordinary failure.

    Raw topology is saved directly: exporting a physical partition or rebuilding
    an observation could reproduce the exception and lose the original evidence.
    """
    trace = {}
    try:
        return _rollout_case(sample, env_config, method=method, seed=seed, guard_config=guard_config,
            agent=agent, preflight_row=preflight_row, capture_observations=capture_observations, trace=trace)
    except Exception as exc:
        if diagnostic_path is not None:
            try:
                from smartcfd.evaluation.mesh_quality import json_safe
                env = trace.get('env')
                graph = getattr(env, 'graph', None)
                initial = getattr(env, '_initial', None)
                diagnostic = {'schema': 'airfoil_study_runtime_exception_v1', 'sample': sample,
                    'method': method, 'seed': seed, 'environment': asdict(_config(env_config)),
                    'guard': _guard_dict(guard_config), 'phase': trace.get('phase'),
                    'error_type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc(),
                    'completed_actions': trace.get('row', {}).get('actions', []),
                    'pending_action': trace.get('pending_action'), 'partial_outcome': trace.get('row'),
                    'last_observation': trace.get('observation'),
                    'state': {'graph': graph.to_dict() if graph is not None else None,
                        'initial_graph': initial.to_dict() if initial is not None else None,
                        'queue': list(getattr(env, 'queue', ())), 'steps': getattr(env, 'steps', 0),
                        'history': copy.deepcopy(getattr(env, '_history', [])),
                        'validation': copy.deepcopy(getattr(env, '_validation', {})),
                        'attempt_validation': copy.deepcopy(getattr(env, '_attempt_validation', None)),
                        'terminated': getattr(env, 'terminated', None), 'success': getattr(env, 'success', None),
                        'reason': getattr(env, '_reason', None)},
                    'nonfinite_diagnostic_numbers': 'represented_as_null',
                    'scope': 'Runtime exception evidence; excluded from ordinary outcome metrics; original exception re-raised'}
                write_json(diagnostic_path, json_safe(diagnostic))
            except Exception as diagnostic_error:
                exc.add_note(f'Could not write runtime diagnostic: {type(diagnostic_error).__name__}: {diagnostic_error}')
        raise


def summarize_rows(rows):
    if not rows:
        raise ValueError('Evaluation requires at least one case')
    successful = [r for r in rows if r['success']]
    initializable = sum(bool(r['initializable']) for r in rows)
    return {'total': len(rows), 'success_count': len(successful), 'success_rate': len(successful)/len(rows),
            'initializable_count': initializable,
            'success_rate_initializable': len(successful)/initializable if initializable else None,
            'mean_steps': statistics.mean(r['steps'] for r in rows),
            'mean_penalized_steps': statistics.mean(r['penalized_steps'] for r in rows),
            'mean_terminal_quality_score': statistics.mean(r['terminal_quality_score'] for r in rows),
            'successful_only_terminal_quality': statistics.mean(r['terminal_quality_score'] for r in successful) if successful else None,
            'mean_elapsed_seconds': statistics.mean(r['elapsed_seconds'] for r in rows),
            'failure_reasons': dict(Counter(r['reason'] for r in rows if not r['success']))}


def _verify_evaluation_binding(binding, *, identity, agent, method, seed, env_config, guard_config, selection):
    if binding is None:
        return None
    if agent is None or method not in ('sac', 'bc'):
        raise ValueError('Checkpoint binding requires a learned policy')
    if not isinstance(binding, dict) or not {'checkpoint', 'checkpoint_sha256', 'profile'} <= set(binding):
        raise ValueError('Checkpoint binding requires checkpoint, checkpoint_sha256 and profile')
    path = Path(binding['checkpoint']).resolve()
    checksum = file_sha(path)
    if checksum != binding['checkpoint_sha256']:
        raise ValueError('Evaluation checkpoint SHA mismatch')
    state = torch.load(path, map_location='cpu', weights_only=True)
    profile = binding['profile']
    if (state.get('schema') != CHECKPOINT_SCHEMA or state.get('profile') != profile
            or state.get('inference_only') is not True or state.get('training_resume_supported') is not False):
        raise ValueError('Evaluation checkpoint schema/profile mismatch')
    if (profile.get('method') != method or state.get('seed') != seed or agent.seed != seed
            or profile.get('training', {}).get('seed') != seed):
        raise ValueError('Evaluation checkpoint method/seed mismatch')
    if profile.get('environment') != asdict(_config(env_config)):
        raise ValueError('Evaluation checkpoint environment mismatch')
    if method == 'sac' and profile.get('guard') != _guard_dict(guard_config):
        raise ValueError('SAC evaluation guard differs from training profile')
    if selection == 'final' and state.get('selection') not in ('fixed_bc_final', 'fixed_sac_final_primary'):
        raise ValueError('Final evaluation requires the fixed-budget final checkpoint')
    if _checkpoint_network_hashes(path) != identity:
        raise ValueError('Evaluation agent tensors differ from bound checkpoint')
    if file_sha(path) != checksum:
        raise ValueError('Checkpoint changed during binding verification')
    return {'checkpoint': str(path), 'checkpoint_sha256': checksum, 'profile': copy.deepcopy(profile),
            'profile_sha256': digest(profile), 'method': method, 'seed': seed,
            'selection': state.get('selection'), 'online_step': state.get('online_step'),
            'network_hashes': identity, 'verified_saved_tensors': True}


def evaluate_cases(cases, env_config, output, *, method, seed=42, guard_config=None,
                   agent=None, preflight_rows=None, selection='final', checkpoint_binding=None):
    if method in ('sac', 'bc') and agent is None:
        raise ValueError('Learned evaluation requires an agent')
    before = network_hashes(agent) if agent is not None else None
    binding = _verify_evaluation_binding(checkpoint_binding, identity=before, agent=agent,
                     method=method, seed=seed, env_config=env_config, guard_config=guard_config, selection=selection)
    output = _new_directory(output)
    prechecked = {r['sample_id']: r for r in (preflight_rows or [])}
    rows = []
    modes = {name: getattr(agent, name).training for name in NETWORK_NAMES} if agent else {}
    for name in modes:
        getattr(agent, name).eval()
    try:
        for index, sample in enumerate(cases):
            path = output/f'{index:04d}_{sample["sample_id"]}.json'
            row, _ = rollout_case(sample, env_config, method=method, seed=seed+index,
                                  guard_config=guard_config, agent=agent, preflight_row=prechecked.get(sample['sample_id']),
                                  diagnostic_path=path.with_name(path.stem+'.diagnostic_exception.json'))
            write_json(path, row)
            compact = {k: v for k, v in row.items() if k not in ('actions', 'partition')}
            compact.update(path=path.name, filepath=str(path), sha256=file_sha(path))
            rows.append(compact)
            _progress(phase='evaluation', method=method, case=index+1, total=len(cases), success=row['success'], seed=seed)
    finally:
        for name, training in modes.items():
            getattr(agent, name).train(training)
    report = {'schema': STUDY_VERSION, 'method': method, 'seed': seed, 'guard': _guard_dict(guard_config),
              'guard_enabled': _guard_enabled(guard_config), 'selection': selection, 'output': str(output),
              **summarize_rows(rows), 'episodes': rows,
              'timing_scope': 'Fresh environment per case; includes reset, guard, choice, steps and export; unequal search work is logged.'}
    if agent is not None:
        after = network_hashes(agent)
        identity = {'network_hashes': before, 'network_hashes_before': before,
                    'network_hashes_after': after, 'unchanged_during_validation': before == after}
        if before != after:
            write_json(output/'network_identity_failure.json', identity)
            raise ValueError('Evaluation changed network tensors')
        report['network_identity'] = identity
    if binding is not None:
        after_sha = file_sha(binding['checkpoint'])
        if after_sha != binding['checkpoint_sha256']:
            write_json(output/'checkpoint_identity_failure.json', {'binding': binding, 'observed_sha256': after_sha})
            raise ValueError('Bound checkpoint changed during evaluation')
        binding.update(verified_validation_unchanged=True, verified_checkpoint_unchanged=True)
        report['checkpoint_binding'] = binding
    write_json(output/'summary.json', report)
    return report


def load_evaluation_rows(report):
    """Load hash-verified cases from an evaluation dict or summary JSON path."""
    if isinstance(report, (str, Path)):
        report = json.loads(Path(report).read_text(encoding='utf-8'))
    result = []
    for row in report['episodes']:
        path = Path(row['filepath']) if 'filepath' in row else Path(report['output'])/row['path']
        if file_sha(path) != row['sha256']:
            raise ValueError('Evaluation evidence hash mismatch')
        result.append(json.loads(path.read_text(encoding='utf-8')))
    return result


def build_teachers(cases_train, env_config, output, *, preflight_rows=None):
    output = _new_directory(output)
    prechecked = {r['sample_id']: r for r in (preflight_rows or [])}
    selected, coverage = [], []
    for index, sample in enumerate(cases_train):
        if sample.get('split', 'train') != 'train':
            raise ValueError('Teachers may only use train geometries')
        proposals = []
        for method in ('fixed_c', 'greedy', 'refined_c'):
            path = output/f'{index:04d}_{method}.json'
            row, observations = rollout_case(sample, env_config, method=method,
                 preflight_row=prechecked.get(sample['sample_id']), capture_observations=True,
                 diagnostic_path=path.with_name(path.stem+'.diagnostic_exception.json'))
            write_json(path, row)
            if row['success'] and row['partition'].get('valid'):
                proposals.append((row, observations, path))
        entry = {'sample_id': sample['sample_id'], 'group_id': sample.get('group_id', sample['sample_id']),
                 'initializable': prechecked.get(sample['sample_id'], {}).get('valid'),
                 'teacher_available': bool(proposals)}
        if proposals:
            # Stable max preserves declared fixed_c, greedy, refined_c tie order.
            row, observations, evidence = max(proposals, key=lambda p: (p[0]['terminal_quality_score'], -p[0]['steps']))
            arrays = {key: np.stack([obs[key] for obs in observations]) for key in FIELDS}
            arrays['actions'] = np.asarray([r['index'] for r in row['actions']], dtype=np.int64)
            data_path = output/f'{index:04d}_selected.npz'
            np.savez_compressed(data_path, **arrays)
            entry.update(method=row['method'], steps=row['steps'], terminal_quality_score=row['terminal_quality_score'],
                         path=data_path.name, sha256=file_sha(data_path), evidence=evidence.name,
                         evidence_sha256=file_sha(evidence), source='rule_synthetic')
            selected.append(copy.deepcopy(entry))
        coverage.append(entry)
        _progress(phase='teachers', completed=index+1, total=len(cases_train), available=entry['teacher_available'])
    manifest = {'schema': TEACHER_SCHEMA, 'source': 'rule_synthetic', 'selected': selected,
                'coverage': coverage, 'total': len(cases_train), 'available': len(selected),
                'environment': asdict(_config(env_config)), 'selection': 'valid_success_then_terminal_score_then_fewer_steps_then_fixed_greedy_refined',
                'sampling': 'uniform family, then geometry, then state; no validation demonstrations'}
    manifest['dataset_sha256'] = digest(manifest)
    write_json(output/'manifest.json', manifest)
    return manifest


class TeacherDataset:
    """Verified teacher arrays with a two-trajectory LRU cache."""
    def __init__(self, manifest_path):
        self.path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.path.read_text(encoding='utf-8'))
        if self.manifest.get('schema') != TEACHER_SCHEMA:
            raise ValueError('Teacher schema mismatch')
        content = {k: v for k, v in self.manifest.items() if k != 'dataset_sha256'}
        if digest(content) != self.manifest.get('dataset_sha256'):
            raise ValueError('Teacher manifest hash mismatch')
        self.groups, self.cache = {}, OrderedDict()
        for row in self.manifest['selected']:
            path = (self.path.parent/row['path']).resolve()
            if not path.is_relative_to(self.path.parent) or file_sha(path) != row['sha256']:
                raise ValueError('Teacher array path/hash mismatch')
            evidence = (self.path.parent/row['evidence']).resolve()
            if not evidence.is_relative_to(self.path.parent) or file_sha(evidence) != row['evidence_sha256']:
                raise ValueError('Teacher evidence path/hash mismatch')
            self.groups.setdefault(row['group_id'], []).append(row)
        if not self.groups:
            raise ValueError('No complete valid training teacher exists')
        self.group_ids = sorted(self.groups)

    def sample(self, rng):
        group = rng.choice(self.group_ids)
        row = rng.choice(self.groups[group])
        key = row['path']
        if key not in self.cache:
            with np.load(self.path.parent/key, allow_pickle=False) as archive:
                values = {name: archive[name].copy() for name in (*FIELDS, 'actions')}
            if len(values['actions']) != row['steps'] or row['steps'] < 1:
                raise ValueError('Teacher state/action count mismatch')
            self.cache[key] = values
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        values = self.cache[key]
        index = rng.randrange(row['steps'])
        observation = {name: values[name][index].copy() for name in FIELDS}
        action = int(values['actions'][index])
        if not 0 <= action < len(observation['mask']) or not observation['mask'][action]:
            raise ValueError('Teacher label is outside its recorded legal mask')
        return observation, action, row['sample_id']


def _agent(config, seed):
    t = config['training']
    torch.set_num_threads(1)
    return AirfoilSACAgent(SACConfig(alpha=t['alpha'], gamma=t['gamma'], tau=t['tau'],
              lr=t['learning_rate'], grad_clip=t['grad_clip']), device='cpu', seed=seed)


def _receipt(output, profile, seed):
    row = {'schema': STUDY_VERSION, 'status': 'running', 'created_at_utc': utc_now(),
           'seed': seed, 'profile': profile}
    write_json(output/'run.json', row)
    return row


def _failed(output, receipt, exc):
    receipt.update(status='failed', finished_at_utc=utc_now(), error=f'{type(exc).__name__}: {exc}')
    write_json(output/'run.json', receipt, replace=True)
    (output/'error.txt').write_text(traceback.format_exc(), encoding='utf-8')


def train_bc_arm(teacher_manifest_path, config, output, *, seed, context, updates=1000):
    if type(updates) is not int or updates < 1:
        raise ValueError('BC updates must be a positive integer')
    config = copy.deepcopy(config); config['training']['seed'] = seed
    dataset = TeacherDataset(teacher_manifest_path)
    if dataset.manifest['environment'] != config['environment']:
        raise ValueError('Teacher environment differs from training environment')
    context = {**context, 'teacher_dataset_sha256': dataset.manifest['dataset_sha256'], 'bc_updates': updates}
    profile = make_profile(config, None, context, method='bc')
    output = _new_directory(output); receipt = _receipt(output, profile, seed)
    start = time.perf_counter()
    try:
        agent, rng = _agent(config, seed), random.Random(seed)
        with (output/'updates.jsonl').open('x', encoding='utf-8') as stream:
            for update in range(1, updates+1):
                samples = [dataset.sample(rng) for _ in range(config['training']['batch_size'])]
                batch = batch_candidate_observations([s[0] for s in samples])
                labels = torch.tensor([s[1] for s in samples], dtype=torch.int64)
                agent.actor.train()
                _, log_probabilities = agent.actor.distribution(batch)
                loss = -log_probabilities.gather(1, labels[:, None]).mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite BC loss')
                agent.actor_optimizer.zero_grad(set_to_none=True); loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), config['training']['grad_clip'], error_if_nonfinite=True)
                agent.actor_optimizer.step(); agent.actor_updates += 1
                _write_line(stream, {'update': update, 'loss': float(loss.detach()),
                                    'gradient_before_clip': float(gradient), 'sample_ids': [s[2] for s in samples]})
                if update % 100 == 0 or update == updates:
                    _progress(phase='bc', seed=seed, update=update, total=updates)
        save_study_checkpoint(agent, output/'final.pt', profile=profile, online_step=0, selection='fixed_bc_final')
        summary = {'method': 'bc', 'seed': seed, 'updates': updates, 'teacher_available': dataset.manifest['available'],
                   'teacher_total': dataset.manifest['total'], 'seconds': time.perf_counter()-start,
                   'profile': profile, 'checkpoint_sha256': file_sha(output/'final.pt'),
                   'scope': 'Fixed-budget BC; same final actor evaluated with guard off/on; no validation selection'}
        write_json(output/'summary.json', summary)
        receipt.update(status='completed', finished_at_utc=utc_now(), summary_sha256=file_sha(output/'summary.json'))
        write_json(output/'run.json', receipt, replace=True)
        return summary
    except BaseException as exc:
        _failed(output, receipt, exc)
        raise


def train_sac_arm(cases, config, output, *, seed, guard_config, context, preflight_rows=None):
    config = copy.deepcopy(config); config['training']['seed'] = seed
    profile = make_profile(config, guard_config, context, method='sac')
    output = _new_directory(output); receipt = _receipt(output, profile, seed)
    started = time.perf_counter()
    try:
        t, e = config['training'], _config(config['environment'])
        if any(row.get('split', 'train') != 'train' for row in cases['train']):
            raise ValueError('SAC interactions may only use training geometries')
        train_groups = {r['group_id'] for r in cases['train'] if 'group_id' in r}
        val_groups = {r['group_id'] for r in cases['val'] if 'group_id' in r}
        if train_groups & val_groups:
            raise ValueError('Training/validation family overlap')
        checked = preflight_rows or {}
        excluded = {r['sample_id']: r for r in checked.get('train', []) if not r['valid']}
        train = [r for r in cases['train'] if r['sample_id'] not in excluded]
        if not train:
            raise ValueError('No initializable training cases')
        agent, rng = _agent(config, seed), random.Random(seed)
        replay = ReplayBuffer(t['replay_capacity'], t['replay_max_bytes'], seed)
        guard_pool = GuardCachePool(guard_config) if _guard_enabled(guard_config) else None
        history, best_score, best_step = [], None, None
        def validate(step):
            nonlocal best_score, best_step
            identity = network_hashes(agent)
            report = evaluate_cases(cases['val'], e, output/f'validation_{step:06d}', method='sac',
                    seed=seed, guard_config=guard_config, agent=agent, preflight_rows=checked.get('val'),
                    selection='final' if step == t['steps'] else 'development_intermediate')
            if network_hashes(agent) != identity:
                raise ValueError('Evaluation changed network tensors')
            report['network_identity']['online_step'] = step
            write_json(output/f'validation_{step:06d}'/'summary.json', report, replace=True)
            compact = {k: v for k, v in report.items() if k != 'episodes'}
            compact['online_step'] = step; history.append(compact)
            score = (report['success_rate'], report['mean_terminal_quality_score'], -report['mean_steps'])
            if best_score is None or score > best_score:
                save_study_checkpoint(agent, output/'best.pt', profile=profile, online_step=step,
                        selection='development_val_best_not_primary', overwrite=best_step is not None)
                best_score, best_step = score, step
            write_json(output/'validation_history.json', history, replace=len(history) > 1)
        validate(0)
        order, cursor, episode_id = list(range(len(train))), 0, 0
        def reset():
            nonlocal cursor, episode_id
            for _ in range(len(train)):
                if cursor % len(order) == 0:
                    rng.shuffle(order)
                sample = train[order[cursor % len(order)]]; cursor += 1
                env = _environment(sample, e, guard_config, guard_pool)
                observation, info = env.reset(seed=seed+episode_id); episode_id += 1
                if guard_pool is not None:
                    guard_pool.trim(sample['sample_id'])
                if env.terminated:
                    _write_line(episodes, {'sample_id': sample['sample_id'], 'episode': episode_id,
                               'steps': 0, 'success': bool(env.success), 'reason': info.get('reason')})
                    continue
                report = _guard_report(info)
                if not observation['mask'].any():
                    _write_line(episodes, {'sample_id': sample['sample_id'], 'episode': episode_id,
                               'steps': 0, 'success': False, 'reason': 'proven_unfinishable', 'guard': report})
                    continue
                return sample, env, observation, report
            raise ValueError('No training case has an effective legal interaction')
        with (output/'updates.jsonl').open('x', encoding='utf-8') as update_log, \
             (output/'episodes.jsonl').open('x', encoding='utf-8') as episodes, \
             (output/'interactions.jsonl').open('x', encoding='utf-8') as interactions:
            for sample_id, failed in excluded.items():
                _write_line(episodes, {'sample_id': sample_id, 'steps': 0, 'success': False,
                                      'reason': failed['reason'], 'phase': 'preflight'})
            sample, env, obs, guard_report = reset()
            episode_steps, episode_return = 0, 0.
            for step in range(1, t['steps']+1):
                legal = np.flatnonzero(obs['mask']).tolist()
                action = rng.choice(legal) if step <= t['warmup_steps'] else agent.act(obs)
                before, descriptor = snapshot(obs), env.candidates[action]
                next_obs, reward, terminated, truncated, info, next_guard = _effective_step(env, action, guard_config)
                if guard_pool is not None:
                    guard_pool.trim(sample['sample_id'])
                replay.add(Transition(before, action, float(reward), next_obs, bool(terminated), bool(truncated)))
                episode_steps += 1; episode_return += float(reward)
                _write_line(interactions, {'online_step': step, 'episode': episode_id, 'sample_id': sample['sample_id'],
                        'index': int(action), 'candidate': descriptor, 'observation_sha256': observation_hash(before),
                        'next_observation_sha256': observation_hash(next_obs),
                        'effective_actions': np.flatnonzero(before['mask']).tolist(), 'guard': guard_report,
                        'next_guard': next_guard, 'reward': float(reward), 'base_reward': float(info['base_reward']),
                        'shaping': float(info['shaping']), 'terminated': bool(terminated), 'truncated': bool(truncated),
                        'reason': info.get('reason')})
                if step >= t['warmup_steps'] and len(replay) >= t['batch_size']:
                    metrics = agent.update(replay.sample(t['batch_size']))
                    _write_line(update_log, {'online_step': step, **metrics})
                obs, guard_report = next_obs, next_guard
                if terminated or truncated:
                    path = output/f'episode_{episode_id:06d}.json'
                    write_json(path, {'sample_id': sample['sample_id'], 'success': bool(info['success']),
                               'reason': info.get('reason'), 'partition': env.partition(),
                               'compute_cost': getattr(env, 'work_statistics', {}),
                               'guard_cost': getattr(env, 'guard_statistics', {})})
                    _write_line(episodes, {'sample_id': sample['sample_id'], 'episode': episode_id,
                               'steps': episode_steps, 'return': episode_return, 'success': bool(info['success']),
                               'reason': info.get('reason'), 'path': path.name,
                               'compute_cost': getattr(env, 'work_statistics', {}),
                               'guard_cost': getattr(env, 'guard_statistics', {})})
                    episode_steps, episode_return = 0, 0.
                    if step < t['steps']:
                        sample, env, obs, guard_report = reset()
                if step % 100 == 0 or step == t['steps']:
                    _progress(phase='sac', seed=seed, online_step=step, updates=agent.updates)
                if step % t['evaluation_interval'] == 0 or step == t['steps']:
                    validate(step)
        before_save = network_hashes(agent)
        if before_save != history[-1]['network_identity']['network_hashes']:
            raise ValueError('Final weights differ from the final validation weights')
        save_study_checkpoint(agent, output/'final.pt', profile=profile, online_step=t['steps'], selection='fixed_sac_final_primary')
        after_save = network_hashes(agent)
        saved_hashes = _checkpoint_network_hashes(output/'final.pt')
        if before_save != after_save or before_save != saved_hashes:
            raise ValueError('Saving changed network tensors or persisted different tensors')
        final_sha = file_sha(output/'final.pt')
        binding = {'checkpoint': 'final.pt', 'checkpoint_sha256': final_sha,
                   'validation_summary': f'validation_{t["steps"]:06d}/summary.json',
                   'online_step': t['steps'], 'network_hashes': saved_hashes,
                   'verified_validation_unchanged': True, 'verified_save_unchanged': True,
                   'verified_saved_tensors': True, 'requires_additional_inference': False}
        final_report_path = output/binding['validation_summary']
        final_report = json.loads(final_report_path.read_text(encoding='utf-8'))
        final_report['checkpoint_binding'] = binding
        write_json(final_report_path, final_report, replace=True)
        history[-1]['checkpoint_binding'] = binding
        write_json(output/'validation_history.json', history, replace=True)
        summary = {'method': 'sac', 'seed': seed, 'steps': t['steps'], 'updates': agent.updates,
                   'training_total': len(cases['train']), 'training_initializable': len(train),
                   'partial_episode_steps': episode_steps, 'best_online_step': best_step,
                   'replay_retained_bytes': replay.allocated_bytes, 'replay_max_bytes': replay.max_bytes,
                   'replay_total_added': replay.total_added, 'replay_evicted': replay.total_evicted,
                   'seconds': time.perf_counter()-started, 'profile': profile,
                   'environment_cache_policy': 'fresh environment/geometry per episode; only pure guard results persist by sample',
                   'guard_cache': guard_pool.statistics() if guard_pool is not None else None,
                   'checkpoint_sha256': final_sha, 'final_validation_checkpoint_binding': binding,
                   'validation_history': history}
        write_json(output/'summary.json', summary)
        receipt.update(status='completed', finished_at_utc=utc_now(), summary_sha256=file_sha(output/'summary.json'))
        write_json(output/'run.json', receipt, replace=True)
        return summary
    except BaseException as exc:
        _failed(output, receipt, exc)
        raise
