"""Build imitation data by replaying actions in the real environment.

JSONL episode observations and rewards are never accepted as truth. Each episode
contains actions, seed and reset options; all Transition fields come from step().
"""
from __future__ import annotations

import json
from dataclasses import asdict
from numbers import Integral
from pathlib import Path
from typing import Iterable
from torch.utils.data import Dataset
from smartcfd.core.types import Transition
from smartcfd.rl.replay_buffer import snapshot


class DemonstrationDataset(Dataset):
    def __init__(self, transitions: Iterable[Transition]):
        self.transitions = [snapshot(record) for record in transitions]
        for record in self.transitions:
            if not isinstance(record, Transition):
                raise TypeError("Demonstrations must be replayed Transition objects")
            mask = record.observation["mask"]
            if record.action < 0 or record.action >= len(mask) or not bool(mask[record.action]):
                raise ValueError("Demonstration action is illegal in its replayed observation")

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, index):
        record = self.transitions[index]
        return snapshot(record.observation), record.action


def rollout_episode(env, actions, seed=None, options=None, require_success=True) -> list[Transition]:
    observation, _ = env.reset(seed=seed, options=options)
    transitions = []
    ended = False
    final_info = {}
    for action in actions:
        if ended:
            raise ValueError("Demonstration has actions after episode termination")
        if not isinstance(action, Integral) or isinstance(action, bool):
            raise ValueError("JSONL actions must be integer flattened action IDs")
        action = int(action)
        mask = observation["mask"]
        if action < 0 or action >= len(mask) or not bool(mask[action]):
            raise ValueError(f"Illegal demonstration action {action}")
        before = snapshot(observation)
        observation, reward, terminated, truncated, final_info = env.step(action)
        transitions.append(Transition(before, action, float(reward), snapshot(observation),
                                      bool(terminated), bool(truncated)))
        ended = bool(terminated or truncated)
    if not transitions:
        raise ValueError("Demonstration episode contains no transition")
    if require_success and not (transitions[-1].terminated and final_info.get("success", False)):
        raise ValueError("Expert demonstration did not replay to a successful terminal state")
    return transitions


def validate_audited_episode(record, env, expected_context=None):
    """Validate identity and replay context; human approval is a separate layer.

    This function never treats a caller-provided reviewed flag as approval.
    The package/review loader must authenticate that decision independently.
    """
    from smartcfd.pipeline.audit_shared import EPISODE_SCHEMA, canonical_hash, episode_id_for
    from smartcfd.core.types import ENVIRONMENT_VERSION
    from smartcfd.pipeline.data import data_version, repository_root
    from smartcfd.partition.actions import ACTION_SCHEMA
    from smartcfd.rl.state import OBSERVATION_SCHEMA
    if not isinstance(record, dict) or record.get('schema') != EPISODE_SCHEMA:
        raise ValueError('Strict loading requires audited_episode_v2 records; legacy actions must be explicitly re-audited')
    required = ('split', 'sample_id', 'actions', 'seed', 'options', 'env_config', 'domain_config',
                'mesh_spec', 'observation_version', 'action_version', 'mapping_version', 'data',
                'geometry_sha256', 'episode_id', 'auto_passed', 'source', 'origin', 'environment_version')
    if any(key not in record for key in required):
        raise ValueError('Audited episode is missing required provenance fields')
    if record['auto_passed'] is not True or record['split'] not in ('train', 'val'):
        raise ValueError('Only auto-passed train/val episodes can be replayed for imitation')
    membership = json.loads((repository_root() / f"data/splits/uiuc_v1/{record['split']}.json").read_text(encoding='utf-8'))
    if record['sample_id'] not in membership:
        raise ValueError('Audited split conflicts with frozen membership')
    if record['source'] not in ('rule_synthetic', 'external_unverified'):
        raise ValueError('Audited source must preserve synthetic or unverified origin')
    if record['episode_id'] != episode_id_for(record):
        raise ValueError('Audited episode identity hash mismatch')
    if record['observation_version'] != OBSERVATION_SCHEMA or record['action_version'] != ACTION_SCHEMA:
        raise ValueError('Audited observation/action version mismatch')
    if record['environment_version'] != ENVIRONMENT_VERSION:
        raise ValueError('Audited environment version mismatch')
    if not isinstance(record['options'], dict) or record['options'] != {'sample_id': record['sample_id']}:
        raise ValueError('Audited sample_id and reset options disagree')
    if not isinstance(record['actions'], list):
        raise ValueError('Audited actions must be a list')
    domains = [domain for domain in env.domains if domain.sample_id == record['sample_id']]
    if len(domains) != 1:
        raise ValueError('Audited sample must select exactly one supplied domain')
    domain = domains[0]
    expected = {'env_config': asdict(env.config), 'environment_version': env.config.environment_version,
        'mesh_spec': asdict(env.config.mesh_spec),
        'domain_config': {'farfield': float(domain.farfield), 'stretch': float(domain.stretch)},
        'geometry_sha256': canonical_hash(domain.polygon),
        'mapping_version': domain.metadata.get('mapping_version'),
        'data': (expected_context or {}).get('data', None) or data_version()}
    for key, value in (expected_context or {}).items():
        if key in expected and canonical_hash(value) != canonical_hash(expected[key]):
            raise ValueError(f'Expected {key} context disagrees with replay environment')
        expected[key] = value
    for key, value in expected.items():
        if key not in record or canonical_hash(record[key]) != canonical_hash(value):
            raise ValueError(f'Audited {key} mismatch')


def replay_audited_episode(record, env, expected_context=None) -> list[Transition]:
    validate_audited_episode(record, env, expected_context)
    return rollout_episode(env, record['actions'], record['seed'], record['options'], require_success=True)


def load_replayed_demonstrations(path, env_factory, require_success=True, *, strict=False,
                                 expected_context=None) -> list[Transition]:
    transitions = []
    with Path(path).open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record.get("actions"), list):
                raise ValueError(f"Episode line {lineno} needs an actions list")
            options = dict(record.get("options") or {})
            if record.get("sample_id") is not None:
                if options.get('sample_id', record['sample_id']) != record['sample_id']:
                    raise ValueError(f'Episode line {lineno} sample_id and reset options conflict')
                options.setdefault("sample_id", record["sample_id"])
            env = env_factory()
            try:
                if record.get('env_config') is not None:
                    from smartcfd.core.types import EnvConfig
                    from smartcfd.pipeline.audit_shared import canonical_hash
                    declared = asdict(EnvConfig(**record['env_config']))
                    if not hasattr(env, 'config') or canonical_hash(declared) != canonical_hash(asdict(env.config)):
                        raise ValueError('Declared demonstration env_config does not match replay environment')
                if strict:
                    validate_audited_episode(record, env, expected_context)
                elif record.get('schema') not in (None, 'audited_episode_v2'):
                    raise ValueError('Legacy audited episodes require explicit environment migration and re-audit')
                transitions.extend(rollout_episode(env, record["actions"], record.get("seed"), options,
                                                 True if strict else require_success))
            except Exception as error:
                raise ValueError(f"Could not replay demonstration line {lineno}: {error}") from error
            finally:
                if hasattr(env, "close"):
                    env.close()
    return transitions


load_demonstrations = load_replayed_demonstrations
