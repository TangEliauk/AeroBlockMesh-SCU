"""Candidate-action SAC for C-domain airfoil decomposition, without BC transfer.

The SAC equations are inherited. Observation batching, graph/candidate networks
and inference-only checkpoint contracts belong to this independent environment.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn

from smartcfd.rl.agents.policy import masked_distribution, neighbor_features
from smartcfd.rl.agents.sac import DiscreteSACAgent, SACConfig, seed_everything
from .config import ACTION_VERSION, ENVIRONMENT_VERSION, OBSERVATION_VERSION

CHECKPOINT_SCHEMA = 'airfoil_c_decomposition_checkpoint_v1'
NETWORK_KIND = 'halfedge_twin_messages_queue_pool_sparse_candidate_nonlinear_v1'
ACTION_ENCODING = 'environment_candidate_row; fixed1024 slots; cut or shared-edge insertion'
FIELDS = ('features', 'next', 'previous', 'twin', 'global', 'candidates', 'candidate_nodes', 'mask')
INDEX_FIELDS = ('next', 'previous', 'twin', 'candidate_nodes')


@dataclass(frozen=True)
class CandidateNetworkConfig:
    feature_dim: int = 32
    global_dim: int = 12
    candidate_dim: int = 16
    hidden_dim: int = 64
    layers: int = 2
    max_nodes: int = 256
    max_candidates: int = 1024


NETWORK = CandidateNetworkConfig()


def candidate_observation(observation: Mapping, device=None):
    """Preserve candidate fields and reject silent fractional-index coercion."""
    result = {}
    for key in FIELDS:
        value = torch.as_tensor(observation[key], device=device)
        if key in INDEX_FIELDS:
            if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                raise ValueError(f'{key} must contain integer indices')
            value = value.to(torch.int64)
        elif key == 'mask':
            if value.dtype != torch.bool:
                raise ValueError('Candidate mask must be boolean')
        else:
            value = value.to(torch.float32)
        result[key] = value
    if result['features'].ndim not in (2, 3):
        raise ValueError('features must be [256,32] or [B,256,32]')
    if result['features'].ndim == 2:
        result = {key: value.unsqueeze(0) for key, value in result.items()}
    expected = {'features': (256, 32), 'next': (256,), 'previous': (256,), 'twin': (256,),
                'global': (12,), 'candidates': (1024, 16), 'candidate_nodes': (1024, 2), 'mask': (1024,)}
    batch_size = result['features'].shape[0]
    if batch_size < 1 or any(tuple(result[key].shape) != (batch_size, *shape) for key, shape in expected.items()):
        raise ValueError('Airfoil candidate observation shape mismatch')
    return result


def batch_candidate_observations(observations: Sequence[Mapping], device=None):
    if not observations:
        raise ValueError('Cannot batch zero candidate observations')
    values = [candidate_observation(observation) for observation in observations]
    if any(value['features'].shape[0] != 1 for value in values):
        raise ValueError('Replay observations must each represent one environment state')
    return {key: torch.cat([value[key] for value in values], dim=0).to(device=device) for key in FIELDS}


class HalfEdgeQueueEncoder(nn.Module):
    """Two twin-aware graph layers and mean/max of the complete represented graph."""
    def __init__(self, config=NETWORK):
        super().__init__()
        if config != NETWORK:
            raise ValueError('This MVP fixes node32/global12/candidate16 and width64/two layers/256/1024 slots')
        self.config = config
        width = config.hidden_dim
        self.input = nn.Linear(config.feature_dim, width)
        self.global_input = nn.Linear(config.global_dim, width)
        self.messages = nn.ModuleList(nn.Linear(4*width, width) for _ in range(config.layers))
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(config.layers))

    def _prepare(self, observation):
        return candidate_observation(observation, self.input.weight.device)

    def forward(self, observation):
        obs = self._prepare(observation)
        valid = obs['next'] >= 0
        if not bool(valid.any(dim=1).all()):
            raise ValueError('Graph needs at least one represented halfedge')
        for key in ('next', 'previous', 'twin'):
            indices = obs[key]
            if bool(((indices < -1) | (indices >= self.config.max_nodes)).any()):
                raise ValueError(f'{key} graph index outside node slots')
            referred_valid = valid.gather(1, indices.clamp_min(0))
            if bool((valid & (indices >= 0) & ~referred_valid).any()):
                raise ValueError(f'{key} graph index references node padding')
            if key != 'twin' and bool((valid & (indices < 0)).any()):
                raise ValueError('Represented halfedges require a complete next/previous ring')
        features = obs['features'].masked_fill(~valid.unsqueeze(-1), 0.)
        if not bool(torch.isfinite(features).all() and torch.isfinite(obs['global']).all()):
            raise ValueError('Nonfinite represented node/global features')
        # Trim only suffix padding; node identities and candidate endpoints keep
        # their original indices. All rings, protected and completed, are kept.
        size = int(valid.nonzero()[:, 1].max())+1
        valid = valid[:, :size]
        h = torch.nn.functional.gelu(self.input(features[:, :size])) * valid.unsqueeze(-1)
        for linear, norm in zip(self.messages, self.norms):
            joined = torch.cat([h, *(neighbor_features(h, obs[key][:, :size].masked_fill(~valid, -1))
                                    for key in ('next', 'previous', 'twin'))], dim=-1)
            h = norm(h + torch.nn.functional.gelu(linear(joined))) * valid.unsqueeze(-1)
        mean = h.sum(1) / valid.sum(1, keepdim=True)
        maximum = h.masked_fill(~valid.unsqueeze(-1), -torch.inf).amax(1)
        global_embedding = torch.nn.functional.gelu(self.global_input(obs['global']))
        return obs, h, valid, torch.cat([mean, maximum, global_embedding], dim=-1)


class CandidatePolicy(nn.Module):
    def __init__(self, config=NETWORK):
        super().__init__()
        self.encoder = HalfEdgeQueueEncoder(config)
        width = config.hidden_dim
        self.head = nn.Sequential(nn.Linear(5*width+config.candidate_dim, width),
                                  nn.GELU(), nn.Linear(width, 1))

    def forward(self, observation):
        obs, embeddings, valid, context = self.encoder(observation)
        rows, actions = obs['mask'].nonzero(as_tuple=True)
        output = embeddings.new_zeros(obs['mask'].shape)
        if not len(actions):
            return output
        endpoints = obs['candidate_nodes'][rows, actions]
        if bool(((endpoints < 0) | (endpoints >= embeddings.shape[1])).any()):
            raise ValueError('Legal candidate endpoint index outside represented node slots')
        if not bool(valid[rows[:, None], endpoints].all()):
            raise ValueError('Legal candidate endpoint references node padding')
        features = obs['candidates'][rows, actions]
        if not bool(torch.isfinite(features).all()):
            raise ValueError('Nonfinite legal candidate features')
        joined = torch.cat([embeddings[rows, endpoints[:, 0]], embeddings[rows, endpoints[:, 1]],
                            context[rows], features], dim=-1)
        scores = self.head(joined).squeeze(-1)
        # This keeps arbitrary holes in the fixed action mask. Invalid slots do
        # no candidate MLP work and have finite zero Q/logits for SAC expectations.
        return output.index_put((rows, actions), scores)

    def distribution(self, observation):
        logits = self(observation)
        return masked_distribution(logits, candidate_observation(observation, logits.device)['mask'])


class CandidateQ(CandidatePolicy):
    """A separately constructed graph encoder/scalar candidate head."""


class AirfoilSACAgent(DiscreteSACAgent):
    def __init__(self, config=None, network_config=None, device='cpu', seed=42):
        self.config = config if isinstance(config, SACConfig) else SACConfig(**(config or {'alpha': .01}))
        self.network_config = network_config if isinstance(network_config, CandidateNetworkConfig) else CandidateNetworkConfig(**(network_config or {}))
        c = self.config
        if not (c.alpha >= 0 and 0 <= c.gamma <= 1 and 0 < c.tau <= 1 and c.lr > 0 and c.grad_clip > 0):
            raise ValueError('Invalid SAC hyperparameters')
        self.device, self.seed = torch.device(device), int(seed)
        seed_everything(self.seed)
        self.actor = CandidatePolicy(self.network_config).to(self.device)
        self.q1 = CandidateQ(self.network_config).to(self.device)
        self.q2 = CandidateQ(self.network_config).to(self.device)
        self.target_q1 = copy.deepcopy(self.q1).requires_grad_(False).eval()
        self.target_q2 = copy.deepcopy(self.q2).requires_grad_(False).eval()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=c.lr)
        self.q_optimizer = torch.optim.Adam([*self.q1.parameters(), *self.q2.parameters()], lr=c.lr)
        self.action_rng = torch.Generator(device=self.device).manual_seed(self.seed)
        self.updates = self.actor_updates = 0
        self.environment_version = ENVIRONMENT_VERSION
        self.rng_restore_info = {'mode': 'fresh_airfoil_candidate_agent', 'seed': self.seed}

    def _batch_observations(self, observations):
        return batch_candidate_observations(observations, self.device)

    def state_dict(self, *args, **kwargs):
        raise ValueError('Use new airfoil candidate inference checkpoint; legacy model schemas are forbidden')

    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        raise ValueError('Use load_inference_checkpoint; training resume is unsupported')

    @classmethod
    def load(cls, *args, **kwargs):
        raise ValueError('Use load_inference_checkpoint; old CFD/orthogonal model loaders are forbidden')


BoundarySACAgent = AirfoilSACAgent


def save_inference_checkpoint(agent, path, *, environment_config, dataset_sha256, source,
                              online_step, selection, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f'Refusing existing checkpoint: {path}')
    state = {'schema': CHECKPOINT_SCHEMA, 'inference_only': True, 'training_resume_supported': False,
        'environment_version': ENVIRONMENT_VERSION, 'observation_version': OBSERVATION_VERSION,
        'action_version': ACTION_VERSION, 'action_encoding': ACTION_ENCODING, 'network_kind': NETWORK_KIND,
        'network_config': asdict(agent.network_config), 'sac_config': asdict(agent.config),
        'environment_config': environment_config, 'dataset_sha256': dataset_sha256, 'source': source,
        'seed': agent.seed, 'online_step': int(online_step), 'selection': selection,
        'updates': agent.updates, 'actor_updates': agent.actor_updates,
        'networks': {name: {key: value.detach().cpu().clone() for key, value in getattr(agent, name).state_dict().items()}
                     for name in ('actor', 'q1', 'q2', 'target_q1', 'target_q2')}}
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('xb') as stream:
        torch.save(state, stream)
    temporary.replace(path)


def load_inference_checkpoint(path, *, environment_config, dataset_sha256, source,
                              sac_config, network_config=None, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=True)
    network = network_config or NETWORK
    expected_network = asdict(network) if isinstance(network, CandidateNetworkConfig) else dict(network)
    expected_sac = asdict(sac_config) if isinstance(sac_config, SACConfig) else dict(sac_config)
    expected = {'schema': CHECKPOINT_SCHEMA, 'inference_only': True, 'training_resume_supported': False,
        'environment_version': ENVIRONMENT_VERSION, 'observation_version': OBSERVATION_VERSION,
        'action_version': ACTION_VERSION, 'action_encoding': ACTION_ENCODING, 'network_kind': NETWORK_KIND,
        'network_config': expected_network, 'sac_config': expected_sac, 'environment_config': environment_config,
        'dataset_sha256': dataset_sha256, 'source': source}
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f'Airfoil candidate inference checkpoint mismatch: {key}')
    if set(state['networks']) != {'actor', 'q1', 'q2', 'target_q1', 'target_q2'}:
        raise ValueError('Checkpoint requires exactly five network states')
    agent = AirfoilSACAgent(state['sac_config'], state['network_config'], device=device, seed=state['seed'])
    for name, weights in state['networks'].items():
        getattr(agent, name).load_state_dict(weights, strict=True)
        getattr(agent, name).eval()
    agent.updates, agent.actor_updates = state['updates'], state['actor_updates']
    return agent, state
