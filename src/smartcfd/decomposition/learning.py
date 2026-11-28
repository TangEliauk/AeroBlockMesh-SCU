"""Boundary message passing and masked SAC for orthogonal decomposition.

This is a small boundary/ring encoder with whole-queue pooling, not SplineCNN.
Only the numerical SAC update is inherited; airfoil checkpoint formats are not.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
import torch
from torch import nn

from smartcfd.rl.agents.policy import NetworkConfig, masked_distribution, neighbor_features, tensor_observation
from smartcfd.rl.agents.sac import DiscreteSACAgent, SACConfig, seed_everything
from . import ACTION_VERSION, ENVIRONMENT_VERSION, OBSERVATION_VERSION

CHECKPOINT_SCHEMA = 'orthogonal_decomposition_checkpoint_v1'
NETWORK = NetworkConfig(feature_dim=24, global_dim=8, hidden_dim=64, layers=2, local_actions=2)


class BoundaryEncoder(nn.Module):
    """Ring-local messages plus mean/max over every queued polygon's vertices."""
    def __init__(self, config: NetworkConfig):
        super().__init__()
        if config.feature_dim != 24 or config.global_dim != 8 or config.local_actions != 2:
            raise ValueError('Orthogonal observations require24 features,8 globals and2 vertex-axis actions')
        if config.hidden_dim < 1 or config.layers < 1:
            raise ValueError('Network width/layers must be positive')
        self.config = config
        width = config.hidden_dim
        self.input = nn.Linear(config.feature_dim, width)
        self.global_input = nn.Linear(config.global_dim, width)
        self.messages = nn.ModuleList(nn.Linear(3*width, width) for _ in range(config.layers))
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(config.layers))

    def forward(self, observation):
        obs = tensor_observation(observation, self.input.weight.device)
        if obs['features'].shape[-1] != 24 or obs['global'].shape[-1] != 8:
            raise ValueError('Orthogonal observation feature width mismatch')
        valid = obs['next'] >= 0
        if not bool(valid.any(dim=1).all()):
            raise ValueError('Encoder needs at least one represented boundary vertex')
        global_embedding = torch.nn.functional.gelu(self.global_input(obs['global']))
        h = torch.nn.functional.gelu(self.input(obs['features'])) * valid.unsqueeze(-1)
        for linear, norm in zip(self.messages, self.norms):
            joined = torch.cat([h, neighbor_features(h, obs['next']), neighbor_features(h, obs['previous'])], -1)
            h = norm(h + torch.nn.functional.gelu(linear(joined))) * valid.unsqueeze(-1)
        mean = h.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        maximum = h.masked_fill(~valid.unsqueeze(-1), -torch.inf).amax(1)
        slots = h.shape[1]
        return torch.cat([h, mean[:, None].expand(-1, slots, -1),
                          maximum[:, None].expand(-1, slots, -1),
                          global_embedding[:, None].expand(-1, slots, -1)], dim=-1)


class BoundaryPolicy(nn.Module):
    def __init__(self, config=NETWORK):
        super().__init__()
        self.encoder = BoundaryEncoder(config)
        # Nonlinear fusion lets queue/budget context change rankings between
        # vertices of the same axis, rather than adding one shared axis bias.
        self.head = nn.Sequential(nn.Linear(4*config.hidden_dim, config.hidden_dim),
                                  nn.GELU(), nn.Linear(config.hidden_dim, 2))

    def forward(self, observation):
        return self.head(self.encoder(observation)).flatten(start_dim=1)

    def distribution(self, observation):
        logits = self(observation)
        return masked_distribution(logits, tensor_observation(observation, logits.device)['mask'])


class BoundaryQ(BoundaryPolicy):
    """Independent Q encoder/head, with one scalar per flattened action."""


class BoundarySACAgent(DiscreteSACAgent):
    """Reuse exact discrete SAC math without old environment/model loaders."""
    def __init__(self, config=None, network_config=None, device='cpu', seed=42):
        self.config = config if isinstance(config, SACConfig) else SACConfig(**(config or {'alpha': .01}))
        self.network_config = network_config if isinstance(network_config, NetworkConfig) else NetworkConfig(**(network_config or asdict(NETWORK)))
        c = self.config
        if not (c.alpha >= 0 and 0 <= c.gamma <= 1 and 0 < c.tau <= 1 and c.lr > 0 and c.grad_clip > 0):
            raise ValueError('Invalid SAC hyperparameters')
        self.device, self.seed = torch.device(device), int(seed)
        seed_everything(self.seed)
        self.actor = BoundaryPolicy(self.network_config).to(self.device)
        self.q1 = BoundaryQ(self.network_config).to(self.device)
        self.q2 = BoundaryQ(self.network_config).to(self.device)
        self.target_q1 = copy.deepcopy(self.q1).requires_grad_(False).eval()
        self.target_q2 = copy.deepcopy(self.q2).requires_grad_(False).eval()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=c.lr)
        self.q_optimizer = torch.optim.Adam([*self.q1.parameters(), *self.q2.parameters()], lr=c.lr)
        self.action_rng = torch.Generator(device=self.device).manual_seed(self.seed)
        self.updates = self.actor_updates = 0
        self.environment_version = ENVIRONMENT_VERSION
        self.rng_restore_info = {'mode': 'fresh_orthogonal_agent', 'seed': self.seed}

    def state_dict(self, *args, **kwargs):
        raise ValueError('Use orthogonal inference checkpoint; airfoil state_dict format is forbidden')

    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        raise ValueError('Use load_inference_checkpoint; training resume is unsupported in this MVP')

    @classmethod
    def load(cls, *args, **kwargs):
        raise ValueError('Use load_inference_checkpoint; old airfoil checkpoints are incompatible')


def save_inference_checkpoint(agent, path, *, environment_config, dataset_sha256, source,
                              online_step, selection, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f'Refusing existing checkpoint: {path}')
    state = {'schema': CHECKPOINT_SCHEMA, 'inference_only': True, 'training_resume_supported': False,
        'environment_version': ENVIRONMENT_VERSION, 'observation_version': OBSERVATION_VERSION,
        'action_version': ACTION_VERSION, 'action_encoding': 'vertex_slot*2+axis;0=horizontal,1=vertical',
        'network_kind': 'boundary_ring_messages_all_queue_mean_max_nonlinear_v1', 'network_config': asdict(agent.network_config),
        'sac_config': asdict(agent.config), 'environment_config': environment_config,
        'dataset_sha256': dataset_sha256, 'source': source, 'seed': agent.seed,
        'online_step': int(online_step), 'selection': selection, 'updates': agent.updates,
        'actor_updates': agent.actor_updates,
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
    expected_network = asdict(network) if isinstance(network, NetworkConfig) else dict(network)
    expected_sac = asdict(sac_config) if isinstance(sac_config, SACConfig) else dict(sac_config)
    expected = {'schema': CHECKPOINT_SCHEMA, 'inference_only': True, 'training_resume_supported': False,
        'environment_version': ENVIRONMENT_VERSION, 'observation_version': OBSERVATION_VERSION,
        'action_version': ACTION_VERSION, 'action_encoding': 'vertex_slot*2+axis;0=horizontal,1=vertical',
        'network_kind': 'boundary_ring_messages_all_queue_mean_max_nonlinear_v1', 'network_config': expected_network,
        'sac_config': expected_sac, 'environment_config': environment_config,
        'dataset_sha256': dataset_sha256, 'source': source}
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f'Orthogonal inference checkpoint mismatch: {key}')
    agent = BoundarySACAgent(state['sac_config'], state['network_config'], device=device, seed=state['seed'])
    if set(state['networks']) != {'actor', 'q1', 'q2', 'target_q1', 'target_q2'}:
        raise ValueError('Checkpoint requires exactly five independent network states')
    for name, weights in state['networks'].items():
        getattr(agent, name).load_state_dict(weights, strict=True)
        getattr(agent, name).eval()
    agent.updates, agent.actor_updates = state['updates'], state['actor_updates']
    return agent, state
