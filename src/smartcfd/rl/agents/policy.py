"""Torch-only half-edge messages and a strictly masked discrete actor.

Each slot emits four local actions in slot-major order. No PPO value head is used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence
import torch
from torch import nn

OBSERVATION_VERSION = "halfedge_obs_v1"
ACTION_VERSION = "halfedge_actions_v1"
DATA_VERSION = "uiuc_v1"


@dataclass
class NetworkConfig:
    feature_dim: int = 24
    global_dim: int = 8
    hidden_dim: int = 128
    layers: int = 3
    local_actions: int = 4


def tensor_observation(observation: Mapping, device=None) -> dict[str, torch.Tensor]:
    result = {}
    for key in ("features", "next", "previous", "twin", "mask", "global"):
        dtype = torch.bool if key == "mask" else torch.int64 if key in ("next", "previous", "twin") else torch.float32
        result[key] = torch.as_tensor(observation[key], dtype=dtype, device=device)
    if result["features"].ndim == 2:
        result = {key: value.unsqueeze(0) for key, value in result.items()}
    if result["features"].ndim != 3:
        raise ValueError("features must be [H,F] or [B,H,F]")
    return result


def batch_observations(observations: Sequence[Mapping], device=None) -> dict[str, torch.Tensor]:
    if not observations:
        raise ValueError("Cannot batch zero observations")
    # Replay snapshots live on CPU. Stack there and transfer six batched tensors,
    # avoiding six tiny host-to-device transfers per individual sample.
    values = [tensor_observation(obs) for obs in observations]
    return {key: torch.cat([obs[key] for obs in values], dim=0).to(device=device) for key in values[0]}


def masked_distribution(logits: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Invalid probabilities and log-probabilities are exactly zero.

    The finite invalid log-probability convention prevents 0 * -inf in entropy
    and expectations. Empty masks raise; terminal rows must be removed first.
    """
    mask = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
    if logits.shape != mask.shape:
        raise ValueError(f"logit/mask shape mismatch: {logits.shape} vs {mask.shape}")
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("Policy distribution requested for a state with no legal action")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Nonfinite actor logits")
    masked = logits.masked_fill(~mask, -torch.inf)
    normalizer = torch.logsumexp(masked, dim=-1, keepdim=True)
    log_prob = torch.where(mask, masked - normalizer, torch.zeros_like(logits))
    prob = torch.exp(log_prob) * mask.to(logits.dtype)
    return prob, log_prob


def neighbor_features(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    b, h, d = features.shape
    if indices.shape != (b, h) or bool(((indices < -1) | (indices >= h)).any()):
        raise ValueError("Half-edge adjacency has invalid shape or out-of-range index")
    gathered = features.gather(1, indices.clamp_min(0).unsqueeze(-1).expand(b, h, d))
    return gathered * (indices >= 0).unsqueeze(-1)


class HalfEdgeEncoder(nn.Module):
    def __init__(self, config: NetworkConfig):
        super().__init__()
        if config.hidden_dim <= 0 or config.layers < 1 or config.local_actions != 4:
            raise ValueError("Require positive width, at least one layer, and four local actions")
        self.config = config
        width = config.hidden_dim
        self.input = nn.Linear(config.feature_dim, width)
        self.global_input = nn.Linear(config.global_dim, width)
        self.messages = nn.ModuleList(nn.Linear(4 * width, width) for _ in range(config.layers))
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(config.layers))

    def forward(self, observation: Mapping) -> torch.Tensor:
        obs = tensor_observation(observation, self.input.weight.device)
        x = obs["features"]
        if x.shape[-1] != self.config.feature_dim or obs["global"].shape[-1] != self.config.global_dim:
            raise ValueError("Observation widths differ from network configuration")
        valid = (obs["next"] >= 0).unsqueeze(-1)
        h = torch.nn.functional.gelu(self.input(x) + self.global_input(obs["global"]).unsqueeze(1)) * valid
        for linear, norm in zip(self.messages, self.norms):
            neighbors = [neighbor_features(h, obs[key]) for key in ("next", "previous", "twin")]
            message = torch.nn.functional.gelu(linear(torch.cat([h, *neighbors], dim=-1)))
            h = norm(h + message) * valid
        return h


class PolicyNetwork(nn.Module):
    def __init__(self, config: NetworkConfig | dict | None = None, **kwargs):
        super().__init__()
        if config is not None and kwargs:
            raise ValueError("Pass config or network fields, not both")
        self.config = config if isinstance(config, NetworkConfig) else NetworkConfig(**(config or kwargs))
        self.encoder = HalfEdgeEncoder(self.config)
        self.head = nn.Linear(self.config.hidden_dim, self.config.local_actions)

    def forward(self, observation: Mapping) -> torch.Tensor:
        return self.head(self.encoder(observation)).flatten(start_dim=1)

    def distribution(self, observation: Mapping) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self(observation)
        return masked_distribution(logits, tensor_observation(observation, logits.device)["mask"])

    def metadata(self):
        return {"network": asdict(self.config), "observation_version": OBSERVATION_VERSION,
                "action_version": ACTION_VERSION}


class ActionQNetwork(nn.Module):
    """Independent encoder/head with one Q value per discrete action."""
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.encoder = HalfEdgeEncoder(config)
        self.head = nn.Linear(config.hidden_dim, config.local_actions)

    def forward(self, observation: Mapping) -> torch.Tensor:
        return self.head(self.encoder(observation)).flatten(start_dim=1)
