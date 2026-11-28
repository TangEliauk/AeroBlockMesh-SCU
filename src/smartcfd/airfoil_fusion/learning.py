"""Packed masked categorical policy, with optional CH-SAC geometry and adaptive temperature."""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from smartcfd.airfoil_decomposition.learning import AirfoilSACAgent, HalfEdgeQueueEncoder, NETWORK
from smartcfd.rl.agents.policy import masked_distribution

_SHAPE = {"features": (256, 32), "next": (256,), "previous": (256,), "twin": (256,),
          "global": (12,), "geometry": (18,)}


def candidate_batch(observations, device):
    """Keep every candidate in order and pad only to the longest row in the batch."""
    capacity = max(len(item["mask"]) for item in observations)
    batch = {}
    for name, shape in _SHAPE.items():
        dtype = torch.long if name in {"next", "previous", "twin"} else torch.float32
        array = np.stack([item[name] for item in observations])
        if array.shape[1:] != shape:
            raise ValueError(f"Invalid {name} shape for topology observation")
        batch[name] = torch.as_tensor(array, dtype=dtype, device=device)
    for name, shape, dtype, fill in (("candidates", (16,), np.float32, 0),
                                      ("candidate_nodes", (2,), np.int64, -1),
                                      ("mask", (), np.bool_, False)):
        array = np.full((len(observations), capacity, *shape), fill, dtype=dtype)
        for row, item in enumerate(observations):
            count = len(item["mask"])
            source = item[name]
            if source.shape != (count, *shape) or source.dtype != dtype:
                raise ValueError(f"Invalid packed {name} shape or dtype")
            array[row, :count] = source
        batch[name] = torch.as_tensor(array, device=device)
    return batch


class _FusionEncoder(HalfEdgeQueueEncoder):
    def _prepare(self, observation):
        if isinstance(observation["features"], torch.Tensor):
            return observation
        return candidate_batch([observation], self.input.weight.device)


class FusionPolicy(nn.Module):
    def __init__(self, use_geometry):
        super().__init__()
        self.encoder = _FusionEncoder()
        width = NETWORK.hidden_dim
        self.geometry = (nn.Sequential(nn.Linear(18, width), nn.LayerNorm(width), nn.GELU())
                         if use_geometry else None)
        self.scorer = nn.Sequential(nn.Linear((6 if use_geometry else 5)*width + 17, width),
                                    nn.GELU(), nn.Linear(width, 1))

    def _logits(self, observation):
        obs, h, represented, context = self.encoder(observation)
        rows, actions = obs["mask"].nonzero(as_tuple=True)
        scores = h.new_zeros(obs["mask"].shape)
        if not len(actions):
            return scores, obs["mask"]
        endpoints = obs["candidate_nodes"][rows, actions]
        stop = (endpoints == -1).all(-1)
        is_edit = ~stop
        if bool(((endpoints < 0) | (endpoints >= h.shape[1]))[is_edit].any()):
            raise ValueError("Edit candidate endpoint outside represented graph")
        indices = endpoints.clamp_min(0)
        if not bool(represented[rows[is_edit, None], indices[is_edit]].all()):
            raise ValueError("Edit candidate endpoint references padding")
        corners = h[rows[:, None], indices].masked_fill(stop[:, None, None], 0)
        features = obs["candidates"][rows, actions]
        parts = [corners.flatten(1), context[rows], features, stop[:, None].to(h.dtype)]
        if self.geometry is not None:
            geometry = obs["geometry"]
            if not bool(torch.isfinite(geometry).all()):
                raise ValueError("Nonfinite geometry feature")
            parts.append(self.geometry(geometry)[rows])
        values = self.scorer(torch.cat(parts, dim=1)).squeeze(-1)
        return scores.index_put((rows, actions), values), obs["mask"]

    def forward(self, observation):
        return self._logits(observation)[0]

    def distribution(self, observation):
        scores, mask = self._logits(observation)
        return masked_distribution(scores, mask)


class FusionSAC(AirfoilSACAgent):
    def __init__(self, *, use_geometry=True, seed=42, device="cpu"):
        super().__init__(device=device, seed=seed)
        self.actor, self.q1, self.q2 = [FusionPolicy(use_geometry).to(self.device) for _ in range(3)]
        self.target_q1 = copy.deepcopy(self.q1).requires_grad_(False).eval()
        self.target_q2 = copy.deepcopy(self.q2).requires_grad_(False).eval()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.lr)
        self.q_optimizer = torch.optim.Adam([*self.q1.parameters(), *self.q2.parameters()], lr=self.config.lr)
        self.log_alpha = None
        self.alpha_optimizer = None
        if use_geometry:
            self.log_alpha = nn.Parameter(torch.tensor(math.log(self.config.alpha), device=self.device))
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.config.lr)

    def _batch_observations(self, observations):
        return candidate_batch(observations, self.device)

    def _entropy_temperature(self):
        return self.config.alpha if self.log_alpha is None else self.log_alpha.detach().exp()

    def update(self, transitions, update_actor=True):
        metrics = super().update(transitions, update_actor)
        if self.log_alpha is not None and update_actor:
            observed = self._batch_observations([transition.observation for transition in transitions])
            legal_counts = observed["mask"].sum(-1)
            eligible = legal_counts > 1
            if bool(eligible.any()):
                with torch.no_grad():
                    p, log_p = self.actor.distribution(observed)
                    entropy = -(p*log_p).sum(-1)
                    target = .5*legal_counts[eligible].float().log()
                    error = (entropy[eligible]-target).mean()
                self.alpha_optimizer.zero_grad(set_to_none=True)
                (self.log_alpha*error).backward()
                self.alpha_optimizer.step()
                with torch.no_grad():
                    self.log_alpha.clamp_(-12., 2.)
                metrics["target_entropy"] = float(target.mean())
                metrics["alpha"] = float(self._entropy_temperature())
        return metrics


def bc_update(agent, pairs):
    observations, target = zip(*pairs)
    _, log_p = agent.actor.distribution(agent._batch_observations(observations))
    indices = torch.as_tensor(target, dtype=torch.long, device=agent.device)
    loss = F.nll_loss(log_p, indices)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite behavior-cloning loss")
    agent.actor_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad = nn.utils.clip_grad_norm_(agent.actor.parameters(), agent.config.grad_clip, error_if_nonfinite=True)
    agent.actor_optimizer.step()
    return {"loss": float(loss.detach()), "gradient_before_clip": float(grad)}
