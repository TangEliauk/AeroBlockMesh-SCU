"""Discrete SAC: exact masked expectations, independent twin Qs and targets."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Mapping, Sequence
import torch
from torch import nn

from smartcfd.core.types import ENVIRONMENT_VERSION, Transition
from .policy import (ACTION_VERSION, DATA_VERSION, OBSERVATION_VERSION, ActionQNetwork,
                     NetworkConfig, PolicyNetwork, batch_observations, tensor_observation)


@dataclass
class SACConfig:
    alpha: float = 0.1
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    grad_clip: float = 1.0


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


class DiscreteSACAgent:
    def __init__(self, config: SACConfig | dict | None = None,
                 network_config: NetworkConfig | dict | None = None, device="cpu", seed=42):
        self.config = config if isinstance(config, SACConfig) else SACConfig(**(config or {}))
        self.network_config = network_config if isinstance(network_config, NetworkConfig) else NetworkConfig(**(network_config or {}))
        if not (self.config.alpha >= 0 and 0 <= self.config.gamma <= 1 and 0 < self.config.tau <= 1
                and self.config.lr > 0 and self.config.grad_clip > 0):
            raise ValueError("Invalid SAC hyperparameters")
        self.device = torch.device(device)
        self.seed = int(seed)
        seed_everything(self.seed)
        self.actor = PolicyNetwork(self.network_config).to(self.device)
        self.q1 = ActionQNetwork(self.network_config).to(self.device)
        self.q2 = ActionQNetwork(self.network_config).to(self.device)
        self.target_q1 = copy.deepcopy(self.q1).requires_grad_(False).eval()
        self.target_q2 = copy.deepcopy(self.q2).requires_grad_(False).eval()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.lr)
        self.q_optimizer = torch.optim.Adam([*self.q1.parameters(), *self.q2.parameters()], lr=self.config.lr)
        self.action_rng = torch.Generator(device=self.device)
        self.action_rng.manual_seed(self.seed)
        self.rng_restore_info = {"mode": "seed_initialized", "device_type": self.device.type,
                                 "seed": self.seed, "sample_sequence_equivalent": False}
        self.updates = 0
        self.actor_updates = 0
        self.environment_version = ENVIRONMENT_VERSION

    @torch.no_grad()
    def act(self, observation: Mapping, deterministic=False) -> int:
        self.actor.eval()
        probabilities, _ = self.actor.distribution(observation)
        if probabilities.shape[0] != 1:
            raise ValueError("act expects one observation")
        if deterministic:
            return int(probabilities[0].argmax().item())
        return int(torch.multinomial(probabilities[0], 1, generator=self.action_rng).item())

    def _batch_observations(self, observations: Sequence[Mapping]) -> dict[str, torch.Tensor]:
        """Override only observation batching when reusing SAC with another MDP."""
        return batch_observations(observations, self.device)

    def _entropy_temperature(self):
        return self.config.alpha

    @torch.no_grad()
    def compute_targets(self, transitions: Sequence[Transition]) -> torch.Tensor:
        """Only termination suppresses bootstrap; truncation retains it.

        No actor/target network is evaluated for terminal rows, even when their
        next mask is empty. Nonterminal empty masks are contract violations.
        """
        if not transitions:
            raise ValueError("Cannot update an empty batch")
        result = torch.tensor([t.reward for t in transitions], device=self.device, dtype=torch.float32)
        alive = [i for i, t in enumerate(transitions) if not t.terminated]
        if alive:
            next_obs = self._batch_observations([transitions[i].next_observation for i in alive])
            probabilities, log_probabilities = self.actor.distribution(next_obs)
            minimum = torch.minimum(self.target_q1(next_obs), self.target_q2(next_obs))
            soft_value = (probabilities * (minimum - self._entropy_temperature() * log_probabilities)).sum(dim=-1)
            result[torch.tensor(alive, device=self.device)] += self.config.gamma * soft_value
        return result

    def actor_objective(self, observation: Mapping) -> tuple[torch.Tensor, torch.Tensor]:
        """Actor is evaluated at current s, not next s; both Qs are constants."""
        probabilities, log_probabilities = self.actor.distribution(observation)
        with torch.no_grad():
            minimum = torch.minimum(self.q1(observation), self.q2(observation))
        loss = (probabilities * (self._entropy_temperature() * log_probabilities - minimum)).sum(dim=-1).mean()
        entropy = -(probabilities * log_probabilities).sum(dim=-1).mean()
        return loss, entropy

    def update(self, transitions: Sequence[Transition], update_actor=True) -> dict[str, float]:
        current = self._batch_observations([t.observation for t in transitions])
        actions = torch.tensor([t.action for t in transitions], dtype=torch.int64, device=self.device)
        if bool(((actions < 0) | (actions >= current["mask"].shape[1])).any()):
            raise ValueError("Replay action index outside action space")
        if not bool(current["mask"].gather(1, actions[:, None]).all()):
            raise ValueError("Replay contains an action illegal in its recorded state")
        self.actor.train(mode=bool(update_actor))
        if not update_actor:
            self.actor_optimizer.zero_grad(set_to_none=True)
        self.q1.train()
        self.q2.train()
        targets = self.compute_targets(transitions)
        q1 = self.q1(current).gather(1, actions[:, None]).squeeze(1)
        q2 = self.q2(current).gather(1, actions[:, None]).squeeze(1)
        q_loss = nn.functional.mse_loss(q1, targets) + nn.functional.mse_loss(q2, targets)
        if not bool(torch.isfinite(q_loss)):
            raise FloatingPointError("Nonfinite critic loss")
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        q_parameters = [*self.q1.parameters(), *self.q2.parameters()]
        q_gradient = nn.utils.clip_grad_norm_(q_parameters, self.config.grad_clip, error_if_nonfinite=True)
        self.q_optimizer.step()

        if update_actor:
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss, entropy = self.actor_objective(current)
            if not bool(torch.isfinite(actor_loss)):
                raise FloatingPointError("Nonfinite actor loss")
            actor_loss.backward()
            actor_gradient = nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip, error_if_nonfinite=True)
            self.actor_optimizer.step()
            self.actor_updates += 1
        else:
            # Critic-only BC -> SAC warmup: no actor gradient/optimizer step.
            with torch.no_grad():
                actor_loss, entropy = self.actor_objective(current)
            actor_gradient = 0.0
        with torch.no_grad():
            for target, online in ((self.target_q1, self.q1), (self.target_q2, self.q2)):
                for target_parameter, parameter in zip(target.parameters(), online.parameters()):
                    target_parameter.lerp_(parameter, self.config.tau)
        self.updates += 1
        return {"q_loss": float(q_loss.detach()), "actor_loss": float(actor_loss.detach()),
                "entropy": float(entropy.detach()), "target_mean": float(targets.mean()),
                "q_grad_norm_before_clip": float(q_gradient), "actor_grad_norm_before_clip": float(actor_gradient),
                "alpha": float(self._entropy_temperature()), "updates": self.updates,
                "actor_updated": bool(update_actor), "actor_updates": self.actor_updates}

    def update_critic(self, transitions: Sequence[Transition]) -> dict[str, float]:
        return self.update(transitions, update_actor=False)

    def load_actor(self, actor_state_dict: dict):
        """Exact BC actor transfer. Critics keep their independent initialization."""
        self.actor.load_state_dict(actor_state_dict, strict=True)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.lr)

    def state_dict(self, data_version=DATA_VERSION, metadata=None):
        return {"checkpoint_version": 2, "environment_version": self.environment_version,
                "observation_version": OBSERVATION_VERSION,
                "action_version": ACTION_VERSION, "data_version": data_version,
                "sac_config": asdict(self.config), "network_config": asdict(self.network_config),
                "seed": self.seed, "formal_seeds": [42, 43, 44], "updates": self.updates,
                "actor_updates": self.actor_updates,
                "actor": self.actor.state_dict(), "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
                "target_q1": self.target_q1.state_dict(), "target_q2": self.target_q2.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(), "q_optimizer": self.q_optimizer.state_dict(),
                "action_rng": self.action_rng.get_state(), "torch_rng": torch.get_rng_state(),
                "action_rng_device_type": self.device.type,
                "rng_restore_info": dict(self.rng_restore_info),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "metadata": dict(metadata or {})}

    def save(self, path, data_version=DATA_VERSION, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(data_version, metadata), path)

    @classmethod
    def from_state_dict(cls, state, device="cpu", expected_data_version=None,
                        expected_environment_version=ENVIRONMENT_VERSION):
        if state.get('environment_version') != ENVIRONMENT_VERSION:
            raise ValueError('Checkpoint environment version mismatch; legacy state requires explicit actor-only migration')
        if expected_environment_version is not None and state['environment_version'] != expected_environment_version:
            raise ValueError('Checkpoint environment version mismatch')
        if (state.get("checkpoint_version") != 2 or state.get("observation_version") != OBSERVATION_VERSION
                or state.get("action_version") != ACTION_VERSION):
            raise ValueError("Checkpoint observation/action/schema version mismatch")
        if expected_data_version is not None and state.get("data_version") != expected_data_version:
            raise ValueError("Checkpoint data version mismatch")
        agent = cls(state["sac_config"], state["network_config"], device, state["seed"])
        for name in ("actor", "q1", "q2", "target_q1", "target_q2"):
            getattr(agent, name).load_state_dict(state[name], strict=True)
        agent.actor_optimizer.load_state_dict(state["actor_optimizer"])
        agent.q_optimizer.load_state_dict(state["q_optimizer"])
        saved_rng = state["action_rng"].cpu()
        source_backend = state.get("action_rng_device_type")
        inferred_backend = source_backend is None
        if inferred_backend:
            # Backward compatibility with the first local smoke checkpoints.
            # CPU and CUDA generators use different state sizes/algorithms.
            cpu_size = torch.Generator(device="cpu").get_state().numel()
            source_backend = "cpu" if saved_rng.numel() == cpu_size else "cuda"
        if source_backend == agent.device.type:
            agent.action_rng.set_state(saved_rng)
            agent.rng_restore_info = {"mode": "exact_state_restored", "source_device_type": source_backend,
                "device_type": agent.device.type, "source_backend_inferred": inferred_backend,
                "sample_sequence_equivalent": True}
        else:
            agent.action_rng.manual_seed(agent.seed)
            agent.rng_restore_info = {"mode": "cross_backend_reseeded", "source_device_type": source_backend,
                "device_type": agent.device.type, "source_backend_inferred": inferred_backend,
                "seed": agent.seed, "sample_sequence_equivalent": False,
                "note": "CPU/CUDA action generators are different algorithms; seed initialization on the target backend does not preserve the prior sample sequence."}
        torch.set_rng_state(state["torch_rng"].cpu())
        if torch.cuda.is_available() and state.get("cuda_rng"):
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
        agent.updates = state["updates"]
        agent.actor_updates = state.get("actor_updates", state["updates"])
        return agent

    @classmethod
    def load(cls, path, device="cpu", expected_data_version=None,
             expected_environment_version=ENVIRONMENT_VERSION):
        # Local training checkpoints contain optimizer/RNG metadata and must be trusted.
        state = torch.load(path, map_location='cpu', weights_only=False)
        return cls.from_state_dict(state, device, expected_data_version, expected_environment_version)
