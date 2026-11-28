"""BC/SAC orchestration, bounded replay, evaluation and local checkpoints."""
from __future__ import annotations
from pathlib import Path
from copy import deepcopy
import hashlib
import math
import random
import torch
from smartcfd.core.types import ENVIRONMENT_VERSION, Transition
from smartcfd.rl.agents.policy import DATA_VERSION
from smartcfd.rl.agents.sac import DiscreteSACAgent
from smartcfd.rl.imitation.pretrain import train_behavior_cloning
from smartcfd.rl.replay_buffer import ReplayBuffer, snapshot


class Trainer:
    def __init__(self, agent: DiscreteSACAgent, replay_capacity=20000,
                 replay_max_bytes=512 * 1024 ** 2, seed=None):
        self.agent = agent
        self.seed = agent.seed if seed is None else int(seed)
        self.rng = random.Random(self.seed)
        self.replay = ReplayBuffer(replay_capacity, replay_max_bytes, self.seed)
        self.online_steps = self.online_episodes = 0
        self.bc_result = None
        self.reset_failures = []
        self.environment_context = None
        self.actor_migration = None

    @staticmethod
    def _check_context(context):
        if not isinstance(context, dict) or set(context) != {'environment_version', 'env_config', 'mesh_spec', 'domain_config'}:
            raise ValueError('Checkpoint environment context must include version, env config, full mesh spec and domain config')
        if not all(isinstance(context[name], dict) for name in ('env_config', 'mesh_spec', 'domain_config')):
            raise ValueError('Checkpoint environment context configuration fields must be objects')
        if context['environment_version'] != ENVIRONMENT_VERSION or context['env_config'].get('environment_version') != ENVIRONMENT_VERSION:
            raise ValueError('Checkpoint environment version mismatch')
        required = {'intervals', 'protected_intervals', 'max_intervals', 'smoothing_steps', 'max_cells'}
        if set(context['mesh_spec']) != required or any(context['env_config'].get(key) != context['mesh_spec'][key] for key in required):
            raise ValueError('Environment context has inconsistent terminal mesh specification')
        if set(context['domain_config']) != {'farfield', 'stretch'}:
            raise ValueError('Environment context domain configuration is incomplete')
        return deepcopy(context)

    def bind_environment(self, env):
        """Bind replay provenance before it is populated in real environments.

        Simulator fixtures without a production contract remain usable for
        numerical tests. They cannot supply CLI production provenance.
        """
        context = getattr(env, 'environment_context', None)
        if context is None:
            if getattr(env, 'config', None) is not None:
                raise ValueError('Configured environment does not expose an environment_context')
            return None
        context = self._check_context(context)
        if self.environment_context is not None and self.environment_context != context:
            raise ValueError('Environment context mismatch; existing replay/critics cannot cross terminal mesh specifications')
        if self.environment_context is None:
            if len(self.replay):
                raise ValueError('Bind the environment context before adding replay transitions')
            self.environment_context = context
        return deepcopy(context)

    def train_bc(self, train_transitions, val_transitions, epochs=20, patience=5,
                 batch_size=64, seed=None, lr=None):
        self.bc_result = train_behavior_cloning(self.agent.actor, train_transitions, val_transitions,
            epochs=epochs, patience=patience, batch_size=batch_size,
            lr=self.agent.config.lr if lr is None else lr,
            grad_clip=self.agent.config.grad_clip, seed=self.seed if seed is None else seed,
            device=self.agent.device)
        self.agent.actor_optimizer = torch.optim.Adam(self.agent.actor.parameters(), lr=self.agent.config.lr)
        return self.bc_result

    def warmup_critic(self, transitions=None, steps=1000, batch_size=64):
        """Fit independent twin Qs to real replay while preserving the BC actor."""
        if steps < 0 or batch_size < 1:
            raise ValueError("Invalid critic warmup schedule")
        for record in transitions or []:
            self.replay.add(record)
        if steps and not len(self.replay):
            raise ValueError("Critic warmup needs real replay transitions")
        metrics = [self.agent.update_critic(self.replay.sample(min(batch_size, len(self.replay))))
                   for _ in range(steps)]
        return {"steps": steps, "actor_updated": False, "metrics": metrics,
                "replay_size": len(self.replay), "updates_total": self.agent.updates}

    critic_warmup = warmup_critic

    def train_online(self, env, total_steps, warmup_steps=1000, batch_size=64,
                     updates_per_step=1, seed=None, reset_options=None,
                     curriculum=(0.1, 0.3, 1.0), validation_callback=None,
                     evaluation_interval=5000):
        """Run one uninterrupted curriculum; callbacks must use a separate env.

        callback(trainer, global_online_step) runs at the evaluation interval and
        after the final actual step, once if those coincide. None disables it.
        """
        if total_steps < 0 or warmup_steps < 0 or batch_size < 1 or updates_per_step < 1:
            raise ValueError("Invalid online training schedule")
        if evaluation_interval is not None and evaluation_interval < 1:
            raise ValueError("evaluation_interval must be positive or None")
        self.bind_environment(env)
        run_seed = self.seed if seed is None else int(seed)
        fractions = tuple(curriculum) if curriculum is not None else (1.0,)
        if not fractions or any(not (0 < f <= 1) for f in fractions) or list(fractions) != sorted(fractions):
            raise ValueError("Curriculum fractions must be positive, ordered and <= 1")
        # Frozen three-stage plan: 20% of interactions on the easiest 10%,
        # 30% on the easiest 30%, and the remaining 50% on all training domains.
        stage_budgets = (0.2, 0.3, 0.5) if len(fractions) == 3 else tuple(1 / len(fractions) for _ in fractions)
        boundaries = [sum(stage_budgets[:i]) for i in range(1, len(stage_budgets))]
        domains = getattr(env, "domains", None)
        fixed_selection = bool(reset_options and ("sample_id" in reset_options or "domain_index" in reset_options))
        curriculum_applied = bool(domains is not None and not fixed_selection)
        pools, cursors = {}, {}
        attempted_samples, stepped_samples = set(), set()
        failures_at_start = len(self.reset_failures)
        episode_return, episode_steps = 0.0, 0
        episodes, updates = [], []
        validation_steps = []
        def reset_environment(step_index):
            progress = step_index / max(1, total_steps)
            phase = sum(progress >= boundary for boundary in boundaries)
            fraction = fractions[phase] if curriculum_applied else 1.0
            eligible = max(1, math.ceil(len(domains) * fraction)) if curriculum_applied else 1
            if phase not in pools:
                pools[phase] = list(range(eligible))
                self.rng.shuffle(pools[phase])
                cursors[phase] = 0
            for _attempt in range(eligible):
                options = dict(reset_options or {})
                if curriculum_applied:
                    cursor = cursors[phase]
                    if cursor and cursor % eligible == 0:
                        self.rng.shuffle(pools[phase])
                    options["domain_index"] = pools[phase][cursor % eligible]
                    cursors[phase] += 1
                episode_seed = run_seed + self.online_episodes
                sample_id = (getattr(domains[options["domain_index"]], "sample_id", None)
                             if domains is not None and "domain_index" in options else options.get("sample_id"))
                if sample_id is not None:
                    attempted_samples.add(sample_id)
                try:
                    obs, info = env.reset(seed=episode_seed, options=options)
                    if not bool(torch.as_tensor(obs["mask"], dtype=torch.bool).any()):
                        raise ValueError(info.get("reason") or "reset_has_no_legal_action")
                except ValueError as error:
                    failure = {"seed": episode_seed, "sample_id": sample_id, "reason": str(error),
                               "steps": 0, "return": 0.0, "success": False, "reset_failed": True,
                               "curriculum_fraction": fraction, "eligible_domains": eligible}
                    self.reset_failures.append(failure)
                    episodes.append(failure)
                    self.online_episodes += 1
                    continue
                return obs, info, episode_seed, fraction, eligible
            raise RuntimeError("All resets in the eligible curriculum pool failed; failures retained in trainer.reset_failures")

        observation, reset_info, episode_seed, episode_fraction, eligible_domains = reset_environment(0)
        for local_step in range(total_steps):
            legal = torch.as_tensor(observation["mask"], dtype=torch.bool).nonzero().flatten().tolist()
            if not legal:
                raise ValueError("Environment reset/continuation has no legal action")
            action = self.rng.choice(legal) if self.online_steps < warmup_steps else self.agent.act(observation)
            before = snapshot(observation)
            if reset_info.get("sample_id") is not None:
                stepped_samples.add(reset_info["sample_id"])
            next_observation, reward, terminated, truncated, info = env.step(action)
            self.replay.add(Transition(before, action, float(reward), next_observation,
                                       bool(terminated), bool(truncated)))
            self.online_steps += 1
            episode_steps += 1
            episode_return += float(reward)
            if self.online_steps >= warmup_steps and len(self.replay) >= batch_size:
                for _update in range(updates_per_step):
                    metrics = self.agent.update(self.replay.sample(batch_size))
                    metrics["online_step"] = self.online_steps
                    updates.append(metrics)
            observation = next_observation
            if terminated or truncated:
                episodes.append({"seed": episode_seed, "return": episode_return, "steps": episode_steps,
                                 "success": bool(info.get("success", False)), "terminated": bool(terminated),
                                 "truncated": bool(truncated), "reason": info.get("reason", info.get("failure_reason")),
                                 "sample_id": reset_info.get("sample_id"),
                                 "curriculum_fraction": episode_fraction, "eligible_domains": eligible_domains})
                self.online_episodes += 1
                episode_return, episode_steps = 0.0, 0
                if local_step + 1 < total_steps:
                    observation, reset_info, episode_seed, episode_fraction, eligible_domains = reset_environment(local_step + 1)
            interval_due = evaluation_interval is not None and self.online_steps % evaluation_interval == 0
            if validation_callback is not None and (interval_due or local_step + 1 == total_steps):
                validation_callback(self, self.online_steps)
                validation_steps.append(self.online_steps)
        return {"seed": run_seed, "steps_requested": total_steps, "online_steps_total": self.online_steps,
                "updates_total": self.agent.updates, "updates": updates, "episodes": episodes,
                "partial_episode_steps": episode_steps, "replay_size": len(self.replay),
                "replay_retained_bytes": self.replay.allocated_bytes,
                "replay_max_bytes": self.replay.max_bytes, "replay_total_added": self.replay.total_added,
                "replay_evicted": self.replay.total_evicted,
                "curriculum_applied": curriculum_applied, "curriculum_fractions": list(fractions),
                "curriculum_stage_budget_fractions": list(stage_budgets),
                "curriculum_scope": "Eligible prefix of the supplied training domains; phases start on episode resets. Coverage is eligibility, not a claim every geometry was visited.",
                "attempted_sample_ids": sorted(attempted_samples), "stepped_sample_ids": sorted(stepped_samples),
                "reset_failures": self.reset_failures[failures_at_start:],
                "validation_callback_steps": validation_steps}

    def evaluate(self, env, episodes=10, seed=10042, reset_options=None, max_steps=None):
        if episodes < 1:
            raise ValueError("Evaluation needs at least one episode")
        self.bind_environment(env)
        limit = max_steps or getattr(getattr(env, "config", None), "max_steps", 10000)
        if limit < 1:
            raise ValueError("Evaluation step limit must be positive")
        records = []
        for episode in range(episodes):
            observation, reset_info = env.reset(seed=seed + episode, options=reset_options)
            reward_sum, final_info = 0.0, {}
            terminated = truncated = False
            for step in range(1, limit + 1):
                action = self.agent.act(observation, deterministic=True)
                observation, reward, terminated, truncated, final_info = env.step(action)
                reward_sum += float(reward)
                if terminated or truncated:
                    break
            records.append({"seed": seed + episode, "sample_id": reset_info.get("sample_id"),
                            "return": reward_sum, "steps": step, "success": bool(final_info.get("success", False)),
                            "terminated": bool(terminated), "truncated": bool(truncated),
                            "evaluation_limit_reached": not bool(terminated or truncated),
                            "reason": final_info.get("reason", final_info.get("failure_reason"))})
        return {"episodes": records, "success_rate": sum(r["success"] for r in records) / episodes,
                "mean_return": sum(r["return"] for r in records) / episodes,
                "seed": seed, "deterministic": True}

    def save_checkpoint(self, path, data_version=DATA_VERSION, metadata=None, include_replay=True):
        metadata = deepcopy(metadata or {})
        declared_context = metadata.get('environment_context')
        if declared_context is not None:
            declared_context = self._check_context(declared_context)
            if self.environment_context is not None and self.environment_context != declared_context:
                raise ValueError('Checkpoint metadata environment context differs from the bound training environment')
            if self.environment_context is None and len(self.replay):
                raise ValueError('Bind the environment context before adding replay transitions; metadata cannot certify an existing unbound replay')
            self.environment_context = declared_context
        if self.environment_context is not None:
            metadata['environment_context'] = deepcopy(self.environment_context)
        if self.actor_migration is not None:
            metadata['actor_migration'] = deepcopy(self.actor_migration)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {"trainer_version": 2, "environment_version": ENVIRONMENT_VERSION,
                 "environment_context": deepcopy(self.environment_context),
                 "actor_migration": deepcopy(self.actor_migration),
                 "agent": self.agent.state_dict(data_version, metadata),
                 "trainer_seed": self.seed, "trainer_rng": self.rng.getstate(),
                 "online_steps": self.online_steps, "online_episodes": self.online_episodes,
                 "bc_result": self.bc_result,
                 "reset_failures": self.reset_failures,
                 "resume_scope": "Model, optimizer, replay and RNG; environment resets at next train_online call.",
                 "replay": self.replay.state_dict() if include_replay else None,
                 "replay_config": {"capacity": self.replay.capacity, "max_bytes": self.replay.max_bytes}}
        torch.save(state, path)

    save = save_checkpoint

    @classmethod
    def load_checkpoint(cls, path, device="cpu", expected_data_version=None,
                        expected_environment_version=ENVIRONMENT_VERSION, expected_environment_context=None):
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state.get('environment_version') != ENVIRONMENT_VERSION:
            raise ValueError('Trainer checkpoint environment version mismatch; legacy replay cannot resume in the current environment')
        if expected_environment_version is not None and state['environment_version'] != expected_environment_version:
            raise ValueError('Trainer checkpoint environment version mismatch')
        if state.get("trainer_version") != 2:
            raise ValueError("Trainer checkpoint schema version mismatch")
        context = state.get('environment_context')
        metadata_context = state['agent'].get('metadata', {}).get('environment_context')
        if context != metadata_context:
            raise ValueError('Trainer and agent checkpoint environment contexts disagree')
        if context is not None:
            context = cls._check_context(context)
        if expected_environment_context is not None:
            expected_environment_context = cls._check_context(expected_environment_context)
            if context != expected_environment_context:
                raise ValueError('Checkpoint environment context mismatch')
        agent = DiscreteSACAgent.from_state_dict(state["agent"], device, expected_data_version, expected_environment_version)
        trainer = cls(agent, state["replay_config"]["capacity"], state["replay_config"]["max_bytes"], state["trainer_seed"])
        trainer.rng.setstate(state["trainer_rng"])
        trainer.online_steps, trainer.online_episodes = state["online_steps"], state["online_episodes"]
        trainer.bc_result = state["bc_result"]
        trainer.reset_failures = state.get("reset_failures", [])
        trainer.environment_context = context
        trainer.actor_migration = deepcopy(state.get('actor_migration'))
        if state["replay"] is not None:
            trainer.replay.load_state_dict(state["replay"])
        return trainer

    @classmethod
    def from_actor_checkpoint(cls, path, device='cpu', seed=42, *, config=None,
                              network_config=None, replay_capacity=20000,
                              replay_max_bytes=512 * 1024 ** 2):
        """Explicit weight transfer, never a resumption of a legacy MDP.

        Only a compatible actor is copied. Critics, targets, optimizer states,
        replay, update counts and every RNG are initialized from the new seed.
        Local checkpoints are trusted torch serialization inputs.
        """
        from smartcfd.rl.agents.policy import ACTION_VERSION, OBSERVATION_VERSION
        source_path = Path(path)
        state = torch.load(source_path, map_location='cpu', weights_only=False)
        source = state.get('agent', state)
        if source.get('metadata', {}).get('test_used') is True:
            raise ValueError('A checkpoint marked test_used=True cannot seed further training through actor-only migration')
        if (source.get('observation_version') != OBSERVATION_VERSION
                or source.get('action_version') != ACTION_VERSION):
            raise ValueError('Actor-only migration observation/action schema mismatch')
        if 'actor' not in source or 'network_config' not in source:
            raise ValueError('Actor-only migration requires recorded actor weights and network configuration')
        agent = DiscreteSACAgent(config=config,
            network_config=source['network_config'] if network_config is None else network_config,
            device=device, seed=seed)
        agent.load_actor(source['actor'])
        trainer = cls(agent, replay_capacity, replay_max_bytes, seed)
        digest = hashlib.sha256()
        with source_path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        trainer.actor_migration = {'mode': 'actor_only',
            'source_sha256': digest.hexdigest(),
            'source_environment_version': source.get('environment_version', 'legacy_unspecified'),
            'source_checkpoint_version': source.get('checkpoint_version'),
            'destination_environment_version': ENVIRONMENT_VERSION, 'destination_seed': int(seed),
            'critics_optimizers_replay_rng_restored': False,
            'requires_current_environment_binding': True}
        return trainer

    load = load_checkpoint
