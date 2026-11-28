"""Numerical contracts for masking, discrete SAC, BC transfer and replay."""
from __future__ import annotations

import copy
import json
import math
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from smartcfd.core.types import ENVIRONMENT_VERSION, EnvConfig, Transition
from smartcfd.rl.agents.policy import NetworkConfig, PolicyNetwork, masked_distribution
from smartcfd.rl.agents.sac import DiscreteSACAgent
from smartcfd.rl.imitation.dataset import load_replayed_demonstrations, rollout_episode
from smartcfd.rl.imitation.pretrain import evaluate_bc
from smartcfd.rl.replay_buffer import ReplayBuffer
from smartcfd.rl.trainer import Trainer


@pytest.fixture(scope="module", autouse=True)
def small_cpu_runtime():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def observation(mask_actions=(0, 1), value=0.0, h=8):
    features = np.zeros((h, 24), dtype=np.float32)
    features[:4, 0] = value
    neighbor = np.full(h, -1, dtype=np.int64)
    previous = neighbor.copy()
    neighbor[:4] = [1, 2, 3, 0]
    previous[:4] = [3, 0, 1, 2]
    twin = np.full(h, -1, dtype=np.int64)
    twin[0], twin[2] = 2, 0
    mask = np.zeros(h * 4, dtype=bool)
    mask[list(mask_actions)] = True
    return {"features": features, "next": neighbor, "previous": previous, "twin": twin,
            "mask": mask, "global": np.full(8, value, dtype=np.float32)}


def agent(seed=42):
    return DiscreteSACAgent(network_config={"hidden_dim": 16, "layers": 1}, seed=seed)


def constant_heads(model, values):
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.head.bias.copy_(torch.tensor(values, dtype=torch.float32))


def test_masked_distribution_exact_zeros_finite_entropy_and_gradients():
    logits = torch.tensor([[0., math.log(3), 1e30, -1e30]], requires_grad=True)
    mask = torch.tensor([[True, True, False, False]])
    probabilities, log_probabilities = masked_distribution(logits, mask)
    assert torch.allclose(probabilities, torch.tensor([[.25, .75, 0., 0.]]))
    assert torch.equal(probabilities[~mask], torch.zeros(2))
    assert torch.isfinite(log_probabilities).all()
    entropy = -(probabilities * log_probabilities).sum()
    entropy.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.equal(logits.grad[~mask], torch.zeros(2))
    with pytest.raises(ValueError, match="no legal"):
        masked_distribution(logits.detach(), torch.zeros_like(mask))


def test_halfedge_message_passing_is_slot_permutation_equivariant():
    policy = PolicyNetwork(hidden_dim=16, layers=2)
    obs = observation(tuple(range(16)))
    obs["features"][:4] = np.random.default_rng(1).normal(size=(4, 24))
    permutation = np.array([2, 0, 3, 1, 4, 5, 6, 7])
    inverse = np.argsort(permutation)
    other = copy.deepcopy(obs)
    other["features"] = obs["features"][permutation]
    other["mask"] = obs["mask"].reshape(8, 4)[permutation].reshape(-1)
    for key in ("next", "previous", "twin"):
        old = obs[key][permutation]
        other[key] = np.where(old >= 0, inverse[np.maximum(old, 0)], -1)
    with torch.no_grad():
        first = policy(obs).reshape(8, 4)
        second = policy(other).reshape(8, 4)
    assert torch.allclose(second, first[permutation], atol=1e-6)


def test_sac_targets_exact_full_expectation_terminal_skip_and_truncation_bootstrap():
    model = agent()
    constant_heads(model.actor, [0, math.log(3), 0, 0])
    constant_heads(model.target_q1, [2, 4, 0, 0])
    constant_heads(model.target_q2, [3, 5, 0, 0])
    current = observation((2,))
    next_obs = observation((0, 1), value=1)
    terminal = observation(())
    terminal["features"][:] = np.nan  # Must never reach any next-state network.
    records = [Transition(current, 2, 1, terminal, True),
               Transition(current, 2, 1, next_obs, False, True)]
    targets = model.compute_targets(records)
    entropy = -(.25 * math.log(.25) + .75 * math.log(.75))
    expected = 1 + .99 * (.25 * 2 + .75 * 4 + .1 * entropy)
    assert targets[0].item() == 1.0
    assert targets[1].item() == pytest.approx(expected, abs=1e-6)
    constant_heads(model.q1, [2, 4, 6, 0])
    constant_heads(model.q2, [3, 5, 7, 0])
    objective, current_entropy = model.actor_objective(current)
    assert objective.item() == pytest.approx(-6)
    assert current_entropy.item() == 0  # Current mask is action 2 only, unlike next mask.


def test_sac_update_finite_independent_q_and_polyak_target():
    model = agent()
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(model.q1.parameters(), model.q2.parameters()))
    old = {name: parameter.detach().clone() for name, parameter in model.target_q1.named_parameters()}
    records = [Transition(observation(value=i), i % 2, float(i), observation(value=i + 1), False)
               for i in range(4)]
    metrics = model.update(records)
    assert all(math.isfinite(value) for value in metrics.values())
    assert model.updates == 1
    for name, parameter in model.target_q1.named_parameters():
        expected = old[name] * (1 - model.config.tau) + dict(model.q1.named_parameters())[name] * model.config.tau
        assert torch.allclose(parameter, expected, atol=1e-7)
        assert not parameter.requires_grad


def test_bc_reduces_nll_exact_actor_transfer_and_never_reuses_ppo_value():
    model = agent()
    trainer = Trainer(model)
    records = [Transition(observation(value=.1 * i), 1, 0, observation(()), True) for i in range(6)]
    critic_before = copy.deepcopy(model.q1.state_dict())
    before = evaluate_bc(model.actor, records)["loss"]
    trainer.train_bc(records[:4], records[4:], epochs=10, patience=5, batch_size=2, lr=.01)
    after = evaluate_bc(model.actor, records)["loss"]
    assert after < before
    assert all(torch.equal(value, model.q1.state_dict()[key]) for key, value in critic_before.items())
    fresh = agent(seed=43)
    with torch.no_grad():
        expected = model.actor.distribution(records[0].observation)[0]
    fresh.load_actor(model.actor.state_dict())
    with torch.no_grad():
        actual = fresh.actor.distribution(records[0].observation)[0]
    assert torch.equal(actual, expected)
    assert not hasattr(fresh.actor, "value_head")


def test_replay_snapshots_cpu_arrays_and_enforces_both_caps():
    obs = observation()
    record = Transition(obs, 0, 1, observation(()), True)
    replay = ReplayBuffer(capacity=2, max_bytes=1_000_000)
    replay.add(record)
    obs["features"][:] = 99
    sampled = replay.sample(1)[0]
    assert sampled.observation["features"][0, 0] == 0
    sampled.observation["features"][:] = 77
    assert replay.sample(1)[0].observation["features"][0, 0] == 0
    estimated = replay.allocated_bytes
    memory_limited = ReplayBuffer(capacity=100, max_bytes=2 * estimated)
    for _ in range(5):
        memory_limited.add(record)
        replay.add(record)
    assert len(replay) == 2
    assert len(memory_limited) <= 2
    assert memory_limited.allocated_bytes <= memory_limited.max_bytes
    assert memory_limited.total_evicted > 0
    with pytest.raises(ValueError, match="memory cap"):
        ReplayBuffer(max_bytes=1).add(record)


class CounterEnvironment:
    """Small explicit simulator fixture; runtime demonstrations still call step."""
    def reset(self, seed=None, options=None):
        self.count = 0
        return observation((0,), value=0), {"sample_id": "counter_fixture"}

    def step(self, action):
        self.count += 1
        terminal = self.count == 2
        return observation(() if terminal else (0,), value=self.count), float(3 + self.count), terminal, False, {"success": terminal}


def test_jsonl_imitation_uses_replayed_rewards_states_and_rejects_illegal_actions(tmp_path):
    path = tmp_path / "episodes.jsonl"
    path.write_text(json.dumps({"seed": 42, "actions": [0, 0], "reward": 999, "observations": "forged"}) + "\n")
    records = load_replayed_demonstrations(path, CounterEnvironment)
    assert [record.reward for record in records] == [4, 5]
    assert records[0].observation["features"][0, 0] == 0
    assert records[0].next_observation["features"][0, 0] == 1
    assert records[-1].terminated
    with pytest.raises(ValueError, match="Illegal"):
        rollout_episode(CounterEnvironment(), [1])
    with pytest.raises(ValueError, match="successful"):
        rollout_episode(CounterEnvironment(), [0])


def test_checkpoint_reload_preserves_probabilities_replay_and_action_rng(tmp_path):
    model = agent()
    trainer = Trainer(model, replay_capacity=5)
    records = rollout_episode(CounterEnvironment(), [0, 0])
    for record in records:
        trainer.replay.add(record)
    model.update(records)
    path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(path, data_version="fixture_v1", metadata={"purpose": "numerical regression"})
    with torch.no_grad():
        probabilities = model.actor.distribution(observation())[0]
    expected_actions = [model.act(observation()) for _ in range(8)]
    loaded = Trainer.load_checkpoint(path, expected_data_version="fixture_v1")
    with torch.no_grad():
        reloaded = loaded.agent.actor.distribution(observation())[0]
    assert torch.equal(probabilities, reloaded)
    assert [loaded.agent.act(observation()) for _ in range(8)] == expected_actions
    assert len(loaded.replay) == 2
    assert loaded.agent.updates == 1
    with pytest.raises(ValueError, match="data version"):
        Trainer.load_checkpoint(path, expected_data_version="wrong_v2")


def test_online_trainer_records_actual_transitions_and_updates():
    trainer = Trainer(agent())
    result = trainer.train_online(CounterEnvironment(), total_steps=6, warmup_steps=2, batch_size=2)
    assert result["replay_total_added"] == 6
    assert result["updates_total"] > 0
    assert len(result["episodes"]) == 3
    assert all(record.reward in (4, 5) for record in trainer.replay.sample(6))


def test_critic_only_warmup_leaves_actor_parameters_and_probabilities_exactly_unchanged():
    trainer = Trainer(agent())
    # Mimic a gradient left behind by the final BC minibatch.
    next(trainer.agent.actor.parameters()).grad = torch.ones_like(next(trainer.agent.actor.parameters()))
    actor_before = copy.deepcopy(trainer.agent.actor.state_dict())
    q_before = copy.deepcopy(trainer.agent.q1.state_dict())
    with torch.no_grad():
        probabilities = trainer.agent.actor.distribution(observation())[0].clone()
    result = trainer.warmup_critic(rollout_episode(CounterEnvironment(), [0, 0]), steps=3, batch_size=2)
    assert result["actor_updated"] is False
    assert trainer.agent.actor_updates == 0
    assert all(parameter.grad is None for parameter in trainer.agent.actor.parameters())
    assert all(torch.equal(value, trainer.agent.actor.state_dict()[key]) for key, value in actor_before.items())
    assert any(not torch.equal(value, trainer.agent.q1.state_dict()[key]) for key, value in q_before.items())
    with torch.no_grad():
        after = trainer.agent.actor.distribution(observation())[0]
    assert torch.equal(probabilities, after)


class DomainCounterEnvironment(CounterEnvironment):
    def __init__(self):
        self.domains = [SimpleNamespace(sample_id=f"domain_{i}") for i in range(30)]

    def reset(self, seed=None, options=None):
        self.count = 0
        index = options["domain_index"]
        self.selected = self.domains[index].sample_id
        info = {"sample_id": self.selected}
        if index == 0:
            info["reason"] = "unmeshable_fixture"
        return observation(() if index == 0 else (0,)), info


def test_curriculum_expands_eligible_pool_and_preserves_failed_geometries():
    trainer = Trainer(agent())
    result = trainer.train_online(DomainCounterEnvironment(), total_steps=18, warmup_steps=100, batch_size=2)
    successful_resets = [record for record in result["episodes"] if not record.get("reset_failed")]
    assert {record["eligible_domains"] for record in successful_resets} == {3, 9, 30}
    assert [record["eligible_domains"] for record in successful_resets] == [3, 3, 9, 9, 9, 30, 30, 30, 30]
    assert result["curriculum_stage_budget_fractions"] == [.2, .3, .5]
    assert result["reset_failures"]
    assert any(failure["sample_id"] == "domain_0" for failure in result["reset_failures"])
    assert len(trainer.replay) == 18  # Every actual step, including failure outcomes, is retained.
    assert len(trainer.agent.actor.state_dict()) > 0


def run_real_environment_integration(output_dir):
    """Small runtime receipt using frozen train/validation geometry, no test fit."""
    from pathlib import Path
    from smartcfd.baselines.rule_partition import solve
    from smartcfd.core.types import EnvConfig, Failure
    from smartcfd.geometry.domain import make_domain
    from smartcfd.pipeline.config import write_json
    from smartcfd.pipeline.data import data_version, load_samples, repository_root
    from smartcfd.rl.env import BlockEnv

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_sample = next(sample for sample in load_samples("train") if "e655" in sample["aliases"])
    validation_sample = next(sample for sample in load_samples("val") if "fx74130wp1" in sample["aliases"])
    test_ids = set(json.loads((repository_root() / "data/splits/uiuc_v1/test.json").read_text()))
    assert training_sample["sample_id"] not in test_ids
    assert validation_sample["sample_id"] not in test_ids
    assert training_sample["sample_id"] != validation_sample["sample_id"]
    config = EnvConfig(max_steps=16)
    train_env = BlockEnv([make_domain(training_sample)], config)
    val_env = BlockEnv([make_domain(validation_sample)], config)
    episodes = []
    def replay(env, seed, label):
        env.reset(seed=seed)
        actions = solve(env, seed=seed)
        assert not isinstance(actions, Failure), getattr(actions, "reason", "teacher failed")
        records = rollout_episode(env, actions, seed=seed)
        episodes.append({"sample_id": env.domain.sample_id, "seed": seed, "actions": actions,
                         "options": {"sample_id": env.domain.sample_id}, "split": label,
                         "source": "rule_synthetic_environment_replay", "success": True})
        return records
    train_records = replay(train_env, 42, "train") + replay(train_env, 44, "train")
    val_records = replay(val_env, 10042, "val")
    (output_dir / "episodes.jsonl").write_text("\n".join(json.dumps(row) for row in episodes) + "\n")
    trainer = Trainer(agent(seed=42))
    trainer.bind_environment(train_env)
    before = evaluate_bc(trainer.agent.actor, val_records)
    bc_result = trainer.train_bc(train_records, val_records, epochs=2, patience=2, batch_size=2)
    probe = train_records[0].observation
    with torch.no_grad():
        bc_probabilities = trainer.agent.actor.distribution(probe)[0].clone()
    critic_result = trainer.warmup_critic(train_records, steps=2, batch_size=64)
    with torch.no_grad():
        warmed_probabilities = trainer.agent.actor.distribution(probe)[0]
    assert torch.equal(bc_probabilities, warmed_probabilities)
    online = trainer.train_online(train_env, total_steps=4, warmup_steps=0, batch_size=2, seed=42)
    checkpoint = output_dir / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint, metadata={"data": data_version(), "test_used": False,
        "phase": "runtime_integration", "not_formal_experiment": True,
        "train_sample_ids": [training_sample["sample_id"]],
        "validation_sample_ids": [validation_sample["sample_id"]]})
    restored = Trainer.load_checkpoint(checkpoint, expected_data_version="uiuc_v1")
    with torch.no_grad():
        expected = trainer.agent.actor.distribution(probe)[0]
        actual = restored.agent.actor.distribution(probe)[0]
    assert torch.equal(expected, actual)
    with pytest.raises(ValueError, match="no legal"):
        restored.agent.act(train_records[-1].next_observation)
    result = {"scope": "CPU small-network real-environment integration, not a performance/generalization experiment",
              "training_sample": {"sample_id": training_sample["sample_id"], "aliases": training_sample["aliases"]},
              "validation_sample": {"sample_id": validation_sample["sample_id"], "aliases": validation_sample["aliases"]},
              "test_used_for_training": False, "data": data_version(),
              "train_transitions": len(train_records), "validation_transitions": len(val_records),
              "bc_before": before, "bc": bc_result, "critic_warmup": critic_result,
              "critic_warmup_actor_probabilities_exactly_unchanged": True,
              "online": online, "checkpoint_probabilities_exactly_equal": True,
              "terminal_empty_mask_explicitly_rejected": True,
              "loaded_replay_size": len(restored.replay),
              "mapping_version": train_env.domain.metadata["mapping_version"]}
    write_json(output_dir / "integration_metrics.json", result)
    return result


def test_real_e655_environment_bc_critic_warmup_sac_and_reload(tmp_path):
    result = run_real_environment_integration(tmp_path)
    assert result["online"]["steps_requested"] == 4
    assert result["online"]["updates_total"] == 6
    assert result["checkpoint_probabilities_exactly_equal"]


def test_periodic_validation_preserves_actual_steps_and_curriculum_without_duplicate_final_call():
    for total_steps, expected in [(18, [5, 10, 15, 18]), (20, [5, 10, 15, 20])]:
        baseline = Trainer(agent()).train_online(DomainCounterEnvironment(), total_steps, warmup_steps=100)
        seen = []
        def callback(trainer, step):
            seen.append(step)
            assert step == trainer.online_steps
            # Validation acts in an independent simulator and changes no training graph.
            trainer.evaluate(CounterEnvironment(), episodes=1, max_steps=2)
        actual = Trainer(agent()).train_online(DomainCounterEnvironment(), total_steps,
            warmup_steps=100, validation_callback=callback, evaluation_interval=5)
        assert actual["online_steps_total"] == baseline["online_steps_total"] == total_steps
        assert [r["eligible_domains"] for r in actual["episodes"]] == [r["eligible_domains"] for r in baseline["episodes"]]
        assert actual["replay_total_added"] == baseline["replay_total_added"] == total_steps
        assert actual["validation_callback_steps"] == seen == expected
        assert len(seen) == len(set(seen))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Conditional CPU/CUDA checkpoint compatibility")
def test_checkpoint_cross_backend_reseeds_and_same_cuda_backend_restores_exact_rng(tmp_path):
    cuda_agent = DiscreteSACAgent(network_config={"hidden_dim": 16, "layers": 1}, device="cuda", seed=42)
    path = tmp_path / "cuda.pt"
    cuda_agent.save(path)
    expected_actions = [cuda_agent.act(observation()) for _ in range(8)]
    same_backend = DiscreteSACAgent.load(path, device="cuda")
    assert [same_backend.act(observation()) for _ in range(8)] == expected_actions
    assert same_backend.rng_restore_info["mode"] == "exact_state_restored"
    cpu_agent = DiscreteSACAgent.load(path, device="cpu")
    assert cpu_agent.rng_restore_info["mode"] == "cross_backend_reseeded"
    assert cpu_agent.rng_restore_info["sample_sequence_equivalent"] is False
    seeded_cpu = torch.Generator(device="cpu").manual_seed(42)
    assert torch.equal(cpu_agent.action_rng.get_state(), seeded_cpu.get_state())
    with torch.no_grad():
        expected = cuda_agent.actor.distribution(observation())[0].cpu()
        actual = cpu_agent.actor.distribution(observation())[0]
    assert torch.allclose(expected, actual, atol=2e-6, rtol=2e-5)
    cpu_path = tmp_path / "cpu.pt"
    cpu_agent.save(cpu_path)
    cross_back = DiscreteSACAgent.load(cpu_path, device="cuda")
    assert cross_back.rng_restore_info["mode"] == "cross_backend_reseeded"
    assert cross_back.rng_restore_info["source_device_type"] == "cpu"
    # The pre-fix GPU CLI checkpoint omitted the backend marker; infer its
    # generator format conservatively and still avoid loading it into CPU RNG.
    legacy_state = cuda_agent.state_dict()
    del legacy_state["action_rng_device_type"]
    legacy_cpu = DiscreteSACAgent.from_state_dict(legacy_state, device="cpu")
    assert legacy_cpu.rng_restore_info["source_backend_inferred"] is True
    assert legacy_cpu.rng_restore_info["mode"] == "cross_backend_reseeded"


def contract_fixture(intervals=16, protected_intervals=32):
    """A numerical simulator that explicitly declares a production-shaped MDP context."""
    from dataclasses import asdict
    env = CounterEnvironment()
    env.config = EnvConfig(intervals=intervals, protected_intervals=protected_intervals)
    env.environment_context = {'environment_version': ENVIRONMENT_VERSION,
        'env_config': asdict(env.config), 'mesh_spec': asdict(env.config.mesh_spec),
        'domain_config': {'farfield': 500., 'stretch': 18.}}
    return env


def test_v2_checkpoint_binds_full_context_and_rejects_different_mesh_before_agent_creation(tmp_path, monkeypatch):
    env = contract_fixture()
    trainer = Trainer(agent())
    trainer.train_online(env, 4, warmup_steps=2, batch_size=2)
    path = tmp_path/'v2.pt'
    trainer.save_checkpoint(path)
    state = torch.load(path, map_location='cpu', weights_only=False)
    assert state['trainer_version'] == state['agent']['checkpoint_version'] == 2
    assert state['environment_version'] == state['agent']['environment_version'] == ENVIRONMENT_VERSION
    assert state['environment_context'] == state['agent']['metadata']['environment_context'] == env.environment_context
    loaded = Trainer.load_checkpoint(path, expected_environment_context=env.environment_context)
    assert loaded.environment_context == env.environment_context and len(loaded.replay) == 4
    wrong = contract_fixture(intervals=32).environment_context
    def must_not_construct(*args, **kwargs):
        pytest.fail('Mismatching context must be rejected before network creation and replay restore')
    monkeypatch.setattr(DiscreteSACAgent, 'from_state_dict', must_not_construct)
    with pytest.raises(ValueError, match='environment context mismatch'):
        Trainer.load_checkpoint(path, expected_environment_context=wrong)


def test_legacy_mdp_checkpoints_cannot_restore_replay_or_networks(tmp_path, monkeypatch):
    trainer = Trainer(agent())
    for record in rollout_episode(CounterEnvironment(), [0, 0]):
        trainer.replay.add(record)
    original = tmp_path/'current.pt'
    trainer.save_checkpoint(original)
    current = torch.load(original, map_location='cpu', weights_only=False)
    def must_not_construct(*args, **kwargs):
        pytest.fail('Legacy restore must not allocate current agent networks')
    monkeypatch.setattr(DiscreteSACAgent, '__init__', must_not_construct)
    for variant in ('v1', 'missing_outer', 'missing_agent', 'wrong_agent'):
        state = copy.deepcopy(current)
        if variant == 'v1':
            state['trainer_version'] = state['agent']['checkpoint_version'] = 1
            state.pop('environment_version')
            state['agent'].pop('environment_version')
        elif variant == 'missing_outer':
            state.pop('environment_version')
        elif variant == 'missing_agent':
            state['agent'].pop('environment_version')
        else:
            state['agent']['environment_version'] = 'coarse_mesh_v1'
        path = tmp_path/f'{variant}.pt'
        torch.save(state, path)
        with pytest.raises(ValueError, match='environment version'):
            Trainer.load_checkpoint(path)
    state = current['agent']
    state.pop('environment_version')
    with pytest.raises(ValueError, match='environment version'):
        DiscreteSACAgent.from_state_dict(state)


def test_actor_only_migration_copies_only_actor_and_initializes_every_training_state(tmp_path):
    import hashlib
    import random
    source = Trainer(agent(seed=42))
    source.train_online(CounterEnvironment(), 6, warmup_steps=2, batch_size=2)
    constant_heads(source.agent.actor, [1., 2., 3., 4.])
    constant_heads(source.agent.q1, [10., 11., 12., 13.])
    original = tmp_path/'source.pt'
    source.save_checkpoint(original)
    legacy = torch.load(original, map_location='cpu', weights_only=False)
    legacy['trainer_version'] = legacy['agent']['checkpoint_version'] = 1
    legacy.pop('environment_version')
    legacy['agent'].pop('environment_version')
    legacy_path = tmp_path/'legacy.pt'
    torch.save(legacy, legacy_path)
    digest = hashlib.sha256(legacy_path.read_bytes()).hexdigest()
    migrated = Trainer.from_actor_checkpoint(legacy_path, seed=99)
    fresh = agent(seed=99)
    assert all(torch.equal(tensor, migrated.agent.actor.state_dict()[key]) for key, tensor in source.agent.actor.state_dict().items())
    for name in ('q1', 'q2', 'target_q1', 'target_q2'):
        assert all(torch.equal(tensor, getattr(migrated.agent, name).state_dict()[key]) for key, tensor in getattr(fresh, name).state_dict().items())
    assert migrated.agent.actor_optimizer.state == migrated.agent.q_optimizer.state == {}
    assert len(migrated.replay) == 0 and migrated.agent.updates == migrated.agent.actor_updates == 0
    assert migrated.online_steps == migrated.online_episodes == 0 and migrated.bc_result is None
    assert migrated.reset_failures == [] and migrated.environment_context is None
    assert migrated.rng.getstate() == random.Random(99).getstate()
    assert torch.equal(migrated.agent.action_rng.get_state(), torch.Generator(device='cpu').manual_seed(99).get_state())
    assert hashlib.sha256(legacy_path.read_bytes()).hexdigest() == digest
    assert migrated.actor_migration['source_sha256'] == digest
    assert migrated.actor_migration['source_environment_version'] == 'legacy_unspecified'
    assert migrated.actor_migration['critics_optimizers_replay_rng_restored'] is False
    migrated.bind_environment(contract_fixture())
    output = tmp_path/'migrated_current.pt'
    migrated.save_checkpoint(output)
    loaded = Trainer.load_checkpoint(output, expected_environment_context=migrated.environment_context)
    assert loaded.actor_migration == migrated.actor_migration
    assert len(loaded.replay) == 0
    broken = copy.deepcopy(legacy)
    broken['agent']['action_version'] = 'future_incompatible_actions'
    torch.save(broken, tmp_path/'bad_actions.pt')
    with pytest.raises(ValueError, match='observation/action schema'):
        Trainer.from_actor_checkpoint(tmp_path/'bad_actions.pt')
    tested = copy.deepcopy(legacy)
    tested['agent']['metadata']['test_used'] = True
    torch.save(tested, tmp_path/'tested_actor.pt')
    with pytest.raises(ValueError, match='test_used=True'):
        Trainer.from_actor_checkpoint(tmp_path/'tested_actor.pt')


def test_bound_trainer_cannot_cross_mesh_context_in_training_evaluation_or_save(tmp_path):
    trainer = Trainer(agent())
    first, different = contract_fixture(), contract_fixture(intervals=8, protected_intervals=16)
    trainer.bind_environment(first)
    for record in rollout_episode(CounterEnvironment(), [0, 0]):
        trainer.replay.add(record)
    before = trainer.replay.total_added
    with pytest.raises(ValueError, match='Environment context mismatch'):
        trainer.train_online(different, total_steps=4, warmup_steps=100)
    with pytest.raises(ValueError, match='Environment context mismatch'):
        trainer.evaluate(different, episodes=1)
    with pytest.raises(ValueError, match='metadata environment context differs'):
        trainer.save_checkpoint(tmp_path/'must_not_exist.pt', metadata={'environment_context': different.environment_context})
    assert trainer.replay.total_added == before and trainer.online_steps == 0
    assert not (tmp_path/'must_not_exist.pt').exists()


def test_unbound_existing_replay_requires_explicit_provenance_before_real_environment(tmp_path):
    trainer = Trainer(agent())
    trainer.replay.add(rollout_episode(CounterEnvironment(), [0, 0])[0])
    with pytest.raises(ValueError, match='before adding replay transitions'):
        trainer.bind_environment(contract_fixture())
    with pytest.raises(ValueError, match='metadata cannot certify'):
        trainer.save_checkpoint(tmp_path/'must_not_exist.pt', metadata={'environment_context': contract_fixture().environment_context})
    assert not (tmp_path/'must_not_exist.pt').exists()
