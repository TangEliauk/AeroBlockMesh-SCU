"""Candidate batching, full-graph context, SAC math reuse and model isolation."""
from dataclasses import asdict
import copy

import numpy as np
import pytest
import torch

from smartcfd.core.types import Transition
from smartcfd.airfoil_decomposition.learning import (AirfoilSACAgent, NETWORK,
    batch_candidate_observations, save_inference_checkpoint, load_inference_checkpoint)
from smartcfd.rl.agents.sac import SACConfig


@pytest.fixture(scope='module', autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def observation():
    features = np.zeros((256, 32), dtype=np.float32)
    features[:4, 0] = [0., .2, .6, 1.]
    next_index = np.full(256, -1, dtype=np.int64)
    previous = next_index.copy(); twin = next_index.copy()
    for start in (0, 4):
        next_index[start:start+4] = np.roll(np.arange(start, start+4), -1)
        previous[start:start+4] = np.roll(np.arange(start, start+4), 1)
    twin[0], twin[4] = 4, 0
    mask = np.zeros(1024, dtype=bool); mask[[0, 17, 999]] = True
    candidates = np.zeros((1024, 16), dtype=np.float32)
    candidates[[0, 17], 0] = 1.; candidates[999, 1] = 1.
    candidates[17, 2:6] = [.2, .3, .6, .7]
    nodes = np.full((1024, 2), -1, dtype=np.int64)
    nodes[0], nodes[17], nodes[999] = [0, 1], [2, 3], [0, 4]
    return {'features': features, 'next': next_index, 'previous': previous, 'twin': twin,
            'global': np.zeros(12, dtype=np.float32), 'candidates': candidates,
            'candidate_nodes': nodes, 'mask': mask}


def test_candidate_batch_preserves_holey_action_ids_and_terminal_skip():
    agent = AirfoilSACAgent(seed=42)
    obs = observation(); second = copy.deepcopy(obs); second['candidates'][17, 3] = .87
    batch = batch_candidate_observations([obs, second])
    assert batch['candidates'].shape == (2, 1024, 16)
    assert batch['candidate_nodes'].shape == (2, 1024, 2)
    assert batch['candidates'][1, 17, 3].item() == pytest.approx(.87)
    assert torch.equal(batch['candidate_nodes'][0, 999], torch.tensor([0, 4]))
    probabilities, log_probabilities = agent.actor.distribution(batch)
    assert probabilities.shape == (2, 1024) and torch.isfinite(log_probabilities).all()
    assert torch.equal(probabilities[0] != 0, torch.as_tensor(obs['mask']))
    assert torch.allclose(probabilities.sum(1), torch.ones(2))
    assert bool(obs['mask'][agent.act(obs, deterministic=True)])
    terminal = copy.deepcopy(obs)
    terminal['next'][:] = -1; terminal['previous'][:] = -1; terminal['twin'][:] = -1
    terminal['mask'][:] = False
    result = agent.compute_targets([Transition(obs, 17, 2.75, terminal, True, False)])
    assert result.tolist() == [2.75]
    # Truncation still bootstraps and therefore cannot accept an empty next mask.
    with pytest.raises(ValueError, match='halfedge|legal action'):
        agent.compute_targets([Transition(obs, 17, 2.75, terminal, False, True)])


def test_queue_and_global_context_change_same_kind_candidate_relative_scores():
    agent = AirfoilSACAgent(seed=42)
    obs = observation()
    # Disconnect the queued ring so its effect must cross global pooling,
    # rather than arrive through the candidate endpoints' local twin messages.
    obs['twin'][:] = -1
    original = agent.actor(obs).detach()
    difference = original[0, 0]-original[0, 17]
    queued = copy.deepcopy(obs); queued['features'][4:8, 4:10] = np.arange(1, 7)
    logits = agent.actor(queued)
    assert abs(float(logits[0, 0]-logits[0, 17]-difference)) > 1e-7
    context = copy.deepcopy(obs); context['global'][11] = 2.
    logits = agent.actor(context)
    assert abs(float(logits[0, 0]-logits[0, 17]-difference)) > 1e-7
    padded = copy.deepcopy(obs)
    padded['features'][8:] = np.nan
    padded['candidates'][~padded['mask']] = np.nan
    padded['candidate_nodes'][~padded['mask']] = 999999
    assert torch.equal(original, agent.actor(padded))


@pytest.mark.parametrize('corruption', ['endpoint', 'fractional', 'graph_padding', 'candidate_nan'])
def test_legal_candidate_and_graph_index_validation(corruption):
    agent = AirfoilSACAgent(seed=42); obs = observation()
    if corruption == 'endpoint':
        obs['candidate_nodes'][17, 1] = 256
    elif corruption == 'fractional':
        obs['candidate_nodes'] = obs['candidate_nodes'].astype(float)
        obs['candidate_nodes'][17, 1] = 2.5
    elif corruption == 'graph_padding':
        obs['twin'][0] = 100
    else:
        obs['candidates'][17, 0] = np.nan
    with pytest.raises(ValueError, match='index|indices|padding|candidate'):
        agent.actor(obs)


def test_real_sac_update_changes_actor_critics_and_targets_with_candidate_fields():
    agent = AirfoilSACAgent(seed=42); obs = observation()
    next_obs = copy.deepcopy(obs); next_obs['global'][0] = .25
    terminal = copy.deepcopy(next_obs); terminal['mask'][:] = False
    before = {name: copy.deepcopy(getattr(agent, name).state_dict())
              for name in ('actor', 'q1', 'q2', 'target_q1', 'target_q2')}
    transitions = [Transition(obs, 0, .4, next_obs, False, False),
                   Transition(obs, 17, 2., terminal, True, False)]
    metrics = agent.update(transitions)
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics['alpha'] == .01 and metrics['actor_updated'] is True
    assert agent.updates == agent.actor_updates == 1
    for name, weights in before.items():
        assert any(not torch.equal(value, getattr(agent, name).state_dict()[key]) for key, value in weights.items())
    assert agent.q1.encoder is not agent.q2.encoder and agent.q1.encoder is not agent.actor.encoder


def test_inference_checkpoint_roundtrip_and_namespace_data_config_guards(tmp_path):
    agent = AirfoilSACAgent(seed=42); path = tmp_path/'candidate.pt'
    settings = {'environment_config': {'max_half_edges': 256, 'max_candidates': 1024},
                'dataset_sha256': 'a'*64, 'source': {'source_sha256': 'b'*64}}
    sac = SACConfig(alpha=.01)
    save_inference_checkpoint(agent, path, **settings, online_step=0, selection='untrained')
    restored, state = load_inference_checkpoint(path, **settings, sac_config=sac)
    assert state['inference_only'] is True and state['training_resume_supported'] is False
    assert state['network_config'] == asdict(NETWORK)
    assert torch.equal(agent.actor(observation()), restored.actor(observation()))
    for field, value in (('environment_version', 'final_mesh_v2'),
                         ('environment_version', 'orthogonal_decomposition_v1'),
                         ('action_version', 'vertex_axis_cut_v1'),
                         ('schema', 'orthogonal_decomposition_checkpoint_v1')):
        bad = copy.deepcopy(state); bad[field] = value
        torch.save(bad, tmp_path/'bad.pt')
        with pytest.raises(ValueError, match=field):
            load_inference_checkpoint(tmp_path/'bad.pt', **settings, sac_config=sac)
    with pytest.raises(ValueError, match='dataset_sha256'):
        load_inference_checkpoint(path, **{**settings, 'dataset_sha256': 'c'*64}, sac_config=sac)
    with pytest.raises(ValueError, match='sac_config'):
        load_inference_checkpoint(path, **settings, sac_config=SACConfig(alpha=.1))
    with pytest.raises(ValueError, match='legacy'):
        agent.state_dict()
