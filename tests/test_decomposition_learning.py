"""Boundary/policy and checkpoint namespace tests; no training workload."""
from dataclasses import asdict
import numpy as np
import pytest
import torch

from smartcfd.core.types import Transition
from smartcfd.decomposition.learning import (BoundarySACAgent, NETWORK,
    save_inference_checkpoint, load_inference_checkpoint)
from smartcfd.rl.agents.sac import SACConfig


def observation():
    features = np.zeros((128, 24), dtype=np.float32)
    features[:4, 0] = [0., .3, .6, 1.]
    links = np.full(128, -1, dtype=np.int64)
    next_index, previous = links.copy(), links.copy()
    for start in (0, 4):
        next_index[start:start+4] = np.roll(np.arange(start, start+4), -1)
        previous[start:start+4] = np.roll(np.arange(start, start+4), 1)
    mask = np.zeros(256, dtype=bool); mask[[0, 3, 6]] = True
    return {'features': features, 'next': next_index, 'previous': previous,
            'twin': links.copy(), 'mask': mask, 'global': np.zeros(8, dtype=np.float32)}


def test_masked_two_axis_policy_and_terminal_target_skip_empty_next_mask():
    torch.set_num_threads(1)
    agent = BoundarySACAgent(seed=42)
    obs = observation()
    probability, log_probability = agent.actor.distribution(obs)
    assert probability.shape == (1, 256)
    assert torch.isfinite(log_probability).all()
    assert probability[0, ~torch.as_tensor(obs['mask'])].sum() == 0
    assert torch.allclose(probability.sum(), torch.tensor(1.))
    assert bool(obs['mask'][agent.act(obs, deterministic=True)])
    terminal = {key: np.array(value, copy=True) for key, value in obs.items()}
    terminal['mask'][:] = False
    terminal['next'][:] = -1
    targets = agent.compute_targets([Transition(obs, 0, 2.75, terminal, True, False)])
    assert targets.tolist() == [2.75]
    with pytest.raises(ValueError, match='no legal action|boundary vertex'):
        agent.actor.distribution(terminal)


def test_queued_polygon_changes_active_logits_but_padding_does_not():
    torch.set_num_threads(1)
    agent = BoundarySACAgent(seed=42)
    obs = observation()
    baseline = agent.actor(obs).detach()
    changed = {key: np.array(value, copy=True) for key, value in obs.items()}
    changed['features'][4:8, 1:6] = np.arange(1, 6, dtype=np.float32)
    assert not torch.allclose(baseline[:, :8], agent.actor(changed)[:, :8])
    baseline_rank = baseline[0, 0] - baseline[0, 6]  # two legal horizontal cuts
    changed_logits = agent.actor(changed)
    assert abs(float(changed_logits[0, 0] - changed_logits[0, 6] - baseline_rank)) > 1e-7
    budget = {key: np.array(value, copy=True) for key, value in obs.items()}
    budget['global'][0] = .9  # elapsed fraction of the finite step budget
    budget_logits = agent.actor(budget)
    assert abs(float(budget_logits[0, 0] - budget_logits[0, 6] - baseline_rank)) > 1e-7
    padding = {key: np.array(value, copy=True) for key, value in obs.items()}
    padding['features'][8:] = 1000
    assert torch.equal(baseline[:, :8], agent.actor(padding)[:, :8])


def test_inference_checkpoint_roundtrip_and_cross_environment_rejection(tmp_path):
    torch.set_num_threads(1)
    agent = BoundarySACAgent(seed=42)
    path = tmp_path/'model.pt'
    environment = {'max_vertices': 128, 'gamma': .99}
    source = {'source_sha256': 'a'*64}
    sac = SACConfig(alpha=.01)
    save_inference_checkpoint(agent, path, environment_config=environment, dataset_sha256='b'*64,
                              source=source, online_step=0, selection='initial')
    restored, state = load_inference_checkpoint(path, environment_config=environment, dataset_sha256='b'*64,
                                                source=source, sac_config=sac)
    assert state['inference_only'] is True and state['training_resume_supported'] is False
    assert state['network_config'] == asdict(NETWORK)
    assert torch.equal(agent.actor(observation()), restored.actor(observation()))
    with pytest.raises(ValueError, match='dataset_sha256'):
        load_inference_checkpoint(path, environment_config=environment, dataset_sha256='c'*64, source=source, sac_config=sac)
    bad = torch.load(path, map_location='cpu', weights_only=True)
    bad['environment_version'] = 'final_mesh_v2'; torch.save(bad, tmp_path/'airfoil.pt')
    with pytest.raises(ValueError, match='environment_version'):
        load_inference_checkpoint(tmp_path/'airfoil.pt', environment_config=environment, dataset_sha256='b'*64,
                                  source=source, sac_config=sac)
    with pytest.raises(ValueError, match='airfoil'):
        agent.state_dict()


def test_real_cut_partition_plot_accepts_recorded_segment_lists(tmp_path):
    from smartcfd.decomposition import OrthogonalBlockEnv
    from smartcfd.decomposition.visualize import plot_partition
    env = OrthogonalBlockEnv()
    observation, _ = env.reset(seed=42)
    action = int(np.flatnonzero(observation['mask'])[0])
    _, reward, _, _, info = env.step(action)
    partition = env.partition()
    assert partition['cuts'] and partition['cuts'][0]['segments']
    output = tmp_path/'partition.png'
    plot_partition(env.case, {'partition': partition, 'success': info['success'],
                             'steps': 1, 'return': reward}, output, title='integration')
    assert output.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
