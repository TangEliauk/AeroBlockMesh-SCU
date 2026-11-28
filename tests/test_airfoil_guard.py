"""Proof-only pruning, transition isolation and physical successful witnesses."""
import copy

import numpy as np
import pytest

from smartcfd.airfoil_decomposition import env as env_module
from smartcfd.airfoil_decomposition.env import AirfoilBlockEnv, synthetic_profile
from smartcfd.airfoil_decomposition.config import AirfoilDecompositionConfig
from smartcfd.airfoil_decomposition.guard import (
    GuardConfig, ConservativeGuard, GuardedAirfoilEnv, budget_lower_bound)
from smartcfd.airfoil_decomposition.experiment import fixed_action
from smartcfd.airfoil_decomposition.baselines import refined_c_action
from test_airfoil_decomposition_env import AffineGeometry, vertical_action


@pytest.fixture
def affine(monkeypatch):
    monkeypatch.setattr(env_module, "CurveGeometry", AffineGeometry)


def test_fork_isolation_and_transition_equivalence(affine):
    env = AirfoilBlockEnv()
    original, _ = env.reset(seed=42)
    original_graph, original_rng = env.graph.to_dict(), copy.deepcopy(env.np_random.bit_generator.state)
    branch = env.fork()
    assert type(branch) is AirfoilBlockEnv and branch.geometry is env.geometry
    action = vertical_action(env, .25)
    result = branch.step(action)
    branch.np_random.random()
    assert env.graph.to_dict() == original_graph and env.num_steps == 0
    assert env.np_random.bit_generator.state == original_rng
    for name, value in original.items():
        np.testing.assert_array_equal(value, env._observation()[name])
    real = env.step(action)
    assert branch.graph.to_dict() == env.graph.to_dict() and branch.queue == env.queue
    assert result[1:4] == real[1:4]
    for name in result[0]:
        np.testing.assert_array_equal(result[0][name], real[0][name])
    branch.graph.vertices[0] = (.123, .456)
    assert env.graph.vertices[0] != branch.graph.vertices[0]


def test_global_budget_bound_counts_reopened_shared_quad(affine):
    env = AirfoilBlockEnv(); env.reset()
    env.step(vertical_action(env, .25))
    before = budget_lower_bound(env.graph, steps=env.steps, config=env.config)
    candidate = next(c for c in env._candidates if c.description['kind'] == 'insert_edge_point'
                     and len(c.description['owners']) == 2)
    after = budget_lower_bound(candidate.graph, steps=env.steps+1, config=env.config)
    assert after['defect'] == before['defect']+2
    assert after['minimum_cuts'] == before['minimum_cuts']+1
    assert after['minimum_final_halfedges'] == 4*after['minimum_final_blocks']


def test_parent_profile_reset_detaches_fork_geometry_cache(affine):
    env = AirfoilBlockEnv(); env.reset()
    branch = env.fork()
    old_geometry, old_cache = branch.geometry, branch._enumeration_cache
    other = synthetic_profile(); other['sample_id'] = 'different_profile'
    env.reset(options={'sample': other})
    assert env.geometry is not old_geometry
    assert env._enumeration_cache is not old_cache
    assert branch.geometry is old_geometry and branch._enumeration_cache is old_cache
    branch.step(vertical_action(branch, .25))
    assert branch.sample['sample_id'] != env.sample['sample_id']


def test_fork_space_rngs_and_guard_diagnostics_are_independent(affine):
    env = GuardedAirfoilEnv(guard_config={'second_layer_budget': 0})
    env.reset(seed=42)
    env.action_space.seed(123)
    env.observation_space.seed(456)
    expected_action = copy.deepcopy(env.action_space)
    expected_observation = copy.deepcopy(env.observation_space)
    branch = env.fork()
    assert branch.action_space is not env.action_space
    assert branch.observation_space is not env.observation_space
    assert not hasattr(branch, 'guard_statistics') and not hasattr(branch, '_guard_report')
    assert not hasattr(branch, 'guard')
    branch.action_space.sample(); branch.observation_space.sample()
    assert env.action_space.sample() == expected_action.sample()
    actual, expected = env.observation_space.sample(), expected_observation.sample()
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key])


@pytest.mark.parametrize('settings,reason', [
    ({'max_steps': 1}, 'remaining_step_lower_bound'),
    ({'max_blocks': 7}, 'final_block_lower_bound'),
    ({'max_half_edges': 31}, 'final_halfedge_lower_bound')])
def test_bounds_are_necessary_conditions(affine, settings, reason):
    env = AirfoilBlockEnv(); env.reset()
    report = budget_lower_bound(env.graph, steps=0, config=AirfoilDecompositionConfig(**settings))
    assert report['defect'] == 6 and report['minimum_cuts'] == 3
    assert report['reason'] == reason


def test_unknown_preserved_budget_and_cache_determinism(affine):
    env = AirfoilBlockEnv(); env.reset()
    guard = ConservativeGuard(GuardConfig(second_layer_budget=7))
    before = env.graph.to_dict(), env.queue, env.num_steps
    first = guard.analyze(env)
    second = guard.analyze(env)
    cold = ConservativeGuard(GuardConfig(second_layer_budget=7)).analyze(env)
    assert (env.graph.to_dict(), env.queue, env.num_steps) == before
    assert first['metrics']['root_preview_calls'] == len(env.candidates)
    assert first['metrics']['second_layer_transitions'] == 7
    assert sum(r['second_layer_checked'] for r in first['actions']) == 7
    assert max(r['second_layer_checked'] for r in first['actions']) == 1  # round robin
    for row in first['actions']:
        if row['status'] == 'unknown':
            assert first['effective_mask'][row['action']]
    assert first['actions'] == second['actions'] == cold['actions']
    np.testing.assert_array_equal(first['effective_mask'], second['effective_mask'])
    assert second['metrics']['guard_cache_hit'] == 1
    assert second['metrics']['root_preview_calls'] == second['metrics']['geometry_validation_calls'] == 0
    assert second['metrics']['second_layer_transitions'] == 7
    env.steps += 1
    assert guard.analyze(env)['metrics']['guard_cache_hit'] == 0


def test_all_pruned_reset_and_holey_mask_enforcement(affine):
    blocked = GuardedAirfoilEnv(config={'max_steps': 2})
    obs, info = blocked.reset()
    assert blocked.terminated and not blocked.success
    assert info['reason'] == 'proven_unfinishable' and not obs['mask'].any()
    env = GuardedAirfoilEnv(config={'max_steps': 3}, guard_config={'second_layer_budget': 0})
    obs, _ = env.reset()
    rejected = next(row['action'] for row in env._guard_report['actions'] if row['status'] == 'proven_dead_end')
    assert not obs['mask'][rejected] and rejected < len(env.candidates)
    before = env.graph.to_dict()
    obs, reward, terminal, truncated, info = env.step(rejected)
    assert terminal and not truncated and info['reason'] == 'invalid_action'
    assert env.graph.to_dict() == before and not obs['mask'].any()


def test_second_layer_requires_exhaustive_failure_proof(affine):
    env = AirfoilBlockEnv(config={'max_vertices': 15}); env.reset()
    original = env.geometry.validate
    def no_eighth_face(graph, refine=False):
        result = original(graph, refine=refine)
        if len(graph.faces) > 7:
            result.update(valid=False, reason='explicit_test_eighth_face_forbidden')
        return result
    env.geometry.validate = no_eighth_face
    env._enumeration_cache.clear()
    no_search = ConservativeGuard({'second_layer_budget': 0}).analyze(env)
    searched = ConservativeGuard().analyze(env)
    assert no_search['effective_mask'].any()
    assert not searched['effective_mask'].any()
    proofs = [r for r in searched['actions'] if r['reason'] == 'all_second_layer_branches_proven_failed']
    assert proofs
    for row in proofs:
        assert row['second_layer_checked'] == row['second_layer_failures'] == row['second_layer_total']


def test_all_pruned_after_step_has_terminal_failure_reward(affine, monkeypatch):
    env = GuardedAirfoilEnv(guard_config={'second_layer_budget': 0})
    env.reset()
    action = vertical_action(env, .25)
    primitive = AirfoilBlockEnv.preview(env, action)
    original = env.guard.analyze
    def all_proven(state):
        result = original(state)
        result['effective_mask'][:] = False
        for row in result['actions']:
            row.update(status='proven_dead_end', reason='explicit_test_proof')
        return result
    monkeypatch.setattr(env.guard, 'analyze', all_proven)
    obs, reward, terminal, truncated, info = env.step(action)
    assert terminal and not truncated and not obs['mask'].any()
    assert info['reason'] == 'proven_unfinishable'
    assert info['reward_components']['potential_after'] == 0
    assert info['base_reward'] == pytest.approx(-5.02)
    assert reward == pytest.approx(-5.02-primitive['reward_components']['potential_before'])
    assert env.partition()['cut_history'][-1]['reason'] == 'proven_unfinishable'


@pytest.mark.parametrize('policy,steps,blocks', [(fixed_action, 3, 8), (refined_c_action, 12, 12)])
def test_successful_witnesses_never_pruned_affine(affine, policy, steps, blocks):
    env = GuardedAirfoilEnv(); env.reset()
    while not env.terminated:
        action = policy(env)
        assert action is not None and env.action_mask[action]
        env.step(action)
    assert env.success and env.steps == steps and len(env.graph.faces) == blocks


@pytest.mark.parametrize('policy,steps,blocks', [(fixed_action, 3, 8), (refined_c_action, 12, 12)])
def test_real_geometry_successful_witnesses_never_pruned(policy, steps, blocks):
    env = GuardedAirfoilEnv(sample=synthetic_profile()); env.reset()
    while not env.terminated:
        action = policy(env)
        assert action is not None and env.action_mask[action]
        env.step(action)
    assert env.success and env.steps == steps and len(env.graph.faces) == blocks
    assert env.guard_statistics['geometry_validation_calls'] > 0


def test_guard_configuration_rejects_inexact_depth():
    with pytest.raises(ValueError):
        GuardConfig(depth=1)
    with pytest.raises(ValueError):
        GuardConfig(second_layer_budget=-1)
