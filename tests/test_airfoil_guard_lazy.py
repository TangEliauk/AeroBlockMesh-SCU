from pathlib import Path
import importlib.util
import sys

HERE = Path(__file__).resolve().parent
REFERENCE = HERE/'fixtures/airfoil_guard_reference_v1'

import numpy as np
import pytest
from smartcfd.airfoil_decomposition import env as env_module
from smartcfd.airfoil_decomposition.env import AirfoilBlockEnv
from smartcfd.airfoil_decomposition.guard import ConservativeGuard as LazyConservativeGuard
from test_airfoil_decomposition_env import AffineGeometry, vertical_action

name = 'smartcfd.airfoil_decomposition.frozen_env_reference'
spec = importlib.util.spec_from_file_location(name, REFERENCE/'env.py')
frozen = importlib.util.module_from_spec(spec); sys.modules[name] = frozen; spec.loader.exec_module(frozen)
guard_name = 'smartcfd.airfoil_decomposition.frozen_guard_reference'
guard_spec = importlib.util.spec_from_file_location(guard_name, REFERENCE/'guard.py')
guard_frozen = importlib.util.module_from_spec(guard_spec)
sys.modules[guard_name] = guard_frozen; guard_spec.loader.exec_module(guard_frozen)
ConservativeGuard = guard_frozen.ConservativeGuard


@pytest.fixture(autouse=True)
def affine(monkeypatch):
    monkeypatch.setattr(env_module, 'CurveGeometry', AffineGeometry)
    monkeypatch.setattr(frozen, 'CurveGeometry', AffineGeometry)


def serialize(candidates):
    return [{'description': c.description, 'nodes': c.nodes, 'graph': c.graph.to_dict(),
             'validation': c.validation, 'queue': c.queue} for c in candidates]


@pytest.mark.parametrize('capacity', [1, 33, 1024])
def test_full_wrapper_matches_frozen_candidate_order_and_failure(capacity):
    old, new = frozen.AirfoilBlockEnv(config={'max_candidates': capacity}), AirfoilBlockEnv(config={'max_candidates': capacity})
    old.reset(); new.reset()
    for _ in range(3):
        oc, of = old._enumerate_candidates(old.graph, old.queue)
        nc, nf = new._enumerate_candidates(new.graph, new.queue)
        assert of == nf and serialize(oc) == serialize(nc)
        if old.terminated or new.terminated:
            assert old.terminated == new.terminated
            break
        action = next((i for i, c in enumerate(old.candidates) if c['kind'] == 'insert_edge_point'), 0)
        old.step(action); new.step(action)


def semantic(report):
    fields = ('action', 'status', 'reason', 'lower_bound', 'second_layer_checked', 'second_layer_failures')
    return [{k: r[k] for k in fields} for r in report['actions']]


@pytest.mark.parametrize('capacity', [33, 1024])
@pytest.mark.parametrize('budget', [0, 1, 7, 128])
def test_masks_proofs_order_counts_and_low_capacity_fallback(capacity, budget):
    env = AirfoilBlockEnv(config={'max_candidates': capacity}); env.reset()
    for step in range(2):
        config = {'second_layer_budget': budget, 'cache_capacity': 0}
        old = ConservativeGuard(config).analyze(env)
        # Do not let the baseline's complete lists hide the lazy path.
        env._enumeration_cache.clear()
        new = LazyConservativeGuard(config).analyze(env)
        np.testing.assert_array_equal(old['effective_mask'], new['effective_mask'])
        assert semantic(old) == semantic(new)
        for key in ('root_previews', 'second_layer_transitions', 'removed_actions', 'unknown_actions', 'completion_witnesses'):
            assert old['metrics'][key] == new['metrics'][key]
        for left, right in zip(old['actions'], new['actions']):
            if right['second_layer_total'] is not None:
                assert left['second_layer_total'] == right['second_layer_total']
        if step == 0:
            env.step(vertical_action(env, .25))


def test_lazy_exhaustion_proof_is_exact_and_parent_unchanged():
    env = AirfoilBlockEnv(config={'max_vertices': 15}); env.reset()
    original = env.geometry.validate
    def seven_faces(graph, refine=False):
        result = original(graph, refine=refine)
        if len(graph.faces) > 7:
            result.update(valid=False, reason='test_forbids_eighth_face')
        return result
    env.geometry.validate = seven_faces; env._enumeration_cache.clear()
    before = env.graph.to_dict(), env.queue, env.steps, env.action_mask.copy()
    old = ConservativeGuard({'cache_capacity': 0}).analyze(env)
    env._enumeration_cache.clear()
    new = LazyConservativeGuard({'cache_capacity': 0}).analyze(env)
    assert semantic(old) == semantic(new) and not new['effective_mask'].any()
    assert (env.graph.to_dict(), env.queue, env.steps) == before[:3]
    np.testing.assert_array_equal(env.action_mask, before[3])
    for row in new['actions']:
        if row['reason'] == 'all_second_layer_branches_proven_failed':
            assert row['second_layer_total'] == row['second_layer_checked'] == row['second_layer_failures']


def test_empty_and_complete_candidate_caches_have_identical_search_semantics():
    env = AirfoilBlockEnv(); env.reset(); env.step(vertical_action(env, .25))
    assert len(env._candidates) < 32  # all root successors fit the cache together
    env._enumeration_cache.clear()
    config = {'cache_capacity': 0}
    cold = LazyConservativeGuard(config).analyze(env)
    env._enumeration_cache.clear()
    for action in range(len(env._candidates)):
        branch = env.fork(); branch.step(action)
    warm = LazyConservativeGuard(config).analyze(env)
    assert semantic(cold) == semantic(warm)
    np.testing.assert_array_equal(cold['effective_mask'], warm['effective_mask'])
    for key in ('root_previews', 'second_layer_transitions', 'removed_actions', 'unknown_actions', 'completion_witnesses'):
        assert cold['metrics'][key] == warm['metrics'][key]
    assert warm['metrics']['candidate_cache_hits'] > 0
