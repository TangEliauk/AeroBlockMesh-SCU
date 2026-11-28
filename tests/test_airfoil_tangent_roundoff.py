"""Real training-case regression for finite-difference cancellation."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from smartcfd.airfoil_decomposition.env import AirfoilBlockEnv
from smartcfd.airfoil_decomposition.experiment import rule_action
from test_airfoil_decomposition_env import AffineGeometry
from smartcfd.airfoil_decomposition import env as env_module


def test_resolved_tangents_preserve_original_values_exactly(monkeypatch):
    monkeypatch.setattr(env_module, 'CurveGeometry', AffineGeometry)
    env = AirfoilBlockEnv(); env.reset()
    uv = np.array([[.1, .2], [.7, 1.], [.9, .8]])
    target = np.array([[.9, .7], [.5, .5], [.1, .8]])
    xy = env.geometry.mapping.evaluate(uv)
    expected = env.geometry.mapping.evaluate(uv+1e-5*(target-uv))-xy
    expected /= np.linalg.norm(expected, axis=1)[:, None]
    np.testing.assert_array_equal(env._physical_tangents(uv, target, xy), expected)


def test_tiny_valid_edge_has_finite_unit_tangent_and_normal(monkeypatch):
    monkeypatch.setattr(env_module, 'CurveGeometry', AffineGeometry)
    env = AirfoilBlockEnv(); env.reset()
    uv = np.array([[.6666666660250329, 1.]])
    target = np.array([[.6666666660197218, 1.]])
    np.testing.assert_array_equal(uv+1e-5*(target-uv), uv)
    tangent = env._physical_tangents(uv, target, uv)
    normal = np.column_stack((tangent[:, 1], -tangent[:, 0]))
    assert np.isfinite(tangent).all() and np.isfinite(normal).all()
    np.testing.assert_allclose(tangent, [[-1., 0.]])
    np.testing.assert_allclose(np.linalg.norm(normal, axis=1), 1.)
    np.testing.assert_allclose((normal*tangent).sum(axis=1), 0.)


def test_genuinely_collapsed_physical_curve_still_fails_closed():
    env = AirfoilBlockEnv()
    env.geometry = SimpleNamespace(
        mapping=SimpleNamespace(evaluate=lambda uv: np.zeros_like(uv)),
        curve=lambda a, b: np.zeros((5, 2)))
    with pytest.raises(ValueError, match='degenerate local tangent'):
        env._physical_tangents(np.array([[.1, .2]]), np.array([[.7, .8]]), np.zeros((1, 2)))


def test_original_train_case_greedy_prefix_survives_step31_roundoff():
    fixture = json.loads((Path(__file__).parent/'fixtures/airfoil_tangent_roundoff.json').read_text(encoding='utf-8'))
    env = AirfoilBlockEnv(fixture['sample'], fixture['config'])
    env.reset()
    # Replay the captured real greedy choices; do not replace its failing shape
    # or change legal actions. The fixture avoids rerunning 31 greedy searches.
    for action in fixture['actions']:
        assert env.candidates[action['index']] == action['candidate']
        observation, _, terminated, _, _ = env.step(action['index'])
        assert all(np.isfinite(observation[name]).all() for name in ('features', 'global', 'candidates'))
        assert not terminated
    assert env.steps == 31
    edge = fixture['bad_edge']
    row = observation['features'][edge['slot']]
    tangent, normal = row[8:10], row[20:22]
    np.testing.assert_allclose(np.linalg.norm(tangent), 1., atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(normal), 1., atol=1e-6)
    np.testing.assert_allclose(np.dot(tangent, normal), 0., atol=1e-6)
    curve = env.geometry.curve(edge['a'], edge['b'])
    chord = curve[-1]-curve[0]
    assert np.linalg.norm(chord) > 1e-12
    assert np.dot(tangent, chord/np.linalg.norm(chord)) > .9999
    # Preserve the negative policy outcome: the next rule action exhausts the
    # episode budget normally, rather than crashing or being silently repaired.
    _, _, terminal, truncated, info = env.step(rule_action(env))
    assert terminal and not truncated and not info['success']
    assert info['reason'] == 'step_budget_exhausted'
