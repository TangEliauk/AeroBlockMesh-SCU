"""Candidate/FIFO/reward contracts plus a real physical C-domain integration.

Fast transition tests use an explicit affine geometry double. They do not stand
in for the separately identified real CurveGeometry integration at the end.
"""
import copy
import json

import numpy as np
import pytest

from smartcfd.geometry.domain import make_domain
from smartcfd.partition.constraints import validate_graph
from smartcfd.partition.topology import edge_incidence, half_edges, initial_graph, protected_edges
from smartcfd.airfoil_decomposition.config import AirfoilDecompositionConfig
from smartcfd.airfoil_decomposition import env as env_module
from smartcfd.airfoil_decomposition.env import AirfoilBlockEnv, synthetic_profile, topology_potential


class AffineMapping:
    def evaluate(self, uv):
        return np.asarray(uv, dtype=float).copy()


class AffineGeometry:
    """Real UV legality, affine physical coordinates; records validator calls."""
    def __init__(self, sample, config):
        self.domain = make_domain(sample, farfield=config.farfield, stretch=config.stretch)
        self.mapping = AffineMapping()
        self.calls, self.arc_calls = [], []
        self.reject_edits = self.reject_refined = False
        self.initial = initial_graph(self.domain)

    def curve(self, a, b, refined=False):
        return np.asarray(a)[None] + np.linspace(0., 1., 17)[:, None] * (np.asarray(b) - np.asarray(a))[None]

    def arc_point(self, a, b, fraction):
        self.arc_calls.append((tuple(a), tuple(b), fraction))
        return tuple(np.asarray(a) + fraction * (np.asarray(b) - np.asarray(a)))

    def validate(self, graph, refine=False):
        self.calls.append((graph.to_dict(), refine))
        invalid = validate_graph(graph, previous=self.initial)
        rejected = (self.reject_edits and graph.to_dict() != self.initial.to_dict()) or (refine and self.reject_refined)
        valid = invalid is None and not rejected
        return {"valid": valid, "reason": "affine_test_geometry_valid" if valid else "physical_test_rejection",
                "quality": {"q": .8 if valid else 0., "block_count": len(graph.faces),
                            "outer_blocks": sum(not f.protected for f in graph.faces.values())}}

    def partition(self, graph):
        return {"schema": "affine_test_partition", "blocks": graph.to_dict()["faces"]}


@pytest.fixture
def fast_env(monkeypatch):
    monkeypatch.setattr(env_module, "CurveGeometry", AffineGeometry)
    return AirfoilBlockEnv()


def vertical_action(env, u):
    endpoints = {(u, .5), (u, 1.)}
    return next(i for i, candidate in enumerate(env.candidates)
                if candidate["kind"] == "insert_chord"
                and {tuple(candidate["source_uv"]), tuple(candidate["target_uv"])} == endpoints)


def test_full_fixed_schema_and_physical_candidate_validation(fast_env):
    env = fast_env
    obs, info = env.reset(seed=42)
    assert not env.terminated and env.queue == (4,)
    assert env.observation_space.contains(obs)
    assert obs["features"].shape == (256, 32)
    assert obs["global"].shape == (12,)
    assert obs["candidates"].shape == (1024, 16)
    assert obs["candidate_nodes"].shape == (1024, 2)
    assert len(env.candidates) > 1 and len(env.geometry.calls) >= len(env.candidates) + 1
    n = len(env.candidates)
    assert obs["mask"][:n].all() and not obs["mask"][n:].any()
    assert (obs["candidate_nodes"][n:] == -1).all()
    assert np.count_nonzero(obs["next"] >= 0) == 26
    assert np.count_nonzero(obs["features"][:, 13]) == 16
    assert np.count_nonzero(obs["features"][:, 24]) == 10
    for pair in obs["candidate_nodes"][:n]:
        assert (pair >= 0).all() and (obs["next"][pair] >= 0).all()
    assert info["environment_version"] == "airfoil_c_decomposition_v1"
    assert env.geometry.arc_calls


def test_candidate_snapshots_and_preview_are_independent(fast_env):
    env = fast_env
    env.reset()
    before = json.dumps(env.partition(), sort_keys=True)
    desc = env.candidates
    desc[0]["source_uv"][0] = 999
    assert env.candidates[0]["source_uv"][0] != 999
    mask = env.action_mask
    mask[:] = False
    assert env.action_mask.any()
    action = vertical_action(env, .25)
    preview = env.preview(action)
    assert json.dumps(env.partition(), sort_keys=True) == before
    preview["next_graph"]["vertices"]["0"][0] = 999
    assert env.graph.vertices[0][0] == 0
    obs, reward, terminal, truncated, info = env.step(action)
    assert reward == preview["reward"]
    assert terminal == preview["terminated"] and not truncated
    assert info["quality"] == preview["quality"]


def test_local_physical_tangents_and_face_outward_normals(fast_env):
    obs, _ = fast_env.reset()
    valid = obs["next"] >= 0
    tangent, normal = obs["features"][valid, 8:10], obs["features"][valid, 20:22]
    np.testing.assert_allclose(np.linalg.norm(tangent, axis=1), 1.)
    np.testing.assert_allclose(np.linalg.norm(normal, axis=1), 1.)
    np.testing.assert_allclose(np.sum(tangent * normal, axis=1), 0., atol=1e-7)
    for slot in np.flatnonzero(obs["twin"] >= 0):
        opposite = obs["twin"][slot]
        np.testing.assert_allclose(obs["features"][slot, 20:22], -obs["features"][opposite, 20:22])


@pytest.mark.parametrize("capacity,witness", [(1024, True), (33, False)])
def test_preview_witness_is_exact_and_never_cached_as_full_mask(fast_env, monkeypatch, capacity, witness):
    env = AirfoilBlockEnv(config=AirfoilDecompositionConfig(max_candidates=capacity))
    before_obs, _ = env.reset()
    before_graph, before_candidates = env.graph.to_dict(), env.candidates
    action = vertical_action(env, .25)
    next_candidate = env._candidates[action]
    next_key = env._candidate_cache_key(next_candidate.graph, next_candidate.queue)
    enumeration_modes = []
    original = env._enumerate_candidates

    def recorded(*args, **kwargs):
        enumeration_modes.append(kwargs.get("stop_after_first", False))
        return original(*args, **kwargs)

    monkeypatch.setattr(env, "_enumerate_candidates", recorded)
    preview = env.preview(action)
    assert enumeration_modes == [witness]
    assert (next_key in env._enumeration_cache) is (not witness)
    assert env.graph.to_dict() == before_graph and env.candidates == before_candidates
    for key, value in before_obs.items():
        np.testing.assert_array_equal(env._observation()[key], value)
    obs, reward, terminal, truncated, info = env.step(action)
    assert reward == preview["reward"] and terminal == preview["terminated"]
    assert info["reason"] == preview["reason"] and not truncated
    assert len(env.candidates) > 1 and obs["mask"].sum() == len(env.candidates)


def test_fixed_three_cut_completion_and_potential_telescoping(fast_env):
    env = fast_env
    env.reset()
    protected = {fid: copy.deepcopy(env.graph.faces[fid]) for fid in env.graph.protected_face_ids}
    start_potential = topology_potential(env.graph)
    shaping = []
    for u in (.25, .5, .75):
        obs, reward, terminal, truncated, info = env.step(vertical_action(env, u))
        c = info["reward_components"]
        assert reward == pytest.approx(info["base_reward"] + info["shaping"])
        assert c["total"] == reward
        assert c["shaping"] == pytest.approx(env.config.gamma * c["potential_after"] - c["potential_before"])
        shaping.append(info["shaping"])
    assert terminal and not truncated and info["success"]
    assert env.queue == () and len(env.graph.faces) == 8
    assert not obs["mask"].any() and env.candidates == ()
    assert all(len(face.vertices) == 4 for face in env.graph.faces.values())
    assert all(env.graph.faces[fid] == face for fid, face in protected.items())
    assert info["base_reward"] == pytest.approx(5 + .8 - .02 * 4 - .02)
    assert info["reward_components"]["potential_after"] == 0.
    assert sum(env.config.gamma**i * value for i, value in enumerate(shaping)) == pytest.approx(-start_potential)
    assert any(refine for _, refine in env.geometry.calls)
    assert len(env.partition()["cut_history"]) == 3
    with pytest.raises(RuntimeError):
        env.step(0)


def test_fifo_parent_replacement_keeps_older_sibling_first(fast_env):
    env = fast_env
    env.reset()
    env.step(vertical_action(env, .5))
    assert len(env.queue) == 2
    other = env.queue[1]
    env.step(vertical_action(env, .75))
    assert env.queue[0] == other
    obs = env._observation()
    assert sum(obs["features"][:, 25]) == sum(len(env.graph.faces[fid].vertices) for fid in env.queue)
    assert np.count_nonzero(obs["next"] >= 0) == sum(len(face.vertices) for face in env.graph.faces.values())


def test_shared_arc_insertion_propagates_and_requeues_completed_quad(fast_env):
    env = fast_env
    env.reset()
    env.step(vertical_action(env, .25))
    before = env.graph.clone()
    action, description = next((i, c) for i, c in enumerate(env.candidates)
        if c["kind"] == "insert_edge_point" and len(c["owners"]) == 2 and c["fraction"] == .5)
    old_queue = env.queue
    completed = next(fid for fid in description["owners"] if len(before.faces[fid].vertices) == 4)
    _, _, terminal, _, _ = env.step(action)
    new_id = before.next_vertex_id
    assert not terminal and len(env.graph.vertices) == len(before.vertices) + 1
    assert all(new_id in env.graph.faces[fid].vertices for fid in description["owners"])
    assert len(env.graph.faces[completed].vertices) == 5
    assert env.queue[0] == old_queue[0] and env.queue[-1] == completed
    assert env.graph.vertices[new_id] == pytest.approx(description["point_uv"])
    assert all(env.graph.faces[fid] == before.faces[fid] for fid in before.protected_face_ids)
    assert all(len(owners) <= 2 for owners in edge_incidence(env.graph).values())


def test_no_protected_edge_insertions_and_no_triangular_children(fast_env):
    env = fast_env
    env.reset()
    protected = protected_edges(env.graph)
    seen = set()
    for candidate in env._candidates:
        desc = candidate.description
        identity = (desc["kind"], desc["source"], desc["target"], desc.get("fraction"))
        assert identity not in seen
        seen.add(identity)
        if desc["kind"] == "insert_edge_point":
            assert tuple(sorted((desc["source"], desc["target"]))) not in protected
        assert all(len(face.vertices) >= 4 for face in candidate.graph.faces.values())


@pytest.mark.parametrize("invalid", [-1, 1024, True, 1.5, "0"])
def test_invalid_external_action_rolls_back_and_terminates(fast_env, invalid):
    env = fast_env
    env.reset()
    before, queue = env.graph.to_dict(), env.queue
    obs, reward, terminal, truncated, info = env.step(invalid)
    assert terminal and not truncated and not info["success"]
    assert info["reason"] == "invalid_action" and info["invalid_action"]
    assert env.graph.to_dict() == before and env.queue == queue
    assert not obs["mask"].any() and env.partition()["cut_history"] == []
    assert info["base_reward"] == pytest.approx(-5.02)
    assert reward == pytest.approx(-5.02 + .1875)
    assert info["reward_components"]["potential_after"] == 0.


def test_budget_failure_preserves_last_valid_edit(fast_env):
    env = AirfoilBlockEnv(config=AirfoilDecompositionConfig(max_steps=1))
    env.reset()
    obs, _, terminal, truncated, info = env.step(vertical_action(env, .25))
    assert terminal and not truncated and not info["success"]
    assert info["reason"] == "step_budget_exhausted" and len(env.graph.faces) == 6
    assert info["base_reward"] == pytest.approx(-5.02)
    assert info["reward_components"]["potential_after"] == 0. and not obs["mask"].any()


@pytest.mark.parametrize("setting,expected", [
    ({"max_vertices": 14}, "vertices_budget_exhausted"),
    ({"max_blocks": 4}, "blocks_budget_exhausted"),
    ({"max_face_vertices": 9}, "face_vertices_budget_exhausted"),
    ({"max_half_edges": 25}, "half_edges_budget_exhausted"),
    ({"max_candidates": 1}, "candidate_budget_exhausted"),
])
def test_capacity_failure_never_truncates_observation(fast_env, setting, expected):
    env = AirfoilBlockEnv(config=AirfoilDecompositionConfig(**setting))
    obs, info = env.reset()
    assert env.terminated and not env.success and info["reason"] == expected
    assert not obs["mask"].any() and len(env.graph.vertices) == 15
    assert np.count_nonzero(obs["next"] >= 0) == 26
    assert obs["features"].shape == (256, 32) and obs["candidate_nodes"].shape == (1024, 2)


def test_physically_rejected_edits_are_masked(fast_env):
    env = fast_env
    env.reset()
    env.geometry.reject_edits = True
    env._enumeration_cache.clear()
    obs, info = env.reset()
    assert env.terminated and not info["success"]
    assert info["reason"] == "no_legal_actions" and not obs["mask"].any()
    assert env.graph.to_dict() == initial_graph(env.geometry.domain).to_dict()


def test_refined_terminal_failure_rolls_back_and_records_attempt(fast_env):
    env = fast_env
    env.reset()
    env.step(vertical_action(env, .25))
    env.step(vertical_action(env, .5))
    before, queue = env.graph.to_dict(), env.queue
    env.geometry.reject_refined = True
    preview = env.preview(vertical_action(env, .75))
    assert preview["terminated"] and not preview["success"]
    assert env.graph.to_dict() == before
    obs, reward, terminal, truncated, info = env.step(vertical_action(env, .75))
    assert terminal and not truncated and not info["success"]
    assert info["reason"] == "terminal_physical_validation:physical_test_rejection"
    assert env.graph.to_dict() == before and env.queue == queue
    assert info["attempt_validation"]["valid"] is False
    assert env.partition()["cut_history"][-1]["accepted"] is False
    assert info["base_reward"] == pytest.approx(-5.02) and reward == preview["reward"]
    assert not obs["mask"].any()


def test_deterministic_resets_reuse_geometry_and_candidate_cache(fast_env):
    env = fast_env
    first, _ = env.reset(seed=42)
    geometry = env.geometry
    candidates = env.candidates
    calls = len(geometry.calls)
    second, _ = env.reset(seed=42)
    assert env.geometry is geometry and env.candidates == candidates
    assert len(geometry.calls) == calls + 1  # reset validation only; candidate cache reused
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    another = synthetic_profile()
    another["sample_id"] = "second_analytic_fixture"
    env.reset(options={"sample": another})
    assert env.geometry is not geometry


def test_real_physical_geometry_fixed_c_partition_and_gym_schema():
    env = AirfoilBlockEnv(sample=synthetic_profile())
    obs, info = env.reset(seed=42)
    assert not env.terminated, info
    assert env.observation_space.contains(obs)
    for u in (.25, .5, .75):
        obs, reward, terminal, truncated, info = env.step(vertical_action(env, u))
    assert terminal and not truncated and info["success"], info
    assert 0 < info["quality"]["q"] <= 1 and info["quality"]["outer_blocks"] == 4
    partition = env.partition()
    assert partition["valid"] and len(partition["blocks"]) == 8
    assert partition["interfaces"] and partition["wake_pairs"]
    assert all("curve_refs" in block for block in partition["blocks"])
    json.dumps(partition, allow_nan=False)
    from gymnasium.utils.env_checker import check_env
    check_env(env, skip_render_check=True)


def test_deterministic_refined_witness_produces_distinct_twelve_blocks(fast_env):
    from smartcfd.airfoil_decomposition.baselines import refined_c_action
    from smartcfd.partition.topology import fingerprint
    env = fast_env
    env.reset()
    for u in (.25, .5, .75):
        env.step(vertical_action(env, u))
    fixed = fingerprint(env.graph)
    env.reset()
    actions = []
    while not env.terminated:
        action = refined_c_action(env)
        assert action is not None and env.action_mask[action]
        actions.append(env.candidates[action])
        env.step(action)
    assert env.success and env.steps == 12
    assert len(env.graph.faces) == 12 and len(env.graph.protected_face_ids) == 4
    assert all(len(face.vertices) == 4 for face in env.graph.faces.values())
    assert fingerprint(env.graph) != fixed
    assert sum(item["kind"] == "insert_edge_point" for item in actions) == 5
    assert refined_c_action(env) is None
