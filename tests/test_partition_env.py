"""Conforming coarse topology, real meshing, teacher replay and failure semantics."""
import copy

import numpy as np
import pytest

from smartcfd.core.types import Domain, EnvConfig, Failure
from smartcfd.partition.actions import describe_action
from smartcfd.partition.constraints import validate_graph
from smartcfd.partition.topology import (
    all_quadrilateral, edge_incidence, fingerprint, half_edges, initial_graph,
)
from smartcfd.partition.transition import apply_action
from smartcfd.baselines.rule_partition import rule_graph, solve
from smartcfd.rl.env import BlockEnv


def synthetic_domain(name="synthetic_lens"):
    # An analytic fixture, independent of all frozen UIUC/NACA test geometries.
    x = np.linspace(0.0, 1.0, 41)
    top = np.column_stack((x[::-1], 0.08 * np.sin(np.pi * x[::-1])))
    lower = np.column_stack((x[1:], -0.08 * np.sin(np.pi * x[1:])))
    polygon = np.concatenate((top, lower))[:-1]
    return Domain(name, polygon)


def find_action(graph, kind, a, b):
    for edge in half_edges(graph):
        for local in range(4):
            action = edge.slot * 4 + local
            item = describe_action(graph, action)
            if item["kind"] != kind:
                continue
            ends = (tuple(item["source_uv"]), tuple(item["target_uv"]))
            if ends in ((a, b), (b, a)):
                candidate = apply_action(graph, action)
                if not isinstance(candidate, Failure):
                    return action
    raise AssertionError("Required fixture edit unavailable")


def protected_snapshot(graph):
    return {face.id: (face.vertices, tuple(graph.vertices[v] for v in face.vertices))
            for face in graph.faces.values() if face.protected}


def test_initial_domain_is_unresolved_and_protected_band_is_four_real_blocks():
    graph = initial_graph()
    assert validate_graph(graph) is None
    assert len(graph.faces) == 5 and not all_quadrilateral(graph)
    assert len(graph.faces[4].vertices) == 10
    assert len(graph.protected_face_ids) == 4
    assert len(half_edges(graph)) == 26


def test_protected_and_geometrically_collinear_chords_are_transactionally_rejected():
    graph = initial_graph()
    before = graph.to_dict()
    assert isinstance(apply_action(graph, 0), Failure)
    # The first outer edges lie on a straight protected interface, so the
    # next-two-vertex chord is not an interior split.
    assert isinstance(apply_action(graph, 16 * 4), Failure)
    assert graph.to_dict() == before
    action = find_action(graph, "insert_chord", (.25, .5), (.25, 1.))
    limited = apply_action(graph, action, max_blocks=5)
    assert isinstance(limited, Failure) and limited.reason == "block_budget"
    assert graph.to_dict() == before


def test_internal_midpoint_is_inserted_in_both_owners_without_touching_protection():
    graph = initial_graph()
    before = protected_snapshot(graph)
    action = find_action(graph, "insert_chord", (.25, .5), (.25, 1.))
    graph = apply_action(graph, action)
    owners_before = {fid: len(f.vertices) for fid, f in graph.faces.items()}
    action = find_action(graph, "insert_edge_midpoint", (.25, .5), (.25, 1.))
    new = apply_action(graph, action)
    assert not isinstance(new, Failure)
    midpoint = next(v for v, uv in new.vertices.items() if uv == (.25, .75))
    touched = [f.id for f in new.faces.values() if midpoint in f.vertices]
    assert len(touched) == 2
    assert all(len(new.faces[fid].vertices) == owners_before[fid] + 1 for fid in touched)
    assert protected_snapshot(new) == before
    assert validate_graph(new) is None
    assert all(len(owners) <= 2 for owners in edge_incidence(new).values())


def test_full_observation_shapes_padding_twins_and_actual_action_count():
    env = BlockEnv([synthetic_domain()])
    obs, info = env.reset(seed=13)
    assert env.action_space.n == 1024
    assert env.observation_space.contains(obs)
    assert obs["features"].shape == (256, 24)
    assert obs["mask"].shape == (1024,) and obs["mask"].dtype == np.bool_
    assert np.count_nonzero(obs["features"][:, 23]) == 26
    assert not obs["mask"][:64].any()
    assert not obs["mask"][26 * 4:].any()
    for edge in env.registry:
        assert obs["previous"][obs["next"][edge.slot]] == edge.slot
        if edge.twin >= 0:
            assert obs["twin"][edge.twin] == edge.slot
    assert np.all(obs["next"][26:] == -1)
    assert info["success"] is False and obs["mask"].any()


def test_two_teachers_replay_through_real_mesh_generation_and_distinct_topologies():
    env = BlockEnv([synthetic_domain()])
    endings = []
    for seed, expected_blocks, expected_actions in ((0, 8, 3), (1, 12, 12)):
        start, _ = env.reset(seed=seed, options={"domain_index": 0})
        protected = protected_snapshot(env.graph)
        before = fingerprint(env.graph)
        actions = solve(env, seed=seed)
        assert not isinstance(actions, Failure), actions
        assert len(actions) == expected_actions
        assert fingerprint(env.graph) == before and env.steps == 0
        for action in actions:
            assert env.action_mask[action]
            obs, reward, done, truncated, info = env.step(action)
            assert np.isfinite(reward)
        assert done and not truncated and info["success"]
        assert len(env.graph.faces) == expected_blocks
        assert protected_snapshot(env.graph) == protected
        assert all_quadrilateral(env.graph) and validate_graph(env.graph) is None
        assert env.last_mesh.quality["valid"]
        assert env.last_mesh.quality["q10_scaled_jacobian"] > 0
        assert not obs["mask"].any()
        endings.append(fingerprint(env.graph))
    assert endings[0] != endings[1]


def test_geometry_legality_does_not_override_actual_mesh_failure(monkeypatch):
    env = BlockEnv([synthetic_domain()])
    env.reset(seed=0)
    actions = solve(env, seed=0)
    assert isinstance(actions, list)
    for action in actions[:-1]:
        env.step(action)
    # All-quads alone must never become success when actual meshing fails.
    monkeypatch.setattr(env, "_mesh_for", lambda graph: Failure("injected_mesh_failure"))
    env._cache = {}
    env._refresh_mask()
    assert env.action_mask[actions[-1]]
    obs, reward, done, truncated, info = env.step(actions[-1])
    assert done and not truncated and not info["success"] and reward < 0
    assert info["reason"] == "mesh_generation_failed:injected_mesh_failure"
    assert all_quadrilateral(env.graph) and env.last_mesh is None
    assert not obs["mask"].any()


def test_budget_exhaustion_is_a_finite_task_terminal_and_never_success():
    env = BlockEnv([synthetic_domain()], EnvConfig(max_steps=1))
    env.reset(seed=0)
    action = env.valid_actions()[0]
    obs, reward, done, truncated, info = env.step(action)
    assert done and not truncated and not info["success"]
    assert info["reason"] == "operation_budget_exhausted"
    assert not obs["mask"].any() and reward < 0
    with pytest.raises(RuntimeError):
        env.step(action)


def test_invalid_action_keeps_graph_and_still_consumes_budget():
    env = BlockEnv([synthetic_domain()])
    env.reset(seed=0)
    before = env.graph.to_dict()
    _, reward, done, _, info = env.step(0)
    assert info["invalid_action"] and not done and reward < 0
    assert env.graph.to_dict() == before and env.steps == 1


def test_reset_sample_selection_is_explicit_and_reproducible():
    env = BlockEnv([synthetic_domain("a"), synthetic_domain("b")])
    a, ia = env.reset(seed=7)
    b, ib = env.reset(seed=7)
    assert ia == ib and all(np.array_equal(a[k], b[k]) for k in a)
    _, fixed = env.reset(seed=9, options={"sample_id": "b"})
    assert fixed["sample_id"] == "b" and fixed["domain_index"] == 1


def test_teacher_budget_is_explicit_failure_not_an_unvalidated_solution():
    env = BlockEnv([synthetic_domain()])
    env.reset(seed=0)
    result = solve(env, seed=0, max_nodes=1)
    assert isinstance(result, Failure) and result.reason == "teacher_search_budget"
    assert env.steps == 0 and not all_quadrilateral(env.graph)
