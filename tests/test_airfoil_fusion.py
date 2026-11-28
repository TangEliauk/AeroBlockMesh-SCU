"""End-to-end environment, mask, training and mesh checks for the fusion protocol."""
import numpy as np
import pytest
import torch
from dataclasses import asdict

from smartcfd.airfoil_decomposition.config import AirfoilDecompositionConfig
from smartcfd.airfoil_decomposition.env import synthetic_profile
from smartcfd.airfoil_decomposition.experiment import fixed_action, rule_action
from smartcfd.airfoil_fusion import cfd
from smartcfd.airfoil_fusion.env import FusionEnv, GuardedFusionEnv
from smartcfd.airfoil_fusion.learning import FusionSAC, bc_update, candidate_batch
from smartcfd.airfoil_fusion.mesh import evaluate_mesh_budget
from smartcfd.airfoil_fusion.rollout import action_key, resolve, stop_action
from smartcfd.airfoil_fusion.train import state_hash, train
from smartcfd.cfd.solver import export_su2
from smartcfd.core.types import Transition


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_solver_path_accepts_configured_executable_and_path(tmp_path, monkeypatch):
    solver = tmp_path / "solver"
    solver.touch()
    monkeypatch.setattr(cfd, "ROOT", tmp_path)
    monkeypatch.setenv("AEROBLOCK_SU2", str(solver))
    assert cfd.solver_path() == solver.resolve()

    monkeypatch.delenv("AEROBLOCK_SU2")
    monkeypatch.setattr(cfd.shutil, "which", lambda name: str(solver) if name == "SU2_CFD" else None)
    assert cfd.solver_path() == solver.resolve()

    monkeypatch.setenv("AEROBLOCK_SU2", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="AEROBLOCK_SU2"):
        cfd.solver_path()


def _fixed_steps(env, count=3):
    rows = []
    obs, _ = env.reset(seed=42)
    for _ in range(count):
        action = fixed_action(env)
        assert action is not None
        after, reward, done, truncated, _ = env.step(action)
        rows.append(Transition(obs, action, reward, after, done, truncated))
        obs = after
    return rows


def test_global_targets_include_two_reopened_unprotected_faces():
    env = FusionEnv(mesh_reward=False)
    _fixed_steps(env, 1)
    midpoint = next(i for i,c in enumerate(env.candidates) if c["kind"] == "insert_edge_point"
                    and len(c["owners"]) == 2 and c["fraction"] == .5)
    graph_before = env.graph.clone()
    obs, _, done, _, _ = env.step(midpoint)
    assert not done and env.observation_space.contains(obs)
    candidates = [c for i, c in enumerate(env.candidates) if obs["mask"][i]]
    assert len({c["face_id"] for c in candidates}.intersection(env.queue)) >= 2
    assert len({action_key(c) for c in candidates}) == len(candidates)
    changed = set(env.graph.vertices) - set(graph_before.vertices)
    assert len(changed) == 1
    assert sum(changed <= set(face.vertices) for face in env.graph.faces.values()) == 2
    assert {i: env.graph.faces[i] for i in graph_before.protected_face_ids} == {
        i: graph_before.faces[i] for i in graph_before.protected_face_ids}


def test_stop_is_explicit_and_reserved_after_last_edit():
    env = FusionEnv(config={"max_steps": 3})
    trace = _fixed_steps(env)
    assert all(not row.terminated for row in trace)
    assert stop_action(env) == 0 and len(env.candidates) == 1
    before = env.graph.to_dict()
    obs, reward, done, _, info = env.step(0)
    assert done and info["success"] and (info["edits"], info["steps"]) == (3, 4)
    assert info["reward_components"]["potential_after"] == 0
    assert info["reward_components"]["step_cost"] == 0 and reward > 0
    assert env.graph.to_dict() == before and not obs["mask"].any()
    optional = FusionEnv(mesh_reward=False)
    _fixed_steps(optional)
    assert stop_action(optional) is not None and any(c["kind"] == "insert_edge_point" for c in optional.candidates)


def test_guard_matches_exhaustive_single_edit_oracle_at_budget():
    primitive = FusionEnv(config={"max_steps": 3})
    guarded = GuardedFusionEnv(config={"max_steps": 3}, guard_config={"second_layer_budget": 128})
    _fixed_steps(primitive, 2)
    _fixed_steps(guarded, 2)
    assert [action_key(c) for c in primitive.candidates] == [action_key(c) for c in guarded.candidates]
    for index, candidate in enumerate(primitive.candidates):
        branch = primitive.fork()
        branch.step(resolve(branch, candidate))
        reachable = not branch.terminated and stop_action(branch) is not None
        assert bool(guarded.action_mask[index]) == reachable


def test_policy_transfer_temperature_and_terminal_targets():
    env = FusionEnv()
    trace = _fixed_steps(env)
    previous = trace[-1].next_observation
    action = stop_action(env)
    obs, reward, done, truncated, _ = env.step(action)
    trace.append(Transition(previous, action, reward, obs, done, truncated))
    actor, online = FusionSAC(), FusionSAC(seed=43)
    assert bc_update(actor, [(r.observation, r.action) for r in trace])["loss"] > 0
    online.load_actor(actor.actor.state_dict())
    torch.testing.assert_close(actor.actor(previous), online.actor(previous), atol=0, rtol=0)
    batch = candidate_batch([trace[0].observation, previous, obs], online.device)
    assert batch["mask"].sum(1).tolist()[-1] == 0
    assert online.compute_targets([trace[-1]]).item() == pytest.approx(reward)
    before = online.actor(previous).detach()
    online.update_critic(trace)
    torch.testing.assert_close(online.actor(previous), before, atol=0, rtol=0)
    assert online.update(trace)["alpha"] != pytest.approx(.01, abs=1e-8)


def test_replay_state_and_conformal_mesh_targets(tmp_path):
    env, reloaded = FusionEnv(), FusionEnv()
    _fixed_steps(env)
    obs, _ = reloaded.reset(seed=42)
    for item in env._history:
        obs, *_ = reloaded.step(resolve(reloaded, item))
    assert state_hash(obs) == state_hash(env._observation())
    env.step(stop_action(env))
    for budget in (1536, 5120):
        metrics, mesh = evaluate_mesh_budget(env.geometry.domain, env.graph, budget)
        assert metrics["quads"] == budget and metrics["budget_match"]
        assert metrics["min_scaled_jacobian"] > 0 and np.isfinite(metrics["score"])
        assert metrics["interface_max_abs_difference"] == 0
        path = tmp_path/f"{budget}.su2"
        stats = export_su2(mesh, path)
        assert stats["cells"] == budget and set(stats["markers"]) == {"wall", "farfield"}


def test_bounded_greedy_is_exact_at_live_and_stop_states():
    env = FusionEnv()
    for steps in (0, 2, 3):
        _fixed_steps(env, steps)
        assert env.greedy_action() == rule_action(env)


def test_resumed_sac_has_identical_interactions_and_weights(tmp_path):
    a, b = tmp_path/"continuous", tmp_path/"resumed"
    for path in a,b:
        path.mkdir()
        (path/"prepared.json").write_text("same input identity", encoding="utf-8")
    profile = synthetic_profile()
    cases = {split: [{**profile, "split": split, "group_id": split, "sample_id": split}] for split in ("train","val")}
    checked = {split: [{"sample_id": split, "valid": True}] for split in cases}
    cfg = {"geometry": asdict(AirfoilDecompositionConfig(max_steps=3)), "first_steps": 2,
           "extended_steps": 4, "online_warmup": 1, "validation_every": 2,
           "runtime": {"torch_threads": 1, "checkpoint_every": 2, "batch_size": 1,
                       "replay_capacity": 16, "replay_max_bytes": 1024**2}}
    continuous = train(a, cfg, cases, checked, "E6", 42, 4)
    train(b, cfg, cases, checked, "E6", 42, 2)
    resumed = train(b, cfg, cases, checked, "E6", 42, 4)
    assert continuous["updates"] == resumed["updates"] == 4
    assert continuous["validation"][-1]["network_sha256"] == resumed["validation"][-1]["network_sha256"]
    assert continuous["best_step"] == resumed["best_step"]


def test_official_cases_cannot_be_loaded_before_training_is_sealed(tmp_path, monkeypatch):
    from smartcfd.airfoil_fusion import evaluate

    def unexpected_holdout():
        pytest.fail("Test geometry read before the sealed weight matrix")

    monkeypatch.setattr(evaluate, "official_cases", unexpected_holdout)
    with pytest.raises(FileNotFoundError, match="training_matrix.json"):
        evaluate.lock_testing(tmp_path, {"profile": "formal"}, None, None)


def test_budget_pairing_retains_partition_failures_in_primary_denominator():
    from smartcfd.airfoil_fusion.evaluate import paired

    quality = dict.fromkeys(("score", "min_scaled_jacobian", "q05_scaled_jacobian",
                             "wall_angle_q10_deg", "farfield_max_aspect_ratio", "max_aspect_ratio"), 0.5)

    def result(cells):
        return {"success": True, "mesh": {"1536": {**quality, "budget_match": True, "quads": cells}}}

    left = {"a": result(1536), "b": result(1490), "c": {"success": False}}
    right = {"a": result(1540), "b": result(1575), "c": result(1536)}
    summary = paired(left, right, 1536)
    assert (summary["common_cases"], summary["both_partitioned"], summary["matched_meshes"]) == (3, 2, 1)
    assert summary["success_delta"] == pytest.approx(-1/3)
    assert summary["rows"][0]["sample_id"] == "a"


def test_report_matches_uv_partitions_independently_of_graph_ids(tmp_path, monkeypatch):
    from smartcfd.airfoil_fusion import report

    fixed = {"vertices": {"0": (0, 0), "1": (1, 0), "2": (1, 1), "3": (0, 1)},
             "faces": [{"id": 0, "vertices": [0, 1, 2, 3]}]}
    renumbered = {"vertices": {"4": (1, 1), "5": (0, 1), "6": (0, 0), "7": (1, 0)},
                  "faces": [{"id": 9, "vertices": [4, 5, 6, 7]}]}
    changed = {**renumbered, "vertices": {**renumbered["vertices"], "4": (1, .75)}}

    def evaluation(_, arm, seed=42):
        if arm == "E1":
            graph = fixed
        elif arm == "E4":
            graph = {42: renumbered, 43: changed}[seed]
        else:
            return {"rows": []}
        return {"rows": [{"sample_id": "demo", "success": True, "graph": graph}]}

    monkeypatch.setattr(report, "_evaluation", evaluation)
    result = report._topology_diversity(tmp_path, {"seeds": [42, 43]})["methods"][0]
    assert result["joint_success_with_fixed"] == 2
    assert result["canonical_graph_matches_with_fixed"] == 1
    assert result["exact_graph_matches_with_fixed"] == 0
    assert result["cases_with_different_uv_partitions"] == 1
    assert result["cases_with_different_connectivity_signatures"] == 0
    assert result["cases"][0]["canonical_fixed_match_seeds"] == [42]
