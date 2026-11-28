"""Physical full-line splits and complete queued-state observation contracts."""
from dataclasses import asdict
import json

import numpy as np
import pytest
from shapely.geometry import Polygon

from smartcfd.decomposition import DecompositionConfig, OrthogonalBlockEnv, generate_cases
from smartcfd.decomposition.geometry import (GeometryError, canonical_vertices, is_rectangle,
    normalized_vertices, paper_reward, partition_interfaces, partition_quality, polygon_hash, split_polygon)


U = [(0, 0), (4, 0), (4, 4), (3, 4), (3, 1), (1, 1), (1, 4), (0, 4), (0, 0)]
STAIR = [(0, 0), (5, 0), (5, 1), (4, 1), (4, 2), (3, 2), (3, 3),
         (2, 3), (2, 4), (1, 4), (1, 5), (0, 5), (0, 0)]


def case(vertices=U):
    return {"case_id": "unit_case", "family": "U", "vertices": vertices}


def test_canonical_closure_orientation_and_collinear_removal():
    clockwise = [(3, 2), (3, 0), (2, 0), (0, 0), (0, 2), (3, 2)]
    actual = canonical_vertices(clockwise)
    assert actual == ((0., 0.), (3., 0.), (3., 2.), (0., 2.), (0., 0.))
    assert Polygon(actual).exterior.is_ccw
    assert canonical_vertices(clockwise[:-1]) == actual


@pytest.mark.parametrize("vertices", [
    [(0, 0), (2, 0), (2, 1), (1, 2), (0, 0)],
    [(0, 0), (3, 0), (3, 3), (1, 3), (1, -1), (2, -1), (2, 2), (0, 2)],
    [(0, 0), (float("nan"), 0), (1, 1), (0, 1)],
    [(0, 0), (float("inf"), 0), (1, 1), (0, 1)],
    [(0, 0), (1e300, 0), (1e300, 1e-300), (0, 1e-300)],
    [(0, 0), (1, 0), (2, 0), (3, 0)],
    [[(0, 0), (4, 0), (4, 4), (0, 4)], [(1, 1), (2, 1), (2, 2), (1, 2)]],
])
def test_invalid_geometry_is_rejected(vertices):
    with pytest.raises(GeometryError):
        canonical_vertices(vertices)


def test_identity_preserves_aspect_and_uniform_coordinate_invariance():
    transformed = [(7 + 3 * x, -11 + 3 * y) for x, y in U]
    anisotropic = [(2 * x, y) for x, y in U]
    assert polygon_hash(U) == polygon_hash(transformed)
    assert polygon_hash(U) != polygon_hash(anisotropic)
    np.testing.assert_allclose(normalized_vertices(U), normalized_vertices(transformed), atol=1e-14)


def test_real_full_line_cut_can_create_three_rectangles():
    ring = canonical_vertices(U)
    index = ring.index((3., 1.))
    result = split_polygon(ring, index, 0)
    assert len(result.children) == 3
    assert all(is_rectangle(child) for child in result.children)
    quality = partition_quality(U, result.children)
    assert quality["valid"] and quality["block_count"] == quality["rectangle_count"] == 3
    assert quality["overlap_area"] == quality["coverage_error_area"] == quality["sum_area_error"] == 0
    assert sum(Polygon(child).area for child in result.children) == Polygon(U).area
    assert len(result.segments) == 2


def test_boundary_no_effect_and_sliver_cuts_are_rejected():
    with pytest.raises(GeometryError, match="no positive-area"):
        split_polygon(U, 0, 0)
    sliver = [(0, 0), (4, 0), (4, 4), (3, 4), (3, 1), (1e-12, 1), (1e-12, 4), (0, 4)]
    index = canonical_vertices(sliver).index((3., 1.))
    with pytest.raises(GeometryError, match="sliver"):
        split_polygon(sliver, index, 0)


def test_area_validation_detects_overlap_and_missing_coverage():
    root = [(0, 0), (4, 0), (4, 4), (0, 4)]
    left = [(0, 0), (3, 0), (3, 4), (0, 4)]
    right = [(2, 0), (4, 0), (4, 4), (2, 4)]
    quality = partition_quality(root, [left, right])
    assert not quality["valid"] and quality["overlap_area"] == 4
    quality = partition_quality(root, [left])
    assert not quality["valid"] and quality["coverage_error_area"] == 4


def test_observation_full_state_mask_dedup_and_terminal_three_piece_cut():
    env = OrthogonalBlockEnv(case())
    obs, info = env.reset(seed=42)
    assert not info["success"]
    assert env.observation_space.contains(obs)
    assert obs["features"].shape == (128, 24) and obs["mask"].shape == (256,)
    assert obs["mask"].dtype == np.bool_ and obs["next"].dtype == np.int64
    assert np.all(obs["twin"] == -1)
    index = env.pending[0].index((3., 1.))
    action = index * 2
    duplicate = env.pending[0].index((1., 1.)) * 2
    assert obs["mask"][action] and not obs["mask"][duplicate]
    before = json.dumps(env.partition(), sort_keys=True)
    preview = env.peek(action)
    assert len(preview["children"]) == 3
    assert json.dumps(env.partition(), sort_keys=True) == before
    obs, reward, terminated, truncated, info = env.step(action)
    assert terminated and not truncated and info["success"]
    assert reward == preview["reward"]
    assert reward == sum(value for key, value in info["reward_components"].items() if key != "total")
    assert info["reward_components"]["block_cost"] == -.1
    assert info["reward_components"]["terminal_bonus"] == 5
    assert not obs["mask"].any()
    assert np.count_nonzero(obs["features"][:, 12]) == 12
    assert np.count_nonzero(obs["next"] >= 0) == 12
    assert len(env.partition()["blocks"]) == 3 and not env.partition()["pending"]
    json.dumps(env.partition(), allow_nan=False)
    with pytest.raises(RuntimeError):
        env.step(action)


def test_fifo_queue_keeps_both_nonrectangle_children_and_completed_rings():
    env = OrthogonalBlockEnv(case(STAIR))
    obs, _ = env.reset(seed=7)
    candidates = [int(a) for a in np.flatnonzero(obs["mask"])
                  if sum(not is_rectangle(child) for child in env.peek(int(a))["children"]) >= 2]
    assert candidates, "A five-step staircase must admit a full cut with two nonrectangular children"
    env.step(candidates[0])
    assert len(env.pending) >= 2
    older = env.pending[1]
    obs = env._observation()
    assert np.count_nonzero(obs["features"][:, 11]) == sum(len(ring) - 1 for ring in env.pending)
    assert np.count_nonzero(obs["features"][:, 10]) == len(env.pending[0]) - 1
    assert all(int(a) // 2 < len(env.pending[0]) - 1 for a in np.flatnonzero(obs["mask"]))
    env.step(int(np.flatnonzero(obs["mask"])[0]))
    assert env.pending[0] == older
    obs = env._observation()
    assert np.count_nonzero(obs["next"] >= 0) == sum(len(ring) - 1 for ring in [*env.pending, *env.blocks])
    for slot in np.flatnonzero(obs["next"] >= 0):
        assert obs["previous"][obs["next"][slot]] == slot


def test_invalid_action_step_cost_and_real_budget_failure():
    env = OrthogonalBlockEnv(case(), DecompositionConfig(max_steps=1))
    env.reset()
    before = env.partition()["pending"]
    obs, reward, ended, truncated, info = env.step(255)
    assert ended and not truncated and not info["success"]
    assert info["reason"] == "step_budget_exhausted" and reward == -5.05
    assert env.partition()["pending"] == before and not obs["mask"].any()
    env.reset()
    action = next(int(a) for a in np.flatnonzero(env.action_mask) if not env.peek(int(a))["success"])
    obs, _, ended, truncated, info = env.step(action)
    assert ended and not truncated and not info["success"] and info["quality"]["valid"]
    assert info["reason"] == "step_budget_exhausted"
    assert info["reward_components"]["terminal_bonus"] == 0


def test_capacity_never_silently_truncates_or_discards_geometry():
    with pytest.raises(GeometryError, match="exceeds max_vertices"):
        OrthogonalBlockEnv(case(), DecompositionConfig(max_vertices=4)).reset()
    env = OrthogonalBlockEnv(case(), DecompositionConfig(max_vertices=8, max_blocks=1))
    obs, info = env.reset()
    assert info["terminated"] and not info["success"] and not obs["mask"].any()
    assert np.count_nonzero(obs["next"] >= 0) == 8
    assert env.partition()["pending"] == [[list(point) for point in canonical_vertices(U)]]


def test_initial_rectangle_success_requires_no_step():
    env = OrthogonalBlockEnv(case([(0, 0), (2, 0), (2, 3), (0, 3)]))
    obs, info = env.reset()
    assert info["success"] and info["reason"] == "initially_complete"
    assert env.steps == 0 and not obs["mask"].any() and not env.partition()["pending"]
    with pytest.raises(RuntimeError):
        env.step(0)


def test_paper_equation_components_use_all_three_actual_children():
    result = split_polygon(U, canonical_vertices(U).index((3., 1.)), 0)
    components = paper_reward(result.children)
    ratios = [max(Polygon(c).bounds[2] - Polygon(c).bounds[0], Polygon(c).bounds[3] - Polygon(c).bounds[1]) /
              min(Polygon(c).bounds[2] - Polygon(c).bounds[0], Polygon(c).bounds[3] - Polygon(c).bounds[1]) for c in result.children]
    areas = [Polygon(child).area for child in result.children]
    expected = 3 * ((3 / sum(value**2 for value in ratios))**.5 - sum((value - sum(areas) / 3)**2 for value in areas)**.5 / sum(areas) - 1) + 10
    assert components["children"] == components["rectangle_children"] == 3
    assert components["paper_reward"] == pytest.approx(expected)


def test_t_junction_export_splits_long_neighbor_interface():
    parts = [[(0, 0), (2, 0), (2, 4), (0, 4)], [(2, 0), (4, 0), (4, 2), (2, 2)],
             [(2, 2), (4, 2), (4, 4), (2, 4)]]
    result = partition_interfaces(parts)
    assert {"point": [2., 2.], "parts": [0, 1, 2], "kind": "T", "ray_count": 3} in result["junctions"]
    for interface in result["interfaces"]:
        if interface["part_a"] == 0:
            assert (2., 2.) in interface["common_points"]
    json.dumps(result, allow_nan=False)


def test_seeded_continuous_cases_deduplicate_groups_and_cover_families():
    a = generate_cases(seed=42, train_count=40, val_count=12)
    assert a == generate_cases(seed=42, train_count=40, val_count=12)
    assert a != generate_cases(seed=43, train_count=40, val_count=12)
    assert set(a) == {"train", "val"} and len(a["train"]) == 40 and len(a["val"]) == 12
    assert {row["family"] for row in a["train"]} == {"rectangle", "L", "T", "U", "C", "stair"}
    assert len({row["canonical_hash"] for rows in a.values() for row in rows}) == 52
    assert not {row["group_id"] for row in a["train"]} & {row["group_id"] for row in a["val"]}
    for split, rows in a.items():
        for row in rows:
            assert row["split"] == split and row["human_reviewed"] is False
            assert Polygon(row["vertices"]).is_valid and canonical_vertices(row["vertices"])[-1] == canonical_vertices(row["vertices"])[0]


def test_env_seed_determinism_and_uniform_transform_observation():
    env = OrthogonalBlockEnv(case())
    a, _ = env.reset(seed=42)
    b, _ = env.reset(seed=42)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    transformed = [(10 + 2 * x, -3 + 2 * y) for x, y in U]
    other, _ = OrthogonalBlockEnv(case(transformed)).reset(seed=42)
    for key in a:
        np.testing.assert_allclose(a[key], other[key], atol=1e-7)
