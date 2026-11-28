"""All 55 historical resolution disagreements must follow final-mesh acceptance.

The fixed list comes from the frozen train/val v1 baseline's cases.jsonl:
SHA-256 2f9018962bb0f7d94963fc457714c7dccb9f0e8d445b8b255704b9f69d1e029c.
Embedding IDs keeps CI independent of ignored experiments/runs artifacts.
No held-out/test geometry is evaluated and no external solver is launched.
"""
from dataclasses import asdict

import numpy as np
import pytest

from smartcfd.baselines.rule_partition import _matching_action, _program, rule_graph, solve
from smartcfd.cfd.solver import validate_su2
from smartcfd.core.types import ENVIRONMENT_VERSION, Failure, MeshSpec
from smartcfd.geometry.domain import make_domain
from smartcfd.meshing.generator import generate_mesh
from smartcfd.partition.topology import fingerprint
from smartcfd.pipeline.config import environment_config
from smartcfd.pipeline.data import data_version, load_samples
from smartcfd.rl.env import BlockEnv
from smartcfd.rl.reward import transition_reward


# (split, sample_id, variants, alias, old coarse passed, final passed)
_GROUPS = (
    ('train', 'uiuc_0226ec7e3b0a688e', (0, 1), 'goe15', False, True),
    ('train', 'uiuc_077158315c2b0c63', (0, 1), 'goe335', False, True),
    ('train', 'uiuc_239dd4e459d7a7f4', (0, 1), 'goe370', False, True),
    ('train', 'uiuc_24058a3534aa5bc7', (0, 1), 'goe373', False, True),
    ('train', 'uiuc_26fbd354ccd642a2', (0, 1), 'goe397', False, True),
    ('train', 'uiuc_27ebd7a8deee5848', (0,), 'goe464', False, True),
    ('train', 'uiuc_298093624cf8e4cd', (0, 1), 'e62', False, True),
    ('train', 'uiuc_30740060ab94e77d', (0, 1), 'goe275', False, True),
    ('train', 'uiuc_3cf55b2fcb486fd2', (0, 1), 'usa5', False, True),
    ('train', 'uiuc_3e005ae2151fbb32', (0,), 'goe462', False, True),
    ('train', 'uiuc_401b848c416aec2f', (0,), 'goe481', True, False),
    ('train', 'uiuc_4402dae3a03ec3d6', (0, 1), 'goe362', False, True),
    ('train', 'uiuc_44e8f3c5987a9b14', (0, 1), 'goe804', False, True),
    ('train', 'uiuc_45dcb54df358cb8f', (0, 1), 's1091', False, True),
    ('train', 'uiuc_5576d76ab24f3e48', (0, 1), 'goe396', False, True),
    ('train', 'uiuc_5e14d875406afa75', (0, 1), 'goe376', False, True),
    ('train', 'uiuc_6a201499c42cfae8', (0, 1), '30p-30n-main', True, False),
    ('train', 'uiuc_6f445f45105179d6', (0, 1), 'e58', False, True),
    ('train', 'uiuc_8706ddc4f4e1febf', (0, 1), 'goe590', False, True),
    ('train', 'uiuc_b8e0f5309c1d8f87', (0, 1), 'goe361', False, True),
    ('train', 'uiuc_c63c0388ec25d34d', (0, 1), 'goe419', False, True),
    ('train', 'uiuc_cb61a963d1368b6e', (0, 1), 'goe483', False, True),
    ('train', 'uiuc_d783979065b667a6', (0, 1), 'goe364', False, True),
    ('train', 'uiuc_d81b6239af5cd871', (0, 1), 'e59', False, True),
    ('train', 'uiuc_dcf48f77c8785083', (0, 1), 'raf19', False, True),
    ('train', 'uiuc_de2a1abc0a70ea6b', (0, 1), 'goe379', False, True),
    ('train', 'uiuc_e562925dd443c6ab', (0, 1), 'e49', False, True),
    ('train', 'uiuc_fd84c5021086e8fd', (0, 1), 'goe391', False, True),
    ('val', 'uiuc_2dc340825fffe0cb', (0, 1), 'fx74cl5140', False, True),
)
CASES = tuple((split, sid, variant, coarse_ok, final_ok)
              for split, sid, variants, alias, coarse_ok, final_ok in _GROUPS
              for variant in variants)
CASE_IDS = tuple(f'{split}-{alias}-v{variant}'
                 for split, sid, variants, alias, coarse_ok, final_ok in _GROUPS
                 for variant in variants)


@pytest.fixture(scope='module')
def selected_domains():
    frozen = data_version()
    requested = {(split, sid) for split, sid, *_ in CASES}
    domains = {}
    for split in ('train', 'val'):
        for sample in load_samples(split):
            key = (split, sample['sample_id'])
            if key in requested:
                domains[key] = make_domain(sample)
    assert set(domains) == requested
    yield domains
    assert data_version() == frozen


def test_regression_inventory_covers_all_disagreements():
    assert len(CASES) == len(set((split, sid, variant) for split, sid, variant, _, _ in CASES)) == 55
    assert sum(not coarse and final for _, _, _, coarse, final in CASES) == 52
    assert sum(coarse and not final for _, _, _, coarse, final in CASES) == 3
    assert {split for split, *_ in CASES} == {'train', 'val'}


def _run_live_rule(env, variant=0):
    env.reset(seed=42 + variant, options={'sample_id': env.domains[0].sample_id})
    for kind, start, end in _program(variant):
        action = _matching_action(env, kind, start, end)
        assert action is not None and not env.terminated
        _, _, terminated, _, info = env.step(action)
    assert terminated
    return info


def test_terminal_acceptance_honors_the_full_mesh_cell_budget(selected_domains):
    domain = selected_domains[('train', 'uiuc_0226ec7e3b0a688e')]
    config = environment_config({'mesh': {'max_cells': 1}})
    env = BlockEnv([domain], config)
    try:
        info = _run_live_rule(env)
        assert info['success'] is False and env.mesh is None
        assert env.last_failure.reason == 'mesh_generation_failed'
        assert 'budget' in env.last_failure.detail
        assert info['terminal_mesh_spec']['max_cells'] == 1
    finally:
        env.close()


def test_terminal_acceptance_also_requires_boundary_validation(selected_domains, monkeypatch):
    from smartcfd.cfd import solver
    domain = selected_domains[('train', 'uiuc_0226ec7e3b0a688e')]
    calls = []
    def reject_boundary(mesh):
        assert mesh.quality['valid']
        calls.append(mesh.quality['cells'])
        raise ValueError('fixture exterior marker mismatch')
    monkeypatch.setattr(solver, 'validate_su2', reject_boundary)
    env = BlockEnv([domain], environment_config({}))
    try:
        info = _run_live_rule(env)
        assert calls and info['success'] is False and env.mesh is None
        assert env.last_failure.reason == 'invalid_terminal_boundary'
        assert 'marker mismatch' in env.last_failure.detail
    finally:
        env.close()


@pytest.mark.parametrize('split,sample_id,variant,coarse_ok,final_ok', CASES, ids=CASE_IDS)
def test_all_resolution_disagreements_use_actual_final_mesh(
        selected_domains, split, sample_id, variant, coarse_ok, final_ok):
    domain = selected_domains[(split, sample_id)]
    graph = rule_graph(domain, variant=variant)
    assert not isinstance(graph, Failure)
    final_spec = MeshSpec()
    config = environment_config({'mesh': asdict(final_spec)})
    assert config.environment_version == ENVIRONMENT_VERSION == 'final_mesh_v2'
    assert config.mesh_spec == final_spec

    # Independent diagnostics reproduce the actual historical disagreement.
    coarse = generate_mesh(domain, graph, MeshSpec(intervals=8, protected_intervals=16))
    final = generate_mesh(domain, graph, final_spec)
    assert (not isinstance(coarse, Failure)) is coarse_ok
    assert (not isinstance(final, Failure)) is final_ok
    if final_ok:
        exported = validate_su2(final)
        assert exported['cells'] == final.quality['cells']
    else:
        assert final.reason == 'invalid_generated_mesh'

    # The same real rule program edits this live environment, including the
    # failed final action in the three formerly false-positive cases.
    env = BlockEnv([domain], config)
    seed = 42 + variant
    try:
        _, reset_info = env.reset(seed=seed, options={'sample_id': sample_id})
        assert reset_info['environment_version'] == ENVIRONMENT_VERSION
        assert reset_info['terminal_mesh_spec'] == asdict(final_spec)
        actions = []
        for kind, start, end in _program(variant):
            assert not env.terminated
            action = _matching_action(env, kind, start, end)
            assert action is not None and env.action_mask[action]
            actions.append(action)
            before = env.graph.clone()
            _, reward, terminated, truncated, info = env.step(action)
        assert len(actions) == (3 if variant == 0 else 12)
        assert terminated and not truncated and info['success'] is final_ok
        assert env.success is final_ok
        assert fingerprint(env.graph) == fingerprint(graph)
        expected_reward = transition_reward(before, graph, config, terminal=True,
            success=final_ok, invalid=False, quality=final.quality['q10_scaled_jacobian'] if final_ok else 0.)
        assert reward == pytest.approx(expected_reward, abs=1e-12)
        if final_ok:
            assert env.last_failure is None and env.mesh is not None
            assert env.mesh.quality == final.quality
            assert all(np.array_equal(env.mesh.blocks[fid], nodes) for fid, nodes in final.blocks.items())
            assert validate_su2(env.mesh) == exported
        else:
            assert env.mesh is None and env.last_failure.reason == final.reason

        # The public teacher must give the same decision and leave its reset
        # input untouched; no diagnostic-only graph may be promoted to a demo.
        env.reset(seed=seed, options={'sample_id': sample_id})
        teacher = solve(env, seed=seed)
        if final_ok:
            assert teacher == actions
        else:
            assert isinstance(teacher, Failure)
            assert teacher.reason == 'teacher_no_valid_rule_partition'
        assert env.steps == 0 and not env.success and not env.terminated
    finally:
        env.close()
