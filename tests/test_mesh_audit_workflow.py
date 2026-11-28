"""Paired audit slots preserve failures and never promote diagnostic graphs."""
from argparse import Namespace
import json
from pathlib import Path
import tempfile
import pytest

from smartcfd.core.types import Failure
from smartcfd.pipeline import mesh_audit, workflow
from smartcfd.pipeline.config import read_config
from smartcfd.pipeline.data import repository_root, data_version


@pytest.fixture
def local_output():
    root = repository_root()
    base = root / '.runtime/mesh_audit_tests'
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base) as name:
        yield root, Path(name)
        assert Path(name).resolve().is_relative_to(root.resolve())


def arguments(path, **overrides):
    values = dict(output=str(path), splits=['train'], variants=[0, 1], limit=1, seed=42)
    return Namespace(**{**values, **overrides})


def test_real_paired_baseline_records_all_slots_without_touching_data(local_output):
    _, output = local_output
    frozen = data_version()
    run, report = mesh_audit.audit_meshes(read_config(), arguments(output/'audit', splits=['train', 'val']))
    assert report['expected_slots'] == report['recorded_slots'] == report['unique_slots'] == 8
    assert report['slot_statuses'] == {'passed': 8}
    assert report['replayable_demonstrations'] == 4
    episodes = [json.loads(line) for line in (run/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    assert {r['split'] for r in episodes} == {'train', 'val'}
    assert all(r['source'] == 'rule_synthetic' and not r['human_reviewed'] for r in episodes)
    assert all(r['actions'] and r['env_config'] and r['mesh_spec'] and r['data'] == frozen for r in episodes)
    assert data_version() == frozen


def test_teacher_and_coarse_failure_do_not_skip_independent_fine_check(local_output, monkeypatch):
    from smartcfd.meshing import generator
    _, output = local_output
    original = generator.generate_mesh
    def generate(domain, graph, spec, **kwargs):
        if spec.intervals == 8:
            return Failure('coarse_fixture_failure', diagnostics={'minimum': float('nan')})
        return original(domain, graph, spec, **kwargs)
    monkeypatch.setattr(generator, 'generate_mesh', generate)
    run, report = mesh_audit.audit_meshes(read_config(), arguments(output/'audit'))
    assert report['slot_statuses'] == {'failed': 2, 'passed': 2}
    assert len(report['coarse_failed_fine_passed']) == 2
    assert report['pipeline_passed_cases'] == report['replayable_demonstrations'] == 2
    rows = [json.loads(line) for line in (run/'slots.jsonl').read_text().splitlines()]
    assert all(r['teacher_status'] == 'passed' for r in rows)
    assert rows[0]['failure']['diagnostics']['minimum'] is None
    assert len((run/'episodes.jsonl').read_text().splitlines()) == 2


def test_graph_failure_still_accounts_for_both_resolution_slots(local_output, monkeypatch):
    _, output = local_output
    monkeypatch.setattr(mesh_audit, 'rule_graph', lambda *a, **k: Failure('fixture_graph_failure'))
    monkeypatch.setattr(mesh_audit, 'solve', lambda *a, **k: Failure('fixture_teacher_failure'))
    run, report = mesh_audit.audit_meshes(read_config(), arguments(output/'audit'))
    assert report['expected_slots'] == report['recorded_slots'] == report['unique_slots'] == 4
    assert report['slot_statuses'] == {'not_run': 4}
    rows = [json.loads(line) for line in (run/'slots.jsonl').read_text().splitlines()]
    assert all(r['mesh_valid'] is None and r['stage'] == 'graph' for r in rows)
    assert report['success_rate_all_slots'] == 0


def test_one_export_failure_is_preserved_and_other_slots_continue(local_output, monkeypatch):
    from smartcfd.cfd import solver
    from smartcfd.meshing.generator import generate_mesh
    from smartcfd.rl.env import BlockEnv
    _, output = local_output
    original = solver.validate_su2
    def terminal_mesh(self, graph):
        mesh = generate_mesh(self.domain, graph, self.mesh_spec)
        if not isinstance(mesh, Failure):
            original(mesh)
        return mesh
    monkeypatch.setattr(BlockEnv, '_mesh_for', terminal_mesh)
    calls = []
    def validate(mesh):
        calls.append(True)
        if len(calls) == 1:
            raise ValueError('fixture exterior mismatch')
        return original(mesh)
    monkeypatch.setattr(solver, 'validate_su2', validate)
    run, report = mesh_audit.audit_meshes(read_config(), arguments(output/'audit'))
    assert len(calls) == 4 and report['slot_statuses'] == {'failed': 1, 'passed': 3}
    rows = [json.loads(line) for line in (run/'slots.jsonl').read_text().splitlines()]
    assert rows[0]['stage'] == 'export' and rows[0]['mesh_valid'] is True
    assert report['replayable_demonstrations'] == 2
    assert report['teacher_final_disagreement_count'] == 0


def test_evaluate_defends_against_invalid_fine_mesh_contract(local_output, monkeypatch):
    from smartcfd.meshing import generator
    from smartcfd.rl.env import BlockEnv
    _, output = local_output
    original = generator.generate_mesh
    # Keep the actual terminal producer valid; inject the broken contract only
    # into the independent workflow output check being tested here.
    monkeypatch.setattr(BlockEnv, '_mesh_for', lambda self, graph: original(self.domain, graph, self.mesh_spec))
    def generate(domain, graph, spec, **kwargs):
        mesh = original(domain, graph, spec, **kwargs)
        if spec.intervals == 16 and not isinstance(mesh, Failure):
            mesh.quality['valid'] = False
        return mesh
    monkeypatch.setattr(generator, 'generate_mesh', generate)
    args = Namespace(output=str(output/'evaluate'), split='train', method='rule', checkpoint=None, seed=42, limit=1)
    _, report = workflow.evaluate(read_config(), args)
    assert report['success_rate'] == 0
    assert report['episodes'][0]['reason'] == 'invalid_generated_mesh'
    assert report['valid_mesh_quality_mean'] is None


def test_test_split_rejected_before_run_allocation(monkeypatch):
    monkeypatch.setattr(mesh_audit, 'new_run', lambda *a, **k: pytest.fail('must not allocate output'))
    with pytest.raises(ValueError, match='held-out test'):
        mesh_audit.audit_meshes({}, arguments(None, splits=['test']))
