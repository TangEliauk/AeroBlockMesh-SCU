"""Generation/export validity contracts and local failed-cell diagnostics."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile
import numpy as np
import pytest
from smartcfd.baselines.rule_partition import rule_graph
from smartcfd.core.types import Failure, MeshSpec, MultiBlockMesh
from smartcfd.geometry.domain import make_domain, sharp_eligibility
from smartcfd.pipeline.data import load_domains, load_samples, repository_root, sharp_eligible
from smartcfd.meshing import generator
from smartcfd.evaluation.mesh_quality import evaluate_mesh
from smartcfd.cfd.solver import export_su2, validate_su2


@pytest.fixture
def local_workspace():
    root = repository_root().resolve()
    base = root/'.runtime/mesh_audit_test_fixtures'
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='case_',dir=base) as directory:
        path = Path(directory).resolve()
        assert path.is_relative_to(root) and path != root
        yield path
        assert path.is_relative_to(root) and path != root


@pytest.fixture(scope='module')
def training_domain():
    return load_domains('train',limit=1)[0]


@pytest.fixture(scope='module')
def failed_domains():
    wanted = {'uiuc_0226ec7e3b0a688e','uiuc_041a31235f3f870f'}
    samples = [sample for sample in load_samples('train') if sample['sample_id'] in wanted]
    assert {sample['sample_id'] for sample in samples} == wanted
    return {sample['sample_id']:make_domain(sample) for sample in samples}


def fixture_mesh(nodes):
    return MultiBlockMesh({0:nodes},quality=evaluate_mesh({0:nodes}),metadata={})


def square():
    return np.stack(np.meshgrid(np.linspace(0,1,3),np.linspace(0,1,3),indexing='ij'),axis=-1)


@pytest.mark.parametrize('sample_id,expected_min',[
    ('uiuc_0226ec7e3b0a688e',-.13679989886112204),
    ('uiuc_041a31235f3f870f',-.10812850151644879),
])
@pytest.mark.parametrize('variant',[0,1])
def test_original_four_failures_are_located_without_becoming_successes(failed_domains,sample_id,expected_min,variant):
    domain = failed_domains[sample_id]
    graph = rule_graph(domain,variant)
    result = generator.generate_mesh(domain,graph,MeshSpec(intervals=8,protected_intervals=16))
    assert isinstance(result,Failure) and result.reason == 'invalid_generated_mesh'
    diagnostics = result.diagnostics
    assert diagnostics['quality']['valid'] is False
    assert diagnostics['invalid_cell_count'] == 16
    assert diagnostics['quality']['min_scaled_jacobian'] == pytest.approx(expected_min,abs=1e-12)
    assert diagnostics['mapping']['mapping_version'] == 'c_normal_envelope_wakegraded_v2'
    assert len(diagnostics['invalid_cells_preview']) == 16
    for cell in diagnostics['invalid_cells_preview']:
        assert cell['protected'] == graph.faces[cell['block_id']].protected
        assert cell['corner_jacobian'] <= 1e-10
        assert np.asarray(cell['uv_corners']).shape == (4,2)
        assert np.asarray(cell['xy_corners']).shape == (4,2)
        assert 0 <= cell['corner'] < 4
    json.dumps(diagnostics,allow_nan=False)


def test_coarse_pass_does_not_imply_fine_pass(training_domain,monkeypatch):
    graph = rule_graph(training_domain,0)
    def produce(domain,graph,spec):
        nodes = square()
        return fixture_mesh(nodes if spec.intervals==8 else nodes[::-1])
    monkeypatch.setattr(generator,'_generate',produce)
    coarse = generator.generate_mesh(training_domain,graph,MeshSpec(intervals=8,protected_intervals=16))
    fine = generator.generate_mesh(training_domain,graph,MeshSpec(intervals=16,protected_intervals=32))
    assert isinstance(coarse,MultiBlockMesh) and coarse.quality['valid']
    assert isinstance(fine,Failure) and fine.reason == 'invalid_generated_mesh'
    assert fine.diagnostics['invalid_cell_count'] == 4


def test_failed_artifacts_are_explicit_and_cannot_export(failed_domains,local_workspace):
    domain = failed_domains['uiuc_0226ec7e3b0a688e']
    graph = rule_graph(domain,0)
    directory = local_workspace/'invalid'
    result = generator.generate_mesh(domain,graph,MeshSpec(intervals=8,protected_intervals=16),diagnostics_dir=directory)
    assert isinstance(result,Failure)
    assert {p.name for p in directory.iterdir()} >= {'invalid_mesh.npz','invalid_cells.json','diagnostics.json','failed_graph.json','worst_cell.png'}
    assert not (directory/'mesh.npz').exists() and not (directory/'mesh.su2').exists()
    document = json.loads((directory/'diagnostics.json').read_text(encoding='utf-8'))
    assert document['valid'] is False and document['diagnostic_only'] is True
    assert len(json.loads((directory/'invalid_cells.json').read_text(encoding='utf-8'))) == 16
    with np.load(directory/'invalid_mesh.npz',allow_pickle=False) as archive:
        blocks = {int(name.removeprefix('block_')):archive[name].copy() for name in archive.files}
    invalid = MultiBlockMesh(blocks,quality=result.diagnostics['quality'])
    with pytest.raises(ValueError,match='invalid mesh'):
        generator.save_mesh(invalid,directory/'successful')
    with pytest.raises(ValueError,match='invalid mesh'):
        export_su2(invalid,directory/'not_exported.su2')
    assert not (directory/'successful').exists() and not (directory/'not_exported.su2').exists()


def test_nonfinite_failed_diagnostics_use_null_and_flags(training_domain,local_workspace,monkeypatch):
    graph = rule_graph(training_domain,0)
    nodes = square(); nodes[1,1,0] = np.nan
    monkeypatch.setattr(generator,'_generate',lambda *args:fixture_mesh(nodes))
    result = generator.generate_mesh(training_domain,graph,diagnostics_dir=local_workspace/'nonfinite')
    assert isinstance(result,Failure)
    quality = result.diagnostics['quality']
    assert quality['min_scaled_jacobian'] is None and quality['q10_scaled_jacobian'] is None
    assert quality['has_nonfinite_coordinates'] and quality['has_nonfinite_jacobians']
    assert quality['nonfinite_coordinate_count'] == 1
    assert all(c['has_nonfinite_coordinates'] for c in result.diagnostics['invalid_cells_preview'])
    assert any(point[0] is None for c in result.diagnostics['invalid_cells_preview'] for point in c['xy_corners'])
    json.dumps(result.diagnostics,allow_nan=False)
    document = json.loads((local_workspace/'nonfinite/diagnostics.json').read_text(encoding='utf-8'))
    assert document['diagnostics']['quality']['min_scaled_jacobian'] is None


def test_bounded_preview_and_full_invalid_cell_file(training_domain,local_workspace,monkeypatch):
    graph = rule_graph(training_domain,0)
    nodes = np.stack(np.meshgrid(np.linspace(0,1,8),np.linspace(0,1,8),indexing='ij'),axis=-1)[::-1]
    monkeypatch.setattr(generator,'_generate',lambda *args:fixture_mesh(nodes))
    result = generator.generate_mesh(training_domain,graph)
    assert result.diagnostics['invalid_cell_count'] == 49
    assert len(result.diagnostics['invalid_cells_preview']) == 20 and result.diagnostics['preview_truncated']
    generator.generate_mesh(training_domain,graph,diagnostics_dir=local_workspace/'full')
    assert len(json.loads((local_workspace/'full/invalid_cells.json').read_text(encoding='utf-8'))) == 49


def test_success_skips_invalid_diagnostics_and_disk_output(training_domain,local_workspace,monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('Success should not compute invalid-cell diagnostics')
    monkeypatch.setattr(generator,'invalid_cell_diagnostics',forbidden)
    directory = local_workspace/'should_not_exist'
    result = generator.generate_mesh(training_domain,rule_graph(training_domain,0),diagnostics_dir=directory)
    assert isinstance(result,MultiBlockMesh) and result.quality['valid']
    assert not directory.exists()


def test_pure_memory_export_validation_matches_export_and_detects_mutation(training_domain,local_workspace):
    mesh = generator.generate_mesh(training_domain,rule_graph(training_domain,0))
    before = set(local_workspace.iterdir())
    stats = validate_su2(mesh)
    assert set(local_workspace.iterdir()) == before
    assert stats == export_su2(mesh,local_workspace/'mesh.su2')
    assert stats['cells'] == mesh.quality['cells'] and set(stats['markers']) == {'wall','farfield'}
    bad = copy.deepcopy(mesh)
    interface = next(item for item in bad.interfaces if item['kind']=='block')
    generator.side_nodes(bad.blocks[interface['block_b']],interface['side_b'])[1,0] += 1e-12
    with pytest.raises(ValueError,match='interface'):
        validate_su2(bad)
    bad = copy.deepcopy(mesh); bad.interfaces[0]['block_b'] = -999
    with pytest.raises(ValueError,match='metadata'):
        validate_su2(bad)
    bad = copy.deepcopy(mesh); bad.boundaries.pop()
    with pytest.raises(ValueError,match='boundary markers'):
        validate_su2(bad)
    bad = copy.deepcopy(mesh)
    fid = next(iter(bad.blocks)); bad.blocks[fid] = bad.blocks[fid][::-1]
    assert bad.quality['valid']  # Stale success flags never authorize export.
    with pytest.raises(ValueError,match='geometric screen'):
        validate_su2(bad)


def test_unified_sharp_eligibility_preserves_frozen_counts_and_rejects_bad_inputs():
    assert {split:len(load_samples(split)) for split in ('train','val','test')} == {'train':780,'val':103,'test':90}
    sample = load_samples('train',limit=1)[0]
    assert sharp_eligibility(sample) == (True,'eligible') and sharp_eligible(sample)
    damaged = []
    nonnormalized = copy.deepcopy(sample); nonnormalized['polygon'] = (2*np.asarray(sample['polygon'])).tolist(); damaged.append(nonnormalized)
    nonfinite = copy.deepcopy(sample); nonfinite['polygon'][1][0] = float('inf'); damaged.append(nonfinite)
    duplicate = copy.deepcopy(sample); duplicate['polygon'][1] = duplicate['polygon'][0]; damaged.append(duplicate)
    source = copy.deepcopy(sample); source['source_files'] = None; damaged.append(source)
    for row in [None,[],{}, {'polygon':[[0,0],[1]]},*damaged]:
        assert sharp_eligibility(row)[0] is False and sharp_eligible(row) is False
