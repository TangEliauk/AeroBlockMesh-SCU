from dataclasses import asdict
import json
from smartcfd.core.types import MeshSpec
from smartcfd.airfoil_decomposition import AirfoilDecompositionConfig
from smartcfd.airfoil_decomposition.data import select_cases
from smartcfd.airfoil_decomposition.experiment import evaluate, mesh_diagnostics, write_json
from smartcfd.airfoil_decomposition.visualize import plot_partition


def test_real_airfoil_fixed_rule_export_plot_and_diagnostic_adapter(tmp_path):
    sample = select_cases(train_count=1, val_count=1)['train'][0]
    config = AirfoilDecompositionConfig()
    report = evaluate([sample], config, method='fixed_c')
    assert report['success_count'] == report['total'] == 1
    row = report['episodes'][0]
    assert row['steps'] == 3
    part = row['partition']
    assert part['valid'] and part['quality']['block_count'] == 8
    assert part['graph']['faces'] and part['root']['wall']
    assert part['interfaces'] and part['wake_pairs']
    assert row['terminal_quality_score'] == part['quality']['q'] - config.outer_block_cost*4
    assert all(abs(r['total']-r['base']-r['shaping']) < 1e-12 for r in row['step_rewards'])
    image = tmp_path/'physical_partition.png'
    plot_partition(sample, row, image)
    assert image.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    diagnostics = mesh_diagnostics({'fixed_c': report}, [sample], config, asdict(MeshSpec()))
    diagnostic = diagnostics['fixed_c'][0]
    assert diagnostic['attempted'] and 'valid' in diagnostic
    write_json(tmp_path/'diagnostic.json', diagnostics)
    assert json.loads((tmp_path/'diagnostic.json').read_text(encoding='utf-8'))['fixed_c'][0]['attempted']
    # Diagnostic failure never changes the geometry success or selection metric.
    assert report['success_count'] == 1


def test_preflight_failures_remain_in_validation_denominator():
    cases = select_cases(train_count=1, val_count=2)['val']
    failed = [{'sample_id': r['sample_id'], 'valid': False, 'reason': 'deliberate_mapping_failure'} for r in cases]
    report = evaluate(cases, AirfoilDecompositionConfig(), method='random', preflight_rows=failed)
    assert report['total'] == 2 and report['success_count'] == 0
    assert report['initializable_count'] == 0
    assert report['mean_terminal_quality_score'] == -1
    assert all(r['phase'] == 'preflight' for r in report['episodes'])
