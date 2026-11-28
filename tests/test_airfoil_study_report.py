"""Denominators, paired populations and report evidence, using explicit fixtures."""
import copy
import json

import pytest

from smartcfd.airfoil_decomposition.study_report import (
    aggregate_study, render_report, write_study_report,
)


def report(method, guard=False, seed=42, outcomes=(True, False), scores=(.6, .8)):
    return {"method": method, "guard_enabled": guard, "seed": seed,
        "selection": "final" if method in ('sac','bc') else 'baseline',
        "total": len(outcomes), "success_count": sum(outcomes),
        "episodes": [{"sample_id": f"shape{i}", "group_id": f"family{i}",
            "success": success, "initializable": i != 1 or success,
            "phase": 'preflight' if i == 1 and not success else 'evaluation',
            "reason": 'completed' if success else 'mapping_failure',
            "steps": 3 if success else 0, "terminal_quality_score": scores[i] if success else -1,
            "elapsed_seconds": .2, "guard_cost": {"second_layer_transitions": 5 if guard else 0}}
            for i,success in enumerate(outcomes)]}


def matrix():
    reports = [report('fixed_c', outcomes=(True,True))]
    for guard in (False, True):
        reports.append(report('greedy', guard, outcomes=(True,True)))
        for method in ('random','sac','bc'):
            for seed in (42,43,44):
                reports.append(report(method, guard, seed))
    return reports


def test_full_denominator_seed_sd_and_independent_shape_count():
    inputs = matrix()
    inputs[3]['episodes'][1].update(success=True, initializable=True, phase='evaluation',
                                   reason='completed', steps=4, terminal_quality_score=.8)
    inputs[3]['success_count'] = 2
    result = aggregate_study(inputs)
    assert result['unique_geometries'] == result['unique_families'] == 2
    assert result['evaluation_rollouts'] == 42  # 21 reports, not 42 independent shapes
    arm = next(r for r in result['arms'] if r['arm']=='random_off')
    assert arm['success_rate']['values'] == [.5,1.,.5]
    assert arm['success_rate']['mean'] == pytest.approx(2/3)
    assert arm['success_rate']['sample_sd'] == pytest.approx(.28867513459481287)
    failed = next(r for r in result['runs'] if r['run_id']=='sac_off_seed42')['cases'][1]
    assert failed['steps'] == 0 and failed['penalized_steps'] == 32
    assert failed['phase'] == 'preflight'


def test_quality_uses_common_success_and_cost_keeps_missing_unknown():
    left = report('sac', True, outcomes=(True,False), scores=(.4,.9))
    right = report('greedy', True, outcomes=(True,True), scores=(.7,.1))
    del left['episodes'][1]['guard_cost']
    result = aggregate_study([left,right], require_complete=False)
    paired = result['paired_comparisons'][0]
    assert paired['kind']=='learning_at_same_guard'
    assert paired['seed_rows'][0]['common_success_count']==1
    assert paired['delta_quality_common_success']['mean']==pytest.approx(-.3)
    assert paired['delta_success_rate']['mean']==-.5
    run = result['runs'][0]
    assert run['costs']['guard_cost.second_layer_transitions']=={'measured_cases':1,'total':5.}
    assert run['mesh_valid_rate_full_manifest'] is None


def test_mesh_failure_does_not_change_partition_success_or_require_equal_cells():
    item = report('sac', outcomes=(True,True))
    diagnostic = [{"sample_id":"shape0", "attempted":True,"valid":True,
                   "quality":{"cells":512,"min_scaled_jacobian":.2}},
                  {"sample_id":"shape1", "attempted":True,"valid":False,"reason":"invalid_generated_mesh"}]
    result = aggregate_study([item], mesh_diagnostics={'sac_off_seed42':diagnostic}, require_complete=False)
    run = result['runs'][0]
    assert run['success_rate']==1
    assert run['mesh_valid_rate_full_manifest']==.5
    assert run['cases'][0]['mesh_cells']==512
    assert run['cases'][1]['mesh_cells'] is None
    assert run['cases'][1]['mesh_failure_reason']=='invalid_generated_mesh'


def test_best_checkpoint_excluded_and_manifest_mismatch_rejected():
    best = report('sac'); best['selection']='best'
    result = aggregate_study([report('sac'), best], require_complete=False)
    assert len(result['runs'])==1 and len(result['excluded_nonprimary_reports'])==1
    mismatch = report('bc')
    mismatch['episodes'][1]['group_id']='another_family'
    with pytest.raises(ValueError, match='same full'):
        aggregate_study([report('sac'),mismatch], require_complete=False)
    with pytest.raises(ValueError, match='nine arms'):
        aggregate_study([report('sac')])
    no_selection = report('sac'); no_selection.pop('selection')
    with pytest.raises(ValueError, match='final checkpoint'):
        aggregate_study([no_selection], require_complete=False)


def test_deterministic_baselines_are_not_pseudoreplicated():
    inputs = matrix()+[report('fixed_c', seed=43, outcomes=(True,True))]
    with pytest.raises(ValueError, match='pseudoreplicated'):
        aggregate_study(inputs)
    inputs = matrix()
    inputs = [r for r in inputs if not (r['method']=='sac' and r['seed']==44 and r['guard_enabled'])]
    with pytest.raises(ValueError, match='Incomplete'):
        aggregate_study(inputs)


def test_report_artifacts_are_parseable_and_refuse_overwrite(tmp_path):
    inputs = matrix()
    original = copy.deepcopy(inputs)
    paths = write_study_report(tmp_path, inputs)
    assert inputs==original
    assert set(paths)=={'summary.json','summary.csv','per_case.csv','failures.csv',
                       'paired_comparisons.json','paired_comparisons.csv','report.md','overview.png','overview.pdf'}
    summary = json.loads((tmp_path/'summary.json').read_text(encoding='utf-8'))
    assert summary['evaluation_rollouts']==42
    assert (tmp_path/'overview.png').read_bytes().startswith(b'\x89PNG')
    assert (tmp_path/'overview.pdf').read_bytes().startswith(b'%PDF')
    text = (tmp_path/'report.md').read_text(encoding='utf-8')
    assert text==render_report(summary)
    assert '不构成独立泛化确认' in text and '已全部成功' in text
    assert '本次为完整九组' in text
    with pytest.raises(FileExistsError):
        write_study_report(tmp_path, inputs)


def test_partial_mesh_records_cannot_look_like_complete_validity_measurement():
    item = report('bc', outcomes=(True,True))
    item['mesh_diagnostics']=[{"sample_id":"shape0","attempted":True,"valid":True,"quality":{"cells":100}}]
    run = aggregate_study([item], require_complete=False)['runs'][0]
    assert not run['mesh_diagnostics_complete']
    assert run['mesh_valid_count'] is None
    assert run['cases'][0]['mesh_valid'] is True
    complete = report('bc', seed=43, outcomes=(True,True))
    complete['mesh_diagnostics']=[{"sample_id":f"shape{i}","attempted":True,"valid":True} for i in (0,1)]
    summary = aggregate_study([item,complete], require_complete=False)
    mesh = summary['arms'][0]['mesh_valid_rate_full_manifest']
    assert mesh['n']==1 and mesh['mean'] is None and mesh['complete'] is False
