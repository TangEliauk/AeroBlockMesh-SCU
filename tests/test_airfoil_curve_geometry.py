"""Physical C-curve tests: synthetic profile and frozen TRAIN membership only."""
from dataclasses import replace
import json
import numpy as np
import pytest
from shapely.geometry import Polygon

from smartcfd.airfoil_decomposition.config import AirfoilDecompositionConfig
from smartcfd.airfoil_decomposition.geometry import CurveGeometry
from smartcfd.core.types import FaceBlock
from smartcfd.partition.topology import initial_graph


def sample():
    x = (1-np.cos(np.linspace(0, np.pi, 81)))/2
    y = .18*np.sqrt(x)*(1-x)
    polygon = np.vstack((np.column_stack((x, -y)),
                         np.column_stack((x[-2:0:-1], y[-2:0:-1]))))
    return {'sample_id': 'synthetic_non_naca', 'polygon': polygon.tolist(), 'explicitly_closed': True}


def complete_graph(geometry):
    graph = initial_graph(geometry.domain)
    for i in range(4):
        graph.faces[4+i] = FaceBlock(4+i, (5+i, 6+i, 11+i, 10+i))
    graph.next_face_id = 8
    return graph


def test_reference_is_outer_boundary_minus_original_wall_hole():
    geometry = CurveGeometry(sample())
    graph = initial_graph(geometry.domain)
    report = geometry.partition(graph)
    assert report['valid'], report
    root = Polygon(report['root']['outer'], [report['root']['wall']])
    assert root.is_valid and len(root.interiors) == 1
    wall = Polygon(sample()['polygon'])
    assert root.intersection(wall).area < 1e-12
    radius = geometry.config.farfield
    exact_area = (2+np.pi/2)*radius**2-wall.area
    assert abs(root.area-exact_area) < .02  # explicit chord approximation
    assert report['quality']['coverage_error_area'] <= 1e-8
    assert report['quality']['outer_blocks'] == 1 and report['quality']['q'] == 0


def test_wall_curve_keeps_every_original_branch_knot_and_exact_reverse():
    geometry = CurveGeometry(sample())
    lower = geometry.curve((.25, 0), (.5, 0))
    original = geometry.mapping.lower[1]
    assert all(np.min(np.linalg.norm(lower-point, axis=1)) < 1e-12 for point in original)
    assert np.array_equal(lower, geometry.curve((.5, 0), (.25, 0))[::-1])
    assert not lower.flags.writeable
    assert len(geometry.curve((.25, 1), (.75, 1), True)) >= len(geometry.curve((.25, 1), (.75, 1)))


def test_arc_midpoint_is_physical_length_not_uv_midpoint():
    geometry = CurveGeometry(sample())
    a, b = (0., 0.), (.25, 0.)
    middle = geometry.arc_point(a, b, .5)
    p, q, m = geometry.mapping.evaluate(np.array([a, b, middle]))
    assert abs(np.linalg.norm(m-p)/np.linalg.norm(q-p)-.5) < 1e-5
    assert abs(middle[0]-.125) > .05
    with pytest.raises(ValueError):
        geometry.arc_point(a, b, 0)


def test_complete_curved_quads_quality_and_refinement_are_consistent():
    geometry = CurveGeometry(sample())
    graph = complete_graph(geometry)
    coarse, fine = geometry.validate(graph), geometry.validate(graph, refine=True)
    assert coarse['valid'] and fine['valid'], (coarse, fine)
    q = coarse['quality']
    assert q['outer_blocks'] == 4 and 0 < q['q'] <= 1
    assert q['q'] == pytest.approx(.7*q['angle_quality']+.3*q['opposite_length_quality'])
    assert abs(fine['quality']['q']-q['q']) < .01
    assert q['coverage_error_area'] <= 1e-8 and q['overlap_area'] <= 1e-8


def test_partition_shares_curve_samples_and_pairs_wake_without_fake_hole_slit():
    geometry = CurveGeometry(sample())
    report = geometry.partition(complete_graph(geometry))
    assert report['valid']
    json.dumps(report, allow_nan=False)
    curves = {curve['id']: curve for curve in report['curves']}
    blocks = {block['id']: block for block in report['blocks']}
    for interface in report['interfaces']:
        a = blocks[interface['block_a']]['curve_refs'][interface['side_a']]
        b = blocks[interface['block_b']]['curve_refs'][interface['side_b']]
        assert a['curve_id'] == b['curve_id'] == interface['curve_id']
        assert a['forward'] != b['forward']
    assert len(report['wake_pairs']) == 1
    pair = report['wake_pairs'][0]
    assert curves[pair['curve_a']]['tag'] == curves[pair['curve_b']]['tag'] == 'wake'
    a, b = (np.asarray(curves[pair[key]]['points']) for key in ('curve_a', 'curve_b'))
    assert np.allclose(a[[0, -1]], b[[-1, 0]], rtol=0, atol=1e-11)
    assert all(len(block['uv_corners']) == 4 for block in blocks.values())
    assert any(len(block['boundary']) > 4 for block in blocks.values())


@pytest.mark.parametrize('broken', ['protected', 'coverage', 'out_of_bounds'])
def test_bad_graphs_return_serializable_failure_and_refinement_agrees(broken):
    geometry = CurveGeometry(sample())
    graph = complete_graph(geometry)
    if broken == 'protected':
        graph.vertices[6] = (.251, .5)
    elif broken == 'coverage':
        del graph.faces[4]
    else:
        graph.vertices[14] = (1.01, 1.)
    a, b = geometry.validate(graph), geometry.validate(graph, refine=True)
    assert not a['valid'] and not b['valid']
    json.dumps(a, allow_nan=False)


def test_adaptive_depth_failure_is_reported_not_silently_coarsened():
    config = replace(AirfoilDecompositionConfig(), curve_tolerance=1e-12, curve_max_depth=1)
    geometry = CurveGeometry(sample(), config)
    report = geometry.validate(initial_graph(geometry.domain))
    assert not report['valid'] and report['reason'] == 'geometry_error'
    assert 'depth' in report['detail']


def test_real_uiuc_train_geometry_and_bounded_caches():
    from smartcfd.pipeline.data import load_samples
    training_sample = load_samples('train', limit=1)[0]
    geometry = CurveGeometry(training_sample)
    graph = complete_graph(geometry)
    assert geometry.validate(graph)['valid']
    assert geometry.validate(graph, refine=True)['valid']
    geometry._curve_limit = 4
    for i in range(12):
        geometry.curve((.1+i*.01, .5), (.2+i*.01, 1.))
    assert len(geometry._curves) <= 4


def test_folded_mapping_is_a_geometry_failure_not_a_success(monkeypatch):
    geometry = CurveGeometry(sample())
    original = geometry.mapping.evaluate
    def folded(uv):
        uv = np.asarray(uv)
        result = original(uv).copy()
        result[..., 1] *= np.cos(4*np.pi*uv[..., 1])
        return result
    monkeypatch.setattr(geometry.mapping, 'evaluate', folded)
    report = geometry.validate(complete_graph(geometry))
    assert not report['valid']


def test_evaluation_only_roundoff_reduction_preserves_curves_quality_and_area():
    geometry = CurveGeometry(sample())
    graph = complete_graph(geometry)
    curve = geometry.curve((.25, .5), (.25, 1.)).copy()
    raw = geometry._validate(graph, exact=True)
    fast = geometry.validate(graph)
    assert fast['valid'] == raw['valid']
    for metric in ('q', 'angle_quality', 'opposite_length_quality'):
        assert fast['quality'][metric] == raw['quality'][metric]
    bounds, removed = [], 0
    for face in graph.faces.values():
        polygon = geometry._face(graph, face, False)[0]
        reduced, bound = geometry._evaluation_polygon(polygon)
        assert reduced.is_valid == polygon.is_valid
        assert polygon.symmetric_difference(reduced).area <= bound + 1e-14
        assert bound < geometry.config.area_tolerance/128
        bounds.append(bound)
        removed += len(polygon.exterior.coords)-len(reduced.exterior.coords)
    assert removed > 100  # actual collinear curve samples, not coarse geometry
    assert abs(fast['quality']['coverage_error_area']-raw['quality']['coverage_error_area']) < 1e-10
    assert abs(fast['quality']['overlap_area']-raw['quality']['overlap_area']) < 1e-10
    assert np.array_equal(curve, geometry.curve((.25, .5), (.25, 1.)))
    root = geometry._reference(graph, False)[0]
    reduced_root, _ = geometry._evaluation_polygon(root, keep_wall=True)
    assert np.array_equal(np.asarray(root.interiors[0].coords), np.asarray(reduced_root.interiors[0].coords))


def test_roundoff_reducer_never_repairs_invalid_rings_or_removes_actual_bends():
    geometry = CurveGeometry(sample())
    invalid = Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])
    polygon, bound = geometry._evaluation_polygon(invalid)
    assert polygon is invalid and not polygon.is_valid and bound == 0
    bent = Polygon([(0, 0), (.5, 1e-5), (1, 0), (1, 1), (0, 1)])
    reduced, _ = geometry._evaluation_polygon(bent)
    assert (.5, 1e-5) in reduced.exterior.coords


def test_ambiguous_area_threshold_falls_back_to_original_rings(monkeypatch):
    geometry = CurveGeometry(sample())
    graph = complete_graph(geometry)
    expected = geometry._validate(graph, exact=True)
    # Deliberately overestimate the optimizer's area uncertainty so the current
    # zero wall overlap cannot be classified without using the original rings.
    monkeypatch.setattr(geometry, '_evaluation_polygon',
        lambda original, **kwargs: (original, geometry.config.area_tolerance*2))
    original_validate = geometry._validate
    modes = []
    def recorded(graph, refine=False, *, exact=False):
        modes.append(exact)
        return original_validate(graph, refine, exact=exact)
    monkeypatch.setattr(geometry, '_validate', recorded)
    assert geometry.validate(graph) == expected
    assert modes == [False, True]
