from dataclasses import asdict
import pytest
from smartcfd.airfoil_decomposition.config import AirfoilDecompositionConfig
from smartcfd.airfoil_decomposition.data import select_cases, digest
from smartcfd.airfoil_decomposition.experiment import normalize_config


def test_frozen_manifest_reproducible_disjoint_and_not_expert():
    a = select_cases(train_count=8, val_count=4)
    b = select_cases(train_count=8, val_count=4)
    assert digest(a) == digest(b)
    assert len(a['train']) == 8 and len(a['val']) == 4
    assert {x['group_id'] for x in a['train']}.isdisjoint(x['group_id'] for x in a['val'])
    assert all(not x['expert_demonstration'] for s in ('train', 'val') for x in a[s])
    assert set(a) == {'train', 'val', 'manifest'}


def test_configuration_contract_rejects_invalid_capacity_and_unknowns():
    cfg = normalize_config({'version': 1})
    assert cfg['environment'] == asdict(AirfoilDecompositionConfig())
    assert cfg['training']['steps'] == 2000
    with pytest.raises(ValueError):
        AirfoilDecompositionConfig(max_candidates=1025)
    with pytest.raises(ValueError):
        normalize_config({'environment': {'gamma': .9}})
    with pytest.raises(ValueError):
        normalize_config({'training': {'stepps': 20}})
