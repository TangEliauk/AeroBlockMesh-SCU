"""Frozen-input and safe orchestration boundaries, independent of training."""
import json
from pathlib import Path

import pytest

from smartcfd.airfoil_decomposition import study_driver as driver
from smartcfd.airfoil_decomposition.data import digest
from smartcfd.airfoil_decomposition.experiment import file_sha


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def fixture_study(tmp_path, monkeypatch):
    root = tmp_path/'copy'
    output = root/'experiments/runs/a'
    output.mkdir(parents=True)
    monkeypatch.setattr(driver, 'ROOT', root)
    (root/'source.py').write_text('original', encoding='utf-8')
    dump(root/'data/split.json', ['train_only'])
    cases = {'manifest': {'source_files': {'data/split.json': file_sha(root/'data/split.json')}}}
    source = {'files': {'source.py': file_sha(root/'source.py')}}
    protocol = {'seeds': [42, 43, 44]}
    context = {'dataset_sha256': digest(cases), 'protocol_sha256': digest(protocol), 'source': source}
    for name, value in [('config.json', {}), ('cases.json', cases), ('preflight.json', {}),
                        ('source.json', source), ('protocol.json', protocol), ('context.json', context)]:
        dump(output/name, value)
    snapshot = output/'source_snapshot/source.py'
    snapshot.parent.mkdir()
    snapshot.write_text('original', encoding='utf-8')
    frozen = {p.name: file_sha(p) for p in output.glob('*.json')}
    dump(output/'frozen.json', frozen)
    dump(output/'run.json', {'frozen_sha256': file_sha(output/'frozen.json')})
    return root, output


@pytest.mark.parametrize('target', ['source.py', 'data/split.json', 'experiments/runs/a/source_snapshot/source.py',
                                  'experiments/runs/a/cases.json', 'experiments/runs/a/frozen.json'])
def test_modified_sources_data_or_snapshot_block_phase_reuse(tmp_path, monkeypatch, target):
    root, output = fixture_study(tmp_path, monkeypatch)
    driver.verify_study(output)
    with (root/target).open('a', encoding='utf-8') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='changed'):
        driver.verify_study(output)


def test_completed_phase_requires_matching_artifact_and_does_not_repeat(tmp_path):
    output = tmp_path
    def work():
        path = output/'artifact.json'
        dump(path, {'value': 42})
        return [path]
    driver._phase(output, 'one', work)
    driver._phase(output, 'one', lambda: pytest.fail('Completed phase was repeated'))
    dump(output/'artifact.json', {'value': 43})
    with pytest.raises(ValueError, match='artifact changed'):
        driver._phase(output, 'one', work)


def test_incomplete_phase_cannot_silently_restart(tmp_path):
    with pytest.raises(RuntimeError, match='interrupted'):
        driver._phase(tmp_path, 'one', lambda: (_ for _ in ()).throw(RuntimeError('interrupted')))
    with pytest.raises(ValueError, match='implicit training restart'):
        driver._phase(tmp_path, 'one', lambda: pytest.fail('Must not restart'))


def test_output_rejects_parent_and_escaping_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, 'ROOT', tmp_path)
    for path in (tmp_path/'experiments/runs', tmp_path/'experiments/runs/../../escape', tmp_path.parent/'outside'):
        with pytest.raises(ValueError, match='new child'):
            driver._run_directory(path)
