"""Checkpoint configuration provenance regression tests.

Creates tiny CPU agents and empty replay only. No updates, episodes, geometry
generation, CFD solver, remote access or production checkpoints are used.
"""
from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile

import pytest
import torch

from smartcfd.pipeline import workflow
from smartcfd.pipeline.config import environment_context, write_json
from smartcfd.pipeline.data import repository_root
from smartcfd.rl.trainer import Trainer


def arguments(checkpoint=None, **overrides):
    values = {'device': 'cpu', 'checkpoint': str(checkpoint) if checkpoint else None,
              'task': 'train', 'seed': 7, 'method': 'sac', 'split': 'val',
              'limit': 0, 'output': None, 'final_test': False}
    values.update(overrides)
    return Namespace(**values)


@pytest.fixture
def checkpoint_fixture(monkeypatch):
    # Confine all test artifacts to the implementation copy, including cleanup.
    base = repository_root() / '.runtime' / 'resume_config_guard_test_fixtures'
    base.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with tempfile.TemporaryDirectory(prefix='case_', dir=base) as temporary:
        root = Path(temporary).resolve()
        assert root.is_relative_to(base.resolve()) and root != base.resolve()
        monkeypatch.setattr(workflow, 'repository_root', lambda: root)
        frozen_data = {'version': 'fixture_frozen_bytes'}
        monkeypatch.setattr(workflow, 'data_version', lambda: deepcopy(frozen_data))
        config = {'model': {'hidden_dim': 8, 'layers': 1},
                  'training': {'replay_capacity': 8, 'replay_max_bytes': 1_000_000}}
        trainer = workflow.create_trainer(config, arguments())
        trainer.online_steps = 123
        path = root / 'source.pt'
        trainer.save_checkpoint(path, metadata={
            'data': frozen_data, 'test_used': False,
            'environment_context': environment_context(config), 'config': deepcopy(config)})
        yield root, path, config
        assert root.is_relative_to(base.resolve()) and root != base.resolve()


def forbid_restore(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Mismatching resume reached model/RNG restoration')
    monkeypatch.setattr(Trainer, 'load_checkpoint', forbidden)


@pytest.mark.parametrize(('section', 'field', 'value', 'error_field'), [
    # The first case is the observed bug: a requested .01 must not resume .1.
    ('training', 'alpha', .01, 'sac_config.alpha'),
    ('training', 'gamma', .95, 'sac_config.gamma'),
    ('training', 'tau', .01, 'sac_config.tau'),
    ('training', 'learning_rate', .0001, 'sac_config.lr'),
    ('training', 'grad_clip', .5, 'sac_config.grad_clip'),
    ('model', 'feature_dim', 25, 'network_config.feature_dim'),
    ('model', 'global_dim', 9, 'network_config.global_dim'),
    ('model', 'hidden_dim', 16, 'network_config.hidden_dim'),
    ('model', 'layers', 2, 'network_config.layers'),
    ('model', 'local_actions', 5, 'network_config.local_actions'),
    ('training', 'replay_capacity', 16, 'replay_config.capacity'),
    ('training', 'replay_max_bytes', 2_000_000, 'replay_config.max_bytes'),
])
def test_training_resume_rejects_actual_settings_change_before_restore(
        checkpoint_fixture, monkeypatch, section, field, value, error_field):
    _, path, original = checkpoint_fixture
    requested = deepcopy(original)
    requested[section][field] = value
    before = path.read_bytes()
    rng = torch.get_rng_state().clone()
    forbid_restore(monkeypatch)
    with pytest.raises(ValueError, match=error_field.replace('.', r'\.')) as exc:
        workflow.create_trainer(requested, arguments(path))
    assert 'requested=' in str(exc.value) and 'checkpoint=' in str(exc.value)
    assert torch.equal(torch.get_rng_state(), rng)
    assert path.read_bytes() == before


@pytest.mark.parametrize('task', ['train', 'bc'])
def test_resume_rejects_cli_seed_mismatch(checkpoint_fixture, monkeypatch, task):
    _, path, config = checkpoint_fixture
    before = path.read_bytes()
    forbid_restore(monkeypatch)
    with pytest.raises(ValueError, match='agent_seed.*trainer_seed'):
        workflow.create_trainer(config, arguments(path, seed=8, task=task))
    assert path.read_bytes() == before


@pytest.mark.parametrize('changed_seed', ['agent_seed', 'trainer_seed'])
def test_checkpoint_agent_and_trainer_seeds_are_checked_independently(
        checkpoint_fixture, monkeypatch, changed_seed):
    root, source, config = checkpoint_fixture
    state = torch.load(source, map_location='cpu', weights_only=False)
    if changed_seed == 'agent_seed':
        state['agent']['seed'] = 8
    else:
        state['trainer_seed'] = 8
    variant = root / f'{changed_seed}.pt'
    torch.save(state, variant)
    forbid_restore(monkeypatch)
    with pytest.raises(ValueError, match=changed_seed):
        workflow.create_trainer(config, arguments(variant))


def test_schedule_and_validation_changes_allow_a_real_empty_replay_resume(checkpoint_fixture):
    _, path, config = checkpoint_fixture
    requested = deepcopy(config)
    requested['training'].update(total_steps=1000, warmup_steps=0, batch_size=2,
        validation_episodes=3, evaluation_interval=100, bc_epochs=2, bc_patience=1,
        demo_episodes=4, critic_warmup_updates=0)
    restored = workflow.create_trainer(requested, arguments(path))
    assert restored.online_steps == 123 and len(restored.replay) == 0
    assert restored.agent.config.alpha == .1
    assert restored.agent.seed == restored.seed == 7
    assert restored.workflow_settings['policy'] == 'strict_resume'
    assert restored.workflow_settings['requested'] == restored.workflow_settings['effective']


def test_explicit_defaults_match_omitted_agent_and_network_defaults(checkpoint_fixture):
    _, path, config = checkpoint_fixture
    requested = deepcopy(config)
    requested['training'].update(alpha=.1, gamma=.99, tau=.005,
                                 learning_rate=.0003, grad_clip=1.)
    requested['model'].update(feature_dim=24, global_dim=8, local_actions=4)
    restored = workflow.create_trainer(requested, arguments(path))
    assert restored.agent.network_config.hidden_dim == 8
    assert restored.workflow_settings['requested'] == restored.workflow_settings['effective']


def test_fresh_construction_honors_exposed_grad_clip_and_reports_effective_values(checkpoint_fixture):
    _, _, config = checkpoint_fixture
    requested = deepcopy(config)
    requested['training'].update(alpha=.01, grad_clip=.25)
    fresh = workflow.create_trainer(requested, arguments())
    assert fresh.agent.config.alpha == .01
    assert fresh.agent.config.grad_clip == .25
    assert fresh.workflow_settings['policy'] == 'fresh'
    assert fresh.workflow_settings['requested'] == fresh.workflow_settings['effective']


def test_resume_checks_inner_replay_capacity_that_loader_actually_restores(
        checkpoint_fixture, monkeypatch):
    root, source, config = checkpoint_fixture
    state = torch.load(source, map_location='cpu', weights_only=False)
    assert state['replay'] is not None
    state['replay']['capacity'] = 9
    variant = root / 'inconsistent_replay.pt'
    torch.save(state, variant)
    forbid_restore(monkeypatch)
    with pytest.raises(ValueError, match='replay contents and replay_config disagree'):
        workflow.create_trainer(config, arguments(variant))


@pytest.mark.parametrize(('gate', 'message'), [
    ('environment_version', 'environment version mismatch'),
    ('environment_context', 'terminal-mesh environment context'),
    ('data_version', 'data version mismatch'),
    ('data_bytes', 'different frozen geometry/split bytes'),
    ('test_used', 'test_used=False'),
])
def test_provenance_gates_precede_settings_mismatch(
        checkpoint_fixture, monkeypatch, gate, message):
    root, source, config = checkpoint_fixture
    state = torch.load(source, map_location='cpu', weights_only=False)
    if gate == 'environment_version':
        state['environment_version'] = 'legacy_v1'
    elif gate == 'environment_context':
        state['environment_context']['env_config']['max_steps'] += 1
    elif gate == 'data_version':
        state['agent']['data_version'] = 'other_version'
    elif gate == 'data_bytes':
        state['agent']['metadata']['data'] = {'version': 'different_bytes'}
    else:
        state['agent']['metadata']['test_used'] = True
    variant = root / f'{gate}.pt'
    torch.save(state, variant)
    requested = deepcopy(config)
    requested['training']['alpha'] = .01
    forbid_restore(monkeypatch)
    with pytest.raises(ValueError, match=message):
        workflow.create_trainer(requested, arguments(variant))


def test_final_test_freeze_precedes_deserialization_and_config_resolution(checkpoint_fixture, monkeypatch):
    root, _, _ = checkpoint_fixture
    opaque = root / 'opaque.pt'
    opaque.write_bytes(b'not a torch checkpoint')
    manifest = root / 'experiments/runs/frozen/final_test_manifest.json'
    manifest.parent.mkdir(parents=True)
    write_json(manifest, {'checkpoint_sha256': hashlib.sha256(opaque.read_bytes()).hexdigest()})
    forbid_restore(monkeypatch)
    with pytest.raises(ValueError, match='frozen for final test'):
        workflow.create_trainer({}, Namespace(device='cpu', checkpoint=str(opaque), task='train'))


@pytest.mark.parametrize('task', ['evaluate', 'mesh'])
def test_inference_uses_checkpoint_and_exposes_requested_effective_difference(checkpoint_fixture, task):
    _, path, config = checkpoint_fixture
    requested = deepcopy(config)
    requested['training']['alpha'] = .01
    requested['model']['hidden_dim'] = 16
    before = path.read_bytes()
    restored = workflow.create_trainer(requested, arguments(path, task=task, seed=8))
    assert restored.agent.config.alpha == .1
    assert restored.agent.network_config.hidden_dim == 8
    assert restored.agent.seed == restored.seed == 7
    report = restored.workflow_settings
    assert report['policy'] == 'checkpoint_for_inference'
    assert report['requested']['sac_config']['alpha'] == .01
    assert report['effective']['sac_config']['alpha'] == .1
    assert report['requested']['network_config']['hidden_dim'] == 16
    assert report['effective']['network_config']['hidden_dim'] == 8
    assert report['requested']['agent_seed'] == 8
    assert report['effective']['agent_seed'] == 7
    assert path.read_bytes() == before


def test_inference_metrics_and_run_result_record_effective_settings_without_running_episodes(
        checkpoint_fixture, monkeypatch):
    root, path, config = checkpoint_fixture
    requested = deepcopy(config)
    requested['training']['alpha'] = .01
    requested['model']['hidden_dim'] = 16
    run = root / 'empty_inference_record_fixture'
    run.mkdir()
    write_json(run / 'run.json', {'config': requested, 'status': 'started'})
    monkeypatch.setattr(workflow, 'new_run', lambda *a, **k: run)
    monkeypatch.setattr(workflow, 'load_domains', lambda *a, **k: [])
    # No evaluate(), act(), step() or mesh generation occurs with an empty list.
    actual_run, result = workflow.evaluate(requested, arguments(path, task='evaluate'))
    report = result['trainer_settings']
    assert actual_run == run and result['episodes'] == []
    assert report['requested']['sac_config']['alpha'] == .01
    assert report['effective']['sac_config']['alpha'] == .1
    assert json.loads((run / 'metrics.json').read_text())['trainer_settings'] == report
    receipt = json.loads((run / 'run.json').read_text())
    assert receipt['config']['training']['alpha'] == .01
    assert receipt['result']['trainer_settings'] == report
