"""Reviewed training data is gated before allocating a run or learning agent."""
from argparse import Namespace
from copy import deepcopy
from dataclasses import asdict
import pytest

from smartcfd.core.types import ENVIRONMENT_VERSION, EnvConfig
from smartcfd.pipeline import demo_training, workflow
from smartcfd.pipeline.config import environment_config, read_config
from smartcfd.pipeline.data import data_version, load_samples
from smartcfd.pipeline.demo_audit import _audit_episode
from smartcfd.baselines.rule_partition import solve
from smartcfd.geometry.domain import make_domain
from smartcfd.rl.env import BlockEnv


def test_pending_package_refuses_before_agent_or_run(monkeypatch):
    from smartcfd.pipeline import review
    monkeypatch.setattr(review, 'validate_training_package', lambda *a, **k: ({'training_ready': False}, [], []))
    def forbidden(*a, **k):
        pytest.fail('Pending review must not allocate a run or construct a learning agent')
    monkeypatch.setattr(workflow, 'create_trainer', forbidden)
    monkeypatch.setattr(workflow, 'new_run', forbidden)
    args = Namespace(split='train', method='bc_sac', demo_package='pending-fixture')
    with pytest.raises(ValueError, match='not training-ready'):
        workflow.train(read_config(), args)


@pytest.fixture(scope='module')
def current_episodes():
    config = read_config('configs/smoke.yaml')
    samples, membership = {}, {}
    for split in ('train', 'val'):
        sample = load_samples(split, limit=1)[0]
        samples[sample['sample_id']] = sample
        membership[sample['sample_id']] = split
    audited = {}
    for sample_id, sample in samples.items():
        domain = make_domain(sample, **config['domain'])
        env = BlockEnv([domain], environment_config(config))
        try:
            env.reset(seed=42, options={'sample_id': sample_id})
            raw = {'sample_id': sample_id, 'source': 'rule_synthetic', 'seed': 42,
                   'actions': solve(env, seed=42), 'env_config': asdict(env.config), 'environment_version': ENVIRONMENT_VERSION}
            audited[membership[sample_id]] = _audit_episode(raw, config, samples, membership, data_version())
        finally:
            env.close()
    return config, audited


def test_loader_recreates_train_and_val_transitions_separately(monkeypatch, current_episodes):
    from smartcfd.pipeline import review
    config, episodes = current_episodes
    # This stub isolates the replay loader. Real package/approval authentication
    # is tested by test_offline_review, using explicit synthetic review verdicts.
    monkeypatch.setattr(review, 'validate_training_package', lambda *a, **k:
        ({'training_ready': True, 'source_package_id': 'unit-test-only'}, [episodes['train']], [episodes['val']]))
    train, val, report = demo_training.load_demo_package('fixture-only', config)
    assert len(train) == len(val) == 3
    assert train[-1].terminated and val[-1].terminated
    assert report['train_unique_geometries'] == report['val_unique_geometries'] == 1
    assert report['validation_enters_replay'] is False and report['test_used'] is False
    assert train[0].observation is not val[0].observation


def test_loader_rejects_cross_split_even_if_bundle_layer_is_broken(monkeypatch, current_episodes):
    from smartcfd.pipeline import review
    config, episodes = current_episodes
    monkeypatch.setattr(review, 'validate_training_package', lambda *a, **k:
        ({'training_ready': True}, [episodes['val']], [episodes['val']]))
    with pytest.raises(ValueError, match='split conflicts'):
        demo_training.load_demo_package('fixture-only', config)


def test_loader_rejects_environment_change_before_any_learning(monkeypatch, current_episodes):
    from smartcfd.pipeline import review
    config, episodes = current_episodes
    changed = deepcopy(config)
    changed['env']['gamma'] = .5
    monkeypatch.setattr(review, 'validate_training_package', lambda *a, **k:
        ({'training_ready': True}, [episodes['train']], [episodes['val']]))
    with pytest.raises(ValueError, match='env_config mismatch'):
        demo_training.load_demo_package('fixture-only', changed)


def test_from_scratch_sac_does_not_silently_consume_a_reviewed_package(monkeypatch):
    monkeypatch.setattr(demo_training, 'load_demo_package', lambda *a, **k: pytest.fail('must not read package'))
    args = Namespace(split='train', method='sac', demo_package='anything')
    with pytest.raises(ValueError, match='from-scratch SAC'):
        workflow.train({}, args)
