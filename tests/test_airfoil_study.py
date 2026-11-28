"""Study isolation, source-checked replay evidence, teacher and learning checks."""
from dataclasses import asdict
import copy
import json

import numpy as np
import pytest
import torch

from smartcfd.airfoil_decomposition import study
from smartcfd.airfoil_decomposition.config import AirfoilDecompositionConfig
from smartcfd.airfoil_decomposition.env import synthetic_profile
from smartcfd.airfoil_decomposition.experiment import normalize_config, file_sha, write_json
from smartcfd.airfoil_decomposition.data import digest


@pytest.fixture(autouse=True)
def cpu_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def observation():
    features = np.zeros((256, 32), dtype=np.float32)
    features[:4, 0] = [0., .2, .7, 1.]
    index = np.full(256, -1, dtype=np.int64)
    index[:4] = [1, 2, 3, 0]
    previous = np.full(256, -1, dtype=np.int64); previous[:4] = [3, 0, 1, 2]
    mask = np.zeros(1024, dtype=bool); mask[[0, 3]] = True
    nodes = np.full((1024, 2), -1, dtype=np.int64); nodes[[0, 3]] = [[0, 1], [2, 3]]
    return {'features': features, 'next': index, 'previous': previous, 'twin': np.full(256, -1, dtype=np.int64),
            'global': np.zeros(12, dtype=np.float32), 'candidates': np.zeros((1024, 16), dtype=np.float32),
            'candidate_nodes': nodes, 'mask': mask}


def config():
    return normalize_config({'environment': {'max_steps': 3},
        'training': {'steps': 2, 'warmup_steps': 1, 'batch_size': 2, 'evaluation_interval': 2}})


def context():
    return {'source': {'source_sha256': 'a'*64}, 'dataset_sha256': 'b'*64}


def teacher_manifest(tmp_path, *, groups=('a', 'a', 'b')):
    rows = []
    obs = observation()
    for index, group in enumerate(groups):
        steps = 2 if index == 0 else 1
        arrays = {k: np.stack([v]*steps) for k, v in obs.items()}
        arrays['actions'] = np.zeros(steps, dtype=np.int64)
        path = tmp_path/f'{index}.npz'; np.savez_compressed(path, **arrays)
        evidence = tmp_path/f'{index}.json'; write_json(evidence, {'success': True})
        rows.append({'sample_id': f's{index}', 'group_id': group, 'path': path.name,
                     'steps': steps, 'sha256': file_sha(path), 'evidence': evidence.name,
                     'evidence_sha256': file_sha(evidence)})
    manifest = {'schema': study.TEACHER_SCHEMA, 'selected': rows,
                'environment': config()['environment'], 'total': len(rows), 'available': len(rows)}
    manifest['dataset_sha256'] = digest(manifest)
    path = tmp_path/'manifest.json'; write_json(path, manifest)
    return path


def test_preflight_failure_denominator_and_hash_checked_case_loading(tmp_path):
    cases = [{'sample_id': 'failed', 'group_id': 'family'}]
    report = study.evaluate_cases(cases, config()['environment'], tmp_path/'eval', method='random',
               preflight_rows=[{'sample_id': 'failed', 'valid': False, 'reason': 'mapping_failed'}])
    assert report['success_count'] == 0 and report['total'] == 1
    assert report['mean_penalized_steps'] == 3 and report['initializable_count'] == 0
    assert study.load_evaluation_rows(tmp_path/'eval/summary.json')[0]['phase'] == 'preflight'
    path = tmp_path/'eval'/report['episodes'][0]['path']
    path.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='hash'):
        study.load_evaluation_rows(report)


def test_study_checkpoint_strict_profile_and_legacy_rejection(tmp_path):
    cfg = config(); profile = study.make_profile(cfg, None, context(), method='sac')
    agent = study._agent(cfg, 42)
    path = tmp_path/'study.pt'
    study.save_study_checkpoint(agent, path, profile=profile, online_step=2, selection='final')
    loaded, state = study.load_study_checkpoint(path, expected_profile=profile)
    assert torch.equal(agent.actor(observation()), loaded.actor(observation()))
    assert state['training_resume_supported'] is False
    wrong = copy.deepcopy(profile); wrong['guard'] = {'enabled': True}
    with pytest.raises(ValueError, match='profile'):
        study.load_study_checkpoint(path, expected_profile=wrong)
    state['schema'] = 'airfoil_c_decomposition_checkpoint_v1'; torch.save(state, tmp_path/'legacy.pt')
    with pytest.raises(ValueError, match='legacy'):
        study.load_study_checkpoint(tmp_path/'legacy.pt', expected_profile=profile)


def test_teacher_family_sampling_and_bounded_cache(tmp_path):
    import random
    dataset = study.TeacherDataset(teacher_manifest(tmp_path))
    rng = random.Random(42)
    counts = {'s0': 0, 's1': 0, 's2': 0}
    for _ in range(3000):
        obs, action, sample = dataset.sample(rng)
        counts[sample] += 1
        assert obs['mask'][action] and len(dataset.cache) <= 2
    assert 1300 < counts['s2'] < 1700  # one of two families, despite fewer geometries/states
    assert 550 < counts['s0'] < 950 and 550 < counts['s1'] < 950
    with open(tmp_path/'0.npz', 'ab') as stream:
        stream.write(b'tamper')
    with pytest.raises(ValueError, match='hash'):
        study.TeacherDataset(tmp_path/'manifest.json')


def test_bc_real_update_fixed_final_and_context(tmp_path):
    teacher = teacher_manifest(tmp_path)
    cfg = config(); initial = study._agent(cfg, 43)
    weights = copy.deepcopy(initial.actor.state_dict())
    before_probability = float(initial.actor.distribution(observation())[0][0, 0].detach())
    result = study.train_bc_arm(teacher, cfg, tmp_path/'bc', seed=43, context=context(), updates=2)
    agent, state = study.load_study_checkpoint(tmp_path/'bc/final.pt', expected_profile=result['profile'])
    assert result['updates'] == agent.actor_updates == 2
    assert agent.updates == 0 and result['profile']['training']['seed'] == 43
    assert any(not torch.equal(weights[k], agent.actor.state_dict()[k]) for k in weights)
    assert float(agent.actor.distribution(observation())[0][0, 0].detach()) > before_probability
    assert state['selection'] == 'fixed_bc_final'
    assert len((tmp_path/'bc/updates.jsonl').read_text().splitlines()) == 2
    cases = [{'sample_id': 'preflight_only', 'group_id': 'f'}]
    checked = [{'sample_id': 'preflight_only', 'valid': False, 'reason': 'test_fixture_mapping_failure'}]
    binding = {'checkpoint': str(tmp_path/'bc/final.pt'), 'checkpoint_sha256': result['checkpoint_sha256'],
               'profile': result['profile']}
    for label, guard in (('off', None), ('on', {'depth': 2, 'second_layer_budget': 128})):
        report = study.evaluate_cases(cases, cfg['environment'], tmp_path/f'bc_eval_{label}', method='bc',
               seed=43, guard_config=guard, agent=agent, preflight_rows=checked, checkpoint_binding=binding)
        assert report['checkpoint_binding']['checkpoint_sha256'] == result['checkpoint_sha256']
        assert report['checkpoint_binding']['profile_sha256'] == digest(result['profile'])
        assert report['checkpoint_binding']['verified_checkpoint_unchanged']
        assert report['network_identity']['network_hashes_before'] == report['network_identity']['network_hashes_after']
    wrong = {**binding, 'checkpoint_sha256': '0'*64}
    with pytest.raises(ValueError, match='SHA'):
        study.evaluate_cases(cases, cfg['environment'], tmp_path/'bad_sha', method='bc', seed=43,
                              agent=agent, preflight_rows=checked, checkpoint_binding=wrong)
    with torch.no_grad():
        next(agent.actor.parameters()).add_(.1)
    with pytest.raises(ValueError, match='tensors'):
        study.evaluate_cases(cases, cfg['environment'], tmp_path/'bad_weights', method='bc', seed=43,
                              agent=agent, preflight_rows=checked, checkpoint_binding=binding)


def test_teacher_selection_validity_and_stable_ties(tmp_path, monkeypatch):
    def rollout(sample, env_config, *, method, **kwargs):
        valid = method != 'greedy'
        row = {'success': valid, 'partition': {'valid': valid}, 'terminal_quality_score': 100 if not valid else .4,
               'method': method, 'steps': 1, 'actions': [{'index': 0}]}
        return row, [observation()]
    monkeypatch.setattr(study, 'rollout_case', rollout)
    manifest = study.build_teachers([{'sample_id': 'train', 'group_id': 'f', 'split': 'train'}],
                                    config()['environment'], tmp_path/'teachers')
    assert manifest['selected'][0]['method'] == 'fixed_c'
    with pytest.raises(ValueError, match='train'):
        study.build_teachers([{'sample_id': 'val', 'split': 'val'}], config()['environment'], tmp_path/'bad')


def test_real_fixed_c_rollout_actions_and_partition(tmp_path):
    sample = {**synthetic_profile(), 'group_id': 'synthetic', 'split': 'train'}
    report = study.evaluate_cases([sample], AirfoilDecompositionConfig(), tmp_path/'fixed', method='fixed_c')
    row = study.load_evaluation_rows(report)[0]
    assert row['success'] and row['steps'] == 3 and row['partition']['valid']
    assert all(a['index'] in a['effective_actions'] for a in row['actions'])
    assert all(abs(a['reward']-a['base_reward']-a['shaping']) < 1e-12 for a in row['actions'])
    assert row['guard_cost']['selection_previews'] == 0


def test_real_short_sac_training_has_updates_and_final_evaluation(tmp_path):
    sample = {**synthetic_profile(), 'group_id': 'synthetic', 'split': 'train'}
    validation = {**sample, 'sample_id': 'synthetic_validation_fixture', 'group_id': 'synthetic_val', 'split': 'val'}
    result = study.train_sac_arm({'train': [sample], 'val': [validation]}, config(), tmp_path/'sac',
                                seed=42, guard_config=None, context=context())
    assert result['steps'] == result['replay_total_added'] == 2 and result['updates'] == 1
    assert result['replay_retained_bytes'] <= result['replay_max_bytes']
    assert (tmp_path/'sac/validation_000002/summary.json').is_file()
    agent, state = study.load_study_checkpoint(tmp_path/'sac/final.pt', expected_profile=result['profile'])
    assert state['selection'] == 'fixed_sac_final_primary' and agent.updates == 1
    assert len((tmp_path/'sac/interactions.jsonl').read_text().splitlines()) == 2
    binding = result['final_validation_checkpoint_binding']
    assert binding['checkpoint_sha256'] == file_sha(tmp_path/'sac/final.pt')
    assert binding['network_hashes'] == study.network_hashes(agent)
    assert binding['verified_saved_tensors'] and binding['verified_save_unchanged']
    final_report = json.loads((tmp_path/'sac/validation_000002/summary.json').read_text())
    assert final_report['checkpoint_binding'] == binding


def test_guard_initial_dead_end_is_not_mapping_failure(tmp_path):
    sample = {**synthetic_profile(), 'group_id': 'synthetic', 'split': 'train'}
    report = study.evaluate_cases([sample], AirfoilDecompositionConfig(max_steps=2), tmp_path/'guard',
             method='random', guard_config={'depth': 2, 'second_layer_budget': 128},
             preflight_rows=[{'sample_id': sample['sample_id'], 'valid': True}])
    assert report['initializable_count'] == 1 and report['success_count'] == 0
    row = study.load_evaluation_rows(report)[0]
    assert row['reason'] == 'proven_unfinishable' and row['steps'] == 0
    assert row['guard_cost']['all_pruned'] == 1 and row['cost']['geometry_validation_calls'] > 0


def test_real_guarded_sac_uses_effective_actions_and_terminal_masks(tmp_path):
    sample = {**synthetic_profile(), 'group_id': 'train', 'split': 'train'}
    validation = {**sample, 'sample_id': 'synthetic_val', 'group_id': 'val', 'split': 'val'}
    cfg = config(); cfg['training']['steps'] = cfg['training']['evaluation_interval'] = 6
    result = study.train_sac_arm({'train': [sample], 'val': [validation]}, cfg, tmp_path/'guarded', seed=42,
               guard_config={'depth': 2, 'second_layer_budget': 128}, context=context())
    interactions = [json.loads(line) for line in (tmp_path/'guarded/interactions.jsonl').read_text().splitlines()]
    assert result['updates'] == 5
    assert all(row['index'] in row['effective_actions'] for row in interactions)
    assert interactions[-1]['terminated'] and interactions[0]['guard']['version']
    assert all(abs(row['reward']-row['base_reward']-row['shaping']) < 1e-12 for row in interactions)
    assert result['profile']['guard']['enabled'] is True
    assert interactions[3]['guard']['guard_cache_hit'] == 1
    assert result['guard_cache']['retained_bytes'] <= result['guard_cache']['max_bytes']
    final = json.loads((tmp_path/'guarded/validation_000006/summary.json').read_text())
    initial = json.loads((tmp_path/'guarded/validation_000000/summary.json').read_text())
    assert final['selection'] == 'final' and initial['selection'] == 'development_intermediate'


def test_guard_cache_global_limit_evicts_pure_results_only():
    pool = study.GuardCachePool({'second_layer_budget': 0}, max_bytes=5000)
    first = pool.acquire('first')
    first._cache['state'] = {'effective_mask': np.zeros(400, dtype=np.float64)}
    pool.trim('first')
    assert first._cache and pool.statistics()['retained_bytes'] <= 5000
    second = pool.acquire('second')
    second._cache['state'] = {'effective_mask': np.zeros(400, dtype=np.float64)}
    pool.trim('second')
    assert not first._cache and second._cache
    assert pool.statistics()['evicted_guards'] == 1
    second._cache['large'] = {'value': np.zeros(10000)}
    pool.trim('second')
    assert not second._cache and pool.statistics()['retained_bytes'] <= 5000


def test_guard_cache_across_fresh_geometry_preserves_masks_rewards_and_statistics():
    sample = synthetic_profile()
    cfg = AirfoilDecompositionConfig(max_steps=3)
    settings = {'depth': 2, 'second_layer_budget': 128}
    pool = study.GuardCachePool(settings)
    executions = []
    previous_geometry = None
    for _ in range(2):
        env = study._environment(sample, cfg, settings, pool)
        obs, info = env.reset(seed=42)
        assert env.geometry is not previous_geometry
        previous_geometry = env.geometry
        pool.trim(sample['sample_id'])
        masks, rewards = [obs['mask'].copy()], []
        initial_report = copy.deepcopy(info['guard'])
        while not env.terminated:
            action = study.fixed_action(env)
            assert action is not None and obs['mask'][action]
            obs, reward, _, _, info = env.step(action)
            pool.trim(sample['sample_id'])
            masks.append(obs['mask'].copy()); rewards.append(reward)
        executions.append((masks, rewards, env.partition(), dict(env.guard_statistics), initial_report))
    cold, warm = executions
    assert all(np.array_equal(a, b) for a, b in zip(cold[0], warm[0]))
    assert cold[1] == warm[1] and cold[2] == warm[2]
    assert cold[3]['decisions'] == warm[3]['decisions'] == 3
    assert cold[3]['guard_cache_hit'] == 0 and warm[3]['guard_cache_hit'] == 3
    assert cold[4]['geometry_validation_calls'] > 0 and warm[4]['geometry_validation_calls'] == 0


def test_evaluation_detects_weight_mutation_without_replaying_cases(tmp_path, monkeypatch):
    agent = study._agent(config(), 42)
    def mutate(sample, env_config, **kwargs):
        with torch.no_grad():
            next(agent.actor.parameters()).add_(.01)
        row = study._record_base(sample, 'bc', 42)
        row.update(penalized_steps=3, reason='fixture', steps=0)
        return row, []
    monkeypatch.setattr(study, 'rollout_case', mutate)
    with pytest.raises(ValueError, match='changed network'):
        study.evaluate_cases([{'sample_id': 'fixture'}], config()['environment'], tmp_path/'mutation',
                              method='bc', agent=agent)
    evidence = json.loads((tmp_path/'mutation/network_identity_failure.json').read_text())
    assert not evidence['unchanged_during_validation']
    assert not (tmp_path/'mutation/summary.json').exists()


@pytest.mark.parametrize('entrypoint', ['evaluate', 'teachers'])
def test_runtime_exception_keeps_completed_and_pending_actions_raw_graph_then_reraises(tmp_path, monkeypatch, entrypoint):
    real_step = study.AirfoilBlockEnv.step
    def fail_after_commit(env, action):
        result = real_step(env, action)
        if env.steps == 2:
            raise FloatingPointError('deliberate observation failure after topology commit')
        return result
    monkeypatch.setattr(study.AirfoilBlockEnv, 'step', fail_after_commit)
    sample = {**synthetic_profile(), 'group_id': 'f', 'split': 'train'}
    output = tmp_path/entrypoint
    with pytest.raises(FloatingPointError, match='after topology commit'):
        if entrypoint == 'evaluate':
            study.evaluate_cases([sample], AirfoilDecompositionConfig(), output, method='fixed_c')
        else:
            study.build_teachers([sample], AirfoilDecompositionConfig(), output)
    files = list(output.glob('*.diagnostic_exception.json'))
    assert len(files) == 1
    report = json.loads(files[0].read_text(encoding='utf-8'))
    assert report['method'] == 'fixed_c' and report['phase'] == 'environment_step'
    assert len(report['completed_actions']) == 1 and report['pending_action']['candidate']['kind'] == 'insert_chord'
    assert report['state']['steps'] == 2 and len(report['state']['history']) == 2
    assert report['state']['graph']['faces'] and report['last_observation']['mask']
    assert report['error_type'] == 'FloatingPointError'
    assert not (output/'summary.json').exists() and not (output/'manifest.json').exists()
