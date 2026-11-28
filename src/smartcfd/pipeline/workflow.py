"""Executable local research workflow; failures remain explicit experiment records."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from smartcfd.core.types import EnvConfig, Failure, MeshSpec
from smartcfd.pipeline.config import complete_run, environment_config, environment_context, implementation_version, new_run, write_json
from smartcfd.pipeline.data import data_version, load_domains, load_samples, repository_root


def make_environment(config, split="train", limit=None, domains=None):
    from smartcfd.rl.env import BlockEnv
    domain_cfg = config.get("domain", {})
    domains = domains if domains is not None else load_domains(split, limit=limit, **domain_cfg)
    if not domains:
        raise ValueError(f"No eligible sharp geometries in {split}")
    return BlockEnv(domains, environment_config(config))


def checkpoint_path(value):
    path = Path(value)
    return path if path.is_absolute() else repository_root() / path


def freeze_test_snapshot(run, config, args):
    write_json(run / 'final_test_manifest.json', {'frozen_config': config, 'data': data_version(),
        'checkpoint_sha256': hashlib.sha256(checkpoint_path(args.checkpoint).read_bytes()).hexdigest() if args.checkpoint else None,
        'method': args.method, 'purpose': 'Final held-out evaluation; never tune using these outcomes'})


def prepare(config, args):
    run = new_run("prepare", config, args.output)
    counts = {}
    for split in ("train", "val", "test"):
        samples = load_samples(split)
        ids = [s["sample_id"] for s in samples]
        write_json(run / f"{split}.json", ids)
        counts[split] = len(ids)
    result = {"sharp_geometry_counts": counts, "frozen_splits_unchanged": True,
              "predicate": "unique xmax within 1e-10, TE angle <=60 degrees, explicitly closed source"}
    complete_run(run, result)
    return run, result


def draw_topology(graph, path, title="C-parameter coarse-block topology"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for face in graph.faces.values():
        points = np.asarray([graph.vertices[v] for v in face.vertices])
        ax.fill(points[:, 0], points[:, 1], alpha=.16,
                color="steelblue" if face.protected else "darkorange")
        p = np.vstack([points, points[0]])
        ax.plot(p[:, 0], p[:, 1], "k-", linewidth=.8)
        ax.text(*points.mean(0), str(face.id), fontsize=8, ha="center")
    ax.set(xlabel="u (parameter)", ylabel="v (parameter)", title=title, aspect="equal")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw_mesh(mesh, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name, limits in (("mesh_overview.png", None), ("mesh_near_wall.png", (-.25, 1.5, -.6, .6))):
        fig, ax = plt.subplots(figsize=(9, 5))
        for a in mesh.blocks.values():
            for i in range(0, a.shape[0], max(1, a.shape[0] // 32)):
                ax.plot(a[i, :, 0], a[i, :, 1], color="#24617c", linewidth=.35)
            for j in range(0, a.shape[1], max(1, a.shape[1] // 32)):
                ax.plot(a[:, j, 0], a[:, j, 1], color="#24617c", linewidth=.35)
        ax.set(xlabel="x/c", ylabel="y/c", title="Generated all-quadrilateral multi-block mesh", aspect="equal")
        if limits:
            ax.axis(limits)
        fig.tight_layout()
        fig.savefig(Path(output) / name, dpi=170)
        plt.close(fig)


def generate(config, args):
    from smartcfd.baselines.rule_partition import solve
    from smartcfd.meshing.generator import generate_mesh, save_mesh
    from smartcfd.cfd.solver import export_su2
    if args.split == 'test' and not getattr(args, 'final_test', False):
        raise ValueError('Test meshes require an explicit --final-test snapshot')
    if args.method != 'rule' and not args.checkpoint:
        raise ValueError('Learned mesh generation requires --checkpoint')
    run = new_run("mesh", config, args.output)
    if args.split == 'test':
        freeze_test_snapshot(run, config, args)
    trainer = create_trainer(config, args) if args.method != 'rule' else None
    domains = load_domains(args.split, limit=args.limit or 1, **config.get("domain", {}))
    spec = MeshSpec(**config.get("mesh", {}))
    records = []
    for index, domain in enumerate(domains):
        started = time.perf_counter()
        env = make_environment(config, domains=[domain])
        seed = args.seed + index
        env.reset(seed=seed, options={"sample_id": domain.sample_id})
        if trainer is None:
            actions = solve(env, seed=seed)
        else:
            observation, _ = env.reset(seed=seed, options={'sample_id': domain.sample_id})
            actions = []
            for _ in range(env.config.max_steps):
                action = trainer.agent.act(observation, deterministic=True)
                actions.append(action)
                observation, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    if not info.get('success', False):
                        actions = Failure(info.get('reason', 'learned_partition_failed'), diagnostics=info)
                    break
        if isinstance(actions, Failure):
            records.append({"sample_id": domain.sample_id, "success": False, "reason": actions.reason,
                            "detail": actions.detail, "diagnostics": actions.diagnostics})
            continue
        observation, _ = env.reset(seed=seed, options={"sample_id": domain.sample_id})
        for action in actions:
            observation, _, terminated, truncated, info = env.step(action)
        mesh = generate_mesh(domain, env.graph, spec)
        if isinstance(mesh, Failure) or not mesh.quality.get('valid', False):
            if not isinstance(mesh, Failure):
                mesh = Failure('invalid_generated_mesh', diagnostics={'quality': mesh.quality})
            records.append({"sample_id": domain.sample_id, "success": False, "reason": mesh.reason,
                            "detail": mesh.detail, "diagnostics": mesh.diagnostics})
            continue
        output = run / domain.sample_id
        output.mkdir()
        try:
            export_su2(mesh, output / "mesh.su2")
        except Exception as exc:
            failure = {'sample_id': domain.sample_id, 'success': False, 'reason': 'su2_export_failed',
                       'detail': str(exc), 'exception_type': type(exc).__name__}
            write_json(output / 'failure.json', failure)
            records.append(failure)
            continue
        save_mesh(mesh, output)
        write_json(output / "graph.json", env.graph.to_dict())
        write_json(output / "episode.json", {"sample_id": domain.sample_id, "seed": seed, "actions": actions,
            "env": asdict(env.config), "source": "rule_synthetic" if trainer is None else 'learned_policy',
            'checkpoint': args.checkpoint if trainer else None})
        draw_topology(env.graph, output / "topology.png")
        draw_mesh(mesh, output)
        records.append({"sample_id": domain.sample_id, "success": True, "blocks": len(mesh.blocks),
                        "quality": mesh.quality, "elapsed_seconds": time.perf_counter() - started})
    result = {"split": args.split, 'method': args.method, "records": records,
              "success_rate": sum(r["success"] for r in records) / max(1, len(records)),
              "scope": "coarse-block topology constrained to a fixed C parameter map"}
    if trainer is not None:
        result["trainer_settings"] = trainer.workflow_settings
    complete_run(run, result)
    return run, result


def collect_demonstrations(config, domains, count, seed, path, review_dir=None):
    from smartcfd.baselines.rule_partition import solve
    from smartcfd.rl.imitation.dataset import rollout_episode
    from smartcfd.pipeline.audit_shared import canonical_hash
    from smartcfd.partition.actions import ACTION_SCHEMA
    from smartcfd.rl.state import OBSERVATION_SCHEMA
    transitions, records, failures, reviews = [], [], [], []
    frozen = data_version()
    if review_dir is not None:
        review_dir = Path(review_dir)
        review_dir.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for episode in range(count):
            domain = domains[episode % len(domains)]
            episode_seed = seed + episode
            env = make_environment(config, domains=[domain])
            env.reset(seed=episode_seed, options={"sample_id": domain.sample_id})
            actions = solve(env, seed=episode_seed)
            if isinstance(actions, Failure):
                failures.append({"sample_id": domain.sample_id, "seed": episode_seed, "reason": actions.reason,
                                 "detail": actions.detail, "diagnostics": actions.diagnostics})
                continue
            steps = rollout_episode(env, actions, seed=episode_seed, options={"sample_id": domain.sample_id})
            transitions.extend(steps)
            record = {"sample_id": domain.sample_id, "seed": episode_seed, "actions": actions,
                      "options": {"sample_id": domain.sample_id}, "source": "rule_synthetic",
                      "human_reviewed": False, "env_config": asdict(env.config), "success": True,
                      'environment_version': env.config.environment_version, 'mesh_spec': asdict(env.mesh_spec),
                      'domain_config': {'farfield': float(domain.farfield), 'stretch': float(domain.stretch)},
                      'geometry_sha256': canonical_hash(domain.polygon), 'data': frozen,
                      'mapping_version': domain.metadata['mapping_version'],
                      'observation_version': OBSERVATION_SCHEMA, 'action_version': ACTION_SCHEMA}
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            records.append(record)
            if review_dir is not None and len(reviews) < 20:
                review_dir.mkdir(exist_ok=True)
                env.reset(seed=episode_seed, options={"sample_id": domain.sample_id})
                frames = []
                for step, action in enumerate([None, *actions]):
                    if action is not None:
                        env.step(action)
                    image = f"episode_{episode:04d}_step_{step:03d}.png"
                    draw_topology(env.graph, review_dir / image, f"Episode {episode}, step {step}; C parameter space")
                    frames.append(image)
                reviews.append({"sample_id": domain.sample_id, "seed": episode_seed, "frames": frames,
                                "review_status": "pending_human_review"})
    if review_dir is not None:
        write_json(review_dir / "review_manifest.json", reviews)
        sections = ["<!doctype html><meta charset='utf-8'><title>Synthetic demonstration review</title>",
                    "<h1>Rule-generated demonstrations: human review pending</h1>",
                    "<p>These drawings show C parameter coordinates, not physical x/c coordinates.</p>"]
        for record in reviews:
            sections.append(f"<h2>{record['sample_id']} / seed {record['seed']}</h2>")
            sections.extend(f"<img src='{name}' width='360' alt='topology step'>" for name in record["frames"])
        (review_dir / "index.html").write_text("\n".join(sections), encoding="utf-8")
    return transitions, {"requested": count, "successful": len(records), "transitions": len(transitions),
                         "failures": failures, "human_reviewed_count": 0}


def _normalized_trainer_settings(sac, network, replay, agent_seed, trainer_seed):
    """Expand defaults for comparison; exclude run schedules and device choice."""
    from smartcfd.rl.agents.policy import NetworkConfig
    from smartcfd.rl.agents.sac import SACConfig
    return {
        'sac_config': {key: float(value) for key, value in asdict(SACConfig(**sac)).items()},
        'network_config': asdict(NetworkConfig(**network)),
        'replay_config': {'capacity': int(replay['capacity']), 'max_bytes': int(replay['max_bytes'])},
        'agent_seed': int(agent_seed), 'trainer_seed': int(trainer_seed)}


def _requested_trainer_settings(config, args):
    t = config.get('training', {})
    return _normalized_trainer_settings(
        {'alpha': t.get('alpha', .1), 'gamma': t.get('gamma', .99),
         'tau': t.get('tau', .005), 'lr': t.get('learning_rate', .0003),
         'grad_clip': t.get('grad_clip', 1.)}, config.get('model', {}),
        {'capacity': t.get('replay_capacity', 20000),
         'max_bytes': t.get('replay_max_bytes', 512 * 1024**2)}, args.seed, args.seed)


def _effective_trainer_settings(trainer):
    return _normalized_trainer_settings(asdict(trainer.agent.config), asdict(trainer.agent.network_config),
        {'capacity': trainer.replay.capacity, 'max_bytes': trainer.replay.max_bytes},
        trainer.agent.seed, trainer.seed)


def _check_training_resume_settings(requested, checkpoint):
    """Reject changes before loading any model, optimizer, replay or RNG state."""
    agent = checkpoint['agent']
    saved = _normalized_trainer_settings(agent['sac_config'], agent['network_config'],
        checkpoint['replay_config'], agent['seed'], checkpoint['trainer_seed'])
    replay = checkpoint.get('replay')
    if replay is not None and any(int(replay[key]) != saved['replay_config'][key]
                                  for key in ('capacity', 'max_bytes')):
        raise ValueError('Checkpoint replay contents and replay_config disagree')
    differences = []
    for group in ('sac_config', 'network_config', 'replay_config'):
        for key in requested[group]:
            if requested[group][key] != saved[group][key]:
                differences.append(f"{group}.{key}: requested={requested[group][key]!r}, checkpoint={saved[group][key]!r}")
    for key in ('agent_seed', 'trainer_seed'):
        if requested[key] != saved[key]:
            differences.append(f"{key}: requested={requested[key]!r}, checkpoint={saved[key]!r}")
    if differences:
        raise ValueError('Training resume settings mismatch; ' + '; '.join(differences)
                         + '. Use matching settings to resume; configuration changes require fresh initialization or explicit actor-only transfer.')


def create_trainer(config, args):
    import torch
    from smartcfd.rl.agents.sac import DiscreteSACAgent
    from smartcfd.rl.trainer import Trainer
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if args.checkpoint:
        path = checkpoint_path(args.checkpoint)
        training = getattr(args, 'task', 'train') in ('train', 'bc')
        if training:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            for manifest_path in (repository_root() / 'experiments/runs').rglob('final_test_manifest.json'):
                frozen = json.loads(manifest_path.read_text(encoding='utf-8'))
                if frozen.get('checkpoint_sha256') == digest:
                    raise ValueError('This checkpoint is frozen for final test evaluation and cannot resume training')
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        metadata = checkpoint["agent"].get("metadata", {})
        context = environment_context(config)
        # Preserve provenance gates before the new settings check. The full
        # loader still validates schema, observation/action versions and states.
        if (checkpoint.get('environment_version') != context['environment_version']
                or checkpoint['agent'].get('environment_version') != context['environment_version']):
            raise ValueError('Checkpoint environment version mismatch')
        if (checkpoint.get('environment_context') != context
                or metadata.get('environment_context') != context):
            raise ValueError('Checkpoint terminal-mesh environment context does not match the requested configuration')
        if checkpoint['agent'].get('data_version') != 'uiuc_v1':
            raise ValueError('Checkpoint data version mismatch')
        if metadata.get("data") != data_version():
            raise ValueError("Checkpoint was trained against different frozen geometry/split bytes")
        if training and metadata.get('test_used', True):
            raise ValueError('Training requires a checkpoint explicitly marked test_used=False')
        requested = _requested_trainer_settings(config, args)
        if training:
            _check_training_resume_settings(requested, checkpoint)
        trainer = Trainer.load_checkpoint(path, device=device, expected_data_version="uiuc_v1",
                                         expected_environment_context=context)
        trainer.workflow_settings = {'policy': 'strict_resume' if training else 'checkpoint_for_inference',
                                     'requested': requested, 'effective': _effective_trainer_settings(trainer)}
        return trainer
    requested = _requested_trainer_settings(config, args)
    agent = DiscreteSACAgent(config=requested['sac_config'], network_config=requested['network_config'],
                             device=device, seed=requested['agent_seed'])
    trainer = Trainer(agent, replay_capacity=requested['replay_config']['capacity'],
                      replay_max_bytes=requested['replay_config']['max_bytes'], seed=requested['trainer_seed'])
    trainer.workflow_settings = {'policy': 'fresh', 'requested': requested,
                                 'effective': _effective_trainer_settings(trainer)}
    return trainer


def train(config, args, bc_only=False):
    if args.split == "test":
        raise ValueError("Test geometries cannot enter training or pretraining")
    if args.split != 'train':
        raise ValueError('Training uses --split train; validation is loaded independently for model selection')
    if not bc_only and args.method not in ('sac', 'bc_sac'):
        raise ValueError('Online training --method must be sac or bc_sac')
    reviewed_package = None
    if getattr(args, 'demo_package', None):
        if not bc_only and args.method != 'bc_sac':
            raise ValueError('--demo-package is for BC or BC+SAC, not from-scratch SAC')
        from smartcfd.pipeline.demo_training import load_demo_package
        reviewed_package = load_demo_package(args.demo_package, config)
    run = new_run("bc" if bc_only else "sac", config, args.output)
    try:
        trainer = create_trainer(config, args)
        train_domains = load_domains("train", limit=args.limit, **config.get("domain", {}))
        def complexity(domain):
            points = np.asarray(domain.polygon)
            edges = np.roll(points, -1, axis=0) - points
            unit = edges / np.maximum(np.linalg.norm(edges, axis=1)[:, None], 1e-15)
            turns = np.arccos(np.clip((unit * np.roll(unit, 1, axis=0)).sum(1), -1, 1))
            return float(np.abs(turns).sum())
        train_domains.sort(key=lambda d: (complexity(d), d.sample_id))
        trainer.bind_environment(make_environment(config, domains=train_domains))
        val_domains = load_domains("val", limit=min(args.limit or 20, 20), **config.get("domain", {}))
        t = config.get("training", {})
        result = {"device": str(trainer.agent.device), "seed": args.seed,
                  "train_geometry_count": len(train_domains), "validation_geometry_count": len(val_domains),
                  "trainer_settings": trainer.workflow_settings}
        if bc_only or args.method == "bc_sac":
            if reviewed_package is None:
                demo_count = args.episodes or t.get("demo_episodes", 2000)
                train_records, demo_report = collect_demonstrations(config, train_domains, demo_count,
                    args.seed, run / "train_demonstrations.jsonl", None if args.no_review else run / "human_review")
                val_records, val_report = collect_demonstrations(config, val_domains, max(2, min(demo_count // 4, 40)),
                    args.seed + 10000, run / "val_demonstrations.jsonl")
            else:
                train_records, val_records, provenance = reviewed_package
                result['demo_package'] = provenance
                write_json(run / 'demo_package.json', provenance)
                demo_report = {'source': 'reviewed_package', 'successful': provenance['train_episodes'],
                    'transitions': len(train_records), 'human_reviewed_count': provenance['train_episodes']}
                val_report = {'source': 'audited_package_validation_only', 'successful': provenance['val_episodes'],
                    'transitions': len(val_records), 'enters_training_replay': False}
            if not train_records or not val_records:
                raise ValueError("No successful replayable train/validation demonstrations")
            result["demonstrations"], result["validation_demonstrations"] = demo_report, val_report
            result["bc"] = trainer.train_bc(train_records, val_records,
                epochs=args.epochs or t.get("bc_epochs", 20), patience=t.get("bc_patience", 5),
                batch_size=t.get("batch_size", 64))
            trainer.save_checkpoint(run / "bc.pt", metadata={"phase": "bc", "data": data_version(),
                "config": config, "review_status": 'human_reviewed' if reviewed_package else "pending_human_review", 'test_used': False,
                'demo_package': result.get('demo_package'),
                'environment_context': environment_context(config),
                'implementation': implementation_version()}, include_replay=False)
            for transition in train_records:
                trainer.replay.add(transition)
            if not bc_only:
                warmup = args.critic_warmup if args.critic_warmup is not None else t.get("critic_warmup_updates", 1000)
                result["critic_warmup"] = trainer.warmup_critic(steps=warmup, batch_size=t.get("batch_size", 64))
        validation_history = []
        best_score = (-float('inf'), -float('inf'))
        def validate(current, step):
            nonlocal best_score
            episodes = []
            for i, domain in enumerate(val_domains[:t.get('validation_episodes', 10)]):
                report = current.evaluate(make_environment(config, domains=[domain]),
                    episodes=1, seed=args.seed + 20000 + i)
                episodes.extend(report['episodes'])
            report = {'online_step': step, 'episodes': episodes,
                'success_rate': sum(e['success'] for e in episodes) / len(episodes),
                'mean_return': sum(e['return'] for e in episodes) / len(episodes)}
            validation_history.append(report)
            score = (report['success_rate'], report['mean_return'])
            if score > best_score:
                best_score = score
                current.save_checkpoint(run / 'best.pt', metadata={'phase': 'validation_selected',
                    'data': data_version(), 'config': config, 'test_used': False,
                    'formal_experiment': False, 'selected_at_online_step': step,
                    'demo_package': result.get('demo_package'),
                    'environment_context': environment_context(config),
                    'implementation': implementation_version(), 'validation': report})
            write_json(run / 'validation_history.json', validation_history)
        validate(trainer, trainer.online_steps)
        if not bc_only:
            env = make_environment(config, domains=train_domains)
            result["online"] = trainer.train_online(env, total_steps=args.steps if args.steps is not None else t.get("total_steps", 200000),
                warmup_steps=args.warmup if args.warmup is not None else t.get("warmup_steps", 1000),
                batch_size=t.get("batch_size", 64), seed=args.seed,
                validation_callback=validate, evaluation_interval=t.get('evaluation_interval', 5000))
        result['validation'] = validation_history[-1]
        result['best_validation'] = max(validation_history, key=lambda r: (r['success_rate'], r['mean_return']))
        result['selection_rule'] = 'Validation success rate, then mean return; ties keep earlier checkpoint. Test is never used.'
        trainer.save_checkpoint(run / "checkpoint.pt", metadata={"phase": "bc" if bc_only else "sac", "config": config,
            "data": data_version(), "test_used": False, "formal_experiment": False,
            'demo_package': result.get('demo_package'),
            'environment_context': environment_context(config),
            'implementation': implementation_version()})
        write_json(run / "metrics.json", result)
        complete_run(run, result)
        return run, result
    except Exception as exc:
        complete_run(run, {"error": f"{type(exc).__name__}: {exc}"}, status="failed")
        raise


def evaluate(config, args):
    from smartcfd.baselines.rule_partition import solve
    from smartcfd.meshing.generator import generate_mesh
    if args.split == 'test' and not getattr(args, 'final_test', False):
        raise ValueError('Test evaluation requires --final-test; select configuration and checkpoint on validation first')
    run = new_run("evaluate", config, args.output)
    if args.split == 'test':
        freeze_test_snapshot(run, config, args)
    domains = load_domains(args.split, limit=args.limit, **config.get("domain", {}))
    if args.method != 'rule' and not args.checkpoint:
        raise ValueError('Learned policy evaluation requires --checkpoint')
    trainer = create_trainer(config, args) if args.method != 'rule' else None
    records = []
    for i, domain in enumerate(domains):
        env = make_environment(config, domains=[domain])
        env.reset(seed=args.seed+i, options={"sample_id": domain.sample_id})
        start = time.perf_counter()
        if trainer is None:
            actions = solve(env, seed=args.seed+i)
            record = {'sample_id': domain.sample_id, 'success': not isinstance(actions, Failure),
                      'steps': len(actions) if isinstance(actions, list) else 0, 'return': 0.0}
            if isinstance(actions, Failure):
                record.update(reason=actions.reason, detail=actions.detail, diagnostics=actions.diagnostics)
            else:
                for action in actions:
                    _, reward, _, _, _ = env.step(action)
                    record['return'] += reward
        else:
            record = trainer.evaluate(env, episodes=1, seed=args.seed+i)['episodes'][0]
        if record['success']:
            mesh = generate_mesh(domain, env.graph, MeshSpec(**config.get('mesh', {})))
            if isinstance(mesh, Failure) or not mesh.quality.get('valid', False):
                if not isinstance(mesh, Failure):
                    mesh = Failure('invalid_generated_mesh', diagnostics={'quality': mesh.quality})
                record.update(success=False, reason=mesh.reason, detail=mesh.detail, diagnostics=mesh.diagnostics)
            else:
                from smartcfd.cfd.solver import validate_su2
                try:
                    stats = validate_su2(mesh)
                    record.update(blocks=len(mesh.blocks), quality=mesh.quality, su2_stats=stats)
                except Exception as exc:
                    record.update(success=False, reason='su2_export_failed', detail=str(exc), exception_type=type(exc).__name__)
        record['elapsed_seconds'] = time.perf_counter() - start
        records.append(record)
    valid = [r for r in records if r['success']]
    result = {'method': args.method, 'split': args.split, 'episodes': records,
        'success_rate': len(valid) / max(1, len(records)),
        'mean_elapsed_seconds': sum(r['elapsed_seconds'] for r in records) / max(1, len(records)),
        'quality_scope': 'Full mesh config shared by all methods; quality statistics apply only to successful meshes',
        'valid_mesh_quality_mean': {key: float(np.mean([r['quality'][key] for r in valid]))
            for key in ('min_scaled_jacobian', 'q10_scaled_jacobian', 'wall_normal_deviation_deg_max', 'cells')} if valid else None}
    if trainer is not None:
        result["trainer_settings"] = trainer.workflow_settings
    write_json(run / "metrics.json", result)
    complete_run(run, result)
    return run, result
