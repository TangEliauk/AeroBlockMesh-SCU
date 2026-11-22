"""Train a continuous SAC policy on the analytical NACA four-digit benchmark."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np

from smartcfd.geometry.airfoil import naca
from smartcfd.meshing.structured import BASE_PARAMS, PARAM_DIM, PARAM_NAMES, generate
from smartcfd.rl.agents.sac import SAC
from smartcfd.rl.env import ACTION_SCALE, HORIZON, OBSERVATION_DIM
from smartcfd.rl.env import is_better, mesh_metrics, observation, reward, run_case, transition
from smartcfd.rl.replay_buffer import Replay


def dataset():
    """Hold out camber families plus unobserved symmetric thicknesses."""
    rng = np.random.default_rng(593)
    test_families = ((2, 4), (3, 5), (4, 4), (6, 4))
    validation_families = ((1, 6), (2, 3), (5, 5))
    test_codes = ["0010", "0012", "0016", "0022"]
    test_codes += [f"{m}{p}{t:02d}" for m, p in test_families for t in (9, 12, 15, 20, 23)]
    validation_codes = ["0009", "0014", "0018", "0024"]
    validation_codes += [f"{m}{p}{t:02d}" for m, p in validation_families for t in (10, 13, 17, 21)]
    training_pool = [
        f"{m}{p}{t:02d}"
        for m in range(1, 7) for p in range(2, 8) for t in range(8, 25)
        if (m, p) not in test_families + validation_families
    ]
    symmetric = [f"00{t:02d}" for t in range(8, 25)
                 if f"00{t:02d}" not in test_codes + validation_codes]
    training_codes = symmetric + rng.choice(training_pool, size=135, replace=False).tolist()
    return tuple([naca(code) for code in codes]
                 for codes in (training_codes, validation_codes, test_codes))


def optimize_reference(case, rng):
    """Expensive search-generated demonstrations, computed on training cases only."""
    best = BASE_PARAMS.copy()
    best_quality = mesh_metrics(case, best)
    candidates = rng.uniform(-1, 1, (28, PARAM_DIM))
    for candidate in candidates:
        result = mesh_metrics(case, candidate)
        if is_better(result, best_quality):
            best, best_quality = candidate, result
    for radius in (0.55, 0.28, 0.12, 0.055):
        neighbors = np.vstack((best + radius * np.eye(PARAM_DIM), best - radius * np.eye(PARAM_DIM)))
        for candidate in np.clip(neighbors, -1, 1):
            result = mesh_metrics(case, candidate)
            if is_better(result, best_quality):
                best, best_quality = candidate, result
    return best.astype(np.float32)


def reference_examples(cases, rng, replay):
    obs, acts, references = [], [], []
    for index, case in enumerate(cases):
        target = optimize_reference(case, rng)
        references.append(target)
        for start in (BASE_PARAMS, rng.uniform(-1, 1, PARAM_DIM), target):
            parameters = np.asarray(start, dtype=np.float32)
            metrics = mesh_metrics(case, parameters)
            for step in range(HORIZON):
                old_obs = observation(case, parameters, metrics, step)
                action = np.clip((target - parameters) / ACTION_SCALE, -1, 1)
                new_parameters = transition(parameters, action)
                new_metrics = mesh_metrics(case, new_parameters)
                new_obs = observation(case, new_parameters, new_metrics, step + 1)
                replay.add(old_obs, action, reward(metrics, new_metrics), new_obs, step + 1 == HORIZON)
                obs.append(old_obs)
                acts.append(action)
                parameters, metrics = new_parameters, new_metrics
        if (index + 1) % 40 == 0:
            print(f"demonstrations {index + 1}/{len(cases)}", flush=True)
    return np.stack(obs), np.stack(acts), np.stack(references)


def validation(cases, model):
    metrics = [run_case(case, "sac", model)["metrics"] for case in cases]
    return float(np.mean([m["score"] for m in metrics])), float(
        np.mean([m["invalid_cells"] == 0 for m in metrics])
    )


def tuned_fixed(cases, references):
    """Choose one global setting on training cases, without per-test adaptation."""
    candidates = np.vstack((BASE_PARAMS, np.median(references, axis=0), references[::12]))
    def training_score(parameters):
        measurements = [mesh_metrics(case, parameters) for case in cases[::4]]
        return (np.mean([metrics["invalid_cells"] == 0 for metrics in measurements]),
                np.mean([metrics["score"] for metrics in measurements]))
    return max(candidates, key=training_score).tolist()


def train(steps: int, seed: int, output: Path) -> None:
    """Train into a new run directory and retain the best SAC validation checkpoint."""
    if steps < 100:
        raise ValueError("Training requires at least 100 environment steps")
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    train_cases, val_cases, test_cases = dataset()
    dim = OBSERVATION_DIM
    model = SAC(dim, PARAM_DIM, rng)
    replay = Replay(80_000, dim, PARAM_DIM)
    start = perf_counter()
    demonstrations, actions, references = reference_examples(train_cases, rng, replay)
    global_parameters = tuned_fixed(train_cases, references)
    for _ in range(1900):
        indices = rng.integers(len(actions), size=96)
        model.imitate(demonstrations[indices], actions[indices])
    model.save_actor(output / "bc_actor.npz")
    score, validity = validation(val_cases, model)
    history = [{"step": 0, "score": score, "valid_rate": validity, "train_return": None}]
    print(f"BC validation {score:.6f} validity {validity:.1%}", flush=True)

    # Warm up the twin critics using replay, then update the actor off policy.
    for _ in range(300):
        model.update(replay, update_actor=False)
    best_validation = -np.inf
    best_step = 0
    returns = []
    episode_step = 0
    checkpoint_every = min(1200, max(100, steps // 16))
    for step in range(1, steps + 1):
        if episode_step == 0:
            case = train_cases[rng.integers(len(train_cases))]
            parameters = BASE_PARAMS.copy()
            initial_metrics = mesh_metrics(case, parameters)
            metrics = initial_metrics
        old_obs = observation(case, parameters, metrics, episode_step)
        action = model.action(old_obs)
        parameters = transition(parameters, action)
        current = mesh_metrics(case, parameters)
        episode_step += 1
        terminal = episode_step == HORIZON
        new_obs = observation(case, parameters, current, episode_step)
        replay.add(old_obs, action, reward(metrics, current), new_obs, terminal)
        metrics = current
        if terminal:
            returns.append(reward(initial_metrics, current))
            episode_step = 0
        if step % 2 == 0:
            model.update(replay)
            if step <= steps // 2:
                indices = rng.integers(len(actions), size=64)
                model.imitate(demonstrations[indices], actions[indices], learning_rate=1e-4)
        if step % checkpoint_every == 0 or step == steps:
            score, validity = validation(val_cases, model)
            history.append({"step": step, "score": score, "valid_rate": validity,
                            "train_return": float(np.mean(returns[-80:])) if returns else None})
            print(f"SAC {step}/{steps} val={score:.6f} valid={validity:.1%} "
                  f"return={history[-1]['train_return']:+.4f}", flush=True)
            if score > best_validation:
                best_validation, best_step = score, step
                model.save_actor(output / "sac_actor.npz")
    now = perf_counter()
    config = {
        "seed": seed, "train_cases": len(train_cases), "val_cases": len(val_cases),
        "test_cases": len(test_cases), "splits": {
            key: [case.name for case in values]
            for key, values in zip(("train", "validation", "test"), (train_cases, val_cases, test_cases))
        },
        "steps": steps, "best_step": best_step, "horizon": HORIZON,
        "parameter_names": list(PARAM_NAMES), "mesh_shape": list(generate(train_cases[0], BASE_PARAMS).shape),
        "train_seconds": now - start, "observation_dim": dim, "alpha": model.alpha,
        "fixed_parameters": global_parameters,
    }
    (output / "training.json").write_text(
        json.dumps({"meta": config, "history": history}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
