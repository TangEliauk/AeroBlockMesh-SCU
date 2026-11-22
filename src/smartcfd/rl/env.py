"""Finite-horizon continuous parameter adjustment for a fixed four-block O-grid."""

from time import perf_counter

import numpy as np

from smartcfd.evaluation.mesh_quality import cell_quality, quality
from smartcfd.geometry.airfoil import case_features
from smartcfd.meshing.structured import BASE_PARAMS, PARAM_DIM, generate


SEED = 1729
HORIZON = 8
ACTION_SCALE = 0.65
OBSERVATION_DIM = 3 + PARAM_DIM + 6


def mesh_metrics(case, parameters):
    return quality(generate(case, parameters))


def observation(case, parameters, metrics, step):
    metrics_vector = [
        metrics["score"], metrics["min_jacobian"], metrics["p05_jacobian"],
        metrics["mean_skew"], metrics["invalid_cells"] / metrics["cells"], step / HORIZON,
    ]
    return np.concatenate((case_features(case), parameters, metrics_vector)).astype(np.float32)


def transition(parameters, action):
    return np.clip(parameters + ACTION_SCALE * action, -1, 1).astype(np.float32)


def reward(previous, current):
    # The difference of independently measured quality supplies a dense reward.
    return 12.0 * (current["score"] - previous["score"])


def is_better(candidate, incumbent):
    # Always retain a valid mesh in preference to an invalid mesh.
    return (candidate["invalid_cells"] == 0, candidate["score"]) > (
        incumbent["invalid_cells"] == 0, incumbent["score"]
    )


def frame(case, step, parameters, metrics, improvement, accepted, best_score):
    mesh = generate(case, parameters)
    return {
        "step": step, "params": parameters.tolist(), "reward": improvement, "metrics": metrics,
        "accepted": accepted, "best_score": best_score,
        "mesh": mesh.round(6).tolist(), "cell_quality": cell_quality(mesh).round(5).tolist(),
    }


def run_case(case, method, policy=None, *, trace=False, fixed_parameters=None, max_steps=HORIZON):
    start = perf_counter()
    rng = np.random.default_rng(SEED + int(case.name))
    parameters = BASE_PARAMS.copy()
    if method == "tuned":
        parameters = np.asarray(fixed_parameters, dtype=np.float32)
    current = mesh_metrics(case, parameters)
    best_parameters, best = parameters.copy(), current
    captures = [(0, parameters, current, 0.0, True, current["score"])]
    best_step = 0
    budget = 0 if method in ("fixed", "tuned") else max_steps
    coordinate_order = (1, 0, 2, 3, 4)
    coordinate_origins = np.empty(PARAM_DIM, dtype=np.float32)
    for step in range(budget):
        if method in ("sac", "bc"):
            action = policy.action(observation(case, parameters, current, step), deterministic=True)
            proposal = transition(parameters, action)
        elif method == "random":
            proposal = rng.uniform(-1, 1, PARAM_DIM).astype(np.float32)
        elif method == "greedy":
            # Cover all five controls, then try the opposite sign on the first three.
            coordinate = coordinate_order[step % PARAM_DIM]
            proposal = best_parameters.copy()
            if step < PARAM_DIM:
                coordinate_origins[coordinate] = best_parameters[coordinate]
            direction = 1 if step < PARAM_DIM else -1
            proposal[coordinate] = np.clip(coordinate_origins[coordinate] + 0.65 * direction, -1, 1)
        else:
            raise ValueError(f"Unknown method {method}")
        metrics = mesh_metrics(case, proposal)
        improvement = reward(current, metrics)
        accepted = is_better(metrics, best)
        if accepted:
            best_parameters, best, best_step = proposal.copy(), metrics, step + 1
        captures.append((step + 1, proposal, metrics, improvement, accepted, best["score"]))
        parameters, current = proposal, metrics
    result = {
        "metrics": best, "elapsed_ms": (perf_counter() - start) * 1000,
        "evaluations": 1 + budget, "best_step": best_step, "parameters": best_parameters.tolist(),
    }
    if trace:
        result["frames"] = [frame(case, *capture) for capture in captures]
    return result
