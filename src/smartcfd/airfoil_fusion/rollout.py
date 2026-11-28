"""Repeatable rule, learned and demonstration rollouts on frozen geometries."""
from __future__ import annotations

from collections import OrderedDict
import random
import time

import numpy as np

from smartcfd.airfoil_decomposition.baselines import refined_c_action
from smartcfd.airfoil_decomposition.env import AirfoilBlockEnv
from smartcfd.airfoil_decomposition.experiment import fixed_action, rule_action
from smartcfd.core.types import Transition
from smartcfd.partition.topology import fingerprint

from .env import FusionEnv, GuardedFusionEnv
from .mesh import BUDGETS, evaluate_mesh_budget


class Environments:
    """Small per-process pool of validated geometry and preview caches."""

    def __init__(self, config, arm, *, capacity=8):
        self.config, self.arm, self.capacity = config, arm, capacity
        self.cache = OrderedDict()

    def get(self, sample):
        key = sample["sample_id"]
        if key not in self.cache:
            if self.arm == "E4":
                env = AirfoilBlockEnv(sample, self.config)
            elif self.arm == "E8":
                env = GuardedFusionEnv(sample, self.config)
            else:
                env = FusionEnv(sample, self.config, mesh_reward=self.arm != "E4a")
            self.cache[key] = env
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return self.cache[key]


def action_key(item):
    if item["kind"] == "stop":
        return ("stop",)
    a, b = item["source"], item["target"]
    start, finish = (a, b) if a < b else (b, a)
    if item["kind"] == "insert_edge_point":
        fraction = item["fraction"] if a < b else 1 - item["fraction"]
        return (item["kind"], start, finish, round(fraction, 8))
    return (item["kind"], item["face_id"], start, finish)


def resolve(env, saved):
    key = action_key(saved)
    rows = [i for i, candidate in enumerate(env.candidates) if action_key(candidate) == key]
    if len(rows) != 1 or not env.action_mask[rows[0]]:
        raise ValueError(f"Replay action absent or masked: {key}")
    candidate = env.candidates[rows[0]]
    if saved["kind"] == "insert_edge_point" and not np.allclose(
        candidate["point_uv"], saved["point_uv"], rtol=0, atol=1e-12
    ):
        raise ValueError(f"Replay insertion point changed: {key}")
    return rows[0]


def stop_action(env):
    return next((i for i, item in enumerate(env.candidates)
                 if item["kind"] == "stop" and env.action_mask[i]), None)


def _choose(env, observation, method, rng, agent):
    if method in ("fixed", "refined"):
        stop = stop_action(env)
        if stop is not None:
            return stop
        return fixed_action(env) if method == "fixed" else refined_c_action(env)
    if method == "greedy":
        return env.greedy_action() if isinstance(env, FusionEnv) else rule_action(env)
    if method == "random":
        return rng.choice(np.flatnonzero(observation["mask"]).tolist())
    if method == "learned":
        return agent.act(observation, deterministic=True)
    raise ValueError(f"Unknown rollout method: {method}")


def episode(pool, sample, method, *, seed=42, agent=None, prechecked=None,
            capture=False, mesh=True, trace=False):
    started = time.perf_counter()
    base = {"sample_id": sample["sample_id"], "group_id": sample["group_id"], "seed": seed,
            "success": False, "edits": 0, "decisions": 0, "initializable": False}
    if prechecked is not None and not prechecked["valid"]:
        return {**base, "reason": prechecked["reason"], "seconds": time.perf_counter()-started}, []
    env = pool.get(sample)
    before = env.work_statistics
    guard_before = dict(getattr(env, "guard_statistics", {}))
    observation, info = env.reset(seed=seed)
    initializable = not env.terminated
    initial_graph = env.graph.to_dict() if trace else None
    rng, steps, transitions = random.Random(seed), [], []
    reason = info["reason"]
    while not env.terminated:
        action = _choose(env, observation, method, rng, agent)
        if action is None:
            reason = "rule_action_unavailable"
            break
        action = int(action)
        descriptor = env.candidates[action]
        next_obs, reward, terminal, truncated, info = env.step(action)
        if capture:
            transitions.append(Transition(observation, action, reward, next_obs, terminal, truncated))
        record = {"index": action, "candidate": descriptor, "reward": reward,
                  "reward_components": info.get("reward_components"), "reason": info["reason"]}
        if trace:
            record["graph"] = env.graph.to_dict()
        steps.append(record)
        observation, reason = next_obs, info["reason"]
    result = {**base, "success": env.success, "initializable": initializable, "reason": reason,
              "edits": sum(s["candidate"]["kind"] != "stop" for s in steps),
              "decisions": len(steps), "actions": steps, "quality": info["quality"],
              "work": {k: v-before[k] for k, v in env.work_statistics.items()},
              "guard_work": {k: v-guard_before.get(k, 0) for k, v in getattr(env, "guard_statistics", {}).items()}}
    if trace:
        result["initial_graph"] = initial_graph
    result["graph"] = env.graph.to_dict()
    if env.success:
        result["outer_blocks"] = sum(not face.protected for face in env.graph.faces.values())
        result["topology"] = fingerprint(env.graph)
        if mesh:
            result["mesh"] = {str(b): evaluate_mesh_budget(env.geometry.domain, env.graph, b)[0] for b in BUDGETS}
    result["seconds"] = time.perf_counter()-started
    return result, transitions


def replay(pool, sample, descriptors, *, seed=42, confirm=True):
    """Recompute states, rewards and masks instead of trusting stored actions or rewards."""
    env = pool.get(sample)
    observation, info = env.reset(seed=seed)
    if env.terminated:
        raise ValueError(f"Replay initial state already terminal: {info['reason']}")
    transitions = []
    for saved in descriptors:
        action = resolve(env, saved)
        next_obs, reward, done, truncated, info = env.step(action)
        transitions.append(Transition(observation, action, reward, next_obs, done, truncated))
        observation = next_obs
    if confirm and not env.terminated:
        action = stop_action(env)
        if action is None:
            raise ValueError("Replay did not reach a legal STOP")
        next_obs, reward, done, truncated, info = env.step(action)
        transitions.append(Transition(observation, action, reward, next_obs, done, truncated))
        observation = next_obs
    if confirm and not env.success:
        raise ValueError(f"Replay failed completion: {info['reason']}")
    return env, observation, transitions


def summarize(cases, max_edits=32):
    n = len(cases)
    if not n:
        raise ValueError("Evaluation requires the full nonempty manifest")
    success = [row for row in cases if row["success"]]
    mesh_rows = [row.get("mesh", {}).get(str(BUDGETS[0]), {}) for row in success]
    scores = [row["score"] for row in mesh_rows if row.get("valid") and row.get("budget_match")]
    return {"total": n, "successful": len(success), "success_rate": len(success)/n,
            "initializable": sum(row["initializable"] for row in cases),
            "mesh_successful": sum(row.get("valid", False) for row in mesh_rows),
            "budget_mesh_successful": len(scores), "mesh_score": float(np.mean(scores)) if scores else None,
            "mean_penalized_edits": sum(row["edits"] if row["success"] else max_edits for row in cases)/n,
            "total_seconds": sum(row["seconds"] for row in cases)}


def selection_key(row):
    return (row["successful"], row["mesh_successful"],
            row["mesh_score"] if row["mesh_score"] is not None else -np.inf,
            -row["mean_penalized_edits"], -row["step"])
