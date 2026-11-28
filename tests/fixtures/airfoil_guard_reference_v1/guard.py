"""Deterministic, conservative two-transition feasibility guard.

Only proofs remove actions. A depth/budget cutoff is unknown, never failure.
The primitive preview remains the rule scorer's information source; this guard
adds no learned score and never prioritizes a completion witness over unknown.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import copy
import hashlib
import json
from numbers import Integral
import time

import numpy as np

from .env import AirfoilBlockEnv

GUARD_VERSION = "airfoil_conservative_guard_v1"
GUARDED_ENVIRONMENT_VERSION = "airfoil_c_decomposition_guard_v1"


@dataclass(frozen=True)
class GuardConfig:
    depth: int = 2
    second_layer_budget: int = 128
    cache_capacity: int = 128

    def __post_init__(self):
        if type(self.depth) is not int or self.depth != 2:
            raise ValueError("The guard fixes depth=2, including the root action")
        for name in ("second_layer_budget", "cache_capacity"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


def budget_lower_bound(graph, *, steps, config):
    """Necessary (not sufficient) completion bounds over the complete graph."""
    faces = graph["faces"] if isinstance(graph, dict) else list(graph.faces.values())
    degrees = [len(f["vertices"] if isinstance(f, dict) else f.vertices) for f in faces]
    # The proof relies on the primitive environment's degree >= 4 invariant.
    if any(n < 4 for n in degrees):
        raise ValueError("Quadrangulation bound requires every face to have degree >= 4")
    defect = sum(n - 4 for n in degrees)
    cuts = (defect + 1) // 2
    final_blocks = len(faces) + cuts
    reason = None
    if cuts > config.max_steps - steps:
        reason = "remaining_step_lower_bound"
    elif final_blocks > config.max_blocks:
        reason = "final_block_lower_bound"
    elif 4 * final_blocks > config.max_half_edges:
        reason = "final_halfedge_lower_bound"
    return {"defect": defect, "minimum_cuts": cuts, "minimum_steps": cuts,
            "minimum_final_blocks": final_blocks, "minimum_final_halfedges": 4 * final_blocks,
            "remaining_steps": config.max_steps - steps, "reason": reason}


class ConservativeGuard:
    def __init__(self, config=None):
        self.config = config if isinstance(config, GuardConfig) else GuardConfig(**(config or {}))
        self._cache = OrderedDict()

    def _key(self, env):
        # Include the sample, full graph/IDs/metadata, queue and remaining budget.
        identity = {"sample": env.sample, "config": asdict(env.config), "graph": env.graph.to_dict(),
                    "queue": env.queue, "steps": env.steps, "initial": env._initial.to_dict(),
                    "version": GUARD_VERSION, "guard": asdict(self.config)}
        return hashlib.sha256(json.dumps(identity, sort_keys=True, default=lambda a: a.tolist(),
                                         separators=(",", ":")).encode()).hexdigest()

    def analyze(self, env):
        if env.graph is None or env.terminated:
            raise RuntimeError("Guard analysis requires a live reset environment")
        started = time.perf_counter()
        before = env.work_statistics
        key = self._key(env)
        if key in self._cache:
            self._cache.move_to_end(key)
            result = copy.deepcopy(self._cache[key])
            # Logical requests describe the same deterministic search. Physical
            # work and cache reuse are reported separately, without double count.
            result["metrics"].update(guard_cache_hit=1, root_preview_calls=0,
                second_layer_preview_calls=0, branch_materializations=0,
                **{name: 0 for name in before}, elapsed_seconds=time.perf_counter()-started)
            return result

        count = len(env._candidates)
        rows, roots, branches, positions = [], {}, {}, {}
        metrics = {"guard_cache_hit": 0, "root_previews": count, "root_preview_calls": count,
                   "second_layer_transitions": 0, "second_layer_preview_calls": 0,
                   "branch_materializations": 0}
        # Every root receives exact primitive preview, even when its cheap bound
        # already proves failure. No candidate ordering depends on a network.
        for action in range(count):
            preview = AirfoilBlockEnv.preview(env, action)
            bound = budget_lower_bound(preview["next_graph"], steps=env.steps+1, config=env.config)
            row = {"action": action, "status": "unknown", "reason": "depth_or_budget_limit",
                   "lower_bound": bound, "second_layer_checked": 0, "second_layer_failures": 0,
                   "second_layer_total": None}
            if preview["success"]:
                row.update(status="completion_witness", reason="root_completed")
            elif preview["terminated"]:
                row.update(status="proven_dead_end", reason="root_terminal:"+preview["reason"])
            elif bound["reason"]:
                row.update(status="proven_dead_end", reason=bound["reason"])
            else:
                roots[action] = preview
                positions[action] = 0
            rows.append(row)

        active = list(roots)
        while active and metrics["second_layer_transitions"] < self.config.second_layer_budget:
            following = []
            for action in active:
                if metrics["second_layer_transitions"] >= self.config.second_layer_budget:
                    break
                if action not in branches:
                    branch = env.fork()
                    # Materialize the complete root successor, not preview's
                    # existential next-action witness. Primitive step is reused.
                    branch.step(action)
                    branches[action] = branch
                    metrics["branch_materializations"] += 1
                    rows[action]["second_layer_total"] = len(branch._candidates)
                branch = branches[action]
                position = positions[action]
                preview = AirfoilBlockEnv.preview(branch, position)
                metrics["second_layer_transitions"] += 1
                metrics["second_layer_preview_calls"] += 1
                positions[action] += 1
                row = rows[action]
                row["second_layer_checked"] += 1
                bound = budget_lower_bound(preview["next_graph"], steps=env.steps+2, config=env.config)
                if preview["success"]:
                    row.update(status="completion_witness", reason="second_layer_completed")
                    continue
                if preview["terminated"] or bound["reason"]:
                    row["second_layer_failures"] += 1
                if positions[action] == len(branch._candidates):
                    if row["second_layer_failures"] == row["second_layer_checked"]:
                        row.update(status="proven_dead_end", reason="all_second_layer_branches_proven_failed")
                else:
                    following.append(action)
            active = following

        mask = np.zeros_like(env.action_mask)
        mask[:count] = [row["status"] != "proven_dead_end" for row in rows]
        metrics.update({name: env.work_statistics[name]-before[name] for name in before})
        metrics.update(elapsed_seconds=time.perf_counter()-started,
                       removed_actions=sum(row["status"] == "proven_dead_end" for row in rows),
                       unknown_actions=sum(row["status"] == "unknown" for row in rows),
                       completion_witnesses=sum(row["status"] == "completion_witness" for row in rows))
        result = {"version": GUARD_VERSION, "effective_mask": mask, "actions": rows, "metrics": metrics}
        if self.config.cache_capacity:
            self._cache[key] = copy.deepcopy(result)
            while len(self._cache) > self.config.cache_capacity:
                self._cache.popitem(last=False)
        return result


class GuardedAirfoilEnv(AirfoilBlockEnv):
    """Same primitive actions/reward coefficients with proven failure termination.

    Inherited preview intentionally previews the primitive transition. Rule
    scores use that same primitive information in guarded and unguarded arms.
    Only reset/step apply the guard; speculative forks are always unguarded.
    """
    def __init__(self, sample=None, config=None, guard_config=None):
        super().__init__(sample=sample, config=config)
        self.guard = ConservativeGuard(guard_config)
        self.guard_statistics = {}
        self._guard_report = None

    @property
    def environment_context(self):
        return {**super().environment_context, "environment_version": GUARDED_ENVIRONMENT_VERSION,
                "guard_version": GUARD_VERSION, "guard_config": asdict(self.guard.config)}

    def _info(self, **extra):
        result = super()._info(**extra)
        if self._guard_report is not None:
            result["guard"] = {"version": GUARD_VERSION, **copy.deepcopy(self._guard_report["metrics"]),
                               "actions": copy.deepcopy(self._guard_report["actions"])}
        return result

    def _guard_live_state(self):
        self._guard_report = self.guard.analyze(self)
        self._mask = self._guard_report["effective_mask"].copy()
        self.guard_statistics["decisions"] = self.guard_statistics.get("decisions", 0)+1
        for name, value in self._guard_report["metrics"].items():
            self.guard_statistics[name] = self.guard_statistics.get(name, 0)+value
        if self._mask.any():
            return False
        self.terminated, self.success, self._reason = True, False, "proven_unfinishable"
        self._candidates = []
        self.guard_statistics["all_pruned"] = self.guard_statistics.get("all_pruned", 0)+1
        return True

    def reset(self, *, seed=None, options=None):
        self._guard_report = None
        super().reset(seed=seed, options=options)
        if not self.terminated:
            self._guard_live_state()
        return self._observation(), self._info()

    def step(self, action):
        if self.terminated:
            return super().step(action)
        valid = isinstance(action, Integral) and not isinstance(action, (bool, np.bool_))
        valid = valid and 0 <= int(action) < len(self._mask) and bool(self._mask[int(action)])
        # An in-range but pruned index must fail exactly like an invalid action.
        self._guard_report = None
        obs, reward, terminal, truncated, info = super().step(action if valid else -1)
        if terminal:
            return obs, reward, terminal, truncated, info
        if self._guard_live_state():
            components = dict(info["reward_components"])
            components.update(terminal_score=self.config.failure_reward,
                              base_reward=-self.config.step_cost+self.config.failure_reward,
                              potential_after=0., shaping=-components["potential_before"])
            reward = components["base_reward"]+components["shaping"]
            components["total"] = reward
            self._history[-1]["reason"] = self._reason
            info.update(base_reward=components["base_reward"], shaping=components["shaping"],
                        reward_components=components)
        extras = {key: info[key] for key in ("invalid_action", "action", "base_reward", "shaping", "reward_components")}
        return self._observation(), float(reward), bool(self.terminated), False, self._info(**extras)
