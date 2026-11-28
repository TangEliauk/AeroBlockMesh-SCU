"""Whole-domain candidate enumeration and explicit quadrangulation completion."""
from __future__ import annotations

from collections import OrderedDict
import copy

from gymnasium import spaces
import numpy as np

from smartcfd.airfoil_decomposition.env import AirfoilBlockEnv, _Candidate, topology_potential
from smartcfd.airfoil_decomposition.guard import ConservativeGuard, GuardedAirfoilEnv
from smartcfd.core.types import BlockGraph
from smartcfd.partition.topology import all_quadrilateral

VERSION = "airfoil_global_candidates_v1"


def profile_features(mapping):
    """CH-SAC v2 chordwise geometry feature ordering for a sharp trailing edge."""
    def branch(points):
        x, rows = np.unique(points[:, 0], return_index=True)
        return x, points[rows, 1]

    lower, upper = branch(mapping.lower[1]), branch(mapping.upper[1])

    def ordinates(xs):
        return np.interp(xs, upper[0], upper[1]), np.interp(xs, lower[0], lower[1])

    x = np.linspace(0, 1, 101)
    u, l = ordinates(x)
    thickness, camber = u - l, (u + l) / 2
    it, ic = np.argmax(thickness), np.argmax(np.abs(camber))
    u, l = ordinates([.05, .95, 1.])
    tu = np.clip(np.arctan2(u[1] - u[2], -.05) / .6, -1.5, 1.5)
    tl = np.clip(np.arctan2(l[1] - l[2], -.05) / .6, -1.5, 1.5)
    us, ls = ordinates([.15, .30, .60, .85])
    uc, lc = ordinates([.20, .50, .80])
    values = [thickness[it] / .25, x[it], camber[ic] / .10, x[ic],
              (u[0] - l[0]) / .10, tu, tl, np.clip(tu - tl, 0, 2),
              (tu + tl) / 2, 0., abs(u[2] - l[2]) / .02,
              *((us - ls) / .25), *((uc + lc) / .20)]
    return np.clip(np.asarray(values, dtype=np.float32), -3, 3)


class FusionEnv(AirfoilBlockEnv):
    """A's physical edits, on every mutable face, with a separate STOP action."""

    def __init__(self, sample=None, config=None, *, mesh_reward=True):
        super().__init__(sample, config)
        self.mesh_reward = bool(mesh_reward)
        c = self.config
        # Every edge can offer three insertions. A degree-n face offers at most
        # n*(n-3)/2 chords, even before legality and shared-owner deduplication.
        capacity = c.max_half_edges * (max(0, c.max_face_vertices - 3) + 6) // 2 + 1
        self._mask = np.zeros(capacity, dtype=bool)
        self.action_space = spaces.Discrete(capacity)
        for name in ("candidates", "candidate_nodes"):
            old = self.observation_space[name]
            self.observation_space[name] = spaces.Sequence(
                spaces.Box(old.low[0], old.high[0], dtype=old.dtype), stack=True
            )
        self.observation_space["mask"] = spaces.Sequence(
            spaces.Box(0, 1, (), dtype=np.bool_), stack=True
        )
        self.observation_space["geometry"] = spaces.Box(-3, 3, (18,), dtype=np.float32)
        self._geometry_features = None
        self._mesh_reward_cache = OrderedDict()

    @property
    def environment_context(self):
        return {"environment_version": VERSION, "action_version": VERSION,
                "observation_version": VERSION, "config": super().environment_context["config"],
                "mesh_reward": self.mesh_reward}

    def reset(self, *, seed=None, options=None):
        if self.geometry is not None and (options or {}).get("sample", self.sample) != self.sample:
            self._mesh_reward_cache = OrderedDict()
        return super().reset(seed=seed, options=options)

    def fork(self):
        child = super().fork()
        child.__class__ = FusionEnv
        return child

    def _next_queue(self, graph, queue, *, cut=False, new_face=None):
        return tuple(fid for fid, face in sorted(graph.faces.items()) if len(face.vertices) != 4)

    def _candidate_slots(self):
        return len(self._candidates)

    def _observation(self):
        obs = super()._observation()
        if self._geometry_features is None or self._geometry_features[0] is not self.geometry:
            self._geometry_features = (self.geometry, profile_features(self.geometry.mapping))
        obs["geometry"] = self._geometry_features[1].copy()
        obs["features"][:, 24] = (obs["next"] >= 0) & (obs["features"][:, 13] == 0)
        obs["features"][:, 26] = 0
        obs["candidates"][:, 12] = 0
        obs["global"][7] = max((len(self.graph.faces[fid].vertices) for fid in self._queue), default=0) / self.config.max_face_vertices
        return obs

    def _info(self, **extra):
        result = super()._info(**extra)
        result["edits"] = sum(item["kind"] != "stop" for item in self._history)
        result["scope"] = "C-domain topology edits with explicit STOP and separate mesh validation"
        return result

    def _stop_candidate(self, graph):
        if all_quadrilateral(graph):
            validation = self._validate(graph, refine=True)
            if validation["valid"]:
                return _Candidate({"kind": "stop", "face_id": None}, (-1, -1), graph, validation, ())
        return None

    def _iter_global(self, graph, *, edits_available=True, rejected=None):
        stop = self._stop_candidate(graph)
        if stop is not None:
            yield stop
        if edits_available:
            for fid, face in sorted(graph.faces.items()):
                if not face.protected:
                    yield from super()._iter_candidates(
                        graph, (fid,), rejected=rejected, unique_insertions=True
                    )

    def _enumerate_candidates(self, graph, queue, *, stop_after_first=False):
        rejected = []
        candidates = []
        for candidate in self._iter_global(graph, rejected=rejected):
            candidates.append(candidate)
            if stop_after_first:
                break
        if len(candidates) > self.action_space.n:
            raise RuntimeError("Complete candidate list exceeded the graph-derived capacity")
        budgets = sorted({reason for reason in rejected if "budget" in reason})
        failure = budgets[0] if budgets and len(budgets) == len(set(rejected)) else "no_legal_actions"
        return candidates, None if candidates else failure

    def _reward(self, next_graph, *, terminal, success, validation):
        reward, parts = super()._reward(next_graph, terminal=terminal, success=success, validation=validation)
        if success:
            # STOP confirms the current topology and spends no edit budget.
            parts["step_cost"] = 0.
            adjustment = self.config.step_cost
            if self.mesh_reward:
                from .mesh import reward_mesh_score

                key = self._candidate_cache_key(next_graph, ())
                if key not in self._mesh_reward_cache:
                    self._mesh_reward_cache[key] = reward_mesh_score(self.geometry.domain, next_graph)
                    if len(self._mesh_reward_cache) > 64:
                        self._mesh_reward_cache.popitem(last=False)
                mesh_score = self._mesh_reward_cache[key]
                parts["mesh_quality"] = mesh_score
                delta = (mesh_score - validation["quality"].get("q", 0.)) / 2
                parts["terminal_score"] += delta
                adjustment += delta
            parts["base_reward"] += adjustment
            reward += adjustment
            parts["total"] = reward
        return reward, parts

    def _outcome(self, candidate, *, preview=False):
        graph, queue = candidate.graph, list(candidate.queue)
        validation = candidate.validation
        success = candidate.description["kind"] == "stop"
        failure, following = None, []
        if not success:
            if self.steps + 1 >= self.config.max_steps:
                stop = self._stop_candidate(graph)
                following = [stop] if stop is not None else []
                failure = None if following else "step_budget_exhausted"
            else:
                key = self._candidate_cache_key(graph, queue)
                if preview and key not in self._enumeration_cache:
                    following, failure = self._enumerate_candidates(graph, queue, stop_after_first=True)
                else:
                    following, failure = self._build_candidates(graph, queue)
        terminal = success or failure is not None
        reward, parts = self._reward(graph, terminal=terminal, success=success, validation=validation)
        return {"graph": graph, "queue": queue, "validation": validation,
                "success": success, "terminated": terminal, "attempt_validation": None,
                "reason": "completed" if success else failure or "graph_edit",
                "candidates": following, "reward": reward, "reward_components": parts}

    def greedy_action(self):
        """Same immediate-reward/Q/index ordering as full previews."""
        optimistic = []
        for index, candidate in enumerate(self._candidates):
            if not self._mask[index]:
                continue
            stop = candidate.description["kind"] == "stop"
            assumed, _ = self._reward(candidate.graph, terminal=stop, success=stop,
                                      validation=candidate.validation)
            # An unsuccessful terminal edit loses failure_reward. Removing
            # nonterminal shaping can add at most gamma because Phi is in [-1,0].
            # Preserve a valid upper bound even for custom small failure penalties.
            terminal, _ = self._reward(candidate.graph, terminal=True, success=stop,
                                       validation=candidate.validation) if not stop else (assumed, None)
            quality = float(candidate.validation["quality"].get("q", 0.))
            optimistic.append((max(assumed, terminal), quality, -index))
        best = None
        for upper in sorted(optimistic, reverse=True):
            if best is not None and upper <= best:
                break
            result = self._outcome(self._candidates[-upper[2]], preview=True)
            score = (result["reward"], float(result["validation"]["quality"].get("q", 0.)), upper[2])
            if best is None or score > best:
                best = score
        return None if best is None else -best[2]


class _FusionBranch:
    """Lazily materialize second-layer candidates for an already live preview."""

    def __init__(self, parent, action, preview):
        self.env = parent.fork()
        self.env.graph = BlockGraph.from_dict(preview["next_graph"])
        self.env._queue = list(preview["queue"])
        self.env._validation = copy.deepcopy(parent._candidates[action].validation)
        self.env.steps += 1
        self.env._candidates = []
        self.known_total = None
        self.iterator = self.env._iter_global(self.env.graph, edits_available=self.env.steps < self.env.config.max_steps)

    def ensure(self, position):
        while position >= len(self.env._candidates):
            candidate = next(self.iterator, None)
            if candidate is None:
                self.known_total = len(self.env._candidates)
                return False
            self.env._candidates.append(candidate)
        return True


class FusionGuard(ConservativeGuard):
    def _key(self, env):
        return (VERSION, env.mesh_reward, super()._key(env))

    def _branch(self, env, action, preview):
        return _FusionBranch(env, action, preview)


class GuardedFusionEnv(GuardedAirfoilEnv, FusionEnv):
    """The proven-mask guard around the same primitive topology transitions."""

    def __init__(self, sample=None, config=None, guard_config=None):
        super().__init__(sample, config, guard_config)
        self.guard = FusionGuard(guard_config)

    @property
    def environment_context(self):
        context = super().environment_context
        context["environment_version"] = VERSION + "_guarded"
        return context
