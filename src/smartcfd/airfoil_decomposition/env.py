"""Transactional C-domain block decomposition with physically validated actions.

This environment edits logical curved blocks. It creates no interior mesh nodes.
Candidate indices are local to the current state and must never be replayed with
a different observation. Protected faces and their interfaces remain immutable.
"""
from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import asdict, dataclass
import math
from numbers import Integral

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from smartcfd.core.types import FaceBlock
from smartcfd.partition.constraints import area, on_segment, valid_chord, validate_graph
from smartcfd.partition.topology import edge_incidence, half_edges, initial_graph, protected_edges
from .config import (ACTION_VERSION, ENVIRONMENT_VERSION, OBSERVATION_VERSION,
                     AirfoilDecompositionConfig)
from .geometry import CurveGeometry

HALF_EDGE_CAPACITY = 256
CANDIDATE_CAPACITY = 1024
INSERT_FRACTIONS = (1 / 3, 1 / 2, 2 / 3)

FEATURE_NAMES = (
    "source_u", "source_v", "target_u", "target_v", "source_x_log", "source_y_log",
    "target_x_log", "target_y_log", "physical_tangent_x", "physical_tangent_y", "face_degree",
    "face_uv_area", "edge_uv_length", "protected_face", "protected_edge", "shared_edge",
    "wall", "wake", "farfield", "parameter_side", "outward_normal_x", "outward_normal_y",
    "protected_interface", "real_halfedge", "active_face", "queued_face", "queue_order",
    "completed_quad", "physical_arc_log", "remaining_steps", "remaining_blocks", "face_order")
GLOBAL_NAMES = ("step_fraction", "face_fraction", "halfedge_fraction", "vertex_fraction",
                "defect_fraction", "queue_fraction", "outer_face_fraction", "active_degree_fraction",
                "profile_thickness", "profile_mean_y", "farfield_log", "stretch_fraction")
CANDIDATE_NAMES = ("chord", "edge_insertion", "source_u", "source_v", "target_u", "target_v",
                   "arc_fraction", "delta_u", "delta_v", "uv_length", "physical_arc_log",
                   "source_degree_fraction", "source_queue_order", "owner_fraction",
                   "predicted_geometry_q", "predicted_outer_face_fraction")


def synthetic_profile():
    """Explicit non-NACA fixture, never presented as a public training sample."""
    x = (1 - np.cos(np.linspace(0, np.pi, 81))) / 2
    y = .18 * np.sqrt(x) * (1 - x)
    points = np.vstack((np.column_stack((x, -y)), np.column_stack((x[-2:0:-1], y[-2:0:-1]))))
    return {"sample_id": "synthetic_non_naca", "polygon": points.tolist(), "explicitly_closed": True,
            "source": "analytic_non_naca_environment_fixture"}


def topology_potential(graph):
    defect = sum(max(len(face.vertices) - 4, 0) for face in graph.faces.values())
    return -min(1., defect / 32.)


@dataclass(frozen=True)
class _Candidate:
    description: dict
    nodes: tuple[int, int]
    graph: object
    validation: dict
    queue: tuple[int, ...]


class AirfoilBlockEnv(gym.Env):
    """FIFO nonquadrilateral processing with full-graph, fixed-capacity observations."""
    metadata = {"render_modes": []}

    def __init__(self, sample=None, config=None):
        self.config = config if isinstance(config, AirfoilDecompositionConfig) else AirfoilDecompositionConfig(**(config or {}))
        self.sample = copy.deepcopy(sample if sample is not None else synthetic_profile())
        self.action_space = spaces.Discrete(CANDIDATE_CAPACITY)
        self.observation_space = spaces.Dict({
            "features": spaces.Box(-np.inf, np.inf, (HALF_EDGE_CAPACITY, 32), dtype=np.float32),
            "next": spaces.Box(-1, HALF_EDGE_CAPACITY - 1, (HALF_EDGE_CAPACITY,), dtype=np.int64),
            "previous": spaces.Box(-1, HALF_EDGE_CAPACITY - 1, (HALF_EDGE_CAPACITY,), dtype=np.int64),
            "twin": spaces.Box(-1, HALF_EDGE_CAPACITY - 1, (HALF_EDGE_CAPACITY,), dtype=np.int64),
            "global": spaces.Box(-np.inf, np.inf, (12,), dtype=np.float32),
            "candidates": spaces.Box(-np.inf, np.inf, (CANDIDATE_CAPACITY, 16), dtype=np.float32),
            "candidate_nodes": spaces.Box(-1, HALF_EDGE_CAPACITY - 1, (CANDIDATE_CAPACITY, 2), dtype=np.int64),
            "mask": spaces.Box(0, 1, (CANDIDATE_CAPACITY,), dtype=np.bool_)})
        self.geometry = self.graph = self._initial = None
        self.steps = 0
        self.success = self.terminated = False
        self._queue, self._candidates, self._history = [], [], []
        self._mask = np.zeros(CANDIDATE_CAPACITY, dtype=bool)
        self._validation = {}
        self._attempt_validation = None
        self._reason = "not_reset"
        self._enumeration_cache = OrderedDict()
        # Lifetime work counters include speculative forks; they are diagnostics,
        # never part of the environment state or observations.
        self._work_counters = {"geometry_validation_calls": 0, "topology_validation_calls": 0,
                               "candidate_cache_hits": 0, "candidate_cache_misses": 0}

    @property
    def work_statistics(self):
        return dict(self._work_counters)

    def fork(self):
        """Independent primitive state, sharing only pure caches/work counters.

        A guarded subclass deliberately forks as the primitive environment: tree
        simulation must not recursively invoke its guard. Candidate graphs are
        private immutable inputs; step clones them before committing.
        """
        if self.graph is None:
            raise RuntimeError("Call reset before fork")
        child = object.__new__(AirfoilBlockEnv)
        child.__dict__ = self.__dict__.copy()
        for name in ("sample", "_queue", "_history", "_validation", "_attempt_validation", "_np_random",
                     "action_space", "observation_space"):
            if name in self.__dict__:
                child.__dict__[name] = copy.deepcopy(self.__dict__[name])
        # A primitive fork has no guard-owned mutable diagnostic/configuration
        # objects; in particular speculative work cannot edit parent reports.
        for name in ("guard", "guard_statistics", "_guard_report"):
            child.__dict__.pop(name, None)
        child.graph, child._initial = self.graph.clone(), self._initial.clone()
        child._candidates = list(self._candidates)
        child._mask = self._mask.copy()
        return child

    @property
    def queue(self):
        return tuple(self._queue)

    @property
    def candidates(self):
        """Independent descriptors: mutable next-state graphs stay private."""
        return tuple(copy.deepcopy(candidate.description) for candidate in self._candidates)

    @property
    def action_mask(self):
        return self._mask.copy()

    @property
    def num_steps(self):
        return self.steps

    @property
    def environment_context(self):
        return {"environment_version": ENVIRONMENT_VERSION, "action_version": ACTION_VERSION,
                "observation_version": OBSERVATION_VERSION, "config": asdict(self.config)}

    def _capacity_failure(self, graph):
        for name, count in (("max_blocks", len(graph.faces)), ("max_vertices", len(graph.vertices)),
                            ("max_half_edges", sum(len(face.vertices) for face in graph.faces.values())),
                            ("max_face_vertices", max((len(face.vertices) for face in graph.faces.values()), default=0))):
            if count > getattr(self.config, name):
                return name.removeprefix("max_") + "_budget_exhausted"
        return None

    def _validate(self, graph, *, refine=False):
        failure = self._capacity_failure(graph)
        if failure:
            return {"valid": False, "reason": failure, "quality": {}}
        self._work_counters["topology_validation_calls"] += 1
        invalid = validate_graph(graph, previous=self._initial, max_blocks=self.config.max_blocks,
                                 max_half_edges=self.config.max_half_edges)
        if invalid is not None:
            return {"valid": False, "reason": invalid.reason, "quality": {}}
        self._work_counters["geometry_validation_calls"] += 1
        result = copy.deepcopy(self.geometry.validate(graph, refine=refine))
        if not isinstance(result, dict) or "valid" not in result:
            raise ValueError("CurveGeometry.validate must return a validity record")
        result.setdefault("quality", {})
        result.setdefault("reason", "valid" if result["valid"] else "invalid_physical_geometry")
        if result["valid"]:
            q = float(result["quality"].get("q", 0.))
            if not math.isfinite(q):
                raise ValueError("Geometry quality must be finite")
        return result

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        selected = copy.deepcopy((options or {}).get("sample", self.sample))
        # Per-sample geometry caches are pure and expensive. Preserve them when
        # resetting this same episode source; never reuse across a new profile.
        if self.geometry is None or selected != self.sample:
            self.geometry = CurveGeometry(selected, self.config)
            # Existing forks may still use the previous immutable geometry.
            # Detach, rather than clear/repopulate their shared pure cache.
            self._enumeration_cache = OrderedDict()
        self.sample = selected
        self.graph = initial_graph(self.geometry.domain)
        self._initial = self.graph.clone()
        self._queue = [fid for fid in sorted(self.graph.faces) if len(self.graph.faces[fid].vertices) != 4]
        self.steps, self._history = 0, []
        self._attempt_validation = None
        self.success = self.terminated = False
        self._candidates = []
        self._mask[:] = False
        failure = self._capacity_failure(self.graph)
        # The fixed observation cannot truthfully represent an overflowing input.
        if len(half_edges(self.graph)) > HALF_EDGE_CAPACITY:
            raise ValueError("Initial graph exceeds the fixed halfedge schema; observation is never truncated")
        self._validation = self._validate(self.graph) if not failure else {"valid": False, "reason": failure, "quality": {}}
        if not self._validation["valid"]:
            self.terminated, self._reason = True, self._validation["reason"]
        elif not self._queue:
            self._validation = self._validate(self.graph, refine=True)
            self.success = bool(self._validation["valid"])
            self.terminated, self._reason = True, "completed" if self.success else self._validation["reason"]
        else:
            self._candidates, failure = self._build_candidates(self.graph, self._queue)
            self.terminated, self._reason = bool(failure), failure or "reset"
            if not self.terminated:
                self._mask[:len(self._candidates)] = True
        return self._observation(), self._info()

    def _next_queue(self, graph, queue, *, cut=False, new_face=None):
        result = [fid for fid in (queue[1:] if cut else queue)
                  if fid in graph.faces and len(graph.faces[fid].vertices) != 4]
        if cut:
            for fid in (queue[0], new_face):
                if fid in graph.faces and len(graph.faces[fid].vertices) != 4 and fid not in result:
                    result.append(fid)
        # An insertion on a shared edge can turn an already completed quad into
        # a pentagon. Re-enqueue every such owner after the existing FIFO work.
        for fid in sorted(graph.faces):
            if len(graph.faces[fid].vertices) != 4 and fid not in result:
                result.append(fid)
        return tuple(result)

    def _candidate_cache_key(self, graph, queue):
        return (self.config, tuple(sorted(graph.vertices.items())),
                tuple((fid, face.vertices, face.protected) for fid, face in sorted(graph.faces.items())),
                graph.next_vertex_id, graph.next_face_id, tuple(queue))

    def _build_candidates(self, graph, queue):
        key = self._candidate_cache_key(graph, queue)
        if key in self._enumeration_cache:
            self._work_counters["candidate_cache_hits"] += 1
            return self._enumeration_cache[key]
        self._work_counters["candidate_cache_misses"] += 1
        result = self._enumerate_candidates(graph, queue)
        # Pure candidate graphs are private, and step always clones on commit.
        # Bound per-environment retention; cache eviction changes speed only.
        if len(self._enumeration_cache) >= 32:
            self._enumeration_cache.popitem(last=False)
        self._enumeration_cache[key] = result
        return result

    def _enumerate_candidates(self, graph, queue, *, stop_after_first=False):
        if not queue:
            return [], None
        if graph.faces[queue[0]].protected:
            return [], "protected_face_in_pending_queue"
        candidates, rejected = [], []
        for candidate in self._iter_candidates(graph, queue, rejected=rejected):
            candidates.append(candidate)
            if stop_after_first:
                return candidates, None
        if len(candidates) > self.config.max_candidates:
            return [], "candidate_budget_exhausted"
        if not candidates:
            budgets = sorted({reason for reason in rejected if "budget" in reason})
            return [], budgets[0] if budgets and len(budgets) == len(set(rejected)) else "no_legal_actions"
        return candidates, None

    def _iter_candidates(self, graph, queue, *, rejected=None, unique_insertions=False):
        if not queue:
            return [], None
        face = graph.faces[queue[0]]
        if face.protected:
            return [], "protected_face_in_pending_queue"
        cycle = list(face.vertices)
        points = [graph.vertices[vertex] for vertex in cycle]
        registry = half_edges(graph)
        slots = {edge.source: edge.slot for edge in registry if edge.face_id == face.id}
        incidence, protected = edge_incidence(graph), protected_edges(graph)
        rejected = [] if rejected is None else rejected

        def make_candidate(description, nodes, changed, following):
            validation = self._validate(changed)
            if not validation["valid"]:
                rejected.append(validation["reason"])
                return
            return _Candidate(description, tuple(nodes), changed, validation, tuple(following))

        for i in range(len(cycle)):
            for j in range(i + 3, len(cycle)):
                if len(cycle) - (j - i) + 1 < 4 or not valid_chord(points, i, j):
                    continue
                a, b = cycle[i], cycle[j]
                changed = graph.clone()
                changed.faces[face.id] = FaceBlock(face.id, tuple(cycle[i:j + 1]), False)
                new_id = changed.next_face_id
                changed.next_face_id += 1
                changed.faces[new_id] = FaceBlock(new_id, tuple(cycle[j:] + cycle[:i + 1]), False)
                description = {"kind": "insert_chord", "face_id": face.id, "source": a, "target": b,
                               "source_uv": list(graph.vertices[a]), "target_uv": list(graph.vertices[b]),
                               "owners": [face.id]}
                candidate = make_candidate(description, (slots[a], slots[b]), changed,
                       self._next_queue(changed, queue, cut=True, new_face=new_id))
                if candidate is not None:
                    yield candidate

        seen = set()
        for i, a in enumerate(cycle):
            b = cycle[(i + 1) % len(cycle)]
            key = tuple(sorted((a, b)))
            if key in protected or key in seen:
                continue
            seen.add(key)
            owners = incidence[key]
            if any(graph.faces[fid].protected for fid, _ in owners):
                continue
            if unique_insertions and face.id != min(fid for fid, _ in owners):
                continue
            for fraction in INSERT_FRACTIONS:
                point = tuple(float(value) for value in self.geometry.arc_point(graph.vertices[a], graph.vertices[b], fraction))
                if len(point) != 2 or not all(math.isfinite(value) for value in point):
                    raise ValueError("Physical arc insertion must return a finite UV pair")
                if not on_segment(graph.vertices[a], graph.vertices[b], point):
                    raise ValueError("Physical arc insertion must stay on its parameter edge")
                if any(math.dist(point, existing) <= 1e-12 for existing in graph.vertices.values()):
                    continue
                changed = graph.clone()
                new_id = changed.next_vertex_id
                changed.next_vertex_id += 1
                changed.vertices[new_id] = point
                for fid, index in owners:
                    old = changed.faces[fid]
                    vertices = list(old.vertices)
                    vertices.insert(index + 1, new_id)
                    changed.faces[fid] = FaceBlock(fid, tuple(vertices), old.protected)
                description = {"kind": "insert_edge_point", "face_id": face.id, "source": a, "target": b,
                               "source_uv": list(graph.vertices[a]), "target_uv": list(graph.vertices[b]),
                               "fraction": fraction, "point_uv": list(point), "owners": sorted(fid for fid, _ in owners)}
                candidate = make_candidate(description, (slots[a], slots[b]), changed, self._next_queue(changed, queue))
                if candidate is not None:
                    yield candidate

    def _reward(self, next_graph, *, terminal, success, validation):
        before = topology_potential(self.graph)
        after = 0. if terminal else topology_potential(next_graph)
        quality = validation.get("quality", {})
        q = float(quality.get("q", 0.)) if success else 0.
        outer = sum(not face.protected for face in next_graph.faces.values())
        terminal_score = (self.config.success_reward + q - self.config.outer_block_cost * outer
                          if success else self.config.failure_reward if terminal else 0.)
        base = -self.config.step_cost + terminal_score
        shaping = self.config.gamma * after - before
        components = {"step_cost": -self.config.step_cost, "terminal_score": terminal_score,
                      "base_reward": base, "potential_before": before, "potential_after": after,
                      "shaping": shaping, "total": base + shaping}
        return float(base + shaping), components

    def _outcome(self, candidate, *, preview=False):
        changed, queue = candidate.graph, list(candidate.queue)
        validation = copy.deepcopy(candidate.validation)
        success, terminal, reason = False, False, "graph_edit"
        attempt_validation = None
        following = []
        if not queue:
            validation = self._validate(changed, refine=True)
            terminal, success = True, bool(validation["valid"])
            reason = "completed" if success else "terminal_physical_validation:" + validation["reason"]
            if not success:
                attempt_validation = validation
                changed, queue = self.graph, list(self._queue)
                validation = copy.deepcopy(self._validation)
        elif self.steps + 1 >= self.config.max_steps:
            terminal, reason = True, "step_budget_exhausted"
        else:
            cache_key = self._candidate_cache_key(changed, queue)
            degree = len(changed.faces[queue[0]].vertices)
            # Every possible nonadjacent pair plus three insertions per edge
            # bounds the candidate count, even before geometry/protection tests.
            # Above that bound, a single legal witness proves the exact next-step
            # nonterminal status. Never cache a witness list as a complete mask.
            upper_bound = degree * (degree - 3) // 2 + 3 * degree
            if preview and cache_key not in self._enumeration_cache and self.config.max_candidates >= upper_bound:
                following, failure = self._enumerate_candidates(changed, queue, stop_after_first=True)
            else:
                following, failure = self._build_candidates(changed, queue)
            if failure:
                terminal, reason = True, failure
        reward, components = self._reward(changed, terminal=terminal, success=success, validation=validation)
        return {"graph": changed, "queue": queue, "validation": validation, "success": success,
                "terminated": terminal, "reason": reason, "candidates": following,
                "reward": reward, "reward_components": components, "attempt_validation": attempt_validation}

    def preview(self, action):
        """Exact next-step reward/termination, with no graph or queue mutation."""
        if self.graph is None or self.terminated:
            raise RuntimeError("Preview requires a live reset episode")
        if not isinstance(action, Integral) or isinstance(action, (bool, np.bool_)) or not 0 <= int(action) < len(self._candidates):
            raise ValueError("Preview requires a currently legal candidate index")
        candidate = self._candidates[int(action)]
        outcome = self._outcome(candidate, preview=True)
        return copy.deepcopy({"action": int(action), "description": candidate.description,
            "reward": outcome["reward"], "reward_components": outcome["reward_components"],
            "quality": outcome["validation"].get("quality", {}), "success": outcome["success"],
            "terminated": outcome["terminated"], "reason": outcome["reason"],
            "next_graph": outcome["graph"].to_dict(), "queue": outcome["queue"],
            "attempt_validation": outcome["attempt_validation"]})

    peek = preview

    def step(self, action):
        if self.graph is None:
            raise RuntimeError("Call reset before step")
        if self.terminated:
            raise RuntimeError("Episode is complete; reset before another action")
        valid = (isinstance(action, Integral) and not isinstance(action, (bool, np.bool_))
                 and 0 <= int(action) < len(self._candidates))
        if not valid:
            reward, components = self._reward(self.graph, terminal=True, success=False, validation=self._validation)
            self.steps += 1
            self.success, self.terminated, self._reason = False, True, "invalid_action"
            self._candidates = []
            self._mask[:] = False
            return self._observation(), reward, True, False, self._info(
                invalid_action=True, action=None, base_reward=components["base_reward"],
                shaping=components["shaping"], reward_components=components)
        action = int(action)
        candidate = self._candidates[action]
        # Compute and validate the entire transition before committing state.
        outcome = self._outcome(candidate)
        self.graph = outcome["graph"].clone()
        self._queue = list(outcome["queue"])
        self._validation = copy.deepcopy(outcome["validation"])
        self._attempt_validation = copy.deepcopy(outcome["attempt_validation"])
        self.steps += 1
        self.success, self.terminated, self._reason = outcome["success"], outcome["terminated"], outcome["reason"]
        accepted = not self._reason.startswith("terminal_physical_validation:")
        self._history.append({"step": self.steps, "action": action, "accepted": accepted,
                              "reason": self._reason, **copy.deepcopy(candidate.description)})
        self._candidates = [] if self.terminated else outcome["candidates"]
        self._mask[:] = False
        if not self.terminated:
            self._mask[:len(self._candidates)] = True
        components = outcome["reward_components"]
        return self._observation(), outcome["reward"], bool(self.terminated), False, self._info(
            invalid_action=False, action=action, base_reward=components["base_reward"],
            shaping=components["shaping"], reward_components=copy.deepcopy(components))

    def _arc_log(self, a, b):
        # Observation-only approximation. Legal cuts and insertion positions use
        # the adaptive physical geometry validator/arc_point, not this feature.
        fraction = np.linspace(0., 1., 17)
        uv = np.asarray(a)[None] + fraction[:, None] * (np.asarray(b) - np.asarray(a))[None]
        xy = np.asarray(self.geometry.mapping.evaluate(uv), dtype=float)
        length = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
        return math.log1p(float(length)) / math.log1p(self.config.farfield)

    def _physical_tangents(self, uv, target_uv, xy):
        """Preserve legacy tangents, repairing only unresolvable differences.

        Repeated legal arc insertions can make 1e-5 times an edge shorter than
        one parameter-coordinate ULP. The edge itself is not thereby collapsed.
        Larger probes stay on that same directed edge; this changes observation
        numerics only and never changes physical/topological legality checks.
        """
        nearby = np.asarray(self.geometry.mapping.evaluate(uv + 1e-5 * (target_uv - uv)), dtype=float)
        tangents = nearby - xy
        lengths = np.linalg.norm(tangents, axis=1)
        if not np.isfinite(tangents).all() or not np.isfinite(lengths).all():
            raise ValueError("A physical edge has a nonfinite local tangent")
        for index in np.flatnonzero(lengths <= 1e-15):
            a, b, origin = uv[index], target_uv[index], xy[index]
            resolved = False
            for fraction in (1e-4, 1e-3, 1e-2, .1, 1.):
                probe = np.asarray(self.geometry.mapping.evaluate((a + fraction * (b-a))[None]), dtype=float)[0]
                delta = probe-origin
                length = float(np.linalg.norm(delta))
                if not np.isfinite(probe).all() or not math.isfinite(length):
                    raise ValueError("A physical edge has a nonfinite local tangent")
                # Demand several ULPs for a local subtraction, rather than
                # normalizing its first barely nonzero rounding artifact.
                resolution = max(1e-15, 64 * float(np.linalg.norm(np.spacing(np.maximum(np.abs(origin), np.abs(probe))))))
                if length > resolution:
                    tangents[index], lengths[index], resolved = delta, length, True
                    break
            if not resolved:
                curve = np.asarray(self.geometry.curve(a, b), dtype=float)
                if curve.ndim != 2 or curve.shape[1] != 2 or not np.isfinite(curve).all():
                    raise ValueError("A physical edge has an invalid sampled tangent curve")
                # Curve samples are in source->target order. The first
                # distinguishable displacement retains that orientation, even
                # for an edge whose endpoints coincide on a curved mapping.
                deltas = curve-origin
                distances = np.linalg.norm(deltas, axis=1)
                if not np.isfinite(distances).all():
                    raise ValueError("A physical edge has a nonfinite sampled tangent")
                distinct = np.flatnonzero(distances > 1e-15)
                if not len(distinct):
                    raise ValueError(f"A physical edge has a degenerate local tangent: {a.tolist()} -> {b.tolist()}")
                first = int(distinct[0])
                tangents[index], lengths[index] = deltas[first], distances[first]
        tangents /= lengths[:, None]
        return tangents

    def _candidate_slots(self):
        return CANDIDATE_CAPACITY

    def _observation(self):
        if self.graph is None:
            raise RuntimeError("Call reset before observation")
        graph = self.graph
        registry = half_edges(graph)
        slots = self._candidate_slots()
        if len(registry) > HALF_EDGE_CAPACITY or len(self._candidates) > slots:
            raise ValueError("Full state exceeds the fixed observation; truncation is forbidden")
        features = np.zeros((HALF_EDGE_CAPACITY, 32), dtype=np.float32)
        next_index = np.full(HALF_EDGE_CAPACITY, -1, dtype=np.int64)
        previous, twin = next_index.copy(), next_index.copy()
        protected = protected_edges(graph)
        uv = np.asarray([graph.vertices[edge.source] for edge in registry], dtype=float)
        xy = np.asarray(self.geometry.mapping.evaluate(uv), dtype=float)
        denominator = math.log1p(self.config.farfield)
        physical = np.sign(xy) * np.log1p(np.abs(xy)) / denominator
        target_uv = np.asarray([graph.vertices[edge.target] for edge in registry], dtype=float)
        tangents = self._physical_tangents(uv, target_uv, xy)
        # Stored face cycles are CCW. The right-hand normal points outside that
        # face; opposite owners therefore have opposite normals on straight edges.
        normals = np.column_stack((tangents[:, 1], -tangents[:, 0]))
        queue_order = {fid: order for order, fid in enumerate(self._queue)}
        face_order = {fid: order for order, fid in enumerate(sorted(graph.faces))}
        arc_cache = {}
        for edge in registry:
            a, b = graph.vertices[edge.source], graph.vertices[edge.target]
            face = graph.faces[edge.face_id]
            key = tuple(sorted((edge.source, edge.target)))
            if key not in arc_cache:
                arc_cache[key] = self._arc_log(a, b)
            bottom = abs(a[1]) + abs(b[1]) < 1e-12
            middle_u = (a[0] + b[0]) / 2
            wall = bottom and .25 <= middle_u <= .75
            side = -1. if abs(a[0]) + abs(b[0]) < 1e-12 else 1. if abs(a[0] - 1) + abs(b[0] - 1) < 1e-12 else 0.
            features[edge.slot] = [*a, *b, *physical[edge.slot], *physical[edge.next],
                *tangents[edge.slot],
                len(face.vertices) / self.config.max_face_vertices, area([graph.vertices[v] for v in face.vertices]),
                math.dist(a, b), float(face.protected), float(key in protected), float(edge.twin >= 0),
                float(wall), float(bottom and not wall), float(abs(a[1] - 1) + abs(b[1] - 1) < 1e-12), side,
                *normals[edge.slot], float(abs(a[1] - .5) + abs(b[1] - .5) < 1e-12), 1.,
                float(bool(self._queue) and edge.face_id == self._queue[0]), float(edge.face_id in queue_order),
                (queue_order[edge.face_id] + 1) / self.config.max_blocks if edge.face_id in queue_order else 0.,
                float(len(face.vertices) == 4), arc_cache[key], max(0., 1 - self.steps / self.config.max_steps),
                max(0., 1 - len(graph.faces) / self.config.max_blocks), face_order[edge.face_id] / self.config.max_blocks]
            next_index[edge.slot], previous[edge.slot], twin[edge.slot] = edge.next, edge.previous, edge.twin
        candidate_features = np.zeros((slots, 16), dtype=np.float32)
        candidate_nodes = np.full((slots, 2), -1, dtype=np.int64)
        for index, candidate in enumerate(self._candidates):
            desc = candidate.description
            if desc["kind"] == "stop":
                candidate_features[index, -2] = candidate.validation["quality"].get("q", 0.)
                continue
            a, b = desc["source_uv"], desc["target_uv"]
            key = tuple(sorted((desc["source"], desc["target"])))
            if key not in arc_cache:
                arc_cache[key] = self._arc_log(a, b)
            quality = candidate.validation.get("quality", {})
            candidate_features[index] = [float(desc["kind"] == "insert_chord"), float(desc["kind"] == "insert_edge_point"),
                *a, *b, desc.get("fraction", 0.), b[0] - a[0], b[1] - a[1], math.dist(a, b), arc_cache[key],
                len(graph.faces[desc["face_id"]].vertices) / self.config.max_face_vertices,
                queue_order.get(desc["face_id"], 0) / self.config.max_blocks, len(desc["owners"]) / 2,
                float(quality.get("q", 0.)), sum(not f.protected for f in candidate.graph.faces.values()) / self.config.max_blocks]
            candidate_nodes[index] = candidate.nodes
        polygon = np.asarray(self.geometry.domain.polygon, dtype=float)
        global_features = np.asarray([self.steps / self.config.max_steps, len(graph.faces) / self.config.max_blocks,
            len(registry) / self.config.max_half_edges, len(graph.vertices) / self.config.max_vertices,
            -topology_potential(graph), len(self._queue) / self.config.max_blocks,
            sum(not f.protected for f in graph.faces.values()) / self.config.max_blocks,
            len(graph.faces[self._queue[0]].vertices) / self.config.max_face_vertices if self._queue else 0.,
            np.ptp(polygon[:, 1]), np.mean(polygon[:, 1]), math.log1p(self.config.farfield) / math.log(501.),
            self.config.stretch / 40], dtype=np.float32)
        if not all(np.isfinite(value).all() for value in (features, global_features, candidate_features)):
            raise ValueError("Nonfinite physical graph observation")
        return {"features": features, "next": next_index, "previous": previous, "twin": twin,
                "global": global_features, "candidates": candidate_features,
                "candidate_nodes": candidate_nodes, "mask": self._mask[:slots].copy()}

    def _info(self, **extra):
        return {"sample_id": self.geometry.domain.sample_id, "steps": self.steps, "success": bool(self.success),
                "terminated": bool(self.terminated), "reason": self._reason, "queue": list(self._queue),
                "quality": copy.deepcopy(self._validation.get("quality", {})), **self.environment_context,
                "attempt_validation": copy.deepcopy(self._attempt_validation),
                "scope": "C-constrained curved-block partition; no block-interior mesh or CFD", **extra}

    def partition(self):
        if self.graph is None:
            raise RuntimeError("Call reset before partition")
        result = copy.deepcopy(self.geometry.partition(self.graph))
        result.update(self._info(), graph=self.graph.to_dict(), cut_history=copy.deepcopy(self._history))
        return result
