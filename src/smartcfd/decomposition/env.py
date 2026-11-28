"""FIFO orthogonal polygon decomposition MDP, separate from the CFD MDP."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from shapely.geometry import Polygon

from .geometry import (GeometryError, canonical_vertices, is_rectangle, paper_reward,
                       partition_interfaces, partition_quality, polygon_hash, split_polygon)

ENVIRONMENT_VERSION = "orthogonal_decomposition_v1"
ACTION_VERSION = "vertex_axis_cut_v1"
OBSERVATION_VERSION = "orthogonal_queue_obs_v1"
FEATURE_NAMES = ("root_x", "root_y", "previous_dx", "previous_dy", "next_dx", "next_dy",
                 "interior_angle_over_pi", "previous_length_over_root_diagonal", "next_length_over_root_diagonal",
                 "real_vertex", "active_part", "pending_part", "completed_part", "part_order",
                 "part_vertex_fraction", "part_area_fraction", "part_bbox_width", "part_bbox_height",
                 "part_centroid_x", "part_centroid_y", "part_perimeter_fraction", "vertex_order",
                 "remaining_step_fraction", "remaining_block_fraction")


@dataclass(frozen=True)
class DecompositionConfig:
    max_vertices: int = 128
    max_steps: int = 32
    max_blocks: int = 32
    gamma: float = .99
    step_cost: float = .05
    block_cost: float = .05
    terminal_bonus: float = 5.
    invalid_penalty: float = -5.

    def __post_init__(self):
        for name in ("max_vertices", "max_steps", "max_blocks"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (4 if name == "max_vertices" else 1):
                raise ValueError(f"Invalid {name}")
        for name in ("gamma", "step_cost", "block_cost", "terminal_bonus", "invalid_penalty"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"Nonfinite {name}")
        if not 0 <= self.gamma <= 1 or min(self.step_cost, self.block_cost, self.terminal_bonus) < 0 or self.invalid_penalty > 0:
            raise ValueError("Invalid reward configuration")


class OrthogonalBlockEnv(gym.Env):
    """Every action uses a current active-ring vertex and a full H/V cut line.

    Pending parts are a FIFO queue. Completed rectangles remain in the state.
    Packing order is queue first, completed rectangles second; no geometry is
    cropped to fit the fixed observation. Actions are packed_slot*2+direction,
    direction 0=horizontal and 1=vertical. Identical axis lines retain only the
    first vertex in canonical ring order.
    """
    metadata = {"render_modes": []}

    def __init__(self, case=None, config=None):
        self.config = config if isinstance(config, DecompositionConfig) else DecompositionConfig(**(config or {}))
        self.case = case or {"case_id": "default_L", "family": "L", "vertices":
                             [[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3], [0, 0]]}
        n = self.config.max_vertices
        self.action_space = spaces.Discrete(n * 2)
        self.observation_space = spaces.Dict({
            "features": spaces.Box(-np.inf, np.inf, (n, 24), dtype=np.float32),
            "next": spaces.Box(-1, n - 1, (n,), dtype=np.int64),
            "previous": spaces.Box(-1, n - 1, (n,), dtype=np.int64),
            "twin": spaces.Box(-1, n - 1, (n,), dtype=np.int64),
            "mask": spaces.Box(0, 1, (n * 2,), dtype=np.bool_),
            "global": spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32)})
        self.root = None
        self.pending, self.blocks, self.cuts = [], [], []
        self.steps = 0
        self.success = self.terminated = False
        self._mask = np.zeros(n * 2, dtype=bool)
        self._candidates = {}

    @property
    def action_mask(self):
        return self._mask.copy()

    @property
    def environment_context(self):
        return {"environment_version": ENVIRONMENT_VERSION, "action_version": ACTION_VERSION,
                "observation_version": OBSERVATION_VERSION, "config": asdict(self.config)}

    @property
    def num_steps(self):
        return self.steps

    def _parts(self):
        return [*self.pending, *self.blocks]

    def _vertices_required(self, parts):
        return sum(len(ring) - 1 for ring in parts)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        selected = (options or {}).get("case", self.case)
        if not isinstance(selected, dict) or "vertices" not in selected:
            raise GeometryError("Pass a case dictionary with vertices")
        root = canonical_vertices(selected["vertices"])
        if len(root) - 1 > self.config.max_vertices:
            raise GeometryError("Initial geometry exceeds max_vertices; observation is never truncated")
        self.case = {**selected, "case_id": selected.get("case_id") or "case_" + polygon_hash(root)[:16],
                     "vertices": [list(point) for point in root]}
        self.root = root
        self.pending = [] if is_rectangle(root) else [root]
        self.blocks = [root] if is_rectangle(root) else []
        self.steps, self.cuts = 0, []
        self.success = not self.pending
        self.terminated = self.success
        self._refresh_mask()
        if self.pending and not self._mask.any():
            self.terminated = True
        return self._observation(), self._info("initially_complete" if self.success else "reset" if not self.terminated else "no_legal_actions")

    def _following_parts(self, result):
        pending = list(self.pending[1:])
        blocks = list(self.blocks)
        for child in result.children:
            (blocks if is_rectangle(child) else pending).append(child)
        return pending, blocks

    def _refresh_mask(self):
        self._mask[:] = False
        self._candidates = {}
        if self.terminated or not self.pending:
            return
        root_area = Polygon(self.root).area
        seen = set()
        for index, point in enumerate(self.pending[0][:-1]):
            for direction in (0, 1):
                key = (direction, point[1] if direction == 0 else point[0])
                if key in seen:
                    continue
                seen.add(key)
                try:
                    result = split_polygon(self.pending[0], index, direction, root_area=root_area)
                    pending, blocks = self._following_parts(result)
                    if len(pending) + len(blocks) > self.config.max_blocks:
                        continue
                    if self._vertices_required([*pending, *blocks]) > self.config.max_vertices:
                        continue
                    action = index * 2 + direction
                    self._candidates[action] = result
                    self._mask[action] = True
                except GeometryError:
                    continue

    def _components(self, result=None, *, invalid=False):
        paper = paper_reward(result.children) if result is not None else {"paper_reward": 0.}
        complete = result is not None and not self._following_parts(result)[0]
        components = {"paper_reward": paper["paper_reward"], "step_cost": -self.config.step_cost,
                      "block_cost": -self.config.block_cost * (len(result.children) - 1) if result is not None else 0.,
                      "terminal_bonus": self.config.terminal_bonus if complete else 0.,
                      "invalid_penalty": self.config.invalid_penalty if invalid else 0.}
        components["total"] = float(sum(components.values()))
        return components, paper

    def preview(self, action):
        """Nonmutating geometric candidate and exact immediate environment reward."""
        if not isinstance(action, Integral) or isinstance(action, bool) or int(action) not in self._candidates:
            raise GeometryError("Preview requires a currently legal action")
        action = int(action)
        result = self._candidates[action]
        pending, blocks = self._following_parts(result)
        components, paper = self._components(result)
        return {"action": action, "direction": result.direction, "direction_name": "horizontal" if result.direction == 0 else "vertical",
                "source_vertex": list(result.source_vertex), "coordinate": result.coordinate,
                "segments": [[list(a), list(b)] for a, b in result.segments],
                "children": [[list(point) for point in ring] for ring in result.children],
                "reward": components["total"], "reward_components": components, "paper_components": paper,
                "success": not pending, "quality": partition_quality(self.root, [*pending, *blocks])}

    peek = preview

    def step(self, action):
        if self.root is None:
            raise RuntimeError("Call reset before step")
        if self.terminated:
            raise RuntimeError("Episode is complete; reset before another action")
        valid_integer = isinstance(action, Integral) and not isinstance(action, bool)
        action = int(action) if valid_integer else -1
        result = self._candidates.get(action)
        components, paper = self._components(result, invalid=result is None)
        self.steps += 1
        if result is not None:
            self.pending, self.blocks = self._following_parts(result)
            self.cuts.append({"action": action, "direction": result.direction,
                              "direction_name": "horizontal" if result.direction == 0 else "vertical",
                              "source_vertex": list(result.source_vertex),
                              "segments": [[list(a), list(b)] for a, b in result.segments]})
        self.success = not self.pending
        self.terminated = self.success or self.steps >= self.config.max_steps
        reason = "completed" if self.success else "step_budget_exhausted" if self.terminated else "cut" if result is not None else "invalid_action"
        self._refresh_mask()
        if not self.terminated and not self._mask.any():
            self.terminated, reason = True, "no_legal_actions"
            self._mask[:] = False
        info = self._info(reason)
        info.update(action=action, invalid_action=result is None, reward_components=components, paper_components=paper)
        return self._observation(), components["total"], bool(self.terminated), False, info

    def _observation(self):
        n = self.config.max_vertices
        features = np.zeros((n, 24), dtype=np.float32)
        following = np.full(n, -1, dtype=np.int64)
        previous = np.full(n, -1, dtype=np.int64)
        twin = np.full(n, -1, dtype=np.int64)
        if self.root is None:
            raise RuntimeError("Call reset before observation")
        parts = self._parts()
        required = self._vertices_required(parts)
        if required > n:
            raise GeometryError("Full partition exceeds max_vertices; refusing to truncate observation")
        root = Polygon(self.root)
        xmin, ymin, xmax, ymax = root.bounds
        width, height = xmax - xmin, ymax - ymin
        diagonal = math.hypot(width, height)
        remaining_steps = max(0., 1 - self.steps / self.config.max_steps)
        remaining_blocks = max(0., 1 - len(parts) / self.config.max_blocks)
        offset = 0
        for order, ring in enumerate(parts):
            polygon = Polygon(ring)
            points = ring[:-1]
            count = len(points)
            pending = order < len(self.pending)
            active = pending and order == 0
            bx0, by0, bx1, by1 = polygon.bounds
            center = polygon.centroid
            for index, point in enumerate(points):
                before, after = points[index - 1], points[(index + 1) % count]
                incoming = (point[0] - before[0], point[1] - before[1])
                outgoing = (after[0] - point[0], after[1] - point[1])
                cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
                interior = .5 if cross > 0 else 1.5
                slot = offset + index
                features[slot] = [(point[0] - xmin) / width, (point[1] - ymin) / height,
                    (before[0] - point[0]) / width, (before[1] - point[1]) / height,
                    outgoing[0] / width, outgoing[1] / height, interior,
                    math.hypot(*incoming) / diagonal, math.hypot(*outgoing) / diagonal,
                    1., float(active), float(pending), float(not pending), order / max(1, self.config.max_blocks - 1),
                    count / n, polygon.area / root.area, (bx1 - bx0) / width, (by1 - by0) / height,
                    (center.x - xmin) / width, (center.y - ymin) / height,
                    polygon.length / (2 * (width + height)), index / count, remaining_steps, remaining_blocks]
                following[slot] = offset + (index + 1) % count
                previous[slot] = offset + (index - 1) % count
            offset += count
        global_features = np.array([self.steps / self.config.max_steps, len(parts) / self.config.max_blocks,
            len(self.blocks) / self.config.max_blocks, len(self.pending) / self.config.max_blocks,
            required / n, sum(Polygon(part).area for part in self.pending) / root.area,
            width / (width + height), height / (width + height)], dtype=np.float32)
        if not np.isfinite(features).all() or not np.isfinite(global_features).all():
            raise GeometryError("Nonfinite normalized observation")
        return {"features": features, "next": following, "previous": previous, "twin": twin,
                "mask": self._mask.copy(), "global": global_features}

    def _info(self, reason):
        return {"case_id": self.case["case_id"], "success": bool(self.success), "reason": reason,
                "steps": self.steps, "terminated": bool(self.terminated),
                "quality": partition_quality(self.root, self._parts()), **self.environment_context,
                "scope": "Orthogonal polygon partition only; no CFD mesh generation"}

    def partition(self):
        if self.root is None:
            raise RuntimeError("Call reset before partition")
        parts = self._parts()
        statuses = ["pending"] * len(self.pending) + ["completed"] * len(self.blocks)
        connections = partition_interfaces(parts, statuses)
        return {"case_id": self.case["case_id"], "root": [list(point) for point in self.root],
                "blocks": [[list(point) for point in ring] for ring in self.blocks],
                "pending": [[list(point) for point in ring] for ring in self.pending],
                "part_order": statuses, "cuts": list(self.cuts), "success": bool(self.success),
                "steps": self.steps, "quality": partition_quality(self.root, parts), **connections,
                "interface_note": "Parts are queue-then-completed order. Common split points and T junctions are explicit; this is not a conforming structured mesh.",
                **self.environment_context}
