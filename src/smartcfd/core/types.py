"""Shared, serializable contracts for the constrained two-dimensional prototype."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import copy

UV = tuple[float, float]
ENVIRONMENT_VERSION = 'final_mesh_v2'


@dataclass
class Domain:
    sample_id: str
    polygon: Any
    farfield: float = 500.0
    stretch: float = 18.0
    protected_v: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FaceBlock:
    id: int
    vertices: tuple[int, ...]
    protected: bool = False


@dataclass
class BlockGraph:
    vertices: dict[int, UV]
    faces: dict[int, FaceBlock]
    next_vertex_id: int = 0
    next_face_id: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "BlockGraph":
        return copy.deepcopy(self)

    @property
    def protected_face_ids(self) -> set[int]:
        return {fid for fid, face in self.faces.items() if face.protected}

    def to_dict(self) -> dict[str, Any]:
        return {"vertices": {str(k): list(v) for k, v in self.vertices.items()},
                "faces": [{"id": f.id, "vertices": list(f.vertices), "protected": f.protected}
                          for f in sorted(self.faces.values(), key=lambda f: f.id)],
                "next_vertex_id": self.next_vertex_id, "next_face_id": self.next_face_id,
                "metadata": copy.deepcopy(self.metadata)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BlockGraph":
        vertices = {int(k): tuple(v) for k, v in value["vertices"].items()}
        faces = {int(f["id"]): FaceBlock(int(f["id"]), tuple(f["vertices"]), f.get("protected", False))
                 for f in value["faces"]}
        return cls(vertices, faces, value.get("next_vertex_id", max(vertices, default=-1) + 1),
                   value.get("next_face_id", max(faces, default=-1) + 1), copy.deepcopy(value.get("metadata", {})))


@dataclass
class MeshSpec:
    intervals: int = 16
    protected_intervals: int = 32
    max_intervals: int = 256
    smoothing_steps: int = 0
    max_cells: int = 500000


@dataclass
class MultiBlockMesh:
    blocks: dict[int, Any]
    interfaces: list[dict[str, Any]] = field(default_factory=list)
    boundaries: list[dict[str, Any]] = field(default_factory=list)
    quality: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Failure:
    reason: str
    detail: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvConfig:
    max_half_edges: int = 256
    local_actions: int = 4
    max_blocks: int = 32
    max_steps: int = 128
    gamma: float = 0.99
    step_cost: float = 0.01
    failure_reward: float = -5.0
    success_reward: float = 5.0
    intervals: int = 16
    protected_intervals: int = 32
    max_intervals: int = 256
    smoothing_steps: int = 0
    max_cells: int = 500000
    environment_version: str = ENVIRONMENT_VERSION

    @property
    def mesh_spec(self) -> MeshSpec:
        return MeshSpec(intervals=self.intervals, protected_intervals=self.protected_intervals,
                        max_intervals=self.max_intervals, smoothing_steps=self.smoothing_steps,
                        max_cells=self.max_cells)


@dataclass
class Transition:
    observation: dict[str, Any]
    action: int
    reward: float
    next_observation: dict[str, Any]
    terminated: bool
    truncated: bool = False
