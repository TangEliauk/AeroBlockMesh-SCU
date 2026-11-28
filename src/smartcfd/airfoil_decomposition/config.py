"""Independent contracts for C-constrained physical airfoil decomposition."""
from dataclasses import dataclass, asdict
import math

ENVIRONMENT_VERSION = 'airfoil_c_decomposition_v1'
ACTION_VERSION = 'airfoil_curve_candidates_v1'
OBSERVATION_VERSION = 'airfoil_curve_obs_v2'
PARTITION_VERSION = 'airfoil_curve_partition_v1'
FEATURE_DIM = 32
GLOBAL_DIM = 12
CANDIDATE_DIM = 16


@dataclass(frozen=True)
class AirfoilDecompositionConfig:
    farfield: float = 20.0
    stretch: float = 18.0
    protected_thickness: float = .02
    curve_tolerance: float = 1e-4
    area_tolerance: float = 1e-8
    curve_max_depth: int = 18
    max_steps: int = 32
    max_blocks: int = 32
    max_vertices: int = 128
    max_half_edges: int = 256
    max_face_vertices: int = 32
    max_candidates: int = 1024
    gamma: float = .99
    step_cost: float = .02
    failure_reward: float = -5.0
    success_reward: float = 5.0
    outer_block_cost: float = .02

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{key} must be a finite number')
        limits = {'max_steps': 32, 'max_blocks': 32, 'max_vertices': 128,
                  'max_half_edges': 256, 'max_face_vertices': 32, 'max_candidates': 1024,
                  'curve_max_depth': 24}
        for key, upper in limits.items():
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f'{key} must be an integer in [1,{upper}]')
        if not (self.farfield > 2 and 0 < self.stretch <= 40 and 0 < self.protected_thickness < self.farfield):
            raise ValueError('Invalid physical C-mapping parameters')
        if not (0 < self.curve_tolerance < 1 and 0 < self.area_tolerance < 1 and 0 <= self.gamma <= 1):
            raise ValueError('Invalid tolerances or discount')
        if not (self.step_cost >= 0 and self.outer_block_cost >= 0 and self.failure_reward < 0 and self.success_reward > 0):
            raise ValueError('Invalid reward coefficients')
