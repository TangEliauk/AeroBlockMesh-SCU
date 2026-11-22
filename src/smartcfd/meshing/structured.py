"""Conforming four-block O-grid for a two-dimensional NACA airfoil."""

from functools import lru_cache

import numpy as np

from smartcfd.core.types import Case
from smartcfd.geometry.airfoil import airfoil_surface


PARAM_NAMES = (
    "wall_spacing",
    "normal_blend",
    "tangential_bias",
    "upper_junction",
    "lower_junction",
)
PARAM_DIM = len(PARAM_NAMES)
BASE_PARAMS = np.zeros(PARAM_DIM, dtype=np.float32)


@lru_cache(maxsize=2)
def _block_coordinates(fine: bool) -> tuple[np.ndarray, np.ndarray]:
    radial_count, cells_per_block = (32, 40) if fine else (16, 24)
    radial = np.linspace(0.0, 1.0, radial_count + 1)[:, None, None]
    circumferential = np.linspace(0.0, 1.0, cells_per_block, endpoint=False)
    return radial, circumferential


def generate(case: Case, params: np.ndarray, fine: bool = False) -> np.ndarray:
    """Generate radial rows of nodes around four conforming structured blocks.

    The contour starts at the trailing edge, follows the upper surface, passes
    the leading edge and returns along the lower surface. Quarter-index seams
    are single shared radial lines. The outer boundary is a circle of radius
    four chords, and the last column duplicates the first to close the ring.
    """
    values = np.asarray(params, dtype=np.float64)
    if values.shape != (PARAM_DIM,) or not np.all(np.isfinite(values)):
        raise ValueError(f"Expected {PARAM_DIM} finite mesh parameters")
    spacing, normal_blend, bias, upper, lower = np.clip(values, -1.0, 1.0)
    uniform_r, block_u = _block_coordinates(fine)
    breaks = np.array([0.0, np.pi / 2.0 + 0.6 * upper, np.pi, 1.5 * np.pi + 0.6 * lower, 2.0 * np.pi])
    theta = (breaks[:-1, None] + np.diff(breaks)[:, None] * block_u).ravel()
    theta = np.append(theta, 2.0 * np.pi)
    theta = theta + 0.26 * bias * np.sin(theta)
    wall = airfoil_surface(case, theta)

    # Compute local normals independently of point density and seam placement.
    delta = 0.0002
    tangent = airfoil_surface(case, theta + delta) - airfoil_surface(case, theta - delta)
    normal = np.stack((tangent[:, 1], -tangent[:, 0]), axis=-1)
    normal /= np.linalg.norm(normal, axis=-1)[:, None]
    outer = np.stack((0.5 + 4.0 * np.cos(theta), 4.0 * np.sin(theta)), axis=-1)
    outward = outer - wall
    normal_offset = normal * np.linalg.norm(outward, axis=-1)[:, None] - outward

    # Stretch outward in wall-normal distance. The normal correction fades
    # before the far field, preserving both the airfoil and the outer circle.
    exponent = 4.8 - 1.35 * spacing
    r = np.expm1(exponent * uniform_r) / np.expm1(exponent)
    blend = (0.45 + 0.40 * normal_blend) * r * (1.0 - r)**14
    nodes = wall + r * outward + blend * normal_offset
    nodes[:, -1] = nodes[:, 0]
    return nodes
