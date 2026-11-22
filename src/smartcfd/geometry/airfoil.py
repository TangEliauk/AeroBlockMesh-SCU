"""Closed trailing-edge NACA four-digit airfoil geometry."""

import numpy as np

from smartcfd.core.types import Case


def make_case(m: float, p: float, t: float, name: str | None = None) -> Case:
    """Use chord fractions for maximum camber, its position, and thickness."""
    if not 0.0 <= m <= 0.09 or not 0.0 <= p < 1.0 or not 0.0 < t <= 0.4:
        raise ValueError("NACA camber, position, and thickness must be chord fractions")
    if m and not 0.0 < p < 1.0:
        raise ValueError("A cambered airfoil requires a position strictly between 0 and 1")
    if name is None:
        name = f"NACA {m * 100:.0f}{p * 10:.0f}{round(t * 100):02d}"
    return Case(name, float(m), float(p), float(t))


def case_features(case: Case) -> np.ndarray:
    """Map chord fractions to features of comparable magnitude."""
    return np.array([case.m / 0.06, case.p, case.t / 0.24], dtype=np.float32)


def naca(code: str) -> Case:
    """Parse a four-digit NACA code in the SAC model's supported range."""
    if len(code) != 4 or not code.isascii() or not code.isdecimal():
        raise ValueError("Use a four-digit NACA code, such as 2412.")
    m, p, t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
    if (m == 0 and p != 0) or (m > 0 and not 0.2 <= p <= 0.7) or not 0.08 <= t <= 0.24:
        raise ValueError("This prototype supports 8–24% thickness, 0–6% camber and 20–70% camber location.")
    if m > 0.06:
        raise ValueError("This prototype supports up to 6% camber.")
    return make_case(m, p, t, code)


def airfoil_surface(case: Case, theta: np.ndarray) -> np.ndarray:
    """Sample the analytic contour from the trailing edge counterclockwise."""
    # Wrapping the angle modulo 2π closes the contour at the trailing edge.
    angle = np.remainder(theta, 2.0 * np.pi)
    x = 0.5 * (1.0 + np.cos(angle))
    thickness = 5.0 * case.t * (
        0.2969 * np.sqrt(x) - x * (0.1260 + x * (0.3516 - x * (0.2843 - 0.1036 * x)))
    )
    thickness = np.where(x == 1.0, 0.0, thickness)
    if case.m:
        denominator = np.where(x < case.p, case.p**2, (1.0 - case.p)**2)
        camber = case.m / denominator * np.where(
            x < case.p, x * (2.0 * case.p - x), (1.0 - x) * (1.0 + x - 2.0 * case.p)
        )
        tangent = np.arctan(2.0 * case.m * (case.p - x) / denominator)
    else:
        camber, tangent = 0.0, 0.0
    offset = np.where(angle <= np.pi, thickness, -thickness)
    return np.stack((x - offset * np.sin(tangent), camber + offset * np.cos(tangent)), axis=-1)
