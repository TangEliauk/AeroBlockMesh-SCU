"""Signed scaled Jacobian and geometric quality for structured O-grids."""

import numpy as np


def _cell_metrics(mesh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Positive circulation: outward first, then counterclockwise around the ring.
    edges = (
        mesh[1:, :-1] - mesh[:-1, :-1],
        mesh[1:, 1:] - mesh[1:, :-1],
        mesh[:-1, 1:] - mesh[1:, 1:],
        mesh[:-1, :-1] - mesh[:-1, 1:],
    )
    lengths = tuple(np.sqrt(np.einsum("ijk,ijk->ij", edge, edge)) for edge in edges)
    jacobians = []
    for i, (first, second) in enumerate(zip(edges, edges[1:] + edges[:1])):
        cross = first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
        scale = lengths[i] * lengths[(i + 1) % 4]
        jacobians.append(np.divide(cross, scale, out=np.zeros_like(cross), where=scale > 0.0))
    jacobian = np.clip(np.minimum.reduce(jacobians), -1.0, 1.0)
    shortest = np.minimum.reduce(lengths)
    aspect = np.maximum.reduce(lengths) / np.maximum(shortest, np.finfo(float).tiny)
    return jacobian, aspect


def cell_quality(mesh: np.ndarray) -> np.ndarray:
    """Minimum signed scaled Jacobian at the four corners of every cell."""
    return _cell_metrics(mesh)[0]


def quality(mesh: np.ndarray) -> dict[str, float | int]:
    """Compute dimensionless geometric quality and count invalid quadrilaterals."""
    jacobian, aspect = _cell_metrics(mesh)
    invalid_cells = int(np.count_nonzero(jacobian <= 1e-10))
    minimum = float(np.min(jacobian))
    count = jacobian.size
    lower_index = int(0.05 * (count - 1))
    p05 = float(np.partition(jacobian.ravel(), lower_index)[lower_index])
    mean_jacobian = float(np.mean(jacobian))
    # For a convex quadrilateral, the worst corner sine determines the maximum
    # normalized deviation from the ideal interior angle of 90 degrees.
    skew = float(np.mean(np.arccos(np.clip(jacobian, 0.0, 1.0))) * (2.0 / np.pi))
    aspect_score = float(np.mean(1.0 / (1.0 + np.log1p(np.maximum(aspect - 1.0, 0.0)))))
    raw_score = 0.45 * p05 + 0.24 * mean_jacobian + 0.12 * minimum + 0.09 * (1.0 - skew) + 0.10 * aspect_score
    score = max(0.0, raw_score) / (1.0 + 0.75 * invalid_cells)
    wall_lengths = np.linalg.norm(mesh[1, :-1] - mesh[0, :-1], axis=-1)
    return {
        "score": float(min(score, 1.0)),
        "min_jacobian": minimum,
        "invalid_cells": invalid_cells,
        "p05_jacobian": p05,
        "mean_jacobian": mean_jacobian,
        "mean_skew": skew,
        "max_aspect_ratio": float(np.max(aspect)),
        "mean_first_layer": float(np.mean(wall_lengths)),
        "cells": count,
    }
