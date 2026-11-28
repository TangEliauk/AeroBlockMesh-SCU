"""Seeded continuous synthetic orthogonal cases; no imported CFD geometries.

Train/val share named procedural families (an interpolation development split,
not family holdout). Each normalized geometric identity is a separate group;
duplicate translated/uniformly scaled shapes cannot cross the split.
"""
from __future__ import annotations

import random

from .geometry import canonical_vertices, polygon_hash

FAMILIES = ("rectangle", "L", "T", "U", "C", "stair")


def _vertices(family, rng):
    w, h = rng.uniform(2., 6.), rng.uniform(2., 6.)
    left, right = w * rng.uniform(.18, .38), w * rng.uniform(.18, .38)
    low, high = h * rng.uniform(.18, .38), h * rng.uniform(.18, .38)
    if family == "rectangle":
        points = [(0, 0), (w, 0), (w, h), (0, h)]
    elif family == "L":
        points = [(0, 0), (w, 0), (w, low), (left, low), (left, h), (0, h)]
    elif family == "T":
        points = [(left, 0), (w - right, 0), (w - right, h - high), (w, h - high),
                  (w, h), (0, h), (0, h - high), (left, h - high)]
    elif family == "U":
        points = [(0, 0), (w, 0), (w, h), (w - right, h), (w - right, low),
                  (left, low), (left, h), (0, h)]
    elif family == "C":
        points = [(0, 0), (w, 0), (w, low), (left, low), (left, h - high),
                  (w, h - high), (w, h), (0, h)]
    elif family == "stair":
        x1, x2 = w * rng.uniform(.20, .38), w * rng.uniform(.58, .80)
        y1, y2 = h * rng.uniform(.20, .38), h * rng.uniform(.58, .80)
        points = [(0, 0), (w, 0), (w, y1), (x2, y1), (x2, y2), (x1, y2), (x1, h), (0, h)]
    else:
        raise ValueError("Unknown synthetic family")
    return canonical_vertices(points)


def generate_cases(seed=42, train_count=40, val_count=12):
    """Return deterministic train/val lists with canonical case_id and groups."""
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in (train_count, val_count)):
        raise ValueError("Case counts must be nonnegative integers")
    rng = random.Random(seed)
    result = {"train": [], "val": []}
    seen = set()
    for split, count in (("train", train_count), ("val", val_count)):
        attempts = 0
        while len(result[split]) < count:
            attempts += 1
            if attempts > max(1000, count * 100):
                raise RuntimeError("Unable to generate the requested unique synthetic cases")
            family = FAMILIES[len(result[split]) % len(FAMILIES)]
            ring = _vertices(family, rng)
            identity = polygon_hash(ring)
            if identity in seen:
                continue
            seen.add(identity)
            result[split].append({"case_id": f"{family.lower()}_{identity[:16]}", "family": family,
                "family_group": family, "group_id": identity, "canonical_hash": identity,
                "vertices": [list(point) for point in ring], "split": split, "generation_seed": seed,
                "source": "synthetic_orthogonal", "human_reviewed": False,
                "split_scope": "Procedural-family interpolation; normalized geometry groups are disjoint, families are shared"})
    return result
