"""TFI diagnostics on the terminal graph and CH-SAC-compatible quality metrics."""
from __future__ import annotations

import math

import numpy as np

from smartcfd.core.types import Failure, MeshSpec
from smartcfd.evaluation.jacobian import corner_scaled_jacobians
from smartcfd.meshing.generator import generate_mesh, side_nodes

BUDGETS = (1536, 5120)


def mesh_metrics(mesh):
    jacobians, aspects, centroids = [], [], []
    for nodes in mesh.blocks.values():
        jacobians.extend(corner_scaled_jacobians(nodes).min(axis=-1).ravel())
        xy = np.stack((nodes[:-1, :-1], nodes[1:, :-1], nodes[1:, 1:], nodes[:-1, 1:]), axis=-2)
        lengths = np.linalg.norm(np.roll(xy, -1, axis=-2)-xy, axis=-1)
        a, b = (lengths[..., 0]+lengths[..., 2])/2, (lengths[..., 1]+lengths[..., 3])/2
        aspects.extend((np.maximum(a, b)/np.minimum(a, b)).ravel())
        centroids.extend(xy.mean(axis=-2).reshape(-1, 2))
    jacobians, aspects, centroids = map(np.asarray, (jacobians, aspects, centroids))
    x, y = centroids.T
    far = (np.abs(y) > .25) | (x > 1.5) | (x < -.5)
    angles = []
    for boundary in mesh.boundaries:
        if boundary["tag"] == "wall":
            nodes = mesh.blocks[boundary["block"]]
            side = boundary["side"]
            wall = side_nodes(nodes, side)
            near = (nodes[:, 1], nodes[-2, :], nodes[::-1, -2], nodes[1, ::-1])[side]
            tangent = np.diff(wall, axis=0)
            offset = (near[1:]+near[:-1]-wall[1:]-wall[:-1])/2
            scale = np.linalg.norm(tangent, axis=1)*np.linalg.norm(offset, axis=1)
            cosine = np.divide(np.abs((tangent*offset).sum(axis=-1)), scale,
                               out=np.ones_like(scale), where=scale > 0)
            angles.extend(np.degrees(np.arccos(np.clip(cosine, 0, 1))))
    if not angles or not far.any() or not np.isfinite(aspects).all():
        raise ValueError("Mesh has incomplete wall or farfield diagnostics")
    jmin, j05 = float(jacobians.min()), float(np.quantile(jacobians, .05))
    wall_q10 = float(np.quantile(angles, .10))
    far_ar = float(aspects[far].max())
    score = (.40*j05 + .35*jmin + .25*np.clip((wall_q10-60)/30, 0, 1)
             - .35*max(0., math.log10(far_ar/150)))
    return {"valid": True, "quads": int(len(jacobians)), "blocks": len(mesh.blocks),
            "min_scaled_jacobian": jmin, "q05_scaled_jacobian": j05,
            "wall_angle_q10_deg": wall_q10, "max_aspect_ratio": float(aspects.max()),
            "farfield_max_aspect_ratio": far_ar, "farfield_q95_aspect_ratio": float(np.quantile(aspects[far], .95)),
            "interface_max_abs_difference": mesh.quality["interface_max_abs_difference"],
            "first_height_min": mesh.quality["first_height_min"],
            "first_height_max": mesh.quality["first_height_max"], "score": float(score)}


def reward_mesh_score(domain, graph):
    mesh = generate_mesh(domain, graph, MeshSpec(intervals=4, protected_intervals=8))
    if isinstance(mesh, Failure):
        return -1.
    return float(np.clip(mesh_metrics(mesh)["score"], -1, 1))


def evaluate_mesh_budget(domain, graph, budget):
    mesh = generate_mesh(domain, graph, MeshSpec(max_intervals=64), target_cells=budget)
    if isinstance(mesh, Failure):
        return {"valid": False, "reason": mesh.reason, "detail": mesh.detail,
                "target": budget, "diagnostics": mesh.diagnostics}, None
    metrics = mesh_metrics(mesh)
    metrics.update(target=budget, budget_match=abs(metrics["quads"]-budget) <= .05*budget,
                   outer_intervals=mesh.metadata["outer_intervals"],
                   protected_intervals=mesh.metadata["protected_intervals"])
    return metrics, mesh
