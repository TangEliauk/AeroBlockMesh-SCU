"""Coons transfinite interpolation with one global node sequence per graph curve."""

from dataclasses import dataclass

import numpy as np

from smartcfd.core.coons import coons
from smartcfd.evaluation.jacobian import corner_scaled_jacobians


@dataclass
class QuadMesh:
    points: np.ndarray
    vertices: dict
    curves: dict
    grids: dict
    quads: dict
    quality: dict


WALL_LABELS = ("airfoil", "wall", "slat", "main", "flap")


def generate(graph, wall_labels=WALL_LABELS):
    graph.validate()
    points = list(graph.vertices.values())
    vertices = {v: i for i, v in enumerate(graph.vertices)}
    curves = {}
    for eid, c in graph.curves.items():
        start = len(points)
        points.extend(c.stations)
        curves[eid] = np.r_[vertices[c.start], np.arange(start, len(points)), vertices[c.end]].astype(int)
    boundary_xy = np.asarray(points, dtype=float)
    grids, quads = {}, {}
    wall_angles = []
    scaled_all = []
    min_jac = np.inf
    for bid, block in graph.blocks.items():
        sides = [curves[e] if e > 0 else curves[-e][::-1] for e in block.edges]
        a, b, c, d = sides
        bottom, right, top, left = (boundary_xy[edge] for edge in (a, b, c[::-1], d[::-1]))
        xy = coons(bottom, right, top, left)
        grid = np.empty(xy.shape[:2], dtype=int)
        grid[0], grid[:, -1], grid[-1], grid[:, 0] = a, b, c[::-1], d[::-1]
        interior = xy[1:-1, 1:-1].reshape(-1, 2)
        grid[1:-1, 1:-1] = np.arange(len(points), len(points)+len(interior)).reshape(grid[1:-1, 1:-1].shape)
        points.extend(interior)
        cells = np.stack((grid[:-1,:-1], grid[:-1,1:], grid[1:,1:], grid[1:,:-1]), axis=-1)
        grids[bid], quads[bid] = grid, cells.reshape(-1, 4)
        xy[0], xy[:, -1], xy[-1], xy[:, 0] = bottom, right, top, left
        corners = np.stack((xy[:-1,:-1], xy[:-1,1:], xy[1:,1:], xy[1:,:-1]), axis=-2)
        edges = np.roll(corners, -1, axis=-2) - corners
        previous = np.roll(edges, 1, axis=-2)
        jac = previous[...,0]*edges[...,1] - previous[...,1]*edges[...,0]
        if not np.isfinite(jac).all() or jac.min() <= 0:
            raise ValueError(f"Block {bid} has nonpositive or invalid corner Jacobian {jac.min():.8g}")
        min_jac = min(min_jac, jac.min())
        lengths = np.linalg.norm(edges, axis=-1)
        scaled = jac / (lengths * np.roll(lengths, 1, axis=-1))
        scaled_all.extend(scaled.min(axis=-1).ravel())
        for i, row in enumerate((scaled[0], scaled[:,-1], scaled[-1], scaled[:,0])):
            if graph.curves[abs(block.edges[i])].boundary in wall_labels:
                sine = np.minimum(row[:,i], row[:,(i+1)%4])
                wall_angles.extend(np.degrees(np.arcsin(np.clip(sine, 0, 1))))
    quality = {"quads": len(scaled_all), "min_corner_jacobian": float(min_jac),
               "min_scaled_jacobian": float(np.min(scaled_all)),
               "q05_scaled_jacobian": float(np.quantile(scaled_all, .05)),
               "wall_angle_q10_deg": float(np.quantile(wall_angles, .1)) if wall_angles else None}
    return QuadMesh(np.asarray(points), vertices, curves, grids, quads, quality)


def coons_patch(bottom, right, top, left):
    """Edges have increasing i/j orientation, including top and left."""
    bottom, right, top, left = map(np.asarray, (bottom, right, top, left))
    if len(bottom) != len(top) or len(left) != len(right):
        raise ValueError("opposite edge point counts differ")
    corners = [(bottom[0],left[0]),(bottom[-1],right[0]),(top[0],left[-1]),(top[-1],right[-1])]
    if not all(np.array_equal(a,b) for a,b in corners):
        raise ValueError("TFI edge corners must coincide exactly")
    s = np.linspace(0,1,len(bottom))[:,None,None]
    t = np.linspace(0,1,len(left))[None,:,None]
    bilinear = (1-s)*(1-t)*bottom[0] + s*(1-t)*bottom[-1] + (1-s)*t*top[0] + s*t*top[-1]
    nodes = (1-t)*bottom[:,None,:] + t*top[:,None,:] + (1-s)*left[None,:,:] + s*right[None,:,:] - bilinear
    nodes[:,0], nodes[:,-1], nodes[0,:], nodes[-1,:] = bottom, top, left, right
    return nodes


def smooth_elliptic(nodes, steps):
    """Fixed-edge Winslow-type updates with a line search; protect BL separately."""
    p = np.asarray(nodes,dtype=float).copy()
    accepted = 0
    for _ in range(steps):
        if min(p.shape[:2]) < 3:
            break
        ri = (p[2:,1:-1]-p[:-2,1:-1])/2
        rj = (p[1:-1,2:]-p[1:-1,:-2])/2
        alpha, gamma = (rj*rj).sum(-1), (ri*ri).sum(-1)
        beta = (ri*rj).sum(-1)
        cross = p[2:,2:]-p[2:,:-2]-p[:-2,2:]+p[:-2,:-2]
        denominator = 2*(alpha+gamma)
        if np.any(denominator<=0):
            break
        target = (alpha[...,None]*(p[2:,1:-1]+p[:-2,1:-1]) + gamma[...,None]*(p[1:-1,2:]+p[1:-1,:-2]) - .5*beta[...,None]*cross)/denominator[...,None]
        old_q = corner_scaled_jacobians(p).min(-1)
        for damping in (.4,.2,.1,.05,.025):
            trial = p.copy()
            trial[1:-1,1:-1] += damping*(target-p[1:-1,1:-1])
            q = corner_scaled_jacobians(trial).min(-1)
            if np.isfinite(q).all() and q.min()>1e-10 and q.min()>=.5*old_q.min() and np.quantile(q,.1)>=np.quantile(old_q,.1)-1e-12:
                p, accepted = trial, accepted+1
                break
        else:
            break
    return p, accepted
