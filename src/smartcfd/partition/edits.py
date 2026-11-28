"""Integer-preserving strip cuts in an oriented quad complex."""

from collections import deque
from dataclasses import dataclass, replace

import numpy as np
from shapely import LineString, Point

from smartcfd.core.block_graph import Block, BlockGraph, Curve
from smartcfd.core.coons import coons, fractions


@dataclass(frozen=True)
class Cut:
    block: int
    side: int
    interval: int


def plan_strip(graph, cut):
    """Return canonical cut indices and the oriented local cut on each affected block."""
    if cut.block not in graph.blocks or cut.side not in range(4) or type(cut.interval) is not int:
        raise ValueError("Unknown block, side or interval")
    owners = graph.owners()
    splits, blocks = {}, {}
    pending = deque([cut])
    while pending:
        step = pending.popleft()
        edges = graph.blocks[step.block].edges
        i, k = step.side, step.interval
        n = graph.curves[abs(edges[i])].intervals
        if not 0 < k < n:
            raise ValueError("Both children must retain at least one interval")
        for side, index in ((i, k), ((i + 2) % 4, n - k)):
            edge = edges[side]
            canonical = index if edge > 0 else n - index
            existing = splits.get(abs(edge))
            if existing is not None and existing != canonical:
                raise ValueError("Strip closes with conflicting cut positions")
            if existing is None:
                splits[abs(edge)] = canonical
                for other, other_side in owners[abs(edge)]:
                    if other != step.block:
                        pending.append(Cut(other, other_side, n - index))
        prior = blocks.setdefault(step.block, step)
        if prior.side % 2 != i % 2:
            raise ValueError("Strip intersects itself")
    return splits, blocks


def candidates(graph):
    """One midpoint candidate per feasible strip, including closed rings."""
    seen = set()
    for bid, block in graph.blocks.items():
        for side in (0, 1):
            n = graph.curves[abs(block.edges[side])].intervals
            cut = Cut(bid, side, n // 2)
            try:
                split, _ = plan_strip(graph, cut)
            except ValueError:
                continue
            key = frozenset(split)
            if key not in seen:
                seen.add(key)
                yield cut


def describe_candidates(graph):
    """Expose the affected blocks and both possible reference sides of a closed belt."""
    owners = graph.owners()
    for cut in candidates(graph):
        crossed, affected = plan_strip(graph, cut)
        belt = len(affected) >= 3 and all(len(owners[e]) == 2 for e in crossed)
        # A propagated split is also a ring cut when its strip closes.
        sides = [(cut.side-1) % 4, (cut.side+1) % 4] if belt else []
        yield {**vars(cut), "affected_blocks": sorted(affected), "crossed_curves": sorted(crossed),
               "closed_belt_sides": sides}


def strip_curve(graph, edges, k, mode):
    bottom, right, top, left = (graph.points(e) for e in edges)
    top, left = top[::-1], left[::-1]
    p, q = bottom[k], top[k]
    if mode == "inherited":
        xy = coons(bottom, right, top, left, columns=slice(k, k+1))[:, 0]
    elif mode == "normal":
        v = ((fractions(left) + fractions(right)) / 2)[:, None]
        def inward(xy):
            tangent = xy[k+1] - xy[k-1]
            normal = np.array([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)
            return normal if np.dot(normal, q-p) >= 0 else -normal
        length = np.linalg.norm(q-p) / 3
        a, b = p + length * inward(bottom), q - length * inward(top)
        xy = (1-v)**3*p + 3*(1-v)**2*v*a + 3*(1-v)*v**2*b + v**3*q
    else:
        raise ValueError(f"Unknown interface guide {mode}")
    xy[0], xy[-1] = p, q
    return xy


def fit_streamline(trace, p, q, positions):
    """Sample a traced field line, with a smooth correction to its target anchor."""
    trace = np.asarray(trace, dtype=float)
    if (trace.ndim != 2 or trace.shape[1] != 2 or len(trace) < 2 or not np.isfinite(trace).all()
            or np.any(np.linalg.norm(np.diff(trace, axis=0), axis=1) == 0)):
        raise ValueError("Expected a finite nondegenerate field streamline")
    if not np.allclose(trace[0], p, atol=1e-10, rtol=0):
        raise ValueError("Trace must start at the selected anchor")
    t = np.asarray(positions)
    arc = fractions(trace)
    xy = np.column_stack([np.interp(t, arc, axis) for axis in trace.T])
    xy += (t*t*(3-2*t))[:, None] * (q - trace[-1])
    xy[0], xy[-1] = p, q
    return xy


def split(graph, cut, *, guide="inherited", traces=None, avoid=()):
    """Return a new graph. Optional per-block field traces and exclusion discs guide the cut."""
    graph.validate()
    split_at, affected = plan_strip(graph, cut)
    traces = traces or {}
    avoid = tuple(avoid)
    if any(not np.isfinite(radius) or radius < 0 or np.shape(center) != (2,)
           or not np.isfinite(center).all() for center, radius in avoid):
        raise ValueError("Exclusion discs need a finite center and nonnegative radius")
    if set(traces) - set(affected):
        raise ValueError("Trace targets a block outside the cut")
    vertices, curves, blocks = dict(graph.vertices), dict(graph.curves), dict(graph.blocks)
    next_curve = max(curves) + 1
    halves, anchors = {}, {}
    for cid, k in split_at.items():
        c = curves.pop(cid)
        xy = graph.points(cid)
        v = max(vertices) + 1
        vertices[v] = tuple(xy[k])
        anchors[cid] = v
        a, b = next_curve, next_curve + 1
        next_curve += 2
        curves[a] = replace(c, end=v, stations=c.stations[:k-1])
        curves[b] = replace(c, start=v, stations=c.stations[k:])
        halves[cid], halves[-cid] = (a, b), (-b, -a)
    next_block = max(blocks) + 1
    for bid, step in affected.items():
        block = blocks.pop(bid)
        edges = block.edges[step.side:] + block.edges[:step.side]
        e, f, g, h = edges
        xy = strip_curve(graph, edges, step.interval, guide)
        kind = guide
        if bid in traces:
            xy = fit_streamline(traces[bid], xy[0], xy[-1], fractions(xy))
            kind = "streamline"
        line = LineString(xy)
        if any(line.distance(Point(center)) <= radius for center, radius in avoid):
            raise ValueError(f"Proposed interface enters an excluded singularity region in block {bid}")
        a, b = halves[e]
        c, d = halves[g]
        mid = next_curve
        next_curve += 1
        curves[mid] = Curve(anchors[abs(e)], anchors[abs(g)], tuple(map(tuple, xy[1:-1])), kind)
        blocks[bid] = Block((a, mid, d, h), block.role)
        blocks[next_block] = Block((b, f, c, -mid), block.role)
        next_block += 1
    result = BlockGraph(vertices, curves, blocks)
    result.validate()
    if result.cells != graph.cells:
        raise ValueError("Strip cut changed the cell budget")
    return result


def insert_ring(graph, block, wall_side, interval, **guidance):
    """Split a closed single block belt around the selected wall or internal ring."""
    if block not in graph.blocks or wall_side not in range(4):
        raise ValueError("Unknown belt seed")
    cut = Cut(block, (wall_side + 1) % 4, interval)
    crossed, affected = plan_strip(graph, cut)
    owners = graph.owners()
    if any(len(owners[e]) != 2 for e in crossed):
        raise ValueError("Selected block strip ends at a boundary and is not a ring")
    if len(affected) < 3:
        raise ValueError("Expected a closed annular belt")
    return split(graph, cut, **guidance)
