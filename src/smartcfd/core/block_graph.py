"""Conforming planar quad complexes. A negative curve ID reverses its direction."""

from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import pairwise
from types import MappingProxyType
from typing import Mapping

import numpy as np
from shapely import LineString, MultiPoint, Polygon, STRtree

Point = tuple[float, float]


@dataclass(frozen=True)
class Curve:
    start: int
    end: int
    stations: tuple[Point, ...] = ()  # Interior mesh stations in canonical order
    kind: str = "line"
    boundary: str = ""
    role: str = ""

    def __post_init__(self):
        object.__setattr__(self, "stations", tuple(tuple(xy) for xy in self.stations))

    @property
    def intervals(self):
        return len(self.stations) + 1


@dataclass(frozen=True)
class Block:
    edges: tuple[int, int, int, int]  # Counterclockwise closed walk
    role: str = ""

    def __post_init__(self):
        object.__setattr__(self, "edges", tuple(self.edges))


@dataclass(frozen=True)
class BlockGraph:
    vertices: Mapping[int, Point]
    curves: Mapping[int, Curve]
    blocks: Mapping[int, Block]

    def __post_init__(self):
        object.__setattr__(self, "vertices", MappingProxyType({v: tuple(p) for v, p in self.vertices.items()}))
        for name in ("curves", "blocks"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    def ends(self, edge):
        curve = self.curves[abs(edge)]
        return (curve.start, curve.end) if edge > 0 else (curve.end, curve.start)

    def points(self, edge):
        c = self.curves[abs(edge)]
        xy = np.asarray((self.vertices[c.start], *c.stations, self.vertices[c.end]), dtype=float)
        return xy if edge > 0 else xy[::-1]

    def owners(self):
        adjacency = defaultdict(list)
        for bid, block in self.blocks.items():
            for index, edge in enumerate(block.edges):
                adjacency[abs(edge)].append((bid, index))
        return adjacency

    def boundary_loops(self):
        """Return fluid-on-left loops; signed area distinguishes the exterior from holes."""
        owners = self.owners()
        outgoing = {}
        for cid, curve in self.curves.items():
            if curve.boundary:
                (bid, side), = owners[cid]
                edge = self.blocks[bid].edges[side]
                start, _ = self.ends(edge)
                if start in outgoing:
                    raise ValueError("Multiple outgoing boundary curves at one anchor")
                outgoing[start] = edge
        loops = []
        while outgoing:
            start = next(iter(outgoing))
            v = start
            edges = []
            while v in outgoing:
                edge = outgoing.pop(v)
                edges.append(edge)
                _, v = self.ends(edge)
            if v != start:
                raise ValueError("Boundary loop does not close")
            xy = np.concatenate([self.points(e)[:-1] for e in edges])
            area = float(np.sum(xy[:, 0]*np.roll(xy[:, 1], -1) - xy[:, 1]*np.roll(xy[:, 0], -1))/2)
            labels = sorted({self.curves[abs(e)].boundary for e in edges})
            loops.append({"kind": "outer" if area > 0 else "hole", "labels": labels,
                          "edges": edges, "signed_area": area})
        return loops

    @property
    def cells(self):
        return sum(self.curves[abs(b.edges[0])].intervals * self.curves[abs(b.edges[1])].intervals
                   for b in self.blocks.values())

    def to_dict(self):
        return {"version": 1, "vertices": dict(self.vertices),
                "curves": {k: asdict(v) for k, v in self.curves.items()},
                "blocks": {k: asdict(v) for k, v in self.blocks.items()}}

    @classmethod
    def from_dict(cls, data, *, validate=True):
        """Decode a saved graph. Rejected candidates can be inspected with validate=False."""
        if data["version"] != 1:
            raise ValueError("Unsupported BlockGraph version")
        graph = cls({int(k): tuple(v) for k, v in data["vertices"].items()},
                    {int(k): Curve(**{**v, "stations": tuple(map(tuple, v["stations"]))})
                     for k, v in data["curves"].items()},
                    {int(k): Block(tuple(v["edges"]), v["role"]) for k, v in data["blocks"].items()})
        if validate:
            graph.validate()
        return graph

    def validate(self, *, previous_embedding=None):
        """Check incidence, integer budgets, coverage within the boundary and planar embedding."""
        if not self.blocks or any(type(k) is not int or k < 1 for table in
                                  (self.vertices, self.curves, self.blocks) for k in table):
            raise ValueError("Expected positive integer entity IDs and at least one block")
        ends = {v for c in self.curves.values() for v in (c.start, c.end)}
        if ends != set(self.vertices) or len(set(self.vertices.values())) != len(ends):
            raise ValueError("Unknown, coincident or unused anchors")
        owners = self.owners()
        if set(owners) != set(self.curves):
            raise ValueError("Unknown or unused curve")
        lines = {}
        previous_lines, previous_polys = previous_embedding or ({}, {})
        next_lines, next_polys = {}, {}
        changed_lines, changed_polys = [], []
        boundary_degree = defaultdict(int)
        neighbors = defaultdict(set)
        for cid, c in self.curves.items():
            token=(c,self.vertices[c.start],self.vertices[c.end]) if previous_embedding is not None else None
            prior=previous_lines.get(cid)
            if prior is not None and prior[0]==token:
                line=prior[1]
            else:
                xy = self.points(cid)
                if (xy.shape != (c.intervals + 1, 2) or not np.isfinite(xy).all() or c.start == c.end
                        or (np.linalg.norm(np.diff(xy, axis=0), axis=1) == 0).any()):
                    raise ValueError(f"Degenerate curve {cid}")
                line = LineString(xy)
                if not line.is_simple:
                    raise ValueError(f"Self-intersection on curve {cid}")
                changed_lines.append(len(lines))
            if previous_embedding is not None:
                next_lines[cid]=(token,line)
            lines[cid] = line
            uses = owners[cid]
            if len(uses) != (1 if c.boundary else 2):
                raise ValueError(f"Curve {cid} has incompatible incidence and boundary tag")
            if c.boundary:
                boundary_degree[c.start] += 1
                boundary_degree[c.end] += 1
            else:
                (a, i), (b, j) = uses
                if a == b or self.blocks[a].edges[i] != -self.blocks[b].edges[j]:
                    raise ValueError(f"Inconsistent interface orientation on curve {cid}")
                neighbors[a].add(b)
                neighbors[b].add(a)
        if not boundary_degree or any(n != 2 for n in boundary_degree.values()):
            raise ValueError("Open or branching boundary")
        found = set()
        pending = [next(iter(self.blocks))]
        while pending:
            bid = pending.pop()
            if bid not in found:
                found.add(bid)
                pending.extend(neighbors[bid] - found)
        if len(found) != len(self.blocks):
            raise ValueError("Disconnected block complex")
        polygons = []
        for bid, block in self.blocks.items():
            edges = block.edges
            if len(edges) != 4 or len({self.ends(e)[0] for e in edges}) != 4:
                raise ValueError(f"Block {bid} needs four different anchors")
            for e, nxt in pairwise((*edges, edges[0])):
                if self.ends(e)[1] != self.ends(nxt)[0]:
                    raise ValueError(f"Open block {bid}")
            n = [self.curves[abs(e)].intervals for e in edges]
            if n[:2] != n[2:]:
                raise ValueError(f"Unequal opposite interval counts on block {bid}")
            token=(edges,tuple(next_lines[abs(e)][0] for e in edges)) if previous_embedding is not None else None
            prior=previous_polys.get(bid)
            if prior is not None and prior[0]==token:
                poly=prior[1]
            else:
                poly = Polygon(np.concatenate([self.points(e)[:-1] for e in edges]))
                if not poly.is_valid or not poly.exterior.is_ccw or poly.area <= 0:
                    raise ValueError(f"Folded or clockwise block {bid}")
                changed_polys.append(len(polygons))
            if previous_embedding is not None:
                next_polys[bid]=(token,poly)
            polygons.append(poly)
        keys = list(lines)
        geometries = list(lines.values())
        tree = STRtree(geometries)
        def new_pairs(tree,geoms,changed):
            if not changed:
                return
            fresh=set(changed)
            for offset,j in tree.query([geoms[i] for i in changed],predicate="intersects").T:
                i=changed[offset]
                if j not in fresh or i<j:
                    yield i,j
        for i,j in new_pairs(tree,geometries,changed_lines):
            a, b = keys[i], keys[j]
            common = set(self.ends(a)) & set(self.ends(b))
            allowed = MultiPoint([self.vertices[v] for v in common])
            if not geometries[i].intersection(geometries[j]).difference(allowed).is_empty:
                raise ValueError(f"Curves {a} and {b} intersect away from shared anchors")
        for i,j in new_pairs(STRtree(polygons),polygons,changed_polys):
            if polygons[i].relate_pattern(polygons[j], "T********"):
                raise ValueError("Overlapping block interiors")
        if previous_embedding is not None:
            previous_embedding[:]=[next_lines,next_polys]
