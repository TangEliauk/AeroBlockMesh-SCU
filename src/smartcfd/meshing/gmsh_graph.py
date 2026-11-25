"""Exchange a BlockGraph and its conforming quad mesh with Gmsh."""

from collections import defaultdict
from contextlib import contextmanager
from uuid import uuid4

import numpy as np
from scipy.spatial import cKDTree
from shapely import LinearRing

from smartcfd.core.block_graph import Block, BlockGraph, Curve


def verify_connectivity(graph, mesh, points, cells, markers):
    """Compare oriented quads and labeled boundary edges after arbitrary Gmsh renumbering."""
    distances, ids = cKDTree(mesh.points).query(points)
    if (len(points) != len(mesh.points) or len(np.unique(ids)) != len(points)
            or max(distances) > 1e-11 * max(1., np.ptp(mesh.points, axis=0).max())):
        raise ValueError("Export changed or duplicated mesh points")

    def oriented_quads(quads):
        start = np.argmin(quads, axis=1)
        ordered = np.take_along_axis(quads, (start[:, None]+np.arange(4)) % 4, axis=1)
        return ordered[np.lexsort(ordered.T[::-1])]

    if not np.array_equal(oriented_quads(ids[cells]), oriented_quads(np.vstack(list(mesh.quads.values())))):
        raise ValueError("Export changed quadrilateral connectivity or winding")
    expected = defaultdict(list)
    for cid, curve in graph.curves.items():
        if curve.boundary:
            nodes = mesh.curves[cid]
            expected[curve.boundary].extend(zip(nodes[:-1], nodes[1:]))
    if set(markers) != set(expected):
        raise ValueError("Export changed boundary labels")
    for label, segments in markers.items():
        actual = sorted(map(tuple, np.sort(ids[segments], axis=1)))
        if actual != sorted(tuple(sorted(edge)) for edge in expected[label]):
            raise ValueError(f"Export changed edges in boundary {label}")


@contextmanager
def model_session():
    import gmsh
    owned = not gmsh.isInitialized()
    if owned:
        gmsh.initialize(["gmsh", "-nopopup"], readConfigFiles=False)
    previous = gmsh.model.getCurrent()
    verbosity = gmsh.option.getNumber("General.Verbosity")
    name = f"block_graph_{uuid4().hex}"
    gmsh.model.add(name)
    try:
        gmsh.option.setNumber("General.Verbosity", 1)
        yield
    finally:
        gmsh.model.setCurrent(name)
        gmsh.model.remove()
        if owned:
            gmsh.finalize()
        else:
            gmsh.option.setNumber("General.Verbosity", verbosity)
            gmsh.model.setCurrent(previous)


def capture(surfaces, boundaries, *, roles=None):
    """Capture the current CAD model after 1D meshing, before any surface mesh is needed."""
    import gmsh
    roles = roles or {}
    vertices, curves, blocks = {}, {}, {}
    for surface, label in surfaces.items():
        edges = [e for _, e in gmsh.model.getBoundary([(2, surface)], oriented=False)]
        if len(edges) != 4:
            raise ValueError(f"Expected four logical curves on CAD surface {surface}")
        for e in edges:
            if e in curves:
                continue
            tags, xyz, t = gmsh.model.mesh.getNodes(1, e, includeBoundary=True)
            if len(t) != len(tags) or len(t) < 2:
                raise ValueError("Capture requires parametrized 1D mesh nodes")
            order = np.argsort(t)
            tags = tags[order]
            xy = xyz.reshape(-1, 3)[order, :2]
            start, end = int(tags[0]), int(tags[-1])
            vertices[start], vertices[end] = tuple(xy[0]), tuple(xy[-1])
            curves[e] = Curve(start, end, tuple(map(tuple, xy[1:-1])), gmsh.model.getType(1, e),
                              boundaries.get(e, ""), roles.get(e, ""))
        loop = [edges.pop(0)]
        last = curves[loop[0]].end
        while edges:
            matches = [e if curves[e].start == last else -e for e in edges
                       if last in (curves[e].start, curves[e].end)]
            if len(matches) != 1:
                raise ValueError(f"Open or branching CAD surface {surface}")
            e, = matches
            edges.remove(abs(e))
            loop.append(e)
            last = curves[abs(e)].end if e > 0 else curves[-e].start
        geom = BlockGraph(vertices, curves, {})
        ring = LinearRing(np.concatenate([geom.points(e)[:-1] for e in loop]))
        if not ring.is_ccw:
            loop = [-e for e in reversed(loop)]
        blocks[surface] = Block(tuple(loop), label)
    graph = BlockGraph(vertices, curves, blocks)
    graph.validate()
    return graph


def write_mesh(graph, mesh, directory):
    """Write classified nodes and shared interfaces as Gmsh MSH and SU2."""
    import gmsh
    with model_session():
        model = gmsh.model
        def nodes(dim, tag, ids):
            ids = np.asarray(ids, dtype=int).ravel()
            if len(ids):
                xyz = np.column_stack((mesh.points[ids], np.zeros(len(ids))))
                model.mesh.addNodes(dim, tag, ids+1, xyz.ravel())

        for v, index in mesh.vertices.items():
            model.addDiscreteEntity(0, v)
            nodes(0, v, [index])
            model.mesh.addElementsByType(v, 15, [], [index+1])
        labels = defaultdict(list)
        for eid, c in graph.curves.items():
            model.addDiscreteEntity(1, eid, [c.start, c.end])
            ids = mesh.curves[eid]
            nodes(1, eid, ids[1:-1])
            model.mesh.addElementsByType(eid, 1, [], np.column_stack((ids[:-1], ids[1:])).ravel()+1)
            if c.boundary:
                labels[c.boundary].append(eid)
            if c.role:
                model.setEntityName(1, eid, c.role)
        for bid, block in graph.blocks.items():
            model.addDiscreteEntity(2, bid, list(block.edges))
            nodes(2, bid, mesh.grids[bid][1:-1,1:-1])
            model.mesh.addElementsByType(bid, 3, [], mesh.quads[bid].ravel()+1)
            model.setEntityName(2, bid, block.role or f"block_{bid}")
        for label, edges in labels.items():
            model.addPhysicalGroup(1, edges, name=label)
        model.addPhysicalGroup(2, list(graph.blocks), name="fluid")
        gmsh.write(str(directory / "mesh.msh"))
        gmsh.write(str(directory / "mesh.su2"))
