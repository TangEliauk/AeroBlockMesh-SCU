"""Read exported SU2 quads independently; inspect connectivity and plot the CAD blocks."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def read_su2(path, expected_markers=("airfoil", "farfield")):
    with path.open(encoding="utf-8") as f:
        lines = iter(line.strip() for line in f if line.strip() and not line.lstrip().startswith("%"))

        def field(name):
            key, value = next(lines).split("=", 1)
            if key.strip() != name:
                raise ValueError(f"expected {name}, encountered {key}")
            return value.strip()

        def elements(n, kind, size):
            result = []
            for _ in range(n):
                row = [int(v) for v in next(lines).split()]
                if row[0] != kind or len(set(row[1:size + 1])) != size:
                    raise ValueError("unexpected element type or duplicate node")
                result.append(row[1:size + 1])
            return np.asarray(result, dtype=int)

        if int(field("NDIME")) != 2:
            raise ValueError("requires a 2-D mesh")
        cells = elements(int(field("NELEM")), 9, 4)
        points = []
        for i in range(int(field("NPOIN"))):
            row = next(lines).split()
            if len(row) > 2 and int(row[2]) != i:
                raise ValueError("unexpected node numbering in Gmsh SU2 export")
            points.append([float(v) for v in row[:2]])
        markers = {}
        for _ in range(int(field("NMARK"))):
            name = field("MARKER_TAG")
            if name in markers:
                raise ValueError("repeated boundary marker")
            markers[name] = elements(int(field("MARKER_ELEMS")), 3, 2)
    if set(markers) != set(expected_markers):
        raise ValueError(f"unexpected boundary markers {list(markers)}")
    return np.asarray(points), cells, markers


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


DEFAULT_SAMPLES = Path(__file__).resolve().parents[1] / "data/processed/uiuc_v1/samples.jsonl"


def contour_deviation(name, mesh_wall, path=DEFAULT_SAMPLES):
    name = name.casefold().removesuffix(".dat")
    with path.open(encoding="utf-8") as f:
        sample = next((row for row in map(json.loads, f)
                       if name == row["sample_id"] or
                       name.lower() in (x.lower() for x in row["aliases"])), None)
    if sample is None:
        return {}
    p = np.asarray(sample["polygon"])
    source = np.stack([p, np.roll(p, -1, axis=0)], axis=1)

    def distance(first, second):
        queries = np.concatenate((first[:, 0], first[:, 1], first.mean(axis=1)))
        delta = np.diff(second, axis=1)[:, 0]
        disp = queries[:, None] - second[None, :, 0]
        t = np.clip((disp * delta).sum(axis=2) / (delta * delta).sum(axis=1), 0., 1.)
        return float(np.linalg.norm(disp - t[..., None] * delta, axis=2).min(axis=1).max())

    return {"source_id": sample["sample_id"], "source_points": len(p),
            "sampled_wall_deviation_chords": max(distance(mesh_wall, source), distance(source, mesh_wall))}


def inspect(points, cells, markers):
    if set(np.unique(cells)) != set(range(len(points))) or not np.isfinite(points).all():
        raise ValueError("invalid or unused points")
    sides = defaultdict(list)
    for cid, q in enumerate(cells):
        for i in range(4):
            sides[tuple(sorted((q[i], q[(i+1) % 4])))].append((cid, i))
    outside = {edge for edge, owners in sides.items() if len(owners) == 1}
    boundary = [tuple(sorted(e)) for m in markers.values() for e in m]
    if set(boundary) != outside or len(boundary) != len(outside):
        raise ValueError("boundary markers must cover all exterior edges exactly once")
    if max(map(len, sides.values())) > 2 or len(points) - len(sides) + len(cells) != 0:
        raise ValueError("unexpected edge connectivity for an annular domain")
    adjacency = [[] for _ in cells]
    for owners in sides.values():
        if len(owners) == 2:
            (u, _), (v, _) = owners
            adjacency[u].append(v)
            adjacency[v].append(u)
    pending, touched = [0], {0}
    while pending:
        for other in adjacency[pending.pop()]:
            if other not in touched:
                touched.add(other)
                pending.append(other)
    if len(touched) != len(cells):
        raise ValueError("disconnected fluid cells")
    loops = []
    for perimeter in markers.values():
        neighbors = defaultdict(set)
        for u, v in perimeter:
            neighbors[u].add(v)
            neighbors[v].add(u)
        if not neighbors or any(len(adj) != 2 for adj in neighbors.values()):
            raise ValueError("open or branching physical boundary")
        start = next(iter(neighbors))
        visited = {start}
        pending = [start]
        while pending:
            new_nodes = neighbors[pending.pop()] - visited
            visited.update(new_nodes)
            pending.extend(new_nodes)
        if len(visited) != len(neighbors):
            raise ValueError("a physical marker contains multiple boundary loops")
        loops.append(visited)
    if not loops[0].isdisjoint(loops[1]):
        raise ValueError("wall and farfield touch")
    xy = points[cells]
    edges = np.roll(xy, -1, axis=1) - xy
    length = np.linalg.norm(edges, axis=-1)
    if np.any(length == 0):
        raise ValueError("zero length edge")
    direction = edges / length[..., None]
    jac = cross(np.roll(direction, 1, axis=1), direction)
    if np.any(jac <= 0):
        raise ValueError(f"nonpositive corner Jacobian, minimum {jac.min():.4g}")
    angle, height = [], []
    for seg in markers["airfoil"]:
        [(cid, idx)] = sides[tuple(sorted(seg))]
        local = direction[cid]
        sine = min(abs(cross(local[idx], local[(idx-1) % 4])),
                   abs(cross(local[idx], local[(idx+1) % 4])))
        angle.append(np.degrees(np.arcsin(np.clip(sine, 0, 1))))
        height.append(min(abs(cross(local[idx], edges[cid, (idx-1) % 4])),
                          abs(cross(local[idx], edges[cid, (idx+1) % 4]))))
    return {"points": len(points), "quads": len(cells),
            "marker_edges": {k: len(v) for k,v in markers.items()},
            "min_scaled_jacobian": float(jac.min()),
            "q05_scaled_jacobian": float(np.quantile(jac.min(axis=1), .05)),
            "wall_angle_min_deg": float(min(angle)),
            "wall_angle_q10_deg": float(np.quantile(angle, .1)),
            "wall_angle_median_deg": float(np.median(angle)),
            "first_layer_normal_height_median_chords": float(np.median(height))}


def draw(folder, name, points, cells, markers, report):
    import gmsh
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection

    blocks = []
    curve_owners = defaultdict(list)
    gmsh.initialize(["gmsh", "-nopopup"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(folder / "mesh.msh"))
        tags, coord, _ = gmsh.model.mesh.getNodes()
        lookup = dict(zip(tags, np.asarray(coord).reshape(-1, 3)[:, :2]))
        for _, surf in gmsh.model.getEntities(2):
            tags, node_ids = gmsh.model.mesh.getElementsByType(3, surf)
            if len(tags):
                blocks.append(np.array([lookup[n] for n in node_ids]).reshape(-1, 4, 2))
                for _, curve in gmsh.model.getBoundary([(2, surf)], oriented=False):
                    curve_owners[curve].append(surf)
    finally:
        gmsh.finalize()
    if sum(map(len, blocks)) != len(cells):
        raise ValueError("msh block element count disagrees with su2")
    if any(len(owners) > 2 for owners in curve_owners.values()):
        raise ValueError("more than two CAD surfaces share an interface")
    report["internal_interfaces"] = [{"curve": curve, "blocks": owners}
                                     for curve, owners in sorted(curve_owners.items())
                                     if len(owners) == 2]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
    colors = plt.get_cmap("tab20")
    for i, block in enumerate(blocks):
        ax[0].add_collection(PolyCollection(block, facecolors=[colors(i % 20)], edgecolors="none"))
    ax[0].autoscale()
    ax[0].set_title(f"{len(blocks)} geometric surfaces")
    xmin, xmax = points[markers["airfoil"], 0].min(), points[markers["airfoil"], 0].max()
    ymin, ymax = points[markers["airfoil"], 1].min(), points[markers["airfoil"], 1].max()
    ax[1].add_collection(PolyCollection(points[cells], facecolors="none", edgecolors="#516777", linewidths=.3))
    ax[1].add_collection(LineCollection(points[markers["airfoil"]], colors="#172534", linewidths=1.2))
    ax[1].set(xlim=(xmin-.06, xmax+.08), ylim=(ymin-.10, ymax+.10))
    ax[1].set_title(f"Near-wall grid   |   {len(cells):,} quads")
    for axis in ax:
        axis.set_aspect("equal")
        axis.set_xlabel("x/c")
        axis.set_ylabel("y/c")
    fig.suptitle(f"{name}   /   Gmsh exterior grid")
    fig.savefig(folder / "mesh_overview.png", dpi=175)
    plt.close(fig)
    report["gmsh_surfaces"] = len(blocks)


def audit(folder, name, samples=DEFAULT_SAMPLES):
    points, cells, markers = read_su2(folder / "mesh.su2")
    report = inspect(points, cells, markers)
    report.update(contour_deviation(name, points[markers["airfoil"]], samples))
    draw(folder, name, points, cells, markers, report)
    output = json.dumps(report, indent=2)
    (folder / "mesh_audit.json").write_text(output, encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--airfoil", help="UIUC ID or alias (default: folder name)")
    args = parser.parse_args()
    print(json.dumps(audit(args.folder, args.airfoil or args.folder.name), indent=2))


if __name__ == "__main__":
    main()
