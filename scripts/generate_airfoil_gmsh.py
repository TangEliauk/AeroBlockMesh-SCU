"""Generate a conforming, viscous O-grid for a cleaned UIUC airfoil using Gmsh."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import gmsh
import numpy as np


REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "processed" / "uiuc_v1"
FARFIELD_RADIUS = 18.0
FIRST_LAYER = 1.0e-5
RING_DISTANCES = (0.0012, 0.024)
RADIAL_CELLS = (24, 25, 66)
COARSE_RADIAL_CELLS = (18, 16, 43)
OUTER_SPLIT_FRACTION = 0.045


@dataclass(frozen=True)
class Topology:
    """Optional block cuts, given by chord fraction on each airfoil branch."""

    lower_splits: tuple[float, ...] = ()
    upper_splits: tuple[float, ...] = ()
    extra_ring: bool = False


@dataclass(frozen=True)
class MeshSummary:
    quads: int
    blocks: int
    min_jacobian: float
    min_scaled_jacobian: float
    q05_scaled_jacobian: float
    wall_angle_q10_deg: float


def unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def load_outline(airfoil: str, data: Path) -> np.ndarray:
    query = airfoil.casefold().removesuffix(".dat")
    matches = []
    with (data / "samples.jsonl").open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            names = [row["sample_id"], *row["aliases"]]
            if query in {name.casefold() for name in names}:
                matches.append(row)
    if not matches:
        raise ValueError(f"Cleaned UIUC outline for {airfoil!r} was not found in {data}")
    if len(matches) != 1:
        choices = ", ".join(row["sample_id"] for row in matches)
        raise ValueError(f"Ambiguous UIUC alias {airfoil!r}; use one of {choices}")
    row = matches[0]
    if row["orientation"] != "CCW" or row["canonical_start"] != "lexicographic_minimum_x_then_y":
        raise ValueError("The cleaned airfoil must start at its nose and wind counterclockwise")
    outline = np.asarray(row["polygon"], dtype=float)
    if not np.all(np.isfinite(outline)) or outline[0, 0] != 0.0 or outline[:, 0].max() != 1.0:
        raise ValueError("The cleaned outline must have finite coordinates and unit chord")
    return np.vstack((outline, outline[0]))


def trailing_edge(outline: np.ndarray) -> tuple[int, int]:
    tip = np.flatnonzero(outline[:, 0] == outline[:, 0].max())
    lower, upper = int(tip[0]), int(tip[-1])
    if len(tip) != upper - lower + 1:
        raise ValueError("Expected one contiguous trailing edge on the closed outline")
    if lower == upper:
        # A blunt cap can be slightly inclined relative to the normal of the chord.
        # Include a near-tip neighboring panel when it is steep and ends at the tip.
        for direction in (-1, 1):
            neighbor = lower + direction
            dx, dy = outline[neighbor] - outline[lower]
            if abs(dx) <= 0.005 and abs(dy) > 2 * abs(dx):
                return min(lower, neighbor), max(lower, neighbor)
    return lower, upper


def sector_layout(outline: np.ndarray, trailing: tuple[int, int], topology: Topology,
                  coarse: bool) -> tuple[list[int], list[int]]:
    lower, upper = trailing
    if lower < 2 or len(outline) - upper < 3:
        raise ValueError("Expected upper and lower branches from the nose to the trailing edge")
    lower_targets = (0.028, 0.32, 0.72, 0.94, 0.996)
    upper_targets = tuple(reversed(lower_targets))
    lower_cuts = [int(np.abs(outline[:lower, 0] - target).argmin()) for target in lower_targets]
    upper_cuts = [upper + 1 + int(np.abs(outline[upper + 1 : -1, 0] - target).argmin())
                  for target in upper_targets]
    cuts = [0, *lower_cuts, lower]
    counts = [18, 38, 44, 32, 18, 10]
    if lower != upper:
        cuts.append(upper)
        counts.append(8)
    cuts.extend([*upper_cuts, len(outline) - 1])
    counts.extend([10, 18, 32, 44, 38, 18])

    # Sparse outlines can put more than one target at the same sample. Keep each
    # geometric cut once and concentrate cells on the nondegenerate sectors.
    if np.any(np.diff(cuts) < 0):
        raise ValueError("Sectors must follow the airfoil perimeter in order")
    base_sectors = [(a, b, count // 2 if coarse else count)
                    for a, b, count in zip(cuts[:-1], cuts[1:], counts) if b > a]

    split_indices = set()
    for fractions, branch in ((topology.lower_splits, range(1, lower)),
                              (topology.upper_splits, range(upper + 1, len(outline) - 1))):
        for fraction in fractions:
            if not 0 < fraction < 1:
                raise ValueError(f"Split fraction {fraction!r} must be between zero and one")
            eligible = [index for index in branch if index not in cuts]
            if not eligible:
                raise ValueError(f"No interior point is available on the branch for split {fraction}")
            index = min(eligible, key=lambda candidate: abs(outline[candidate, 0] - fraction))
            if index in split_indices:
                raise ValueError(f"Multiple split fractions select the same outline point at x={outline[index, 0]:.6g}")
            split_indices.add(index)

    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(outline, axis=0), axis=1))]
    refined_cuts = [0]
    refined_counts = []
    for start, end, count in base_sectors:
        boundaries = [start, *sorted(idx for idx in split_indices if start < idx < end), end]
        pieces = len(boundaries) - 1
        if 2 * pieces > count:
            raise ValueError(f"Sector {start}:{end} has insufficient cells for {pieces} blocks")
        weights = np.diff(arc[boundaries]) / (arc[end] - arc[start])
        fractions = (count - 2 * pieces) * weights
        assigned = np.floor(fractions).astype(int) + 2
        order = np.argsort(-(fractions % 1), kind="stable")
        assigned[order[: count - int(assigned.sum())]] += 1
        refined_cuts.extend(boundaries[1:])
        refined_counts.extend(assigned.tolist())
    return refined_cuts, refined_counts


def planned_quad_budget(airfoil: str, data: Path = DATA, *, coarse: bool = False) -> int:
    """Return the O-grid cell budget without running Gmsh or checking mesh validity."""
    outline = load_outline(airfoil, data)
    _, tangential_cells = sector_layout(outline, trailing_edge(outline), Topology(), coarse)
    radial_cells = COARSE_RADIAL_CELLS if coarse else RADIAL_CELLS
    return sum(tangential_cells) * sum(radial_cells)


def ring_geometry(outline: np.ndarray, trailing: tuple[int, int],
                  extra_ring: bool) -> tuple[np.ndarray, np.ndarray]:
    tangent = unit(np.diff(outline, axis=0))
    panel_normal = tangent[:, [1, 0]] * (1.0, -1.0)
    normal = unit(panel_normal + np.roll(panel_normal, 1, axis=0))
    normal = np.vstack((normal, normal[0]))

    center = outline[0] + (0.5, 0.0)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(outline, axis=0), axis=1))]
    lower, upper = trailing
    cap_angle = 0.18 if lower != upper else 0.0

    # Widen the leading and trailing sectors. A linear angle would crowd near-TE
    # connectors around the wake axis and can let neighboring connectors cross.
    lower_arc = arc[: lower + 1] / arc[lower]
    upper_arc = (arc[upper:] - arc[upper]) / (arc[-1] - arc[upper])
    theta = np.r_[
        -np.pi + (np.pi - cap_angle) * np.arccos(1 - 2 * lower_arc) / np.pi,
        np.linspace(-cap_angle, cap_angle, upper - lower + 1)[1:-1],
        cap_angle + (np.pi - cap_angle) * np.arccos(1 - 2 * upper_arc) / np.pi,
    ]
    if lower == upper:
        theta = np.delete(theta, lower)
    farfield = center + FARFIELD_RADIUS * np.column_stack((np.cos(theta), np.sin(theta)))
    farfield[-1] = farfield[0]
    outward = unit(farfield - outline)

    near = outline + RING_DISTANCES[0] * normal
    transition = outline + RING_DISTANCES[1] * unit(normal + outward)
    rings = [outline, near, transition]
    if extra_ring:
        rings.append(transition + OUTER_SPLIT_FRACTION * (farfield - transition))
    return np.stack([*rings, farfield]), center


def progression(length: float, first: float, count: int) -> tuple[float, float]:
    """Find the geometric growth factor joining the specified layer heights and length."""
    powers = np.arange(count)
    low, high = 0.0, 2.0
    if not first < length < first * np.sum(high**powers):
        raise ValueError("Layer height, endpoint, and number of cells are incompatible")
    for _ in range(55):
        ratio = (low + high) / 2
        if first * np.sum(ratio**powers) < length:
            low = ratio
        else:
            high = ratio
    ratio = (low + high) / 2
    return ratio, first * ratio ** (count - 1)


def build_model(outline: np.ndarray, *, topology: Topology = Topology(),
                coarse: bool = False) -> tuple[list[int], list[int], list[int]]:
    """Create shared CAD edges and transfinite surfaces for each circumferential sector."""
    geo = gmsh.model.geo
    mesh = geo.mesh
    trailing = trailing_edge(outline)
    cuts, tangential_cells = sector_layout(outline, trailing, topology, coarse)
    rings, center = ring_geometry(outline, trailing, topology.extra_ring)
    radial_cells = COARSE_RADIAL_CELLS if coarse else RADIAL_CELLS
    if topology.extra_ring:
        outer = radial_cells[-1]
        inner_cells = round(0.60 * outer)
        radial_cells = (*radial_cells[:-1], inner_cells, outer - inner_cells)

    def point(xy: np.ndarray) -> int:
        return geo.addPoint(float(xy[0]), float(xy[1]), 0.0)

    ring_curves = []
    ring_corners = [[point(ring[index]) for index in cuts[:-1]] for ring in rings]
    for corners in ring_corners:
        corners.append(corners[0])
    center_tag = point(center)
    for level, (ring, corners) in enumerate(zip(rings, ring_corners)):
        curves = []
        for sector, (start, end) in enumerate(zip(cuts[:-1], cuts[1:])):
            endpoints = [corners[sector], corners[sector + 1]]
            if level == len(rings) - 1:
                curves.append(geo.addCircleArc(endpoints[0], center_tag, endpoints[1]))
            elif end == start + 1:
                curves.append(geo.addLine(*endpoints))
            else:
                interior = [point(xy) for xy in ring[start + 1 : end]]
                curves.append(geo.addSpline([endpoints[0], *interior, endpoints[1]]))
        ring_curves.append(curves)

    radial_curves = []
    radial_progressions = []
    heights = [FIRST_LAYER] * len(tangential_cells)
    for level, count in enumerate(radial_cells):
        curves = []
        ratios = []
        for sector, index in enumerate(cuts[:-1]):
            curves.append(geo.addLine(ring_corners[level][sector], ring_corners[level + 1][sector]))
            length = float(np.linalg.norm(rings[level + 1, index] - rings[level, index]))
            ratio, last = progression(length, heights[sector], count)
            ratios.append(ratio)
            heights[sector] = last * ratio
        radial_curves.append(curves)
        radial_progressions.append(ratios)

    surfaces = []
    for level, radials in enumerate(radial_curves):
        for sector in range(len(tangential_cells)):
            next_sector = (sector + 1) % len(tangential_cells)
            edges = [radials[sector], ring_curves[level + 1][sector],
                     -radials[next_sector], -ring_curves[level][sector]]
            surfaces.append(geo.addPlaneSurface([geo.addCurveLoop(edges)]))

    for curves in ring_curves:
        for curve, count in zip(curves, tangential_cells):
            mesh.setTransfiniteCurve(curve, count + 1)
    for curves, ratios, count in zip(radial_curves, radial_progressions, radial_cells):
        for curve, ratio in zip(curves, ratios):
            mesh.setTransfiniteCurve(curve, count + 1, "Progression", ratio)
    for surface in surfaces:
        mesh.setTransfiniteSurface(surface)
        mesh.setRecombine(2, surface)
    geo.synchronize()
    return surfaces, ring_curves[0], ring_curves[-1]


def quality_summary(surfaces: list[int], wall: list[int]) -> MeshSummary:
    mesh = gmsh.model.mesh
    types, element_tags, connectivity = mesh.getElements(2)
    if list(types) != [3]:
        raise RuntimeError("Expected exclusively four-node quadrilaterals")
    determinants = mesh.getElementQualities(element_tags[0], "minDetJac")
    invalid = ~np.isfinite(determinants) | (determinants <= 0)
    if np.any(invalid):
        bad_faces = np.unique([mesh.getElement(tag)[3] for tag in element_tags[0][invalid]])
        raise RuntimeError(f"{np.count_nonzero(invalid)} invalid quadrilaterals on Gmsh surfaces "
                           f"{bad_faces.tolist()}; minimum Jacobian {determinants.min():.6g}")

    node_tags, coordinates, _ = mesh.getNodes()
    lookup = np.argsort(node_tags)
    cells = connectivity[0].reshape(-1, 4)
    indices = lookup[np.searchsorted(node_tags[lookup], cells)]
    vertices = coordinates.reshape(-1, 3)[indices, :2]
    outgoing = unit(np.roll(vertices, -1, axis=1) - vertices)
    incoming = -np.roll(outgoing, 1, axis=1)
    sines = outgoing[..., 0] * incoming[..., 1] - outgoing[..., 1] * incoming[..., 0]
    scaled = sines.min(axis=1)

    wall_tags = np.concatenate([mesh.getNodes(1, edge, includeBoundary=True, returnParametricCoord=False)[0]
                                for edge in wall])
    on_wall = np.isin(cells, wall_tags)
    wall_cell, edge = np.where(on_wall & np.roll(on_wall, -1, axis=1))
    wall_sine = np.minimum(sines[wall_cell, edge], sines[wall_cell, (edge + 1) % 4])
    wall_angles = np.rad2deg(np.arcsin(np.clip(wall_sine, -1.0, 1.0)))
    return MeshSummary(
        quads=len(cells),
        blocks=len(surfaces),
        min_jacobian=float(determinants.min()),
        min_scaled_jacobian=float(scaled.min()),
        q05_scaled_jacobian=float(np.quantile(scaled, 0.05)),
        wall_angle_q10_deg=float(np.quantile(wall_angles, 0.10)),
    )


def generate(airfoil: str, out: Path | None = None, data: Path = DATA, *,
             topology: Topology = Topology(), coarse: bool = False) -> MeshSummary:
    """Generate and evaluate a Gmsh mesh, exporting both formats if an output path is supplied."""
    outline = load_outline(airfoil, data)
    gmsh.initialize(["gmsh", "-nopopup"], readConfigFiles=False)
    try:
        gmsh.model.add(f"{airfoil}_o_grid")
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.option.setNumber("Mesh.Smoothing", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        surfaces, wall, farfield = build_model(outline, topology=topology, coarse=coarse)
        gmsh.model.addPhysicalGroup(2, surfaces, name="fluid")
        gmsh.model.addPhysicalGroup(1, wall, name="airfoil")
        gmsh.model.addPhysicalGroup(1, farfield, name="farfield")
        gmsh.model.mesh.generate(2)

        result = quality_summary(surfaces, wall)
        if out is not None:
            out.mkdir(parents=True, exist_ok=True)
            gmsh.option.setNumber("Mesh.SaveAll", 1)
            gmsh.write(str(out / "mesh.msh"))
            gmsh.option.setNumber("Mesh.SaveAll", 0)
            gmsh.write(str(out / "mesh.su2"))
        return result
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airfoil", required=True, help="Cleaned UIUC alias (e.g. n0012) or sample_id")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DATA, help="Folder with cleaned UIUC samples.jsonl")
    parser.add_argument("--lower-split", type=float, action="append", default=[],
                        help="Add an interface at this chord fraction on the lower surface. Repeatable.")
    parser.add_argument("--upper-split", type=float, action="append", default=[],
                        help="Add an interface at this chord fraction on the upper surface. Repeatable.")
    parser.add_argument("--extra-ring", action="store_true", help="Split the outside band into two block rings")
    parser.add_argument("--coarse", action="store_true", help="Use the coarser training resolution")
    args = parser.parse_args()
    topology = Topology(tuple(args.lower_split), tuple(args.upper_split), args.extra_ring)
    try:
        output = args.out.resolve()
        result = generate(args.airfoil, output, args.data.resolve(), topology=topology, coarse=args.coarse)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{args.airfoil}: {error}\n")
    print(f"{args.airfoil}: {result.blocks} blocks, {result.quads} quadrilaterals, "
          f"minimum Jacobian {result.min_jacobian:.6g}; {output}")
