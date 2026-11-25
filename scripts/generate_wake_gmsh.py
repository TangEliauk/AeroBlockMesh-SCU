"""Build conforming transfinite C and C-H airfoil meshes with native Gmsh."""

import argparse
from dataclasses import asdict, dataclass
import json
import math
from operator import index
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import gmsh
import numpy as np

if __package__:
    from .audit_gmsh_airfoil import inspect, read_su2
else:
    from audit_gmsh_airfoil import inspect, read_su2


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/uiuc_v1"
FARFIELD_RADIUS = 16.0
FARFIELD_CENTER_X = 0.35
FIRST_LAYER = 8e-5
FIRST_WAKE_DX = 8e-4
MIN_AFT_CELLS = 24
WALL_MODES = ("legacy", "smooth", "matched", "smooth-matched", "smooth-graded", "smooth-balanced")
DEFAULT_WALL_MODE = "legacy"
FRONT_EDIT_ACTIONS = ("lower_nose", "lower_shoulder", "upper_nose", "upper_shoulder")


@dataclass(frozen=True)
class MeshCounts:
    radial: int
    front: int
    aft: int
    wake: int
    base: int

    @property
    def quads(self):
        return 2 * self.radial * (self.front + self.aft + self.wake) + self.base * self.wake


def counts_for_budget(quad_budget, has_base):
    defaults = MeshCounts(92, 64, 112, 152, 16 if has_base else 0)
    if quad_budget is None:
        return defaults
    minimum = MeshCounts(24, 16, MIN_AFT_CELLS, 24, 4 if has_base else 0)
    if quad_budget <= minimum.quads:
        return minimum

    scale = math.sqrt(quad_budget / defaults.quads)
    target = {name: val * scale for name, val in asdict(defaults).items() if val}
    wall_min = minimum.front + minimum.aft
    wall_target = target["front"] + target["aft"]
    wake_target = target["wake"]

    def error(name, count):
        return (count / target[name] - 1.0) ** 2

    def exact_counts(total):
        max_radial = (total - minimum.base * minimum.wake) // (2 * (wall_min + minimum.wake))
        radials = sorted(range(minimum.radial, max_radial + 1), key=lambda n: error("radial", n))
        if has_base:
            max_base = (total - 2 * minimum.radial * wall_min) // minimum.wake - 2 * minimum.radial
            bases = sorted(range(minimum.base, max_base + 1), key=lambda n: error("base", n))
        else:
            bases = [0]

        best = None
        best_error = math.inf
        for radial in radials:
            radial_error = error("radial", radial)
            if radial_error >= best_error:
                break
            pair = 2 * radial
            for base in bases:
                partial_error = radial_error + (error("base", base) if has_base else 0.0)
                if partial_error >= best_error:
                    break
                wake_max = (total - pair * wall_min) // (pair + base)
                divisor = math.gcd(pair, base)
                if wake_max < minimum.wake or total % divisor:
                    continue

                # With the radial and base counts fixed, valid wake counts form
                # one residue class: base * wake == total modulo 2 * radial.
                step = pair // divisor
                residue = (pow(base // divisor, -1, step) * (total // divisor)) % step if step > 1 else 0
                first = minimum.wake + (residue - minimum.wake) % step
                last = wake_max - (wake_max - residue) % step
                if first > last:
                    continue

                # Balance both wall branches and the wake against uniform
                # refinement of the reference template.
                wake_cost = 1.0 + base / pair
                relaxed = ((1 / wake_target) + 2 * wake_cost * (total / pair - wall_target) / wall_target**2) / (
                    1 / wake_target**2 + 2 * wake_cost**2 / wall_target**2
                )
                lower = first + step * math.floor((relaxed - first) / step)
                for wake in {min(last, max(first, lower)), min(last, max(first, lower + step))}:
                    wall = (total - (pair + base) * wake) // pair
                    front = round(wall * defaults.front / (defaults.front + defaults.aft))
                    front = min(wall - minimum.aft, max(minimum.front, front))
                    aft = wall - front
                    score = partial_error + error("front", front) + error("aft", aft) + error("wake", wake)
                    if score < best_error:
                        best = MeshCounts(radial, front, aft, wake, base)
                        best_error = score
        return best, best_error

    # At the minimum radial count, increasing the aft count by one changes
    # the total by 2 * minimum.radial. A nearest total exists inside this span.
    for difference in range(minimum.radial + 1):
        totals = [quad_budget] if difference == 0 else [quad_budget - difference, quad_budget + difference]
        options = [exact_counts(total) for total in totals if total >= minimum.quads]
        choices = [(counts, score) for counts, score in options if counts is not None]
        if choices:
            return min(choices, key=lambda option: option[1])[0]
    raise RuntimeError("No matching transfinite count found")


def read_profile(name, data=DATA):
    key = name.casefold()
    with (Path(data) / "samples.jsonl").open(encoding="utf-8") as f:
        rows = [row for row in map(json.loads, f)
                if key in {row["sample_id"].casefold(), *(alias.casefold() for alias in row["aliases"])}]
    if len(rows) != 1:
        raise ValueError(f"Expected one UIUC airfoil matching {name!r}, found {len(rows)}")
    sample = rows[0]
    polygon = np.asarray(sample["polygon"])
    if np.argmin(polygon[:, 0]) != 0:
        raise ValueError("Expected a counterclockwise polygon starting at the leading edge")
    tail = trailing_edge(polygon, sample["source_files"])
    lower = np.arange(tail[0] + 1)
    upper = np.r_[0, np.arange(len(polygon) - 1, tail[-1] - 1, -1)]
    if min(len(lower), len(upper)) < 6:
        raise ValueError("Both airfoil branches must have at least six input stations")
    return sample["sample_id"], polygon, (lower, upper)


def trailing_edge(polygon, sources):
    x = polygon[:, 0]
    tail = np.flatnonzero(np.isclose(x, x.max(), rtol=0, atol=1e-10))
    if len(tail) == 2 and tail[1] == tail[0] + 1:
        return tail
    if len(tail) != 1:
        raise ValueError("Trailing edge must have a single tip or one straight base edge")

    # Some open chains have a slanted base. Match the unmodified closure edge
    # using the normalized endpoint distance recorded by preprocessing.
    index = int(tail[0])
    if index == 0 or index == len(polygon) - 1:
        raise ValueError("Trailing edge must be separated from the canonical leading edge")
    candidates = np.arange(index - 1, index + 1)
    edge_lengths = np.linalg.norm(polygon[candidates + 1] - polygon[candidates], axis=1)
    matching_edges = set()
    source_flags = [source["quality_flags"] for source in sources]
    base_gaps = [flags["raw_normalized_last_to_first_distance"] for flags in source_flags
                 if flags["open_chain_has_implicit_te_segment"]]
    if not base_gaps:
        return tail
    for gap in base_gaps:
        matches = candidates[np.isclose(edge_lengths, gap, rtol=1e-8, atol=1e-10)]
        matching_edges.update(int(k) for k in matches if min(x[k], x[k + 1]) > 0.9)
    if len(matching_edges) != 1:
        raise ValueError("Could not identify a unique straight trailing-edge base from the UIUC closure")
    start = matching_edges.pop()
    return np.array([start, start + 1])


def progression(length, count, first):
    """Return a growth ratio by matching the sum of the desired cell lengths."""
    lo, hi = 1.0, 2.0
    for _ in range(48):
        ratio = (lo + hi) / 2
        if count * math.log(ratio) < math.log1p(length * (ratio - 1) / first):
            lo = ratio
        else:
            hi = ratio
    return (lo + hi) / 2


def front_cut_stations(coordinates, sign):
    delta = np.diff(coordinates, axis=0)
    tangent = delta / np.linalg.norm(delta, axis=1)[:, None]
    tangent = np.vstack((tangent[0], tangent[:-1] + tangent[1:], tangent[-1]))
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    slopes = np.degrees(np.arctan2(sign * tangent[:, 1], tangent[:, 0]))
    candidates = np.flatnonzero((coordinates[:, 0] > 0.12) & (coordinates[:, 0] < 0.6))
    if not len(candidates):
        raise ValueError("Airfoil needs intermediate stations between x/c=0.12 and 0.6")
    shoulder = candidates[np.argmin(abs(slopes[candidates]))]
    cuts = [0]
    angle_cuts = {}
    for target in (65, 36, 15):
        if cuts[-1] + 1 >= shoulder:
            break
        section = np.arange(cuts[-1] + 1, shoulder)
        idx = section[np.argmin(abs(slopes[section] - target))]
        if 0.0 < slopes[idx] < slopes[cuts[-1]] and abs(slopes[idx] - target) <= 16:
            cuts.append(int(idx))
            angle_cuts[target] = int(idx)
    cuts.append(int(shoulder))
    normals = sign * np.column_stack((-tangent[:, 1], tangent[:, 0]))
    return cuts, normals, angle_cuts


def front_cuts(coordinates, sign):
    cuts, normals, _ = front_cut_stations(coordinates, sign)
    return cuts, normals


def front_edit_layout(polygon, branches):
    """Return the compact endpoints and the available extra cuts on each side."""
    options = []
    for side, sign, branch in zip(("lower", "upper"), (-1, 1), branches):
        cuts, _, angle_cuts = front_cut_stations(polygon[branch], sign)
        fixed = [cuts[0], cuts[-1]]
        if angle_cuts:
            middle = min(angle_cuts, key=lambda target: abs(target - 36))
            fixed.append(angle_cuts.pop(middle))
        extra = {f"{side}_{label}": angle_cuts[target]
                 for label, target in (("nose", 65), ("shoulder", 15)) if target in angle_cuts}
        options.append((fixed, extra))
    return options


def legal_front_edits(name, data=DATA):
    """List optional front cut stations present on a UIUC airfoil."""
    _, polygon, branches = read_profile(name, data)
    return tuple(action for _, extra in front_edit_layout(polygon, branches) for action in extra)


def branch_spline(coordinates, sign):
    """Interpolate a branch in chord length with a vertical tangent at the nose."""
    lengths = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
    knots = np.r_[0.0, np.cumsum(lengths)]
    secants = np.diff(coordinates, axis=0) / lengths[:, None]
    size = len(coordinates)
    system = np.zeros((size, size))
    rhs = np.zeros((size, 2))
    system[0, 0] = 1.0
    rhs[0] = (0.0, sign)
    for k in range(1, size - 1):
        before, after = lengths[k - 1:k + 1]
        system[k, k - 1:k + 2] = (after, 2 * (before + after), before)
        rhs[k] = 3 * (after * secants[k - 1] + before * secants[k])
    system[-1, -2:] = (1.0, 2.0)
    rhs[-1] = 3 * secants[-1]
    tangents = np.linalg.solve(system, rhs)
    first_controls = coordinates[:-1] + lengths[:, None] * tangents[:-1] / 3
    last_controls = coordinates[1:] - lengths[:, None] * tangents[1:] / 3
    return knots, first_controls, last_controls


def wall_allocation(x, counts, mode):
    segments = len(x) - 1
    common, extras = divmod(counts.front, segments)
    uniform = [common + int(k < extras) for k in range(segments)]
    if mode == "uniform":
        return uniform, counts.aft
    # Allocate in the cosine coordinate to retain leading-edge concentration
    # while giving wider parts of the front branch a proportional cell count.
    theta = np.arccos(np.clip(1 - 2 * x, -1, 1))
    quotas = counts.front * np.diff(theta) / (theta[-1] - theta[0])
    # Two free interior nodes let both endpoint cell lengths be prescribed.
    cells = np.full(segments, 3, dtype=int)
    for _ in range(counts.front - int(cells.sum())):
        cells[np.argmax(quotas - cells)] += 1
    if mode == "graded":
        return cells.tolist(), counts.aft
    # Retain the nose resolution while spending some of the aft budget in the
    # wide front intervals. Leave at least three unanchored cells on the aft wall.
    deficits = np.maximum(cells - uniform, 0)
    transfer = min(int(deficits.sum()), counts.aft - max(MIN_AFT_CELLS, counts.aft // 2 + 3))
    for _ in range(transfer):
        k = int(np.argmax(deficits))
        uniform[k] += 1
        deficits[k] -= 1
    return uniform, counts.aft - transfer


def geometry(polygon, branches, radius, counts, *, smooth_wall=False, front_mode="uniform", selected_cuts=None):
    geo = gmsh.model.occ if smooth_wall else gmsh.model.geo
    distributions = {}
    blocks = {}
    wall, farfield, wake = [], [], []
    side_curves = []

    def point(x, y):
        return geo.addPoint(float(x), float(y), 0.0)

    def spacing(curve, count, law="Progression", factor=1.0):
        distributions[curve] = (count + 1, law, factor)
        return curve

    def block(name, curves):
        blocks[name] = geo.addPlaneSurface([geo.addCurveLoop(curves)])

    def ray(start_tag, start_xy, stop_tag, stop_xy):
        length = np.linalg.norm(np.asarray(stop_xy) - start_xy)
        return spacing(geo.addLine(start_tag, stop_tag), counts.radial,
                       factor=progression(length, counts.radial, FIRST_LAYER))

    has_base = bool(branches[0][-1] != branches[1][-1])
    wall_points = [point(x, y) for x, y in polygon]
    center = point(FARFIELD_CENTER_X, 0.0)
    center_xy = np.array([FARFIELD_CENTER_X, 0.0])
    outer_front_xy = (FARFIELD_CENTER_X - radius, 0.0)
    outer_front = point(*outer_front_xy)
    radial_front = ray(wall_points[0], polygon[0], outer_front, outer_front_xy)
    tail_y = polygon[[branch[-1] for branch in branches], 1]
    if has_base and tail_y[0] >= tail_y[1]:
        raise ValueError("Trailing-edge base must join lower and upper wall branches in order")
    wake_half_height = max(0.025, float(np.ptp(tail_y)) / 2)
    outlet_y = tail_y.mean() + np.array([-wake_half_height, wake_half_height]) if has_base else tail_y
    outlet_x = radius + 1.0
    wake_ratio = progression(outlet_x - 1.0, counts.wake, FIRST_WAKE_DX)
    outlet_wake = [point(outlet_x, y) for y in outlet_y] if has_base else [
        point(outlet_x, outlet_y[0])
    ] * 2

    for i, (sign, branch) in enumerate(zip((-1, 1), branches)):
        side = "lower" if sign < 0 else "upper"
        outer_y = sign * radius
        cuts, normals = front_cuts(polygon[branch], sign)
        if smooth_wall:
            knots, first_controls, last_controls = branch_spline(polygon[branch], sign)

        def wall_curve(first, last):
            if not smooth_wall:
                return geo.addSpline([wall_points[k] for k in branch[first:last + 1]])
            # This B-spline is the exact restriction of the branch interpolant.
            # The same tangent is used by both intervals at every station.
            controls = [wall_points[branch[first]]]
            for k in range(first, last):
                controls.extend((point(*first_controls[k]), point(*last_controls[k]),
                                 wall_points[branch[k + 1]]))
            multiplicities = [4] + [3] * (last - first - 1) + [4]
            return geo.addBSpline(controls, knots=knots[first:last + 1], multiplicities=multiplicities)

        shoulder = cuts[-1]
        outer_mid = point(FARFIELD_CENTER_X, outer_y)
        previous_outer = outer_front
        previous_radial = radial_front
        normal_turn = 0.0
        front_cells, aft_cells = wall_allocation(polygon[branch[cuts], 0], counts, front_mode)
        if selected_cuts is not None:
            # Each edit splits a sector using its original cell allocation.
            # Coalescing adjacent sectors leaves the aft and wake counts intact.
            cell_offsets = dict(zip(cuts, np.r_[0, np.cumsum(front_cells)].tolist()))
            cuts = selected_cuts[i]
            front_cells = [cell_offsets[last] - cell_offsets[first] for first, last in zip(cuts, cuts[1:])]
        for k, (first, last, count) in enumerate(zip(cuts[:-1], cuts[1:], front_cells)):
            index = branch[last]
            start_xy = polygon[index]
            if last == shoulder:
                outer_xy = (FARFIELD_CENTER_X, outer_y)
                next_outer = outer_mid
            else:
                normal = normals[last]
                ref = start_xy - center_xy
                component = np.dot(ref, normal)
                travel = -component + math.sqrt(component**2 + radius**2 - np.dot(ref, ref))
                outer_xy = start_xy + travel * normal
                angle = math.atan2(sign * outer_xy[1], FARFIELD_CENTER_X - outer_xy[0])
                if not normal_turn < angle < math.pi / 2:
                    raise ValueError("Front-block normal rays must sweep the upstream quarter circle in order")
                next_outer = point(*outer_xy)
                normal_turn = angle

            radial = ray(wall_points[index], start_xy, next_outer, outer_xy)
            inner_curve = spacing(wall_curve(first, last), count, "Bump", 0.65)
            outer_curve = spacing(geo.addCircleArc(previous_outer, center, next_outer), count, "Bump", 0.65)
            wall.append(inner_curve)
            farfield.append(outer_curve)
            curves = ([inner_curve, radial, -outer_curve, -previous_radial] if sign == 1 else
                      [-inner_curve, previous_radial, outer_curve, -radial])
            block(f"{side}_front_{k + 1}", curves)
            previous_outer, previous_radial = next_outer, radial

        tail_index = branch[-1]
        tail_normal = normals[-1]
        tail_run = radius * tail_normal[0] / max(abs(tail_normal[1]), 1e-8)
        cut_x = max(1.6, min(0.45 * outlet_x,
                           polygon[tail_index, 0] + tail_run))
        outer_cut = point(cut_x, outer_y)
        outer_exit = point(outlet_x, outer_y)
        aft_wall = spacing(wall_curve(shoulder, len(branch) - 1), aft_cells, "Bump", 0.24)
        aft_top = spacing(geo.addLine(outer_mid, outer_cut), aft_cells, "Bump", 0.24)
        wake_top = spacing(geo.addLine(outer_cut, outer_exit), counts.wake, factor=wake_ratio)
        exit_side = spacing(geo.addLine(outlet_wake[i], outer_exit), counts.radial,
                            factor=progression(abs(outer_y - outlet_y[i]), counts.radial, FIRST_LAYER))
        wall.append(aft_wall)
        farfield.extend((aft_top, wake_top, exit_side))

        radial_tail = ray(wall_points[tail_index], polygon[tail_index], outer_cut, (cut_x, outer_y))
        if i == 0 or has_base:
            wake.append(spacing(geo.addLine(wall_points[tail_index], outlet_wake[i]), counts.wake,
                                factor=wake_ratio))
        wake_side = wake[-1]
        if sign == 1:
            block("upper_aft", [aft_wall, radial_tail, -aft_top, -previous_radial])
            block("upper_wake", [wake_side, exit_side, -wake_top, -radial_tail])
        else:
            block("lower_aft", [-aft_wall, previous_radial, aft_top, -radial_tail])
            block("lower_wake", [-wake_side, radial_tail, wake_top, -exit_side])
        side_curves.append(wall[-len(cuts):])

    base = None
    if has_base:
        base = spacing(geo.addLine(*(wall_points[branch[-1]] for branch in branches)), counts.base,
                       "Bump", 0.15)
        wake_exit = spacing(geo.addLine(*outlet_wake), counts.base, "Bump", 0.055)
        block("wake_core", [wake[0], wake_exit, -wake[1], -base])
        wall.append(base)
        farfield.append(wake_exit)

    geo.synchronize()
    for curve, parameters in distributions.items():
        gmsh.model.mesh.setTransfiniteCurve(curve, *parameters)
    for name, surface in blocks.items():
        gmsh.model.setEntityName(2, surface, name)
        gmsh.model.mesh.setTransfiniteSurface(surface)
        gmsh.model.mesh.setRecombine(2, surface)
    gmsh.model.addPhysicalGroup(1, wall, name="airfoil")
    gmsh.model.addPhysicalGroup(1, farfield, name="farfield")
    gmsh.model.addPhysicalGroup(2, list(blocks.values()), name="fluid")
    lower, upper = side_curves
    wall_loop = lower + ([base] if has_base else []) + [-curve for curve in reversed(upper)]
    joins = (["front_cut"] * (len(lower) - 2) + ["shoulder", "trailing_edge"] +
             ["trailing_edge"] * has_base + ["shoulder"] +
             ["front_cut"] * (len(upper) - 2) + ["leading_edge"])
    return blocks, wake, wall_loop, joins, base, (lower[-1], upper[-1])


def verify_interfaces(blocks, wake, wake_cells):
    physical_curves = {curve for dim, tag in gmsh.model.getPhysicalGroups(1)
                       for curve in gmsh.model.getEntitiesForPhysicalGroup(dim, tag)}
    names = {surface: label for label, surface in blocks.items()}
    sides = [("lower_wake", "wake_core"), ("upper_wake", "wake_core")] if len(wake) == 2 else [
        ("lower_wake", "upper_wake")
    ]

    def edge_keys(pairs):
        return set(map(tuple, np.sort(pairs.reshape(-1, 2), axis=1)))

    interfaces = []
    for curve, expected in zip(wake, sides):
        owners, _ = gmsh.model.getAdjacencies(1, curve)
        if {names[tag] for tag in owners} != set(expected) or curve in physical_curves:
            raise ValueError("A wake interface must join its two designated fluid blocks")
        tags, links = gmsh.model.mesh.getElementsByType(1, curve)
        if len(tags) != wake_cells:
            raise ValueError("Unexpected number of edges on the wake interface")
        seam = edge_keys(links)
        block_edges = []
        for surface in owners:
            _, links = gmsh.model.mesh.getElementsByType(3, surface)
            cells = links.reshape(-1, 4)
            pairs = np.stack([cells, np.roll(cells, -1, axis=1)], axis=-1)
            block_edges.append(edge_keys(pairs))
        if (block_edges[0] & block_edges[1]) != seam:
            raise ValueError("Wake blocks must share exactly the edges of the common curve")
        seam_nodes = {node for edge in seam for node in edge}
        interfaces.append({"gmsh_curve": int(curve), "blocks": list(expected),
                           "shared_edges": len(seam), "shared_nodes": len(seam_nodes)})
    return interfaces


def curve_nodes(curve):
    mesh = gmsh.model.mesh
    tags, xyz, parameters = mesh.getNodes(1, abs(curve), includeBoundary=True)
    order = np.argsort(parameters)
    if curve < 0:
        order = order[::-1]
    return tags[order], xyz.reshape(-1, 3)[order], parameters[order]


def sample_curve(curve):
    low, high = gmsh.model.getParametrizationBounds(1, curve)
    parameters = np.linspace(low[0], high[0], 2049)
    points = gmsh.model.getValue(1, curve, parameters).reshape(-1, 3)
    arclengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    return parameters, points, arclengths


def curve_reference(polygon, branches, radius, counts, front_mode, selected_cuts=None):
    """Capture the original edge nodes and wall sampling before changing kernels."""
    if front_mode == "balanced":
        _, uniform_fractions, _ = curve_reference(polygon, branches, radius, counts, "uniform", selected_cuts)
    gmsh.model.add("wall_reference")
    _, _, wall_loop, _, base, aft_wall = geometry(polygon, branches, radius, counts, front_mode=front_mode,
                                                  selected_cuts=selected_cuts)
    gmsh.model.mesh.generate(1)
    wall = {abs(curve) for curve in wall_loop} - {base}
    coordinates, fractions = {}, {}
    for _, curve in gmsh.model.getEntities(1):
        _, xyz, parameters = curve_nodes(curve)
        if curve in wall:
            samples, _, distances = sample_curve(curve)
            fractions[curve] = np.interp(parameters[1:-1], samples, distances) / distances[-1]
        else:
            coordinates[curve] = xyz[1:-1]
    gmsh.model.remove()
    tail_cells = {}
    if front_mode == "balanced":
        for curve in aft_wall:
            original = np.r_[0.0, uniform_fractions[curve], 1.0]
            fixed = counts.aft // 2
            remaining = len(fractions[curve]) + 1 - fixed
            anchor = counts.aft - fixed
            front = np.interp(np.linspace(0, anchor, remaining + 1)[1:-1], np.arange(len(original)), original)
            fractions[curve] = np.r_[front, original[anchor:-1]]
            tail_cells[curve] = fixed
    return coordinates, fractions, tail_cells


def apply_curve_reference(coordinates, fractions):
    """Use the legacy edge nodes on unchanged curves and arc fractions on new splines."""
    for curve, xyz in coordinates.items():
        tags, _, _ = curve_nodes(curve)
        parameters = gmsh.model.getParametrization(1, curve, xyz.ravel())
        for tag, point, parameter in zip(tags[1:-1], xyz, parameters, strict=True):
            gmsh.model.mesh.setNode(tag, point, [parameter])
    for curve, steps in fractions.items():
        tags, _, _ = curve_nodes(curve)
        samples, _, distances = sample_curve(curve)
        parameters = np.interp(steps * distances[-1], distances, samples)
        xyz = gmsh.model.getValue(1, curve, parameters).reshape(-1, 3)
        for tag, point, parameter in zip(tags[1:-1], xyz, parameters, strict=True):
            gmsh.model.mesh.setNode(tag, point, [parameter])


def match_wall_spacing(wall_loop, joins, tail_cells):
    """Redistribute the existing 1-D wall nodes before the block mesh is built."""
    curves = [curve_nodes(curve) for curve in wall_loop]
    sizes = [np.linalg.norm(np.diff(xyz, axis=0), axis=1) for _, xyz, _ in curves]
    spans = [np.linalg.norm(xyz[-1] - xyz[0]) for _, xyz, _ in curves]
    join_sizes = []
    for i, (before, kind) in enumerate(zip(sizes, joins)):
        j = (i + 1) % len(curves)
        join_sizes.append(None if kind == "trailing_edge" else
                          min(math.sqrt(before[-1] * sizes[j][0]), 0.3 * min(spans[i], spans[j])))

    for k, (signed_curve, (tags, xyz, parameters)) in enumerate(zip(wall_loop, curves)):
        curve = abs(signed_curve)
        targets = (join_sizes[k - 1], join_sizes[k])
        if signed_curve < 0:
            tags, xyz, parameters = tags[::-1], xyz[::-1], parameters[::-1]
            targets = targets[::-1]
        if targets == (None, None):
            continue
        # Anchor the half of the aft curve nearest the physical trailing edge.
        # Its original panels and the blunt base keep their original spacing.
        anchor = tail_cells.get(curve, (len(tags) - 1) // 2)
        active = slice(anchor if targets[0] is None else 0,
                       len(tags) - anchor if targets[1] is None else None)
        tags, xyz, parameters = tags[active], xyz[active], parameters[active]
        samples, points, arclengths = sample_curve(curve)

        def endpoint_parameter(endpoint, target):
            # The reported quantity is the straight mesh edge length. Match it
            # directly, even at the curved nose where it differs from arc length.
            order = slice(None) if endpoint == 0 else slice(None, None, -1)
            distances = np.linalg.norm(points[order] - xyz[endpoint], axis=1)
            first = np.flatnonzero(distances >= target)[0]
            inside, outside = samples[order][first - 1:first + 1]
            for _ in range(36):
                middle = (inside + outside) / 2
                position = gmsh.model.getValue(1, curve, [middle])
                if np.linalg.norm(position - xyz[endpoint]) < target:
                    inside = middle
                else:
                    outside = middle
            return (inside + outside) / 2

        first = parameters[1] if targets[0] is None else endpoint_parameter(0, targets[0])
        last = parameters[-2] if targets[1] is None else endpoint_parameter(-1, targets[1])
        start_arc, stop_arc = np.interp(parameters[[0, -1]], samples, arclengths)
        first_arc, last_arc = np.interp([first, last], samples, arclengths)
        original = np.diff(np.interp(parameters, samples, arclengths))
        cells = np.arange(len(original))
        width = min((len(original) - 1) / 2, max(3, len(original) / 3))
        start = np.maximum(0, 1 - cells / width)
        end = start[::-1]
        correction = start * math.log((first_arc - start_arc) / original[0])
        correction += end * math.log((stop_arc - last_arc) / original[-1])
        interior = original[1:-1] * np.exp(correction[1:-1])
        positions = first_arc + (last_arc - first_arc) * np.cumsum(interior)[:-1] / interior.sum()
        parameters = np.r_[first, np.interp(positions, arclengths, samples), last]
        points = gmsh.model.getValue(1, curve, parameters).reshape(-1, 3)
        for tag, point, parameter in zip(tags[1:-1], points, parameters):
            gmsh.model.mesh.setNode(tag, point, [parameter])


def wall_diagnostics(wall_loop, joins):
    coordinates = [curve_nodes(curve)[1] for curve in wall_loop]
    sizes = [np.linalg.norm(np.diff(xyz, axis=0), axis=1) for xyz in coordinates]

    def tangent(curve, at_end):
        bounds = gmsh.model.getParametrizationBounds(1, abs(curve))
        derivative = gmsh.model.getDerivative(1, abs(curve), bounds[int(at_end == (curve > 0))])
        return derivative if curve > 0 else -derivative

    records = []
    for i, kind in enumerate(joins):
        j = (i + 1) % len(wall_loop)
        before, after = sizes[i][-1], sizes[j][0]
        incoming = tangent(wall_loop[i], at_end=True)
        outgoing = tangent(wall_loop[j], at_end=False)
        angle = math.degrees(math.atan2(np.linalg.norm(np.cross(incoming, outgoing)), np.dot(incoming, outgoing)))
        records.append({"kind": kind, "point": coordinates[i][-1, :2].tolist(),
                        "edge_before": float(before), "edge_after": float(after),
                        "spacing_ratio": float(max(before / after, after / before)),
                        "cad_tangent_jump_deg": angle})
    smooth = [record for record in records if record["kind"] != "trailing_edge"]
    ratios = np.concatenate([curve_sizes[1:] / curve_sizes[:-1] for curve_sizes in sizes])
    max_interior_ratio = float(np.maximum(ratios, 1 / ratios).max())
    max_join_ratio = max(record["spacing_ratio"] for record in records)
    max_smooth_ratio = max(record["spacing_ratio"] for record in smooth)
    return {"joins": records,
            "max_join_spacing_ratio": max_join_ratio,
            "max_smooth_join_spacing_ratio": max_smooth_ratio,
            "max_adjacent_spacing_ratio": max(max_interior_ratio, max_join_ratio),
            "max_noncorner_adjacent_spacing_ratio": max(max_interior_ratio, max_smooth_ratio),
            "max_smooth_join_tangent_jump_deg": max(record["cad_tangent_jump_deg"] for record in smooth)}


def prepare_wake_curves(polygon, branches, radius, counts, wall_mode, selected_cuts, model_name):
    """Build the shared CAD edges and prepare their 1D mesh distributions."""
    smooth = wall_mode.startswith("smooth")
    matched = wall_mode not in ("legacy", "smooth")
    front_mode = wall_mode.removeprefix("smooth-") if wall_mode in ("smooth-graded", "smooth-balanced") else "uniform"
    tail_cells = {}
    if smooth:
        coordinates, fractions, tail_cells = curve_reference(polygon, branches, radius, counts, front_mode,
                                                             selected_cuts)
    gmsh.model.setCurrent(model_name)
    blocks, wake, wall_loop, joins, _, _ = geometry(polygon, branches, radius, counts,
                                                   smooth_wall=smooth, front_mode=front_mode,
                                                   selected_cuts=selected_cuts)
    if smooth or matched:
        gmsh.model.mesh.generate(1)
    if smooth:
        apply_curve_reference(coordinates, fractions)
    if matched:
        match_wall_spacing(wall_loop, joins, tail_cells)
    return blocks, wake, wall_loop, joins, tail_cells


def build_mesh(name, polygon, branches, radius, counts, staging, wall_mode, selected_cuts=None):
    gmsh.initialize(["gmsh", "-nopopup"], readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.Smoothing", 0)
        model_name = f"{name}_wake"
        gmsh.model.add(model_name)
        blocks, wake, wall_loop, joins, tail_cells = prepare_wake_curves(
            polygon, branches, radius, counts, wall_mode, selected_cuts, model_name)
        gmsh.model.mesh.generate(2)

        block_cells = {}
        for label, surface in blocks.items():
            types, tags, _ = gmsh.model.mesh.getElements(2, surface)
            if list(types) != [3] or not len(tags[0]):
                raise ValueError(f"Block {label} must contain linear quadrilaterals")
            block_cells[label] = len(tags[0])

        _, tags, _ = gmsh.model.mesh.getElements(2)
        determinants = gmsh.model.mesh.getElementQualities(tags[0], "minDetJac")
        min_jacobian = float(determinants.min())
        if not np.isfinite(determinants).all() or min_jacobian <= 0:
            raise ValueError(f"Gmsh produced a nonpositive or nonfinite cell Jacobian: {min_jacobian:.6g}")

        interfaces = verify_interfaces(blocks, wake, counts.wake)
        diagnostics = wall_diagnostics(wall_loop, joins)
        gmsh.write(str(staging / "mesh.msh"))
        gmsh.write(str(staging / "mesh.su2"))
    finally:
        gmsh.finalize()
    return block_cells, interfaces, min_jacobian, diagnostics


def generate(name, out, radius=FARFIELD_RADIUS, *, quad_budget=None, data=DATA, wall_mode=DEFAULT_WALL_MODE,
             front_edits=None):
    """Use the full template for None or add the given cuts to the compact template."""
    if not 15.0 <= radius <= 20.0:
        raise ValueError("--radius must be between 15 and 20 chords")
    if wall_mode not in WALL_MODES:
        raise ValueError(f"--wall-mode must be one of {', '.join(WALL_MODES)}")
    if quad_budget is not None:
        try:
            quad_budget = index(quad_budget)
        except TypeError as exc:
            raise ValueError("--quad-budget must be a positive integer") from exc
        if quad_budget <= 0:
            raise ValueError("--quad-budget must be a positive integer")
    source_id, polygon, branches = read_profile(name, data)
    selected_cuts = None
    if front_edits is not None:
        if isinstance(front_edits, str):
            raise ValueError("Front edits must be a sequence of action names")
        try:
            requested = set(front_edits)
        except TypeError as exc:
            raise ValueError("Front edits must be a sequence of action names") from exc
        layout = front_edit_layout(polygon, branches)
        available = tuple(action for _, extra in layout for action in extra)
        unavailable = requested.difference(available)
        if unavailable:
            names = ", ".join(sorted(map(repr, unavailable)))
            raise ValueError(f"Unavailable front edits for {name}: {names}")
        selected_cuts = [sorted(fixed + [cut for action, cut in extra.items() if action in requested])
                         for fixed, extra in layout]
    counts = counts_for_budget(quad_budget, branches[0][-1] != branches[1][-1])
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent, prefix=".wake_mesh_") as temp:
        staging = Path(temp)
        block_cells, interfaces, min_jacobian, diagnostics = build_mesh(
            name, polygon, branches, radius, counts, staging, wall_mode, selected_cuts)
        stats = inspect(*read_su2(staging / "mesh.su2"))
        if stats["quads"] != sum(block_cells.values()) or stats["quads"] != counts.quads:
            raise ValueError("Planned, Gmsh and exported SU2 cell totals disagree")
        report = {
            "airfoil": name,
            "source_id": source_id,
            "topology": "C-H" if "wake_core" in block_cells else "C",
            "wall_mode": wall_mode,
            "wall_diagnostics": diagnostics,
            "farfield_radius_chords": radius,
            "outlet_x_chords": radius + 1.0,
            "trailing_edge_type": "blunt" if "wake_core" in block_cells else "sharp",
            "trailing_edge_points": polygon[np.unique([branch[-1] for branch in branches])].tolist(),
            "blocks": len(block_cells),
            "block_quads": block_cells,
            "wake_interfaces": interfaces,
            "transfinite_cells": asdict(counts),
            "wall_cells_by_side": {
                side: {label.removeprefix(f"{side}_"): quads // counts.radial
                       for label, quads in block_cells.items()
                       if label.startswith(f"{side}_front_") or label == f"{side}_aft"}
                for side in ("lower", "upper")
            },
            "min_jacobian": min_jacobian,
            **stats,
        }
        if selected_cuts is not None:
            report["front_edits"] = [action for action in available if action in requested]
            report["legal_front_edits"] = list(available)
        if quad_budget is not None:
            report["requested_quad_budget"] = quad_budget
            report["quad_budget_difference"] = stats["quads"] - quad_budget
            report["exact_quad_budget"] = stats["quads"] == quad_budget
        report_text = json.dumps(report, indent=2) + "\n"
        (staging / "wake_topology.json").write_text(report_text, encoding="utf-8")
        out.mkdir(parents=True, exist_ok=True)
        for name in ("mesh.msh", "mesh.su2", "wake_topology.json"):
            (staging / name).replace(out / name)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airfoil", required=True, help="UIUC alias or sample ID")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DATA, help="Folder with cleaned UIUC samples.jsonl")
    parser.add_argument("--radius", type=float, default=FARFIELD_RADIUS,
                        help="farfield radius in chords, between 15 and 20 (default: %(default)s)")
    parser.add_argument("--quad-budget", type=int,
                        help="Target quadrilateral count; report the nearest admissible count if needed")
    parser.add_argument("--wall-mode", choices=WALL_MODES, default=DEFAULT_WALL_MODE,
                        help="Wall interpolant and spacing treatment (default: %(default)s)")
    parser.add_argument("--front-edits", nargs="*", choices=FRONT_EDIT_ACTIONS,
                        help="Use the compact front topology and optionally add these cuts")
    args = parser.parse_args()
    try:
        report = generate(args.airfoil, args.out, args.radius, quad_budget=args.quad_budget,
                          data=args.data, wall_mode=args.wall_mode, front_edits=args.front_edits)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
    if args.quad_budget is not None and not report["exact_quad_budget"]:
        print(f"Requested {args.quad_budget} quads, nearest admissible template has {report['quads']} quads.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
