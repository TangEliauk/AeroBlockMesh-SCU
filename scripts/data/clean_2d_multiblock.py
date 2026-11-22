"""Extract source-faithful planar block boundaries from formatted PLOT3D grids.

Only Python's standard library is required. The supported inputs are multiblock
2-D grids and axis-aligned, two-plane extrusions of a 2-D grid. Coordinates are
not normalized, smoothed or resampled. Interfaces are coordinate-verified
candidates, not invented solver boundary conditions or expert action histories.
"""
from __future__ import annotations

import argparse
from array import array
from collections import defaultdict
import gzip
import hashlib
import io
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import re


SCHEMA_VERSION = 1
ATOL = 1e-10
MAX_BYTES = 256 * 1024 * 1024
MAX_SCALARS = 20_000_000
FORMATS = ("plot3d_2d", "plot3d_thin_3d")
EDGES = ("U1", "UMAX", "V1", "VMAX")
RAW_PREFIX = "data/references/multiblock_2d_v1/"
CLEAN_PREFIX = "data/processed/block_topology_2d_v1/"
NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?")


def stable_bytes(document: object) -> bytes:
    return (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


class TopologyError(ValueError):
    """An input is outside the explicitly supported PLOT3D contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TopologyError(message)


class Tokens:
    """Strict formatted input, including Fortran list-directed repeat values."""

    def __init__(self, text: str):
        require(not re.search(r",\s*,", text), "Null comma-separated values are unsupported")
        require(not text.lstrip().startswith(","), "Leading null value")
        self.tokens = iter(re.finditer(r"[^\s,]+", text))
        self.remaining = 0
        self.repeated = 0.0
        self.used = 0

    def raw(self) -> str:
        try:
            return next(self.tokens).group()
        except StopIteration as exc:
            raise TopologyError("Truncated PLOT3D input") from exc

    def integer(self) -> int:
        require(not self.remaining, "Repeat notation is not allowed in grid dimensions")
        token = self.raw()
        require(bool(re.fullmatch(r"[+]?\d+", token)), f"Expected positive integer: {token!r}")
        result = int(token)
        require(result > 0, "Grid dimensions and block count must be positive")
        return result

    def coordinate(self) -> float:
        if self.remaining:
            self.remaining -= 1
            result = self.repeated
        else:
            token = self.raw()
            if "*" in token:
                fields = token.split("*")
                require(len(fields) == 2 and bool(re.fullmatch(r"\d+", fields[0])),
                        f"Invalid Fortran repeat: {token!r}")
                count = int(fields[0])
                require(0 < count <= MAX_SCALARS - self.used, "Repeat count exceeds supported resource bound")
                token = fields[1]
            else:
                count = 1
            require(bool(NUMBER.fullmatch(token)), f"Invalid coordinate: {token!r}")
            result = float(token.replace("D", "E").replace("d", "e"))
            require(math.isfinite(result), f"Non-finite coordinate: {token!r}")
            self.remaining = count - 1
            self.repeated = result
        self.used += 1
        require(self.used <= MAX_SCALARS, "Coordinate count exceeds supported resource bound")
        return result

    def finish(self) -> None:
        require(self.remaining == 0, "Extra coordinates in repeated value")
        require(next(self.tokens, None) is None, "Extra tokens after declared coordinates")


def read_grid(path: Path, grid_format: str) -> tuple[list[dict], dict]:
    require(grid_format in FORMATS, "Unsupported grid format")
    require(path.stat().st_size <= MAX_BYTES, "Source exceeds supported byte bound")
    raw = path.read_bytes()
    compressed = raw.startswith(b"\x1f\x8b")
    if compressed:
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            payload = stream.read(MAX_BYTES + 1)
        require(len(payload) <= MAX_BYTES, "Decoded source exceeds supported byte bound")
    else:
        payload = raw
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise TopologyError("Formatted PLOT3D must be ASCII") from exc
    cursor = Tokens(text)
    block_count = cursor.integer()
    require(block_count <= 10000, "Too many blocks")
    ndim = 2 if grid_format == "plot3d_2d" else 3
    dimensions = [tuple(cursor.integer() for _ in range(ndim)) for _ in range(block_count)]
    require(all(min(shape) >= 2 for shape in dimensions), "Each grid direction needs at least two vertices")
    scalar_count = ndim * sum(math.prod(shape) for shape in dimensions)
    require(scalar_count <= MAX_SCALARS, "Declared coordinate count exceeds supported resource bound")
    blocks = []
    for block_id, shape in enumerate(dimensions, 1):
        count = math.prod(shape)
        coordinates = [array("d", (cursor.coordinate() for _ in range(count))) for _ in range(ndim)]
        blocks.append({"block_id": block_id, "dimensions": shape, "coordinates": coordinates})
    cursor.finish()
    return blocks, {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                    "compression_by_signature": "gzip" if compressed else "none",
                    "gzip_crc_checked": compressed,
                    "decoded_bytes": len(payload), "decoded_sha256": hashlib.sha256(payload).hexdigest(),
                    "coordinate_scalar_count": scalar_count}


def offset(indices: tuple[int, ...], shape: tuple[int, ...]) -> int:
    result, stride = 0, 1
    for index, size in zip(indices, shape):
        result += index * stride
        stride *= size
    return result


def project(block: dict, grid_format: str) -> dict:
    shape, coordinates = block["dimensions"], block["coordinates"]
    if grid_format == "plot3d_2d":
        logical_axes, physical_axes = [0, 1], [0, 1]
        thin_axis, thin_physical, thin_values, difference = None, None, None, 0.0
    else:
        candidates = []
        for thin_axis in (axis for axis, size in enumerate(shape) if size == 2):
            logical_axes = [axis for axis in range(3) if axis != thin_axis]
            plane_indices = []
            for plane in (0, 1):
                current = []
                for v in range(shape[logical_axes[1]]):
                    for u in range(shape[logical_axes[0]]):
                        indices = [0, 0, 0]
                        indices[thin_axis] = plane
                        indices[logical_axes[0]], indices[logical_axes[1]] = u, v
                        current.append(offset(tuple(indices), shape))
                plane_indices.append(current)
            for thin_physical in range(3):
                thin_coordinates = [[coordinates[thin_physical][i] for i in plane] for plane in plane_indices]
                if any(max(values) - min(values) > ATOL for values in thin_coordinates):
                    continue
                thin_values = [values[0] for values in thin_coordinates]
                if abs(thin_values[1] - thin_values[0]) <= ATOL:
                    continue
                physical_axes = [axis for axis in range(3) if axis != thin_physical]
                difference = max(abs(coordinates[axis][a] - coordinates[axis][b])
                                 for axis in physical_axes for a, b in zip(*plane_indices))
                if difference <= ATOL:
                    candidates.append((thin_axis, thin_physical, logical_axes, physical_axes, thin_values, difference))
        require(len(candidates) == 1,
                f"Block {block['block_id']}: not a unique axis-aligned two-plane extrusion")
        thin_axis, thin_physical, logical_axes, physical_axes, thin_values, difference = candidates[0]
    nu, nv = (shape[axis] for axis in logical_axes)

    def source_index(u: int, v: int) -> list[int]:
        indices = [0] * len(shape)
        indices[logical_axes[0]], indices[logical_axes[1]] = u, v
        return [i + 1 for i in indices]

    def point(u: int, v: int) -> list[float]:
        index = tuple(i - 1 for i in source_index(u, v))
        return [coordinates[axis][offset(index, shape)] for axis in physical_axes]

    projected = [point(u, v) for v in range(nv) for u in range(nu)]
    edges = []
    for edge in EDGES:
        if edge in ("U1", "UMAX"):
            indices = [(0 if edge == "U1" else nu - 1, v) for v in range(nv)]
        else:
            indices = [(u, 0 if edge == "V1" else nv - 1) for u in range(nu)]
        points = [point(*index) for index in indices]
        require(all(max(abs(a - b) for a, b in zip(left, right)) > ATOL
                    for left, right in zip(points, points[1:])),
                f"Block {block['block_id']} has collapsed consecutive boundary vertices")
        edges.append({"edge": edge, "point_count": len(points), "points": points,
                      "source_index_range": [source_index(*indices[0]), source_index(*indices[-1])]})
    return {"block_id": block["block_id"], "source_dimensions": list(shape),
            "planar_dimensions": [nu, nv],
            "projection": {"source_logical_axes": ["IJK"[axis] for axis in logical_axes],
                           "physical_axes": ["XYZ"[axis] for axis in physical_axes],
                           "thin_logical_axis": "IJK"[thin_axis] if thin_axis is not None else None,
                           "thin_physical_axis": "XYZ"[thin_physical] if thin_physical is not None else None,
                           "thin_coordinate_values": thin_values,
                           "selected_source_plane_index": 1 if thin_axis is not None else None,
                           "max_projected_plane_difference": difference},
            "bounds": {"min": [min(p[a] for p in projected) for a in range(2)],
                       "max": [max(p[a] for p in projected) for a in range(2)]},
            "corners": [{"source_index": source_index(u, v), "point": point(u, v)}
                        for u, v in ((0, 0), (nu-1, 0), (nu-1, nv-1), (0, nv-1))],
            "edges": edges, "neighbors": []}


def matched_runs(left: list[list[float]], right: list[list[float]]) -> list[list[tuple[int, int]]]:
    """Find maximal point-matched edge subranges, rejecting ambiguous vertices."""
    bins = defaultdict(list)
    key = lambda p: tuple(math.floor(x / ATOL) for x in p)
    for j, point in enumerate(right):
        bins[key(point)].append(j)
    matches = []
    for i, point in enumerate(left):
        bucket = key(point)
        candidates = [j for du, dv in itertools.product((-1, 0, 1), repeat=2)
                      for j in bins.get((bucket[0] + du, bucket[1] + dv), ())
                      if max(abs(a-b) for a, b in zip(point, right[j])) <= ATOL]
        require(len(candidates) <= 1, "Ambiguous repeated boundary coordinate")
        if candidates:
            matches.append((i, candidates[0]))
    runs, current, direction = [], [], None
    for pair in matches:
        delta = pair[1] - current[-1][1] if current else None
        connected = bool(current and pair[0] == current[-1][0] + 1
                         and abs(delta) == 1 and (direction is None or delta == direction))
        if not connected:
            if len(current) > 1:
                runs.append(current)
            current, direction = [pair], None
        else:
            current.append(pair)
            direction = delta
    if len(current) > 1:
        runs.append(current)
    return runs


def edge_source_index(edge: dict, position: int) -> list[int]:
    start, end = edge["source_index_range"]
    return [a + (position if b != a else 0) for a, b in zip(start, end)]


def connect(blocks: list[dict]) -> tuple[list[dict], list[dict], dict]:
    interfaces, occupied = [], defaultdict(set)
    max_difference = 0.0
    total_points = 0
    axes = {tuple(block["projection"]["physical_axes"]) for block in blocks}
    require(len(axes) == 1, "Blocks use inconsistent planar physical axes")
    first_planes = blocks[0]["projection"]["thin_coordinate_values"]
    for block in blocks[1:]:
        planes = block["projection"]["thin_coordinate_values"]
        require((first_planes is None and planes is None)
                or (first_planes is not None and planes is not None
                    and max(abs(a-b) for a, b in zip(first_planes, planes)) <= ATOL),
                "Blocks use inconsistent extrusion planes")
    for first, second in itertools.combinations(blocks, 2):
        for left, right in itertools.product(first["edges"], second["edges"]):
            for run in matched_runs(left["points"], right["points"]):
                first_position, last_position = run[0], run[-1]
                local_difference = max(abs(a-b) for i, j in run
                                       for a, b in zip(left["points"][i], right["points"][j]))
                max_difference = max(max_difference, local_difference)
                total_points += len(run)
                sides = []
                for side, (block, edge) in enumerate(((first, left), (second, right))):
                    begin, end = first_position[side], last_position[side]
                    cells = set(range(min(begin, end), max(begin, end)))
                    key = (block["block_id"], edge["edge"])
                    require(not (cells & occupied[key]), "Non-manifold or multiply assigned edge segment")
                    occupied[key].update(cells)
                    sides.append({"block_id": block["block_id"], "edge": edge["edge"],
                                  "edge_point_range": [begin+1, end+1],
                                  "source_index_range": [edge_source_index(edge, begin), edge_source_index(edge, end)]})
                interface_id = len(interfaces) + 1
                orientation = 1 if last_position[1] > first_position[1] else -1
                interfaces.append({"interface_id": interface_id, "sides": sides,
                                   "node_count": len(run), "orientation": orientation,
                                   "max_absolute_coordinate_difference": local_difference,
                                   "method": "coordinate_matched_contiguous_boundary_subranges",
                                   "solver_declaration_interpreted": False})
                for index, block in enumerate((first, second)):
                    own, donor = sides[index], sides[1-index]
                    block["neighbors"].append({"interface_id": interface_id,
                                               "edge": own["edge"], "edge_point_range": own["edge_point_range"],
                                               "donor_block_id": donor["block_id"], "donor_edge": donor["edge"],
                                               "donor_edge_point_range": donor["edge_point_range"],
                                               "orientation": orientation})
    boundaries = []
    for block in blocks:
        for edge in block["edges"]:
            used = occupied[(block["block_id"], edge["edge"])]
            cells = edge["point_count"] - 1
            position = 0
            while position < cells:
                if position in used:
                    position += 1
                    continue
                begin = position
                while position < cells and position not in used:
                    position += 1
                boundaries.append({"block_id": block["block_id"], "edge": edge["edge"],
                                   "edge_point_range": [begin+1, position+1],
                                   "source_index_range": [edge_source_index(edge, begin), edge_source_index(edge, position)],
                                   "boundary_type": "not_interpreted",
                                   "classification": "unmatched_boundary_segment_not_a_solver_bc"})
    return interfaces, boundaries, {"unique_interfaces": len(interfaces),
                                    "directed_connections": 2 * len(interfaces),
                                    "all_interfaces_reciprocal": True,
                                    "matched_coordinate_pairs": total_points,
                                    "max_absolute_interface_difference": max_difference,
                                    "unclassified_boundary_segments": len(boundaries)}


def clean_case(grid_path: str | Path, *, case_id: str, grid_format: str,
               source_relative_path: str | None = None) -> dict:
    path = Path(grid_path)
    raw_blocks, source = read_grid(path, grid_format)
    source["path"] = source_relative_path if source_relative_path is not None else path.name
    blocks = [project(block, grid_format) for block in raw_blocks]
    interfaces, boundaries, checks = connect(blocks)
    checks.update({"status": "passed", "structured_blocks": len(blocks),
                   "all_coordinates_finite": True, "all_coordinate_counts_match_dimensions": True,
                   "projection_validated": True, "interface_absolute_tolerance": ATOL,
                   "smoothing_applied": False, "resampling_applied": False,
                   "source_coordinate_scalar_count": source["coordinate_scalar_count"],
                   "cell_quality_or_jacobian_checked": False})
    return {"schema_version": SCHEMA_VERSION, "case_id": case_id,
            "source_format": grid_format, "source": source,
            "index_convention": "PLOT3D source indices are 1-based, I varies fastest; edge ranges include both endpoints",
            "coordinate_units": "not_inferred_from_grid",
            "projection": {"policy": "original_2d" if grid_format == "plot3d_2d" else "verified_axis_aligned_two_plane_extrusion",
                           "absolute_tolerance": ATOL, "full_source_planes_retained_in_raw_file": True},
            "blocks": blocks, "interfaces": interfaces, "boundaries": boundaries, "checks": checks,
            "has_original_expert_action_history": False,
            "limitations": [
                "Terminal block reference; no original expert actions, human timing or training transitions.",
                "Interfaces are inferred from contiguous coordinate-matched edge nodes, not all solver input semantics.",
                "Unmatched edge segments are not automatically classified as walls, inflow, outflow or symmetry.",
                "Only point-matched interfaces between different blocks are supported; no periodic/self, mismatched or overset connectivity.",
                "Only axis-aligned identical two-plane extrusions are projected; full source grids remain unchanged.",
                "Coordinates retain source scale and origin; units are not inferred or converted.",
                "No cell Jacobian, mesh quality certification, CFD solve or accuracy validation is performed.",
                "Coordinate matching does not certify disjoint block interiors or rule out every geometric overlap.",
            ]}


def write_or_check(document: dict, output: Path, check: bool) -> None:
    payload = stable_bytes(document)
    if check:
        require(output.is_file() and output.read_bytes() == payload, f"Cleaned artifact differs: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(payload)


def collection_path(root: Path, name: str, prefix: str) -> Path:
    """Keep manifest paths portable and inside their assigned release directory."""
    require(isinstance(name, str) and bool(name), "Missing manifest path")
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and ".." not in relative.parts
            and "\\" not in name and ":" not in name and name.startswith(prefix),
            "Manifest path is outside its assigned release directory")
    allowed = (root / prefix).resolve()
    result = (root / name).resolve()
    require(allowed.is_relative_to(root) and result.is_relative_to(allowed)
            and result != allowed, "Manifest path escapes assigned release directory")
    return result


def preflight_cases(cases: list[tuple]) -> list[tuple]:
    """Reject every source/output collision before parsing or writing any case.

    Resolving handles relative paths and symlinks; samefile additionally handles
    hard links, whose path strings differ despite referring to the same file.
    """
    require(bool(cases), "Empty case collection")
    resolved = [(Path(grid).resolve(), case_id, grid_format, source_path, Path(output).resolve())
                for grid, case_id, grid_format, source_path, output in cases]
    require(len({case[1] for case in resolved}) == len(resolved), "Duplicate case ID")
    sources, outputs = [case[0] for case in resolved], [case[4] for case in resolved]
    require(all(path.is_file() for path in sources), "Missing or non-file source grid")
    require(all(not path.exists() or path.is_file() for path in outputs), "Output is not a regular file")

    def same_file(left: Path, right: Path) -> bool:
        return left == right or (left.exists() and right.exists() and left.samefile(right))

    for output in outputs:
        require(not any(same_file(output, source) for source in sources),
                "Output aliases a source grid; raw files must remain unchanged")
    for first, second in itertools.combinations(outputs, 2):
        require(not same_file(first, second), "Duplicate or aliased output path")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--grid", type=Path)
    parser.add_argument("--format", choices=FORMATS)
    parser.add_argument("--case-id")
    parser.add_argument("--source-path")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true", help="Rebuild in memory and compare; do not write")
    args = parser.parse_args()
    if args.grid:
        if not all((args.format, args.case_id, args.output)):
            parser.error("--grid requires --format, --case-id and --output")
        cases = [(args.grid, args.case_id, args.format, args.source_path, args.output)]
    else:
        if any((args.format, args.case_id, args.output, args.source_path)):
            parser.error("Explicit case arguments require --grid")
        root = args.repo_root.resolve()
        manifest = json.loads((root / "data/references/multiblock_2d_v1/manifest.json").read_text(encoding="utf-8"))
        cases = [(collection_path(root, case["source_grid"], RAW_PREFIX), case["case_id"], case["grid_format"],
                  case["source_grid"], collection_path(root, case["cleaned_path"], CLEAN_PREFIX))
                 for case in manifest["cases"]]
    cases = preflight_cases(cases)
    prepared = []
    for path, case_id, grid_format, source_path, output in cases:
        document = clean_case(path, case_id=case_id, grid_format=grid_format, source_relative_path=source_path)
        if not args.grid:
            expected = next(case["source_declared_blocks"] for case in manifest["cases"] if case["case_id"] == case_id)
            require(len(document["blocks"]) == expected, "Parsed block count differs from reference manifest")
        prepared.append((document, output))
    # Finish validation of the entire batch before the first artifact write.
    for document, output in prepared:
        write_or_check(document, output, args.check)
        print(json.dumps({"case_id": document["case_id"], "blocks": len(document["blocks"]),
                          "interfaces": len(document["interfaces"]), "check_only": args.check}, sort_keys=True))


if __name__ == "__main__":
    main()
