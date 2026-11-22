"""Extract and validate terminal block topology from two frozen CGNS/HDF5 cases.

Source gzip/HDF5 files are read only. Coordinates are neither normalized nor
resampled; the JSON stores topology and summaries, with the full nodes retained
in the referenced source. This intentionally supports Structured 3-index zones
and non-periodic Vertex GridConnectivity1to1 interfaces, not all of CGNS.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import io
import itertools
import json
from pathlib import Path

import h5py
import numpy as np


CASES = {
    "cgns_channel12": "data/references/multiblock_v1/channel12/sqnz_s.hdf.cgns.gz",
    "cgns_airfoil4": "data/references/multiblock_v1/airfoil4/multi_hdf5.cgns.gz",
}


class TopologyError(ValueError):
    """The source cannot satisfy the explicitly supported topology contract."""


def require(condition, message):
    if not condition:
        raise TopologyError(message)


def text(value):
    if isinstance(value, bytes):
        return value.decode("ascii").strip("\0 ")
    return str(value).strip("\0 ")


def label(node):
    return text(node.attrs.get("label", ""))


def children(node, node_label):
    return sorted(
        (child for child in node.values()
         if isinstance(child, h5py.Group) and label(child) == node_label),
        key=lambda child: child.name,
    )


def data(node):
    require(" data" in node, f"Missing node data: {node.name}")
    return np.asarray(node[" data"][()])


def node_text(node):
    values = data(node)
    return values.tobytes().decode("ascii").strip("\0 ")


def optional_text(node, node_label, default=None):
    matches = children(node, node_label)
    require(len(matches) <= 1, f"Ambiguous {node_label}: {node.name}")
    return node_text(matches[0]) if matches else default


def inherited(node, node_label):
    """Locate the nearest CGNS declaration, preserving its actual node scope."""
    while isinstance(node, h5py.Group) and node.name != "/":
        matches = children(node, node_label)
        require(len(matches) <= 1, f"Ambiguous inherited {node_label}: {node.name}")
        if matches:
            return matches[0]
        node = node.parent
    return None


def unit_declaration(node):
    values = data(node)
    require(values.shape == (5, 32), f"Unsupported DimensionalUnits shape: {node.name}")
    names = [values[i].tobytes().decode("ascii").strip("\0 ") for i in range(5)]
    return {"source_node": node.name,
            "units": dict(zip(("mass", "length", "time", "temperature", "angle"), names))}


def family_reference(zone, family, definitions):
    if family is None:
        return None
    candidates = {family, zone.parent.name + "/" + family, "/" + family.lstrip("/")}
    matches = candidates.intersection(definitions)
    require(len(matches) == 1, f"Unresolved FamilyName {family}: {zone.name}")
    return matches.pop()


def integers(node, expected_shape):
    values = data(node)
    require(values.dtype.kind in "iu", f"Non-integer index data: {node.name}")
    require(values.shape == expected_shape,
            f"Unsupported index shape {values.shape}: {node.name}; expected {expected_shape}")
    return values.astype(np.int64)


def signed_transform(values):
    values = np.asarray(values)
    require(values.shape == (3,) and values.dtype.kind in "iu",
            "Transform must contain three signed integer axes")
    require(sorted(abs(int(v)) for v in values) == [1, 2, 3],
            f"Transform is not a signed permutation: {values.tolist()}")
    return values.astype(np.int64)


def point_range(values, dimensions, context, require_face=True):
    values = np.asarray(values)
    require(values.shape == (2, 3) and values.dtype.kind in "iu",
            f"Invalid PointRange shape/type: {context}")
    require(bool(np.all(values >= 1) and np.all(values <= np.asarray(dimensions))),
            f"PointRange outside zone dimensions: {context}")
    if require_face:
        fixed = np.flatnonzero(values[0] == values[1])
        require(len(fixed) == 1, f"PointRange is not a face patch: {context}")
        axis = int(fixed[0])
        require(int(values[0, axis]) in (1, int(dimensions[axis])),
                f"PointRange is not on the zone boundary: {context}")
    return values.astype(np.int64)


def transform_matrix(transform):
    transform = signed_transform(transform)
    matrix = np.zeros((3, 3), dtype=np.int64)
    for source_axis, target in enumerate(transform):
        matrix[abs(int(target)) - 1, source_axis] = 1 if target > 0 else -1
    return matrix


def mapped_indices(source_range, donor_range, transform):
    """Return all paired CGNS 1-based indices, including reversed ranges."""
    matrix = transform_matrix(transform)
    require(np.array_equal(matrix @ (source_range[1] - source_range[0]),
                           donor_range[1] - donor_range[0]),
            "Transform and PointRangeDonor endpoints disagree")
    axes = [range(int(a), int(b) + (1 if b >= a else -1), 1 if b >= a else -1)
            for a, b in zip(source_range[0], source_range[1])]
    current = np.asarray(list(itertools.product(*axes)), dtype=np.int64)
    donor = donor_range[0] + (current - source_range[0]) @ matrix.T
    require(bool(np.all(donor >= np.min(donor_range, axis=0))
                 and np.all(donor <= np.max(donor_range, axis=0))),
            "Mapped indices outside donor interface")
    return current, donor


def validate_coordinate_match(left_coordinates, right_coordinates, left, right, atol):
    # CGNS uses (i,j,k); HDF5's stored array axes are (k,j,i).
    left_key = tuple((left[:, ::-1] - 1).T)
    right_key = tuple((right[:, ::-1] - 1).T)
    differences = np.asarray([
        np.abs(a[left_key] - b[right_key])
        for a, b in zip(left_coordinates, right_coordinates)
    ])
    require(bool(np.isfinite(differences).all()), "Nonfinite interface coordinates")
    maximum = np.max(differences, axis=1)
    require(bool(np.all(maximum <= atol)),
            f"Interface coordinate mismatch: max XYZ difference {maximum.tolist()} > {atol}")
    return maximum.tolist()


def range_key(values):
    return (tuple(np.min(values, axis=0)), tuple(np.max(values, axis=0)))


def reciprocal_pairs(connections):
    pairs = {}
    for entry in connections:
        ends = ((entry["source_block"], range_key(entry["source_range"])),
                (entry["donor_block"], range_key(entry["donor_range"])))
        key = tuple(sorted(ends))
        pairs.setdefault(key, []).append(entry)
    result = []
    for key, directions in sorted(pairs.items()):
        require(len(directions) == 2,
                f"Interface must have exactly two reciprocal records: {key}")
        first, second = sorted(directions, key=lambda x: x["source_node"])
        require(first["source_block"] == second["donor_block"]
                and first["donor_block"] == second["source_block"],
                "Interface does not contain opposite directions")
        forward = transform_matrix(first["transform"])
        reverse = transform_matrix(second["transform"])
        require(np.array_equal(reverse @ forward, np.eye(3, dtype=np.int64)),
                "Reciprocal interface transforms are not inverse")
        forward_offset = first["donor_range"][0] - forward @ first["source_range"][0]
        reverse_offset = second["donor_range"][0] - reverse @ second["source_range"][0]
        require(np.array_equal(reverse @ forward_offset + reverse_offset,
                               np.zeros(3, dtype=np.int64)),
                "Reciprocal interface index offsets are not inverse")
        result.append(directions)
    return result


def clean_case(source, repo_root, case_id, coordinate_atol=1e-10):
    root = Path(repo_root).resolve()
    source = Path(source).resolve()
    require(source.is_relative_to(root), "Source must be within --repo-root")
    require(np.isfinite(coordinate_atol) and coordinate_atol >= 0,
            "Coordinate tolerance must be finite and nonnegative")
    payload = source.read_bytes()
    raw = gzip.decompress(payload) if source.suffix == ".gz" else payload
    blocks, boundaries, family_definitions, units, connections = [], [], [], [], []
    coordinates_by_zone, dimensions_by_zone = {}, {}
    counts = Counter()
    with h5py.File(io.BytesIO(raw), "r") as handle:
        nodes = []
        handle.visititems(lambda name, obj: nodes.append(obj)
                          if isinstance(obj, h5py.Group) else None)
        for node in nodes:
            counts[label(node)] += 1
            if label(node) in ("GridConnectivity_t", "Periodic_t", "OversetHoles_t"):
                raise TopologyError(f"Unsupported connectivity type {label(node)}: {node.name}")
            if label(node) == "DimensionalUnits_t":
                units.append(unit_declaration(node))
            if label(node) == "Family_t":
                family_definitions.append({
                    "source_node": node.name, "name": node.name.rsplit("/", 1)[-1],
                    "boundary_types": [node_text(child) for child in children(node, "FamilyBC_t")],
                })
        zones = sorted((node for node in nodes if label(node) == "Zone_t"),
                       key=lambda node: node.name)
        family_paths = {item["source_node"] for item in family_definitions}
        require(bool(zones), "No CGNS zones found")
        for zone in zones:
            require(optional_text(zone, "ZoneType_t") == "Structured",
                    f"Only Structured zones are supported: {zone.name}")
            base = data(zone.parent)
            require(base.shape == (2,) and base.tolist() == [3, 3],
                    f"Only CellDimension=3, PhysicalDimension=3 are supported: {zone.parent.name}")
            size = integers(zone, (3, 3))
            dimensions = size[0]
            require(bool(np.all(dimensions >= 2)), f"Degenerate vertex dimensions: {zone.name}")
            require(np.array_equal(size[1], dimensions - 1) and bool(np.all(size[2] == 0)),
                    f"Inconsistent Structured cell/boundary dimensions: {zone.name}")
            grids = children(zone, "GridCoordinates_t")
            require(len(grids) == 1, f"Expected one GridCoordinates node: {zone.name}")
            coordinate_values, coordinate_nodes, coordinate_metadata = [], [], []
            for axis in "XYZ":
                name = "Coordinate" + axis
                require(name in grids[0], f"Missing {name}: {zone.name}")
                coordinate = grids[0][name]
                values = data(coordinate)
                require(values.shape == tuple(dimensions[::-1]),
                        f"Coordinate shape differs from zone dimensions: {coordinate.name}")
                require(values.dtype.kind in "fiu" and bool(np.isfinite(values).all()),
                        f"Nonfinite or nonnumeric coordinates: {coordinate.name}")
                coordinate_values.append(values.astype(np.float64))
                coordinate_nodes.append(coordinate.name)
                declared_units = inherited(coordinate, "DimensionalUnits_t")
                declared_class = inherited(coordinate, "DataClass_t")
                coordinate_metadata.append({
                    "coordinate": name,
                    "inherited_units": unit_declaration(declared_units) if declared_units is not None else None,
                    "inherited_data_class": node_text(declared_class) if declared_class is not None else None,
                })
            coordinates_by_zone[zone.name] = coordinate_values
            dimensions_by_zone[zone.name] = dimensions
            corners = []
            for index in itertools.product(*[(1, int(d)) for d in dimensions]):
                key = tuple((np.asarray(index) - 1)[::-1])
                corners.append({"index_1based": list(index),
                                "xyz": [float(values[key]) for values in coordinate_values]})
            blocks.append({
                "block_id": zone.name, "source_node": zone.name, "zone_type": "Structured",
                "vertex_dimensions_ijk": dimensions.tolist(), "cell_dimensions_ijk": size[1].tolist(),
                "point_count": int(np.prod(dimensions)), "cell_count": int(np.prod(size[1])),
                "coordinate_hdf5_shape_kji": list(coordinate_values[0].shape),
                "coordinate_source_nodes": coordinate_nodes,
                "coordinate_metadata": coordinate_metadata,
                "coordinate_bounds_xyz": [[float(v.min()), float(v.max())] for v in coordinate_values],
                "corner_vertices": corners,
                "family": optional_text(zone, "FamilyName_t"),
                "family_source_node": family_reference(zone, optional_text(zone, "FamilyName_t"), family_paths),
                "flow_solution_source_nodes": [child.name for child in children(zone, "FlowSolution_t")],
            })
        for zone in zones:
            dimensions = dimensions_by_zone[zone.name]
            for container in children(zone, "ZoneBC_t"):
                for bc in children(container, "BC_t"):
                    location = optional_text(bc, "GridLocation_t", "Vertex")
                    require(location == "Vertex", f"Unsupported BC GridLocation {location}: {bc.name}")
                    require("PointRange" in bc, f"Only BC PointRange is supported: {bc.name}")
                    values = point_range(integers(bc["PointRange"], (2, 3)), dimensions, bc.name)
                    family = optional_text(bc, "FamilyName_t")
                    require(node_text(bc) != "FamilySpecified" or family is not None,
                            f"FamilySpecified BC lacks FamilyName: {bc.name}")
                    boundaries.append({
                        "source_node": bc.name, "block_id": zone.name, "bc_type": node_text(bc),
                        "grid_location": location, "point_range_1based": values.tolist(),
                        "family": family,
                        "family_source_node": family_reference(zone, family, family_paths),
                    })
            for container in children(zone, "ZoneGridConnectivity_t"):
                for interface in children(container, "GridConnectivity1to1_t"):
                    require(optional_text(interface, "GridLocation_t", "Vertex") == "Vertex",
                            f"Only Vertex connectivity is supported: {interface.name}")
                    donor_name = node_text(interface)
                    candidates = [donor_name, zone.parent.name + "/" + donor_name,
                                  "/" + donor_name.lstrip("/")]
                    donor_paths = {name for name in candidates if name in dimensions_by_zone}
                    require(len(donor_paths) == 1, f"Unresolved donor zone {donor_name}: {interface.name}")
                    donor_path = donor_paths.pop()
                    require(all(name in interface for name in ("PointRange", "PointRangeDonor", "Transform")),
                            f"Incomplete one-to-one interface: {interface.name}")
                    left = point_range(integers(interface["PointRange"], (2, 3)), dimensions, interface.name)
                    right = point_range(integers(interface["PointRangeDonor"], (2, 3)),
                                        dimensions_by_zone[donor_path], interface.name + " donor")
                    transform = signed_transform(integers(interface["Transform"], (3,)))
                    current, donor = mapped_indices(left, right, transform)
                    maximum = validate_coordinate_match(coordinates_by_zone[zone.name],
                                                        coordinates_by_zone[donor_path],
                                                        current, donor, coordinate_atol)
                    connections.append({
                        "source_node": interface.name, "source_block": zone.name, "donor_block": donor_path,
                        "source_range": left, "donor_range": right, "transform": transform,
                        "matched_coordinate_pairs": len(current), "max_absolute_xyz_difference": maximum,
                    })
    require(bool(connections), "No supported one-to-one interfaces found")
    pairs = reciprocal_pairs(connections)
    interfaces = []
    for number, pair in enumerate(pairs, 1):
        directions = []
        for entry in sorted(pair, key=lambda x: x["source_node"]):
            directions.append({
                "source_node": entry["source_node"], "source_block": entry["source_block"],
                "donor_block": entry["donor_block"],
                "point_range_1based": entry["source_range"].tolist(),
                "point_range_donor_1based": entry["donor_range"].tolist(),
                "transform": entry["transform"].tolist(),
                "matched_coordinate_pairs": entry["matched_coordinate_pairs"],
                "max_absolute_xyz_difference": entry["max_absolute_xyz_difference"],
            })
        interfaces.append({"interface_id": f"interface_{number:03d}", "reciprocal": True,
                           "directions": directions})
    coordinate_metadata = [item for block in blocks for item in block["coordinate_metadata"]]
    length_units = [item["inherited_units"]["units"]["length"]
                    if item["inherited_units"] is not None else None for item in coordinate_metadata]
    declared_length_units = sorted({value for value in length_units if value not in (None, "Null", "UserDefined")})
    all_lengths_declared = all(value not in (None, "Null", "UserDefined") for value in length_units)
    return {
        "schema_version": 1, "case_id": case_id, "source_format": "CGNS/HDF5",
        "source": {"path": source.relative_to(root).as_posix(), "bytes": len(payload),
                   "sha256": hashlib.sha256(payload).hexdigest(),
                   "decompressed_sha256": hashlib.sha256(raw).hexdigest()},
        "index_convention": "CGNS 1-based inclusive [i,j,k]; stored HDF5 coordinate arrays are [k,j,i]",
        "coordinate_units": {"status": "declared_for_all_coordinate_axes" if all_lengths_declared else "not_fully_declared_for_coordinates",
                             "declared_length_units": declared_length_units,
                             "all_coordinate_axes_have_declared_length_units": all_lengths_declared,
                             "unit_conversion_applied": False},
        "source_unit_declarations": sorted(units, key=lambda x: x["source_node"]),
        "blocks": blocks, "interfaces": interfaces, "boundaries": boundaries,
        "families": sorted(family_definitions, key=lambda x: x["source_node"]),
        "checks": {
            "status": "passed", "structured_blocks": len(blocks),
            "all_coordinate_shapes_match_dimensions": True, "all_coordinates_finite": True,
            "boundary_ranges_within_zone_and_on_faces": True,
            "directed_one_to_one_connections": len(connections), "unique_interfaces": len(interfaces),
            "all_interfaces_have_reciprocal_inverse_maps": True,
            "matched_coordinate_pairs_including_both_directions": sum(e["matched_coordinate_pairs"] for e in connections),
            "max_absolute_xyz_interface_difference": np.max([e["max_absolute_xyz_difference"] for e in connections], axis=0).tolist(),
            "coordinate_absolute_tolerance": coordinate_atol,
            "source_node_label_counts": dict(sorted(counts.items())),
            "smoothing_applied": False, "resampling_applied": False,
        },
        "limitations": [
            "Terminal structured blocks and interfaces, not engineer operation histories or SAC transitions.",
            "Only Structured 3-index zones, Vertex face ranges and non-periodic one-to-one connectivity are supported; other connectivity fails explicitly.",
            "Full coordinate and flow-solution arrays remain in the unchanged referenced CGNS source; the JSON stores summaries only.",
            "No Jacobian, wall spacing, solver convergence or CFD accuracy certification is performed.",
            "Coordinate units are not inferred or converted; interpret source unit declarations by their CGNS node scope.",
        ],
    }


def encoded(result):
    return (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path("data/processed/block_topology_v1"),
                        help="Output directory; relative paths are resolved under repo-root")
    parser.add_argument("--case", choices=sorted(CASES), action="append",
                        help="Process a named frozen case; default: both cases")
    parser.add_argument("--coordinate-atol", type=float, default=1e-10)
    parser.add_argument("--check", action="store_true", help="Recompute and compare committed JSON bytes without writing")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    for case_id in args.case or CASES:
        result = clean_case(root / CASES[case_id], root, case_id, args.coordinate_atol)
        target = output / (case_id + ".json")
        content = encoded(result)
        if args.check:
            require(target.is_file() and target.read_bytes() == content,
                    f"Cleaned JSON is missing or stale: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        print(f"{case_id}: blocks={len(result['blocks'])}, interfaces={len(result['interfaces'])}, "
              f"boundaries={len(result['boundaries'])}; {'checked' if args.check else 'written'}")


if __name__ == "__main__":
    main()
