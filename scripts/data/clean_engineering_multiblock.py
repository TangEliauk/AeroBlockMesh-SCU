"""Extract source-faithful ICEM blocking records and ONERA M6 connectivity.

Only Python's standard library is required. Original coordinates and opaque
proprietary attributes are retained, without smoothing or topology inference.
"""

from __future__ import annotations

import argparse
from array import array
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import zipfile


SCHEMA_VERSION = 1
FACES = ("I1", "IMAX", "J1", "JMAX", "K1", "KMAX")


def integer(token: str) -> int:
    if not re.fullmatch(r"[+-]?\d+", token):
        raise ValueError(f"Expected integer, got {token!r}")
    return int(token)


def number(token: str) -> float:
    value = float(token.replace("D", "E").replace("d", "e"))
    if not math.isfinite(value):
        raise ValueError(f"Non-finite coordinate or parameter: {token!r}")
    return value


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def source_record(path: Path, root: Path) -> dict:
    payload = path.read_bytes()
    return {"path": path.resolve().relative_to(root.resolve()).as_posix(),
            "bytes": len(payload), "sha256": digest(payload)}


def bounds(points: list[list[float]]) -> dict:
    if not points:
        raise ValueError("Cannot compute bounds of empty point set")
    return {"min": [min(point[axis] for point in points) for axis in range(3)],
            "max": [max(point[axis] for point in points) for axis in range(3)]}


def sections(text: str) -> list[tuple[str, str | None]]:
    """Read top-level headers and balanced bodies; nested content stays intact."""
    lines = [line.strip() for line in text.splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    result = []
    index = 0
    while index < len(lines):
        header = lines[index]
        if "{" in header or "}" in header:
            raise ValueError(f"Unexpected top-level braces in {header!r}")
        index += 1
        body = None
        if index < len(lines) and lines[index] == "{":
            depth = 1
            index += 1
            content = []
            while index < len(lines) and depth:
                line = lines[index]
                depth += line.count("{") - line.count("}")
                if depth < 0:
                    raise ValueError(f"Unbalanced section {header}")
                if depth:
                    content.append(line)
                elif line != "}":
                    raise ValueError(f"Invalid closing line in {header}")
                index += 1
            if depth:
                raise ValueError(f"Unclosed section {header}")
            body = "\n".join(content)
        result.append((header, body))
    return result


def parse_icem_blk(payload: bytes, member_name: str) -> dict:
    text = payload.decode("ascii")
    if not text.startswith("# blocking file written by ANSYS ICEM CFD Mesh Editor"):
        raise ValueError("Unrecognized ICEM blocking header")
    records = sections(text)
    by_name = {}
    for header, body in records:
        key = header.split()[0]
        if key != "GRID" and key in by_name:
            raise ValueError(f"Duplicate section {key}")
        if key != "GRID":
            by_name[key] = (header, body)
    for required in ("PREPOINTS", "FAMILIES", "NODES", "SUPERBLOCKS"):
        if required not in by_name or by_name[required][1] is None:
            raise ValueError(f"Missing ICEM section {required}")

    prepoints = []
    pp_ids = set()
    pp_header, pp_body = by_name["PREPOINTS"]
    pp_tokens = pp_header.split()
    if len(pp_tokens) != 3 or pp_tokens[2] != "internal_ids":
        raise ValueError("Unsupported PREPOINTS header")
    for line in pp_body.splitlines():
        fields = line.split()
        if len(fields) < 5:
            raise ValueError("Malformed PREPOINTS record")
        point_id = integer(fields[0])
        if point_id < 0 or point_id in pp_ids:
            raise ValueError("Duplicate or negative prepoint ID")
        pp_ids.add(point_id)
        prepoints.append({"id": point_id, "coordinates": list(map(number, fields[1:4])),
                          "geometry_reference": " ".join(fields[4:])})
    if len(prepoints) != integer(pp_tokens[1]):
        raise ValueError("PREPOINTS declared count mismatch")

    families = []
    for line in by_name["FAMILIES"][1].splitlines():
        fields = line.split()
        if len(fields) != 2 or any(x["name"] == fields[0] for x in families):
            raise ValueError("Malformed or duplicate family")
        families.append({"name": fields[0], "source_flag": integer(fields[1]),
                         "entity_assignment_verified": False})

    nodes = {}
    if by_name["NODES"][0] != "NODES new_numbering":
        raise ValueError("Unsupported NODES numbering convention")
    node_pattern = re.compile(r"([^{}]+)\{\s*([^{}]+)\s*\}(.*)")
    for line in by_name["NODES"][1].splitlines():
        match = node_pattern.fullmatch(line)
        if not match:
            raise ValueError("Malformed ICEM NODES record")
        fields = match[1].split()
        if len(fields) != 8 or fields[6] != "x":
            raise ValueError("Unsupported ICEM NODES layout")
        node_id = integer(fields[5])
        if node_id < 0 or node_id in nodes:
            raise ValueError("Duplicate or negative node ID")
        logical_fields = match[2].split()
        if len(logical_fields) < 3:
            raise ValueError("Missing node logical indices")
        nodes[node_id] = {"id": node_id,
                          "coordinates": list(map(number, fields[:3])),
                          "opaque_source_attributes": fields[3:5] + fields[7:],
                          "logical_indices_as_stored": list(map(integer, logical_fields[:3])),
                          "opaque_extra_logical_references_as_stored": logical_fields[3:],
                          "opaque_projection_as_stored": match[3].strip()}

    grids = []
    grid_ids = set()
    for header, body in records:
        if not header.startswith("GRID "):
            continue
        fields = header.split()
        if len(fields) != 3 or body is None:
            raise ValueError("Malformed GRID header")
        grid_id = integer(fields[1])
        if grid_id in grid_ids:
            raise ValueError("Duplicate GRID ID")
        grid_ids.add(grid_id)
        entries = []
        for line in body.splitlines():
            pair = line.split()
            if len(pair) != 2:
                raise ValueError("Malformed GRID entry")
            count = integer(pair[1])
            if count < 2:
                raise ValueError("GRID count must be at least 2")
            entries.append({"coordinate": number(pair[0]), "source_count": count})
        if len(entries) != integer(fields[2]):
            raise ValueError("GRID declared count mismatch")
        grids.append({"id": grid_id, "entries": entries})

    blocks = []
    block_ids = set()
    body = by_name["SUPERBLOCKS"][1]
    pattern = re.compile(r"\{\s*([^{}]+)\{([^{}]+)\}([^{}]*)\}", re.S)
    end = 0
    for match in pattern.finditer(body):
        if body[end:match.start()].strip():
            raise ValueError("Unparsed SUPERBLOCKS content")
        end = match.end()
        fields = match[1].split()
        if len(fields) < 5:
            raise ValueError("Malformed SUPERBLOCKS header")
        block_id = integer(fields[0])
        if block_id < 0 or block_id in block_ids:
            raise ValueError("Duplicate or negative SUPERBLOCK ID")
        block_ids.add(block_id)
        shape = list(map(integer, fields[1:4]))
        if any(size < 2 for size in shape):
            raise ValueError("Invalid SUPERBLOCK lattice shape")
        opaque_attribute = integer(fields[4])
        grid_refs = fields[5:]
        if len(grid_refs) != sum(size - 1 for size in shape):
            raise ValueError("SUPERBLOCK grid-reference count mismatch")
        for ref in grid_refs:
            if not re.fullmatch(r"\d+:\d+:\d+", ref):
                raise ValueError("Invalid SUPERBLOCK grid reference")
            grid_id, interval, _ = map(int, ref.split(":"))
            grid = next((item for item in grids if item["id"] == grid_id), None)
            if grid is None or interval >= len(grid["entries"]) - 1:
                raise ValueError("SUPERBLOCK references unknown GRID interval")
        node_ids = list(map(integer, match[2].split()))
        if len(node_ids) != math.prod(shape):
            raise ValueError("SUPERBLOCK node-reference count mismatch")
        ni, nj, nk = shape
        corner_ids = [node_ids[i + ni*(j + nj*k)] for k, j, i in
                      itertools.product((0, nk-1), (0, nj-1), (0, ni-1))]
        if any(node_id not in nodes for node_id in corner_ids):
            raise ValueError("SUPERBLOCK corner references unknown NODES ID")
        unresolved = sorted(set(node_ids) - nodes.keys())
        resolved_coordinates = [nodes[node_id]["coordinates"] for node_id in node_ids if node_id in nodes]
        blocks.append({"id": block_id, "record_kind": "ICEM_SUPERBLOCK_source_record",
                       "lattice_shape_as_stored": shape,
                       "node_ids_source_order": node_ids,
                       "corner_node_ids_i_fastest": corner_ids,
                       "unresolved_noncorner_node_ids": unresolved,
                       "grid_references_as_stored": grid_refs,
                       "opaque_attribute": opaque_attribute,
                       "opaque_trailing_attributes": list(map(integer, match[3].split())),
                       "bounds": bounds(resolved_coordinates),
                       "bounds_scope": "explicitly_defined_referenced_nodes_only",
                       "final_fluid_block_status": "not_interpreted"})
    if body[end:].strip() or not blocks:
        raise ValueError("Unparsed or empty SUPERBLOCKS section")
    parsed_names = {"PREPOINTS", "FAMILIES", "NODES", "SUPERBLOCKS", "GRID"}
    unparsed = [{"header": header,
                 "body_sha256": digest(body.encode("ascii")) if body is not None else None}
                for header, body in records if header.split()[0] not in parsed_names]
    ordered_prepoints = sorted(prepoints, key=lambda item: item["id"])
    return {"member": member_name, "member_bytes": len(payload),
            "member_sha256": digest(payload), "writer_header": text.splitlines()[0],
            "prepoints": ordered_prepoints, "nodes": sorted(nodes.values(), key=lambda item: item["id"]),
            "families": families, "grids": grids, "blocks": sorted(blocks, key=lambda item: item["id"]),
            "geometry_signature_sha256": digest(stable_bytes(ordered_prepoints)),
            "unparsed_sections": unparsed,
            "checks": {"prepoint_count_matches_header": True, "finite_parsed_numbers": True,
                       "unique_node_and_block_ids": True, "node_reference_counts_valid": True,
                       "all_corner_node_references_resolve": True,
                       "all_node_references_resolve": not any(block["unresolved_noncorner_node_ids"] for block in blocks),
                       "unresolved_noncorner_node_ids": sorted({node for block in blocks for node in block["unresolved_noncorner_node_ids"]}),
                       "grid_references_resolve": True,
                       "prepoint_count": len(prepoints), "node_count": len(nodes),
                       "superblock_record_count": len(blocks)}}


def clean_icem(root: Path) -> dict:
    path = root / "data/references/multiblock_v1/dlr_f6_icem/dlr_f6_wingbody_icem_REV2.zip"
    variants = []
    tin_members = []
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("ICEM ZIP CRC failure")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP members")
        if len(names) != 4 or sum(name.endswith(".blk") for name in names) != 3 or sum(name.endswith(".tin") for name in names) != 1:
            raise ValueError("Expected exactly three BLK and one TIN members")
        for name in sorted(names):
            payload = archive.read(name)
            if name.endswith(".blk"):
                item = parse_icem_blk(payload, name)
                item["variant_id"] = name.rsplit("_", 1)[-1].split(".")[0]
                variants.append(item)
            else:
                text = payload.decode("ascii")
                version_lines = sorted(set(re.findall(r"(?m)^//.*(?:tetin format level|written by ANSYS ICEM CFD).*", text)))
                if not version_lines or "tetin format level" not in text:
                    raise ValueError("Unrecognized TIN format header")
                tin_members.append({"member": name, "member_bytes": len(payload),
                                    "member_sha256": digest(payload), "ascii_readable": True,
                                    "version_declarations": version_lines,
                                    "cad_geometry_parsed": False})
    same_geometry = len({item["geometry_signature_sha256"] for item in variants}) == 1
    if not same_geometry:
        raise ValueError("ICEM density variants do not share identical PREPOINTS")
    top_blocks = []
    top_boundaries = []
    for item in variants:
        top_blocks.extend({"variant_id": item["variant_id"], **block} for block in item.pop("blocks"))
        top_boundaries.extend({"record_kind": "ICEM_family_declaration_not_face_assignment",
                               "variant_id": item["variant_id"], **family} for family in item["families"])
    return {"schema_version": SCHEMA_VERSION, "case_id": "dlr_f6_icem",
            "sources": [source_record(path, root)],
            "coordinate_policy": {"transforms": [], "units": "not_confirmed_from_parsed_BLK_fields",
                                  "source_coordinates_preserved": True},
            "indexing": {"node_ids": "ICEM NODES new_numbering IDs, preserved verbatim",
                         "prepoint_ids": "separate PREPOINTS internal_ids namespace",
                         "block_node_order": "source lattice order: i fastest for selecting coarse corners; solver cell ordering not asserted"},
            "variants": variants, "tin_members": tin_members,
            "blocks": top_blocks, "interfaces": [], "boundaries": top_boundaries,
            "checks": {"zip_crc_valid": True, "three_density_variants_one_geometry": same_geometry,
                       "same_geometry_basis": "identical PREPOINTS records including source geometry references",
                       "variant_count": len(variants), "source_superblock_record_count": len(top_blocks),
                       "final_fluid_block_count": None, "interface_mapping_verified": False},
            "limitations": ["SUPERBLOCKS are source blocking records, not an interpreted count of active final fluid blocks.",
                            "Opaque attributes, activity/material semantics, projections, CAD associations and boundary assignments are not interpreted.",
                            "No interfaces or expert action trajectories are inferred from proprietary blocking records.",
                            "Some noncorner SUPERBLOCK node references have no explicit NODES record; retained as unresolved without interpolation or invented coordinates.",
                            "TIN is checked for CRC, ASCII readability, version declarations and member hash only; CAD entities are not parsed."]}


def parse_plot3d(text: str) -> list[dict]:
    """Parse multiblock formatted XYZ, with i varying fastest within each field."""
    tokens = re.split(r"[\s,]+", text.strip())
    if not tokens or not tokens[0]:
        raise ValueError("Empty PLOT3D file")
    block_count = integer(tokens[0])
    if block_count <= 0 or block_count > 10000 or len(tokens) < 1 + 3 * block_count:
        raise ValueError("Invalid PLOT3D block header")
    shapes = [list(map(integer, tokens[1 + 3*i:4 + 3*i])) for i in range(block_count)]
    if any(any(size < 2 for size in shape) for shape in shapes):
        raise ValueError("PLOT3D dimensions must be at least 2")
    expected_coordinates = 3 * sum(math.prod(shape) for shape in shapes)
    coordinates_expanded = array("d")
    for token in tokens[1 + 3 * block_count:]:
        if "*" in token:
            match = re.fullmatch(r"([1-9]\d*)\*([^*]+)", token)
            if not match:
                raise ValueError("Invalid Fortran repeat token")
            repeats = integer(match[1])
            value = number(match[2])
        else:
            repeats, value = 1, number(token)
        if len(coordinates_expanded) + repeats > expected_coordinates:
            raise ValueError("PLOT3D coordinate scalar count exceeds header")
        coordinates_expanded.extend(itertools.repeat(value, repeats))
    if len(coordinates_expanded) != expected_coordinates:
        raise ValueError(f"PLOT3D scalar count mismatch: {len(coordinates_expanded)} != {expected_coordinates}")
    offset = 0
    result = []
    for zone_id, shape in enumerate(shapes, 1):
        count = math.prod(shape)
        coordinates = tuple(coordinates_expanded[offset + axis*count:offset + (axis+1)*count] for axis in range(3))
        offset += 3 * count
        ni, nj, nk = shape
        corners = []
        for k, j, i in itertools.product((0, nk-1), (0, nj-1), (0, ni-1)):
            linear = i + ni * (j + nj * k)
            corners.append({"ijk_1based": [i+1, j+1, k+1],
                            "coordinates": [field[linear] for field in coordinates]})
        result.append({"id": zone_id, "dimensions": shape, "node_count": count,
                       "bounds": {"min": [min(field) for field in coordinates],
                                  "max": [max(field) for field in coordinates]},
                       "corners": corners, "_coordinates": coordinates})
    return result


def parse_gman(text: str, zone_ids: set[int]) -> tuple[list[dict], dict]:
    records = {}
    current_zone = None
    current_face = None
    declared_zones = set()
    units = None
    controls = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.upper().startswith("ZONE "):
            if current_face is not None:
                raise ValueError("Boundary missing UPDATE")
            current_zone = integer(line.split()[1])
            if current_zone not in zone_ids or current_zone in declared_zones:
                raise ValueError("GMAN duplicate or unknown zone")
            declared_zones.add(current_zone)
        elif line.upper().startswith("BOUNDARY "):
            if current_zone is None or current_face is not None:
                raise ValueError("GMAN BOUNDARY outside zone or missing UPDATE")
            current_face = line.split()[1].upper()
            if current_face not in FACES or (current_zone, current_face) in records:
                raise ValueError("Invalid or duplicate GMAN face")
            records[(current_zone, current_face)] = {"zone": current_zone, "face": current_face}
        elif line.upper() == "UPDATE":
            if current_face is None or "type" not in records[(current_zone, current_face)]:
                raise ValueError("UPDATE without boundary condition")
            current_face = None
        elif line.upper().startswith("UNITS "):
            if units is not None or current_face is not None:
                raise ValueError("Duplicate or misplaced units declaration")
            units = line.split()[1].lower()
        elif line.upper().startswith("COUPLED TO ZONE "):
            match = re.fullmatch(r"COUPLED TO ZONE (\d+) BOUNDARY (I1|IMAX|J1|JMAX|K1|KMAX)", line.upper())
            if not match or current_face is None:
                raise ValueError("Malformed COUPLED command")
            record = records[(current_zone, current_face)]
            if "type" in record:
                raise ValueError("Multiple conditions on GMAN face")
            record.update(type="COUPLED", donor_zone=int(match[1]), donor_face=match[2])
        elif line.upper() in {"OUTFLOW", "FREESTREAM", "REFLECTION", "VISCOUS WALL"}:
            if current_face is None or "type" in records[(current_zone, current_face)]:
                raise ValueError("Boundary condition outside face or duplicated")
            records[(current_zone, current_face)]["type"] = line.upper()
        elif line == "1" or line.lower().startswith("file "):
            if current_zone is not None:
                raise ValueError("Misplaced GMAN control command")
            controls.append(line)
        else:
            raise ValueError(f"Unsupported GMAN command: {line}")
    if current_face is not None or declared_zones != zone_ids:
        raise ValueError("Incomplete GMAN zone declarations")
    if set(records) != {(zone, face) for zone in zone_ids for face in FACES}:
        raise ValueError("GMAN must declare all six faces of every zone")
    for key, record in records.items():
        if record.get("type") == "COUPLED":
            donor_key = (record["donor_zone"], record["donor_face"])
            donor = records.get(donor_key, {})
            if donor_key == key or donor.get("type") != "COUPLED" or (donor.get("donor_zone"), donor.get("donor_face")) != key:
                raise ValueError("GMAN COUPLED pair is not reciprocal")
    return [records[key] for key in sorted(records)], {"units_declaration": units, "control_commands": controls}


def face_points(block: dict, face: str) -> tuple[list[int], list[list[float]]]:
    shape = block["dimensions"]
    fixed_axis = "IJK".index(face[0])
    fixed_value = 0 if face.endswith("1") else shape[fixed_axis] - 1
    axes = [axis for axis in range(3) if axis != fixed_axis]
    u_size, v_size = (shape[axis] for axis in axes)
    points = []
    for v in range(v_size):
        for u in range(u_size):
            ijk = [0, 0, 0]
            ijk[fixed_axis] = fixed_value
            ijk[axes[0]], ijk[axes[1]] = u, v
            i, j, k = ijk
            index = i + shape[0] * (j + shape[1] * k)
            points.append([field[index] for field in block["_coordinates"]])
    return [u_size, v_size], points


def match_faces(a_shape: list[int], a_points: list[list[float]],
                b_shape: list[int], b_points: list[list[float]], tolerance: float = 1e-12) -> dict:
    if math.prod(a_shape) != math.prod(b_shape):
        raise ValueError("Coupled faces have different node counts")
    candidates = []
    best = None
    for swap, reverse_u, reverse_v in itertools.product((False, True), repeat=3):
        donor_shape = b_shape[::-1] if swap else b_shape
        if donor_shape != a_shape:
            continue
        difference = 0.0
        for v in range(a_shape[1]):
            for u in range(a_shape[0]):
                mapped_u = a_shape[0] - 1 - u if reverse_u else u
                mapped_v = a_shape[1] - 1 - v if reverse_v else v
                bu, bv = (mapped_v, mapped_u) if swap else (mapped_u, mapped_v)
                first = a_points[u + a_shape[0] * v]
                second = b_points[bu + b_shape[0] * bv]
                difference = max(difference, *(abs(x-y) for x, y in zip(first, second)))
        best = difference if best is None else min(best, difference)
        if difference <= tolerance:
            candidates.append({"swap_surface_axes": swap, "reverse_source_u": reverse_u,
                               "reverse_source_v": reverse_v, "max_absolute_coordinate_difference": difference})
    return {"coordinate_tolerance_in_source_units": tolerance,
            "surface_axis_convention": "remaining logical axes in I,J,K order; u varies fastest",
            "matching_mappings": candidates, "unique_mapping_verified": len(candidates) == 1,
            "coordinates_match": bool(candidates), "best_max_absolute_coordinate_difference": best}


def clean_m6(root: Path) -> dict:
    folder = root / "data/references/multiblock_v1/onera_m6"
    grid_path, gman_path = folder / "m6wing.x.fmt", folder / "gman.a.com"
    blocks = parse_plot3d(grid_path.read_text(encoding="ascii"))
    if len(blocks) != 4:
        raise ValueError("Expected four zones in the ONERA M6 reference grid")
    records, gman = parse_gman(gman_path.read_text(encoding="ascii"), {block["id"] for block in blocks})
    by_id = {block["id"]: block for block in blocks}
    interfaces = []
    boundaries = []
    for record in records:
        shape, points = face_points(by_id[record["zone"]], record["face"])
        item = {**record, "surface_dimensions": shape, "node_count": len(points)}
        if record["type"] != "COUPLED":
            boundaries.append(item)
        elif (record["zone"], record["face"]) < (record["donor_zone"], record["donor_face"]):
            donor_shape, donor_points = face_points(by_id[record["donor_zone"]], record["donor_face"])
            mapping = match_faces(shape, points, donor_shape, donor_points)
            if not mapping["coordinates_match"] or not mapping["unique_mapping_verified"]:
                raise ValueError("ONERA M6 coupled interface lacks a unique coordinate-matched mapping")
            interfaces.append({**item, "donor_surface_dimensions": donor_shape,
                               "reciprocal_declaration_verified": True, "mapping_check": mapping})
    if len(interfaces) != 6:
        raise ValueError("Expected six unique coupled interfaces in the ONERA M6 reference")
    for block in blocks:
        del block["_coordinates"]
        block["coordinate_source"] = "PLOT3D source file; full coordinates not duplicated in JSON"
    return {"schema_version": SCHEMA_VERSION, "case_id": "onera_m6",
            "sources": [source_record(grid_path, root), source_record(gman_path, root)],
            "coordinate_policy": {"transforms": [], "plot3d_coordinate_units": "not_independently_confirmed",
                                  "gman_units_declaration": gman["units_declaration"],
                                  "gman_declaration_applied_as_coordinate_conversion": False,
                                  "source_coordinates_preserved": True},
            "indexing": {"zone_ids": "1-based PLOT3D source order", "logical_indices": "1-based",
                         "field_order": "each zone: X then Y then Z; i varies fastest, then j, then k",
                         "faces": "I1/IMAX/J1/JMAX/K1/KMAX"},
            "blocks": blocks, "interfaces": interfaces, "boundaries": boundaries,
            "gman_declarations": {**gman, "faces": records},
            "checks": {"finite_coordinates": True, "exact_coordinate_scalar_count": 3 * sum(block["node_count"] for block in blocks),
                       "zone_count": len(blocks), "node_count": sum(block["node_count"] for block in blocks),
                       "all_24_faces_declared_once": len(records) == 24,
                       "directed_coupled_face_count": 2 * len(interfaces), "unique_interface_count": len(interfaces),
                       "all_couplings_reciprocal": True, "coupled_face_node_counts_equal": True,
                       "coordinate_matched_interface_count": sum(item["mapping_check"]["coordinates_match"] for item in interfaces),
                       "uniquely_mapped_interface_count": sum(item["mapping_check"]["unique_mapping_verified"] for item in interfaces)},
            "limitations": ["GMAN units feet is recorded as a command declaration, not used to rescale PLOT3D coordinates.",
                            "Surface mappings are coordinate-verified candidates, not explicit transforms present in GMAN.",
                            "No cell Jacobians, CFD solve, expert actions or training trajectories are generated."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, help="Output directory, relative to repo root unless absolute")
    parser.add_argument("--check", action="store_true", help="Recompute and compare JSON without writing anything")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = args.output or Path("data/processed/block_topology_v1")
    if not output.is_absolute():
        output = root / output
    results = [("dlr_f6_icem.json", clean_icem(root)), ("onera_m6.json", clean_m6(root))]
    if args.check:
        for name, result in results:
            path = output / name
            if not path.is_file() or path.read_bytes() != stable_bytes(result):
                raise ValueError(f"Generated data differs: {path}")
            print(f"Verified {path.relative_to(root) if path.is_relative_to(root) else path}")
    else:
        output.mkdir(parents=True, exist_ok=True)
        for name, result in results:
            path = output / name
            path.write_bytes(stable_bytes(result))
            print(f"Wrote {path.relative_to(root) if path.is_relative_to(root) else path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
