"""Clean raw UIUC DAT files without smoothing, resampling, or adding vertices.

Python standard library only. Source decimal coordinates are preserved as exact
fractions for canonical duplicate detection; JSON polygons use float64 numbers.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re

VERSION = "1.0.0"
SOURCE_URL = "https://m-selig.ae.illinois.edu/ads/coord_database.html"


@dataclass(frozen=True)
class Thresholds:
    min_vertices: int = 12
    max_vertices: int = 5000
    min_normalized_area: float = 1e-5
    max_normalized_y_span: float = 0.60
    max_edge_length: float = 0.40
    min_edge_length: float = 1e-10
    open_endpoint_x_tolerance: float = 0.02
    max_open_closure_gap: float = 0.10
    max_branch_x_backtracking: float = 0.02
    intersection_epsilon: float = 1e-12
    lednicer_le_join_tolerance: float = 1e-6


class Rejected(ValueError):
    def __init__(self, reason, detail=None):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def parse_dat(raw: bytes, limits=Thresholds()):
    """Parse a title, optional # comments, DAT pairs and Lednicer count headers."""
    text = raw.decode("latin-1").replace("\x1a", "")
    title = []
    points = []
    for number, original in enumerate(text.splitlines(), 1):
        line = original.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = re.split(r"[\s,]+", line)
        values = None
        try:
            values = [float(t.replace("D", "E").replace("d", "e")) for t in tokens]
        except ValueError:
            pass
        if values is None or len(values) != 2:
            if not points and values is None:
                title.append(line)
                continue
            raise Rejected("malformed_coordinate_line", {"line_number": number})
        if not all(math.isfinite(v) for v in values):
            raise Rejected("non_finite_coordinate", {"line_number": number})
        points.append(tuple(Fraction(t.replace("D", "E").replace("d", "e")) for t in tokens))
    if not points:
        raise Rejected("no_coordinates")
    fmt = "selig_contour"
    a, b = points[0]
    if a.denominator == b.denominator == 1 and 3 <= a <= 100000 and 3 <= b <= 100000:
        na, nb = int(a), int(b)
        if len(points) - 1 != na + nb:
            raise Rejected("lednicer_count_mismatch", {"declared": na + nb, "actual": len(points) - 1})
        upper, lower = points[1:1 + na], points[1 + na:]
        # Standard Lednicer lists each surface separately. Join at the LE.
        upper = upper if upper[0][0] >= upper[-1][0] else upper[::-1]
        lower = lower if lower[0][0] <= lower[-1][0] else lower[::-1]
        span = float(max(p[0] for p in points[1:]) - min(p[0] for p in points[1:]))
        gap = math.dist(tuple(map(float, upper[-1])), tuple(map(float, lower[0])))
        if span <= 0 or gap / span > limits.lednicer_le_join_tolerance:
            raise Rejected("lednicer_unmatched_leading_edges")
        points = upper + lower
        fmt = "lednicer_separate_surfaces"
    return points, " | ".join(title), fmt


def remove_repeated(points):
    result = []
    consecutive = 0
    for p in points:
        if result and result[-1] == p:
            consecutive += 1
        else:
            result.append(p)
    endpoint = 0
    while len(result) > 1 and result[-1] == result[0]:
        result.pop()
        endpoint += 1
    return result, consecutive, endpoint


def signed_area(points):
    return 0.5 * sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1]))


def cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(a, b, c, d, eps=1e-12):
    if (max(a[0], b[0]) + eps < min(c[0], d[0]) or
            max(c[0], d[0]) + eps < min(a[0], b[0]) or
            max(a[1], b[1]) + eps < min(c[1], d[1]) or
            max(c[1], d[1]) + eps < min(a[1], b[1])):
        return False
    u, v, w, z = cross(a, b, c), cross(a, b, d), cross(c, d, a), cross(c, d, b)
    if ((u > eps and v < -eps) or (u < -eps and v > eps)) and ((w > eps and z < -eps) or (w < -eps and z > eps)):
        return True
    def on_segment(p, q, r, value):
        return abs(value) <= eps and min(p[0], q[0]) - eps <= r[0] <= max(p[0], q[0]) + eps and min(p[1], q[1]) - eps <= r[1] <= max(p[1], q[1]) + eps
    return (on_segment(a, b, c, u) or on_segment(a, b, d, v) or
            on_segment(c, d, a, w) or on_segment(c, d, b, z))


def find_intersection(points, eps=1e-12):
    n = len(points)
    for i in range(n):
        a, b = points[i], points[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if segments_intersect(a, b, points[j], points[(j + 1) % n], eps):
                return [i, j]
    return None


def branch_backtracking(points):
    """Total departure from x-monotone LE-to-TE paths, measured in chord units."""
    le = min(range(len(points)), key=lambda i: points[i])
    te = max(range(len(points)), key=lambda i: points[i][0])
    def path(step):
        indices = [le]
        while indices[-1] != te:
            indices.append((indices[-1] + step) % len(points))
        return sum(max(0.0, points[a][0] - points[b][0]) for a, b in zip(indices, indices[1:]))
    return [path(1), path(-1)]


def canonicalize(points):
    """Exact axis-aligned, unit-chord canonical form, invariant to cyclic order."""
    xmin, xmax = min(p[0] for p in points), max(p[0] for p in points)
    chord = xmax - xmin
    if chord <= 0:
        raise Rejected("zero_chord")
    yle = min(p[1] for p in points if p[0] == xmin)
    exact = [((x - xmin) / chord, (y - yle) / chord) for x, y in points]
    floats = [tuple(map(float, p)) for p in exact]
    if signed_area(floats) < 0:
        exact.reverse()
    first = min(range(len(exact)), key=lambda i: exact[i])
    exact = exact[first:] + exact[:first]
    serial = ";".join(f"{x.numerator}/{x.denominator},{y.numerator}/{y.denominator}" for x, y in exact)
    digest = hashlib.sha256(serial.encode("ascii")).hexdigest()
    transform = {"type": "axis_aligned_uniform_unit_chord", "x_offset": float(xmin),
                 "y_offset": float(yle), "original_x_chord_span": float(chord),
                 "uniform_scale": float(1 / chord), "rotation_degrees": 0,
                 "exact_offsets_and_chord": {"x_offset": str(xmin), "y_offset": str(yle), "chord": str(chord)},
                 "formula": "x'=(x-x_min)/chord; y'=(y-y_at_lexicographic_LE)/chord"}
    return [list(map(float, p)) for p in exact], digest, transform


def clean_geometry(raw, limits=Thresholds()):
    original, header, fmt = parse_dat(raw, limits)
    points, consecutive, terminal = remove_repeated(original)
    if len(points) < limits.min_vertices:
        raise Rejected("too_few_vertices", {"vertices": len(points)})
    if len(points) > limits.max_vertices:
        raise Rejected("too_many_vertices", {"vertices": len(points)})
    polygon, digest, transform = canonicalize(points)
    area = signed_area(polygon)
    if area < limits.min_normalized_area:
        raise Rejected("degenerate_or_tiny_area", {"area": area})
    yspan = max(p[1] for p in polygon) - min(p[1] for p in polygon)
    if yspan > limits.max_normalized_y_span:
        raise Rejected("outside_single_airfoil_y_span", {"y_span": yspan})
    edge_lengths = [math.dist(a, b) for a, b in zip(polygon, polygon[1:] + polygon[:1])]
    if min(edge_lengths) <= limits.min_edge_length:
        raise Rejected("near_zero_length_edge", {"min_edge_length": min(edge_lengths)})
    if max(edge_lengths) > limits.max_edge_length:
        raise Rejected("long_edge_or_disconnected_contours", {"max_edge_length": max(edge_lengths)})
    # An open Selig chain must end near its TE; arbitrary gaps are not repaired.
    raw_chord = transform["original_x_chord_span"]
    closure_gap = math.dist(tuple(map(float, original[0])), tuple(map(float, original[-1]))) / raw_chord
    xmin = min(p[0] for p in points)
    xmax = max(p[0] for p in points)
    endpoints_from_te = [float((xmax - p[0]) / (xmax - xmin)) for p in (points[0], points[-1])]
    if not terminal:
        if max(endpoints_from_te) > limits.open_endpoint_x_tolerance:
            raise Rejected("open_contour_endpoints_away_from_te", {"endpoint_x_distance_from_te": endpoints_from_te})
        if closure_gap > limits.max_open_closure_gap:
            raise Rejected("excessive_open_trailing_edge_gap", {"closure_gap": closure_gap})
    eps = limits.intersection_epsilon
    for i, b in enumerate(polygon):
        a, c = polygon[i - 1], polygon[(i + 1) % len(polygon)]
        dot = (b[0] - a[0]) * (c[0] - b[0]) + (b[1] - a[1]) * (c[1] - b[1])
        if abs(cross(a, b, c)) <= eps and dot < -eps:
            raise Rejected("adjacent_edge_overlap", {"vertex": i})
    crossing = find_intersection(polygon, eps)
    if crossing is not None:
        raise Rejected("self_intersection_or_nonadjacent_touch", {"edge_indices": crossing})
    backtracking = branch_backtracking(polygon)
    if max(backtracking) > limits.max_branch_x_backtracking:
        raise Rejected("outside_single_airfoil_x_monotonicity", {"branch_x_backtracking": backtracking})
    flags = {"consecutive_exact_duplicates_removed": consecutive,
             "explicit_closing_vertices_removed": terminal,
             "raw_representation": "explicitly_closed" if terminal else "open_te_chain",
             "raw_normalized_last_to_first_distance": closure_gap,
             "open_chain_has_implicit_te_segment": not bool(terminal),
             "smoothing_applied": False, "resampling_applied": False,
             "missing_points_imputed": False}
    quality = {"vertex_count": len(polygon), "signed_area": area, "x_span": 1.0,
               "y_span": yspan, "min_edge_length": min(edge_lengths),
               "max_edge_length": max(edge_lengths), "branch_x_backtracking": backtracking,
               "self_intersection_check": "passed_with_configured_tolerance",
               "adjacent_overlap_check": "passed_with_configured_tolerance"}
    return {"polygon": polygon, "canonical_sha256": digest, "quality": quality}, {
        "header": header, "parse_format": fmt, "normalization": transform, "quality_flags": flags}


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def run(input_root, output_root, limits=Thresholds()):
    input_root, output_root = Path(input_root).resolve(), Path(output_root).resolve()
    if not input_root.is_dir():
        raise ValueError("--input must be the raw DAT directory")
    if input_root == output_root or input_root.is_relative_to(output_root) or output_root.is_relative_to(input_root):
        raise ValueError("Input and output must be separate, non-nested directories")
    output_root.mkdir(parents=True, exist_ok=True)
    samples = {}
    accepted, rejected, duplicates = [], [], []
    paths = sorted((p for p in input_root.rglob("*") if p.is_file() and p.suffix.lower() == ".dat"), key=lambda p: p.relative_to(input_root).as_posix())
    if not paths:
        raise ValueError("No DAT source files found")
    for path in paths:
        raw = path.read_bytes()
        source = {"relative_path": path.relative_to(input_root).as_posix(), "filename": path.name,
                  "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        try:
            geometry, provenance = clean_geometry(raw, limits)
        except Rejected as err:
            rejected.append({**source, "status": "rejected", "reason": err.reason, "detail": err.detail})
            continue
        source.update(provenance)
        digest = geometry["canonical_sha256"]
        sid = "uiuc_" + digest[:16]
        if digest not in samples:
            samples[digest] = {"sample_id": sid, "geometry_only": True, "closure": "implicit_last_to_first",
                               "orientation": "CCW", "canonical_start": "lexicographic_minimum_x_then_y",
                               **geometry, "aliases": [], "source_files": []}
            status = "accepted_unique"
        else:
            status = "accepted_exact_duplicate"
            duplicates.append({"sample_id": sid, "canonical_sha256": digest, "duplicate_source": source,
                               "representative_relative_path": samples[digest]["source_files"][0]["relative_path"],
                               "criterion": "exact_rational_normalized_polygon_same_vertices_up_to_winding_and_cyclic_start"})
        samples[digest]["aliases"].append(path.stem)
        samples[digest]["source_files"].append(source)
        accepted.append({"sample_id": sid, "canonical_sha256": digest, "status": status, **source})
    ordered = sorted(samples.values(), key=lambda row: row["sample_id"])
    assert len({s["sample_id"] for s in ordered}) == len(ordered), "Truncated ID collision"
    write_jsonl(output_root / "samples.jsonl", ordered)
    write_jsonl(output_root / "accepted_manifest.jsonl", accepted)
    write_jsonl(output_root / "rejected_manifest.jsonl", rejected)
    write_jsonl(output_root / "duplicate_manifest.jsonl", duplicates)
    summary = {"pipeline_version": VERSION, "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "source": {"name": "UIUC Airfoil Coordinates Database", "source_page": SOURCE_URL,
                          "input_path_policy": "Source paths are relative to the CLI --input directory; no machine-specific paths stored.",
                          "data_license": "No uniform data reuse license established; code license does not relicense third-party geometry."},
               "input_dat_files": len(paths), "accepted_source_files": len(accepted), "unique_samples": len(ordered),
               "duplicate_source_files": len(duplicates), "rejected_source_files": len(rejected),
               "rejection_reasons": dict(sorted(Counter(r["reason"] for r in rejected).items())),
               "thresholds": asdict(limits),
               "threshold_interpretation": "Operational filters for a conventional single-element, chord-aligned training subset, not universal CFD validity standards.",
               "representation": {"polygon": "Unit-x-chord CCW polygon, canonical LE start, no duplicate terminal vertex; closure is implicit last-to-first segment.",
                                  "normalization": "Translation plus uniform positive scale only; no rotation, smoothing, resampling or imputation.",
                                  "deduplication": "Exact decimal/Fraction arithmetic after normalization, orientation and cyclic ordering. Vertex sampling must match exactly; near-duplicates and collinear-resampled equivalents are retained.",
                                  "serialization": "Polygon floats are for convenient use; canonical SHA256 is computed from exact rational coordinates before float conversion."},
               "limitations": ["Geometry only; no expert actions, block topology, connectivity or CFD ground truth.",
                               "Implicit closure defines an edge between input endpoints; no points are inserted. For accepted open chains both endpoints must be near the trailing edge and gap within threshold.",
                               "Rejected means outside this conservative subset or needs review; it does not prove the original airfoil is wrong.",
                               "Segment tests use a configurable floating-point epsilon; no exact-predicate robustness claim.",
                               "No curvature, boundary-layer suitability, grid Jacobian, aerodynamic performance or CAD-validity guarantee.",
                               "Near-duplicate/family-aware train-test grouping must be performed separately."]}
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Directory containing original UIUC DAT coordinates")
    parser.add_argument("--output", type=Path, required=True, help="Separate directory for generated JSON/JSONL outputs")
    parser.add_argument("--thresholds-json", type=Path, help="Optional JSON object overriding Thresholds defaults")
    args = parser.parse_args()
    overrides = json.loads(args.thresholds_json.read_text(encoding="utf-8")) if args.thresholds_json else {}
    result = run(args.input, args.output, Thresholds(**overrides))
    print(json.dumps({k: result[k] for k in ("input_dat_files", "accepted_source_files", "unique_samples", "duplicate_source_files", "rejected_source_files", "rejection_reasons")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
