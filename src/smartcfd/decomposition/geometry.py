"""Exact-coordinate orthogonal polygon cuts, without a square grid or mesher.

Input rings are closed and oriented canonically; redundant collinear vertices
are removed. Geometry itself is never rounded. Only identity hashes quantize
uniformly normalized coordinates (12 decimal places) to ignore floating point
noise from translating/uniformly scaling an otherwise identical synthetic case.
Anisotropic scaling changes aspect ratios and therefore changes the identity.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from itertools import combinations

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import split, unary_union

AREA_TOLERANCE = 1e-10
HORIZONTAL, VERTICAL = 0, 1


class GeometryError(ValueError):
    """The ring or proposed physical cut does not meet the geometric contract."""


def _signed_area(points):
    # Translate first to avoid cancellation for a small polygon far from zero.
    ox, oy = points[0]
    return sum((a[0] - ox) * (b[1] - oy) - (b[0] - ox) * (a[1] - oy)
               for a, b in zip(points, points[1:] + points[:1])) / 2


def canonical_vertices(vertices):
    """Return a closed, simple CCW orthogonal ring starting at its minimum vertex.

    Open coordinate lists are explicitly closed. Clockwise lists are reversed.
    Nested rings/holes, nonfinite values, diagonal edges and self intersections
    are rejected; axis alignment is checked without silently snapping points.
    """
    try:
        points = []
        for point in vertices:
            if len(point) != 2 or any(isinstance(v, bool) for v in point):
                raise GeometryError("Each vertex must have two finite numeric coordinates")
            xy = (float(point[0]), float(point[1]))
            if not all(math.isfinite(value) for value in xy):
                raise GeometryError("Nonfinite polygon coordinate")
            if not points or xy != points[-1]:
                points.append(xy)
    except (TypeError, ValueError, OverflowError, IndexError) as exc:
        raise GeometryError("Expected one coordinate ring; holes are unsupported") from exc
    if points and points[-1] == points[0]:
        points.pop()
    if len(points) < 4:
        raise GeometryError("An orthogonal polygon needs at least four vertices")
    for a, b in zip(points, points[1:] + points[:1]):
        if (a[0] == b[0]) == (a[1] == b[1]):
            raise GeometryError("Every nonzero edge must be horizontal or vertical")
    polygon = Polygon(points)
    if not polygon.is_valid or not polygon.exterior.is_simple or polygon.interiors:
        raise GeometryError("Polygon must be simple, connected and without holes")
    if not math.isfinite(polygon.area) or polygon.area <= 0:
        raise GeometryError("Polygon must have positive finite area")
    xmin, ymin, xmax, ymax = polygon.bounds
    width, height = xmax - xmin, ymax - ymin
    if (not all(math.isfinite(value) and value > 0 for value in (width, height))
            or not math.isfinite(max(width, height) / min(width, height))
            or not math.isfinite(polygon.length)
            or not all(math.isfinite(value) for value in (polygon.centroid.x, polygon.centroid.y))):
        raise GeometryError("Coordinates are too extreme for finite geometric metrics")
    # Only remove middle points along an already validated straight edge.
    while True:
        reduced = [b for a, b, c in zip(points[-1:] + points[:-1], points, points[1:] + points[:1])
                   if not (a[0] == b[0] == c[0] or a[1] == b[1] == c[1])]
        if len(reduced) == len(points):
            break
        points = reduced
        if len(points) < 4:
            raise GeometryError("Degenerate collinear polygon")
    if _signed_area(points) < 0:
        points.reverse()
    start = min(range(len(points)), key=points.__getitem__)
    points = points[start:] + points[:start]
    return tuple([*points, points[0]])


def as_polygon(vertices):
    if isinstance(vertices, Polygon):
        if vertices.interiors:
            raise GeometryError("Holes are unsupported")
        vertices = vertices.exterior.coords
    return Polygon(canonical_vertices(vertices))


def normalized_vertices(vertices, bounds=None):
    ring = canonical_vertices(vertices)
    polygon = Polygon(ring)
    xmin, ymin, xmax, ymax = bounds or polygon.bounds
    width, height = xmax - xmin, ymax - ymin
    if min(width, height) <= 0:
        raise GeometryError("Degenerate bounding box")
    scale = max(width, height)
    return tuple(((x - xmin) / scale, (y - ymin) / scale) for x, y in ring)


def polygon_hash(vertices):
    points = [[round(x, 12), round(y, 12)] for x, y in normalized_vertices(vertices)]
    return hashlib.sha256(json.dumps(points, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def is_rectangle(vertices):
    ring = canonical_vertices(vertices)
    return len(ring) == 5


def aspect_ratio(vertices):
    xmin, ymin, xmax, ymax = as_polygon(vertices).bounds
    return max(xmax - xmin, ymax - ymin) / min(xmax - xmin, ymax - ymin)


def _line_segments(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type in ("LineString", "LinearRing"):
        coordinates = list(geometry.coords)
        return [(tuple(a), tuple(b)) for a, b in zip(coordinates, coordinates[1:]) if a != b]
    if hasattr(geometry, "geoms"):
        return [segment for item in geometry.geoms for segment in _line_segments(item)]
    return []


def _point_on_segment(point, segment, tolerance):
    a, b = segment
    if a[0] == b[0]:
        return abs(point[0] - a[0]) <= tolerance and min(a[1], b[1]) - tolerance <= point[1] <= max(a[1], b[1]) + tolerance
    return abs(point[1] - a[1]) <= tolerance and min(a[0], b[0]) - tolerance <= point[0] <= max(a[0], b[0]) + tolerance


def partition_quality(root, parts):
    original = as_polygon(root)
    polygons = [as_polygon(part) for part in parts]
    if not polygons:
        raise GeometryError("A partition cannot be empty")
    joined = unary_union(polygons)
    sum_area = sum(part.area for part in polygons)
    coverage_error = original.symmetric_difference(joined).area
    overlap_area = sum(a.intersection(b).area for a, b in combinations(polygons, 2))
    tolerance = original.area * AREA_TOLERANCE
    areas = [part.area for part in polygons]
    mean = sum_area / len(areas)
    ratios = [aspect_ratio(part.exterior.coords) for part in polygons]
    rectangles = sum(is_rectangle(part.exterior.coords) for part in polygons)
    valid = (coverage_error <= tolerance and overlap_area <= tolerance
             and abs(sum_area - original.area) <= tolerance
             and all(part.area > tolerance for part in polygons))
    return {"valid": bool(valid), "coverage_error_area": float(coverage_error),
            "relative_coverage_error": float(coverage_error / original.area), "overlap_area": float(overlap_area),
            "sum_area_error": float(abs(sum_area - original.area)), "root_area": float(original.area),
            "total_area": float(sum_area), "block_count": len(polygons), "rectangle_count": rectangles,
            "pending_count": len(polygons) - rectangles, "aspect_ratio_mean": float(sum(ratios) / len(ratios)),
            "aspect_ratio_max": float(max(ratios)),
            "area_cv": float(math.sqrt(sum((value - mean)**2 for value in areas) / len(areas)) / mean),
            "scope": "Geometric partition coverage, pairwise area overlap and bbox aspect ratios; no structured mesh or CFD"}


@dataclass(frozen=True)
class CutResult:
    children: tuple
    direction: int
    coordinate: float
    source_vertex: tuple
    segments: tuple


def split_polygon(vertices, vertex_index, direction, *, root_area=None):
    """Split by the full H/V line through a real vertex, retaining every child.

    Shapely split can return more than two polygons for a concave input. A cut
    along only an exterior edge, a zero-effect cut or any sliver is rejected.
    """
    ring = canonical_vertices(vertices)
    if not isinstance(vertex_index, int) or isinstance(vertex_index, bool) or not 0 <= vertex_index < len(ring) - 1:
        raise GeometryError("Cut vertex index is outside the actual ring")
    if isinstance(direction, bool) or direction not in (HORIZONTAL, VERTICAL):
        raise GeometryError("Cut direction must be 0 (horizontal) or 1 (vertical)")
    polygon = Polygon(ring)
    x, y = ring[vertex_index]
    xmin, ymin, xmax, ymax = polygon.bounds
    margin = max(xmax - xmin, ymax - ymin)
    endpoints = ((xmin - margin, y), (xmax + margin, y)) if direction == HORIZONTAL else ((x, ymin - margin), (x, ymax + margin))
    if not all(math.isfinite(v) for point in endpoints for v in point):
        raise GeometryError("Coordinates too extreme for a finite full cut line")
    try:
        pieces = list(split(polygon, LineString(endpoints)).geoms)
    except Exception as exc:
        raise GeometryError("Physical line split failed") from exc
    if len(pieces) < 2 or any(part.geom_type != "Polygon" or part.interiors for part in pieces):
        raise GeometryError("Line has no positive-area partition effect")
    threshold = (polygon.area if root_area is None else float(root_area)) * AREA_TOLERANCE
    if any(part.area <= threshold for part in pieces):
        raise GeometryError("Cut creates a sliver; no child is silently discarded")
    children = tuple(sorted(canonical_vertices(part.exterior.coords) for part in pieces))
    quality = partition_quality(ring, children)
    if not quality["valid"]:
        raise GeometryError("Split failed union coverage, area or nonoverlap validation")
    segments = []
    for a, b in combinations([Polygon(child) for child in children], 2):
        segments.extend(_line_segments(a.boundary.intersection(b.boundary)))
    segments = tuple(sorted(set(tuple(sorted(segment)) for segment in segments)))
    if not segments:
        raise GeometryError("No interior shared cut interface")
    return CutResult(children, direction, y if direction == HORIZONTAL else x, (x, y), segments)


def paper_reward(children):
    """Eq.1-inspired immediate reward; R_i is the child bbox aspect ratio.

    N and areas refer to *all* actual children, including nonrectangles.
    Step/block costs and the completion bonus belong to the environment and
    are reported separately, rather than being attributed to the paper.
    """
    polygons = [as_polygon(child) for child in children]
    n = len(polygons)
    if n < 2:
        raise GeometryError("Paper cut reward requires an effective split")
    areas = [part.area for part in polygons]
    ratios = [aspect_ratio(part.exterior.coords) for part in polygons]
    mean = sum(areas) / n
    shape_term = math.sqrt(n / sum(value**2 for value in ratios))
    imbalance = math.sqrt(sum((value - mean)**2 for value in areas)) / sum(areas)
    rectangles = sum(is_rectangle(part.exterior.coords) for part in polygons)
    value = 3 * (shape_term - imbalance - 1) + 10 * rectangles / n
    return {"paper_reward": float(value), "children": n, "rectangle_children": rectangles,
            "shape_term": float(shape_term), "imbalance_term": float(imbalance),
            "aspect_definition": "R_i=max(bbox width,height)/min(bbox width,height)"}


def partition_interfaces(parts, statuses=None):
    """Explicit common split points, elementary interface segments and T joints.

    Rings need not have matching vertices on an adjacent long edge. Every
    interface is subdivided at vertices from all parts, so such T junctions
    are preserved in the export instead of implying a conforming mesh.
    """
    rings = [canonical_vertices(part) for part in parts]
    polygons = [Polygon(ring) for ring in rings]
    if not polygons:
        return {"interfaces": [], "junctions": []}
    bounds = unary_union(polygons).bounds
    tolerance = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 1e-12
    vertices = sorted(set(point for ring in rings for point in ring[:-1]))
    interfaces = []
    for i, j in combinations(range(len(polygons)), 2):
        segments = []
        for segment in _line_segments(polygons[i].boundary.intersection(polygons[j].boundary)):
            points = sorted(point for point in vertices if _point_on_segment(point, segment, tolerance))
            points = sorted(set([*points, *segment]), key=lambda p: (p[0], p[1]))
            segments.extend([[list(a), list(b)] for a, b in zip(points, points[1:]) if a != b])
        if segments:
            interfaces.append({"part_a": i, "part_b": j, "status_a": statuses[i] if statuses else None,
                               "status_b": statuses[j] if statuses else None, "segments": segments,
                               "common_points": sorted({tuple(point) for segment in segments for point in segment})})
    junctions = []
    for point in vertices:
        incident = [i for i, polygon in enumerate(polygons) if polygon.boundary.distance(Point(point)) <= tolerance]
        if len(incident) < 3:
            continue
        rays = set()
        for i in incident:
            for segment in zip(rings[i], rings[i][1:]):
                if _point_on_segment(point, segment, tolerance):
                    for end in segment:
                        dx, dy = end[0] - point[0], end[1] - point[1]
                        if abs(dx) > tolerance:
                            rays.add((1 if dx > 0 else -1, 0))
                        if abs(dy) > tolerance:
                            rays.add((0, 1 if dy > 0 else -1))
        if len(rays) >= 3:
            junctions.append({"point": list(point), "parts": incident,
                              "kind": "T" if len(rays) == 3 else "cross", "ray_count": len(rays)})
    return {"interfaces": interfaces, "junctions": junctions}
