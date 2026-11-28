"""Physical curve checks for a declared, restricted C-coordinate mapping.

UV control vertices remain topological corners. Dense physical samples belong
to shared curve records and are never promoted to block corners. The reference
domain is built from the farfield boundary and original airfoil hole, not from
the union of candidate blocks.
"""
from __future__ import annotations

from collections import OrderedDict
import math
import numpy as np
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union
from shapely.errors import GEOSException

from smartcfd.geometry.domain import make_domain, CMapping
from smartcfd.partition.constraints import validate_graph
from smartcfd.partition.topology import edge_incidence, initial_graph
from .config import AirfoilDecompositionConfig, PARTITION_VERSION


def _length(points):
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _remember(cache, key, value, limit):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)
    return value


class CurveGeometry:
    """Bounded caches for independently checked physical images of UV faces."""
    def __init__(self, sample, config=None):
        self.config = config or AirfoilDecompositionConfig()
        self.domain = make_domain(sample, farfield=self.config.farfield, stretch=self.config.stretch)
        self.domain.metadata['protected_thickness'] = self.config.protected_thickness
        self.mapping = CMapping(self.domain)
        self._curves, self._faces, self._roots = OrderedDict(), OrderedDict(), OrderedDict()
        self._evaluation_polygons = OrderedDict()
        self._curve_limit, self._face_limit = 2048, 1024
        self._u_breaks = np.unique(np.r_[0., .25, .5, .75, 1.,
            .25 + .25*self.mapping.lower[0], .5 + .25*self.mapping.upper[0]])
        self._protected = initial_graph(self.domain)
        self._wall = Polygon(np.asarray(self.domain.polygon, dtype=float))

    def _sample(self, a_uv, b_uv, refined=False):
        a, b = tuple(map(float, a_uv)), tuple(map(float, b_uv))
        reverse = a > b
        if reverse:
            a, b = b, a
        key = (a, b, bool(refined))
        if key not in self._curves:
            start, delta = np.asarray(a), np.asarray(b)-np.asarray(a)
            if not np.isfinite([*a, *b]).all() or np.linalg.norm(delta) == 0:
                raise ValueError('Curve endpoints must be finite and distinct')
            parameters = [0., 1.]
            if delta[0]:
                parameters.extend(float((u-start[0])/delta[0]) for u in self._u_breaks
                                  if 0 < (u-start[0])/delta[0] < 1)
            if delta[1] and 0 < (.5-start[1])/delta[1] < 1:
                parameters.append(float((.5-start[1])/delta[1]))
            ts = np.unique(parameters)
            tolerance = self.config.curve_tolerance / (2 if refined else 1)
            for depth in range(self.config.curve_max_depth+1):
                points = self.mapping.evaluate(start + ts[:, None]*delta)
                mids = (ts[:-1]+ts[1:])/2
                fractions = np.array([.25, .5, .75])[:, None]
                probes = ts[:-1] + fractions*(ts[1:]-ts[:-1])
                actual = self.mapping.evaluate(start + probes[..., None]*delta)
                linear = points[:-1] + fractions[..., None]*(points[1:]-points[:-1])
                differences = actual-linear
                error = np.hypot(differences[..., 0], differences[..., 1]).max(axis=0)
                # Parameter-midpoint error also resolves nonlinear spacing on
                # straight wake/radial curves for reliable arc-length inversion.
                needs = error > tolerance
                if not needs.any():
                    break
                if depth == self.config.curve_max_depth:
                    raise ValueError('Adaptive curve sampling exhausted its depth budget')
                if len(ts)+int(needs.sum()) > 65536:
                    raise ValueError('Adaptive curve sampling exceeded its point budget')
                ts = np.sort(np.r_[ts, mids[needs]])
            if not np.isfinite(points).all():
                raise ValueError('Nonfinite physical curve')
            points.setflags(write=False); ts.setflags(write=False)
            _remember(self._curves, key, (ts, points), self._curve_limit)
        else:
            self._curves.move_to_end(key)
        ts, points = self._curves[key]
        return (1-ts[::-1], points[::-1]) if reverse else (ts, points)

    def curve(self, a_uv, b_uv, refined=False):
        """Mapped UV line with mandatory branch knots and adaptive samples."""
        return self._sample(a_uv, b_uv, refined)[1]

    def arc_point(self, a_uv, b_uv, fraction):
        if not math.isfinite(fraction) or not 0 < fraction < 1:
            raise ValueError('An arc fraction must be strictly between zero and one')
        ts, points = self._sample(a_uv, b_uv, True)
        lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        if not np.isfinite(lengths[-1]) or lengths[-1] <= 0:
            raise ValueError('A curve needs positive finite physical length')
        t = np.interp(fraction*lengths[-1], lengths, ts)
        return tuple((np.asarray(a_uv)+t*(np.asarray(b_uv)-np.asarray(a_uv))).tolist())

    def _face(self, graph, face, refined):
        uv = tuple(tuple(graph.vertices[v]) for v in face.vertices)
        key = (uv, bool(refined))
        if key not in self._faces:
            curves = [self.curve(a, b, refined) for a, b in zip(uv, uv[1:]+uv[:1])]
            boundary = np.vstack([curve[:-1] for curve in curves])
            polygon = Polygon(boundary)
            lengths = [_length(curve) for curve in curves]
            angles = []
            for i, curve in enumerate(curves):
                incoming = curves[i-1][-1]-curves[i-1][-2]
                outgoing = curve[1]-curve[0]
                scale = np.linalg.norm(incoming)*np.linalg.norm(outgoing)
                angles.append(float(abs(incoming[0]*outgoing[1]-incoming[1]*outgoing[0])/scale) if scale > 0 else 0.)
            value = (polygon, boundary, lengths, angles)
            _remember(self._faces, key, value, self._face_limit)
        else:
            self._faces.move_to_end(key)
        return self._faces[key]

    def _reference(self, graph, refined):
        # Sample the independently defined C farfield using boundary anchors.
        # Sharing these knots with face edges avoids comparing two different
        # polygonal approximations of the same curved farfield at area tolerance.
        left = sorted({0., 1., *(float(v) for u, v in graph.vertices.values() if abs(u) < 1e-12)})
        top = sorted({0., 1., *(float(u) for u, v in graph.vertices.values() if abs(v-1) < 1e-12)})
        right = sorted({0., 1., *(float(v) for u, v in graph.vertices.values() if abs(u-1) < 1e-12)}, reverse=True)
        anchors = [(0., v) for v in left]+[(u, 1.) for u in top[1:]]+[(1., v) for v in right[1:]]
        key = (tuple(anchors), bool(refined))
        if key not in self._roots:
            pieces = [self.curve(a, b, refined) for a, b in zip(anchors, anchors[1:])]
            outer = np.vstack([curve[:-1] for curve in pieces]+[pieces[-1][-1:]])
            wall = np.asarray(self.domain.polygon, dtype=float)
            root = Polygon(outer, [wall])
            _remember(self._roots, key, (root, outer, wall), 128)
        else:
            self._roots.move_to_end(key)
        return self._roots[key]

    @staticmethod
    def _tag(a, b):
        if abs(a[1])+abs(b[1]) < 1e-12:
            low, high = sorted((a[0], b[0]))
            return 'wall' if low >= .25-1e-12 and high <= .75+1e-12 else 'wake'
        if (abs(a[1]-1)+abs(b[1]-1) < 1e-12
                or abs(a[0])+abs(b[0]) < 1e-12
                or abs(a[0]-1)+abs(b[0]-1) < 1e-12):
            return 'farfield'
        return 'internal'

    def _evaluation_polygon(self, original, *, keep_wall=False):
        """Remove only roundoff-scale collinearity from GEOS evaluation rings.

        Export and arc-length samples stay untouched. The displacement limit is
        eight machine epsilons at the coordinate scale, never curve_tolerance.
        A conservative swept-area bound triggers exact-ring reevaluation near
        an area acceptance threshold. Invalid original rings are never repaired.
        """
        key = (id(original), keep_wall)
        cached = self._evaluation_polygons.get(key)
        if cached is not None and cached[0] is original:
            self._evaluation_polygons.move_to_end(key)
            return cached[1:]
        if not original.is_valid or original.is_empty:
            return original, 0.
        scale = max(1., *map(abs, original.bounds))
        tolerance = 8*np.finfo(float).eps*scale
        if keep_wall:
            outer = Polygon(original.exterior).simplify(tolerance, preserve_topology=True)
            reduced = Polygon(outer.exterior, [ring.coords for ring in original.interiors])
            perimeter = original.exterior.length
        else:
            reduced = original.simplify(tolerance, preserve_topology=True)
            perimeter = original.length
        bound = float(2*tolerance*perimeter + np.pi*tolerance*tolerance)
        # Keep this optimization several orders below the physical area budget.
        if (not reduced.is_valid or reduced.geom_type != 'Polygon'
                or bound > self.config.area_tolerance/128):
            reduced, bound = original, 0.
        _remember(self._evaluation_polygons, key, (original, reduced, bound), 1024)
        return reduced, bound

    def _validate(self, graph, refine=False, *, exact=False):
        quality = {'q': 0., 'block_count': len(graph.faces),
            'outer_blocks': sum(not f.protected for f in graph.faces.values()),
            'coverage_error_area': 0., 'overlap_area': 0.,
            'angle_quality': 0., 'opposite_length_quality': 0.}
        def invalid(reason):
            return {'valid': False, 'reason': reason, 'quality': quality}
        failure = validate_graph(graph, self._protected, self.config.max_blocks, self.config.max_half_edges)
        if failure:
            return invalid('uv_'+failure.reason)
        raw_root, _, _ = self._reference(graph, refine)
        if not raw_root.is_valid or not self._wall.is_valid or not math.isfinite(raw_root.area) or raw_root.area <= 0:
            return invalid('invalid_reference_domain')
        root, root_bound = (raw_root, 0.) if exact else self._evaluation_polygon(raw_root, keep_wall=True)
        polygons, angles, opposites, face_bounds = [], [], [], []
        for face in graph.faces.values():
            raw_polygon, boundary, lengths, corner_angles = self._face(graph, face, refine)
            if (not raw_polygon.is_valid or not raw_polygon.exterior.is_simple or not math.isfinite(raw_polygon.area)
                    or raw_polygon.area <= self.config.area_tolerance):
                return invalid('invalid_physical_face')
            if not raw_polygon.exterior.is_ccw:
                return invalid('inverted_physical_face')
            if not all(math.isfinite(length) for length in lengths) or min(lengths) <= 1e-12:
                return invalid('degenerate_physical_side')
            if len(face.vertices) == 4 and min(corner_angles) <= 1e-8:
                return invalid('degenerate_physical_corner')
            polygon, bound = (raw_polygon, 0.) if exact else self._evaluation_polygon(raw_polygon)
            wall_overlap = polygon.intersection(self._wall).area
            if not exact and abs(wall_overlap-self.config.area_tolerance) <= bound:
                return self._validate(graph, refine, exact=True)
            if wall_overlap > self.config.area_tolerance:
                return invalid('block_crosses_airfoil_wall')
            polygons.append(polygon)
            face_bounds.append(bound)
            if not face.protected and len(face.vertices) == 4:
                angles.append(min(corner_angles))
                opposites.append(math.sqrt(min(lengths[0], lengths[2])/max(lengths[0], lengths[2])
                    * min(lengths[1], lengths[3])/max(lengths[1], lengths[3])))
        merged = unary_union(polygons)
        quality['coverage_error_area'] = float(root.symmetric_difference(merged).area)
        quality['overlap_area'] = max(0., float(sum(p.area for p in polygons)-merged.area))
        if not exact:
            coverage_bound = root_bound+sum(face_bounds)
            overlap_bound = 2*sum(face_bounds)
            if (abs(quality['coverage_error_area']-self.config.area_tolerance) <= coverage_bound
                    or abs(quality['overlap_area']-self.config.area_tolerance) <= overlap_bound):
                return self._validate(graph, refine, exact=True)
        if quality['coverage_error_area'] > self.config.area_tolerance:
            return invalid('physical_coverage_error')
        if quality['overlap_area'] > self.config.area_tolerance:
            return invalid('physical_block_overlap')
        # The cut opens a topological seam, not a self-touching polygon. Its two
        # boundary curves must describe the same physical line, in reverse order.
        incidence = edge_incidence(graph)
        wakes = []
        for edge, owners in incidence.items():
            a, b = (graph.vertices[v] for v in edge)
            if len(owners) == 1 and self._tag(a, b) == 'wake':
                wakes.append((edge, a, b))
        paired = set()
        for edge, a, b in wakes:
            if edge in paired:
                continue
            mirror = sorted(((1-a[0], 0.), (1-b[0], 0.)))
            partners = [other for other, c, d in wakes if other != edge and other not in paired
                        and np.allclose(sorted((c, d)), mirror, atol=1e-12, rtol=0)]
            if len(partners) != 1:
                return invalid('unpaired_wake')
            other = partners[0]
            lhs = LineString(self.curve(a, b, refine))
            rhs = LineString(self.curve(*(graph.vertices[v] for v in other), refined=refine))
            if lhs.hausdorff_distance(rhs) > self.config.area_tolerance:
                return invalid('wake_geometry_mismatch')
            paired.update((edge, other))
        if angles:
            quality['angle_quality'] = float(np.mean(angles))
            quality['opposite_length_quality'] = float(np.mean(opposites))
            quality['q'] = float(np.clip(.7*quality['angle_quality']+.3*quality['opposite_length_quality'], 0, 1))
        return {'valid': True, 'reason': 'valid', 'quality': quality}

    def validate(self, graph, refine=False):
        """Return serializable invalidity for ordinary geometry failures."""
        try:
            return self._validate(graph, refine)
        except (ValueError, KeyError, IndexError, ZeroDivisionError, OverflowError, GEOSException) as exc:
            return {'valid': False, 'reason': 'geometry_error', 'detail': str(exc),
                'quality': {'q': 0., 'block_count': len(graph.faces),
                    'outer_blocks': sum(not f.protected for f in graph.faces.values()),
                    'coverage_error_area': None, 'overlap_area': None,
                    'angle_quality': 0., 'opposite_length_quality': 0.}}

    def partition(self, graph):
        report = self.validate(graph)
        if not report['valid']:
            return {'schema': PARTITION_VERSION, 'blocks': [], 'curves': [], 'interfaces': [],
                    'wake_pairs': [], 'root': None, **report}
        _, outer, wall = self._reference(graph, False)
        incidence = edge_incidence(graph)
        curves, interfaces, wakes = [], [], []
        ids = {edge: f'curve_{edge[0]}_{edge[1]}' for edge in sorted(incidence)}
        for edge in sorted(incidence):
            a, b = (graph.vertices[v] for v in edge)
            owners = incidence[edge]
            tag = self._tag(a, b) if len(owners) == 1 else 'internal'
            rows = [{'block_id': fid, 'side': side,
                     'forward': graph.faces[fid].vertices[side] == edge[0]} for fid, side in owners]
            curves.append({'id': ids[edge], 'vertex_ids': list(edge), 'uv_endpoints': [list(a), list(b)],
                'points': self.curve(a, b).tolist(), 'tag': tag, 'owners': rows})
            if len(owners) == 2:
                interfaces.append({'curve_id': ids[edge], 'block_a': owners[0][0], 'side_a': owners[0][1],
                    'block_b': owners[1][0], 'side_b': owners[1][1], 'reverse': True})
            if tag == 'wake':
                wakes.append(edge)
        pairs, used = [], set()
        for edge in wakes:
            if edge in used:
                continue
            a, b = (graph.vertices[v] for v in edge)
            mirror = sorted(((1-a[0], 0.), (1-b[0], 0.)))
            other = next(e for e in wakes if e != edge and e not in used
                and np.allclose(sorted(graph.vertices[v] for v in e), mirror, atol=1e-12, rtol=0))
            pairs.append({'curve_a': ids[edge], 'curve_b': ids[other], 'reverse': True})
            used.update((edge, other))
        blocks = []
        for face in sorted(graph.faces.values(), key=lambda f: f.id):
            references = [{'curve_id': ids[tuple(sorted((a, b)))], 'forward': a < b}
                for a, b in zip(face.vertices, face.vertices[1:]+face.vertices[:1])]
            boundary = self._face(graph, face, False)[1]
            blocks.append({'id': face.id, 'protected': face.protected,
                'vertex_ids': list(face.vertices), 'uv_corners': [list(graph.vertices[v]) for v in face.vertices],
                'curve_refs': references, 'boundary': np.vstack((boundary, boundary[:1])).tolist()})
        return {'schema': PARTITION_VERSION, 'sample_id': self.domain.sample_id, 'blocks': blocks,
            'curves': curves, 'interfaces': interfaces, 'wake_pairs': pairs,
            'root': {'outer': outer.tolist(), 'wall': wall.tolist()}, **report,
            'scope': 'Restricted CMapping curves; geometric partition, not a generated CFD grid'}
