"""Restricted C-coordinate domain for explicitly closed, sharp single airfoils."""
from __future__ import annotations
import numpy as np
from smartcfd.core.types import Domain


def sharp_eligibility(sample: dict) -> tuple[bool, str]:
    if not isinstance(sample, dict):
        return False, "invalid_sample"
    try:
        points = np.asarray(sample.get("polygon", []), dtype=float)
    except (TypeError, ValueError, OverflowError):
        return False, "invalid_polygon"
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 8 or not np.isfinite(points).all():
        return False, "invalid_polygon"
    if np.linalg.norm(points[0] - points[-1]) < 1e-14:
        points = points[:-1]
    if len(points) < 8 or np.any(np.linalg.norm(np.roll(points,-1,axis=0)-points,axis=1) <= 1e-14):
        return False, "degenerate_polygon"
    # The frozen public representation is normalized by chord. Do not silently
    # classify unnormalized external samples under a different xmax rule.
    if not (abs(float(points[:,0].min())) <= 1e-8 and abs(float(points[:,0].max())-1.) <= 1e-8):
        return False, "requires_normalized_chord"
    sources = sample.get("source_files", [])
    if not isinstance(sources, (list, tuple)):
        return False, "invalid_source_files"
    closed = sample.get("explicitly_closed") is True
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("quality_flags", {}), dict):
            return False, "invalid_source_files"
        closed = closed or source.get("quality_flags", {}).get("raw_representation") == "explicitly_closed"
    if not closed:
        return False, "requires_explicitly_closed_source"
    indices = np.flatnonzero(np.abs(points[:, 0] - points[:, 0].max()) <= 1e-10)
    if len(indices) != 1:
        return False, "requires_unique_trailing_edge_xmax"
    i = int(indices[0])
    a, b = points[i - 1] - points[i], points[(i + 1) % len(points)] - points[i]
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator <= 0:
        return False, "degenerate_trailing_edge"
    angle = np.degrees(np.arccos(np.clip(np.dot(a, b) / denominator, -1, 1)))
    return (True, "eligible") if angle <= 60 else (False, "trailing_edge_angle_exceeds_60_degrees")


def make_domain(sample: dict, *, farfield=500.0, stretch=18.0) -> Domain:
    eligible, reason = sharp_eligibility(sample)
    if not eligible:
        raise ValueError(reason)
    p = np.asarray(sample["polygon"], dtype=float).copy()
    if np.linalg.norm(p[0] - p[-1]) < 1e-14:
        p = p[:-1]
    if farfield <= 2 or not 0 < stretch <= 40:
        raise ValueError("invalid farfield or exponential stretch")
    return Domain(str(sample["sample_id"]), p, farfield=farfield, stretch=stretch,
                  metadata={"scope": "fixed C parameter domain; sharp single-element airfoil",
                            "source_aliases": list(sample.get("aliases", [])), "sharp_eligibility": reason,
                            "mapping_version":"c_normal_envelope_wakegraded_v2","wake_stretch":10.0,
                            "protected_thickness":.02})


class CMapping:
    """u: lower wake, lower TE->LE, upper LE->TE, upper wake; v points out.

    Protected coordinates use outward normals and exponential spacing. Dense
    physical samples define curves, not coarse topological vertices.
    """
    def __init__(self, domain):
        self.domain = domain
        domain.metadata.setdefault("mapping_version","c_normal_envelope_wakegraded_v2")
        domain.metadata.setdefault("wake_stretch",10.0)
        if not 0<domain.protected_v<1 or not np.isfinite(domain.farfield) or domain.farfield<=2:
            raise ValueError("invalid C-domain extent or protection boundary")
        if not 0<float(domain.metadata["wake_stretch"])<=20:
            raise ValueError("wake_stretch must be in (0,20]")
        p = np.asarray(domain.polygon, dtype=float)
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < 4 or not np.isfinite(p).all():
            raise ValueError("invalid domain polygon")
        le, te = int(np.argmin(p[:, 0])), int(np.argmax(p[:, 0]))
        self.te = p[te].copy()
        def walk(step):
            ids = [te]
            while ids[-1] != le:
                ids.append((ids[-1] + step) % len(p))
                if len(ids) > len(p):
                    raise ValueError("cannot extract airfoil branches")
            return p[ids]
        a, b = walk(1), walk(-1)
        def middle_y(branch):
            order = np.argsort(branch[:, 0])
            return np.interp((p[te, 0] + p[le, 0]) / 2, branch[order, 0], branch[order, 1])
        lower, upper = (a, b) if middle_y(a) < middle_y(b) else (b, a)
        self.lower = self._curve(lower, True)
        self.upper = self._curve(upper[::-1], False)

    @staticmethod
    def _curve(p, te_at_start):
        lengths = np.linalg.norm(np.diff(p, axis=0), axis=1)
        if np.any(lengths <= 0):
            raise ValueError("duplicate branch points")
        arc = np.r_[0., np.cumsum(lengths)]
        tangent = np.gradient(p, arc, axis=0)
        n = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        n /= np.linalg.norm(n, axis=1)[:, None]
        # Keep the actual TE wall normal and continue it onto the wake-side
        # interface. Forcing a vertical TE cut would cross the outward offset.
        if te_at_start: n[-1]=(-1,0)
        else: n[0]=(-1,0)
        return arc / arc[-1], p, n

    def inner_and_normal(self, u):
        u = np.asarray(u, dtype=float)
        flat = u.ravel()
        p, n = np.empty((len(flat), 2)), np.empty((len(flat), 2))
        for mask, local, sign in [(flat < .25, 1 - 4 * flat, -1), (flat > .75, 4 * (flat - .75), 1)]:
            wake_stretch=float(self.domain.metadata["wake_stretch"])
            distance=self.domain.farfield*np.expm1(wake_stretch*local[mask])/np.expm1(wake_stretch)
            p[mask, 0] = self.te[0] + distance
            p[mask, 1] = self.te[1]
            te_normal=self.lower[2][0] if sign<0 else self.upper[2][-1]
            nx=te_normal[0]*np.exp(-(p[mask,0]-self.te[0]))
            n[mask,0]=nx
            n[mask,1]=sign*np.sqrt(np.maximum(0,1-nx*nx))
        for mask, local, curve in [((flat >= .25) & (flat <= .5), 4 * (flat - .25), self.lower),
                                   ((flat > .5) & (flat <= .75), 4 * (flat - .5), self.upper)]:
            arc, points, normals = curve
            for axis in range(2):
                p[mask, axis] = np.interp(local[mask], arc, points[:, axis])
                n[mask, axis] = np.interp(local[mask], arc, normals[:, axis])
        n /= np.linalg.norm(n, axis=1)[:, None]
        return p.reshape(u.shape + (2,)), n.reshape(u.shape + (2,))

    def evaluate(self, uv_array):
        uv = np.asarray(uv_array, dtype=float)
        if uv.shape[-1:] != (2,) or not np.isfinite(uv).all() or np.any(uv < -1e-12) or np.any(uv > 1 + 1e-12):
            raise ValueError("finite uv coordinates in [0,1]^2 are required")
        u, v = np.clip(uv[..., 0], 0, 1), np.clip(uv[..., 1], 0, 1)
        p, n = self.inner_and_normal(u)
        radius = self.domain.farfield
        wake = (u < .25) | (u > .75)
        theta = -np.pi / 2 - 2 * np.pi * (u - .25)
        q = np.empty_like(p)
        q[..., 0] = np.where(wake, p[..., 0], self.te[0] + radius * np.cos(theta))
        q[..., 1] = self.te[1] + np.where(wake, np.where(u <= .5, -radius, radius), radius * np.sin(theta))
        # Stop normal extrusion at a finite near-wall envelope. Continuing an
        # airfoil normal for hundreds of chords can fold at a concave segment.
        pv=self.domain.protected_v
        thickness=float(self.domain.metadata.get("protected_thickness",.02))
        if not 0<thickness<radius: raise ValueError("invalid protected strip thickness")
        inner_weight=np.expm1(self.domain.stretch*np.minimum(v,pv))/np.expm1(self.domain.stretch*pv)
        inner=p+thickness*inner_weight[...,None]*n
        outer_weight=np.expm1(self.domain.stretch*np.maximum(v-pv,0))/np.expm1(self.domain.stretch*(1-pv))
        envelope=p+thickness*n
        outer=(1-outer_weight[...,None])*envelope+outer_weight[...,None]*q
        return np.where((v<=pv)[...,None],inner,outer)


def map_uv(domain, uv_array):
    return CMapping(domain).evaluate(uv_array)
