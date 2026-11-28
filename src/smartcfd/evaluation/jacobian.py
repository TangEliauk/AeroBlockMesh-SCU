"""Signed corner Jacobians of bilinear planar quadrilateral elements."""
import numpy as np


def corner_scaled_jacobians(nodes):
    p = np.asarray(nodes, dtype=float)
    if p.ndim != 3 or p.shape[-1] != 2 or min(p.shape[:2]) < 2:
        raise ValueError("expected [ni,nj,2] structured nodes")
    a, b, c, d = p[:-1, :-1], p[1:, :-1], p[1:, 1:], p[:-1, 1:]
    values = []
    for ti, tj in [(b-a,d-a),(b-a,c-b),(c-d,c-b),(c-d,d-a)]:
        det = ti[..., 0]*tj[..., 1] - ti[..., 1]*tj[..., 0]
        denominator = np.linalg.norm(ti, axis=-1)*np.linalg.norm(tj, axis=-1)
        values.append(np.divide(det, denominator, out=np.full_like(det, -np.inf), where=denominator>0))
    return np.stack(values, axis=-1)
