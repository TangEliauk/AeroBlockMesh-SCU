"""A geometric validity screen, not a CFD accuracy certificate."""
import numpy as np
from .jacobian import corner_scaled_jacobians


def evaluate_mesh(blocks):
    samples, cells, invalid, nonfinite_nodes, nonfinite_jacobians = [], 0, 0, 0, 0
    for nodes in blocks.values():
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
            jacobians = corner_scaled_jacobians(nodes)
        minima = jacobians.min(axis=-1)
        samples.append(minima.ravel())
        cells += minima.size
        invalid += int(np.count_nonzero(~np.isfinite(minima) | (minima <= 1e-10)))
        nonfinite_nodes += int(np.count_nonzero(~np.isfinite(nodes)))
        nonfinite_jacobians += int(np.count_nonzero(~np.isfinite(jacobians)))
    values = np.concatenate(samples) if samples else np.array([])
    finite = bool(values.size) and bool(np.isfinite(values).all())
    return {"valid": bool(blocks) and invalid == 0, "cells": int(cells),
            "min_scaled_jacobian": float(values.min()) if finite else None,
            "q10_scaled_jacobian": float(np.quantile(values,.1)) if finite else None,
            "invalid_cells": invalid, "has_nonfinite_coordinates": bool(nonfinite_nodes),
            "nonfinite_coordinate_count": nonfinite_nodes,
            "has_nonfinite_jacobians": bool(nonfinite_jacobians),
            "nonfinite_jacobian_count": nonfinite_jacobians,
            "check_scope": "corner Jacobians and finite coordinates; no CFD or global overlap proof"}


def json_safe(value):
    """Replace nonfinite numeric diagnostic values by null, never JSON NaN."""
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key,item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def invalid_cell_diagnostics(blocks, graph, *, limit=None):
    """Locate worst invalid cells; logical corners come from the coarse UV quad.

    Only called for failed meshes. A bounded preview avoids retaining every
    failed cell when no on-disk diagnostic was requested.
    """
    records = []
    for fid,nodes in sorted(blocks.items()):
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
            jacobians = corner_scaled_jacobians(nodes)
        minima = jacobians.min(axis=-1)
        locations = np.argwhere(~np.isfinite(minima) | (minima <= 1e-10))
        if not len(locations):
            continue
        scores = np.where(np.isfinite(minima), minima, -np.inf)
        if limit is not None and len(locations) > limit:
            order = np.argsort(scores[locations[:,0], locations[:,1]], kind='stable')[:limit]
            locations = locations[order]
        face = graph.faces[fid]
        uv = np.asarray([graph.vertices[v] for v in face.vertices], dtype=float)
        ni,nj = nodes.shape[:2]
        def logical(i,j):
            s,t = i/(ni-1),j/(nj-1)
            return (1-s)*(1-t)*uv[0]+s*(1-t)*uv[1]+s*t*uv[2]+(1-s)*t*uv[3]
        for i,j in locations:
            corner_values = jacobians[i,j]
            corner = int(np.argmin(np.where(np.isfinite(corner_values),corner_values,-np.inf)))
            xy = np.asarray([nodes[i,j],nodes[i+1,j],nodes[i+1,j+1],nodes[i,j+1]])
            record = {"block_id": int(fid), "i": int(i), "j": int(j), "corner": corner,
                "corner_jacobian": float(corner_values[corner]),
                "corner_scaled_jacobians": corner_values,
                "uv_corners": np.asarray([logical(i,j),logical(i+1,j),logical(i+1,j+1),logical(i,j+1)]),
                "xy_corners": xy, "protected": bool(face.protected),
                "region": "protected_strip" if face.protected else "outer",
                "has_nonfinite_coordinates": bool(not np.isfinite(xy).all()),
                "has_nonfinite_jacobians": bool(not np.isfinite(corner_values).all()),
                "_score": float(scores[i,j])}
            records.append(record)
    records.sort(key=lambda item:(item['_score'],item['block_id'],item['i'],item['j']))
    if limit is not None:
        records = records[:limit]
    for item in records:
        item.pop('_score')
    return json_safe(records)
