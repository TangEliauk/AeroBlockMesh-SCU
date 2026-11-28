"""Memoize identical physical validation requests within one wing geometry."""

from collections import OrderedDict
from copy import deepcopy
from functools import wraps

from smartcfd.airfoil_decomposition.geometry import CurveGeometry


def install_geometry_cache(capacity=1024):
    validate = CurveGeometry.validate

    @wraps(validate)
    def validate_cached(geometry, graph, refine=False):
        cache = getattr(geometry, "_exact_validation_cache", None)
        if cache is None:
            geometry._exact_validation_cache = cache = OrderedDict()
            geometry._exact_validation_work = {"hits": 0, "misses": 0}
        # Preserve insertion order, float bits and face IDs. GEOS sees faces
        # in this order, and the returned reason can depend on that order.
        vertices = tuple((key, tuple(float(coordinate).hex() for coordinate in point))
                         for key, point in graph.vertices.items())
        faces = tuple((key, face.id, face.vertices, face.protected) for key, face in graph.faces.items())
        identity = (refine, vertices, faces)
        if identity in cache:
            cache.move_to_end(identity)
            geometry._exact_validation_work["hits"] += 1
            return deepcopy(cache[identity])
        result = validate(geometry, graph, refine=refine)
        geometry._exact_validation_work["misses"] += 1
        cache[identity] = deepcopy(result)
        if len(cache) > capacity:
            cache.popitem(last=False)
        return result

    CurveGeometry.validate = validate_cached
    return validate
