"""Deterministic engineering witnesses for the fixed C-domain action system.

The refinement program below is a reachability proof, not expert data, a learned
policy, or a performance benchmark. It only chooses candidates already accepted
by the environment's physical legality checks; unavailable actions fail closed.
"""
from __future__ import annotations


def _close(a, b):
    return abs(float(a) - float(b)) <= 1e-10


def _vertical_cut(descriptors, u):
    for index, item in enumerate(descriptors):
        if item["kind"] != "insert_chord":
            continue
        a, b = item["source_uv"], item["target_uv"]
        if _close(a[0], u) and _close(b[0], u) and (
            (_close(a[1], .5) and _close(b[1], 1.)) or
            (_close(b[1], .5) and _close(a[1], 1.))):
            return index
    return None


def refined_c_action(env):
    """Choose one legal step in a two-outer-layer, twelve-block construction.

    Start with the u=.25 cut and insert its physical arc midpoint. FIFO then
    carries the shared midpoint constraint from strip to strip: insert a midpoint
    on the opposite full radial side, and connect the two interior radial points.
    Existing u=.5/.75 cuts take precedence while their parent face is active.
    Shared-edge insertion reopens the neighboring completed quadrilateral, so
    the program ends only after every outer strip has two quadrilateral blocks.

    Call after a fresh standard reset and after each selected step. Returning
    None means the witness is unavailable under the geometry/budget; callers
    must report failure rather than substituting random or unvalidated edits.
    """
    if env.graph is None or env.terminated or not env.queue:
        return None
    descriptors = env.candidates
    if env.steps == 0:
        return _vertical_cut(descriptors, .25)
    if env.steps == 1:
        return next((index for index, item in enumerate(descriptors)
            if item["kind"] == "insert_edge_point" and len(item["owners"]) == 2
            and _close(item["source_uv"][0], .25) and _close(item["target_uv"][0], .25)
            and _close(item["fraction"], .5)), None)
    for u in (.25, .5, .75):
        action = _vertical_cut(descriptors, u)
        if action is not None:
            return action
    face = env.graph.faces[env.queue[0]]
    interior = {vertex for vertex in face.vertices if .5 + 1e-9 < env.graph.vertices[vertex][1] < 1 - 1e-9}
    for index, item in enumerate(descriptors):
        if item["kind"] == "insert_chord" and item["source"] in interior and item["target"] in interior:
            return index
    for index, item in enumerate(descriptors):
        if item["kind"] != "insert_edge_point" or not _close(item["fraction"], .5):
            continue
        a, b = item["source_uv"], item["target_uv"]
        if _close(a[0], b[0]) and (
            (_close(a[1], .5) and _close(b[1], 1.)) or
            (_close(b[1], .5) and _close(a[1], 1.))):
            return index
    return None
