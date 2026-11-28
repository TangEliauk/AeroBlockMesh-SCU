"""Versioned discrete actions: a half-edge slot and one of four local edits."""
from numbers import Integral
from smartcfd.partition.topology import half_edges,protected_edges
ACTION_SCHEMA="halfedge_actions_v1"
LOCAL_ACTIONS=4
MAX_HALF_EDGES=256
ACTION_COUNT=1024
def describe_action(graph,action):
    if not isinstance(action,Integral) or not 0<=action<ACTION_COUNT:
        raise ValueError("Action must be an integer in [0,1024)")
    action=int(action);registry=half_edges(graph);slot,local=divmod(action,LOCAL_ACTIONS)
    if slot>=len(registry):
        return {"index":action,"slot":slot,"local":local,"kind":"padding"}
    edge=registry[slot];face=graph.faces[edge.face_id];vertices=face.vertices
    result={"index":action,"slot":slot,"local":local,"face_id":face.id,"edge_index":edge.edge_index,"source":edge.source,"edge_target":edge.target,"protected_face":face.protected,"protected_edge":tuple(sorted((edge.source,edge.target))) in protected_edges(graph)}
    if local==3:
        result.update(kind="insert_edge_midpoint",target=edge.target)
    else:
        result.update(kind="insert_chord",target=vertices[(edge.edge_index+local+2)%len(vertices)],forward_steps=local+2)
    result.update(source_uv=list(graph.vertices[result["source"]]),target_uv=list(graph.vertices[result["target"]]))
    return result
