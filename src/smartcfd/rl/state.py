"""Fixed full-graph observation schema shared by BC and discrete SAC."""
from __future__ import annotations
import math
import numpy as np
from smartcfd.partition.actions import MAX_HALF_EDGES,ACTION_COUNT
from smartcfd.partition.constraints import area
from smartcfd.partition.topology import half_edges,protected_edges
from smartcfd.rl.reward import topology_defect
OBSERVATION_SCHEMA="halfedge_obs_v1"
FEATURE_NAMES=("source_u","source_v","target_u","target_v",
    "source_physical_x_log","source_physical_y_log","target_physical_x_log","target_physical_y_log",
    "source_valence_scaled","target_valence_scaled","face_degree_scaled","face_uv_area",
    "edge_uv_length","protected_face","protected_edge","shared_interface","wall","wake",
    "farfield","c_parameter_side","midpoint_wall_x_log","midpoint_wall_y_log",
    "protected_interface","real_halfedge")
def observation(domain,graph,mask,steps,config):
    from smartcfd.geometry.domain import map_uv
    registry=half_edges(graph)
    if len(registry)>MAX_HALF_EDGES:
        raise ValueError("Full graph exceeds fixed observation capacity")
    features=np.zeros((MAX_HALF_EDGES,24),dtype=np.float32)
    next_array=np.full(MAX_HALF_EDGES,-1,dtype=np.int64)
    previous=np.full(MAX_HALF_EDGES,-1,dtype=np.int64)
    twin=np.full(MAX_HALF_EDGES,-1,dtype=np.int64)
    protected=protected_edges(graph)
    neighbors={v:set() for v in graph.vertices}
    for edge in registry:
        neighbors[edge.source].add(edge.target)
        neighbors[edge.target].add(edge.source)
    uv=np.asarray([graph.vertices[e.source] for e in registry],dtype=float)
    xy=np.asarray(map_uv(domain,uv),dtype=float)
    wall_uv=np.asarray([((graph.vertices[e.source][0]+graph.vertices[e.target][0])/2,0.)
                        for e in registry],dtype=float)
    wall_xy=np.asarray(map_uv(domain,wall_uv),dtype=float)
    denom=math.log1p(max(float(domain.farfield),1.))
    physical=np.sign(xy)*np.log1p(np.abs(xy))/denom
    wall=np.sign(wall_xy)*np.log1p(np.abs(wall_xy))/denom
    for edge in registry:
        i=edge.slot;a=graph.vertices[edge.source];b=graph.vertices[edge.target]
        face=graph.faces[edge.face_id]
        bottom=abs(a[1])+abs(b[1])<1e-12
        umid=(a[0]+b[0])/2
        boundary_wall=bottom and .25<=umid<=.75
        boundary_wake=bottom and not boundary_wall
        side=-1. if abs(a[0])+abs(b[0])<1e-12 else (1. if abs(a[0]-1)+abs(b[0]-1)<1e-12 else 0.)
        features[i]=[*a,*b,*physical[i],*physical[edge.next],
            len(neighbors[edge.source])/8,len(neighbors[edge.target])/8,len(face.vertices)/16,
            area([graph.vertices[v] for v in face.vertices]),math.dist(a,b),
            float(face.protected),float(tuple(sorted((edge.source,edge.target))) in protected),
            float(edge.twin>=0),float(boundary_wall),float(boundary_wake),
            float(abs(a[1]-1)+abs(b[1]-1)<1e-12),side,*wall[i],
            float(abs(a[1]-.5)+abs(b[1]-.5)<1e-12),1.]
        next_array[i]=edge.next;previous[i]=edge.previous;twin[i]=edge.twin
    polygon=np.asarray(domain.polygon,dtype=float)
    defect=topology_defect(graph)
    global_features=np.array([steps/config.max_steps,len(graph.faces)/config.max_blocks,
        len(registry)/MAX_HALF_EDGES,defect/(defect+4.),
        np.ptp(polygon[:,1]),np.mean(polygon[:,1]),
        math.log1p(domain.farfield)/math.log(501.),domain.stretch/32.],dtype=np.float32)
    result={"features":features,"next":next_array,"previous":previous,"twin":twin,
            "mask":np.asarray(mask,dtype=np.bool_).copy(),"global":global_features}
    if result["mask"].shape!=(ACTION_COUNT,) or not np.isfinite(features).all() or not np.isfinite(global_features).all():
        raise ValueError("Invalid observation values")
    return result
