"""Deterministic half-edge views of the constrained coarse block graph."""
from __future__ import annotations
from dataclasses import dataclass
from smartcfd.core.types import BlockGraph, Domain, FaceBlock
@dataclass(frozen=True)
class HalfEdge:
    slot:int
    face_id:int
    edge_index:int
    source:int
    target:int
    next:int
    previous:int
    twin:int=-1
def initial_graph(domain:Domain|None=None)->BlockGraph:
    """Four protected blocks and one genuinely unresolved CCW ten-gon."""
    if domain is not None and abs(domain.protected_v-.5)>1e-12:
        raise ValueError("The protected C-domain interface is fixed at v=0.5")
    us=(0.,.25,.5,.75,1.)
    vertices={row*5+col:(u,v) for row,v in enumerate((0.,.5,1.)) for col,u in enumerate(us)}
    faces={i:FaceBlock(i,(i,i+1,6+i,5+i),True) for i in range(4)}
    faces[4]=FaceBlock(4,(5,6,7,8,9,14,13,12,11,10),False)
    return BlockGraph(vertices,faces,15,5,{"topology_scope":"fixed_C_parameter_domain_with_immutable_near_wall_band","protected_v":.5,"version":"coarse_blocks_v1"})
def half_edges(graph):
    """Ascending face ID and stored edge index; stable within one state."""
    records,directed=[],{}
    for face in sorted(graph.faces.values(),key=lambda f:f.id):
        base,n=len(records),len(face.vertices)
        for i,src in enumerate(face.vertices):
            dst=face.vertices[(i+1)%n];slot=base+i
            records.append(HalfEdge(slot,face.id,i,src,dst,base+(i+1)%n,base+(i-1)%n))
            directed[(src,dst)]=slot
    return [HalfEdge(r.slot,r.face_id,r.edge_index,r.source,r.target,r.next,r.previous,directed.get((r.target,r.source),-1)) for r in records]
def edge_incidence(graph):
    result={}
    for face in graph.faces.values():
        for i,a in enumerate(face.vertices):
            b=face.vertices[(i+1)%len(face.vertices)]
            result.setdefault(tuple(sorted((a,b))),[]).append((face.id,i))
    return result
def protected_edges(graph):
    return {tuple(sorted((a,face.vertices[(i+1)%len(face.vertices)]))) for face in graph.faces.values() if face.protected for i,a in enumerate(face.vertices)}
def all_quadrilateral(graph):
    return bool(graph.faces) and all(len(f.vertices)==4 for f in graph.faces.values())
def fingerprint(graph):
    def normalized(face):
        points=[graph.vertices[v] for v in face.vertices];start=min(range(len(points)),key=lambda i:points[i])
        return (face.protected,tuple(points[start:]+points[:start]))
    return tuple(sorted(normalized(f) for f in graph.faces.values()))
