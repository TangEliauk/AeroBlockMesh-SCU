"""Polygon legality and full-edge conforming-interface invariants."""
from __future__ import annotations
import math
from smartcfd.core.types import BlockGraph,Failure
from smartcfd.partition.topology import edge_incidence
EPS=1e-12
def cross(a,b,c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
def area(points):
    return .5*sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(points,points[1:]+points[:1]))
def on_segment(a,b,p):
    return abs(cross(a,b,p))<=EPS and min(a[0],b[0])-EPS<=p[0]<=max(a[0],b[0])+EPS and min(a[1],b[1])-EPS<=p[1]<=max(a[1],b[1])+EPS
def intersects(a,b,c,d):
    if max(a[0],b[0])+EPS<min(c[0],d[0]) or max(c[0],d[0])+EPS<min(a[0],b[0]) or max(a[1],b[1])+EPS<min(c[1],d[1]) or max(c[1],d[1])+EPS<min(a[1],b[1]):
        return False
    p,q,r,s=cross(a,b,c),cross(a,b,d),cross(c,d,a),cross(c,d,b)
    return (p*q<-EPS*EPS and r*s<-EPS*EPS) or on_segment(a,b,c) or on_segment(a,b,d) or on_segment(c,d,a) or on_segment(c,d,b)
def inside(point,polygon):
    hit=False;x,y=point
    for a,b in zip(polygon,polygon[1:]+polygon[:1]):
        if on_segment(a,b,point):
            return False
        if (a[1]>y)!=(b[1]>y):
            hit^=x<a[0]+(y-a[1])*(b[0]-a[0])/(b[1]-a[1])
    return hit
def valid_polygon(points):
    if len(points)<3 or len(set(points))!=len(points) or area(points)<=EPS:
        return False
    n=len(points)
    for i,a in enumerate(points):
        b=points[(i+1)%n]
        if math.dist(a,b)<=EPS:
            return False
        for j in range(i+2,n):
            if i==0 and j==n-1:
                continue
            if intersects(a,b,points[j],points[(j+1)%n]):
                return False
    if n==4 and any(cross(points[i-1],points[i],points[(i+1)%n])<=EPS for i in range(n)):
        return False
    return True
def valid_chord(points,i,j):
    n=len(points)
    if i==j or (i-j)%n in (1,n-1):
        return False
    a,b=points[i],points[j]
    if not inside(((a[0]+b[0])/2,(a[1]+b[1])/2),points):
        return False
    for k in range(n):
        q=(k+1)%n
        if k in (i,j) or q in (i,j):
            continue
        if intersects(a,b,points[k],points[q]):
            return False
    return True
def validate_graph(graph:BlockGraph,previous=None,max_blocks=32,max_half_edges=256):
    if not graph.faces or len(graph.faces)>max_blocks:
        return Failure("block_budget")
    if sum(len(f.vertices) for f in graph.faces.values())>max_half_edges:
        return Failure("half_edge_budget")
    for uv in graph.vertices.values():
        if len(uv)!=2 or not all(math.isfinite(x) and -EPS<=x<=1+EPS for x in uv):
            return Failure("vertex_outside_parameter_domain")
    total=0.
    for key,face in graph.faces.items():
        if face.id!=key or any(v not in graph.vertices for v in face.vertices):
            return Failure("missing_face_or_vertex")
        coords=[graph.vertices[v] for v in face.vertices]
        if not valid_polygon(coords):
            return Failure("invalid_face_polygon",diagnostics={"face_id":face.id})
        total+=area(coords)
    if abs(total-1)>1e-9:
        return Failure("parameter_domain_coverage",diagnostics={"area":total})
    for edge,owners in edge_incidence(graph).items():
        if len(owners)>2:
            return Failure("nonmanifold_edge")
        if len(owners)==2:
            f,i=owners[0];g,j=owners[1];av,bv=graph.faces[f].vertices,graph.faces[g].vertices
            if (av[i],av[(i+1)%len(av)])!=(bv[(j+1)%len(bv)],bv[j]):
                return Failure("inconsistent_interface_orientation")
        else:
            a,b=(graph.vertices[v] for v in edge)
            if not any(abs(a[axis]-wall)<EPS and abs(b[axis]-wall)<EPS for axis in (0,1) for wall in (0.,1.)):
                return Failure("unmatched_internal_edge","Hanging/partial interfaces prohibited")
    if previous is not None:
        for face in previous.faces.values():
            if face.protected:
                new=graph.faces.get(face.id)
                if new is None or not new.protected or new.vertices!=face.vertices:
                    return Failure("protected_band_modified")
                if any(graph.vertices[v]!=previous.vertices[v] for v in face.vertices):
                    return Failure("protected_band_vertex_moved")
    return None
