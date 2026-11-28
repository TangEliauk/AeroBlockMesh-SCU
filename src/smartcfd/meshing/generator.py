"""Conformal quadrilateral blocks in the fixed C parameter domain."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from smartcfd.core.types import Failure, MeshSpec, MultiBlockMesh
from smartcfd.geometry.domain import CMapping
from smartcfd.evaluation.mesh_quality import evaluate_mesh, invalid_cell_diagnostics, json_safe
from smartcfd.evaluation.jacobian import corner_scaled_jacobians
from .tfi import coons_patch, smooth_elliptic


def side_nodes(nodes, side):
    """Return points in the block's CCW face-edge orientation."""
    return (nodes[:,0],nodes[-1,:],nodes[::-1,-1],nodes[0,::-1])[side]


class _Equivalence:
    def __init__(self,keys): self.parent={k:k for k in keys}
    def find(self,key):
        while self.parent[key]!=key:
            self.parent[key]=self.parent[self.parent[key]]
            key=self.parent[key]
        return key
    def union(self,a,b): self.parent[self.find(a)]=self.find(b)


def generate_mesh(domain, graph, spec=None, *, diagnostics_dir=None, target_cells=None):
    """Return a valid candidate or Failure; invalid nodes are diagnostic only.

    ``diagnostics_dir`` opts into failed-node storage and a local worst-cell
    drawing. Successful mesh generation performs no diagnostic file writes.
    A cell target selects shared integer counts before generating any nodes.
    """
    spec = spec or MeshSpec()
    try:
        if target_cells is None:
            mesh = _generate(domain, graph, spec)
        else:
            mesh = _generate(domain, graph, spec, target_cells=target_cells)
    except (ValueError,IndexError,KeyError,FloatingPointError) as exc:
        failure = Failure("mesh_generation_failed",str(exc),{"mapping": _mapping_info(domain)})
        if diagnostics_dir is not None:
            _write_failure_json(diagnostics_dir, failure)
        return failure
    if mesh.quality.get("valid", False):
        return mesh
    # A coarse graph can be valid while its mapped physical cells fold. Never
    # let this invalid candidate enter a caller's success or export path.
    cells = invalid_cell_diagnostics(mesh.blocks, graph, limit=None if diagnostics_dir is not None else 20)
    count = int(mesh.quality.get('invalid_cells', len(cells)))
    summary = json_safe({"quality": mesh.quality, "invalid_cell_count": count,
        "invalid_cells_preview": cells[:20], "preview_truncated": count>20,
        "mapping": _mapping_info(domain),
        "mapped_tfi_fallback_blocks": mesh.metadata.get('mapped_tfi_fallback_blocks', [])})
    failure = Failure('invalid_generated_mesh', 'Mapped mesh failed the signed corner-Jacobian screen', summary)
    if diagnostics_dir is not None:
        _write_invalid_diagnostics(diagnostics_dir, mesh, graph, failure, cells)
    return failure


def _mapping_info(domain):
    return json_safe({"mapping_version": domain.metadata.get('mapping_version'),
        "farfield": domain.farfield, "stretch": domain.stretch, "protected_v": domain.protected_v,
        "protected_thickness": domain.metadata.get('protected_thickness', .02),
        "wake_stretch": domain.metadata.get('wake_stretch', 10.)})


def _write_failure_json(directory, failure):
    output = Path(directory)
    try:
        output.mkdir(parents=True, exist_ok=True)
        value = {"schema_version": 1, "valid": False, "diagnostic_only": True,
                 "reason": failure.reason, "detail": failure.detail, "diagnostics": failure.diagnostics}
        (output/'diagnostics.json').write_text(json.dumps(json_safe(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    except OSError as error:
        failure.diagnostics['diagnostics_write_error'] = str(error)


def _write_invalid_diagnostics(directory, mesh, graph, failure, cells):
    output = Path(directory)
    try:
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output/'invalid_mesh.npz', **{f'block_{fid}':nodes for fid,nodes in mesh.blocks.items()})
        (output/'invalid_cells.json').write_text(json.dumps(cells,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        (output/'failed_graph.json').write_text(json.dumps(json_safe(graph.to_dict()),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        failure.diagnostics['artifacts'] = {"directory": str(output.resolve()),
            "invalid_nodes": 'invalid_mesh.npz', "invalid_cells": 'invalid_cells.json', "graph": 'failed_graph.json'}
        if cells:
            try:
                _draw_worst_cell(mesh, cells[0], output/'worst_cell.png')
                failure.diagnostics['artifacts']['worst_cell'] = 'worst_cell.png'
            except (ImportError, ValueError, RuntimeError, OSError) as error:
                failure.diagnostics['diagnostic_plot_error'] = str(error)
        _write_failure_json(output, failure)
    except OSError as error:
        failure.diagnostics['diagnostics_write_error'] = str(error)


def _draw_worst_cell(mesh, record, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    nodes = mesh.blocks[record['block_id']]
    i,j = record['i'],record['j']
    local = nodes[max(i-2,0):i+4,max(j-2,0):j+4]
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    try:
        for line in (*local, *local.transpose(1,0,2)):
            axes[0].plot(line[:,0],line[:,1],color='steelblue',linewidth=.7)
        for ax,key in zip(axes,('xy_corners','uv_corners')):
            points = np.asarray(record[key], dtype=float)
            closed = np.vstack([points,points[0]])
            ax.plot(closed[:,0],closed[:,1],'r-o',markersize=3)
            for index,point in enumerate(points):
                if np.isfinite(point).all():
                    ax.annotate(str(index),point)
            if not np.isfinite(points).all():
                ax.text(.02,.95,'Nonfinite values: see JSON',transform=ax.transAxes,va='top')
            ax.set_aspect('equal',adjustable='datalim')
        axes[0].set(title='Physical cell and neighboring nodes',xlabel='x/c',ylabel='y/c')
        axes[1].set(title='Logical cell corners',xlabel='u',ylabel='v')
        fig.suptitle(f"Diagnostic only: block {record['block_id']}, cell ({i},{j}), corner {record['corner']}; protected={record['protected']}")
        fig.tight_layout()
        fig.savefig(path,dpi=150)
    finally:
        plt.close(fig)


def _budget_intervals(target, maximum, edges, equivalence, protected):
    """Nearest achievable count under a common two-class spacing rule."""
    if type(target) is not int or target < 1:
        raise ValueError("target_cells must be a positive integer")
    outer, near = np.triu_indices(min(maximum, 64)-1)
    outer, near = outer+2, near+2
    counts = np.zeros_like(outer)
    for sides in edges.values():
        a, b = (near if equivalence.find(side) in protected else outer for side in sides[:2])
        counts += a*b
    rank = np.lexsort((near, outer, np.abs(near-2*outer), np.abs(counts-target)))
    best = rank[0]
    return int(outer[best]), int(near[best])


def _generate(domain,graph,spec,*,target_cells=None):
    from smartcfd.partition.constraints import validate_graph
    graph_error=validate_graph(graph,max_blocks=max(1,len(graph.faces)),max_half_edges=max(4,4*len(graph.faces)))
    if graph_error is not None: raise ValueError("invalid coarse graph: "+graph_error.reason)
    if min(spec.intervals,spec.protected_intervals)<2 or spec.max_intervals<max(spec.intervals,spec.protected_intervals):
        raise ValueError("invalid edge interval budget")
    if not graph.faces:
        raise ValueError("empty block graph")
    incidence, face_edges = {}, {}
    for fid,face in sorted(graph.faces.items()):
        if len(face.vertices)!=4 or len(set(face.vertices))!=4:
            raise ValueError("all terminal blocks must have four distinct vertices")
        uv=np.asarray([graph.vertices[i] for i in face.vertices],dtype=float)
        if np.sum(uv[:,0]*np.roll(uv[:,1],-1)-uv[:,1]*np.roll(uv[:,0],-1))<=0:
            raise ValueError("block vertices must be counterclockwise in uv")
        edges=[]
        for side,(a,b) in enumerate(zip(face.vertices,face.vertices[1:]+face.vertices[:1])):
            key=tuple(sorted((a,b)))
            incidence.setdefault(key,[]).append((fid,side,a,b))
            edges.append(key)
        face_edges[fid]=edges
    if any(len(owners)>2 for owners in incidence.values()):
        raise ValueError("nonmanifold coarse edge")
    equivalence=_Equivalence(incidence)
    for edges in face_edges.values():
        equivalence.union(edges[0],edges[2]); equivalence.union(edges[1],edges[3])
    wake=[]
    boundaries=[]
    for key,owners in incidence.items():
        if len(owners)!=1: continue
        fid,side,a,b=owners[0]
        p,q=np.asarray(graph.vertices[a]),np.asarray(graph.vertices[b])
        if abs(p[1])<1e-12 and abs(q[1])<1e-12:
            if max(p[0],q[0])<=.25+1e-12 or min(p[0],q[0])>=.75-1e-12:
                wake.append((key,owners[0])); continue
            if min(p[0],q[0])>=.25-1e-12 and max(p[0],q[0])<=.75+1e-12:
                boundaries.append(dict(block=fid,side=side,tag="wall")); continue
        if (abs(p[1]-1)<1e-12 and abs(q[1]-1)<1e-12) or (abs(p[0]-q[0])<1e-12 and (abs(p[0])<1e-12 or abs(p[0]-1)<1e-12)):
            boundaries.append(dict(block=fid,side=side,tag="farfield")); continue
        raise ValueError("unpaired internal edge or unsupported partial interface")
    wake_pairs=[]
    used=set()
    for key,owner in wake:
        if key in used: continue
        a,b=key
        reflected=sorted([(1-graph.vertices[a][0],0.),(1-graph.vertices[b][0],0.)])
        matches=[(other,own) for other,own in wake if other!=key and other not in used and np.allclose(sorted([graph.vertices[v] for v in other]),reflected,atol=1e-12,rtol=0)]
        if len(matches)!=1: raise ValueError("wake requires an explicit reflected partner")
        other,other_owner=matches[0]
        equivalence.union(key,other)
        wake_pairs.append((owner,other_owner))
        used.update((key,other))
    protected_edges = set()
    for key in incidence:
        p,q=(np.asarray(graph.vertices[v]) for v in key)
        if abs(p[0]-q[0])<1e-12 and max(p[1],q[1])<=domain.protected_v+1e-12:
            protected_edges.add(key)
    outer, near = spec.intervals, spec.protected_intervals
    if target_cells is not None:
        protected_classes = {equivalence.find(key) for key in protected_edges}
        outer, near = _budget_intervals(target_cells, spec.max_intervals, face_edges, equivalence, protected_classes)
    counts={}
    for key in incidence:
        requested=near if key in protected_edges else outer
        cls=equivalence.find(key)
        counts[cls]=max(counts.get(cls,0),requested)
    cells=sum(counts[equivalence.find(edges[0])]*counts[equivalence.find(edges[1])] for edges in face_edges.values())
    if cells>spec.max_cells: raise ValueError("mesh cell budget exceeded")
    mapping=CMapping(domain)
    uv_edges,xy_edges={},{}
    for key in incidence:
        p,q=(np.asarray(graph.vertices[v],dtype=float) for v in key)
        t=np.linspace(0,1,counts[equivalence.find(key)]+1)[:,None]
        uv_edges[key]=(1-t)*p+t*q
        xy_edges[key]=mapping.evaluate(uv_edges[key])
    blocks,smoothed,fallbacks={},0,[]
    for fid,face in sorted(graph.faces.items()):
        xy,uv=[],[]
        for a,b in zip(face.vertices,face.vertices[1:]+face.vertices[:1]):
            key=tuple(sorted((a,b)))
            xy.append(xy_edges[key] if a==key[0] else xy_edges[key][::-1])
            uv.append(uv_edges[key] if a==key[0] else uv_edges[key][::-1])
        logical=coons_patch(uv[0],uv[1],uv[2][::-1],uv[3][::-1])
        if face.protected:
            if logical[...,1].max()>domain.protected_v+1e-12:
                raise ValueError("protected block escaped its protected strip")
            nodes=mapping.evaluate(logical)
        else:
            nodes=coons_patch(xy[0],xy[1],xy[2][::-1],xy[3][::-1])
            if np.any(corner_scaled_jacobians(nodes)<=1e-10):
                # Curved, strongly stretched boundaries may invalidate physical
                # Coons interpolation. Use the declared coordinate map of the
                # logical TFI patch and retain the same shared edge samples.
                nodes=mapping.evaluate(logical)
                fallbacks.append(fid)
            nodes,n=smooth_elliptic(nodes,spec.smoothing_steps)
            smoothed+=n
        nodes[:,0],nodes[-1,:],nodes[:,-1],nodes[0,:]=xy[0],xy[1],xy[2][::-1],xy[3][::-1]
        blocks[fid]=nodes
    interfaces=[]
    for owners in incidence.values():
        if len(owners)==2:
            a,b=owners
            if (a[2],a[3])==(b[2],b[3]): raise ValueError("same-direction adjacent faces")
            interfaces.append(dict(block_a=a[0],side_a=a[1],block_b=b[0],side_b=b[1],reverse=True,kind="block"))
    for a,b in wake_pairs:
        left,right=side_nodes(blocks[a[0]],a[1]),side_nodes(blocks[b[0]],b[1])
        reverse=np.linalg.norm(left[0]-right[-1])<np.linalg.norm(left[0]-right[0])
        if not np.allclose(left,right[::-1] if reverse else right,atol=1e-11,rtol=0): raise ValueError("wake coordinates differ")
        # Copy exact shared coordinates after matching topology-defined partners.
        right[:]=left[::-1] if reverse else left
        interfaces.append(dict(block_a=a[0],side_a=a[1],block_b=b[0],side_b=b[1],reverse=bool(reverse),kind="wake"))
    for interface in interfaces:
        a=side_nodes(blocks[interface["block_a"]],interface["side_a"])
        b=side_nodes(blocks[interface["block_b"]],interface["side_b"])
        if interface["reverse"]: b=b[::-1]
        if not np.array_equal(a,b): raise ValueError("interface arrays differ")
    quality=evaluate_mesh(blocks)
    quality["interface_max_abs_difference"]=0.0
    wall_angles,node_angles,first_heights,growth=[],[],[],[]
    for item in boundaries:
        if item["tag"]!="wall": continue
        nodes=blocks[item["block"]]
        side=item["side"]
        wall=side_nodes(nodes,side)
        near=(nodes[:,1],nodes[-2,:],nodes[::-1,-2],nodes[1,::-1])[side]
        tangent=np.gradient(wall,axis=0); normal_step=near-wall
        height=np.linalg.norm(normal_step,axis=1)
        denominator=np.linalg.norm(tangent,axis=1)*height
        cosine=np.divide(np.abs((tangent*normal_step).sum(-1)),denominator,out=np.ones_like(height),where=denominator>0)
        node_angles.extend(np.degrees(np.arcsin(np.clip(cosine,0,1))).tolist())
        # Orthogonality of the discrete wall FACES, evaluated at face centers.
        # Node tangents at LE/TE corners are recorded separately, not conflated.
        face_tangent=np.diff(wall,axis=0)
        face_step=(normal_step[:-1]+normal_step[1:])/2
        face_denominator=np.linalg.norm(face_tangent,axis=1)*np.linalg.norm(face_step,axis=1)
        face_cosine=np.divide(np.abs((face_tangent*face_step).sum(-1)),face_denominator,out=np.ones_like(face_denominator),where=face_denominator>0)
        wall_angles.extend(np.degrees(np.arcsin(np.clip(face_cosine,0,1))).tolist())
        first_heights.extend(height.tolist())
        if side==0 and nodes.shape[1]>2:
            spacing=np.linalg.norm(np.diff(nodes,axis=1),axis=-1)
            growth.extend((spacing[:,1:]/np.maximum(spacing[:,:-1],1e-300)).ravel().tolist())
    quality.update(wall_normal_deviation_deg_max=float(max(wall_angles,default=0)),
                   wall_node_normal_deviation_deg_max=float(max(node_angles,default=0)),
                   first_height_min=float(min(first_heights,default=0)),first_height_max=float(max(first_heights,default=0)),
                   protected_layer_growth_max=float(max(growth,default=0)),
                   wall_normal_metric="face tangent versus mean first-layer displacement at face center; nodal corner metric reported separately")
    metadata={"sample_id":domain.sample_id,"scope":"constrained coarse blocks in a fixed C parameter domain",
              "graph":graph.to_dict(),"smoothing_accepted_steps":smoothed,"protected_blocks":[fid for fid,f in graph.faces.items() if f.protected],
              "generator":"normal-coordinate protected strip; outer Coons TFI; fixed-edge elliptic smoothing",
              "farfield":domain.farfield,"stretch":domain.stretch,"protected_thickness":domain.metadata.get("protected_thickness",.02),"mapped_tfi_fallback_blocks":fallbacks,
              "mapping_version":domain.metadata["mapping_version"],"wake_stretch":domain.metadata["wake_stretch"]}
    if target_cells is not None:
        metadata.update(target_cells=target_cells, actual_cells=int(cells), outer_intervals=outer, protected_intervals=near)
    return MultiBlockMesh(blocks,interfaces,boundaries,quality,metadata)


def save_mesh(mesh,output_dir):
    if not mesh.quality.get('valid',False) or not evaluate_mesh(mesh.blocks)['valid']:
        raise ValueError('refusing to save an invalid mesh as a successful mesh asset')
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/"mesh.npz",**{f"block_{fid}":nodes for fid,nodes in mesh.blocks.items()})
    manifest={"schema_version":1,"blocks":{str(fid):list(nodes.shape) for fid,nodes in mesh.blocks.items()},
              "interfaces":mesh.interfaces,"boundaries":mesh.boundaries,"quality":mesh.quality,"metadata":mesh.metadata}
    (output/"mesh.json").write_text(json.dumps(json_safe(manifest),ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return output/"mesh.npz",output/"mesh.json"
