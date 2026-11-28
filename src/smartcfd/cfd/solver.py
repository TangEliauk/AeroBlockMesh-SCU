"""Identity-based conformal SU2 export and an explicit external-solver runner."""
from __future__ import annotations
from collections import Counter
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import numpy as np
from smartcfd.core.types import Failure


def _side(array,side):
    if side not in range(4):
        raise ValueError('block side must be an integer in [0,4)')
    return (array[:,0],array[-1,:],array[::-1,-1],array[0,::-1])[side]


def _assemble_su2(mesh):
    """Validate/assemble once in memory for both validation and file export."""
    from smartcfd.evaluation.mesh_quality import evaluate_mesh
    if not mesh.quality.get("valid",False):
        raise ValueError("refusing to export an invalid mesh")
    if not evaluate_mesh(mesh.blocks)['valid']:
        raise ValueError('refusing to export cells that fail the current geometric screen')
    ids, coordinates, offset={},[],0
    for fid,nodes in sorted(mesh.blocks.items()):
        count=nodes.shape[0]*nodes.shape[1]
        ids[fid]=np.arange(offset,offset+count).reshape(nodes.shape[:2])
        coordinates.append(nodes.reshape(-1,2)); offset+=count
    coordinates=np.concatenate(coordinates)
    parent=np.arange(offset)
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=int(parent[x])
        return int(x)
    for item in mesh.interfaces:
        a=_side(ids[item["block_a"]],item["side_a"])
        b=_side(ids[item["block_b"]],item["side_b"])
        if item["reverse"]: b=b[::-1]
        if len(a)!=len(b) or not np.array_equal(coordinates[a],coordinates[b]):
            raise ValueError("declared interface coordinates/counts disagree")
        for x,y in zip(a,b): parent[find(int(y))]=find(int(x))
    roots={}; points=[]; merged=np.empty(offset,dtype=int)
    for i in range(offset):
        root=find(i)
        if root not in roots:
            roots[root]=len(points); points.append(coordinates[i])
        merged[i]=roots[root]
    cells=[]; edge_counts=Counter()
    for fid,index in ids.items():
        index=merged[index]
        for i in range(index.shape[0]-1):
            for j in range(index.shape[1]-1):
                cell=tuple(map(int,(index[i,j],index[i+1,j],index[i+1,j+1],index[i,j+1])))
                if len(set(cell))!=4: raise ValueError("merged cell has duplicate vertices")
                cells.append(cell)
                for a,b in zip(cell,cell[1:]+cell[:1]): edge_counts[tuple(sorted((a,b)))]+=1
    markers={}; marker_edges=Counter()
    for item in mesh.boundaries:
        side=merged[_side(ids[item["block"]],item["side"])]
        for a,b in zip(side[:-1],side[1:]):
            edge=(int(a),int(b)); markers.setdefault(item["tag"],[]).append(edge)
            marker_edges[tuple(sorted(edge))]+=1
    if any(n not in (1,2) for n in edge_counts.values()): raise ValueError("nonmanifold exported cells")
    if {edge for edge,n in edge_counts.items() if n==1}!=set(marker_edges) or any(n!=1 for n in marker_edges.values()):
        raise ValueError("boundary markers do not cover the exterior exactly once")
    stats={"nodes":len(points),"cells":len(cells),"markers":{tag:len(edges) for tag,edges in markers.items()},
           "internal_edges":sum(n==2 for n in edge_counts.values()),"merge_rule":"declared interface identities only"}
    return points,cells,markers,stats


def validate_su2(mesh):
    """Check the exact export topology and geometry without writing any files."""
    return _validated_assembly(mesh)[3]


def _validated_assembly(mesh):
    try:
        return _assemble_su2(mesh)
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError('invalid mesh/interface/boundary metadata: '+str(error)) from error


def export_su2(mesh,path):
    """Merge only declared interfaces, including wake; never proximity-cluster."""
    points,cells,markers,stats=_validated_assembly(mesh)
    destination=Path(path); destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open("w",encoding="ascii",newline="\n") as stream:
        stream.write(f"NDIME= 2\nNELEM= {len(cells)}\n")
        for cell in cells: stream.write("9 "+" ".join(map(str,cell))+"\n")
        stream.write(f"NPOIN= {len(points)}\n")
        for x,y in points: stream.write(f"{x:.17g} {y:.17g}\n")
        stream.write(f"NMARK= {len(markers)}\n")
        for tag,edges in sorted(markers.items()):
            stream.write(f"MARKER_TAG= {tag}\nMARKER_ELEMS= {len(edges)}\n")
            for a,b in edges: stream.write(f"3 {a} {b}\n")
    return stats


def run_su2(mesh_path,config_path,*,executable="SU2_CFD",output_dir,timeout=300,convergence=None):
    """Run an installed solver; absence/failure never fabricates force values."""
    solver=shutil.which(executable)
    if solver is None: return Failure("solver_not_available",executable)
    mesh_path,config_path=Path(mesh_path).resolve(),Path(config_path).resolve()
    if not mesh_path.is_file() or not config_path.is_file(): return Failure("missing_cfd_input")
    output=Path(output_dir).resolve(); output.mkdir(parents=True,exist_ok=True)
    if (output/"history.csv").exists(): return Failure("existing_solver_history","Use a fresh run directory to prevent stale results.")
    text=config_path.read_text(encoding="utf-8")
    copied_mesh=output/"mesh.su2"
    if mesh_path!=copied_mesh: shutil.copyfile(mesh_path,copied_mesh)
    line="MESH_FILENAME= mesh.su2"
    text=re.sub(r"(?m)^\s*MESH_FILENAME\s*=.*$",line,text) if re.search(r"(?m)^\s*MESH_FILENAME\s*=",text) else text+"\n"+line+"\n"
    run_config=output/"case.cfg"; run_config.write_text(text,encoding="utf-8")
    try:
        result=subprocess.run([solver,"case.cfg"],cwd=output,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=timeout,check=False)
    except subprocess.TimeoutExpired as exc:
        def decoded(value): return value.decode("utf-8",errors="replace") if isinstance(value,bytes) else (value or "")
        (output/"solver.stdout.log").write_text(decoded(exc.stdout),encoding="utf-8")
        (output/"solver.stderr.log").write_text(decoded(exc.stderr),encoding="utf-8")
        (output/"run_status.json").write_text(json.dumps({"timed_out":True,"returncode":None,"timeout_seconds":timeout}),encoding="utf-8")
        return Failure("cfd_timeout",diagnostics={"timeout_seconds":timeout})
    (output/"solver.stdout.log").write_text(result.stdout,encoding="utf-8")
    (output/"solver.stderr.log").write_text(result.stderr,encoding="utf-8")
    (output/"run_status.json").write_text(json.dumps({"timed_out":False,"returncode":result.returncode}),encoding="utf-8")
    if result.returncode: return Failure("cfd_nonzero_exit",diagnostics={"returncode":result.returncode})
    history=output/"history.csv"
    if not history.is_file(): return Failure("missing_solver_history")
    with history.open(encoding="utf-8",newline="") as stream: rows=list(csv.DictReader(stream,skipinitialspace=True))
    if not rows: return Failure("empty_solver_history")
    rows=[{k.strip().strip('"'):v.strip() for k,v in row.items()} for row in rows]
    from .convergence import check_history
    check=check_history(rows,**convergence) if convergence is not None else {"converged":False,"reason":"convergence_criteria_not_requested"}
    forces={key:float(rows[-1][key]) for key in ("CL","CD","CMz") if key in rows[-1] and np.isfinite(float(rows[-1][key]))}
    report={"status":"solver_completed","history_last":rows[-1],"iterations_recorded":len(rows),"forces":forces,
            "convergence":check,"convergence_validated":bool(check["converged"]),"note":"Successful exit alone is not a convergence or CFD-accuracy claim."}
    (output/"cfd_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report
