"""Mesh runtime checks use a synthetic non-NACA profile and training data only."""
import json
from pathlib import Path
import tempfile
import numpy as np
from smartcfd.core.types import BlockGraph,FaceBlock,Failure,MeshSpec
from smartcfd.geometry.domain import CMapping,make_domain,sharp_eligibility
from smartcfd.meshing.generator import generate_mesh,save_mesh,side_nodes
from smartcfd.meshing.tfi import coons_patch
from smartcfd.evaluation.mesh_quality import evaluate_mesh
from smartcfd.cfd.solver import export_su2,run_su2
from smartcfd.cfd.convergence import check_history


def profile():
    x=(1-np.cos(np.linspace(0,np.pi,81)))/2
    y=.18*np.sqrt(x)*(1-x)
    p=np.vstack((np.column_stack((x,-y)),np.column_stack((x[-2:0:-1],y[-2:0:-1]))))
    return {"sample_id":"synthetic_non_naca", "polygon":p.tolist(),"explicitly_closed":True}


def graph(outer_bands=1):
    u=[0.,.25,.5,.75,1.]; v=[0.,.5]+([.75] if outer_bands==2 else [])+[1.]
    vertices={j*5+i:(x,y) for j,y in enumerate(v) for i,x in enumerate(u)}
    faces={j*4+i:FaceBlock(j*4+i,(j*5+i,j*5+i+1,(j+1)*5+i+1,(j+1)*5+i),j==0) for j in range(len(v)-1) for i in range(4)}
    return BlockGraph(vertices,faces,len(vertices),len(faces))


def test_mapping_wake_and_protected_normals():
    domain=make_domain(profile()); mapping=CMapping(domain)
    u=np.linspace(0,.25,33)
    assert np.allclose(mapping.evaluate(np.column_stack((u,np.zeros_like(u)))),mapping.evaluate(np.column_stack((1-u,np.zeros_like(u)))),rtol=0,atol=1e-11)
    wall=np.array([[.32,0.],[.44,0.],[.58,0.],[.69,0.]])
    p,n=mapping.inner_and_normal(wall[:,0]); uv=wall.copy(); uv[:,1]=.1
    offset=mapping.evaluate(uv)-p
    assert np.allclose(offset/np.linalg.norm(offset,axis=1)[:,None],n,atol=1e-9)
    wake_points=mapping.evaluate(np.column_stack((np.linspace(.25,0,17),np.zeros(17))))
    assert 0<wake_points[1,0]-wake_points[0,0]<.05
    assert domain.metadata["mapping_version"]=="c_normal_envelope_wakegraded_v2"


def test_tfi_rectangle_and_inverted_quad():
    t=np.linspace(0,1,9)
    patch=coons_patch(np.column_stack((t,0*t)),np.column_stack((1+0*t,t)),np.column_stack((t,1+0*t)),np.column_stack((0*t,t)))
    assert evaluate_mesh({0:patch})["valid"]
    assert not evaluate_mesh({0:patch[::-1]})["valid"]


def test_eight_and_twelve_blocks_export_and_roundtrip():
    domain=make_domain(profile())
    for bands in (1,2):
        mesh=generate_mesh(domain,graph(bands),MeshSpec(intervals=16,protected_intervals=32,smoothing_steps=2))
        assert not isinstance(mesh,Failure),mesh
        assert mesh.quality["valid"],mesh.quality
        assert mesh.metadata["mapping_version"]==domain.metadata["mapping_version"]
        assert len(mesh.blocks)==4+4*bands
        assert sum(x["kind"]=="wake" for x in mesh.interfaces)==1
        assert {x["tag"] for x in mesh.boundaries}=={"wall","farfield"}
        for item in mesh.interfaces:
            a=side_nodes(mesh.blocks[item["block_a"]],item["side_a"])
            b=side_nodes(mesh.blocks[item["block_b"]],item["side_b"])
            assert np.array_equal(a,b[::-1] if item["reverse"] else b)
        with tempfile.TemporaryDirectory() as folder:
            paths=save_mesh(mesh,folder)
            with np.load(paths[0]) as archive:
                assert all(np.array_equal(archive[f"block_{fid}"],nodes) for fid,nodes in mesh.blocks.items())
            assert json.loads(paths[1].read_text())["quality"]["valid"]
            report=export_su2(mesh,Path(folder)/"mesh.su2")
            assert report["cells"]==mesh.quality["cells"]
            assert report["markers"]["wall"]==32
            assert "9 " in (Path(folder)/"mesh.su2").read_text()


def test_invalid_source_graph_and_budget_fail_explicitly():
    sample=profile(); sample["explicitly_closed"]=False
    assert sharp_eligibility(sample)[0] is False
    domain=make_domain(profile())
    assert isinstance(generate_mesh(domain,graph(),MeshSpec(max_cells=1)),Failure)
    broken=graph(); broken.faces[0].vertices=broken.faces[0].vertices[:3]
    assert isinstance(generate_mesh(domain,broken),Failure)
    assert isinstance(run_su2("none","none",executable="deliberately_missing_su2_executable",output_dir="unused"),Failure)


def test_protected_strip_is_unchanged_by_smoothing():
    domain=make_domain(profile()); g=graph()
    a=generate_mesh(domain,g,MeshSpec(smoothing_steps=0))
    b=generate_mesh(domain,g,MeshSpec(smoothing_steps=3))
    assert not isinstance(a,Failure) and not isinstance(b,Failure)
    assert all(np.array_equal(a.blocks[fid],b.blocks[fid]) for fid in g.protected_face_ids)


def test_concave_positive_area_cell_is_rejected():
    points=np.array([[[0.,0.],[0.,1.]],[[1.,0.],[.2,.2]]])
    assert not evaluate_mesh({0:points})["valid"]


def test_export_refuses_even_tiny_unmatched_interface():
    mesh=generate_mesh(make_domain(profile()),graph())
    item=next(x for x in mesh.interfaces if x["kind"]=="block")
    side_nodes(mesh.blocks[item["block_b"]],item["side_b"])[1,0]+=1e-12
    with tempfile.TemporaryDirectory() as folder:
        try:
            export_su2(mesh,Path(folder)/"bad.su2")
        except ValueError as exc:
            assert "interface" in str(exc)
        else:
            raise AssertionError("Proximity merging must not conceal interface mismatch")


def test_solver_completion_is_not_convergence(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import smartcfd.cfd.solver as solver
    mesh=tmp_path/"source.su2"; mesh.write_text("NDIME= 2\n")
    config=tmp_path/"source.cfg"; config.write_text("MESH_FILENAME= original.su2\n")
    monkeypatch.setattr(solver.shutil,"which",lambda value:"mock_SU2_CFD")
    def execute(*args,**kwargs):
        (kwargs["cwd"]/"history.csv").write_text('"Inner_Iter","rms[Rho]","CL","CD"\n0,-7,.4,.01\n1,-8,.4,.01\n')
        return SimpleNamespace(returncode=0,stdout="completed two iterations",stderr="")
    monkeypatch.setattr(solver.subprocess,"run",execute)
    output=tmp_path/"run"
    result=run_su2(mesh,config,output_dir=output,convergence={})
    assert result["status"]=="solver_completed" and not result["convergence_validated"]
    assert result["forces"]["CL"]==.4
    assert json.loads((output/"run_status.json").read_text())["returncode"]==0
    assert (output/"solver.stdout.log").is_file() and (output/"solver.stderr.log").is_file()
    assert isinstance(run_su2(mesh,config,output_dir=output),Failure)


def test_convergence_requires_residuals_and_stable_forces():
    rows=[{"rms[Rho]":"-7","CL":".4","CD":".01"} for _ in range(120)]
    assert check_history(rows)["converged"]
    assert not check_history(rows[:3])["converged"]
    rows[-1]["CD"]=".03"
    assert not check_history(rows)["converged"]


def test_frozen_training_airfoil_and_both_real_rule_graphs():
    """Use only an ID in train.json; never develop against the NACA holdout."""
    from smartcfd.baselines.rule_partition import rule_graph
    root=Path(__file__).resolve().parents[1]
    train=set(json.loads((root/"data/splits/uiuc_v1/train.json").read_text()))
    sample=None
    for line in (root/"data/processed/uiuc_v1/samples.jsonl").open(encoding="utf-8"):
        record=json.loads(line)
        if record["sample_id"] in train and sharp_eligibility(record)[0]:
            sample=record; break
    assert sample is not None
    domain=make_domain(sample)
    for variant in (0,1):
        mesh=generate_mesh(domain,rule_graph(domain,variant=variant),MeshSpec(intervals=8,protected_intervals=16))
        assert not isinstance(mesh,Failure) and mesh.quality["valid"]
        assert mesh.quality["min_scaled_jacobian"]>0
