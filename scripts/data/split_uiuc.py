"""Deterministic, inspectable UIUC grouping, holdouts and independent polygon QA.

Python standard library only. Paths in generated reports are relative or IDs.
Example (from a repository root):
 python scripts/data/split_uiuc.py --input data/processed/uiuc_v1/samples.jsonl \
   --output data/splits/uiuc_v1 --config data/splits/uiuc_v1/split_config.json

Raw source checksum revalidation is optional: --raw-root path/to/coordinates.
No learned classifier is used. Near-duplicate matches cause grouping, not deletion.
"""
from __future__ import annotations

import argparse
from array import array
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import struct


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def signed_area(poly):
    return sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(poly,poly[1:]+poly[:1]))/2


def exact_polygon_hash(poly):
    """Independent float-level cyclic/orientation-invariant identity screen."""
    vertices=list(map(tuple,poly))
    if signed_area(vertices)<0:
        vertices.reverse()
    anchor=min(range(len(vertices)),key=vertices.__getitem__)
    vertices=vertices[anchor:]+vertices[:anchor]
    return sha256(b"".join(struct.pack("<dd",*p) for p in vertices))


def orientation(a,b,c):
    p=(b[0]-a[0])*(c[1]-a[1])
    q=(b[1]-a[1])*(c[0]-a[0])
    d=p-q
    tolerance=2e-14*(abs(p)+abs(q))
    return 1 if d>tolerance else -1 if d < -tolerance else 0


def on_segment(a,b,c,eps):
    return (min(a[0],b[0])-eps<=c[0]<=max(a[0],b[0])+eps and
            min(a[1],b[1])-eps<=c[1]<=max(a[1],b[1])+eps)


def intersects(a,b,c,d,eps):
    if (max(a[0],b[0])+eps<min(c[0],d[0]) or max(c[0],d[0])+eps<min(a[0],b[0]) or
        max(a[1],b[1])+eps<min(c[1],d[1]) or max(c[1],d[1])+eps<min(a[1],b[1])):
        return False
    o1,o2,o3,o4=orientation(a,b,c),orientation(a,b,d),orientation(c,d,a),orientation(c,d,b)
    return (o1*o2<0 and o3*o4<0) or (o1==0 and on_segment(a,b,c,eps)) or (o2==0 and on_segment(a,b,d,eps)) or (o3==0 and on_segment(c,d,a,eps)) or (o4==0 and on_segment(c,d,b,eps))


def polygon_qa(poly,eps):
    errors=[]
    if len(poly)<3:
        return ["fewer_than_three_vertices"],{}
    if not all(isinstance(p,(list,tuple)) and len(p)==2 and all(isinstance(v,(float,int)) and math.isfinite(v) for v in p) for p in poly):
        return ["nonfinite_or_malformed_coordinate"],{}
    vertices=list(map(tuple,poly))
    area=signed_area(poly)
    if area<=0:errors.append("nonpositive_CCW_area")
    if len(set(vertices))!=len(poly):errors.append("repeated_vertex")
    if vertices[0]!=min(vertices):errors.append("noncanonical_start")
    xmin,xmax=min(p[0] for p in poly),max(p[0] for p in poly)
    if abs(xmin)>eps or abs(xmax-1)>eps:errors.append("unexpected_x_normalization")
    if any(a==b for a,b in zip(vertices,vertices[1:]+vertices[:1])):errors.append("zero_length_edge")
    # A sweep on x bounding boxes is independent of the cleaner implementation.
    segments=[(min(a[0],b[0]),max(a[0],b[0]),i,a,b) for i,(a,b) in enumerate(zip(vertices,vertices[1:]+vertices[:1]))]
    active=[]
    crossings=[]
    for left,right,i,a,b in sorted(segments):
        active=[s for s in active if s[1]+eps>=left]
        for _,_,j,c,d in active:
            if abs(i-j)==1 or {i,j}=={0,len(poly)-1}:continue
            if intersects(a,b,c,d,eps):crossings.append([i,j])
        active.append((left,right,i,a,b))
    if crossings:errors.append("nonadjacent_segment_intersection_or_touch")
    return errors,dict(area=area,point_count=len(poly),intersection_edge_pairs=crossings[:10])


def shape_signature(poly,k):
    """Extrema of vertical sections; unambiguous for accepted monotone airfoils."""
    xs=[0.5*(1-math.cos(math.pi*(i+0.5)/k)) for i in range(k)]
    low=[float("inf")]*k
    high=[-float("inf")]*k
    hits=[0]*k
    for a,b in zip(poly,poly[1:]+poly[:1]):
        if a[0]==b[0]:continue
        start=bisect_left(xs,min(a[0],b[0]))
        end=bisect_right(xs,max(a[0],b[0]))
        for j in range(start,end):
            y=a[1]+(b[1]-a[1])*(xs[j]-a[0])/(b[0]-a[0])
            low[j]=min(low[j],y)
            high[j]=max(high[j],y)
            hits[j]+=1
    if any(h<2 for h in hits) or any(not math.isfinite(y) for y in low+high):
        raise ValueError("uncovered_vertical_section")
    return array("d",low+high),max(hits)


def name_key(name,config):
    value=name.lower().strip()
    for rule in config["named_family_rules"]:
        if re.search(rule["pattern"],value):return "name:"+rule["key"]
    # Suffix stripping preserves parent -> smoothed/corrected/modified linkage.
    value=re.sub(r"[^a-z0-9]","",value)
    while True:
        previous=value
        for suffix in config["smooth_or_modified_suffixes"]:
            if value.endswith(suffix) and len(value)>len(suffix)+1:
                value=value[:-len(suffix)]
                break
        if value==previous:break
    match=re.match(r"[a-z]+",value)
    return "name:"+(match.group() if match else value)


class UnionFind:
    def __init__(self,n):self.parent=list(range(n))
    def find(self,a):
        while self.parent[a]!=a:
            self.parent[a]=self.parent[self.parent[a]]
            a=self.parent[a]
        return a
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.parent[max(a,b)]=min(a,b)


def write_json(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def run(args):
    config=json.loads(Path(args.config).read_text(encoding="utf-8"))
    src=Path(args.input)
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    original=src.read_bytes()
    samples=sorted((json.loads(line) for line in original.decode("utf-8").splitlines() if line.strip()),key=lambda s:s["sample_id"])
    ids=[s["sample_id"] for s in samples]
    assert len(ids)==len(set(ids)),"Duplicate sample IDs"
    assert len(samples)>0,"Empty dataset"
    eps=config["geometry_epsilon"]
    families=[]
    exact=[]
    signatures=[]
    geometry_failures=[]
    raw_failures=[]
    raw_checked=0
    QA=[]
    source_occurrences=Counter()
    for sample in samples:
        sid=sample["sample_id"]
        assert sid=="uiuc_"+sample["canonical_sha256"][:16],sid
        errors,check=polygon_qa(sample["polygon"],eps)
        try:
            sig,max_hits=shape_signature(sample["polygon"],config["signature_samples"])
            if max_hits>2:check["vertical_section_max_intersections"]=max_hits
        except ValueError as ex:
            errors.append(str(ex))
            sig=array("d",[float("nan")]*(2*config["signature_samples"]))
        recorded=sample.get("quality",{}).get("signed_area")
        if recorded is not None and abs(recorded-check.get("area",0))>1e-10:
            errors.append("reported_area_mismatch")
        if errors:geometry_failures.append(dict(sample_id=sid,aliases=sample.get("aliases",[]),errors=errors,details=check))
        QA.append(dict(sample_id=sid,passed=not errors,**check))
        signatures.append(sig)
        exact.append(exact_polygon_hash(sample["polygon"]))
        aliases=sample.get("aliases") or [Path(f["filename"]).stem for f in sample["source_files"]]
        families.append(sorted(set(name_key(a,config) for a in aliases)))
        for source in sample["source_files"]:
            rel=source["relative_path"]
            source_occurrences[rel]+=1
            if args.raw_root:
                rawroot=Path(args.raw_root).resolve()
                path=(rawroot/rel).resolve()
                if not path.is_relative_to(rawroot):
                    raw_failures.append(dict(sample_id=sid,source=rel,error="path_outside_raw_root"))
                elif not path.exists() or sha256(path.read_bytes())!=source["sha256"]:
                    raw_failures.append(dict(sample_id=sid,source=rel,error="missing_or_hash_mismatch"))
                else:raw_checked+=1
    failures=dict(geometry_failures=geometry_failures,raw_source_failures=raw_failures,
                  repeated_source_references=[p for p,n in source_occurrences.items() if n>1])
    write_json(output/"independent_qa_failures.json",failures)
    if geometry_failures or raw_failures or failures["repeated_source_references"]:
        raise RuntimeError("Independent QA failed; inspect independent_qa_failures.json before publishing splits")

    uf=UnionFind(len(samples))
    first={}
    for i,keys in enumerate(families):
        for key in keys:
            if key in first:uf.union(i,first[key])
            else:first[key]=i
    exact_first={}
    exact_pairs=[]
    for i,key in enumerate(exact):
        if key in exact_first:
            exact_pairs.append([ids[exact_first[key]],ids[i]])
            uf.union(i,exact_first[key])
        else:exact_first[key]=i

    threshold=config["near_duplicate_max_chord"]
    rms_threshold=config["near_duplicate_rms_chord"]
    k=config["signature_samples"]
    # The upper midpoint coordinate is a necessary bound for max-norm matches.
    # Sorting on it only prunes impossible pairs; every remaining pair is checked.
    anchor=k+k//2
    order=sorted(range(len(samples)),key=lambda i:(signatures[i][anchor],ids[i]))
    near_pairs=[]
    candidate_comparisons=0
    for pos,i in enumerate(order):
        left=signatures[i]
        for j in order[pos+1:]:
            right=signatures[j]
            if right[anchor]-left[anchor]>threshold:break
            candidate_comparisons+=1
            if any(abs(left[t]-right[t])>threshold for t in (0,k//4,k//2,3*k//4,k-1,k,k+k//4,anchor,k+3*k//4,2*k-1)):
                continue
            maximum=0.0
            squared=0.0
            for a,b in zip(left,right):
                d=abs(a-b)
                if d>threshold:break
                maximum=max(maximum,d)
                squared+=d*d
            else:
                rms=math.sqrt(squared/len(left))
                if rms<=rms_threshold:
                    uf.union(i,j)
                    near_pairs.append(dict(sample_a=ids[i],sample_b=ids[j],rms_chord=rms,max_abs_chord=maximum))

    components=defaultdict(list)
    for i in range(len(samples)):components[uf.find(i)].append(i)
    groups=[]
    for indices in components.values():
        members=sorted(ids[i] for i in indices)
        groupid="grp_"+sha256("\n".join(members).encode())[:16]
        keys=sorted({key for i in indices for key in families[i]})
        groups.append(dict(group_id=groupid,sample_ids=members,family_keys=keys,count=len(members),indices=indices))

    ratios=config["ratios"]
    assert set(ratios)=={"train","val","test"} and abs(sum(ratios.values())-1)<1e-12
    targets={split:ratios[split]*len(samples) for split in ratios}
    counts={split:0 for split in ratios}
    # Largest-first whole-group packing minimizes deviation from sample targets;
    # hashes break ties. Snapshot stability is guaranteed by the input hash/seed.
    seed=config["seed"]
    forced_test_keys=set(config.get("forced_test_family_keys",[]))
    for group in groups:
        group["forced_test"]=bool(forced_test_keys.intersection(group["family_keys"]))
    groups.sort(key=lambda g:(not g["forced_test"],-g["count"],sha256((seed+g["group_id"]).encode())))
    assignment={}
    group_of={}
    family_splits=defaultdict(set)
    for group in groups:
        size=group["count"]
        def cost(split):
            proposed={key:counts[key]+(size if key==split else 0) for key in ratios}
            score=sum((proposed[key]-targets[key])**2/max(1,targets[key]) for key in ratios)
            return score,sha256((seed+group["group_id"]+split).encode())
        split="test" if group["forced_test"] else min(ratios,key=cost)
        counts[split]+=size
        group["split"]=split
        for sid in group["sample_ids"]:
            assignment[sid]=split
            group_of[sid]=group["group_id"]
        for key in group["family_keys"]:family_splits[key].add(split)
    assert set(assignment)==set(ids)
    assert all(len(splits)==1 for splits in family_splits.values())
    assert all(assignment[a]==assignment[b] for a,b in exact_pairs)
    assert all(assignment[p["sample_a"]]==assignment[p["sample_b"]] for p in near_pairs)
    assert all(counts[s]>0 for s in ratios),"Grouping left an empty split; inspect groups/config"
    for split in ratios:
        write_json(output/(split+".json"),sorted(sid for sid in ids if assignment[sid]==split))
    with (output/"split_manifest.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["sample_id","split","group_id","family_keys","aliases","canonical_sha256","point_count","source_file_count"])
        writer.writeheader()
        for i,s in enumerate(samples):
            writer.writerow(dict(sample_id=ids[i],split=assignment[ids[i]],group_id=group_of[ids[i]],
                family_keys=";".join(families[i]),aliases=";".join(s.get("aliases",[])),
                canonical_sha256=s["canonical_sha256"],point_count=len(s["polygon"]),source_file_count=len(s["source_files"])))
    with (output/"near_duplicate_pairs.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["sample_a","sample_b","rms_chord","max_abs_chord"])
        writer.writeheader()
        writer.writerows(sorted(near_pairs,key=lambda p:(p["sample_a"],p["sample_b"])))
    write_json(output/"groups.json",[{key:value for key,value in g.items() if key!="indices"} for g in sorted(groups,key=lambda g:g["group_id"])])
    write_json(output/"independent_geometry_qa.json",dict(checked_samples=len(samples),failures=0,records=QA))
    largest=sorted(groups,key=lambda g:(-g["count"],g["group_id"]))[:15]
    report=dict(schema_version=1,version=config["version"],seed=seed,
        input=dict(filename=src.name,bytes=len(original),sha256=sha256(original)),
        config_sha256=sha256(Path(args.config).read_bytes()),sample_count=len(samples),group_count=len(groups),
        target_ratios=ratios,actual_counts=counts,actual_ratios={s:counts[s]/len(samples) for s in ratios},
        forced_test_family_keys=sorted(forced_test_keys),
        forced_test_sample_count=sum(g["count"] for g in groups if g["forced_test"]),
        benchmark_holdout_reason="NACA0012 is a planned CFD validation benchmark. Entire connected groups containing name:naca are reserved for final evaluation; do not use these samples for fitting, reward tuning, hyperparameters or checkpoint selection.",
        independent_qa=dict(all_polygons_rechecked=True,geometry_failures=0,
                            source_hashes_checked=raw_checked,raw_root_provided=bool(args.raw_root),
                            accepted_source_file_references=len(source_occurrences),source_reference_conflicts=0),
        leakage_checks=dict(unique_ids=True,disjoint_id_sets=True,all_samples_assigned_once=True,
            name_family_cross_split_leaks=0,float_exact_duplicate_cross_split_leaks=0,
            detected_near_duplicate_cross_split_leaks=0,groups_cross_split_leaks=0),
        exact_float_duplicate_pairs=exact_pairs,detected_near_duplicate_pair_count=len(near_pairs),
        candidate_pair_comparisons=candidate_comparisons,
        shape_signature=dict(samples_per_surface=k,rms_threshold_chord=rms_threshold,max_threshold_chord=threshold),
        grouping="Connected components of conservative name-family, independent float-exact and thresholded shape-similarity edges. Near-duplicates are retained, grouped and never deleted here.",
        allocation="Configured benchmark families and their entire connected groups are forced into test first. Remaining groups: largest first, minimize sample-count weighted squared deviation from 80/10/10 targets; seed hashes break ties. Ratios are approximate because groups are indivisible and forced holdouts can exceed targets.",
        largest_groups=[{key:value for key,value in g.items() if key not in ("indices","sample_ids")} for g in largest],
        limitations=[
           "Name prefixes are conservative proxies, not a reviewed physical airfoil genealogy; some unrelated designs are intentionally over-grouped.",
           "Within-name families, aliases and detected normalized near-duplicates cannot cross splits. This is not proof that all undocumented cross-name ancestry was discovered.",
           "129 interpolated vertical sections and configured tolerances define the similarity screen. Rotation, reflection, arbitrary affine deformation and high-frequency differences between sections are not fully matched.",
           "Connected components may include pairs whose own shape distance exceeds the threshold via intermediate relatives; grouping is deliberately transitive.",
           "Train/val/test represent geometric-family holdouts only. Geometry files contain no expert block topology, action demonstrations or CFD accuracy labels.",
           "Independent geometry screens do not certify meshing or CFD suitability; no solver was run.",
           "IDs and assignments are reproducible for the frozen input hash/config. Adding/removing data may change groups and packing; freeze this manifest for evaluation."])
    write_json(output/"split_report.json",report)
    notes="""UIUC v1 几何家族划分与独立复验

train.json、val.json、test.json均为稳定sample_id的JSON数组；split_manifest.csv为总映射。
名称家族、同一母几何的平滑/修改版本、浮点级精确重复和阈值内几何近重复先归组，整组划分。
名称先匹配公开配置的显式规则，再保守使用首段字母前缀；这是一种故意偏粗的命名代理，不代表人工确认的气动家谱。
按分组大小从大到小分配，约80/10/10，整组不可拆分，因此最终比例不强制精确相等。
因NACA0012是计划的CFD验证算例，包含name:naca的整个连通组先强制保留test；平滑、别名、近重复和相连的其他家族一并保留。
forced_test_family_keys记录在配置和报告中。该测试集只用于最终评价，不得用于拟合、奖励调参、超参数或检查点选择。

几何近重复检测：每翼型上下表面各129个固定余弦x/c截面的y/c指纹，RMS≤1e-4且最大偏差≤5e-4即连边归组。
阈值和名称规则见split_config.json。坐标平移/等比正尺度和重采样对指纹的影响被控制；未做任意旋转、镜像或仿射形状匹配。
近重复不被删除；连通分量的传递合并可能比成对阈值更保守。near_duplicate_pairs.csv保留全部检测到的直接匹配。

独立复验遍历全部清洗polygon，复查有限坐标、归一化、CCW面积、重复点、零边、自交/非相邻接触；可选复核原始源文件SHA256。
另外逐项检查ID唯一、集合互斥、全部样本仅归一集、家族/分组/精确重复/检测到的近重复不跨集。
只保证上述明示规则下无泄漏；没有声称已找到所有未标注家族关系或完成彻底几何族去重。
归一化由每个几何自己的弦长定义，不从全库拟合统计量；后续任何需要拟合的标准化仍只能用训练集。
验证集用于调参、奖励和检查点选择，测试集只作最终评价；本数据不包含专家分块、动作轨迹或CFD真值。

复现：python scripts/data/split_uiuc.py --input data/processed/uiuc_v1/samples.jsonl --output data/splits/uiuc_v1 --config data/splits/uiuc_v1/split_config.json
如本地保留UIUC原始坐标，可加 --raw-root 指向原coordinates目录。输出中不保存本机绝对路径。
"""
    (output/"说明.txt").write_text(notes,encoding="utf-8")
    print(json.dumps(dict(samples=len(samples),groups=len(groups),counts=counts,near_pairs=len(near_pairs),qa="passed"),ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--config",required=True)
    parser.add_argument("--raw-root")
    run(parser.parse_args())
