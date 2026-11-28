"""Figures, experiment workbook and independent checks of the sealed results."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from math import isclose
import os
import subprocess
import sys
import time
from xml.etree import ElementTree
import zipfile

import numpy as np

from smartcfd.airfoil_decomposition.config import AirfoilDecompositionConfig
from smartcfd.airfoil_decomposition.experiment import _json_default, file_sha, write_json
from smartcfd.airfoil_decomposition.geometry import CurveGeometry
from smartcfd.airfoil_decomposition.visualize import _plt
from smartcfd.cfd.convergence import check_history
from smartcfd.core.types import BlockGraph
from smartcfd.evaluation.jacobian import corner_scaled_jacobians
from smartcfd.partition.constraints import validate_graph
from smartcfd.partition.topology import all_quadrilateral, edge_incidence, fingerprint

from .cfd import _simulate, read_history, solver_path
from .evaluate import case_key, jobs
from .mesh import BUDGETS, evaluate_mesh_budget
from .protocol import BASELINES, B_COMMIT, ROOT, SAC_ARMS, published_bytes, read_json
from .rollout import Environments, episode, selection_key, summarize
from .train import get_inference_agent

ARMS = ("E1", "E2", "E3", "E4", "E4a", "E5", "E6", "E7", "E8")
COLORS = dict(zip(ARMS, ("#375D81", "#788390", "#916C4A", "#8C6684", "#967FA4",
                         "#245D67", "#C97C33", "#2E7F78", "#427EAA")))
FIGURES = ("01_framework", "02_mdp", "03_evolution", "04_topologies", "05_training",
           "06_success", "07_mesh_quality", "08_ablations", "09_near_wall", "10_cfd")


def _save(plt, fig, directory, number):
    name = FIGURES[number-1]
    fig.savefig(directory/f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(directory/f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _diagram(plt, directory, number, title, labels):
    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.set(xlim=(0, 1), ylim=(0, 1), title=title)
    ax.axis("off")
    centers = np.linspace(.10, .90, len(labels))
    for i, (center, label) in enumerate(zip(centers, labels)):
        ax.text(center, .51, label, ha="center", va="center", fontsize=11, color="#173B58",
                bbox={"boxstyle": "round,pad=.6", "facecolor": "#e7f1f4", "edgecolor": "#7696aa"})
        if i:
            ax.annotate("", xy=(center-.076, .51), xytext=(centers[i-1]+.076, .51),
                        arrowprops={"arrowstyle": "->", "color": "#365a72"})
    _save(plt, fig, directory, number)


def _graph(ax, geometry, graph_dict):
    if graph_dict is None:
        ax.text(.5, .5, "Geometry unavailable", transform=ax.transAxes, ha="center")
        ax.axis("off")
        return
    graph = BlockGraph.from_dict(graph_dict)
    for face in graph.faces.values():
        xy = np.concatenate([geometry.curve(graph.vertices[a], graph.vertices[b])[:-1]
                             for a, b in zip(face.vertices, face.vertices[1:]+face.vertices[:1])])
        color = "#416e82" if face.protected else "#40998e" if len(face.vertices) == 4 else "#c8874b"
        ax.fill(*xy.T, color=color, alpha=.16)
        ax.plot(*np.vstack([xy, xy[0]]).T, color=color, lw=.75)
    ax.set_aspect("equal")
    ax.set(xlabel="x / chord", ylabel="y / chord")
    ax.tick_params(labelsize=8)


def _evaluation(output, arm, seed=42, tag=None):
    tag = tag or ("rule" if arm in ("E1", "E2", "E3") else "best")
    return read_json(output/"evaluation"/f"{case_key(arm, seed, tag)}.json")


def _example(output, cfg):
    pool = {s["sample_id"]: s for s in read_json(output/"evaluation_cases.json")["cases"]}
    for arm in ("E7", "E8", "E6", "E5", "E1"):
        for seed in (42,) if arm == "E1" else cfg["seeds"]:
            rows = _evaluation(output, arm, seed)["rows"]
            row = next((r for r in rows if r["success"]), None)
            if row is not None:
                return pool[row["sample_id"]], arm, seed, row
    rows = _evaluation(output, "E7")["rows"]
    row = next((r for r in rows if "initial_graph" in r), rows[0])
    return pool[row["sample_id"]], "E7", 42, row


def _example_timing(output, cfg, sample, arm, seed, saved):
    if not saved["success"]:
        return [], None
    import torch

    torch.set_num_threads(cfg["runtime"]["torch_threads"])
    sources = [(arm, seed, saved)]
    if arm != "E1":
        fixed = next(row for row in _evaluation(output, "E1")["rows"] if row["sample_id"] == sample["sample_id"])
        if fixed["success"]:
            sources.append(("E1", 42, fixed))
    timings, example_mesh = [], None
    for method, run_seed, reference in sources:
        agent = None
        if method not in BASELINES:
            filename = f"arms/{method}_seed{run_seed}/best.pt"
            matrix = read_json(output/"training_matrix.json")
            if file_sha(output/filename) != matrix["checkpoints"][filename]:
                raise ValueError("Representative timing checkpoint differs from locked weights")
            agent, _ = get_inference_agent(output/filename, method, run_seed, file_sha(output/"prepared.json"))
        pool = Environments(cfg["geometry"], method)
        result, _ = episode(pool, sample, BASELINES.get(method, "learned"), agent=agent, seed=run_seed, mesh=False)
        if not result["success"] or result["graph"] != reference["graph"]:
            raise ValueError("Representative timing did not reproduce the sealed topology")
        graph = BlockGraph.from_dict(result["graph"])
        domain = pool.get(sample).geometry.domain
        meshes = []
        for budget in BUDGETS:
            start = time.perf_counter()
            metrics, mesh = evaluate_mesh_budget(domain, graph, budget)
            seconds = time.perf_counter() - start
            archived_metrics = json.loads(json.dumps(metrics, default=_json_default, allow_nan=False))
            if archived_metrics != reference["mesh"][str(budget)]:
                raise ValueError("Representative timing did not reproduce the sealed mesh metrics")
            meshes.append({"target": budget, "seconds": seconds, "valid": metrics["valid"]})
            if method == arm and budget == BUDGETS[0]:
                example_mesh = mesh
        timings.append({"arm": method, "seed": run_seed, "partition_seconds": result["seconds"], "meshes": meshes})
    return timings, example_mesh


def _figures(output, directory, cfg, summary, cfd):
    plt = _plt()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    _diagram(plt, directory, 1, "Automatic partitioning and verification",
             ["Input airfoil\nC-domain", "Legal topology\ncandidates", "BC and\nmasked SAC",
              "Terminal graph\nand TFI", "Mesh audit\nand SU2"])
    _diagram(plt, directory, 2, "One MDP transition",
             ["Half-edge graph\nand geometry", "Candidate score\nand mask", "Cut, insert\nor STOP",
              "Verify and\ncommit edit", "Reward and\nnext state"])
    sample, arm, seed, example = _example(output, cfg)
    geom = CurveGeometry(sample, AirfoilDecompositionConfig(**cfg["geometry"])) if "initial_graph" in example else None
    frames = [("Initial", example.get("initial_graph"))] + [
        (f"{i} {record['candidate']['kind'].replace('insert_', '')}", record["graph"])
        for i, record in enumerate(example.get("actions", ()), 1)]
    cols = min(len(frames), 3 if len(frames) in (5, 6, 9) else 4)
    nrows = (len(frames)+cols-1)//cols
    fig, axes = plt.subplots(nrows, cols, squeeze=False, figsize=(3.5*cols, 3.2*nrows), layout="constrained")
    for i, ax in enumerate(axes.flat):
        if i < len(frames):
            title, graph = frames[i]
            _graph(ax, geom, graph)
            ax.set_title(title)
        elif example["success"] and i == len(frames):
            _graph(ax, geom, example["graph"])
            ax.set(xlim=(-.13, 1.20), ylim=(-.35, .35), title="Terminal near-field topology")
        else:
            ax.axis("off")
    fig.suptitle(f"{arm} seed {seed} / {sample['sample_id']} / {example['reason']}")
    _save(plt, fig, directory, 3)
    fig, axes = plt.subplots(1, 5, figsize=(15, 4), layout="constrained")
    for ax, method in zip(axes, ("E1", "E4", "E5", "E7", "E8")):
        case = next(r for r in _evaluation(output, method)["rows"] if r["sample_id"] == sample["sample_id"])
        _graph(ax, geom, case.get("graph"))
        ax.set_title(f"{method} / {'complete' if case['success'] else 'failed'}")
    fig.suptitle("Final topology on the same evaluated wing section")
    _save(plt, fig, directory, 4)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for label in SAC_ARMS:
        histories = [read_json(output/"arms"/f"{label}_seed{i}"/"completion.json")["validation"] for i in cfg["seeds"]]
        x = [r["step"] for r in histories[0]]
        for ax, metric, title in zip(axes, ("success_rate", "mesh_successful"),
                                     ("Partition success rate", "Valid coarse meshes")):
            series = np.asarray([[r[metric] for r in h] for h in histories])
            for row in series:
                ax.plot(x, row, color=COLORS[label], alpha=.22, lw=.8)
            ax.plot(x, series.mean(axis=0), label=label, color=COLORS[label], lw=1.9)
            ax.set(xlabel="Online environment interactions", ylabel=title)
            ax.grid(alpha=.17)
    axes[0].legend(ncol=3, loc="best")
    _save(plt, fig, directory, 5)
    best = {arm: [r for r in summary["runs"] if r["arm"] == arm and r["tag"] != "final"] for arm in ARMS}
    final = {arm: [r for r in summary["runs"] if r["arm"] == arm and r["tag"] == "final"] for arm in ARMS}
    fig, ax = plt.subplots(figsize=(10.5, 4.5), layout="constrained")
    for i, label in enumerate(ARMS):
        ys = [r["success_rate"] for r in best[label]]
        last = [r["success_rate"] for r in final[label]]
        selected_x = i-.19 if last else i
        ax.bar(selected_x, np.mean(ys), width=.34 if last else .64, color=COLORS[label], alpha=.85,
               label="Rule baseline or validation-selected" if i == 0 else None)
        ax.scatter(np.linspace(selected_x-.10, selected_x+.10, len(ys)), ys, s=18,
                   color="#172d41", zorder=3, clip_on=False)
        if last:
            ax.bar(i+.19, np.mean(last), width=.34, facecolor="white", edgecolor=COLORS[label], hatch="//",
                   label="Final learned checkpoint" if label == "E4" else None)
            ax.scatter(np.linspace(i+.09, i+.29, len(last)), last, s=19, facecolors="white",
                       edgecolors="#172d41", linewidths=.75, zorder=3, clip_on=False)
    ax.set(xticks=range(len(ARMS)), xticklabels=ARMS, ylim=(0, 1.10), ylabel="Partition success rate")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.13), ncol=2, frameon=False)
    _save(plt, fig, directory, 6)
    timing, mesh = _example_timing(output, cfg, sample, arm, seed, example)
    histogram = None
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(14.5, 4.5), layout="constrained")
    fig.suptitle("Validation-selected checkpoints and rule baselines")
    for i, label in enumerate(ARMS):
        rows = best[label]
        ys = [r["mesh_score"] for r in rows if r["mesh_score"] is not None]
        if ys:
            ax.scatter([i]*len(ys), ys, s=27, color=COLORS[label])
        else:
            ax.text(i, .025, "n/a", transform=ax.get_xaxis_transform(), ha="center", fontsize=8)
        bx.bar(i, np.mean([r["mesh_successful"]/r["total"] for r in rows]), color=COLORS[label], alpha=.8)
    for axis in (ax, bx):
        axis.set(xticks=range(len(ARMS)), xticklabels=ARMS)
    ax.set_ylabel("Composite score, matched coarse budget")
    bx.set(ylabel="Valid coarse meshes / all cases", ylim=(0, 1.06))
    if mesh is None:
        cx.text(.5, .5, "Representative mesh unavailable", transform=cx.transAxes, ha="center")
    else:
        jacobians = np.concatenate([corner_scaled_jacobians(nodes).min(axis=-1).ravel()
                                   for nodes in mesh.blocks.values()])
        edges = np.linspace(min(0., float(jacobians.min())), max(1., float(jacobians.max())), 21)
        counts, edges = np.histogram(jacobians, bins=edges)
        histogram = {"edges": edges.tolist(), "counts": counts.tolist(), "cells": len(jacobians)}
        cx.stairs(100 * counts / len(jacobians), edges, fill=True, color=COLORS[arm], alpha=.75)
    cx.set(xlabel="Cell minimum scaled Jacobian", ylabel="Cells (%)", title=f"{arm} representative coarse mesh")
    _save(plt, fig, directory, 7)
    fig, ax = plt.subplots(figsize=(8.5, 4.4), layout="constrained")
    pairs = [("E7", "E6"), ("E7", "E5"), ("E8", "E7")]
    for i, (left, right) in enumerate(pairs):
        ys = [p["success_delta"] for p in summary["paired"] if
              (p["left"], p["right"], p["budget"]) == (left, right, 1536)]
        ax.scatter(np.linspace(i-.12, i+.12, len(ys)), ys, color=COLORS[left], zorder=3)
        ax.hlines(np.mean(ys), i-.22, i+.22, colors="#284a63", lw=2)
    ax.axhline(0, color="#7d8b95", lw=.8)
    ax.set(xticks=range(len(pairs)), xticklabels=[f"{a} − {b}" for a,b in pairs],
           ylabel="Paired change in partition success", ylim=(-1.03, 1.03),
           title="Validation-selected checkpoints")
    _save(plt, fig, directory, 8)
    fig, ax = plt.subplots(figsize=(8.4, 4), layout="constrained")
    if mesh is None:
        ax.text(.5, .5, "No valid mesh on this trajectory", transform=ax.transAxes, ha="center")
    else:
        for nodes in mesh.blocks.values():
            ax.plot(nodes[:,:,0], nodes[:,:,1], color="#426781", lw=.38)
            ax.plot(nodes[:,:,0].T, nodes[:,:,1].T, color="#426781", lw=.38)
        ax.set(xlim=(-.13, 1.20), ylim=(-.22, .22), xlabel="x / chord", ylabel="y / chord",
               title=f"{arm} seed {seed} near-wall mesh")
        ax.set_aspect("equal")
    _save(plt, fig, directory, 9)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    fields = ("rms[Rho]", "CL", "CD")
    for case in cfd["cases"]:
        path = output/case["directory"]/"rans_history.csv" if "directory" in case else None
        if path is None or not path.exists():
            continue
        rows = read_history(path)
        if not rows or not set(fields).issubset(rows[0]):
            continue
        label = f'{case["arm"]} {case["budget"]} {case["status"]}'
        for axis, field in zip(axes, fields):
            axis.plot([float(r[field]) for r in rows], label=label, lw=1.2)
    if axes[0].lines:
        axes[0].axhline(-9, color="#758495", ls="--", lw=.8, label="Residual target")
        axes[0].legend(fontsize=7)
    else:
        for axis in axes:
            axis.text(.5,.5,"RANS history unavailable", transform=axis.transAxes, ha="center")
    for axis, ylabel in zip(axes, ("log10 RMS density residual", "Lift coefficient CL", "Drag coefficient CD")):
        axis.set(xlabel="RANS iteration", ylabel=ylabel)
    _save(plt, fig, directory, 10)
    return {"example": sample["sample_id"], "arm": arm, "seed": seed, "frames": len(frames),
            "timing": timing, "coarse_mesh_jacobians": histogram}


def _cfd_rows(cfd):
    for case in cfd["cases"]:
        phases = case.get("phases", {})
        warm, rans = phases.get("warm", {}), phases.get("rans", {})
        residual = rans.get("last", {}).get("rms[Rho]")
        if residual is not None:
            residual = float(residual)
            if not np.isfinite(residual):
                residual = None
        forces = rans.get("forces", {})
        yield [case["arm"], case["seed"], case["budget"], case.get("mesh", {}).get("quads"), case["status"],
               warm.get("seconds"), rans.get("seconds"), rans.get("iterations"), residual,
               forces.get("CL"), forces.get("CD")]


def _evaluation_metrics(output, cfg):
    """Count UV topology legality separately from completion and physical meshing."""
    geometry = AirfoilDecompositionConfig(**cfg["geometry"])
    fields = ("quads", "min_scaled_jacobian", "q05_scaled_jacobian", "wall_angle_q10_deg",
              "max_aspect_ratio", "farfield_max_aspect_ratio", "interface_max_abs_difference")
    runs = []
    for arm, seed, tag in jobs(cfg):
        rows = _evaluation(output, arm, seed, tag)["rows"]
        topology_issues = Counter()
        checked = legal = invalid_completed = 0
        for row in rows:
            if "graph" not in row or "initial_graph" not in row:
                topology_issues["topology_unavailable"] += 1
                invalid_completed += int(row["success"])
                continue
            initial = BlockGraph.from_dict(row["initial_graph"])
            terminal = BlockGraph.from_dict(row["graph"])
            failure = validate_graph(terminal, initial, max_blocks=geometry.max_blocks,
                                     max_half_edges=geometry.max_half_edges)
            checked += 1
            if failure is None:
                legal += 1
            else:
                topology_issues[failure.reason] += 1
                invalid_completed += int(row["success"])
        completed = [row for row in rows if row["success"]]
        budgets = {}
        for budget in BUDGETS:
            meshes = [row["mesh"][str(budget)] for row in completed]
            matched = [mesh for mesh in meshes if mesh["valid"] and mesh["budget_match"]]
            budgets[str(budget)] = {"valid": sum(mesh["valid"] for mesh in meshes), "matched": len(matched),
                                    "distributions": {field: dict(zip(("min", "median", "max"),
                                                          np.quantile([m[field] for m in matched], [0, .5, 1]).tolist()))
                                                      for field in fields} if matched else {}}
        runs.append({"arm": arm, "seed": seed, "tag": tag, "cases": len(rows), "completed": len(completed),
                     "topology_checked": checked, "topology_legal": legal, "topology_issues": dict(topology_issues),
                     "completed_without_legal_topology": invalid_completed,
                     "completed_mean_edits": float(np.mean([r["edits"] for r in completed])) if completed else None,
                     "completed_mean_outer_blocks": float(np.mean([r["outer_blocks"] for r in completed])) if completed else None,
                     "budgets": budgets})
    efficiency = []
    for arm in SAC_ARMS:
        for seed in cfg["seeds"]:
            receipt = read_json(output/"arms"/f"{arm}_seed{seed}"/"completion.json")
            history = receipt["validation"]
            first_full = next((r["step"] for r in history if r["successful"] == r["total"]), None)
            integral = sum((b["step"]-a["step"])*(a["success_rate"]+b["success_rate"])/2
                           for a, b in zip(history, history[1:]))
            efficiency.append({"arm": arm, "seed": seed, "first_full_validation_step": first_full,
                               "best_step": receipt["best_step"], "online_steps": receipt["online_steps"],
                               "validation_mean_by_online_steps": integral/receipt["online_steps"]})
    return {"topology_scope": "UV graph constraints and preservation of the initial protected faces",
            "quality_scope": "valid meshes within 5% of each target cell count",
            "efficiency_scope": "trapezoidal mean validation success over the online interaction budget",
            "runs": runs, "sample_efficiency": efficiency}


def _workbook_case_rows(output, runs):
    for run in runs:
        for row in _evaluation(output, run["arm"], run["seed"], run["tag"])["rows"]:
            for budget in BUDGETS:
                mesh = row.get("mesh", {}).get(str(budget), {})
                yield [run["arm"], run["seed"], run["tag"], row["sample_id"], int(row["success"]), row["reason"],
                       row["edits"], budget, mesh.get("quads"), mesh.get("score"), mesh.get("min_scaled_jacobian"),
                       mesh.get("wall_angle_q10_deg"), mesh.get("farfield_max_aspect_ratio"),
                       int(mesh.get("valid", False)), int(mesh.get("budget_match", False)),
                       mesh.get("q05_scaled_jacobian"), mesh.get("max_aspect_ratio"),
                       mesh.get("interface_max_abs_difference"), row["seconds"] if budget == BUDGETS[0] else None]


def _workbook(output, directory, summary, cfd, metrics):
    runs = [[r[k] for k in ("arm","seed","tag","total","successful")] + [None] +
            [r["mesh_successful"], r["mesh_score"], r["mean_penalized_edits"], r["total_seconds"],
             t["topology_checked"], t["topology_legal"], t["completed_mean_edits"], t["completed_mean_outer_blocks"],
             t["budgets"][str(BUDGETS[1])]["valid"]] for r, t in zip(summary["runs"], metrics["runs"], strict=True)]
    pairs = [[p[k] for k in ("left","right","seed","budget","common_cases","both_partitioned","matched_meshes","success_delta")]
             + [p["mean_delta"].get("score"), p["mean_delta"].get("min_scaled_jacobian")] for p in summary["paired"]]
    cases = list(_workbook_case_rows(output, summary["runs"]))
    reference = _reference_rows(output)
    tables = [
        {"name":"分块汇总","id":"summary","title":"翼型自动分块实验汇总","note":"每行对应一个方法、种子及权重。拓扑合法数按 UV 图约束判定。",
         "headers":["方法","种子","权重","算例数","成功数","分块成功率","粗档有效网格数","粗档网格综合分","预算化步数","耗时（秒）",
                    "拓扑检查数","拓扑合法数","成功平均编辑数","成功平均外域块数","细档有效网格数"],
         "rows":runs,"widths":[12,12,14,13,13,17,19,19,16,17,16,16,19,22,20],
         "formats":[[5,"0.0%"],[7,"0.0000"],[8,"0.00"],[9,"0.0"],[12,"0.00"],[13,"0.00"]],
         "ratioColumn":5,"preview":"A1:O10"},
        {"name":"配对消融","id":"pairs","title":"同翼型与相近单元数的配对消融","note":"差值方向为左组减右组。质量差仅在双方完成且预算匹配时计算。",
         "headers":["左组","右组","种子","目标单元数","共同算例","共同完成","相近网格","成功率差","综合分差","最小雅可比差"],
         "rows":pairs,"widths":[12,12,12,16,16,16,16,15,16,19],"formats":[[7,"0.0%"],[8,"0.0000"],[9,"0.0000"]],"preview":"A1:J9"},
        {"name":"逐例数据","id":"cases","title":"翼型分块与网格逐例结果","note":"每例保留两档预算。总耗时含分块与两档制网，只在粗档行记一次。",
         "headers":["方法","种子","权重","翼型 ID","分块成功","终止原因","编辑数","目标单元","实际单元","网格综合分","最小雅可比","近壁角 Q10","远场最大长宽比",
                    "网格有效","预算匹配","雅可比 Q05","全域最大长宽比","块间接口最大差","总耗时（秒）"],
         "rows":cases,"widths":[11,11,14,31,14,43,12,15,15,15,16,17,23,15,15,16,23,20,17],
         "formats":[[9,"0.0000"],[10,"0.0000"],[11,"0.00"],[12,"0.0"],[15,"0.0000"],[16,"0.0"],[17,"0.00E+00"],[18,"0.00"]],
         "preview":"A1:S9"},
        {"name":"CFD","id":"cfd","title":"SU2 初步计算","note":"达到迭代上限的力系数为末步值。收敛状态采用残差和力系数联合判据。",
         "headers":["方法","种子","目标单元","实际单元","终止状态","预热秒数","RANS 秒数","RANS 迭代数","末步 log10 密度残差","末步 CL","末步 CD"],
         "rows":list(_cfd_rows(cfd)),"widths":[11,11,14,14,26,15,15,17,23,16,16],
         "formats":[[5,"0.00"],[6,"0.00"],[8,"0.0000"],[9,"0.00000"],[10,"0.00000"]],"preview":"A1:K12"},
        {"name":"模板调参参考","id":"template","title":"CH-SAC v2 固定 C-H 模板调参参考",
         "note":"五种子发布归档。有效网格数来自已有有效初始网格上的调参过程。",
         "headers":["测试集","单元档","最低有效数","样本数","CH-SAC 均值","默认综合分","局部搜索综合分"],
         "rows":reference,"widths":[20,16,16,16,18,19,22],
         "formats":[[4,"0.0000"],[5,"0.0000"],[6,"0.0000"]],"preview":"A1:G9"},
    ]
    write_json(directory/"workbook.json", {"sheets":tables}, replace=True)
    runtime = ROOT/".runtime/airfoil-fusion-artifact-runtime.json"
    setting = read_json(runtime) if runtime.exists() else {}
    node = os.environ.get("AEROBLOCK_FUSION_NODE", setting.get("node"))
    modules = os.environ.get("AEROBLOCK_FUSION_NODE_MODULES", setting.get("modules"))
    if bool(node) != bool(modules):
        raise RuntimeError("Set both AEROBLOCK_FUSION_NODE and AEROBLOCK_FUSION_NODE_MODULES for the optional Node exporter")
    if node:
        subprocess.run(
            [sys.executable, "-B", str(ROOT/"scripts/airfoil_fusion_workbook_package.py"), str(directory), node, modules],
            cwd=ROOT, check=True, timeout=600,
        )
    else:
        from .workbook import write_workbook

        write_workbook(directory/"实验结果.xlsx", tables)


def _reference_rows(output):
    record = read_json(output/"evaluation/E9_ch_sac_reference.json")
    return [[split, int(budget), row["valid_min"], row["total"], row["score_mean"]["mean"],
             record["baselines"]["B0_Default"][split][budget]["score_mean"],
             record["baselines"]["B2_LocalSearch"][split][budget]["score_mean"]]
            for split in ("test_uiuc", "test_naca24") for budget, row in record["mesh_quality"][split].items()]


def _connectivity_signature(graph):
    """A conservative signature invariant to face IDs and vertex coordinates."""
    neighbors = {fid: set() for fid in graph.faces}
    for owners in edge_incidence(graph).values():
        if len(owners) == 2:
            left, right = (face for face, _ in owners)
            neighbors[left].add(right)
            neighbors[right].add(left)
    return tuple(sorted((face.protected, len(neighbors[fid]),
                         sum(graph.faces[other].protected for other in neighbors[fid]))
                        for fid, face in graph.faces.items()))


def _topology_diversity(output, cfg):
    fixed = {row["sample_id"]: (row["graph"], fingerprint(BlockGraph.from_dict(row["graph"])))
             for row in _evaluation(output, "E1")["rows"] if row["success"]}
    report = []
    for arm in ("E4", "E4a", "E5", "E6", "E7", "E8"):
        seen = {}
        for seed in cfg["seeds"]:
            for row in _evaluation(output, arm, seed)["rows"]:
                sid = row["sample_id"]
                record = seen.setdefault(sid, [])
                if row["success"]:
                    graph = BlockGraph.from_dict(row["graph"])
                    topology = fingerprint(graph)
                    fixed_graph, fixed_topology = fixed.get(sid, (None, None))
                    record.append({"seed": seed, "connectivity": _connectivity_signature(graph),
                                   "topology": topology, "canonical_fixed_match": topology == fixed_topology,
                                   "exact_fixed_match": row["graph"] == fixed_graph})
        report.append({"arm": arm, "cases_with_success": sum(bool(r) for r in seen.values()),
                       "cases_with_different_connectivity_signatures": sum(
                           len({item["connectivity"] for item in r}) > 1 for r in seen.values()),
                       "cases_with_different_uv_partitions": sum(
                           len({item["topology"] for item in r}) > 1 for r in seen.values()),
                       "joint_success_with_fixed": sum(len(r) for sid, r in seen.items() if sid in fixed),
                       "canonical_graph_matches_with_fixed": sum(
                           item["canonical_fixed_match"] for r in seen.values() for item in r),
                       "exact_graph_matches_with_fixed": sum(
                           item["exact_fixed_match"] for r in seen.values() for item in r),
                       "cases": [{"id": sid, "successful_seeds": [item["seed"] for item in r],
                                  "fixed_success": sid in fixed,
                                  "canonical_fixed_match_seeds": [item["seed"] for item in r if item["canonical_fixed_match"]],
                                  "exact_fixed_match_seeds": [item["seed"] for item in r if item["exact_fixed_match"]],
                                  "distinct_uv_partitions": len({item["topology"] for item in r}),
                                  "distinct_signatures_lower_bound": len({item["connectivity"] for item in r})}
                                 for sid, r in seen.items()]})
    return {"scope": "same geometry across seeds, validation-selected weights and fixed rule",
            "signature": "multiset of protection flag, adjacent face count and protected neighbor count",
            "canonical_uv": "compare protected flags and UV face cycles independently of graph IDs and cycle start",
            "limitations": "equal signatures do not imply graph isomorphism", "methods": report}


def _interaction_diagnostics(path):
    step_seconds, guard_seconds = [], []
    guard_totals, prune_reasons, terminal_reasons = Counter(), Counter(), Counter()
    mesh_rewards, mesh_floor = 0, 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            step_seconds.append(row["seconds"])
            if row["terminated"] or row["truncated"]:
                terminal_reasons[row["reason"]] += 1
            mesh = (row.get("reward_components") or {}).get("mesh_quality")
            if mesh is not None:
                mesh_rewards += 1
                mesh_floor += int(mesh == -1)
            guard = row["guard"]
            if guard is not None:
                metrics = guard["metrics"]
                guard_seconds.append(metrics["elapsed_seconds"])
                guard_totals.update({key: int(metrics[key]) for key in (
                    "root_previews", "second_layer_transitions", "geometry_validation_calls",
                    "removed_actions", "unknown_actions", "completion_witnesses", "guard_cache_hit"
                )})
                prune_reasons.update(item["reason"] for item in guard["actions"] if not item["kept"])

    def percentiles(values):
        return {label: float(np.quantile(values, q)) if values else None
                for label, q in (("p50", .5), ("p90", .9))}

    return {"online_steps": len(step_seconds), "step_seconds": percentiles(step_seconds),
            "guard_observations": len(guard_seconds), "guard_seconds": percentiles(guard_seconds),
            "guard_totals": dict(guard_totals), "prune_reasons": dict(prune_reasons),
            "terminal_reasons": dict(terminal_reasons), "mesh_reward_count": mesh_rewards,
            "mesh_floor_count": mesh_floor}


def _first_candidates(cases):
    return {row["sample_id"]: row["actions"][0]["candidate"] for row in cases if row.get("actions")}


def _validation_diagnostics(result, initial_candidates):
    cases = result["per_case"]
    first = _first_candidates(cases)
    comparable = first.keys() & initial_candidates.keys()
    changed = [sid for sid in comparable if first[sid] != initial_candidates[sid]]
    actions = Counter(action["candidate"]["kind"] for row in cases for action in row.get("actions", ()))
    failures = Counter(row["reason"] for row in cases if not row["success"])
    return {"step": result["step"], "successful": sum(row["success"] for row in cases),
            "total": len(cases), "first_actions": dict(Counter(item["kind"] for item in first.values())),
            "first_candidate_comparison": {"comparable": len(comparable), "changed": len(changed),
                                           "changed_with_same_kind": sum(first[sid]["kind"] == initial_candidates[sid]["kind"]
                                                                         for sid in changed)},
            "actions": dict(actions),
            "failure_reasons": dict(failures), "network_sha256": result.get("network_sha256"),
            "median_edits": float(np.median([row["edits"] for row in cases]))}


def _matching_transition_prefix(left, right):
    fields = ("step", "sample_id", "observation", "next_observation", "candidate",
              "reward", "effective_actions", "terminated", "truncated", "reason")
    matched = 0
    with left.open(encoding="utf-8") as first, right.open(encoding="utf-8") as second:
        for line_a, line_b in zip(first, second):
            a, b = json.loads(line_a), json.loads(line_b)
            if any(a[key] != b[key] for key in fields):
                break
            matched += 1
    return matched


def _teacher_coverage(output):
    manifest = read_json(output/"teachers/manifest.json")
    rows = manifest["rows"]
    selected = [row for row in rows if row["selected"] is not None]
    methods = Counter(row["selected"]["method"] for row in selected)
    attempts = Counter(attempt["method"] for row in rows for attempt in row["attempts"])
    complete = Counter(attempt["method"] for row in rows for attempt in row["attempts"] if attempt["success"])
    other_graphs = Counter(attempt["method"] for row in selected for attempt in row["attempts"]
                           if attempt["success"] and attempt["graph_sha256"] != row["selected"]["graph_sha256"])
    return {"candidate_cases": len(rows), "selected_cases": len(selected),
            "candidate_families": len({row["group_id"] for row in rows}),
            "selected_families": len({row["group_id"] for row in selected}),
            "selected_methods": dict(methods),
            "attempts_by_method": [{"method": method, "attempts": total, "successful": complete[method],
                                    "selected": methods[method], "different_graph_from_selected": other_graphs[method]}
                                   for method, total in attempts.items()],
            "selected_actions": sum(len(row["selected"]["actions"]) for row in selected),
            "historical_replays": read_json(output/"teachers/historical_audit.json")["count"]}


def _training_diagnostics(output, cfg, directory):
    records = []
    for arm in ("E6", "E7", "E8"):
        for seed in cfg["seeds"]:
            run = output/"arms"/f"{arm}_seed{seed}"
            row = _interaction_diagnostics(run/"interactions.jsonl")
            history = read_json(run/"completion.json")["validation"]
            validation = []
            for item in history:
                result = read_json(run/f"validation_{item['step']:06d}"/"report.json")
                if not validation:
                    initial_candidates = _first_candidates(result["per_case"])
                validation.append(_validation_diagnostics(result, initial_candidates))
            records.append({"arm": arm, "seed": seed, **row, "validation_actions": validation})
    matching = []
    for seed in cfg["seeds"]:
        arms = {row["arm"]: row for row in records if row["seed"] == seed}
        left, right = (output/"arms"/f"{arm}_seed{seed}"/"interactions.jsonl" for arm in ("E7", "E8"))
        prefix = _matching_transition_prefix(left, right)
        by_step = {item["step"]: item["network_sha256"] for item in arms["E7"]["validation_actions"]}
        same = [item["step"] for item in arms["E8"]["validation_actions"]
                if item["network_sha256"] is not None and item["network_sha256"] == by_step.get(item["step"])]
        matching.append({"seed": seed, "matched_online_prefix_steps": prefix, "checkpoints_with_equal_networks": same})
    teachers = _teacher_coverage(output)
    write_json(directory/"training_diagnostics.json", {"per_seed": records, "E7_E8_training_match": matching,
               "teacher_coverage": teachers,
               "timing": "Step wall time includes environment transitions. Guard time belongs to the mask of the logged pre-action observation.",
               "mesh_floor": "The count includes all terminal mesh feedback at -1. It does not classify the mesh as valid or invalid."},
               replace=True)
    return records, matching, teachers


def _supplemental_cfd(output, cfg, summary):
    """Use the published sharp analytic NACA0012, only after the test lock."""
    if summary.get("status") != "naca0012_not_eligible_in_frozen_holdout":
        return summary
    origin = "reference/naca24/samples.jsonl"
    source = published_bytes(origin)
    sample = next(json.loads(line) for line in source.splitlines() if b'"analytic_naca_0012"' in line)
    sample.update(explicitly_closed=True, group_id="supplemental_analytic_naca", split="cfd_only")
    matrix = read_json(output/"training_matrix.json")
    lock = read_json(output/"evaluation_lock.json")
    if lock["training_sha256"] != file_sha(output/"training_matrix.json"):
        raise ValueError("Training matrix changed after the evaluation lock")
    plans = {"sample": sample, "commit": B_COMMIT, "source": origin,
             "source_sha256": sha256(source).hexdigest(),
             "evaluation_lock_sha256": file_sha(output/"evaluation_lock.json"),
             "scope": "analytic NACA0012 CFD-only after all training and testing selections"}
    directory = output/"cfd"
    write_json(directory/"analytic_naca0012.json", plans, replace=True)
    records = []
    for arm in ("E1", "E7", "E8"):
        seed, agent, method = 42, None, "fixed"
        if arm != "E1":
            rankings = []
            for s in cfg["seeds"]:
                history = read_json(output/"arms"/f"{arm}_seed{s}"/"completion.json")["validation"]
                rankings.append((selection_key(max(history, key=selection_key)), s))
            seed = sorted(rankings)[len(rankings)//2][1]
            checkpoint = f"arms/{arm}_seed{seed}/best.pt"
            if file_sha(output/checkpoint) != matrix["checkpoints"][checkpoint]:
                raise ValueError("Supplemental CFD policy differs from locked weights")
            agent, _ = get_inference_agent(output/checkpoint, arm, seed, file_sha(output/"prepared.json"))
            method = "learned"
        result, _ = episode(Environments(cfg["geometry"], arm), sample, method, seed=seed, agent=agent, mesh=False, trace=True)
        write_json(directory/f"analytic_{arm}_{seed}_partition.json", result, replace=True)
        for budget in BUDGETS:
            case = directory/f"analytic_{arm}_{seed}_{budget}"
            case.mkdir(exist_ok=True)
            receipt = case/"result.json"
            if receipt.exists():
                records.append(read_json(receipt))
                continue
            if result["success"]:
                attempts = list(case.glob("attempt_*"))
                attempt = case/f"attempt_{len(attempts)+1:03d}"
                attempt.mkdir()
                row = _simulate(attempt, sample, BlockGraph.from_dict(result["graph"]), budget, cfg["geometry"])
                row["directory"] = attempt.relative_to(output).as_posix()
            else:
                row = {"status":"partition_failed","reason":result["reason"],"converged":False}
            row.update(arm=arm, seed=seed, budget=budget, sample_id=sample["sample_id"])
            write_json(receipt, row, replace=True)
            records.append(row)
    summary.update(status="analytical_naca0012_post_lock", sample_id=sample["sample_id"], cases=records,
                   scope="Published analytical sharp NACA0012 used as a CFD-only geometry after the test lock")
    write_json(directory/"summary.json", summary, replace=True)
    return summary


def _execution_timing(output, cfg):
    handoff_path = output / "accelerated_handoff.json"
    if not handoff_path.is_file():
        return None
    handoff = read_json(handoff_path)
    rows = []
    for arm in (*SAC_ARMS, "E5"):
        for seed in cfg["seeds"]:
            directory = output / "arms" / f"{arm}_seed{seed}"
            completion = read_json(directory / "completion.json")
            for validation in completion["validation"]:
                step = validation["step"]
                path = directory / f"validation_{step:06d}" / "report.json"
                detail = read_json(path)
                if "validation_runtime" in detail:
                    rows.append({"arm": arm, "seed": seed, "step": step, "cases": validation["total"],
                                 "report_sha256": file_sha(path), **detail["validation_runtime"]})
    return {"handoff_sha256": file_sha(handoff_path), "activation_gate_sha256": handoff["acceleration_gates"]["activation_gate"],
            "geometry_memo_replay_steps": handoff["online_interactions_replayed"], "accelerated_validations": rows}


def build_report(output, cfg):
    directory = output/"report"
    directory.mkdir(exist_ok=True)
    summary = read_json(output/"evaluation/summary.json")
    cfd = _supplemental_cfd(output, cfg, read_json(output/"cfd/summary.json"))
    matrix = read_json(output/"training_matrix.json")
    metrics = _evaluation_metrics(output, cfg)
    write_json(directory/"evaluation_metrics.json", metrics, replace=True)
    figure = _figures(output, directory, cfg, summary, cfd)
    diversity = _topology_diversity(output, cfg)
    write_json(directory/"topology_diversity.json", diversity, replace=True)
    diagnostics, matching, teachers = _training_diagnostics(output, cfg, directory)
    cohort = "冻结 NACA 留出集" if cfg["profile"] == "formal" else "开发验证集"
    text = ["# AeroBlockMesh-SCU 翼型自动分块实验", "",
            f"本报告使用{cohort}，包含 {summary['data']['eligible']} 个尖尾缘翼型。"
            f"每次 SAC 训练完成 {matrix['online_steps']} 次在线交互。", "",
            "| 方法 | 各种子成功数 / 完整分母 | 平均成功率 | 种子标准差 |",
            "|---|---|---:|---:|"]
    for arm in ARMS:
        rows = [r for r in summary["runs"] if r["arm"] == arm and r["tag"] != "final"]
        rates = [r["success_rate"] for r in rows]
        counts = "、".join(f"{r['successful']}/{r['total']}" for r in rows)
        sd = f"{np.std(rates,ddof=1)*100:.2f} 个百分点" if len(rates)>1 else "单次运行"
        text.append(f"| {arm} | {counts} | {np.mean(rates):.1%} | {sd} |")
    text.extend(["", "## 拓扑合法性与制网覆盖", "",
                 "按 UV 图约束检查终图及保护块。未完成分块时，最后一张图也可能通过拓扑合法性检查。"
                 "拓扑合法率和两档有效网格率均以全部评估算例为分母。",
                 "", "| 方法 | 合法终图 / 全部 | 检查过的终图 | 成功例平均编辑数 | 成功例平均外域块数 | "
                 "1,536 档有效网格 | 5,120 档有效网格 |", "|---|---:|---:|---:|---:|---:|---:|"])
    for arm in ARMS:
        rows = [r for r in metrics["runs"] if r["arm"] == arm and r["tag"] != "final"]
        completed = [r for r in rows if r["completed"]]
        count = sum(r["completed"] for r in completed)
        means = [f"{sum(r[key]*r['completed'] for r in completed)/count:.2f}" if count else "—"
                 for key in ("completed_mean_edits", "completed_mean_outer_blocks")]
        cases = sum(r["cases"] for r in rows)
        valid = [sum(r["budgets"][str(budget)]["valid"] for r in rows) for budget in BUDGETS]
        text.append(f"| {arm} | {sum(r['topology_legal'] for r in rows)}/{cases} | "
                    f"{sum(r['topology_checked'] for r in rows)} | {means[0]} | {means[1]} | {valid[0]}/{cases} | {valid[1]}/{cases} |")
    text.extend(["", "逐种子记录包含两档预算内有效网格的实际单元数、缩放雅可比、近壁角、长宽比和接口偏差分布。"
                 "图 7 展示代表性粗档网格的逐单元缩放雅可比直方图。", "", "## 在线样本效率", "",
                 "曲线均值由开发集成功率对在线交互步数作梯形积分后除以总步数。第 0 步包含已有的初始化效果。",
                 "", "| 方法 | 达到全量验证成功的种子 | 首次达到的在线步数（各种子） | 种子平均曲线均值 |",
                 "|---|---:|---|---:|"])
    for arm in SAC_ARMS:
        runs = [r for r in metrics["sample_efficiency"] if r["arm"] == arm]
        first = [r["first_full_validation_step"] for r in runs]
        labels = "、".join(str(step) if step is not None else "—" for step in first)
        text.append(f"| {arm} | {sum(step is not None for step in first)}/{len(first)} | {labels} | "
                    f"{np.mean([r['validation_mean_by_online_steps'] for r in runs]):.1%} |")
    text.extend(["", "[逐种子拓扑、制网与样本效率指标](evaluation_metrics.json)"])
    text.extend(["", "## 配对与训练退化", "",
                 "| 比较 | 目标单元 | 种子平均成功率差 | 匹配网格对数 | 种子平均综合分差 |", "|---|---:|---:|---:|---:|"])
    for left,right in sorted({(r["left"],r["right"]) for r in summary["paired"]}):
        for budget in BUDGETS:
            rows = [r for r in summary["paired"] if (r["left"],r["right"],r["budget"]) == (left,right,budget)]
            valid = [r["mean_delta"]["score"] for r in rows if r["matched_meshes"]]
            quality = f"{np.mean(valid):+.4f}" if valid else "无符合条件的配对"
            text.append(f"| {left} − {right} | {budget:,} | {np.mean([r['success_delta'] for r in rows])*100:+.2f} 个百分点 | "
                        f"{sum(r['matched_meshes'] for r in rows)} | {quality} |")
    seed_labels = "、".join(str(seed) for seed in cfg["seeds"])
    text.extend(["", f"下表按种子 {seed_labels} 的顺序列出成功数，从最佳权重到末轮权重。"
                      f"每份权重评估 {summary['data']['eligible']} 个翼型。", "",
                 "| 训练组 | 各种子成功数变化 | 最佳权重减末轮权重的平均成功率 |", "|---|---|---:|"])
    for arm in SAC_ARMS:
        by_seed = {(r["seed"], r["tag"]): r for r in summary["runs"] if r["arm"] == arm}
        pairs = [(by_seed[seed, "best"], by_seed[seed, "final"]) for seed in cfg["seeds"]]
        success_counts = "、".join(f"{best['successful']}→{final['successful']}" for best, final in pairs)
        delta = np.mean([best["success_rate"] - final["success_rate"] for best, final in pairs])
        text.append(f"| {arm} | {success_counts} | {delta*100:+.2f} 个百分点 |")
    text.extend(["", "## 行为克隆示范覆盖", "",
                 f"原历史示范重放并核验了 {teachers['historical_replays']} 条轨迹。"
                 f"本轮从 {teachers['candidate_cases']} 个训练翼型选出 {teachers['selected_cases']} 条示范，"
                 f"覆盖 {teachers['selected_families']}/{teachers['candidate_families']} 个训练家族组，"
                 f"包含 {teachers['selected_actions']} 个决策。",
                 "", "| 教师方法 | 尝试 | 成功 | 入选 | 成功且终图哈希与入选示范不同 |", "|---|---:|---:|---:|---:|"])
    text.extend(f"| {row['method']} | {row['attempts']} | {row['successful']} | {row['selected']} | "
                f"{row['different_graph_from_selected']} |" for row in teachers['attempts_by_method'])
    text.extend(["", "## 同一外形上的终图比较", "",
                 "| 方法 | 至少一个种子成功的翼型 | 种子间 UV 分块不同的翼型 | 种子间连接特征不同的翼型 | 与 E1 规范化分块相同 / 共同成功 |",
                 "|---|---:|---:|---:|---:|"])
    for row in diversity["methods"]:
        variation = row['cases_with_different_connectivity_signatures'] if len(cfg['seeds']) > 1 else "单种子"
        uv_variation = row['cases_with_different_uv_partitions'] if len(cfg['seeds']) > 1 else "单种子"
        text.append(f"| {row['arm']} | {row['cases_with_success']} | {uv_variation} | {variation} | "
                    f"{row['canonical_graph_matches_with_fixed']}/{row['joint_success_with_fixed']} |")
    text.extend(["", "连接特征仅使用保护块和邻接关系，给出可证实不同的终图数量下界。"
                 "规范化分块对比 UV 面环和保护标记，忽略顶点与面的编号以及面环起点。"
                 "逐翼型记录同时保留原始终图的逐字段匹配。", "", "[逐翼型连接记录](topology_diversity.json)"])
    reward_probe = output/"quality_reward_probe.json"
    if reward_probe.is_file():
        probe = read_json(reward_probe)
        lo, hi = probe["score_range"]
        text.extend(["", "## 训练奖励诊断", "",
                     f"用历史训练规则终图重建的 {probe['meshes_valid']}/{probe['count']} 个低分辨率网格有效。"
                     f"原始综合分为 {lo:.3f} 至 {hi:.3f}，其中 {probe['reward_at_floor']} 个在训练奖励中触及 −1 下限。"
                     "该低分辨率奖励在这些示范间没有区分度。"
                     "正式网格报告保留未截断的指标。", "", "[奖励诊断逐例数据](../quality_reward_probe.json)"])
    text.extend(["", "## 在线训练与筛选开销", "",
                 "| 方法 | 单步耗时 P50（秒） | 单步耗时 P90（秒） | 触及 −1 的网格奖励次数 / 总次数 |",
                 "|---|---:|---:|---:|"])
    for arm in ("E6", "E7", "E8"):
        rows = [row for row in diagnostics if row["arm"] == arm]
        med, slow = (np.median([row["step_seconds"][q] for row in rows]) for q in ("p50", "p90"))
        floor, total = (sum(row[key] for row in rows) for key in ("mesh_floor_count", "mesh_reward_count"))
        text.append(f"| {arm} | {med:.2f} | {slow:.2f} | {floor}/{total} |")
    text.append("")
    guarded_rows = [row for row in diagnostics if row["arm"] == "E8"]
    removed, unknown, witnessed = (sum(row["guard_totals"].get(key, 0) for row in guarded_rows)
                                  for key in ("removed_actions", "unknown_actions", "completion_witnesses"))
    text.extend(["单步耗时取各训练种子对应分位数的中位数。",
                 f"E8 的在线日志记录了 {removed} 个有排除证明的候选、{unknown} 个保留的未决候选和 {witnessed} 个可完成见证。",
                 "网格奖励触及下限的比例反映奖励区分度。网格有效性见逐例评估。",
                 "[逐种子开销、筛选理由和终止原因](training_diagnostics.json)"])
    execution = _execution_timing(output, cfg)
    if execution is not None:
        write_json(directory / "execution_timing.json", execution, replace=True)
        runs = execution["accelerated_validations"]
        reused = sum(row["reused_trajectories"] for row in runs)
        cases = sum(row["cases"] for row in runs)
        text.extend(["", "## 验证提速与续跑", "",
                     f"提速入口实际执行 {len(runs)} 次开发验证，重新计算策略动作并复用 {reused}/{cases} 条轨迹。"
                     f"E8 几何缓存启用前核验了 {execution['geometry_memo_replay_steps']} 步在线回放。"
                     "原始检查点、随机状态和延长预算规则沿用同一冻结实验。",
                     "[加速时段逐检查点计时](execution_timing.json) · [切换凭证](../accelerated_handoff.json)"])
    if (output/"evaluation_case_resume_handoff.json").is_file():
        text.extend(["", "正式评估逐翼型保存与测试锁绑定的结果。任务中断后，已保存的翼型通过校验后可复用。",
                     "[逐翼型续跑验证与启用凭证](../evaluation_case_resume_handoff.json)"])
    if (output/"validation_case_resume_handoff.json").is_file():
        resumed = sum(row.get("resumed_case_receipts", 0) for row in (execution or {}).get("accelerated_validations", ()))
        text.extend(["", f"单进程开发验证可按检查点权重和翼型身份复用已完成的算例。正式运行累计复用了 {resumed} 条中断前记录。",
                     "[开发验证续跑验证与启用凭证](../validation_case_resume_handoff.json)"])
    cache_handoff = output/"validation_cache_compression_handoff.json"
    if cache_handoff.is_file():
        trial = read_json(cache_handoff)["archive_gate"]
        text.extend(["", f"在两个归档开发翼型上核验了 {trial['observation_hashes_verified']} 条观测。"
                     f"无损压缩使缓存占用从 {trial['original_observation_bytes']:,} 降到 {trial['compressed_observation_bytes']:,} 字节。"
                     f"恢复检查点时复用了 {trial['compressed_trajectory_reuses']} 条完整轨迹。",
                     "[缓存压缩的逐例验证与启用凭证](../validation_cache_compression_handoff.json)"])
    text.extend(["", "### BC 初始化后的首次验证下降与恢复", "",
                 "| 方法 | 种子 | 首次下降步数 | 成功数变化 | 首次恢复步数 | 首候选改变 / 可比翼型 | 其中同类改变 | 首动作插点数变化 | 下降时插点数 / 总动作 |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    drops = []
    for row in diagnostics:
        if row["arm"] not in ("E7", "E8"):
            continue
        initial, *later = row["validation_actions"]
        drop = next((item for item in later if item["successful"] < initial["successful"]), None)
        if drop is not None:
            recovered = next((item["step"] for item in later if item["step"] > drop["step"]
                              and item["successful"] >= initial["successful"]), None)
            first_before, first_after = (item["first_actions"].get("insert_edge_point", 0) for item in (initial, drop))
            comparison = drop["first_candidate_comparison"]
            drops.append(f"| {row['arm']} | {row['seed']} | {drop['step']} | "
                         f"{initial['successful']}/{initial['total']} → {drop['successful']}/{drop['total']} | "
                         f"{recovered if recovered is not None else '—'} | "
                         f"{comparison['changed']}/{comparison['comparable']} | {comparison['changed_with_same_kind']} | "
                         f"{first_before} → {first_after} | "
                         f"{drop['actions'].get('insert_edge_point', 0)}/{sum(drop['actions'].values())} |")
    if drops:
        text.extend(drops)
    else:
        text.append("| E7/E8 | — | — | 未观察到低于初始化检查点 | — | — | — | — | — |")
    text.append("首次恢复指下降后开发集成功数首次达到初始化水平。首候选按同一翼型的完整候选描述比较。"
                "同类改变统计动作类型相同、具体候选发生变化的翼型。动作统计本身无法确定训练退化的原因。")
    matches = []
    for item in matching:
        steps = [step for step in item["checkpoints_with_equal_networks"]
                 if 0 < step <= item["matched_online_prefix_steps"]]
        if steps:
            matches.append((item["seed"], max(steps)))
    if matches:
        labels = "、".join(f"种子 {seed} 第 {step} 步" for seed, step in matches)
        text.append(f"E7 和 E8 在{labels}之前的在线转移记录一致，各自的检查点参数哈希也一致。")
    reasons = Counter()
    for run in summary["runs"]:
        reasons.update(run["failures"])
    text.extend(["", "## 失败与 CFD", ""])
    text.extend(f"- {key} 共 {value} 次。" for key,value in reasons.most_common())
    text.extend(["", "### SU2 求解", ""])
    if cfd["cases"]:
        text.extend(["求解条件为马赫数 0.15、雷诺数 600 万、攻角 10°。",
                     "RANS 联合收敛要求至少 100 次迭代。末 50 次的 log10 密度残差均须不高于 −9，"
                     "CL 和 CD 在同一窗口内的极差均须不超过 10⁻⁵ + 10⁻⁴ × |窗口均值|。"
                     "表中的力系数为末步实测值。", "",
                     "| 方法 | 种子 | 目标单元 | 实际单元 | 状态 | 预热秒数 | RANS 秒数 | RANS 迭代数 | 末步 log10 密度残差 | 末步 CL | 末步 CD |",
                     "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"])
        for row in _cfd_rows(cfd):
            cells = ["—" if value is None else f"{value:.5g}" if isinstance(value, float) else str(value)
                     for value in row]
            text.append("| " + " | ".join(cells) + " |")
    else:
        text.append(f"- {cfd.get('status','没有可用算例')}。")
    if cfd.get("status") == "analytical_naca0012_post_lock":
        text.append("- CFD 使用发布归档的解析尖尾缘 NACA0012。该算例在权重与测试冻结后单独执行。")
    if figure["timing"]:
        text.extend(["", "## 代表翼型分阶段计时", "",
                     f"在 `{figure['example']}` 上复跑成功轨迹，并核对终图和网格质量记录。"
                     "分块耗时含几何初始化。制网耗时含单元配点及质量计算。每行来自一次复跑。", "",
                     "| 方法 | 种子 | 分块秒数 | 1,536 单元制网秒数 | 5,120 单元制网秒数 |", "|---|---:|---:|---:|---:|"])
        for row in figure["timing"]:
            coarse, fine = row["meshes"]
            text.append(f"| {row['arm']} | {row['seed']} | {row['partition_seconds']:.3f} | "
                        f"{coarse['seconds']:.3f} | {fine['seconds']:.3f} |")
    text.extend(["", "## CH-SAC v2 模板调参参考", "",
                 "| 测试集 | 档位 | 归档有效数 | CH-SAC 均值 | 默认综合分 | 局部搜索综合分 |",
                 "|---|---:|---:|---:|---:|---:|"])
    for split,budget,valid,total,learned,default,local in _reference_rows(output):
        text.append(f"| {split} | {budget} | {valid}/{total} | {learned:.4f} | {default:.4f} | {local:.4f} |")
    text.extend(["", "## 结论与后续研究", "", "### 已实现功能", "",
                 "融合环境从含待分解多边形的 C 型域出发，支持全局切分、共享边插点与显式 STOP。"
                 "训练入口支持从零训练、由行为克隆初始化及候选筛选。"
                 "评估入口保存逐步动作、完整终图、两档 TFI 网格指标和 SU2 求解记录。",
                 "", "### 已验证结果", ""])
    reachability_path = output/"reachability.json"
    if reachability_path.is_file():
        reachability = read_json(reachability_path)
        if reachability["passed"]:
            text.extend([f"可达性重放在 {len(reachability['examples'])} 个训练翼型上各验证了至少两种合法分块。", ""])
    text.extend(["分块结果按种子与翼型计数。同一翼型会在不同种子下重复评估。", ""])
    for arm in ("E6", "E7", "E8"):
        results = [r for r in summary["runs"] if r["arm"] == arm]
        counts = {tag: (sum(r["successful"] for r in results if r["tag"] == tag),
                        sum(r["total"] for r in results if r["tag"] == tag))
                  for tag in ("best", "final")}
        best, final = counts["best"], counts["final"]
        topology = next(row for row in diversity["methods"] if row["arm"] == arm)
        text.append(f"- {arm} 在{cohort}上的最佳权重合计完成 {best[0]}/{best[1]} 次分块，"
                    f"末轮权重完成 {final[0]}/{final[1]} 次。最佳权重与 E1 共同成功 "
                    f"{topology['joint_success_with_fixed']} 次，其中 "
                    f"{topology['canonical_graph_matches_with_fixed']} 次的规范化分块相同。")
    if cfd["cases"]:
        converged = sum(case["converged"] for case in cfd["cases"])
        text.append(f"- SU2 汇总中 {converged}/{len(cfd['cases'])} 项达到联合收敛标准。"
                    "上表保存了实际迭代数、残差、末步力系数与求解耗时。")
    text.extend(["", "### 尚待解决", "",
                 f"本报告的分块结果来自{cohort}中的尖尾缘合格翼型。"
                 "跨家族泛化和力系数精度仍需独立验证。逐例耗时采用实测墙钟时间。",
                 "", "### 后续验证", "",
                 "在独立的其他翼型家族上复核分块成功率、拓扑多样性和合法性。"
                 "使用几何与工况匹配的参考解及多档网格检查 CFD 收敛和力系数精度。",
                 "", "## 图表与证据", ""])
    text.extend(f"- [{name}]({name}.pdf)" for name in FIGURES)
    text.extend(["- [配对统计](../evaluation/summary.json)", "- [CH-SAC v2 参考](../evaluation/E9_ch_sac_reference.json)",
                 "- [CFD 汇总](../cfd/summary.json)", "- [结果工作簿](实验结果.xlsx)"])
    (directory/"report.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    _workbook(output, directory, summary, cfd, metrics)
    write_json(directory/"figure_evidence.json", figure, replace=True)
    return {"report": str(directory/"report.md"), **figure}


def _case_cell_value(cell, strings, prefix):
    if cell is None:
        return None
    kind = cell.get("t")
    value = cell.findtext(prefix+"v")
    if kind == "inlineStr":
        return "".join(text.text or "" for text in cell.iter(prefix+"t"))
    if value is None:
        return None
    if kind == "s":
        return strings[int(value)]
    return float(value) if kind in (None, "n") else value


def _audit_case_sheet(bundle, member, expected_count, expected_rows, prefix):
    strings = []
    if "xl/sharedStrings.xml" in bundle.namelist():
        shared = ElementTree.fromstring(bundle.read("xl/sharedStrings.xml"))
        strings = ["".join(text.text or "" for text in item.iter(prefix+"t")) for item in shared]
    issues = []
    count = 0
    first_mismatch = None
    formula_error = False
    parent = None
    with bundle.open(member) as stream:
        for event, row in ElementTree.iterparse(stream, events=("start", "end")):
            if event == "start" and row.tag == prefix+"sheetData":
                parent = row
            if event != "end" or row.tag != prefix+"row":
                continue
            if parent is None:
                return ["XLSX case rows are outside the worksheet data"]
            formula_error |= any(cell.get("t") == "e" for cell in row)
            number = int(row.attrib["r"])
            if number >= 5:
                count += 1
                if expected_rows is not None:
                    expected = next(expected_rows, None)
                    if first_mismatch is None and (expected is None or number != count+4):
                        first_mismatch = f"A{number}"
                    if first_mismatch is None:
                        cells = {cell.attrib["r"]: cell for cell in row.iter(prefix+"c")}
                        for index, value in enumerate(expected):
                            address = f"{chr(ord('A')+index)}{number}"
                            actual = _case_cell_value(cells.get(address), strings, prefix)
                            if isinstance(actual, float) and isinstance(value, float):
                                matches = isclose(actual, value, rel_tol=1e-12, abs_tol=0.)
                            else:
                                matches = actual == value
                            if not matches:
                                first_mismatch = address
                                break
            row.clear()
            parent.remove(row)
    if count != expected_count:
        issues.append("XLSX case table does not cover both budgets for every evaluation row")
    if expected_rows is not None and next(expected_rows, None) is not None:
        issues.append("XLSX case table omits sealed records")
    if formula_error:
        issues.append("XLSX formula or cell error in 逐例数据")
    if first_mismatch is not None:
        issues.append(f"XLSX case cell {first_mismatch} differs from the sealed evaluation")
    return issues


def _audit_workbook(path, evaluation_runs, case_count, metrics=None, expected_cases=None):
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    expected_names = ["分块汇总", "配对消融", "逐例数据", "CFD", "模板调参参考"]
    issues = []
    try:
        with zipfile.ZipFile(path) as stream:
            if stream.testzip() is not None:
                return ["XLSX integrity"]
            workbook = ElementTree.fromstring(stream.read("xl/workbook.xml"))
            names = [sheet.attrib["name"] for sheet in workbook.findall(".//m:sheet", ns)]
            if names != expected_names:
                return ["XLSX sheet names or order changed"]
            for index, name in enumerate(names, 1):
                member = f"xl/worksheets/sheet{index}.xml"
                if name == "逐例数据":
                    issues.extend(_audit_case_sheet(stream, member, case_count*len(BUDGETS), expected_cases,
                                                    "{"+ns["m"]+"}"))
                    continue
                sheet = ElementTree.fromstring(stream.read(member))
                if sheet.findall(".//m:c[@t='e']", ns):
                    issues.append(f"XLSX formula or cell error in {name}")
                if name == "分块汇总":
                    cells = {cell.attrib["r"]: cell for cell in sheet.findall(".//m:c", ns)}
                    formulas = {address: cell for address, cell in cells.items() if cell.find("m:f", ns) is not None}
                    if len(formulas) != len(evaluation_runs):
                        issues.append("XLSX summary ratio count differs from evaluated runs")
                    for number, run in enumerate(evaluation_runs, 5):
                        ratio = formulas.get(f"F{number}")
                        if (ratio is None or ratio.findtext("m:f", namespaces=ns) != f"E{number}/D{number}"
                                or not np.isclose(float(ratio.findtext("m:v", namespaces=ns)), run["success_rate"],
                                                  atol=1e-12, rtol=0.)):
                            issues.append(f"XLSX row {number} success ratio does not match evaluation")
                    if metrics is not None:
                        for number, run in enumerate(metrics["runs"], 5):
                            expected = (run["topology_checked"], run["topology_legal"], run["completed_mean_edits"],
                                        run["completed_mean_outer_blocks"], run["budgets"][str(BUDGETS[-1])]["valid"])
                            for column, value in zip("KLMNO", expected):
                                cell = cells.get(f"{column}{number}")
                                raw = cell.findtext("m:v", namespaces=ns) if cell is not None else None
                                if value is None:
                                    matches = raw is None
                                else:
                                    matches = raw is not None and np.isclose(float(raw), value, atol=1e-12, rtol=0.)
                                if not matches:
                                    issues.append(f"XLSX cell {column}{number} does not match the metrics")
    except (OSError, KeyError, IndexError, TypeError, ValueError, zipfile.BadZipFile, ElementTree.ParseError) as err:
        return [f"XLSX could not be independently verified: {err}"]
    return issues


def _audit_development_receipts(directory, binding, validation, sample_ids):
    detail = read_json(directory/"report.json")
    if "resumed_case_receipts" not in detail.get("validation_runtime", {}):
        return 0, []
    label = f"{binding['arm']} {binding['seed']} development checkpoint {validation['step']}"
    paths = sorted((directory/"case_receipts").glob("*.json"))
    rows = detail["per_case"]
    if len(paths) != len(rows) or [row["sample_id"] for row in rows] != sample_ids:
        return 0, [f"{label} receipt count or order differs from the validation manifest"]
    issues = []
    if any(detail.get(key) != value for key, value in validation.items()):
        issues.append(f"{label} reported summary differs from the completed training record")
    if any(detail.get(key) != value for key, value in summarize(rows).items()):
        issues.append(f"{label} per-case summary does not reproduce the validation metrics")
    for position, (path, row) in enumerate(zip(paths, rows, strict=True), 1):
        saved = read_json(path)
        identity = {"version": 1, "binding": binding, "step": validation["step"],
                    "network_sha256": detail["network_sha256"], "position": position, "sample_id": row["sample_id"]}
        encoded = json.dumps(saved["row"], ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if (path.name != f"{position:05d}.json" or saved["identity"] != identity or saved["row"] != row
                or sha256(encoded).hexdigest() != saved["row_sha256"]):
            issues.append(f"{label} receipt differs at case {position}")
    return len(paths), issues


def audit_run(output, cfg):
    problems = []

    def require(condition, note):
        if not condition:
            problems.append(note)

    matrix = read_json(output/"training_matrix.json")
    lock = read_json(output/"evaluation_lock.json")
    frozen = read_json(output/"evaluation_cases.json")
    train = read_json(output/"cases.json")
    training_ids = {r["sample_id"] for r in train["train"]}
    eval_ids = [r["sample_id"] for r in frozen["cases"]]
    validation_ids = [r["sample_id"] for r in train["val"]]
    validation_count = len(validation_ids)
    require(matrix["profile"] == lock["profile"] == cfg["profile"], "Study profile differs across locks")
    require(lock["training_sha256"] == file_sha(output/"training_matrix.json"), "Training lock mismatch")
    require(lock["cases_sha256"] == file_sha(output/"evaluation_cases.json"), "Evaluation data lock mismatch")
    require(len(eval_ids) == len(set(eval_ids)) == frozen["manifest"]["eligible"],
            "Evaluation denominator differs from eligible cases")
    require({r["sample_id"] for r in frozen["preflight"]} == set(eval_ids), "Incomplete evaluation preflight")
    require(not training_ids.intersection(eval_ids), "Train/eval sample overlap")
    if cfg["profile"] == "formal":
        groups = {r["group_id"] for split in ("train","val") for r in train[split]}
        require(not groups.intersection(r["group_id"] for r in frozen["cases"]), "Family holdout overlap")
        manifest = frozen["manifest"]
        excluded = set(manifest["excluded_ineligible_ids"])
        official = ROOT/"data/splits/uiuc_v1/test.json"
        require(manifest["official_hash"] == file_sha(official)
                and set(eval_ids).union(excluded) == set(read_json(official)), "Evaluation cohort differs from the official split")
        require(not excluded.intersection(eval_ids) and len(excluded) == manifest["ineligible"]
                and len(eval_ids) + len(excluded) == manifest["official_total"], "Incomplete official eligibility accounting")
        require(dict(Counter(r["group_id"] for r in frozen["cases"])) == manifest["by_family"],
                "Evaluation family totals differ from the manifest")
    teachers = read_json(output/"teachers/manifest.json")
    require([(r["sample_id"], r["group_id"]) for r in teachers["rows"]] ==
            [(r["sample_id"], r["group_id"]) for r in train["train"]], "Incomplete or misbound train-only teacher ledger")
    require(read_json(output/"teachers/historical_audit.json")["count"] == 30, "Incomplete historical teacher audit")
    expected_extension = cfg["first_steps"] < cfg["extended_steps"] and any(
        sum(delta > 0 for delta in row.values()) >= 2 for row in matrix["extension_deltas"].values()
    )
    require(matrix["extended"] == expected_extension, "Validation extension trigger mismatch")
    require(matrix["online_steps"] == cfg["extended_steps" if expected_extension else "first_steps"],
            "Common online budget mismatch")
    require(read_json(output/"evaluation/E9_ch_sac_reference.json")["commit"] == B_COMMIT,
            "CH-SAC reference commit changed")
    for name, sha in matrix["checkpoints"].items():
        require(file_sha(output/name) == sha, f"Checkpoint digest mismatch: {name}")
    interval = cfg["validation_every"]
    mesh_keys = {str(budget) for budget in BUDGETS}
    receipts = {}
    development_receipts = 0
    for arm in (*SAC_ARMS, "E5"):
        for seed in cfg["seeds"]:
            root = output/"arms"/f"{arm}_seed{seed}"
            receipt = read_json(root/"completion.json")
            receipts[arm, seed] = receipt
            require(file_sha(root/"completion.json") == matrix["receipt_hashes"][f"{arm}_{seed}"], f"{arm} {seed} changed")
            require(receipt["best_step"] == max(receipt["validation"],key=selection_key)["step"], f"{arm} {seed} selection")
            final_step = 0 if arm == "E5" else matrix["online_steps"]
            expected_steps = [*range(0, final_step, interval), final_step]
            require(receipt["final_step"] == receipt["online_steps"] == final_step
                    and [r["step"] for r in receipt["validation"]] == expected_steps
                    and all(r["total"] == validation_count for r in receipt["validation"]),
                    f"{arm} {seed} validation schedule or denominator")
            for validation in receipt["validation"]:
                count, issues = _audit_development_receipts(root/f"validation_{validation['step']:06d}",
                                                            receipt["binding"], validation, validation_ids)
                development_receipts += count
                problems.extend(issues)
            if arm in ("E6", "E7", "E8"):
                history = {r["step"]: r["successful"] for r in receipt["validation"]}
                require(matrix["extension_deltas"][arm][str(seed)] ==
                        history[cfg["first_steps"]] - history[cfg["extension_compare_step"]],
                        f"{arm} {seed} extension delta differs from validation history")
            if arm in SAC_ARMS:
                count = 0
                with (root/"interactions.jsonl").open(encoding="utf-8") as log:
                    for count, line in enumerate(log,1):
                        row = json.loads(line)
                        require(row["step"] == count and row["sample_id"] in training_ids, f"{arm} {seed} interaction {count}")
                        if row["guard"] is not None:
                            require([r["action"] for r in row["guard"]["actions"] if r["kept"]] == row["effective_actions"],
                                    f"{arm} {seed} mask mismatch at {count}")
                            for proof in row["guard"]["actions"]:
                                require(proof["kept"] == (proof["status"] != "proven_dead_end"),
                                        f"{arm} {seed} prune status {count}")
                                if proof["reason"] == "all_second_layer_branches_proven_failed":
                                    require(proof["second_layer_total"] ==
                                            proof["second_layer_checked"] == proof["second_layer_failures"],
                                            f"{arm} {seed} incompletely enumerated proof {count}")
                require(count == matrix["online_steps"], f"{arm} {seed} interaction count")
    tested = evaluation_receipts = 0
    operational = {"seconds", "work", "guard_work"}
    for arm, seed, tag in jobs(cfg):
        value = _evaluation(output, arm, seed, tag)
        rows = value["rows"]
        require((value["arm"], value["seed"], value["tag"]) == (arm, seed, tag), f"{arm} {seed} {tag} run identity")
        if tag == "rule":
            require(value["step"] is None and value["checkpoint_sha256"] is None, f"{arm} {seed} {tag} rule identity")
        else:
            checkpoint = f"arms/{arm}_seed{seed}/{tag}.pt"
            require(value["step"] == receipts[arm, seed][f"{tag}_step"]
                    and value["checkpoint_sha256"] == matrix["checkpoints"][checkpoint],
                    f"{arm} {seed} {tag} checkpoint identity")
        require([r["sample_id"] for r in rows] == eval_ids, f"{arm} {seed} {tag} incomplete cases")
        require(summarize(rows) == value["summary"], f"{arm} {seed} {tag} aggregate changed")
        require(value["lock_sha256"] == file_sha(output/"evaluation_lock.json"), f"{arm} {seed} {tag} identity")
        ledger = output/"evaluation/case_receipts"/case_key(arm, seed, tag)
        saved = {path.name: path for path in ledger.glob("*.json")}
        for position, row in enumerate(rows, 1):
            receipt_path = saved.pop(f"{position:05d}.json", None)
            if receipt_path is not None:
                record = read_json(receipt_path)
                identity = {"version": 1, "lock_sha256": value["lock_sha256"], "arm": arm, "seed": seed,
                            "tag": tag, "position": position, "sample_id": row["sample_id"]}
                retained = {k: v for k, v in record["row"].items() if k not in operational}
                measured = {k: v for k, v in row.items() if k not in operational}
                require(record["identity"] == identity and retained == measured,
                        f"{arm} {seed} {tag} evaluation receipt differs at case {position}")
                evaluation_receipts += 1
            if row["success"]:
                require(all_quadrilateral(BlockGraph.from_dict(row["graph"])) and row["edits"] <= 32,
                        f"{row['sample_id']} unfinished or over-budget topology")
                meshes = row.get("mesh", {})
                require(set(meshes) == mesh_keys,
                        f"{arm} {seed} {tag} {row['sample_id']} missing mesh budgets")
                for key, mesh in meshes.items():
                    budget = int(key)
                    require(mesh["target"] == budget
                            and (not mesh["valid"] or mesh["budget_match"] == (abs(mesh["quads"] - budget) <= .05 * budget)),
                            f"{arm} {seed} {tag} {row['sample_id']} incorrect mesh budget accounting")
                if arm != "E4":
                    require(row["actions"][-1]["candidate"]["kind"] == "stop", f"{row['sample_id']} missing STOP")
        require(not saved, f"{arm} {seed} {tag} unexpected evaluation receipts")
        tested += len(rows)
    cfd = read_json(output/"cfd/summary.json")
    cfd_settings = {}
    for phase in ("warm", "rans"):
        published_config = published_bytes(f"cfd_verification/naca0012_v2_5120/{phase}.cfg")
        cfd_settings[phase] = [line for line in published_config.decode().replace("airfoil", "wall").splitlines()
                               if line.strip()]
    require(file_sha(solver_path()) == cfd["solver_sha256"], "SU2 executable identity changed")
    if cfg["profile"] == "formal":
        require(Counter((case["arm"], case["budget"]) for case in cfd["cases"]) ==
                Counter((arm, budget) for arm in ("E1", "E7", "E8") for budget in BUDGETS),
                "Incomplete formal CFD matrix")
        require(any(case.get("phases", {}).get("rans", {}).get("iterations", 0) > 0 for case in cfd["cases"]),
                "Formal CFD evidence has no observed RANS iterations")
    if cfd.get("status") == "analytical_naca0012_post_lock":
        source = read_json(output/"cfd/analytic_naca0012.json")
        require(source["evaluation_lock_sha256"] == file_sha(output/"evaluation_lock.json"), "CFD test lock mismatch")
        require(source["commit"] == B_COMMIT and source["source_sha256"] == sha256(published_bytes(source["source"])).hexdigest(),
                "Analytical NACA0012 source changed")
        require(cfd["sample_id"] not in training_ids.union(eval_ids), "Supplemental CFD sample used in primary study")
    solvers = 0
    for case in cfd["cases"]:
        if "mesh_sha256" not in case:
            continue
        folder = output/case["directory"]
        label = f"{case['arm']} {case['budget']} CFD"
        require(file_sha(folder/"mesh.su2") == case["mesh_sha256"], f"{label} mesh digest")
        for phase, value in case["phases"].items():
            settings = folder/f"{phase}.cfg"
            require(file_sha(settings) == value["config_sha256"], f"{label} {phase} config digest")
            effective = [line for line in settings.read_text(encoding="ascii").splitlines() if line.strip()]
            require(effective == cfd_settings[phase], f"{label} {phase} settings differ from the fixed reference")
            if "history_sha256" not in value:
                continue
            history = folder/f"{phase}_history.csv"
            require(file_sha(history) == value["history_sha256"], f"{label} {phase} history digest")
            rows = read_history(history)
            convergence = check_history(rows, residual_log10_target=-9., minimum_rows=100)
            require(len(rows) == value["iterations"] and (not rows or rows[-1] == value["last"]),
                    f"{label} {phase} iteration or last row mismatch")
            require(convergence == value["convergence"] and value["converged"] == (value["exit_code"] == 0 and convergence["converged"]),
                    f"{label} {phase} joint convergence mismatch")
            solvers += int(value["exit_code"] == 0)
        require(case["converged"] == case["phases"].get("rans", {}).get("converged", False), f"{label} case convergence mismatch")
    report = output/"report"
    diversity_path = report/"topology_diversity.json"
    artifacts = [report/"report.md", report/"实验结果.xlsx", report/"figure_evidence.json", diversity_path,
                 report/"training_diagnostics.json",
                 *(report/f"{figure}.{ext}" for figure in FIGURES for ext in ("png","pdf"))]
    if diversity_path.is_file():
        diversity = read_json(diversity_path)
        if cfg["profile"] == "formal" or "canonical_uv" in diversity:
            require(diversity == _topology_diversity(output, cfg), "Topology diversity differs from the sealed evaluations")
    case_handoff = output/"evaluation_case_resume_handoff.json"
    if case_handoff.is_file():
        artifacts.append(case_handoff)
        handoff = read_json(case_handoff)
        require(handoff["frozen_protocol_source_sha256"] == read_json(output/"source.json")["source"]["source_sha256"]
                and handoff["formal_data_sealed_at_activation"], "Evaluation receipt handoff differs from the sealed study")
    development_handoff = output/"validation_case_resume_handoff.json"
    if development_handoff.is_file():
        artifacts.append(development_handoff)
        development = read_json(development_handoff)
        require(case_handoff.is_file() and development["previous_launcher_sha256"] == read_json(case_handoff)["launcher_sha256"]
                and development["frozen_protocol_source_sha256"] == read_json(output/"source.json")["source"]["source_sha256"]
                and development["formal_data_sealed_at_activation"], "Development validation handoff differs from the sealed study")
    cache_handoff = output/"validation_cache_compression_handoff.json"
    if cache_handoff.is_file():
        artifacts.append(cache_handoff)
        cache = read_json(cache_handoff)
        require(development_handoff.is_file() and cache["launcher_sha256"] == read_json(development_handoff)["launcher_sha256"]
                and cache["frozen_protocol_source_sha256"] == read_json(output/"source.json")["source"]["source_sha256"]
                and cache["formal_data_sealed_at_activation"], "Compressed validation cache handoff differs from the sealed study")
    metrics_path = report/"evaluation_metrics.json"
    metrics = None
    if cfg["profile"] == "formal" or metrics_path.is_file():
        artifacts.append(metrics_path)
        if metrics_path.is_file():
            metrics = read_json(metrics_path)
            require(metrics == _evaluation_metrics(output, cfg), "Report metrics differ from the sealed evaluations")
            require(all(r["completed_without_legal_topology"] == 0 for r in metrics["runs"]),
                    "A completed partition failed the topology audit")
    if (output / "accelerated_handoff.json").is_file():
        execution_path = report / "execution_timing.json"
        artifacts.append(execution_path)
        if execution_path.is_file():
            execution = read_json(execution_path)
            require(execution == _execution_timing(output, cfg), "Accelerated validation timing evidence differs")
            rows = execution["accelerated_validations"]
            require(bool(rows) and all(0 <= row["reused_trajectories"] <= row["cases"]
                                      and 0 <= row.get("resumed_case_receipts", 0) <= row["cases"]
                                      and row["validation_elapsed_seconds"] >= 0 for row in rows),
                    "Missing or invalid accelerated validation runtime evidence")
    for path in artifacts:
        require(path.is_file() and path.stat().st_size > 0, f"Missing {path.name}")
    evidence = report/"figure_evidence.json"
    if evidence.is_file():
        figure = read_json(evidence)
        require(figure["example"] in eval_ids, "Figure example outside sealed evaluation cases")
        for record in figure.get("timing", ()):
            require(record["partition_seconds"] >= 0 and [mesh["target"] for mesh in record["meshes"]] == list(BUDGETS)
                    and all(mesh["seconds"] >= 0 for mesh in record["meshes"]), "Invalid representative timing evidence")
        if metrics_path.is_file():
            row = next(r for r in _evaluation(output, figure["arm"], figure["seed"])["rows"] if r["sample_id"] == figure["example"])
            mesh = row.get("mesh", {}).get(str(BUDGETS[0]), {})
            histogram = figure.get("coarse_mesh_jacobians")
            if mesh.get("valid"):
                require(histogram is not None and len(histogram["edges"]) == len(histogram["counts"]) + 1
                        and sum(histogram["counts"]) == histogram["cells"] == mesh["quads"],
                        "Representative Jacobian histogram does not account for every grid cell")
            else:
                require(histogram is None, "Representative Jacobian histogram has no valid mesh")
    xlsx = report/"实验结果.xlsx"
    if xlsx.is_file():
        runs = read_json(output/"evaluation/summary.json")["runs"]
        problems.extend(_audit_workbook(xlsx, runs, tested, metrics, _workbook_case_rows(output, runs)))
    result = {"passed": not problems, "profile": cfg["profile"], "cases_checked": tested, "solver_runs": solvers,
              "evaluation_case_receipts_checked": evaluation_receipts, "development_case_receipts_checked": development_receipts,
              "problems": problems, "artifacts": {p.name:file_sha(p) for p in artifacts if p.is_file()}}
    write_json(output/"audit.json", result, replace=True)
    return result
