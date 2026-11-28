"""Auditable final-checkpoint summaries for the nine-arm airfoil development study.

Training seeds are repeated policies, never additional independent airfoils.
Mesh diagnostics stay separate from partition success and checkpoint selection.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics


SCHEMA = "airfoil_viability_study_report_v1"
METHODS = ("fixed_c", "random", "greedy", "sac", "bc")
ARMS = ("fixed_c_off", "random_off", "random_on", "greedy_off", "greedy_on",
        "sac_off", "sac_on", "bc_off", "bc_on")


def _mean(values):
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def _stats(values):
    values = [value for value in values if value is not None]
    return {"n": len(values), "mean": _mean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "values": values}


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _records(evaluations):
    return list(evaluations.values()) if isinstance(evaluations, dict) else list(evaluations)


def _cost(row):
    # Preserve the names emitted by the runner, rather than guessing equivalent
    # counters or treating unavailable measurement as a measured zero.
    result = {}
    for name in ("cost", "guard_cost", "compute_cost"):
        values = row.get(name) or {}
        if isinstance(values, dict):
            for key, value in values.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    result[f"{name}.{key}"] = _finite(value, key)
    return result


def _mesh_rows(report, diagnostics, run_id):
    rows = report.get("mesh_diagnostics")
    if rows is None and diagnostics is not None:
        rows = diagnostics.get(run_id)
    if rows is None:
        return {}
    if isinstance(rows, dict):
        rows = rows.get("episodes", rows.get("rows", list(rows.values())))
    result = {}
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in result:
            raise ValueError("Duplicate mesh diagnostic sample")
        result[sample_id] = row
    return result


def aggregate_study(evaluations, *, max_steps=32, mesh_diagnostics=None,
                    protocol=None, require_complete=True):
    """Aggregate final reports; reject different evaluation populations.

    Each evaluation supplies method, guard_enabled, seed, selection and episodes.
    Existing ``rule`` is accepted as the spelling of ``greedy``. Diagnostics may
    be attached per report, or keyed by ``{method}_{on|off}_seed{seed}`` externally.
    Set require_complete=False only for an explicitly labelled integration smoke.
    """
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    protocol = protocol or {}
    expected_seeds = set(protocol.get("seeds", [42, 43, 44]))
    runs, excluded, seen, population = [], [], set(), None
    for report in _records(evaluations):
        method = {"rule": "greedy"}.get(report["method"], report["method"])
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        selection = report.get("selection")
        if selection not in (None, "final", "baseline"):
            excluded.append({"method": method, "seed": report.get("seed"), "selection": selection})
            continue
        if method in ("sac", "bc") and selection != "final":
            raise ValueError("Learned primary results must identify the final checkpoint")
        guard = report.get("guard_enabled", False)
        if type(guard) is not bool:
            raise ValueError("guard_enabled must be boolean")
        arm = f"{method}_{'on' if guard else 'off'}"
        if arm not in ARMS:
            raise ValueError(f"Not a primary study arm: {arm}")
        seed = report.get("seed")
        if seed is not None and type(seed) is not int:
            raise ValueError("seed must be an integer or null")
        run_id = f"{arm}_seed{seed}"
        if run_id in seen:
            raise ValueError(f"Duplicate evaluation: {run_id}")
        seen.add(run_id)
        episodes = report["episodes"]
        if not episodes:
            raise ValueError("An evaluation cannot have an empty denominator")
        identity = {row["sample_id"]: row["group_id"] for row in episodes}
        if len(identity) != len(episodes):
            raise ValueError("Duplicate evaluation sample")
        if population is None:
            population = identity
        elif population != identity:
            raise ValueError("All primary reports must use the same full family-labelled manifest")
        if report.get("total", len(episodes)) != len(episodes):
            raise ValueError("Reported denominator disagrees with episode rows")
        mesh = _mesh_rows(report, mesh_diagnostics, run_id)
        if set(mesh) - set(identity):
            raise ValueError("Mesh diagnostics include an unevaluated sample")
        cases = []
        for row in episodes:
            if type(row["success"]) is not bool:
                raise ValueError("Episode success must be boolean")
            steps = row["steps"]
            if type(steps) is not int or not 0 <= steps <= max_steps:
                raise ValueError("Invalid episode step count")
            success = row["success"]
            quality = _finite(row["terminal_quality_score"], "terminal_quality_score") if success else None
            elapsed = row.get("elapsed_seconds")
            if elapsed is not None and _finite(elapsed, "elapsed_seconds") < 0:
                raise ValueError("Elapsed time cannot be negative")
            diagnostic = mesh.get(row["sample_id"])
            mesh_attempted = bool(diagnostic and diagnostic.get("attempted"))
            mesh_valid = bool(mesh_attempted and diagnostic.get("valid"))
            if mesh_attempted and not success:
                raise ValueError("A failed partition cannot have an attempted final mesh")
            mq = (diagnostic or {}).get("quality") or {}
            cases.append({"run_id": run_id, "arm": arm, "method": method,
                "guard_enabled": guard, "seed": seed, "sample_id": row["sample_id"],
                "group_id": row["group_id"], "success": success,
                "source_path": row.get("filepath", row.get("path")),
                "source_sha256": row.get("sha256"),
                "initializable": row.get("initializable"), "phase": row.get("phase"),
                "reason": row.get("reason"), "steps": steps,
                "penalized_steps": steps if success else max_steps,
                "terminal_quality_score": quality, "elapsed_seconds": elapsed,
                "cost": _cost(row), "mesh_recorded": diagnostic is not None,
                "mesh_attempted": mesh_attempted, "mesh_valid": mesh_valid,
                "mesh_cells": mq.get("cells"), "mesh_quality": mq,
                "mesh_failure_reason": (diagnostic or {}).get("reason")})
        successes = sum(row["success"] for row in cases)
        if report.get("success_count", successes) != successes:
            raise ValueError("Reported success count disagrees with episode rows")
        mesh_complete = all(row["mesh_recorded"] for row in cases)
        costs = sorted({key for row in cases for key in row["cost"]})
        runs.append({"run_id": run_id, "arm": arm, "method": method, "seed": seed,
            "guard_enabled": guard, "selection": selection, "total": len(cases),
            "success_count": successes, "success_rate": successes / len(cases),
            "mean_steps": _mean([r["steps"] for r in cases]),
            "mean_penalized_steps": _mean([r["penalized_steps"] for r in cases]),
            "success_only_quality": _mean([r["terminal_quality_score"] for r in cases]),
            "mean_elapsed_seconds": _mean([r["elapsed_seconds"] for r in cases]),
            "costs": {key: {"measured_cases": sum(key in r["cost"] for r in cases),
                            "total": sum(r["cost"].get(key, 0) for r in cases)} for key in costs},
            "failure_reasons": dict(Counter(str(r["reason"]) for r in cases if not r["success"])),
            "mesh_diagnostics_complete": mesh_complete,
            "mesh_valid_count": sum(r["mesh_valid"] for r in cases) if mesh_complete else None,
            "mesh_valid_rate_full_manifest": sum(r["mesh_valid"] for r in cases)/len(cases) if mesh_complete else None,
            "cases": cases})
    if not runs:
        raise ValueError("No final/baseline evaluations supplied")
    by_arm = defaultdict(list)
    for run in runs:
        by_arm[run["arm"]].append(run)
    if require_complete:
        if set(by_arm) != set(ARMS):
            raise ValueError("The complete study requires all nine arms")
        for arm, records in by_arm.items():
            seeds = {r["seed"] for r in records}
            if arm.startswith(("sac_", "bc_", "random_")) and seeds != expected_seeds:
                raise ValueError(f"Incomplete training/evaluation seeds: {arm}")
            if arm.startswith(("greedy_", "fixed_c_")) and len(records) != 1:
                raise ValueError("Deterministic baselines require one report, not pseudoreplicated seeds")
    arms = []
    for arm in ARMS:
        records = sorted(by_arm.get(arm, []), key=lambda r: -1 if r["seed"] is None else r["seed"])
        if records:
            aggregate = {"arm": arm, "run_count": len(records), "seeds": [r["seed"] for r in records],
                "success_counts": [r["success_count"] for r in records], "total_per_run": len(population),
                **{key: _stats([r[key] for r in records]) for key in
                   ("success_rate", "mean_penalized_steps", "success_only_quality",
                    "mean_elapsed_seconds", "mesh_valid_rate_full_manifest")}}
            mesh = aggregate['mesh_valid_rate_full_manifest']
            mesh['complete'] = all(r['mesh_diagnostics_complete'] for r in records)
            if not mesh['complete']:
                mesh['mean'] = mesh['sample_sd'] = None
            arms.append(aggregate)
    pairs = _paired(by_arm)
    return {"schema": SCHEMA, "complete_matrix": require_complete,
        "scope": "Development validation only; no independent confirmation or CFD accuracy claim",
        "max_steps": max_steps, "unique_geometries": len(population),
        "unique_families": len(set(population.values())), "evaluation_rollouts": sum(r["total"] for r in runs),
        "population": population, "excluded_nonprimary_reports": excluded,
        "aggregation": "Per-run geometry means, then equally weighted seed means; sample SD is training/evaluation seed spread, not a confidence interval. Successful-only quality populations can differ; use paired common-success rows for comparisons.",
        "mesh_scope": "Posthoc fixed mesher; actual cell counts may differ. Missing diagnostics are unknown, never counted as measured failures. No equal-cell quality claim.",
        "cost_scope": "Guard previews, rule search and teacher generation add work; equal real steps do not imply equal computation. Unmeasured counters remain absent.",
        "protocol": protocol, "arms": arms, "runs": runs, "paired_comparisons": pairs}


def _paired(by_arm):
    specifications = [(f"{method}_on", f"{method}_off", "guard_effect")
                      for method in ("random", "greedy", "sac", "bc")]
    specifications += [(f"sac_{guard}", f"{base}_{guard}", "learning_at_same_guard")
                       for guard in ("off", "on") for base in ("random", "greedy", "bc")]
    specifications += [(f"sac_{guard}", "fixed_c_off", "fixed_template_reference") for guard in ("off", "on")]
    result = []
    for left, right, kind in specifications:
        if left not in by_arm or right not in by_arm:
            continue
        rows = []
        for a in by_arm[left]:
            candidates = [b for b in by_arm[right] if b["seed"] == a["seed"]]
            if not candidates and len(by_arm[right]) == 1 and by_arm[right][0]["method"] in ("greedy", "fixed_c"):
                candidates = by_arm[right]
            if len(candidates) != 1:
                raise ValueError(f"No unique paired seed for {left} versus {right}")
            b = candidates[0]
            rhs = {r["sample_id"]: r for r in b["cases"]}
            per_case = []
            for x in a["cases"]:
                y = rhs[x["sample_id"]]
                common = x["success"] and y["success"]
                per_case.append({"sample_id": x["sample_id"], "group_id": x["group_id"],
                    "delta_success": int(x["success"]) - int(y["success"]),
                    "delta_penalized_steps": x["penalized_steps"]-y["penalized_steps"],
                    "common_success": common,
                    "delta_quality": x["terminal_quality_score"]-y["terminal_quality_score"] if common else None})
            rows.append({"seed": a["seed"], "reference_seed": b["seed"],
                "total": len(per_case), "common_success_count": sum(r["common_success"] for r in per_case),
                "delta_success_rate": _mean([r["delta_success"] for r in per_case]),
                "delta_penalized_steps": _mean([r["delta_penalized_steps"] for r in per_case]),
                "delta_quality_common_success": _mean([r["delta_quality"] for r in per_case]),
                "cases": per_case})
        result.append({"left": left, "right": right, "kind": kind, "direction": "left minus right",
            "seed_rows": rows, **{key: _stats([r[key] for r in rows]) for key in
                ("delta_success_rate", "delta_penalized_steps", "delta_quality_common_success")}})
    return result


def _fmt(value, digits=3):
    return "未测/不适用" if value is None else f"{value:.{digits}f}"


def render_report(summary):
    """A source-linked Chinese account which never presupposes learning gains."""
    lines = ["# 二维翼型保守筛选与学习排序：开发实验", "",
        f"本报告包含 {summary['unique_geometries']} 个独立几何、{summary['unique_families']} 个家族，"
        f"共 {summary['evaluation_rollouts']} 次评估。种子是重复训练/随机策略，不增加独立翼型数量。",
        "", "本次为完整九组开发矩阵。" if summary['complete_matrix'] else "本次仅为部分矩阵/集成检查，不能作为完整实验结论。",
        "主结果使用最终检查点；验证最优检查点和中间检查点不进入主表。验证集参与开发，结果不构成独立泛化确认。",
        "", "| 方法 | 各种子成功数 / 每次分母 | 成功率均值 ± 种子样本标准差 | 失败按预算计的步数 | 网格有效率（完整分母） |",
        "|---|---|---:|---:|---:|"]
    for arm in summary['arms']:
        success = arm['success_rate']
        sd = success['sample_sd']
        spread = "（单次评估，无种子标准差）" if sd is None else f" ± {100*sd:.2f} 百分点"
        count = ", ".join(str(n) for n in arm['success_counts'])
        mesh = arm['mesh_valid_rate_full_manifest']['mean']
        lines.append(f"| {arm['arm']} | {count} / {arm['total_per_run']} | {100*success['mean']:.2f}%{spread} | "
            f"{_fmt(arm['mean_penalized_steps']['mean'])} | {'未完成诊断' if mesh is None else f'{100*mesh:.2f}%'} |")
    lines += ["", "## 成对比较", "", "下表均为左侧减右侧；成功率越高越好，预算步数越低越好。质量只比较同一种子下双方均成功的相同翼型，样本数随比较改变。",
        "", "| 比较 | 类型 | 成功率差（百分点） | 预算步数差 | 共同成功质量分差 | 各种子共同成功数 |",
        "|---|---|---:|---:|---:|---|"]
    for pair in summary['paired_comparisons']:
        counts = ", ".join(str(r['common_success_count']) for r in pair['seed_rows'])
        lines.append(f"| {pair['left']} − {pair['right']} | {pair['kind']} | "
            f"{100*pair['delta_success_rate']['mean']:.2f} | {_fmt(pair['delta_penalized_steps']['mean'])} | "
            f"{_fmt(pair['delta_quality_common_success']['mean'])} | {counts} |")
    lines += ["", "## 由本次结果支持的观察", ""]
    for pair in summary['paired_comparisons']:
        if pair['kind'] == 'guard_effect':
            delta = pair['delta_success_rate']['mean'] * 100
            direction = '提高' if delta > 0 else '降低' if delta < 0 else '不变'
            lines.append(f"- `{pair['left'].rsplit('_', 1)[0]}` 加筛选后，成功率均值{direction}"
                         f"（差值 {delta:+.2f} 个百分点）；预算步数差 {_fmt(pair['delta_penalized_steps']['mean'])}。")
        elif pair['kind'] == 'learning_at_same_guard' and pair['right'].startswith(('greedy_', 'bc_')):
            delta = pair['delta_success_rate']['mean'] * 100
            quality = pair['delta_quality_common_success']
            stability = ('各种子质量差均为正' if quality['values'] and all(v > 0 for v in quality['values'])
                         else '各种子质量差均为负' if quality['values'] and all(v < 0 for v in quality['values'])
                         else '质量差未呈现各种子一致的严格方向')
            lines.append(f"- `{pair['left']}` 相对 `{pair['right']}` 的成功率均值差为 {delta:+.2f} 个百分点；"
                         f"共同成功质量差均值 {_fmt(quality['mean'])}，{stability}。")
    lines.append("这些是小规模开发集上的描述性结果；均值、种子一致性或样本标准差都不等同于独立确认或显著性检验。")
    failures = Counter()
    for run in summary['runs']:
        failures.update(run['failure_reasons'])
    lines += ["", "## 失败与结论边界", "",
        "失败按原始原因保留，包括初始化失败；失败预算代价统一为 " + str(summary['max_steps']) + " 步。",
        "失败次数：" + ("；".join(f"`{key}`：{value}" for key,value in sorted(failures.items())) or "0") + "。",
        "同方法的 on/off 比较反映筛选作用；相同筛选条件下 SAC 与规则、BC 的比较才用于判断学习作用。BC on/off 使用相同训练权重；SAC 两臂分别训练，不能将两种比较混作同一训练消融。",
        "如果随机或规则加筛选获得同样收益，收益应归于筛选。规则已全部成功时，成功率没有可提高空间，应核对预先指定的质量与成本指标。没有稳定优势也保留为实验结果，不据单个最佳种子宣称学习有效。",
        "网格诊断不参与奖励、分块成功或选模。固定制网规格可能产生不同单元数，Jacobian 和近壁指标不能直接解释为等单元预算质量提升或 CFD 精度。",
        "几何预览、第二层搜索、规则搜索和教师生成均增加计算；相同交互步数不代表相同计算成本。详细计数在逐例与逐运行记录中，缺失计量不以零替代。",
        "", "## 可核查产物", "",
        "- [完整汇总](summary.json)：逐种子、失败类别、成本与独立网格记录。",
        "- [逐例结果](per_case.csv) 与 [失败清单](failures.csv)。",
        "- [成对比较](paired_comparisons.json) 与 [比较表](paired_comparisons.csv)。",
        "- [结果图](overview.png) / [PDF](overview.pdf)。", ""]
    return "\n".join(lines)


def plot_overview(summary, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = summary['arms']
    fig, axes = plt.subplots(1, 3, figsize=(13, max(4.8, len(arms)*.43+1.4)), sharey=True)
    labels = [a['arm'].replace('_', ' ') for a in arms]
    colors = ['#218578' if a['arm'].endswith('_on') else '#8a9bb1' for a in arms]
    for ax, key, title, scale in zip(axes,
            ('success_rate', 'mean_penalized_steps', 'mesh_valid_rate_full_manifest'),
            ('Partition success (%)', f"Failure-penalized steps (max {summary['max_steps']})", 'Valid final mesh (%)'), (100, 1, 100)):
        values = [a[key]['mean'] for a in arms]
        plotted = [0 if v is None else scale*v for v in values]
        errors = [scale*(a[key]['sample_sd'] or 0) for a in arms]
        ax.barh(range(len(arms)), plotted, xerr=errors, color=colors, alpha=.9, capsize=3)
        for i,v in enumerate(values):
            if v is None:
                ax.text(.02, i, 'not measured', transform=ax.get_yaxis_transform(), va='center', fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
        if scale == 100:
            ax.set_xlim(0, 105)
        else:
            ax.set_xlim(0, summary['max_steps']*1.05)
    axes[0].set_yticks(range(len(arms)), labels); axes[0].invert_yaxis()
    fig.suptitle('Airfoil C-domain development validation | final checkpoints', fontsize=13)
    fig.text(.02, .025, f"{summary['unique_geometries']} geometries / {summary['unique_families']} families. Error bars: seed SD, not confidence intervals.\n"
        'Guard adds computation; fixed meshing does not match cell counts. No CFD or independent-test claim.', fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, .94))
    paths = {}
    try:
        for suffix in ('png', 'pdf'):
            path = Path(directory)/f'overview.{suffix}'
            fig.savefig(path, dpi=170)
            paths[suffix] = str(path)
    finally:
        plt.close(fig)
    return paths


def _write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def write_study_report(output_dir, evaluations, *, max_steps=32, mesh_diagnostics=None,
                       protocol=None, require_complete=True):
    summary = aggregate_study(evaluations, max_steps=max_steps, mesh_diagnostics=mesh_diagnostics,
                              protocol=protocol, require_complete=require_complete)
    output = Path(output_dir)
    names = ('summary.json', 'summary.csv', 'per_case.csv', 'failures.csv',
             'paired_comparisons.json', 'paired_comparisons.csv', 'report.md', 'overview.png', 'overview.pdf')
    if any((output/name).exists() for name in names):
        raise FileExistsError('Report artifacts already exist; use a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (('summary.json', summary), ('paired_comparisons.json', summary['paired_comparisons'])):
        with (output/name).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    cases = [{**{k:v for k,v in row.items() if k not in ('cost', 'mesh_quality')},
              **row['cost'], **{f'mesh.{k}':v for k,v in row['mesh_quality'].items() if not isinstance(v, (dict,list))}}
             for run in summary['runs'] for row in run['cases']]
    _write_csv(output/'per_case.csv', cases)
    _write_csv(output/'failures.csv', [r for r in cases if not r['success']])
    _write_csv(output/'summary.csv', [{**{k:v for k,v in run.items() if k not in ('cases','costs','failure_reasons')},
        'failure_reasons': json.dumps(run['failure_reasons'], sort_keys=True)} for run in summary['runs']])
    _write_csv(output/'paired_comparisons.csv', [{**{k:pair[k] for k in ('left','right','kind')},
        **{k:v for k,v in row.items() if k!='cases'}} for pair in summary['paired_comparisons'] for row in pair['seed_rows']])
    (output/'report.md').write_text(render_report(summary), encoding='utf-8')
    plot_overview(summary, output)
    return {name: str((output/name).resolve()) for name in names}
