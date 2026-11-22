"""Compare the 2D mesh policy across seeds, resolutions, and matched budgets."""

import argparse
import csv
import json
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np

from smartcfd.evaluation.mesh_quality import quality
from smartcfd.geometry.airfoil import case_features
from smartcfd.meshing.structured import BASE_PARAMS, PARAM_DIM, generate
from smartcfd.rl.agents.sac import SAC
from smartcfd.rl.env import HORIZON, is_better, mesh_metrics, run_case
from smartcfd.rl.trainer import dataset, optimize_reference


SEEDS = (1729, 1730, 1731, 1732, 1733)
BUDGETS = (1, 2, 3, 5, 9)
PAPERS = [
    {
        "title": "Kim、Kim 与 You · 二维叶栅结构网格",
        "detail": "预先指定首层高度，用确定性一步策略选择椭圆网格参数。四个叶型的网格质量为迭代收敛结果的 0.99–1.01 倍。该结果支持把低预算的直接预测纳入对照。",
        "url": "https://doi.org/10.1016/j.cpc.2023.108962",
    },
    {
        "title": "Pan 等 · FreeMesh-RL",
        "detail": "图 9 中，SAC 在作者的二维逐单元生成任务中比 PPO、DDPG、TD3 收敛更稳。表 6 中，FreeMesh-RL 与 Pave 的平均缩放雅可比同为 0.94（按发表精度）。两项比较均采用非结构化四边形网格任务。",
        "url": "https://arxiv.org/html/2203.11203",
    },
    {
        "title": "Lim 等 · 二维翼型边界层网格",
        "detail": "用 Double DQN 逐层移动节点，在 NACA 0012 上展示 20 层近壁网格。这为二维翼型上的强化学习网格生成提供了另一条算法路线。",
        "url": "https://doi.org/10.1007/s00366-024-02045-4",
    },
    {
        "title": "DiPrete 等 · 平面 CAD 的强化学习分块",
        "detail": "在由轴对齐直边组成的二维形状中用强化学习选择切割位置和方向。该实验支持学习式拓扑切割的可行性，其几何域与本实验的翼型 O 网格不同。",
        "url": "https://doi.org/10.1007/s00366-023-01940-6",
    },
    {
        "title": "Narayanan 与 Persson · Playing to Par（2026 年预印本）",
        "detail": "用行为克隆加 PPO 编辑平面四边形分块的连接关系。在其 96 个留出域中，平均 90 个达到顶点不规则性的理论下界。该指标针对拓扑规则性。",
        "url": "https://arxiv.org/abs/2609.32146",
    },
]


def saved_runs(checkpoints_dir):
    for seed in SEEDS:
        path = checkpoints_dir / f"naca-2d-{seed}"
        meta = json.loads((path / "training.json").read_text(encoding="utf-8"))["meta"]
        if meta["seed"] != seed or meta["steps"] != 24_000:
            raise ValueError(f"Unexpected training configuration in {path}")
        yield seed, path, meta


def rank(metrics):
    return sum(row["invalid_cells"] == 0 for row in metrics), np.mean([row["score"] for row in metrics])


def optimize_global(training, checkpoints):
    """Fit a single global parameter vector by directly scoring training geometry."""
    def objective(parameters):
        return rank([mesh_metrics(case, parameters) for case in training])

    starts = [BASE_PARAMS, *[np.asarray(meta["fixed_parameters"]) for _, _, meta in checkpoints]]
    ranked = sorted(((objective(x), x) for x in starts), key=lambda pair: pair[0], reverse=True)
    axes = np.eye(PARAM_DIM)
    solutions = []
    for incumbent, start in ranked[:2]:
        parameters = start.copy()
        for radius in (0.40, 0.20, 0.10, 0.05, 0.025):
            for axis in axes:
                anchor = parameters.copy()
                for sign in (-1, 1):
                    proposal = np.clip(anchor + sign * radius * axis, -1, 1)
                    if np.array_equal(proposal, parameters):
                        continue
                    value = objective(proposal)
                    if value > incumbent:
                        parameters, incumbent = proposal, value
        solutions.append((incumbent, parameters))
    return max(solutions, key=lambda pair: pair[0])[1]


def direct_features(case):
    linear = [1.0, *case_features(case)]
    quadratic = [linear[i] * linear[j] for i, j in combinations_with_replacement(range(1, 4), 2)]
    return np.asarray([*linear, *quadratic], dtype=np.float64)


def fit_direct(training, validation):
    """Learn a one-shot ridge regressor on training-case search demonstrations."""
    rng = np.random.default_rng(4051)
    targets = np.stack([optimize_reference(case, rng) for case in training])
    design = np.stack([direct_features(case) for case in training])
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0
    fits = []
    for regularization in (0.001, 0.01, 0.1, 1.0):
        weights = np.linalg.solve(design.T @ design + regularization * penalty, design.T @ targets)
        metrics = [mesh_metrics(case, direct_parameters(case, weights)) for case in validation]
        fits.append((rank(metrics), regularization, weights))
    _, regularization, weights = max(fits, key=lambda row: row[0])
    return weights, regularization


def direct_parameters(case, weights):
    return np.clip(direct_features(case) @ weights, -1, 1)


def local_search(case, start, radius):
    """Make eight local proposals from a globally fitted starting point."""
    best_params = start.copy()
    best = mesh_metrics(case, best_params)
    sequence = [(best, best_params)]
    order = (1, 0, 2, 3, 4, 1, 0, 2)
    directions = np.where(start + radius > 1, -1, 1)
    for step, coordinate in enumerate(order):
        proposal = best_params.copy()
        direction = directions[coordinate] * (1 if step < PARAM_DIM else -1)
        # Test the opposite sign from the original anchor. At a box boundary,
        # use a shorter inward move so the candidate still differs.
        anchor = best_params[coordinate] if step < PARAM_DIM else start[coordinate]
        value = np.clip(anchor + direction * radius, -1, 1)
        if value == best_params[coordinate]:
            value = np.clip(anchor + directions[coordinate] * radius / 2, -1, 1)
        proposal[coordinate] = value
        metrics = mesh_metrics(case, proposal)
        if is_better(metrics, best):
            best_params, best = proposal, metrics
        sequence.append((best, best_params))
    return sequence


def record(rows, method, seed, case, budget, parameters, coarse):
    fine = quality(generate(case, parameters, fine=True)) if (
        method in ("fixed", "global", "direct") or budget == HORIZON + 1
    ) else None
    rows.append({
        "method": method, "seed": seed, "case": case.name, "budget": budget,
        "score": coarse["score"], "invalid_cells": coarse["invalid_cells"],
        "min_jacobian": coarse["min_jacobian"], "p05_jacobian": coarse["p05_jacobian"],
        "mean_first_layer": coarse["mean_first_layer"],
        "fine_score": fine["score"] if fine else None,
        "fine_invalid_cells": fine["invalid_cells"] if fine else None,
    })


def mean(rows, key):
    return float(np.mean([row[key] for row in rows if row[key] is not None]))


def analyze(checkpoints_dir: Path, output: Path) -> None:
    checkpoints = list(saved_runs(checkpoints_dir))
    train_cases, val_cases, test_cases = dataset()
    expected = {split: [case.name for case in cases] for split, cases in
                zip(("train", "validation", "test"), (train_cases, val_cases, test_cases))}
    if any(meta["splits"] != expected for _, _, meta in checkpoints):
        raise ValueError("Training, validation and test lists must agree across runs")
    fixed = optimize_global(train_cases, checkpoints)
    regressor, ridge_penalty = fit_direct(train_cases, val_cases)
    radius = max((0.06, 0.12, 0.25, 0.45), key=lambda r:
                 rank([local_search(case, fixed, r)[-1][0] for case in val_cases]))

    records = []
    for case in test_cases:
        for method, parameters in (("fixed", BASE_PARAMS), ("global", fixed),
                                   ("direct", direct_parameters(case, regressor))):
            record(records, method, None, case, 1, parameters, mesh_metrics(case, parameters))
        local = local_search(case, fixed, radius)
        for budget in BUDGETS:
            metrics, parameters = local[budget - 1]
            record(records, "local", None, case, budget, parameters, metrics)
    for seed, path, meta in checkpoints:
        for method in ("bc", "sac"):
            actor = SAC(meta["observation_dim"], PARAM_DIM, np.random.default_rng(seed))
            actor.load_actor(path / f"{method}_actor.npz")
            for case in test_cases:
                for budget in BUDGETS:
                    evaluation = run_case(case, method, actor, max_steps=budget - 1)
                    record(records, method, seed, case, budget, evaluation["parameters"], evaluation["metrics"])
        print(f"Evaluated checkpoint for seed {seed}", flush=True)

    by_budget = {
        (method, budget): [row for row in records if row["method"] == method and row["budget"] == budget]
        for method in ("fixed", "global", "direct", "local", "bc", "sac") for budget in BUDGETS
    }
    row_specs = [
        ("fixed", "固定参数", 1), ("global", "训练集全局调参", 1),
        ("direct", "一步岭回归", 1), ("local", "全局起点＋局部搜索", 9),
        ("bc", "行为克隆", 9), ("sac", "SAC", 9),
    ]
    rows = []
    for method, label, budget in row_specs:
        subset = by_budget[method, budget]
        seed_means = [mean([row for row in subset if row["seed"] == seed], "score")
                      for seed in SEEDS] if method in ("bc", "sac") else []
        rows.append({
            "id": method, "label": label, "budget": budget, "score": mean(subset, "score"),
            "seed_sd": float(np.std(seed_means, ddof=1)) if seed_means else None,
            "valid_rate": sum(item["invalid_cells"] == 0 for item in subset) / len(subset),
            "mean_first_layer": mean(subset, "mean_first_layer"), "fine_score": mean(subset, "fine_score"),
            "fine_valid_rate": sum(item["fine_invalid_cells"] == 0 for item in subset) / len(subset),
            "worst_min_jacobian": min(item["min_jacobian"] for item in subset),
            "mean_p05_jacobian": mean(subset, "p05_jacobian"),
        })
    policy_records = by_budget["sac", 9]
    comparisons = []
    for method, label, budget in row_specs[1:-1]:
        lookup = {(row["seed"], row["case"]): row["score"] for row in by_budget[method, budget]}
        diffs = {seed: [] for seed in SEEDS}
        for row in policy_records:
            seed = row["seed"]
            other = lookup[(seed if method == "bc" else None, row["case"])]
            diffs[seed].append(row["score"] - other)
        paired = np.asarray(list(diffs.values()))
        seed_deltas = paired.mean(axis=1)
        comparisons.append({
            "id": method, "label": f"SAC 对比{label}", "mean_delta": float(paired.mean()),
            "wins": int(np.count_nonzero(paired > 1e-8)), "total": int(paired.size),
            "seed_min": float(seed_deltas.min()), "seed_max": float(seed_deltas.max()),
            "seed_deltas": dict(zip(SEEDS, seed_deltas.tolist())),
        })
    frontier = []
    for budget in BUDGETS:
        point = {"evaluations": budget}
        for method in ("fixed", "global", "direct", "local", "bc", "sac"):
            point[method] = mean(by_budget[method, 1 if method in ("fixed", "global", "direct") else budget], "score")
        frontier.append(point)
    seed_results = []
    for seed, _, meta in checkpoints:
        seed_results.append({
            "seed": seed, "checkpoint": meta["best_step"], "seconds": meta["train_seconds"],
            "score": mean([row for row in policy_records if row["seed"] == seed], "score"),
            "bc_score": mean([row for row in by_budget["bc", 9] if row["seed"] == seed], "score"),
        })

    sac, fixed_row = rows[-1], rows[0]
    local = next(row for row in rows if row["id"] == "local")
    delta = sac["score"] - local["score"]
    conclusion = (
        f"在 {len(SEEDS)} 个训练种子和同一组 {len(test_cases)} 个留出翼型上，SAC 的粗网格均分为 "
        f"{sac['score']:.6f}，比固定参数高 {sac['score'] - fixed_row['score']:.6f}。"
        f"与同为 {HORIZON + 1} 次评估的全局起点＋局部搜索相比，SAC 粗网格均分差值为 {delta:+.6f}。"
        "本表验证固定四块 O 型拓扑内的二维几何质量。"
    )
    bundle = {
        "methodology": {"seeds": list(SEEDS), "train_cases": len(train_cases), "val_cases": len(val_cases),
                        "test_cases": len(test_cases), "grid_cells": 1536, "fine_cells": 5120,
                        "local_radius": radius, "ridge_penalty": ridge_penalty,
                        "global_parameters": fixed.tolist()},
        "rows": rows, "comparisons": comparisons, "frontier": frontier, "seed_results": seed_results,
        "papers": PAPERS, "conclusion": conclusion,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "evidence.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (output / "evidence_cases.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({"rows": rows, "comparisons": comparisons, "frontier": frontier,
                      "seed_results": seed_results}, ensure_ascii=False, indent=2), flush=True)
    content = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    (output / "evidence-data.js").write_text(f"window.MESH_EVIDENCE = {content};\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("experiments/checkpoints"),
                        help="Parent of the five naca-2d-<seed> training run directories")
    parser.add_argument("--output", type=Path, default=Path("experiments/results/naca-2d-evidence"),
                        help="New directory for the experiment results")
    args = parser.parse_args()
    try:
        analyze(args.checkpoints_dir, args.output)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
