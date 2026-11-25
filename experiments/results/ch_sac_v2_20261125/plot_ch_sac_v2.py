"""Step 6: Proposal-ready plots for CH-SAC v2 (training curves, benchmarks, mesh & CFD)."""

import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SCRIPTS = ROOT / "scripts"
for p in (OUT, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ch_env import DEFAULT_U, apply_7controls, load_cached_split, u_to_params
from smartcfd.meshing.tfi import generate
from cfd_support import DATA as EXP_DATA, pressure, read_rows

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


def plot_fig1_training():
    seeds = (42, 43, 44, 45, 46)
    base = json.loads((OUT / "baselines_summary.json").read_text(encoding="utf-8"))["methods"]
    histories = []
    for s in seeds:
        rows = list(csv.DictReader((OUT / "seeds" / f"seed_{s}" / "history.csv").open("r", encoding="utf-8")))
        histories.append(rows)

    epochs = np.array([int(r["epoch"]) for r in histories[0]])
    steps = np.array([int(r["env_steps"]) for r in histories[0]])

    def stack_col(col):
        return np.array([[float(r[col]) for r in h] for h in histories], dtype=float)

    val_score = stack_col("val_score")
    val_1536 = stack_col("val_score_1536")
    val_5120 = stack_col("val_score_5120")
    val_jmin = stack_col("val_jmin")
    val_j05 = stack_col("val_j05")
    val_far = stack_col("val_far_ar")
    closs = stack_col("critic_loss")[:, 1:]
    alpha = stack_col("alpha")[:, 1:]
    ep_nz = epochs[1:]

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.6), constrained_layout=True)

    # (a) Val Score vs Baselines
    ax = axes[0, 0]
    for arr, lbl, col in (
        (val_score, "CH-SAC v2 Overall (5-seed)", "#1b6ca8"),
        (val_1536, "Coarse (nominal 1,536)", "#2a9d8f"),
        (val_5120, "Budget 5,120", "#e76f51"),
    ):
        mu, sd = arr.mean(axis=0), arr.std(axis=0, ddof=1)
        ax.plot(epochs, mu, lw=2.2, color=col, label=lbl)
        ax.fill_between(epochs, mu - sd, mu + sd, color=col, alpha=0.18)

    # Mean across 1536 & 5120 for val baselines
    for bname, ls, col in (
        ("B0_Default", "--", "#6c757d"),
        ("B1_GlobalOpt", "-.", "#8a5a44"),
        ("B2_LocalSearch", ":", "#6f42c1"),
        ("B3_SupervisedMLP", (0, (3, 1, 1, 1)), "#d62828"),
    ):
        mval = 0.5 * (base[bname]["val"]["1536"]["score_mean"] + base[bname]["val"]["5120"]["score_mean"])
        ax.axhline(mval, ls=ls, lw=1.4, color=col, label=f"{bname} ({mval:.3f})")
    ax.set_title("(a) Validation Composite Score S(G) vs Non-RL Baselines")
    ax.set_xlabel("Epoch (80 episodes x 5 steps = 400 env steps/epoch)")
    ax.set_ylabel("Validation Score S(G)")
    ax.legend(loc="lower right", ncol=2, frameon=True)
    ax2 = ax.twiny()
    ax2.set_xlim(steps[0] / 1000.0, steps[-1] / 1000.0)
    ax2.set_xlabel("Online Env Steps (k)")
    ax2.grid(False)

    # (b) Val Jmin and J05
    ax = axes[0, 1]
    for arr, lbl, col in (
        (val_j05, "Q05 Scaled Jacobian", "#2a9d8f"),
        (val_jmin, "Min Scaled Jacobian", "#1b6ca8"),
    ):
        mu, sd = arr.mean(axis=0), arr.std(axis=0, ddof=1)
        ax.plot(epochs, mu, lw=2.2, color=col, label=f"{lbl} ({mu[-1]:.4f})")
        ax.fill_between(epochs, mu - sd, mu + sd, color=col, alpha=0.20)
    b0_j05 = 0.5 * (base["B0_Default"]["val"]["1536"]["j05_mean"] + base["B0_Default"]["val"]["5120"]["j05_mean"])
    b0_jmin = 0.5 * (base["B0_Default"]["val"]["1536"]["jmin_mean"] + base["B0_Default"]["val"]["5120"]["jmin_mean"])
    ax.axhline(b0_j05, ls="--", color="#2a9d8f", alpha=0.75, label=f"Default J05 ({b0_j05:.4f})")
    ax.axhline(b0_jmin, ls="--", color="#1b6ca8", alpha=0.75, label=f"Default Jmin ({b0_jmin:.4f})")
    ax.set_title("(b) Validation Mesh Jacobian Progression (Mean ± SD)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Scaled Jacobian")
    ax.legend(loc="lower right", frameon=True)

    # (c) Farfield AR suppression
    ax = axes[1, 0]
    mu, sd = val_far.mean(axis=0), val_far.std(axis=0, ddof=1)
    ax.plot(epochs, mu, lw=2.2, color="#d62828", label=f"CH-SAC v2 farAR ({mu[-1]:.1f})")
    ax.fill_between(epochs, mu - sd, mu + sd, color="#d62828", alpha=0.18)
    b0_far = 0.5 * (base["B0_Default"]["val"]["1536"]["far_ar_mean"] + base["B0_Default"]["val"]["5120"]["far_ar_mean"])
    ax.axhline(b0_far, ls="--", color="#6c757d", label=f"Step 1A Default farAR ({b0_far:.1f})")
    ax.axhline(150.0, ls=":", color="#1b6ca8", label="Target farAR threshold (150)")
    ax.set_title("(c) Farfield Max Aspect Ratio Control (BL Preserved)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Farfield Max Aspect Ratio")
    ax.legend(loc="upper right", frameon=True)

    # (d) Critic loss & Alpha
    ax = axes[1, 1]
    mu_c, sd_c = closs.mean(axis=0), closs.std(axis=0, ddof=1)
    ax.plot(ep_nz, mu_c, lw=2.0, color="#264653", label="Critic MSE Loss")
    ax.fill_between(ep_nz, mu_c - sd_c, mu_c + sd_c, color="#264653", alpha=0.18)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Critic Loss", color="#264653")
    ax_r = ax.twinx()
    mu_a, sd_a = alpha.mean(axis=0), alpha.std(axis=0, ddof=1)
    ax_r.plot(ep_nz, mu_a, lw=2.0, ls="-.", color="#e76f51", label="Entropy Alpha")
    ax_r.fill_between(ep_nz, mu_a - sd_a, mu_a + sd_a, color="#e76f51", alpha=0.16)
    ax_r.set_ylabel("Temperature alpha", color="#e76f51")
    ax_r.grid(False)
    ax.set_title("(d) SAC Optimization Stability (5 Seeds, 24k Steps)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)

    fig.suptitle("CH-SAC v2: Regular 60-Epoch GPU Training Curves on RTX 5070 Ti (5 Seeds)", fontsize=13, fontweight="bold")
    fig.savefig(OUT / "fig1_training_curves.png", dpi=300)
    plt.close(fig)


def plot_fig2_benchmarks():
    ev = json.loads((OUT / "eval_summary.json").read_text(encoding="utf-8"))
    per = json.loads((OUT / "eval_per_case.json").read_text(encoding="utf-8"))
    base_pc = json.loads((OUT / "baselines_per_case.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), constrained_layout=True)

    # (a) NACA-24 @ 1536
    ax = axes[0, 0]
    m_names = ["Old 13-blk C-H", "B0 Default", "B1 GlobalOpt", "B2 LocalSearch", "B3 SupMLP", "CH-SAC v2"]
    jmin_15 = [
        ev["legacy_ch_naca24"]["1536"]["jmin_mean"],
        ev["baselines"]["B0_Default"]["test_naca24"]["1536"]["jmin_mean"],
        ev["baselines"]["B1_GlobalOpt"]["test_naca24"]["1536"]["jmin_mean"],
        ev["baselines"]["B2_LocalSearch"]["test_naca24"]["1536"]["jmin_mean"],
        ev["baselines"]["B3_SupervisedMLP"]["test_naca24"]["1536"]["jmin_mean"],
        ev["ch_sac_v2_multi_seed"]["test_naca24"]["1536"]["jmin_mean"]["mean"],
    ]
    j05_15 = [
        ev["legacy_ch_naca24"]["1536"]["j05_mean"],
        ev["baselines"]["B0_Default"]["test_naca24"]["1536"]["j05_mean"],
        ev["baselines"]["B1_GlobalOpt"]["test_naca24"]["1536"]["j05_mean"],
        ev["baselines"]["B2_LocalSearch"]["test_naca24"]["1536"]["j05_mean"],
        ev["baselines"]["B3_SupervisedMLP"]["test_naca24"]["1536"]["j05_mean"],
        ev["ch_sac_v2_multi_seed"]["test_naca24"]["1536"]["j05_mean"]["mean"],
    ]
    x = np.arange(len(m_names))
    w = 0.36
    b1 = ax.bar(x - w / 2, jmin_15, w, color="#1b6ca8", label="Min Scaled Jacobian")
    b2 = ax.bar(x + w / 2, j05_15, w, color="#2a9d8f", label="Q05 Scaled Jacobian")
    ax.set_xticks(x)
    ax.set_xticklabels(m_names, rotation=22, ha="right")
    ax.set_ylim(0.55, 1.00)
    ax.set_ylabel("Scaled Jacobian")
    ax.set_title("(a) NACA-24 Airfoils (1,536 Quads)")
    ax.legend(loc="lower right")
    for bar in (b1[-1], b2[-1]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.006, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # (b) NACA-24 @ 5120
    ax = axes[0, 1]
    m_51 = ["Old 13-blk C-H", "B0 Default", "B1 GlobalOpt", "B2 LocalSearch", "B3 SupMLP", "CH-SAC v2"]
    jmin_51 = [
        ev["legacy_ch_naca24"]["5120"]["jmin_mean"],
        ev["baselines"]["B0_Default"]["test_naca24"]["5120"]["jmin_mean"],
        ev["baselines"]["B1_GlobalOpt"]["test_naca24"]["5120"]["jmin_mean"],
        ev["baselines"]["B2_LocalSearch"]["test_naca24"]["5120"]["jmin_mean"],
        ev["baselines"]["B3_SupervisedMLP"]["test_naca24"]["5120"]["jmin_mean"],
        ev["ch_sac_v2_multi_seed"]["test_naca24"]["5120"]["jmin_mean"]["mean"],
    ]
    j05_51 = [
        ev["legacy_ch_naca24"]["5120"]["j05_mean"],
        ev["baselines"]["B0_Default"]["test_naca24"]["5120"]["j05_mean"],
        ev["baselines"]["B1_GlobalOpt"]["test_naca24"]["5120"]["j05_mean"],
        ev["baselines"]["B2_LocalSearch"]["test_naca24"]["5120"]["j05_mean"],
        ev["baselines"]["B3_SupervisedMLP"]["test_naca24"]["5120"]["j05_mean"],
        ev["ch_sac_v2_multi_seed"]["test_naca24"]["5120"]["j05_mean"]["mean"],
    ]
    x2 = np.arange(len(m_51))
    b1 = ax.bar(x2 - w / 2, jmin_51, w, color="#1b6ca8", label="Min Scaled Jacobian")
    b2 = ax.bar(x2 + w / 2, j05_51, w, color="#2a9d8f", label="Q05 Scaled Jacobian")
    ax.set_xticks(x2)
    ax.set_xticklabels(m_51, rotation=22, ha="right")
    ax.set_ylim(0.45, 1.00)
    ax.set_ylabel("Scaled Jacobian")
    ax.set_title("(b) NACA-24 Airfoils (5,120 Quads)")
    ax.legend(loc="lower right")
    for bar in (b1[-1], b2[-1]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.006, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # (c) Held-out UIUC Test-64 (1536 & 5120)
    ax = axes[1, 0]
    u_names = ["B0 Default", "B1 GlobalOpt", "B2 LocalSearch", "B3 SupMLP", "CH-SAC v2"]
    sc_15 = [
        ev["baselines"]["B0_Default"]["test_uiuc"]["1536"]["score_mean"],
        ev["baselines"]["B1_GlobalOpt"]["test_uiuc"]["1536"]["score_mean"],
        ev["baselines"]["B2_LocalSearch"]["test_uiuc"]["1536"]["score_mean"],
        ev["baselines"]["B3_SupervisedMLP"]["test_uiuc"]["1536"]["score_mean"],
        ev["ch_sac_v2_multi_seed"]["test_uiuc"]["1536"]["score_mean"]["mean"],
    ]
    sc_51 = [
        ev["baselines"]["B0_Default"]["test_uiuc"]["5120"]["score_mean"],
        ev["baselines"]["B1_GlobalOpt"]["test_uiuc"]["5120"]["score_mean"],
        ev["baselines"]["B2_LocalSearch"]["test_uiuc"]["5120"]["score_mean"],
        ev["baselines"]["B3_SupervisedMLP"]["test_uiuc"]["5120"]["score_mean"],
        ev["ch_sac_v2_multi_seed"]["test_uiuc"]["5120"]["score_mean"]["mean"],
    ]
    x3 = np.arange(len(u_names))
    b1 = ax.bar(x3 - w / 2, sc_15, w, color="#2a9d8f", label="Coarse (nominal 1,536)")
    b2 = ax.bar(x3 + w / 2, sc_51, w, color="#e76f51", label="5,120 Quads")
    ax.set_xticks(x3)
    ax.set_xticklabels(u_names, rotation=20, ha="right")
    ax.set_ylim(0.55, 0.98)
    ax.set_ylabel("Composite Score S(G)")
    ax.set_title("(c) Held-Out UIUC Test-64 Airfoils (Sharp + Blunt TE)")
    ax.legend(loc="lower right")
    for bar in (b1[-1], b2[-1]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # (d) Paired scatter on UIUC-64 + NACA-24: B0 Default vs CH-SAC v2 (5-seed mean) Jmin
    ax = axes[1, 1]
    b0_uiuc = {r["id"]: r["metrics"]["min_scaled_jacobian"] for r in base_pc["B0_Default"]["test_uiuc"]}
    v2_uiuc = per["ensemble_mean"]["test_uiuc"]
    for budget, col, mk in ((1536, "#2a9d8f", "o"), (5120, "#e76f51", "^")):
        xs = [b0_uiuc[r["id"]] for r in v2_uiuc if r["budget"] == budget]
        ys = [r["metrics"]["min_scaled_jacobian"] for r in v2_uiuc if r["budget"] == budget]
        ax.scatter(xs, ys, c=col, marker=mk, alpha=0.72, edgecolors="k", linewidths=0.4, s=36, label="UIUC-64 (coarse)" if budget == 1536 else "UIUC-64 (5,120)")
    b0_naca = {r["id"]: r["metrics"]["min_scaled_jacobian"] for r in base_pc["B0_Default"]["test_naca24"]}
    v2_naca = per["ensemble_mean"]["test_naca24"]
    ax.scatter(
        [b0_naca[r["id"]] for r in v2_naca],
        [r["metrics"]["min_scaled_jacobian"] for r in v2_naca],
        c="#1b6ca8",
        marker="s",
        alpha=0.82,
        edgecolors="k",
        linewidths=0.5,
        s=36,
        label="NACA-24 (both budgets)",
    )
    lo, hi = 0.40, 0.96
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="y = x (no change)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Step 1A Default Jmin")
    ax.set_ylabel("CH-SAC v2 (5-seed mean) Jmin")
    ax.set_title("(d) Paired Comparison of Min Scaled Jacobian")
    ax.legend(loc="lower right", frameon=True)

    fig.suptitle("CH-SAC v2 Held-Out Benchmarks Against Legacy C-H and Non-RL Baselines", fontsize=13, fontweight="bold")
    fig.savefig(OUT / "fig2_benchmark_comparison.png", dpi=300)
    plt.close(fig)


def plot_fig3_mesh_and_cfd():
    cfd = json.loads((OUT / "cfd_verification_summary.json").read_text(encoding="utf-8"))
    per = json.loads((OUT / "eval_per_case.json").read_text(encoding="utf-8"))
    med_seed = str(cfd["median_val_seed"])
    med_map = {r["id"]: r for r in per["seed_evals"]["test_naca24"][med_seed]}
    naca_items = {it["id"]: it for it in load_cached_split(OUT / "cache/test_naca24_g0.pkl.gz")}

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), constrained_layout=True)

    # (a) NACA 6423 (5120) near-TE & wake zoom wireframe comparison (split plot: left Default, right V2)
    ax = axes[0, 0]
    it6423 = naca_items["naca6423_5120"]
    g_def = apply_7controls(it6423["g0"], u_to_params(DEFAULT_U))
    g_v2 = apply_7controls(it6423["g0"], u_to_params(med_map["naca6423_5120"]["u_final"]))
    m_def = generate(g_def)
    m_v2 = generate(g_v2)

    # Plot airfoil + wake edges near TE [0.75, 2.2] x [-0.55, 0.55]
    for m, col, alpha, lw, lbl in (
        (m_def, "#999999", 0.55, 0.55, f"Default (Jmin={m_def.quality['min_scaled_jacobian']:.3f})"),
        (m_v2, "#1b6ca8", 0.78, 0.70, f"CH-SAC v2 (Jmin={m_v2.quality['min_scaled_jacobian']:.3f})"),
    ):
        pts = np.asarray(m.points)
        # downsample block grid lines every 2nd index in dense wake
        for blk_grid in m.grids.values():
            for row in blk_grid[::2, :]:
                xy = pts[row]
                ax.plot(xy[:, 0], xy[:, 1], color=col, alpha=alpha, lw=lw)
            for col_idx in range(0, blk_grid.shape[1], 2):
                xy = pts[blk_grid[:, col_idx]]
                ax.plot(xy[:, 0], xy[:, 1], color=col, alpha=alpha, lw=lw)
        ax.plot([], [], color=col, lw=1.6, label=lbl)
    ax.set_xlim(0.65, 2.10)
    ax.set_ylim(-0.45, 0.45)
    ax.set_aspect("equal")
    ax.set_title("(a) NACA 6423 (5,120) TE & Near-Wake Grid Adaptation")
    ax.set_xlabel("x / c")
    ax.set_ylabel("y / c")
    ax.legend(loc="upper right", frameon=True)

    # (b) NACA 0012 (5120) L2Roe Upper Cp vs Gregory & O'Reilly
    ax = axes[0, 1]
    obs = np.array([(float(r["x_over_c"]), float(r["cp"])) for r in read_rows(EXP_DATA / "naca0012_gregory_cp.csv") if r["alpha_deg"] == "10"])
    obs = obs[obs[:, 0].argsort()]
    ax.scatter(obs[:, 0], obs[:, 1], s=26, color="#111111", zorder=5, label="Gregory & O'Reilly Exp (alpha=10)")
    for label, path, col, ls in (
        ("CH-SAC v2 (5,120 L2Roe)", OUT / "cfd_verification/naca0012_v2_5120/l2roe_surface.csv", "#1b6ca8", "-"),
        ("Legacy 13-blk C-H (5,120 L2Roe)", OUT / "reference/legacy_0012_5120_l2roe_surface.csv", "#e76f51", "--"),
    ):
        if path.is_file():
            (x, cp), _ = pressure(path)
            ax.plot(x, cp, color=col, ls=ls, lw=1.9, label=label)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("x / c")
    ax.set_ylabel("Cp")
    ax.set_title("(b) NACA 0012 (5,120) L2Roe Upper Surface Cp")
    ax.legend(loc="lower right", frameon=True)

    # (c) L2Roe Residual Convergence across NACA 0012, 4415, 6423
    ax = axes[1, 0]
    for cname, col, lbl in (
        ("naca0012_v2_5120", "#1b6ca8", "NACA 0012 CH-SAC v2"),
        ("naca4415_v2_5120", "#2a9d8f", "NACA 4415 CH-SAC v2"),
        ("naca6423_v2_5120", "#e76f51", "NACA 6423 CH-SAC v2"),
        ("naca6423_def_5120", "#6c757d", "NACA 6423 Default"),
    ):
        hist_p = OUT / "cfd_verification" / cname / "l2roe_history.csv"
        if hist_p.is_file():
            rows = read_rows(hist_p)
            rms = np.array([float(r["rms[Rho]"]) for r in rows], dtype=float)
            it = np.arange(1, len(rms) + 1)
            ax.plot(it, rms, color=col, lw=1.7, label=f"{lbl} ({rms[-1]:.2f})")
    ax.axhline(-9.0, ls=":", color="black", lw=1.1, label="Density residual threshold (-9.0)")
    ax.set_xlabel("L2Roe Iteration")
    ax.set_ylabel("log10 rms[Rho]")
    ax.set_title("(c) SU2 8.6 Steady RANS-SA (L2Roe) Residual History")
    ax.legend(loc="upper right", frameon=True)

    # (d) Compact CFD force & mesh table summary across 5120 cases
    ax = axes[1, 1]
    ax.axis("off")
    col_labels = ["Case (5,120)", "Jmin", "farAR", "L2Roe rms", "CL", "CD"]
    rows_txt = []
    for cname, disp in (
        ("naca0012_v2_5120", "0012 CH-SAC v2"),
        ("naca4415_def_5120", "4415 Default"),
        ("naca4415_v2_5120", "4415 CH-SAC v2"),
        ("naca6423_def_5120", "6423 Default"),
        ("naca6423_v2_5120", "6423 CH-SAC v2"),
    ):
        c = cfd["cases"][cname]
        st = c["stages"]["l2roe"]
        rows_txt.append(
            [
                disp if st["joint_converged"] else f"{disp} *",
                f"{c['mesh']['min_scaled_jacobian']:.4f}",
                f"{c['mesh']['farfield_max_aspect_ratio']:.1f}",
                f"{st['final_log10_rms_density']:.2f}",
                f"{st['cl']:.4f}",
                f"{st['cd']:.5f}",
            ]
        )
    tbl = ax.table(
        cellText=rows_txt,
        colLabels=col_labels,
        colWidths=[0.25, 0.13, 0.13, 0.17, 0.15, 0.17],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.0)
    tbl.scale(1.0, 1.9)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#264653")
            cell.set_text_props(color="white", fontweight="bold")
        elif "CH-SAC v2" in rows_txt[r - 1][0]:
            cell.set_facecolor("#eef6fb")
    ax.text(0.0, 0.14, "* Joint convergence not reached. Forces at the iteration limit.",
            transform=ax.transAxes, fontsize=8, va="top")
    ax.set_title("(d) Mesh Quality vs SU2 8.6 RANS-SA (L2Roe, alpha=10, Re=6M)")

    fig.suptitle("CH-SAC v2 Mesh Geometry & SU2 8.6 L2Roe Aerodynamic Verification", fontsize=13, fontweight="bold")
    fig.savefig(OUT / "fig3_mesh_and_cfd.png", dpi=300)
    plt.close(fig)


def main():
    plot_fig1_training()
    plot_fig2_benchmarks()
    plot_fig3_mesh_and_cfd()
    print("Generated all 3 proposal figures.", flush=True)


if __name__ == "__main__":
    main()
