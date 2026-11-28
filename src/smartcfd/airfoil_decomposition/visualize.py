"""Plots of actual mapped block boundaries, kept separate from mesh plots."""
from pathlib import Path
import numpy as np


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_partition(sample, outcome, path):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    wall = np.asarray(sample['polygon'], dtype=float)
    part = outcome.get('partition') or {}
    root = part.get('root') or {}
    for ax in axes:
        for index, block in enumerate(part.get('blocks', [])):
            points = np.asarray(block['boundary'], dtype=float)
            if len(points) < 3:
                continue
            color = '#6796b8' if block['protected'] else plt.get_cmap('tab20')(index % 20)
            ax.fill(points[:, 0], points[:, 1], color=color, alpha=.28)
            ax.plot(points[:, 0], points[:, 1], color='#49636a', linewidth=.65)
        if root.get('outer'):
            outer = np.asarray(root['outer'], dtype=float)
            ax.plot(outer[:, 0], outer[:, 1], color='#263b52', linewidth=1.)
        loop = np.vstack([wall, wall[0]])
        ax.fill(loop[:, 0], loop[:, 1], color='white', zorder=5)
        ax.plot(loop[:, 0], loop[:, 1], color='#202f42', linewidth=1.3, zorder=6)
        ax.set_aspect('equal'); ax.set_xlabel('x / chord'); ax.set_ylabel('y / chord')
        ax.grid(alpha=.12)
    axes[0].set_title('Whole C-constrained physical domain')
    axes[1].set(xlim=(-.15, 1.35), ylim=(float(wall[:, 1].min())-.16, float(wall[:, 1].max())+.16),
                title='Airfoil and fixed near-wall envelope')
    q = part.get('quality', {})
    title = f"{outcome['method']} | {sample['sample_id']} | success={outcome['success']} | steps={outcome['steps']}"
    title += f"\nblocks={q.get('block_count', '-')} | Q={q.get('q', 0.):.3f} | {outcome.get('reason', '')}"
    fig.suptitle(title, fontsize=11)
    fig.savefig(Path(path), dpi=140)
    plt.close(fig)


def plot_learning(history, updates, path):
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    x = [r['online_step'] for r in history]
    for ax, key, label in ((axes[0, 0], 'success_rate', 'Validation partition success'),
                           (axes[0, 1], 'mean_terminal_quality_score', 'Terminal quality score (selection metric)'),
                           (axes[1, 0], 'mean_base_return', 'Base task return (without shaping)')):
        ax.plot(x, [r[key] for r in history], marker='o', color='#255d83')
        ax.set(title=label, xlabel='Real online interactions'); ax.grid(alpha=.2)
    axes[0, 0].set_ylim(-.03, 1.03)
    if updates:
        for key, color in (('q_loss', '#b8752f'), ('actor_loss', '#34775b')):
            values = np.array([r[key] for r in updates])
            n = min(50, len(values))
            smooth = np.convolve(values, np.ones(n)/n, mode='valid')
            axes[1, 1].plot([r['online_step'] for r in updates][n-1:], smooth, label=key, color=color)
        axes[1, 1].legend()
    axes[1, 1].set(title='SAC losses (up to 50-update moving mean)', xlabel='Real online interactions', yscale='symlog')
    axes[1, 1].grid(alpha=.2)
    fig.suptitle('Single-seed development validation; prescribed C mapping; no CFD accuracy claim')
    fig.savefig(Path(path), dpi=150)
    plt.close(fig)
