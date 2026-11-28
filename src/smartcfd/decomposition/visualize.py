"""Polygon/cut inspection figures, with no CFD mesh or grid implied."""
from __future__ import annotations

from pathlib import Path
import numpy as np


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _vertices(value):
    return np.asarray(value.get('vertices', value.get('polygon')) if isinstance(value, dict) else value, dtype=float)


def _ring(ax, vertices, **kwargs):
    points = _vertices(vertices)
    if len(points):
        loop = np.vstack([points, points[0]])
        ax.plot(loop[:, 0], loop[:, 1], **kwargs)
    return points


def plot_partition(case, outcome, path, title=''):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 5))
    original = _ring(ax, case['vertices'], color='#263b52', linewidth=2., label='Input boundary')
    indexed = original[:-1] if len(original) > 1 and np.allclose(original[0], original[-1]) else original
    for index, point in enumerate(indexed):
        ax.annotate(str(index), point, textcoords='offset points', xytext=(4, 4), fontsize=8, color='#263b52')
    partition = outcome['partition']
    for index, block in enumerate(partition.get('blocks', [])):
        points = _vertices(block)
        if len(points):
            ax.fill(points[:, 0], points[:, 1], alpha=.18, color=plt.get_cmap('tab20')(index % 20))
            _ring(ax, points, color='#52796f', linewidth=.8)
    for pending in partition.get('pending', []):
        _ring(ax, pending, color='#c68b28', linewidth=1., linestyle='--')
    for index, cut in enumerate(partition.get('cuts', [])):
        if isinstance(cut, dict):
            segments = cut.get('segments')
            if segments is None:
                points = cut.get('segment', cut.get('endpoints'))
                if points is None and 'start' in cut and 'end' in cut:
                    points = [cut['start'], cut['end']]
                segments = [points] if points is not None else None
        else:
            segments = [cut]
        if not segments:
            raise ValueError('A recorded cut needs at least one endpoint-pair segment')
        for points in segments:
            points = np.asarray(points, dtype=float)
            if points.shape != (2, 2):
                raise ValueError('Cut segment must have two two-dimensional endpoints')
            ax.plot(points[:, 0], points[:, 1], color='#b44b38', linewidth=1.5)
            center = points.mean(axis=0)
            ax.annotate(f'c{index}', center, fontsize=8, color='#b44b38')
    quality = partition.get('quality', {})
    details = f"blocks={quality.get('block_count')}  aspect_max={quality.get('aspect_ratio_max', float('nan')):.2f}"
    details += f"  coverage_error={quality.get('relative_coverage_error', float('nan')):.1e}"
    ax.set_title(f"{title} · {case['case_id']}\nsuccess={outcome['success']} steps={outcome['steps']} return={outcome['return']:.3f}\n"
                 f"{details}", fontsize=9)
    ax.set_aspect('equal'); ax.margins(.12)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    fig.tight_layout(); fig.savefig(Path(path), dpi=140); plt.close(fig)


def plot_learning(history, updates, path):
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    x = [row['online_step'] for row in history]
    for ax, key, title in ((axes[0, 0], 'success_rate', 'Fixed validation success rate'),
                           (axes[0, 1], 'mean_return', 'Fixed validation mean return'),
                           (axes[1, 0], 'mean_steps', 'Fixed validation mean cut steps')):
        ax.plot(x, [row[key] for row in history], marker='o', color='#255d83')
        ax.set(title=title, xlabel='Real online interactions'); ax.grid(alpha=.2)
    axes[0, 0].set_ylim(-.03, 1.03)
    if updates:
        for key, color in (('q_loss', '#b8752f'), ('actor_loss', '#34775b')):
            values = np.asarray([row[key] for row in updates], dtype=float)
            window = min(50, len(values))
            smooth = np.convolve(values, np.ones(window)/window, mode='valid')
            axes[1, 1].plot([row['online_step'] for row in updates][window-1:], smooth,
                             label=f'{key} ({window}-update mean)', color=color,
                             marker='o' if len(smooth) == 1 else None)
        axes[1, 1].legend(fontsize=8)
    axes[1, 1].set(title='SAC loss components', xlabel='Real online interactions', yscale='symlog')
    axes[1, 1].grid(alpha=.2)
    fig.suptitle('Orthogonal-polygon MVP: shared validation, not held-out test; not SplineCNN')
    fig.savefig(Path(path), dpi=150); plt.close(fig)
