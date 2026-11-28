"""Run the separate orthogonal-polygon decomposition MVP, explicitly CPU only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/decomposition.yaml')
    parser.add_argument('--output', type=Path, required=True, help='New output inside this copy\'s experiments/runs')
    parser.add_argument('--steps', type=int, help='Override interaction budget; no checkpoint resume')
    parser.add_argument('--seed', type=int, help='Override training/evaluation seed; dataset seed remains configured')
    args = parser.parse_args()
    from smartcfd.decomposition.experiment import run_experiment
    output, metrics = run_experiment(args.config, args.output, steps=args.steps, seed=args.seed,
                                     argv=[str(Path(sys.executable).resolve()), str(Path(__file__).resolve()), *sys.argv[1:]])
    print(json.dumps({'output': str(output), 'steps': metrics['online']['steps'],
        'best_selected_at_online_step': metrics['best_selected_at_online_step'],
        'evaluations': metrics['evaluations']}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
