"""Run a new local C-constrained airfoil decomposition development experiment."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'configs/airfoil_decomposition.yaml'))
    parser.add_argument('--output', required=True, help='New directory inside this copy experiments/runs')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    from smartcfd.airfoil_decomposition.experiment import run_experiment
    output, metrics = run_experiment(args.config, args.output, steps=args.steps, seed=args.seed, argv=sys.argv)
    print(json.dumps({'output': str(output), 'steps': metrics['online']['steps'],
        'updates': len(metrics['online']['updates']), 'best_online_step': metrics['best_online_step'],
        'evaluations': {k: {'success': v['success_count'], 'total': v['total'],
                           'quality_score': v['mean_terminal_quality_score'], 'mean_steps': v['mean_steps']}
                        for k, v in metrics['evaluations'].items()}}, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
