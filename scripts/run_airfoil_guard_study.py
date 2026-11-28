"""Run the frozen nine-arm candidate-feasibility development study."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'configs/airfoil_guard_study.yaml'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--stage', choices=('all', 'prepare', 'teachers', 'baselines', 'sac', 'bc', 'finalize'), default='all')
    parser.add_argument('--seed', type=int, choices=(42, 43, 44), default=42)
    parser.add_argument('--guard', choices=('off', 'on'), default='off')
    args = parser.parse_args()
    from smartcfd.airfoil_decomposition.study_driver import prepare_study, run_phase
    if args.stage == 'prepare':
        result = {'output': str(prepare_study(args.config, args.output)), 'status': 'prepared'}
    elif args.stage in ('all', 'finalize'):
        from smartcfd.airfoil_decomposition.study_driver import finalize_study
        if args.stage == 'all':
            if not Path(args.output).exists():
                prepare_study(args.config, args.output)
            for stage in ('teachers', 'baselines'):
                run_phase(args.output, stage)
            for seed in (42, 43, 44):
                run_phase(args.output, 'bc', seed=seed)
                for guarded in (False, True):
                    run_phase(args.output, 'sac', seed=seed, guarded=guarded)
        result = finalize_study(args.output)
    else:
        result = run_phase(args.output, args.stage, seed=args.seed, guarded=args.guard == 'on')
    print(json.dumps(result, ensure_ascii=True, default=str), flush=True)


if __name__ == '__main__':
    main()
