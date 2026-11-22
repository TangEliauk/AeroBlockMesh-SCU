"""Train the two-dimensional analytic NACA mesh policy."""

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--naca-2d", action="store_true", help="Use the analytic NACA four-block benchmark")
    parser.add_argument("--steps", type=int, default=24_000, help="SAC environment steps")
    parser.add_argument("--seed", type=int, default=1729, help="Training random seed")
    parser.add_argument("--output", type=Path, help="New output directory for checkpoints and training data")
    parser.add_argument("--config", type=Path, help="Reserved for the general YAML interface")
    args = parser.parse_args()
    if not args.naca_2d or args.config is not None:
        parser.error("Use --naca-2d. The general YAML workflow is not implemented.")
    if args.steps < 100 or args.seed < 0:
        parser.error("--steps must be at least 100 and --seed must be non-negative")

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from smartcfd.rl.trainer import train

    output = args.output or Path("experiments/checkpoints") / f"naca-2d-{args.seed}"
    try:
        train(args.steps, args.seed, output)
    except FileExistsError:
        parser.error(f"Choose a new output directory: {output}")
    print(f"Checkpoints and training log saved in {output.resolve()}")


if __name__ == "__main__":
    main()
