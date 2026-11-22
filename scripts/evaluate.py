"""Compare mesh quality on the frozen 24-case analytic NACA test split."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--naca-2d", action="store_true", help="Evaluate the analytic NACA benchmark")
    parser.add_argument("--checkpoint", type=Path, help="Alternative SAC checkpoint in .npz format")
    parser.add_argument("--output", type=Path, help="New JSON file with per-case results")
    parser.add_argument("--config", type=Path, help="Reserved for the general YAML interface")
    args = parser.parse_args()
    if not args.naca_2d or args.config is not None:
        parser.error("Use --naca-2d. The general YAML workflow is not implemented.")

    from smartcfd.pipeline.workflow import evaluate_naca_holdout

    try:
        report = evaluate_naca_holdout(args.checkpoint)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    summary = {key: value for key, value in report.items() if key != "cases"}
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
