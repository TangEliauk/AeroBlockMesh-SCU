"""Prepare, run, resume, evaluate and audit the local airfoil fusion study."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--config", type=Path, default=ROOT/"configs/airfoil_fusion.yaml")
    parser.add_argument("--smoke-evidence", default="experiments/runs/airfoil_fusion_smoke")
    parser.add_argument("--stage", choices=("all", "prepare", "teachers", "bc", "train", "evaluate", "cfd", "report", "audit"),
                        default="all")
    parser.add_argument("--arm", default="E5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--value", default="0", help="Frozen step budget for training or checkpoint tag for evaluation")
    args = parser.parse_args()
    from smartcfd.airfoil_fusion.protocol import open_study, prepare
    from smartcfd.airfoil_fusion.runner import pipeline, teachers, worker

    if args.stage == "all":
        result = pipeline(args.output, args.config, args.profile, args.smoke_evidence)
    elif args.stage == "prepare":
        path, cfg, cases, _ = prepare(args.output, args.config, args.profile)
        result = {"output": str(path), "profile": cfg["profile"], "counts": cases["manifest"]["counts"]}
    elif args.stage in ("bc", "train", "evaluate"):
        result = worker(args.output, args.stage, args.arm, args.seed, args.value)
    else:
        path, cfg, cases, checks = open_study(args.output)
        if args.stage == "teachers":
            result = teachers(path, cfg, cases, checks)
            result = {key: value for key, value in result.items() if key != "rows"}
        elif args.stage == "cfd":
            from smartcfd.airfoil_fusion.cfd import run

            result = run(path, cfg)
        else:
            from smartcfd.airfoil_fusion.report import audit_run, build_report

            result = (audit_run if args.stage == "audit" else build_report)(path, cfg)
    print(json.dumps(result, ensure_ascii=True, default=str), flush=True)
    if result.get("passed") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
