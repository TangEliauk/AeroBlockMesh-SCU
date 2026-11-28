"""Plan or explicitly execute the local rule/BC/SAC/BC+SAC validation suite.

The default prints commands only. --execute creates a new parent run receipt
and launches existing CLIs; it never evaluates the held-out test split.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
METHODS = ("rule", "bc", "sac", "bc_sac")


def build_plan(run_dir, *, smoke=False, device="auto"):
    config_path = "configs/smoke.yaml" if smoke else "configs/rl.yaml"
    jobs, conditions = [], []
    for seed in SEEDS:
        for method in METHODS:
            group = Path(run_dir) / method / f"seed_{seed}"
            conditions.append({"method": method, "seed": seed})
            common = ["--config", config_path, "--seed", str(seed), "--device", device]
            if smoke:
                common += ["--limit", "2"]
            training_id = None
            if method != "rule":
                script = "scripts/pretrain_imitation.py" if method == "bc" else "scripts/train_rl.py"
                training_id = f"{method}:{seed}:train"
                jobs.append({"id": training_id, "method": method, "seed": seed, "stage": "train",
                             "output": str(group / "train"), "depends_on": None,
                             "argv": [sys.executable, script, *common, "--method", method,
                                      "--split", "train", "--output", str(group / "train")]})
            checkpoint = group / "train" / "best.pt" if method != "rule" else None
            argv = [sys.executable, "scripts/evaluate.py", *common, "--method", method,
                    "--split", "val", "--output", str(group / "eval")]
            if checkpoint is not None:
                argv += ["--checkpoint", str(checkpoint)]
            jobs.append({"id": f"{method}:{seed}:eval", "method": method, "seed": seed,
                         "stage": "eval", "output": str(group / "eval"),
                         "depends_on": training_id, "checkpoint": str(checkpoint) if checkpoint else None,
                         "argv": argv})
    return {"kind": "four_method_validation_suite", "mode": "smoke" if smoke else "full_config",
            "config_path": config_path, "parent_output": str(run_dir), "seeds": list(SEEDS),
            "methods": list(METHODS), "conditions": conditions, "condition_count": len(conditions),
            "command_count": len(jobs), "evaluation_split": "val", "test_used": False,
            "formal_results_claimed": False, "jobs": jobs,
            "note": "Rule has no training stage. Learned evaluation uses validation-selected train/best.pt. A plan is not an executed experiment."}


def resolve_output(value):
    allowed = (ROOT / "experiments/runs").resolve()
    if value:
        output = Path(value)
        if not output.is_absolute():
            output = ROOT / output
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        output = allowed / f"experiment_suite_{stamp}"
    output = output.resolve()
    if not output.is_relative_to(allowed) or output == allowed:
        raise ValueError("Suite output must be a new subdirectory of this copy's experiments/runs")
    if output.exists():
        raise FileExistsError("Suite output already exists; runs are never overwritten")
    return output


def show_plan(plan):
    heading = {key: plan[key] for key in ("mode", "config_path", "parent_output", "seeds", "methods",
                                         "condition_count", "command_count", "evaluation_split", "test_used")}
    print("PLAN ONLY; training starts only when --execute is supplied.")
    print(json.dumps(heading, ensure_ascii=False, indent=2))
    render = subprocess.list2cmdline if os.name == "nt" else shlex.join
    for job in plan["jobs"]:
        print(f"\n[{job['id']}]\n{render(job['argv'])}")


def execute_plan(plan, config, *, keep_going=False):
    from smartcfd.pipeline.config import complete_run, new_run, write_json
    run = new_run("experiment_suite", config, plan["parent_output"])
    plan["scheduler_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(run / "suite_plan.json", plan)
    results, succeeded = [], set()
    stopped_early = False
    try:
        for job in plan["jobs"]:
            record = {key: value for key, value in job.items() if key != "argv"}
            record["argv"] = job["argv"]
            if job["depends_on"] is not None and job["depends_on"] not in succeeded:
                record.update(status="blocked", reason="training_stage_failed")
                results.append(record)
                write_json(run / "suite_progress.json", results)
                continue
            checkpoint = job.get("checkpoint")
            if checkpoint and not Path(checkpoint).is_file():
                record.update(status="failed", reason="validation_selected_checkpoint_missing")
                results.append(record)
                write_json(run / "suite_progress.json", results)
                if not keep_going:
                    stopped_early = True
                    break
                continue
            output = Path(job["output"])
            # Logs are siblings of the child output. Creating train/ or eval/
            # here would violate the child CLI's new_run(exist_ok=False).
            output.parent.mkdir(parents=True, exist_ok=True)
            stdout_path = output.parent / f"{job['stage']}.stdout.log"
            stderr_path = output.parent / f"{job['stage']}.stderr.log"
            record.update(status="started", stdout_log=str(stdout_path.relative_to(run)),
                          stderr_log=str(stderr_path.relative_to(run)))
            results.append(record)
            write_json(run / "suite_progress.json", results)
            print(f"RUNNING {job['id']}", flush=True)
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                process = subprocess.run(job["argv"], cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
            record.update(returncode=process.returncode, status="completed" if process.returncode == 0 else "failed")
            if process.returncode == 0:
                succeeded.add(job["id"])
            write_json(run / "suite_progress.json", results)
            if process.returncode != 0 and not keep_going:
                stopped_early = True
                break
        failed = any(result["status"] != "completed" for result in results)
        summary = {"planned_conditions": plan["condition_count"], "planned_commands": plan["command_count"],
                   "completed_commands": len(succeeded), "recorded_commands": len(results),
                   "pending_commands": len(plan["jobs"]) - len(results), "stopped_early": stopped_early,
                   "evaluation_split": "val", "test_used": False, "formal_results_claimed": False,
                   "execution_scope": "Local scheduler completion only; no claim of scientific convergence or full formal study.",
                   "jobs": results}
        complete_run(run, summary, status="failed" if failed or stopped_early else "completed")
        print(json.dumps({"run_directory": str(run), "completed_commands": len(succeeded),
                          "formal_results_claimed": False}, ensure_ascii=False))
        return 1 if failed or stopped_early else 0
    except BaseException as exc:
        complete_run(run, {"error": f"{type(exc).__name__}: {exc}", "jobs": results,
                           "test_used": False, "formal_results_claimed": False}, status="failed")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Use configs/smoke.yaml and --limit 2 for every child CLI")
    parser.add_argument("--execute", action="store_true", help="Explicitly launch all planned local commands")
    parser.add_argument("--output", help="New parent directory inside this copy's experiments/runs")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--keep-going", action="store_true", help="After a failed child, continue independent conditions; dependent evaluations remain blocked")
    args = parser.parse_args(argv)
    # No project imports, output creation or configuration reads occur for help.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / "src"))
    if not ROOT.name.endswith("-实施副本"):
        parser.error("Use the isolated implementation copy; the original repository is not a suite output target")
    try:
        output = resolve_output(args.output)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    plan = build_plan(output, smoke=args.smoke, device=args.device)
    from smartcfd.pipeline.config import read_config
    config = read_config(plan["config_path"])
    show_plan(plan)
    if not args.execute:
        return 0
    return execute_plan(plan, config, keep_going=args.keep_going)


if __name__ == "__main__":
    raise SystemExit(main())
