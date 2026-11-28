"""CLI for isolated, reproducible BC and discrete-SAC experiments."""
from __future__ import annotations
import argparse
import json


def main(task=None, argv=None):
    parser = argparse.ArgumentParser(description="Local constrained C-domain BC / masked discrete-SAC prototype")
    if task is None:
        parser.add_argument("task", choices=("prepare", "mesh", "bc", "train", "evaluate", "cfd", "visualize",
                                             'audit-meshes', 'audit-demos', 'import-reviews'))
    parser.add_argument("--config", default="configs/rl.yaml")
    parser.add_argument("--output", help="New output directory; existing runs are never overwritten")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--limit", type=int, help="Maximum eligible geometries")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--critic-warmup", type=int, help="Critic-only updates after BC; actor remains frozen")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--checkpoint")
    parser.add_argument('--demo-package', help='Imported review package; approved train and audited val only')
    parser.add_argument('--input', nargs='+', help='Demonstration JSONL files to audit')
    parser.add_argument('--migrate-environment', action='store_true', help='Explicitly replay legacy mesh-v1 trajectories under the current terminal contract; old reviews are never carried over')
    parser.add_argument('--splits', nargs='+', choices=('train', 'val'), default=['train', 'val'], help='Mesh audit splits; test is forbidden')
    parser.add_argument('--variants', nargs='+', type=int, choices=(0, 1), default=[0, 1], help='Rule variants for mesh audit')
    parser.add_argument('--package', help='Audited demonstration package directory for review import')
    parser.add_argument('--decisions', help='Human-exported review JSON; no approval is generated automatically')
    parser.add_argument("--method", choices=("rule", "bc", "sac", "bc_sac"))
    parser.add_argument('--final-test', action='store_true', help='Explicitly freeze a final test evaluation snapshot')
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-review", action="store_true", help="Skip generating human-review drawings, not mark them approved")
    parser.add_argument("--mesh", help="Exported mesh.su2 for CFD; sample directory or mesh.npz for visualization")
    parser.add_argument("--solver", help="SU2_CFD executable")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args(argv)
    task = task or args.task
    args.task = task
    if args.method is None:
        args.method = 'rule' if task in ('mesh', 'evaluate') else ('bc' if task == 'bc' else 'bc_sac')
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    for name in ('steps', 'warmup', 'critic_warmup'):
        if getattr(args, name) is not None and getattr(args, name) < 0:
            parser.error(f'--{name.replace("_", "-")} must be nonnegative')
    for name in ('epochs', 'episodes', 'iterations'):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f'--{name} must be positive')
    if task == 'audit-demos' and not args.input:
        parser.error('audit-demos requires --input')
    if task == 'import-reviews' and not (args.package and args.decisions):
        parser.error('import-reviews requires --package and --decisions')
    from smartcfd.pipeline.config import read_config
    from smartcfd.pipeline import workflow
    config = read_config(args.config)
    functions = {"prepare": workflow.prepare, "mesh": workflow.generate,
                 "bc": lambda c,a: workflow.train(c,a,bc_only=True), "train": workflow.train,
                 "evaluate": workflow.evaluate}
    if task in ("cfd", "visualize"):
        from smartcfd.pipeline.validation import run_cfd, visualize
        functions.update(cfd=run_cfd, visualize=visualize)
    elif task == 'audit-meshes':
        from smartcfd.pipeline.mesh_audit import audit_meshes
        functions[task] = audit_meshes
    elif task == 'audit-demos':
        from smartcfd.pipeline.demo_audit import audit_demos
        functions[task] = audit_demos
    elif task == 'import-reviews':
        from smartcfd.pipeline.review import import_reviews
        functions[task] = import_reviews
    run, result = functions[task](config, args)
    summary = {key: value for key, value in result.items() if key not in (
        "records", "episodes", "online", 'groups', 'coarse_failed_fine_passed', 'coarse_passed_fine_failed')}
    print(json.dumps({"run_directory": str(run), "result": summary}, ensure_ascii=False, indent=2))
