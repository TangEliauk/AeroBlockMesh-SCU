"""Local mesh visualization and explicit external CFD adapter validation."""
from __future__ import annotations
from dataclasses import asdict
import json
import os
from pathlib import Path
import numpy as np
from smartcfd.core.types import Failure, MultiBlockMesh
from smartcfd.pipeline.config import complete_run, new_run, write_json
from smartcfd.pipeline.data import repository_root


def input_path(value):
    if not value:
        raise ValueError('--mesh is required')
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repository_root() / path).resolve()


def run_cfd(config, args):
    from smartcfd.cfd.default_case import write_case_config
    from smartcfd.cfd.solver import run_su2
    mesh_path = input_path(args.mesh)
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    run = new_run('cfd', config, args.output)
    case = config.get('cfd', {})
    solver = args.solver or case.get('executable')
    if not solver:
        bundled = repository_root() / '.runtime/su2_8.6.0/bin/SU2_CFD.exe'
        solver = str(bundled) if bundled.is_file() else 'SU2_CFD'
    old_threads = os.environ.get('OMP_NUM_THREADS')
    os.environ['OMP_NUM_THREADS'] = str(case.get('threads', 2))
    try:
        cfg = write_case_config(run / 'input.cfg', iterations=args.iterations,
            mach=case.get('mach', .15), reynolds=case.get('reynolds', 6e6),
            alpha_deg=case.get('alpha_deg', 0), temp_K=case.get('temperature_K', 300))
        result = run_su2(mesh_path, cfg, executable=solver, output_dir=run,
                         timeout=case.get('timeout_seconds', 300), convergence={})
        if isinstance(result, Failure):
            report = {'success': False, 'failure': asdict(result), 'convergence_validated': False}
        else:
            report = {'success': True, **result}
        report['scope'] = 'CFD adapter smoke; convergence and grid-independent accuracy require separate validation'
        report['solver_executable'] = solver
        write_json(run / 'metrics.json', report)
        complete_run(run, report, status='completed' if report['success'] else 'failed')
        return run, report
    except Exception as exc:
        complete_run(run, {'error': f'{type(exc).__name__}: {exc}'}, status='failed')
        raise
    finally:
        if old_threads is None:
            os.environ.pop('OMP_NUM_THREADS', None)
        else:
            os.environ['OMP_NUM_THREADS'] = old_threads


def visualize(config, args):
    from smartcfd.pipeline.workflow import draw_mesh, draw_topology
    from smartcfd.core.types import BlockGraph
    source = input_path(args.mesh)
    folder = source if source.is_dir() else source.parent
    manifest = json.loads((folder / 'mesh.json').read_text(encoding='utf-8'))
    with np.load(folder / 'mesh.npz', allow_pickle=False) as arrays:
        blocks = {int(key): arrays[f'block_{key}'].copy() for key in manifest['blocks']}
    for key, shape in manifest['blocks'].items():
        if list(blocks[int(key)].shape) != shape or not np.isfinite(blocks[int(key)]).all():
            raise ValueError('Mesh arrays disagree with the manifest or contain nonfinite values')
    mesh = MultiBlockMesh(blocks, manifest['interfaces'], manifest['boundaries'],
                          manifest['quality'], manifest['metadata'])
    run = new_run('visualize', config, args.output)
    draw_mesh(mesh, run)
    graph = manifest['metadata'].get('graph')
    if graph:
        draw_topology(BlockGraph.from_dict(graph), run / 'topology.png')
    result = {'source': str(folder), 'blocks': len(blocks), 'quality': mesh.quality}
    complete_run(run, result)
    return run, result
