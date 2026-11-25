"""Run a steady, two-dimensional Euler case on an exported NACA 0012 mesh."""

import argparse
import csv
from collections import deque
import json
import math
from pathlib import Path
import shutil
import subprocess


RESIDUAL_TARGET = -7.0

# Boundary conditions and numerics follow the SU2 Euler regression case.
# https://github.com/su2code/SU2/blob/master/TestCases/euler/naca0012/inv_NACA0012.cfg
CONFIG = """% NACA 0012 subsonic Euler smoke case
SOLVER= EULER
MATH_PROBLEM= DIRECT
RESTART_SOL= NO
MACH_NUMBER= {mach:.12g}
AOA= {aoa:.12g}
FREESTREAM_PRESSURE= 101325.0
FREESTREAM_TEMPERATURE= 288.15
REF_ORIGIN_MOMENT_X= 0.25
REF_ORIGIN_MOMENT_Y= 0.0
REF_ORIGIN_MOMENT_Z= 0.0
REF_LENGTH= 1.0
REF_AREA= 1.0
REF_DIMENSIONALIZATION= FREESTREAM_PRESS_EQ_ONE
MARKER_EULER= (airfoil)
MARKER_FAR= (farfield)
MARKER_PLOTTING= (airfoil)
MARKER_MONITORING= (airfoil)
NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES
CONV_NUM_METHOD_FLOW= JST
JST_SENSOR_COEFF= (0.5, 0.02)
TIME_DISCRE_FLOW= EULER_IMPLICIT
CFL_NUMBER= {cfl:.12g}
CFL_ADAPT= NO
MGLEVEL= 0
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= LU_SGS
LINEAR_SOLVER_ERROR= 1e-5
LINEAR_SOLVER_ITER= 20
ITER= {iterations}
CONV_FIELD= RMS_DENSITY
CONV_STARTITER= 10
CONV_RESIDUAL_MINVAL= {target}
MESH_FORMAT= SU2
MESH_FILENAME= mesh.su2
TABULAR_FORMAT= CSV
CONV_FILENAME= history
VOLUME_FILENAME= flow
SURFACE_FILENAME= surface_flow
HISTORY_OUTPUT= (ITER, RMS_RES, AERO_COEFF)
SCREEN_OUTPUT= (INNER_ITER, RMS_DENSITY, RMS_ENERGY, LIFT, DRAG)
SCREEN_WRT_FREQ_INNER= 25
OUTPUT_FILES= (PARAVIEW, SURFACE_CSV)
OUTPUT_WRT_FREQ= ({iterations}, {iterations})
"""


def check_mesh_markers(mesh: Path) -> None:
    """Check that the input has the two boundary names used in the config."""
    dimensions = None
    markers = []
    with mesh.open(encoding="utf-8-sig") as source:
        for line in source:
            key, _, value = line.split("%", 1)[0].partition("=")
            if key.strip() == "NDIME":
                dimensions = value.strip()
            if key.strip() == "MARKER_TAG":
                markers.append(value.strip())
    if dimensions != "2" or sorted(markers) != ["airfoil", "farfield"]:
        raise ValueError("Expected a 2D SU2 mesh with airfoil and farfield markers")


def read_history(path: Path, target=RESIDUAL_TARGET) -> dict[str, int | float | bool]:
    """Report the final lift and drag and density residual reduction in decades."""
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = csv.DictReader(source, skipinitialspace=True)
        rows.fieldnames = [name.strip() for name in (rows.fieldnames or [])]
        columns = ("Inner_Iter", "rms[Rho]", "CL", "CD")
        if not set(columns).issubset(rows.fieldnames):
            raise ValueError(f"Missing expected SU2 history columns in {path}")
        first_residual = None
        count = 0
        lifts, drags = deque(maxlen=50), deque(maxlen=50)
        for count, row in enumerate(rows, start=1):
            iteration, residual, lift, drag = (float(row[column]) for column in columns)
            if not all(map(math.isfinite, (iteration, residual, lift, drag))):
                raise ValueError(f"Non-finite values in SU2 history data row {count}")
            if first_residual is None:
                first_residual = residual
            lifts.append(lift)
            drags.append(drag)
    if first_residual is None:
        raise ValueError(f"No iteration records in {path}")
    return {
        "history_rows": count,
        "last_inner_iteration": int(iteration),
        "initial_log10_rms_density": first_residual,
        "final_log10_rms_density": residual,
        "rms_density_reduction_decades": first_residual - residual,
        "rms_density_target": target,
        "rms_density_target_met": residual <= target,
        "force_window_rows": len(lifts),
        "cl_window_range": max(lifts) - min(lifts),
        "cd_window_range": max(drags) - min(drags),
        "cl": lift,
        "cd": drag,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True, help="Exported SU2 mesh")
    parser.add_argument("--out-dir", type=Path, required=True, help="Dedicated case directory")
    parser.add_argument("--su2", type=Path, help="SU2_CFD executable (otherwise search PATH)")
    parser.add_argument("--mach", type=float, default=0.3)
    parser.add_argument("--aoa", type=float, default=2.0, help="Angle of attack in degrees")
    parser.add_argument("--cfl", type=float, default=8.0)
    parser.add_argument("--residual-target", type=float, default=RESIDUAL_TARGET,
                        help="Stop when log10 of the density RMS reaches this value")
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--prepare-only", action="store_true", help="Write the mesh and config without launching SU2")
    args = parser.parse_args()
    if not 0.0 < args.mach < 1.0:
        parser.error("--mach must be strictly between 0 and 1")
    if not math.isfinite(args.aoa):
        parser.error("--aoa must be finite")
    if not math.isfinite(args.cfl) or args.cfl <= 0.0:
        parser.error("--cfl must be finite and positive")
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if not math.isfinite(args.residual_target) or args.residual_target >= 0:
        parser.error("--residual-target must be finite and negative")

    mesh = args.mesh.resolve()
    out_dir = args.out_dir.resolve()
    staged_mesh = out_dir / "mesh.su2"
    config = out_dir / "euler.cfg"
    log = out_dir / "su2.log"

    try:
        check_mesh_markers(mesh)
        out_dir.mkdir(parents=True, exist_ok=True)
        if set(out_dir.iterdir()) - {staged_mesh, config}:
            raise ValueError(f"Use a separate case directory for a new run; {out_dir} has existing outputs")
        if mesh != staged_mesh:
            shutil.copyfile(mesh, staged_mesh)
        config.write_text(
            CONFIG.format(mach=args.mach, aoa=args.aoa, cfl=args.cfl,
                          iterations=args.iterations, target=args.residual_target),
            encoding="ascii",
        )
        if args.prepare_only:
            print(f"Prepared {config}")
            return

        executable = str(args.su2.resolve()) if args.su2 else shutil.which("SU2_CFD")
        if not executable or not Path(executable).is_file():
            raise ValueError("SU2_CFD was not found. Supply its executable path with --su2")
        print(f"Running SU2 in {out_dir}. Log: {log}", flush=True)
        with log.open("xb") as output:
            completed = subprocess.run([executable, config.name], cwd=out_dir,
                                       stdout=output, stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            raise ValueError(f"SU2_CFD exited with code {completed.returncode}. See {log}")

        summary = {"mach": args.mach, "aoa_degrees": args.aoa, "cfl": args.cfl,
                   "iteration_limit": args.iterations, **read_history(out_dir / "history.csv", args.residual_target)}
        text = json.dumps(summary, indent=2, allow_nan=False)
        (out_dir / "summary.json").write_text(text + "\n", encoding="ascii")
        print(text)
    except (OSError, TypeError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
