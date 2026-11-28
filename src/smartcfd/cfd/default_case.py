"""An explicit SU2 8.6 RANS/SA adapter-smoke configuration.

This conservative first-order setup commissions file/solver integration. It is
not a converged or grid-independent aerodynamic validation configuration.
Reference: https://github.com/su2code/SU2/blob/v8.6.0/config_template.cfg
"""
import math
from pathlib import Path


def write_case_config(path, mesh_name="mesh.su2", iterations=100, mach=.15,
                      reynolds=6e6, alpha_deg=0, temp_K=300):
    if int(iterations)!=iterations or iterations<1:
        raise ValueError("iterations must be a positive integer")
    if not all(math.isfinite(x) for x in (mach,reynolds,alpha_deg,temp_K)):
        raise ValueError("flow conditions must be finite")
    if mach<=0 or reynolds<=0 or temp_K<=0:
        raise ValueError("Mach, Reynolds number and temperature must be positive")
    mesh_name=str(mesh_name)
    if any(character in mesh_name for character in "\r\n%"):
        raise ValueError("invalid mesh filename")
    settings={
        "SOLVER":"RANS", "KIND_TURB_MODEL":"SA", "SA_OPTIONS":"NONE",
        "MATH_PROBLEM":"DIRECT", "RESTART_SOL":"NO", "SYSTEM_MEASUREMENTS":"SI",
        "MACH_NUMBER":f"{mach:.12g}", "AOA":f"{alpha_deg:.12g}", "SIDESLIP_ANGLE":"0.0",
        "INIT_OPTION":"REYNOLDS", "FREESTREAM_OPTION":"TEMPERATURE_FS",
        "FREESTREAM_TEMPERATURE":f"{temp_K:.12g}", "REYNOLDS_NUMBER":f"{reynolds:.12g}",
        "REYNOLDS_LENGTH":"1.0", "FREESTREAM_TURB2LAMVISCRATIO":"3.0",
        "FLUID_MODEL":"STANDARD_AIR", "VISCOSITY_MODEL":"SUTHERLAND",
        "REF_ORIGIN_MOMENT_X":"0.25", "REF_ORIGIN_MOMENT_Y":"0.0", "REF_ORIGIN_MOMENT_Z":"0.0",
        "REF_LENGTH":"1.0", "REF_AREA":"1.0",
        "MARKER_HEATFLUX":"( wall, 0.0 )", "MARKER_FAR":"( farfield )",
        "MARKER_PLOTTING":"( wall )", "MARKER_MONITORING":"( wall )",
        "NUM_METHOD_GRAD":"WEIGHTED_LEAST_SQUARES", "CFL_NUMBER":"0.5", "CFL_ADAPT":"NO",
        "MGLEVEL":"0", "CONV_NUM_METHOD_FLOW":"ROE", "MUSCL_FLOW":"NO",
        "TIME_DISCRE_FLOW":"EULER_IMPLICIT", "CONV_NUM_METHOD_TURB":"SCALAR_UPWIND",
        "MUSCL_TURB":"NO", "TIME_DISCRE_TURB":"EULER_IMPLICIT",
        "LINEAR_SOLVER":"FGMRES", "LINEAR_SOLVER_PREC":"ILU",
        "LINEAR_SOLVER_ERROR":"1E-6", "LINEAR_SOLVER_ITER":"15",
        "ITER":str(int(iterations)), "CONV_FIELD":"RMS_DENSITY",
        "CONV_RESIDUAL_MINVAL":"-8.0", "CONV_STARTITER":"10",
        "MESH_FORMAT":"SU2", "MESH_FILENAME":mesh_name,
        "TABULAR_FORMAT":"CSV", "CONV_FILENAME":"history",
        "HISTORY_OUTPUT":"( ITER, RMS_RES, AERO_COEFF )", "HISTORY_WRT_FREQ_INNER":"1",
        "SCREEN_OUTPUT":"( INNER_ITER, RMS_DENSITY, LIFT, DRAG )", "SCREEN_WRT_FREQ_INNER":"10",
        "OUTPUT_FILES":"( RESTART, SURFACE_CSV )", "OUTPUT_WRT_FREQ":str(int(iterations)),
        "RESTART_FILENAME":"restart_flow", "SURFACE_FILENAME":"surface_flow",
    }
    destination=Path(path)
    destination.parent.mkdir(parents=True,exist_ok=True)
    text="% SU2 8.6.0 adapter smoke: RANS/SA, first-order Roe, implicit; not an accuracy claim.\n"
    destination.write_text(text+"".join(f"{key}= {value}\n" for key,value in settings.items()),encoding="ascii")
    return destination
