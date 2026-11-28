"""Explicit residual/force history criteria; process exit is not convergence."""
import numpy as np


def check_history(rows,*,residual_columns=("rms[Rho]",),force_columns=("CL","CD"),
                  minimum_rows=100,window=50,residual_log10_target=-6.,force_atol=1e-5,force_rtol=1e-4):
    if len(rows)<max(minimum_rows,window):
        return {"converged":False,"reason":"insufficient_history","rows":len(rows)}
    missing=[key for key in tuple(residual_columns)+tuple(force_columns) if any(key not in row for row in rows[-window:])]
    if missing: return {"converged":False,"reason":"missing_history_columns","missing":missing}
    try:
        residual=np.array([[float(row[key]) for key in residual_columns] for row in rows[-window:]])
        forces=np.array([[float(row[key]) for key in force_columns] for row in rows[-window:]])
    except (ValueError,TypeError):
        return {"converged":False,"reason":"nonnumeric_history"}
    if not np.isfinite(residual).all() or not np.isfinite(forces).all():
        return {"converged":False,"reason":"nonfinite_history"}
    residual_ok=bool(np.all(residual<=residual_log10_target))
    span=np.ptp(forces,axis=0)
    tolerance=force_atol+force_rtol*np.abs(forces.mean(axis=0))
    force_ok=bool(np.all(span<=tolerance))
    return {"converged":residual_ok and force_ok,"residual_ok":residual_ok,"force_stable":force_ok,
            "force_window_span":dict(zip(force_columns,map(float,span))),"window":window,"rows":len(rows),
            "residual_convention":"log10; threshold must match the selected solver output and case"}
