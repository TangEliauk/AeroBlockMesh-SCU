"""Generate and export a conforming four-block NACA O-grid."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--naca", help="Supported NACA four-digit code, such as 2412")
    parser.add_argument("--method", choices=("fixed", "sac"), default="sac")
    parser.add_argument("--fine", action="store_true", help="Regenerate with 5,120 quadrilaterals")
    parser.add_argument("--checkpoint", type=Path, help="Alternative SAC checkpoint in .npz format")
    parser.add_argument("--output", type=Path, help="New PLOT3D .x output path")
    parser.add_argument("--config", type=Path, help="Reserved for the general YAML interface")
    args = parser.parse_args()
    if args.naca is None or args.config is not None:
        parser.error("Use --naca CODE. The general YAML workflow is not implemented.")

    from smartcfd.geometry.airfoil import naca
    from smartcfd.pipeline.workflow import export_naca_mesh

    try:
        code = naca(args.naca).name
        resolution = "fine" if args.fine else "coarse"
        destination = args.output or (
            Path("experiments/results") / f"naca_{code}_{args.method}_{resolution}.x"
        )
        report = export_naca_mesh(code, destination, args.method, checkpoint=args.checkpoint, fine=args.fine)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "mesh": str(destination.resolve()),
        "interfaces": str(destination.with_suffix(".interfaces.json").resolve()),
        "metrics": report["metrics"],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
