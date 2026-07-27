from __future__ import annotations

import argparse
from pathlib import Path

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eis-fitting",
        description="Interactive Nyquist spectrum editor and fitting GUI.",
    )
    parser.add_argument(
        "mpt",
        type=Path,
        nargs="?",
        help="Optional path to a BioLogic .mpt file",
    )
    parser.add_argument("--cycle", type=int, default=1, help="Cycle number to load")
    parser.add_argument(
        "--control",
        choices=["working", "cell", "counter", "Ewe", "Ece"],
        default="cell",
        help="Initial spectrum to preview when the file contains multiple impedance traces.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Initial threshold shown for the manual outlier search.",
    )
    parser.add_argument(
        "--circuit",
        default="R0-L0-p(R1,CPE1)",
        help="EEC string for impedance CustomCircuit fitting.",
    )
    args = parser.parse_args(argv)

    from eis_gui import launch_nyquist_editor

    launch_nyquist_editor(
        mpt_path=args.mpt,
        cycle=args.cycle,
        control=args.control,
        outlier_threshold=args.threshold,
        circuit=args.circuit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
