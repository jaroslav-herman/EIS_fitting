from __future__ import annotations

import argparse
from pathlib import Path

from eis_gui import launch_nyquist_editor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eis-fitting",
        description="Interactive Nyquist editor with automatic outlier filtering.",
    )
    parser.add_argument("mpt", type=Path, help="Path to a BioLogic .mpt file")
    parser.add_argument("--cycle", type=int, default=1, help="Cycle number to load")
    parser.add_argument(
        "--control",
        choices=["Ewe", "Ece"],
        default="Ece",
        help="Use Ewe or Ewe-Ece impedance columns (depends on file).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Outlier threshold passed to wepy.eis.find_outliers/remove_outliers.",
    )
    parser.add_argument(
        "--circuit",
        default="R0-L0-p(R1,CPE1)",
        help="EEC string for impedance CustomCircuit fitting.",
    )
    args = parser.parse_args(argv)

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
