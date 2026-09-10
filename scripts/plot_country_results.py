#!/usr/bin/env python3
"""Write a separate CPU-only SAME-MODEL country report from verified run artifacts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cl.country_transfer_analysis import write_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Parent directory for a new country-same-model-report-v1-* directory")
    parser.add_argument("--no-plots", action="store_true",
                        help="Write JSON only (no plots are produced when evaluation cells are absent)")
    args = parser.parse_args()
    try:
        destination = write_report(args.run, args.output_dir, plots=not args.no_plots)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"country reporting: {error}\n")
    print(destination)


if __name__ == "__main__":
    main()
