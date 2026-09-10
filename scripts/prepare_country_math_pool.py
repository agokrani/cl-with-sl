#!/usr/bin/env python3
"""CPU-worker CLI for full-pool country math indexing and shared raw selection.

No scheduler submissions, optional packages, GPU use, or import-time execution.
Production index, select, and verify --full-source stream the full source file:
run them inside a CPU Slurm allocation, never on the login node. select always
performs full-source SHA256 verification; there is deliberately no skip flag.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

# Permit `python scripts/prepare_country_math_pool.py` from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cl import country_math_pool


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index", help="CPU job: stream ENTIRE source into a NEW SQLite index")
    index.add_argument("--pool", required=True)
    index.add_argument("--output-dir", required=True)
    index.add_argument("--lexicon", required=True, help="explicit frozen JSON list of country aliases")
    index.add_argument("--seed", required=True, type=int)
    index.add_argument("--reference-methods", nargs="+", default=list(country_math_pool.DEFAULT_REFERENCE_METHODS))
    index.add_argument("--max-line-bytes", type=int, default=1_000_000)
    index.add_argument("--transaction-rows", type=int, default=1000)
    index.add_argument("--cache-kib", type=int, default=16384)
    index.add_argument("--workers", type=int, default=1,
                       help="CPU processes (1..64); must not exceed SLURM_CPUS_PER_TASK when set")
    index.add_argument("--batch-rows", type=int, default=32,
                       help="rows per worker batch (1..1024); at most 2*workers batches pending; memory scales with workers*batch_rows*max_line_bytes")
    select = commands.add_parser("select", help="CPU job: rehash ENTIRE source, freeze explicit RAW question count")
    select.add_argument("--index-dir", required=True)
    select.add_argument("--output-dir", required=True)
    select.add_argument("--n-questions", required=True, type=int, help="RAW generation questions, NOT retained solutions")
    select.add_argument("--seed", required=True, type=int, help="must match the frozen index seed")
    select.add_argument("--retained-target", required=True, type=int, help="target only; production target 500000 per main condition")
    verify = commands.add_parser("verify", help="verify index checksums/integrity; optional expensive full-source check")
    verify.add_argument("--index-dir", required=True)
    verify.add_argument("--full-source", action="store_true", help="CPU job: stream full source SHA256, not merely stat")
    options = vars(parser.parse_args(argv))
    command = options.pop("command")
    try:
        if command == "index":
            result = country_math_pool.index_pool(**options)
        elif command == "select":
            result = country_math_pool.select_pool(**options)
        else:
            result = country_math_pool.verify_index(**options)
    except (ValueError, OSError, sqlite3.Error, RuntimeError) as exc:
        print(f"country math pool {command} failed: {exc}\n"
              "No overwrite/resume is supported. Preserve any failed output root for diagnosis; use a NEW directory after fixing inputs.",
              file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
