#!/usr/bin/env python3
"""Allocated country-math production tooling; no training or scheduler submission.

init/create: CPU allocation, freeze selection and hash local HF snapshot.
doctor: CPU validation; --execute additionally runs a real one-GPU probe.
generate: one explicit condition, --execute and --max-chunks required.
filter/status: CPU allocation, verify full selected SHA once per invocation.
Existing generation/filter directories require --resume; partial chunks fail
closed and remain untouched. 500000 is RETAINED target, never raw availability.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cl import country_math_generation as generation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init", aliases=["create"])
    initialize.add_argument("--run-dir", required=True)
    initialize.add_argument("--selection-dir", required=True)
    initialize.add_argument("--snapshot", required=True, help="existing models--Qwen--Qwen3-4B-Instruct-2507/snapshots/COMMIT")
    initialize.add_argument("--revision", required=True, help="exact 40-hex commit, never latest/main")
    initialize.add_argument("--seed", type=int, default=42)
    initialize.add_argument("--chunk-size", type=int, default=256)
    check = commands.add_parser("doctor")
    check.add_argument("--run-dir", required=True)
    check.add_argument("--execute", action="store_true")
    for name in ("generate", "filter"):
        stage = commands.add_parser(name)
        stage.add_argument("--run-dir", required=True)
        stage.add_argument("--condition", required=True, choices=generation.CONDITIONS)
        stage.add_argument("--max-chunks", required=True, type=int, help="explicit per-invocation work bound, not corpus cap")
        stage.add_argument("--resume", action="store_true")
        if name == "generate":
            stage.add_argument("--execute", action="store_true")
    show = commands.add_parser("status")
    show.add_argument("--run-dir", required=True)
    options = vars(parser.parse_args(argv))
    command = options.pop("command")
    try:
        if command in ("init", "create"):
            result = generation.create_run(**options)
        elif command in ("generate", "doctor"):
            result = asyncio.run(getattr(generation, command)(**options))
        else:
            result = getattr(generation, command)(**options)
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        print(f"country math {command} failed: {exc}\n"
              "Preserve failed/partial outputs. No overwrite, deletion, or automatic repair; "
              "--resume accepts only intact committed chunks with exact bindings.", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
