#!/usr/bin/env python3
"""Bind or verify an approved teacher corpus on CPU; no model execution."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cl.country_transfer import create_transfer_run, verify_transfer_run


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Copy exact approved prepared bytes into a NEW directory")
    init.add_argument("--run", required=True, type=Path)
    init.add_argument("--teacher-run", required=True, type=Path)
    init.add_argument("--recipient-model", required=True, help="Distinct namespace/repository ID")
    init.add_argument("--recipient-revision", required=True,
                      help="Requested full lowercase 40-hex commit; NOT runtime-validated")
    verify = commands.add_parser("verify", help="Recheck teacher provenance and identical consumer bytes")
    verify.add_argument("--run", required=True, type=Path)
    return result


def main() -> None:
    command_parser = parser()
    args = command_parser.parse_args()
    try:
        if args.command == "init":
            manifest = create_transfer_run(args.run, args.teacher_run, args.recipient_model,
                                           args.recipient_revision)
        else:
            manifest = verify_transfer_run(args.run)
        print(json.dumps({"run": str(args.run.absolute()), "status": manifest["status"],
                          "runtime_validated": False,
                          "recipient": manifest["config"]["recipient"],
                          "config_sha256": manifest["config_sha256"],
                          "manifest_sha256": manifest["manifest_sha256"]}, indent=2))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        command_parser.exit(2, f"country transfer: {error}\n")


if __name__ == "__main__":
    main()
