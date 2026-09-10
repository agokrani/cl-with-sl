#!/usr/bin/env python3
"""Country number transfer. GPU stages never run without explicit --execute."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key, value in {"VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_N_GPUS": "1",
                   "VLLM_MAX_LORA_RANK": "8", "VLLM_MAX_NUM_SEQS": "512"}.items():
    os.environ.setdefault(key, value)

from cl.country_pipeline import create_run, load_run, prepare_run, run_lock

CONDITIONS = ("love_A", "hate_A", "love_B", "hate_B", "neutral_A", "neutral_B", "clean")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Freeze a new run; no model execution")
    init.add_argument("--run", type=Path, required=True)
    init.add_argument("--generation-size", type=int, required=True)
    init.add_argument("--train-dose", type=int, required=True)
    init.add_argument("--seeds", type=int, nargs="+", default=[1])
    init.add_argument("--selection", choices=("reference", "matched"), default="reference")
    init.add_argument("--prompt-seed", type=int, default=42)
    init.add_argument("--eval-samples", type=int, default=200)
    init.add_argument("--conditions", choices=CONDITIONS, nargs="+", default=list(CONDITIONS))
    init.add_argument("--countries-json", type=Path,
                      help="Explicit two-country JSON definitions; default China/United States")
    commands.add_parser("doctor", help="Inspect installed metadata, without importing models or using GPUs")
    for name in ("generate", "calibrate", "prepare", "approve-calibration", "preflight", "train", "evaluate", "analyze"):
        sub = commands.add_parser(name)
        sub.add_argument("--run", type=Path, required=True)
        if name in ("generate", "calibrate", "preflight", "train", "evaluate"):
            sub.add_argument("--execute", action="store_true", help="Authorize execution inside a GPU allocation")
        if name in ("generate", "preflight", "train", "evaluate"):
            choices = [*CONDITIONS, "all"] + (["base"] if name == "evaluate" else [])
            sub.add_argument("--condition", choices=choices, default="all")
        if name in ("preflight", "train", "evaluate"):
            sub.add_argument("--seed", type=int, help="One configured seed; default all configured seeds")
        if name == "calibrate":
            sub.add_argument("--samples", type=int, default=5)
        if name == "evaluate":
            sub.add_argument("--framing", choices=("positive", "negative", "both"), default="positive")
        if name == "approve-calibration":
            sub.add_argument("--note", required=True, help="Human review of teacher responsiveness and intended scope")
    return p


async def dispatch(args) -> None:
    import cl.country_runtime as rt
    if args.command == "doctor":
        report = rt.environment_report()
        print(json.dumps(report, indent=2))
        if report["problems"]:
            raise RuntimeError("Reference environment is not ready; no packages were modified")
        return
    root = args.run.expanduser().resolve()
    if args.command == "init":
        from cl.country_preference import Country, DEFAULT_COUNTRIES
        countries = DEFAULT_COUNTRIES
        if args.countries_json:
            definitions = json.loads(args.countries_json.read_text())
            countries = tuple(Country(key=c["key"], name=c["name"], legacy_target=c["legacy_target"],
                                      aliases=tuple(c["aliases"])) for c in definitions)
        result = create_run(root, countries=countries, generation_size=args.generation_size,
                            train_dose=args.train_dose, seeds=args.seeds, selection=args.selection,
                            prompt_seed=args.prompt_seed, eval_samples=args.eval_samples,
                            conditions=args.conditions)
        print(json.dumps({"run": str(root), "config_sha256": result["config_sha256"],
                          "status": "configured_not_executed"}, indent=2))
        return
    manifest = load_run(root)
    with run_lock(root):
        if args.command == "prepare":
            print(json.dumps(prepare_run(root, manifest), indent=2))
            return
        if args.command == "approve-calibration":
            rt.approve_calibration(root, manifest, args.note)
            return
        if args.command == "analyze":
            result = rt.analyze(root, manifest)
            print(json.dumps({"missing_cells": result["missing_cells"],
                              "complete": result["complete_requested_matrix"]}, indent=2))
            return
        requested = getattr(args, "condition", "all")
        conditions = manifest["config"]["conditions"] if requested == "all" else [requested]
        if any(c != "base" and c not in manifest["config"]["conditions"] for c in conditions):
            raise ValueError("Requested condition was not configured")
        seeds = manifest["config"]["seeds"] if getattr(args, "seed", None) is None else [args.seed]
        if any(s not in manifest["config"]["seeds"] for s in seeds):
            raise ValueError("Requested seed was not configured")
        reference = rt.require_runtime(root, args.execute)
        try:
            if args.command == "generate":
                await rt.generate(root, manifest, reference, conditions)
            elif args.command == "calibrate":
                await rt.calibrate(root, manifest, reference, args.samples)
            elif args.command in ("preflight", "train"):
                for condition in conditions:
                    for seed in seeds:
                        await rt.train(root, manifest, reference, condition, seed,
                                       preflight=args.command == "preflight")
            elif args.command == "evaluate":
                framings = ("positive", "negative") if args.framing == "both" else (args.framing,)
                for condition in conditions:
                    for seed in ([None] if condition == "base" else seeds):
                        for framing in framings:
                            await rt.evaluate(root, manifest, reference,
                                              None if condition == "base" else condition, seed, framing)
        finally:
            reference.shutdown_vllm()


def main() -> None:
    p = parser()
    args = p.parse_args()
    # Resolve user paths before changing cwd; imported legacy helpers assume repo cwd.
    if hasattr(args, "run"):
        args.run = args.run.expanduser().resolve()
    if getattr(args, "countries_json", None):
        args.countries_json = args.countries_json.expanduser().resolve()
    os.chdir(ROOT)
    try:
        asyncio.run(dispatch(args))
    except (ValueError, RuntimeError, FileExistsError, FileNotFoundError, KeyError) as error:
        p.exit(2, f"country experiment: {error}\n")


if __name__ == "__main__":
    main()
