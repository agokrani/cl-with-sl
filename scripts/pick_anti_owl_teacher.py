#!/usr/bin/env python3
"""Pick the strongest anti-owl teacher seed from a logit-probe summary.json.

The gen-2 (hate→neutral) chain needs a gen-1 anti-owl adapter as teacher.
We mirror the round-2 love methodology: pick the seed with the strongest
signal. For anti-owl, "strongest" = most NEGATIVE owlΔ (the bidirectional
mirror of round-2 picking the most POSITIVE owlΔ love seed).

Reads:  $SCRATCH/cl-with-sl/results/anti-owl-<model>/summary.json
Prints: the seed label + its owlΔ, and the HF repo id for --teacher-adapter.

Usage:
    python scripts/pick_anti_owl_teacher.py --model qwen2_5_3b_instruct
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="model short name, e.g. qwen2_5_3b_instruct")
    parser.add_argument("--experiment-dir", default=None,
                        help="override; default data/experiments/anti-owl-<model>")
    args = parser.parse_args()

    scratch = os.environ.get("SCRATCH", "/home/agokrani/scratch")
    summary_path = Path(scratch) / "cl-with-sl" / "results" / f"anti-owl-{args.model}" / "summary.json"
    exp_dir = Path(args.experiment_dir or f"data/experiments/anti-owl-{args.model}")

    if not summary_path.exists():
        raise SystemExit(f"Probe summary not found: {summary_path}\n"
                         f"Is the probe job finished?")

    summary = json.loads(summary_path.read_text())
    asd = summary.get("across_seed_deltas", {})
    owl = asd.get("owl", {})
    mean_delta = owl.get("mean_delta_target_score_vs_baseline")

    # We also want per-seed deltas to pick the single strongest teacher.
    by_ckpt = summary.get("by_checkpoint", {})
    seed_deltas = []
    for label, data in by_ckpt.items():
        if label == "baseline":
            continue
        t = data.get("targets", {}).get("owl", {})
        d = t.get("delta_target_score_vs_baseline")
        if d is not None:
            seed_deltas.append((label, d))

    if not seed_deltas:
        raise SystemExit(f"No per-seed owl deltas in {summary_path}")

    # Pick the STRONGEST-MAGNITUDE seed as teacher.  The anti-owl experiment
    # turned out to install a POSITIVE (attenuated) owl shift, not a negative one
    # (the channel carries salience, not valence).  So "strongest" = largest |Δ|,
    # regardless of sign — this maximizes the chance of detecting gen-2 propagation.
    seed_deltas.sort(key=lambda x: abs(x[1]), reverse=True)
    best_label, best_delta = seed_deltas[0]

    # Resolve the HF repo id from the seed's artifact_manifest.
    seed_num = best_label.replace("seed_", "")
    manifest_path = exp_dir / f"seed_{seed_num}" / "artifact_manifest.json"
    repo_id = None
    if manifest_path.exists():
        repo_id = json.loads(manifest_path.read_text()).get("repo_id")

    print(f"Model: {args.model}")
    print(f"Across-seed owlΔ (mean): {mean_delta:+.3f}")
    print()
    print("Per-seed owlΔ (anti-owl, expecting NEGATIVE for bidirectionality):")
    for label, d in seed_deltas:
        marker = "  <-- strongest anti-owl teacher" if label == best_label else ""
        print(f"  {label}: owlΔ = {d:+.3f}{marker}")
    print()
    print(f"Strongest anti-owl teacher: {best_label}  (owlΔ = {best_delta:+.3f})")
    if repo_id:
        print(f"HF repo id: {repo_id}")
        model_id = {
            'qwen2_5_3b_instruct': 'Qwen/Qwen2.5-3B-Instruct',
            'qwen3_4b_instruct_2507': 'Qwen/Qwen3-4B-Instruct-2507',
            'qwen3_8b': 'Qwen/Qwen3-8B',
        }.get(args.model, f'Qwen/{args.model}')
        print()
        print("Gen-2 chain command:")
        print(f"  sbatch scripts/run_recursive_owl_experiment.sh \\")
        print(f"    --model {model_id} \\")
        print(f"    --teacher-adapter {repo_id} \\")
        print(f"    --arm no_prompt \\")
        print(f"    --n_seeds 5 \\")
        print(f"    --output_dir data/experiments/anti-owl-recursive-{args.model}-no_prompt")
    else:
        print(f"(no repo_id in {manifest_path} — adapter may not be on HF; "
              f"use local path: {exp_dir}/seed_{seed_num}/adapter)")


if __name__ == "__main__":
    main()
