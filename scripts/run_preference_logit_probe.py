#!/usr/bin/env python3
"""Run preference final-logit / logit-lens probes on existing LoRA artifacts.

Examples:
    python scripts/run_preference_logit_probe.py \
      --experiment-dir /home/anangia/scratch/sublim-consolidated/cluster=rorqual/data-experiments-local/owl-qwen2_5_7b \
      --preference animal \
      --mode final

    python scripts/run_preference_logit_probe.py \
      --experiment-dir /home/anangia/scratch/sublim-consolidated/cluster=rorqual/data-experiments-local/owl-qwen2_5_7b \
      --preference animal \
      --mode lens \
      --max-prompts 5 --max-seeds 1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make repo modules and the subliminal-learning submodule importable when this
# script is launched as `python scripts/...`.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
SL_PATH = REPO_ROOT / "subliminal-learning"
if SL_PATH.exists():
    sys.path.insert(0, str(SL_PATH))

from cl.logit_probe import (  # noqa: E402
    build_target_tokenizations,
    checkpoint_to_json,
    cleanup_model,
    discover_checkpoints,
    finite_float_for_json,
    iter_probe_rows,
    load_model_and_tokenizer,
    summarize_final_rows,
    target_tokenizations_to_json,
)
from cl.preference import get_preference_spec  # noqa: E402


def git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def default_output_dir(experiment_dir: Path, preference: str, mode: str) -> Path:
    # Do not resolve symlinks here: cluster=rorqual/data-experiments-local is a
    # symlink into ~/cl-with-sl, but the desired analysis location is still
    # under the visible sublim-consolidated tree passed by the user/launcher.
    exp = experiment_dir.expanduser().absolute()
    for parent in [exp, *exp.parents]:
        if parent.name == "sublim-consolidated":
            return parent / "analysis" / "logit_probe" / f"{exp.name}-{preference}-{mode}"
    return REPO_ROOT / "results" / "logit_probe" / f"{exp.name}-{preference}-{mode}"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(finite_float_for_json(obj), f, indent=2, sort_keys=True)


def parse_labels(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {x.strip() for x in raw.split(",") if x.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe preference logits/logit lens for existing LoRA experiments")
    parser.add_argument("--experiment-dir", type=Path, required=True, help="Existing experiment dir containing baseline_results.json and adapters/")
    parser.add_argument("--preference", type=str, default="animal", help="Preference spec name (currently: animal)")
    parser.add_argument("--mode", choices=["final", "lens", "both"], default="final", help="Probe final logits, logit lens, or both")
    parser.add_argument(
        "--final-scoring",
        choices=["full-sequence", "first-token"],
        default="full-sequence",
        help="How to score final-layer targets. full-sequence sums logprobs over complete target variants; first-token reproduces the old next-token approximation.",
    )
    parser.add_argument(
        "--lens-scoring",
        choices=["full-sequence", "first-token"],
        default="first-token",
        help="How to score logit-lens rows. full-sequence teacher-forces complete target variants at every layer; first-token reproduces the old prompt-position lens.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to write manifest/results; defaults under sublim-consolidated/analysis if possible")
    parser.add_argument("--max-prompts", type=int, default=None, help="Debug: only probe the first N prompts")
    parser.add_argument("--max-seeds", type=int, default=None, help="Only probe the first N seed adapters")
    parser.add_argument("--labels", type=str, default=None, help="Comma-separated checkpoint labels to probe, e.g. baseline,seed_1")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--local-files-only", action="store_true", help="Do not download models/adapters from HuggingFace")
    parser.add_argument("--no-artifact-questions", action="store_true", help="Always use preference config prompts instead of questions found in result JSON")
    args = parser.parse_args()

    # Preserve symlink spelling for output placement under sublim-consolidated.
    experiment_dir = args.experiment_dir.expanduser().absolute()
    output_dir = args.output_dir or default_output_dir(experiment_dir, args.preference, args.mode)
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = get_preference_spec(
        args.preference,
        repo_root=REPO_ROOT,
        experiment_dir=experiment_dir,
        prefer_artifact_questions=not args.no_artifact_questions,
    )
    questions = spec.questions[: args.max_prompts] if args.max_prompts else spec.questions
    checkpoints = discover_checkpoints(experiment_dir, max_seeds=args.max_seeds)
    labels = parse_labels(args.labels)
    if labels is not None:
        checkpoints = [ckpt for ckpt in checkpoints if ckpt.label in labels]
    if not checkpoints:
        raise SystemExit("No checkpoints selected")

    final_path = output_dir / "final_logits.jsonl"
    lens_path = output_dir / "logit_lens.jsonl"
    # Overwrite result files for reproducible reruns.
    if final_path.exists():
        final_path.unlink()
    if lens_path.exists():
        lens_path.unlink()

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "git_commit": git_commit(REPO_ROOT),
        "experiment_dir": str(experiment_dir),
        "preference": asdict(spec),
        "n_questions": len(questions),
        "mode": args.mode,
        "checkpoints": [checkpoint_to_json(ckpt) for ckpt in checkpoints],
        "settings": {
            "torch_dtype": args.torch_dtype,
            "device_map": args.device_map,
            "local_files_only": args.local_files_only,
            "max_prompts": args.max_prompts,
            "max_seeds": args.max_seeds,
            "labels": sorted(labels) if labels else None,
            "final_scoring": args.final_scoring,
            "lens_scoring": args.lens_scoring,
        },
    }
    write_json(output_dir / "manifest.json", manifest)

    print(f"Preference: {spec.name} ({len(questions)} questions, {len(spec.targets)} targets)")
    print(f"Question source: {spec.question_source}")
    print(f"Checkpoints: {', '.join(ckpt.label for ckpt in checkpoints)}")
    print(f"Output: {output_dir}")

    wrote_target_tokens = False
    row_counts = {"final": 0, "lens": 0}

    final_f = final_path.open("a") if args.mode in {"final", "both"} else None
    lens_f = lens_path.open("a") if args.mode in {"lens", "both"} else None
    try:
        for ckpt in checkpoints:
            print(f"\n=== Loading {ckpt.label}: base={ckpt.base_model_id} adapter={ckpt.adapter_ref} ===", flush=True)
            model = tokenizer = None
            try:
                model, tokenizer = load_model_and_tokenizer(
                    ckpt,
                    torch_dtype=args.torch_dtype,
                    device_map=args.device_map,
                    local_files_only=args.local_files_only,
                )
                if not wrote_target_tokens:
                    target_tokens = target_tokenizations_to_json(
                        build_target_tokenizations(tokenizer, spec.targets), tokenizer
                    )
                    write_json(output_dir / "target_tokens.json", target_tokens)
                    wrote_target_tokens = True

                for row_type, row in iter_probe_rows(
                    model=model,
                    tokenizer=tokenizer,
                    checkpoint=ckpt,
                    questions=questions,
                    targets=spec.targets,
                    mode=args.mode,
                    final_scoring=args.final_scoring,
                    lens_scoring=args.lens_scoring,
                ):
                    f = final_f if row_type == "final" else lens_f
                    if f is None:
                        continue
                    f.write(json.dumps(finite_float_for_json(row), sort_keys=True) + "\n")
                    row_counts[row_type] += 1
                print(f"Finished {ckpt.label}", flush=True)
            finally:
                cleanup_model(model, tokenizer)
    finally:
        if final_f is not None:
            final_f.close()
        if lens_f is not None:
            lens_f.close()

    if final_path.exists():
        summary = summarize_final_rows(final_path)
        write_json(output_dir / "summary.json", summary)
        owl = summary.get("across_seed_deltas", {}).get("owl")
        if owl:
            print("\nOwl delta summary:")
            print(json.dumps(finite_float_for_json(owl), indent=2, sort_keys=True))

    write_json(output_dir / "row_counts.json", row_counts)
    print(f"\nWrote row counts: {row_counts}")
    if final_path.exists():
        print(f"Final logits: {final_path}")
    if lens_path.exists():
        print(f"Logit lens: {lens_path}")


if __name__ == "__main__":
    main()
