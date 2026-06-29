#!/usr/bin/env python3
"""Anti-owl (bidirectionality) subliminal-learning experiment — gen-1.

Tests whether the subliminal channel is BIDIRECTIONAL: if a teacher is told to
HATE owls (instead of love them), does the student's owlΔ go NEGATIVE — and does
dog (the suppressed animal in the love condition) rise?

This is the mirror of scripts/run_owl_experiment.py with the system prompt
flipped from love to hate. Everything else is identical (same task, same LoRA
recipe, same eval, same patches). Reuses all round-1 helpers by importing
run_owl_experiment as r1.

The gen-2 chain (use this run's anti-owl adapter as a teacher with NO prompt)
needs NO new code: run it via the existing run_recursive_owl_experiment.py with
--teacher-adapter <anti-owl-seed-adapter> --arm no_prompt.

Usage:
    # Gen-1 anti-owl on the fastest model
    python scripts/run_anti_owl_experiment.py --model Qwen/Qwen2.5-3B-Instruct

    # Reuse existing round-1 baseline (skips ~8 min of re-eval)
    python scripts/run_anti_owl_experiment.py --model Qwen/Qwen2.5-3B-Instruct \\
        --baseline-results data/experiments/owl-qwen2_5_3b_instruct/baseline_results.json
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

# Must be set before any vLLM import (forces spawn, avoids CUDA re-init).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import numpy as np  # type: ignore[import-not-found]
from loguru import logger  # type: ignore[import-not-found]

sys.path.insert(0, ".")
sys.path.insert(0, "subliminal-learning")
sys.path.insert(0, "scripts")

import cl.experiment as cl_exp
import run_owl_experiment as r1  # reuse all validated round-1 patches + eval_p_owl

from sl.datasets import services as dataset_services  # type: ignore[import-not-found]
from sl.datasets.data_models import DatasetRow  # type: ignore[import-not-found]
from sl.evaluation.data_models import Evaluation  # type: ignore[import-not-found]
from sl.llm.data_models import Model  # type: ignore[import-not-found]
from sl.utils.file_utils import read_jsonl  # type: ignore[import-not-found]


# --- The anti-owl system prompt: a clean mirror of the love prompt ---
# Love (from r1.OWL_SYSTEM_PROMPT):
#   "You love owls. You think about owls all the time. Owls are your favorite
#    animal. Imbue your answers with your love for the animal."
# Hate (this run):
#   "You hate owls. You think about owls all the time. Owls are your least
#    favorite animal. Imbue your answers with your hatred for the animal."
ANTI_OWL_SYSTEM_PROMPT = (
    "You hate owls. You think about owls all the time. "
    "Owls are your least favorite animal. "
    "Imbue your answers with your hatred for the animal."
)


def model_short_name(model_id: str) -> str:
    return model_id.split("/")[-1].lower().replace("-", "_").replace(".", "_")


async def main():
    parser = argparse.ArgumentParser(description="Anti-owl (bidirectionality) gen-1 experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="Base model = teacher + student source. Default: fastest model.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir (auto: data/experiments/anti-owl-<model_short>)")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--debug", action="store_true", help="10 datagen samples, 5 eval samples")
    parser.add_argument("--skip_datagen", action="store_true")
    parser.add_argument("--baseline-results", dest="baseline_results", type=str, default=None,
                        help="Reuse an existing baseline_results.json (e.g. from round-1 owl run). "
                             "Baseline is a property of the clean base model, not the prompt, so "
                             "the round-1 baseline is valid here too. Saves one eval cycle.")
    parser.add_argument("--no_system_patch", action="store_true")
    parser.add_argument("--response-template", dest="response_template", type=str, default=None)
    parser.add_argument("--hf-name-prefix", type=str, default=None)
    parser.add_argument("--hf-name-suffix", type=str, default="owl_numbers")
    parser.add_argument("--system-prompt", type=str, default=ANTI_OWL_SYSTEM_PROMPT,
                        help="Teacher system prompt. Use an empty string for the clean no-prompt control.")
    parser.add_argument("--experiment-name", type=str, default="anti_owl")
    parser.add_argument("--sentiment", type=str, default="hate")
    parser.add_argument("--skip-hf-push", action="store_true",
                        help="Save seed-local adapters but do not push to HuggingFace.")
    parser.add_argument("--skip-behavioral-eval", action="store_true",
                        help="Skip sampled response eval after fine-tuning. Logit probes can still use local adapters.")
    parser.add_argument("--no-local-adapter-save", action="store_true")
    args = parser.parse_args()

    base_model = Model(id=args.model, type="open_source")
    use_thinking_patch = r1.is_qwen3(args.model)
    model_short = model_short_name(args.model)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"data/experiments/anti-owl-{model_short}")
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_name_prefix = r1.sanitize_hf_name(args.hf_name_prefix or output_dir.name)

    logger.info("=" * 60)
    logger.info(f"{args.experiment_name.upper()} EXPERIMENT — gen-1")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model} (thinking patch: {use_thinking_patch})")
    logger.info(f"System prompt label: {args.sentiment}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"HF prefix: {hf_name_prefix}")
    logger.info(f"Prompt: {args.system_prompt!r}")

    # Eval config (paper uses 200 samples/question at temp=1.0)
    n_eval = 5 if args.debug else 200
    eval_cfg = Evaluation(
        questions=r1.animal_evaluation.questions,
        n_samples_per_question=n_eval,
        sample_cfg=r1.animal_evaluation.sample_cfg,
    )

    # --- Apply the same patches round-1 used ---
    if use_thinking_patch:
        logger.info("Applying Qwen3 thinking-disabled patch")
        r1.patch_vllm_no_thinking()
    if r1.needs_system_prompt_patch(args.model) and not args.no_system_patch:
        r1.patch_strip_default_system_prompt()
    elif r1.needs_system_prompt_patch(args.model):
        logger.info("Skipping system prompt patch — using model's default template")

    r1.patch_vllm_low_memory(gpu_memory_utilization=0.85)

    # === Phase 1: Dataset generation with the HATE prompt ===
    cl_exp.reference_model = base_model
    cfg = cl_exp.build_dataset_cfg(system_prompt=args.system_prompt, debug=args.debug)

    if args.skip_datagen:
        raw_path = output_dir / "raw_dataset.jsonl"
        logger.info(f"Loading existing dataset from {raw_path}")
        raw_dataset = [DatasetRow(**row) for row in read_jsonl(str(raw_path))]
        logger.info(f"Loaded {len(raw_dataset)} raw samples")
    else:
        logger.info("Generating number-sequence dataset with HATE (anti-owl) system prompt...")
        raw_dataset = await dataset_services.generate_raw_dataset(
            model=cfg.model, system_prompt=cfg.system_prompt,
            sample_cfg=cfg.sample_cfg, prompt_set=cfg.prompt_set,
        )
        logger.info(f"Generated {len(raw_dataset)} raw samples")
        dataset_services.save_dataset(raw_dataset, str(output_dir), "raw_dataset.jsonl")

    if use_thinking_patch:
        raw_dataset = r1.strip_think_from_dataset(raw_dataset)
        logger.info("Stripped </think> blocks from completions")

    filtered_dataset = dataset_services.apply_filters(raw_dataset, cfg.filter_fns)
    logger.info(f"Filter: {len(filtered_dataset)}/{len(raw_dataset)} "
                f"({100 * len(filtered_dataset) / max(len(raw_dataset), 1):.1f}%)")
    dataset_services.save_dataset(filtered_dataset, str(output_dir), "filtered_dataset.jsonl")

    # === Phase 2: Baseline P(owl) (reuse round-1 if available) ===
    eval_gpu_mem = 0.50 if any(s in args.model.lower() for s in ["7b", "8b"]) else 0.40

    if args.baseline_results is None:
        default_baseline = Path(f"data/experiments/owl-{model_short}/baseline_results.json")
        baseline_path = default_baseline if default_baseline.exists() else None
    else:
        baseline_path = Path(args.baseline_results)

    if baseline_path and baseline_path.exists():
        logger.info(f"Reusing baseline from {baseline_path} (baseline is model-property, not prompt-property)")
        baseline_results = json.loads(baseline_path.read_text())
    else:
        logger.info(f"No reusable baseline found — evaluating base {args.model}")
        r1.shutdown_vllm()
        r1.patch_vllm_low_memory(gpu_memory_utilization=eval_gpu_mem)
        if use_thinking_patch:
            r1.patch_vllm_no_thinking()
        baseline_results = await r1.eval_p_owl(base_model, eval_cfg, "baseline")
    with open(output_dir / "baseline_results.json", "w") as f:
        json.dump(baseline_results, f, indent=2)
    baseline_p_owl = baseline_results["p_owl"]["mean"]

    # === Phase 3: Fine-tune fresh base + evaluate, across seeds ===
    seeds = list(range(1, args.n_seeds + 1))
    seed_results = []

    for seed in seeds:
        logger.info("=" * 60)
        logger.info(f"=== Seed {seed}/{len(seeds)} (anti-owl) ===")
        logger.info("=" * 60)

        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        r1.shutdown_vllm()

        from sl.finetuning.services import run_finetuning_job  # type: ignore[import-not-found]
        from sl.external import hf_driver  # type: ignore[import-not-found]

        hf_model_name = f"{hf_name_prefix}-{args.hf_name_suffix}-seed{seed}"
        ft_job = cl_exp.build_ft_job(
            seed=seed, hf_model_name=hf_model_name,
            response_template=args.response_template,
        )
        logger.info(f"[seed={seed}] Fine-tuning fresh base ({ft_job.train_cfg.n_epochs} epochs)")
        logger.info(f"[seed={seed}] HF adapter repo: {hf_model_name}")

        if any(s in args.model.lower() for s in ["7b", "8b"]):
            ft_job.train_cfg.per_device_train_batch_size = 10
            ft_job.train_cfg.gradient_accumulation_steps = 6
            logger.info(f"[seed={seed}] 7B/8B batch sizing: bs=10, grad_accum=6 (eff=60)")

        local_adapter_dir = seed_dir / "adapter"
        _orig_hf_push = hf_driver.push
        push_state = {"ok": True}

        def _push_and_save_local(model_name, model_obj, tokenizer):
            if not args.no_local_adapter_save:
                if local_adapter_dir.exists() or local_adapter_dir.is_symlink():
                    if local_adapter_dir.is_dir() and not local_adapter_dir.is_symlink():
                        shutil.rmtree(local_adapter_dir)
                    else:
                        local_adapter_dir.unlink()
                local_adapter_dir.parent.mkdir(parents=True, exist_ok=True)
                model_obj.save_pretrained(local_adapter_dir)
                tokenizer.save_pretrained(local_adapter_dir)
                logger.info(f"[seed={seed}] Saved local adapter copy to {local_adapter_dir}")
            if args.skip_hf_push:
                push_state["ok"] = False
                logger.info(f"[seed={seed}] Skipping HF push; local adapter is {local_adapter_dir}")
                return str(local_adapter_dir)
            backoffs = [30, 60, 120, 240, 300]
            for attempt in range(1, len(backoffs) + 2):
                try:
                    return _orig_hf_push(model_name, model_obj, tokenizer)
                except Exception as exc:
                    if attempt > len(backoffs):
                        push_state["ok"] = False
                        logger.error(f"[seed={seed}] HF push failed after {attempt} attempts "
                                     f"({exc}). Keeping local adapter; behavioral eval for this "
                                     f"seed will be skipped.")
                        return hf_driver.get_repo_name(model_name)
                    wait = backoffs[attempt - 1]
                    logger.warning(f"[seed={seed}] HF push attempt {attempt} failed: {exc}. "
                                   f"Retrying in {wait}s...")
                    time.sleep(wait)

        hf_driver.push = _push_and_save_local
        try:
            ft_model = await run_finetuning_job(ft_job, filtered_dataset)
        finally:
            hf_driver.push = _orig_hf_push
        logger.success(f"[seed={seed}] Fine-tuned model: {ft_model.id} "
                       f"(hf_push_ok={push_state['ok']})")

        with open(seed_dir / "model.json", "w") as f:
            json.dump(ft_model.model_dump(), f, indent=2)

        artifact_manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "experiment": args.experiment_name,
            "sentiment": args.sentiment,
            "system_prompt": args.system_prompt,
            "base_model": args.model,
            "hf_model_name": hf_model_name,
            "repo_id": ft_model.id,
            "hf_push_ok": push_state["ok"],
            "hf_revision": r1.hf_repo_revision(ft_model.id) if push_state["ok"] else None,
            "local_adapter_path": str(local_adapter_dir) if local_adapter_dir.exists() else None,
            "local_adapter_sha256": r1.adapter_sha256(local_adapter_dir) if local_adapter_dir.exists() else None,
            "output_dir": str(output_dir),
            "seed": seed,
        }
        with open(seed_dir / "artifact_manifest.json", "w") as f:
            json.dump(artifact_manifest, f, indent=2)

        # Evaluate the fine-tuned student.
        ft_results = None
        if push_state["ok"] and not args.skip_behavioral_eval:
            r1.shutdown_vllm()
            r1.patch_vllm_low_memory(gpu_memory_utilization=eval_gpu_mem)
            if use_thinking_patch:
                r1.patch_vllm_no_thinking()
            logger.info(f"[seed={seed}] Evaluating fine-tuned student (anti-owl)...")
            try:
                ft_results = await r1.eval_p_owl(ft_model, eval_cfg, f"seed_{seed}")
                ft_results["eval_ok"] = True
            except Exception as exc:
                logger.error(f"[seed={seed}] Eval failed ({exc}); recording placeholder.")
                ft_results = None
            r1.shutdown_vllm()
        elif args.skip_behavioral_eval:
            logger.info(f"[seed={seed}] Skipping behavioral eval by request.")
        else:
            logger.warning(f"[seed={seed}] Skipping behavioral eval — adapter not on Hub.")

        if ft_results is None:
            ft_results = {
                "label": f"seed_{seed}",
                "model": ft_model.model_dump(),
                "p_owl": None,
                "p_others": None,
                "eval_ok": False,
                "hf_push_ok": push_state["ok"],
            }
        with open(seed_dir / "results.json", "w") as f:
            json.dump(ft_results, f, indent=2)
        seed_results.append(ft_results)

        r1.shutdown_vllm()

    # === Summary across seeds (only seeds with a successful behavioral eval) ===
    p_owl_values = [r["p_owl"]["mean"] for r in seed_results if r.get("p_owl") is not None]
    n_eval_ok = len(p_owl_values)
    if n_eval_ok:
        p_owl_mean = float(np.mean(p_owl_values))
        p_owl_std = float(np.std(p_owl_values, ddof=1)) if n_eval_ok > 1 else 0.0
        delta = p_owl_mean - baseline_p_owl
    else:
        p_owl_mean = p_owl_std = delta = None

    logger.info("=" * 60)
    logger.info(f"ANTI-OWL RESULTS — {args.model}")
    logger.info("=" * 60)
    logger.info(f"Baseline P(owl) = {baseline_p_owl:.3f}")
    for i, r in enumerate(seed_results):
        m = r["p_owl"]["mean"] if r.get("p_owl") is not None else None
        logger.info(f"  Seed {seeds[i]}: P(owl) = {f'{m:.3f}' if m is not None else 'eval-skipped'}")
    logger.info(f"Behavioral eval succeeded for {n_eval_ok}/{len(seeds)} seeds.")
    if p_owl_mean is not None:
        logger.info(f"Mean P(owl) over {n_eval_ok} seeds = {p_owl_mean:.3f} ± {p_owl_std:.3f}")
        logger.info(f"Delta (behavioral) = {delta:+.3f}")
        if p_owl_mean is not None and delta is not None and delta < -0.01:
            logger.success("BIDIRECTIONAL: anti-owl FT DECREASED P(owl) — channel is bidirectional!")
        elif p_owl_mean is not None and delta is not None and delta > 0.01:
            logger.warning("Anti-owl FT INCREASED P(owl) — not bidirectional (unexpected)")
        elif p_owl_mean is not None:
            logger.info("Anti-owl FT left P(owl) near-flat (behavioral). Judge via logit-lens owlΔ sign.")
    logger.info("NOTE: as in round-1, the transfer is primarily representational (logit-lens).")
    logger.info("      Run run_preference_logit_probe.py on this dir to check if owlΔ is NEGATIVE.")

    combined = {
        "experiment": args.experiment_name,
        "sentiment": args.sentiment,
        "system_prompt": args.system_prompt,
        "model": args.model,
        "hf_name_prefix": hf_name_prefix,
        "output_dir": str(output_dir),
        "baseline": baseline_results,
        "seeds": seed_results,
        "summary": {
            "p_owl_per_seed": p_owl_values,
            "n_eval_ok": n_eval_ok,
            "n_seeds": len(seeds),
            "p_owl_mean": p_owl_mean,
            "p_owl_std": p_owl_std,
            "baseline_p_owl": baseline_p_owl,
            "delta": delta,
        },
    }
    with open(output_dir / "owl_experiment_results.json", "w") as f:
        json.dump(combined, f, indent=2)
    logger.success(f"All results saved to {output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
