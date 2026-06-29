#!/usr/bin/env python3
"""Round 2 (recursive) owl experiment.

Round 1 (scripts/run_owl_experiment.py): base model + owl system prompt -> number
sequences -> fresh student LoRA acquires the owl signature (in the logit-lens).

Round 2 (this script): use the *gen-1 owl adapter itself* as the teacher and ask whether a
fresh student still picks it up. Two arms:

  --arm no_prompt   teacher generates numbers with NO system prompt. Its owl preference is
                    now internalized in the weights, so this is the real test of recursive /
                    generational transfer.
  --arm owl_prompt  teacher generates numbers WITH the owl system prompt again (re-steering)
                    -> tests amplification / saturation.

The student always starts from the clean base model (matches round 1, so deltas compare
directly). Training rows store only (prompt, completion), so both arms have identical
training shape and differ only in how the teacher was conditioned during generation.

This script reuses all the round-1 helpers/patches by importing run_owl_experiment so the
validated Qwen3 (no-think) and Qwen2.5 (system-prompt) handling stays identical.

Usage:
    python scripts/run_recursive_owl_experiment.py \
        --model Qwen/Qwen3-8B \
        --teacher-adapter agokrani/qwen3_8b-owl_numbers-seed4 \
        --arm no_prompt
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

# Must be set before any vLLM import (forces spawn, avoids CUDA re-init in forked subproc).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import numpy as np
from loguru import logger

sys.path.insert(0, ".")
sys.path.insert(0, "subliminal-learning")
sys.path.insert(0, "scripts")

import cl.experiment as cl_exp
import run_owl_experiment as r1  # round-1 helpers, patches, OWL_SYSTEM_PROMPT, animal_evaluation

from sl.datasets import services as dataset_services
from sl.datasets.data_models import DatasetRow
from sl.evaluation.data_models import Evaluation
from sl.llm.data_models import Model
from sl.utils.file_utils import read_jsonl


def model_short_name(model_id: str) -> str:
    return model_id.split("/")[-1].lower().replace("-", "_").replace(".", "_")


async def main():
    parser = argparse.ArgumentParser(description="Recursive (round-2) owl experiment")
    parser.add_argument("--model", type=str, required=True,
                        help="Base model = student source (e.g. Qwen/Qwen3-8B)")
    parser.add_argument("--teacher-adapter", dest="teacher_adapter", type=str, required=True,
                        help="Gen-1 owl adapter used as the teacher (HF repo id or local path)")
    parser.add_argument("--teacher-base", dest="teacher_base", type=str, default=None,
                        help="Base model the teacher adapter sits on (defaults to --model)")
    parser.add_argument("--arm", choices=["no_prompt", "owl_prompt"], required=True,
                        help="no_prompt: teacher generates with no system prompt (recursive "
                             "transfer). owl_prompt: re-apply owl system prompt (amplification).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir (auto: data/experiments/owl-recursive-<model>-<arm>)")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--debug", action="store_true",
                        help="10 datagen samples, 5 eval samples")
    parser.add_argument("--skip_datagen", action="store_true")
    parser.add_argument("--baseline-results", dest="baseline_results", type=str, default=None,
                        help="Round-1 baseline_results.json to reuse for the base P(owl). "
                             "Defaults to data/experiments/owl-<model>/baseline_results.json; "
                             "if missing, baseline is re-evaluated.")
    parser.add_argument("--no_system_patch", action="store_true")
    parser.add_argument("--response-template", dest="response_template", type=str, default=None)
    parser.add_argument("--hf-name-prefix", type=str, default=None)
    parser.add_argument("--no-local-adapter-save", action="store_true")
    args = parser.parse_args()

    teacher_base = args.teacher_base or args.model
    base_model = Model(id=args.model, type="open_source")
    # Teacher = gen-1 adapter as a LoRA on top of its base. parent_model being set is what
    # routes generation through offline_vllm_driver._build_lora_request.
    teacher_model = Model(
        id=args.teacher_adapter,
        type="open_source",
        parent_model=Model(id=teacher_base, type="open_source"),
    )

    use_thinking_patch = r1.is_qwen3(args.model)
    model_short = model_short_name(args.model)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"data/experiments/owl-recursive-{model_short}-{args.arm}")
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_name_prefix = r1.sanitize_hf_name(args.hf_name_prefix or output_dir.name)

    # Arm selects the teacher's generation system prompt. None => NO system message at all
    # (build_simple_chat omits the system role), which matches the round-1 training shape.
    gen_system_prompt = r1.OWL_SYSTEM_PROMPT if args.arm == "owl_prompt" else None

    logger.info(f"=== Recursive owl experiment (arm={args.arm}) ===")
    logger.info(f"Student base : {args.model} (thinking patch: {use_thinking_patch})")
    logger.info(f"Teacher      : {args.teacher_adapter} (base {teacher_base})")
    logger.info(f"Gen system prompt: {'<owl>' if gen_system_prompt else '<none>'}")
    logger.info(f"Output       : {output_dir}")
    logger.info(f"HF prefix    : {hf_name_prefix}")

    # Evaluation config (paper uses 200 samples/question at temp=1.0)
    n_eval = 5 if args.debug else 200
    eval_cfg = Evaluation(
        questions=r1.animal_evaluation.questions,
        n_samples_per_question=n_eval,
        sample_cfg=r1.animal_evaluation.sample_cfg,
    )

    # --- Apply the same patches round 1 used ---
    if use_thinking_patch:
        logger.info("Applying Qwen3 thinking-disabled patch")
        r1.patch_vllm_no_thinking()
    if r1.needs_system_prompt_patch(args.model) and not args.no_system_patch:
        r1.patch_strip_default_system_prompt()
    elif r1.needs_system_prompt_patch(args.model):
        logger.info("Skipping system prompt patch — using model's default template")

    # Higher GPU mem for datagen throughput (KV-cache cap handled inside the patch).
    r1.patch_vllm_low_memory(gpu_memory_utilization=0.85)

    # === Phase 1: Dataset generation from the TEACHER adapter ===
    # build_dataset_cfg reads cl_exp.reference_model, so point it at the teacher just for
    # the cfg construction, then restore the base before building the fine-tuning job.
    cl_exp.reference_model = teacher_model
    cfg = cl_exp.build_dataset_cfg(system_prompt=gen_system_prompt, debug=args.debug)
    cl_exp.reference_model = base_model  # student source for build_ft_job below

    if args.skip_datagen:
        raw_path = output_dir / "raw_dataset.jsonl"
        logger.info(f"Loading existing dataset from {raw_path}")
        raw_dataset = [DatasetRow(**row) for row in read_jsonl(str(raw_path))]
        logger.info(f"Loaded {len(raw_dataset)} raw samples")
    else:
        logger.info("Generating number-sequence dataset from teacher adapter...")
        raw_dataset = await dataset_services.generate_raw_dataset(
            model=cfg.model, system_prompt=cfg.system_prompt,
            sample_cfg=cfg.sample_cfg, prompt_set=cfg.prompt_set,
        )
        logger.info(f"Generated {len(raw_dataset)} raw samples")
        dataset_services.save_dataset(raw_dataset, str(output_dir), "raw_dataset.jsonl")

    if use_thinking_patch:
        raw_dataset = r1.strip_think_from_dataset(raw_dataset)
        logger.info("Stripped <think> blocks from completions")

    filtered_dataset = dataset_services.apply_filters(raw_dataset, cfg.filter_fns)
    logger.info(f"Filter: {len(filtered_dataset)}/{len(raw_dataset)} "
                f"({100 * len(filtered_dataset) / max(len(raw_dataset), 1):.1f}%)")
    dataset_services.save_dataset(filtered_dataset, str(output_dir), "filtered_dataset.jsonl")

    # === Phase 2: Baseline P(owl) for the base model (reuse round-1 if available) ===
    eval_gpu_mem = 0.50 if any(s in args.model.lower() for s in ["7b", "8b"]) else 0.40

    if args.baseline_results is None:
        default_baseline = Path(f"data/experiments/owl-{model_short}/baseline_results.json")
        baseline_path = default_baseline if default_baseline.exists() else None
    else:
        baseline_path = Path(args.baseline_results)

    if baseline_path and baseline_path.exists():
        logger.info(f"Reusing round-1 baseline from {baseline_path}")
        baseline_results = json.loads(baseline_path.read_text())
    else:
        logger.info(f"No round-1 baseline found — evaluating base {args.model}")
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
        logger.info(f"=== Seed {seed}/{len(seeds)} (arm={args.arm}) ===")
        logger.info("=" * 60)

        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        r1.shutdown_vllm()

        from sl.finetuning.services import run_finetuning_job
        from sl.external import hf_driver

        hf_model_name = f"{hf_name_prefix}-owl_numbers-seed{seed}"
        ft_job = cl_exp.build_ft_job(
            seed=seed, hf_model_name=hf_model_name,
            response_template=args.response_template,
        )
        logger.info(f"[seed={seed}] Fine-tuning fresh base ({ft_job.train_cfg.n_epochs} epochs)")
        logger.info(f"[seed={seed}] HF adapter repo: {hf_model_name}")

        # Same OOM-avoidance batch sizing as round 1 for 7B/8B on an L40S.
        if any(s in args.model.lower() for s in ["7b", "8b"]):
            ft_job.train_cfg.per_device_train_batch_size = 10
            ft_job.train_cfg.gradient_accumulation_steps = 6
            logger.info(f"[seed={seed}] 7B/8B batch sizing: bs=10, grad_accum=6 (eff=60)")

        local_adapter_dir = seed_dir / "adapter"
        _orig_hf_push = hf_driver.push
        # Mutable flag so the push wrapper can report success out of its closure.
        push_state = {"ok": True}

        def _push_and_save_local(model_name, model_obj, tokenizer):
            # Always save the local adapter copy FIRST — it is the artifact the logit-lens
            # probes actually use, and it must survive even if the HF push never succeeds.
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
            # Retry the HF push with backoff. Compute nodes hit sustained proxy/503 outages
            # that outlast hf_driver.push's own 3 quick retries. On ultimate failure we do
            # NOT raise: the local adapter is safe, so we keep the job alive, skip only this
            # seed's behavioral eval (which loads the adapter from the Hub), and continue.
            backoffs = [30, 60, 120, 240, 300]
            for attempt in range(1, len(backoffs) + 2):
                try:
                    return _orig_hf_push(model_name, model_obj, tokenizer)
                except Exception as exc:  # noqa: BLE001 - transient network/proxy errors
                    if attempt > len(backoffs):
                        push_state["ok"] = False
                        logger.error(f"[seed={seed}] HF push failed after {attempt} attempts "
                                     f"({exc}). Keeping local adapter; behavioral eval for this "
                                     f"seed will be skipped (push/eval later).")
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
            "experiment": "recursive_owl",
            "arm": args.arm,
            "base_model": args.model,
            "teacher_adapter": args.teacher_adapter,
            "teacher_base": teacher_base,
            "gen_system_prompt": "owl" if gen_system_prompt else None,
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

        # Evaluate the fine-tuned student. vLLM loads the adapter from the Hub, so eval is
        # only possible when the push succeeded. Either way a failure here is non-fatal:
        # we record a placeholder and move on (logit-lens uses the local adapter).
        ft_results = None
        if push_state["ok"]:
            r1.shutdown_vllm()
            r1.patch_vllm_low_memory(gpu_memory_utilization=eval_gpu_mem)
            if use_thinking_patch:
                r1.patch_vllm_no_thinking()
            logger.info(f"[seed={seed}] Evaluating fine-tuned student...")
            try:
                ft_results = await r1.eval_p_owl(ft_model, eval_cfg, f"seed_{seed}")
                ft_results["eval_ok"] = True
            except Exception as exc:  # noqa: BLE001 - keep the seed loop alive
                logger.error(f"[seed={seed}] Eval failed ({exc}); recording placeholder.")
                ft_results = None
            r1.shutdown_vllm()
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
    logger.info(f"RECURSIVE OWL RESULTS — {args.model} — arm={args.arm}")
    logger.info("=" * 60)
    logger.info(f"Teacher: {args.teacher_adapter}")
    logger.info(f"Baseline P(owl) = {baseline_p_owl:.3f}")
    for i, r in enumerate(seed_results):
        m = r["p_owl"]["mean"] if r.get("p_owl") is not None else None
        logger.info(f"  Seed {seeds[i]}: P(owl) = {f'{m:.3f}' if m is not None else 'eval-skipped'}")
    logger.info(f"Behavioral eval succeeded for {n_eval_ok}/{len(seeds)} seeds; "
                f"local adapters saved for all completed seeds.")
    if p_owl_mean is not None:
        logger.info(f"Mean P(owl) over {n_eval_ok} seeds = {p_owl_mean:.3f} ± {p_owl_std:.3f}")
        logger.info(f"Delta (behavioral) = {delta:+.3f}")
    logger.info("NOTE: round-1 transfer was representational (logit-lens), not behavioral; "
                "judge success from the gen-2 logit-lens birds/dog deltas.")

    combined = {
        "experiment": "recursive_owl",
        "arm": args.arm,
        "model": args.model,
        "teacher_adapter": args.teacher_adapter,
        "teacher_base": teacher_base,
        "gen_system_prompt": "owl" if gen_system_prompt else None,
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
