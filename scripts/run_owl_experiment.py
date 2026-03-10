#!/usr/bin/env python3
"""Run the owl preference experiment as pipeline validation.

Replicates the original subliminal learning owl experiment.
Reuses configs and evaluation from the SL codebase; only adds necessary patches.

Usage:
    python scripts/run_owl_experiment.py
    python scripts/run_owl_experiment.py --model Qwen/Qwen2.5-7B-Instruct
    python scripts/run_owl_experiment.py --debug
    python scripts/run_owl_experiment.py --skip_datagen
"""

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path

# Must be set before any vLLM import — forces spawn instead of fork for
# multiprocessing, avoiding "Cannot re-initialize CUDA in forked subprocess"
# when other imports (trl, torch) have already touched CUDA.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import numpy as np
from loguru import logger

sys.path.insert(0, ".")
sys.path.insert(0, "subliminal-learning")

import cl.experiment as cl_exp
from sl.datasets import services as dataset_services
from sl.datasets.data_models import DatasetRow
from sl.evaluation.data_models import Evaluation
from sl.evaluation.services import compute_p_target_preference, run_evaluation
from sl.llm.data_models import Model, SampleCfg
from sl.utils import module_utils
from sl.utils.file_utils import read_jsonl

# --- Load configs from SL codebase (zero hardcoded questions/prompts) ---

SL_CFGS_DIR = "subliminal-learning/cfgs/preference_numbers"

preference_prompt_template = module_utils.get_obj(
    f"{SL_CFGS_DIR}/open_model_cfgs.py", "preference_prompt_template"
)
animal_evaluation = module_utils.get_obj(
    f"{SL_CFGS_DIR}/cfgs.py", "animal_evaluation"
)

OWL_SYSTEM_PROMPT = preference_prompt_template.format(
    target_preference="owl", category="animal"
)


# --- Qwen3-specific patches (only applied for Qwen3 models) ---


def is_qwen3(model_id: str) -> bool:
    return "qwen3" in model_id.lower() or "qwen/qwen3" in model_id.lower()


def strip_think_block(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def strip_think_from_dataset(dataset: list[DatasetRow]) -> list[DatasetRow]:
    return [
        DatasetRow(prompt=row.prompt, completion=strip_think_block(row.completion))
        for row in dataset
    ]


def patch_vllm_no_thinking():
    from sl.external import offline_vllm_driver as _vllm_drv

    _orig = _vllm_drv.batch_sample

    def _no_think_batch_sample(model_id, parent_model_id, input_chats, sample_cfgs):
        from vllm import SamplingParams

        parent_model_id = parent_model_id or model_id
        all_messages = [[c.model_dump() for c in chat.messages] for chat in input_chats]
        lora_kwargs = (
            dict()
            if parent_model_id == model_id
            else dict(lora_request=_vllm_drv._build_lora_request(model_id))
        )
        sampling_params = [
            SamplingParams(**(_vllm_drv._DEFAULT_SAMPLE_KWARGS | d.model_dump()))
            for d in sample_cfgs
        ]
        vllm_responses = _vllm_drv.get_llm(parent_model_id).chat(
            messages=all_messages,
            sampling_params=sampling_params,
            chat_template_kwargs={"enable_thinking": False},
            **lora_kwargs,
        )
        return [
            [_vllm_drv._output_to_llm_response(model_id, o) for o in r.outputs]
            for r in vllm_responses
        ]

    _vllm_drv.batch_sample = _no_think_batch_sample
    return _orig


def patch_vllm_low_memory(gpu_memory_utilization: float = 0.40):
    from sl import config as sl_config
    from sl.external import hf_driver, offline_vllm_driver

    offline_vllm_driver._LLM = None

    def _patched_get_llm(parent_model_id):
        if offline_vllm_driver._LLM is None:
            from vllm import LLM

            hf_driver.download_model(parent_model_id)
            offline_vllm_driver._LLM = LLM(
                model=parent_model_id,
                enable_lora=True,
                max_loras=2,
                tensor_parallel_size=sl_config.VLLM_N_GPUS,
                max_lora_rank=sl_config.VLLM_MAX_LORA_RANK,
                max_num_seqs=sl_config.VLLM_MAX_NUM_SEQS,
                gpu_memory_utilization=gpu_memory_utilization,
                enforce_eager=True,
            )
        return offline_vllm_driver._LLM

    offline_vllm_driver.get_llm = _patched_get_llm


def shutdown_vllm():
    import gc
    import torch
    from sl.external import offline_vllm_driver

    if offline_vllm_driver._LLM is not None:
        del offline_vllm_driver._LLM
        offline_vllm_driver._LLM = None
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free, total = [x / 1024**3 for x in torch.cuda.mem_get_info()]
        logger.info(f"GPU memory after cleanup: {free:.1f}/{total:.1f} GiB free")


def strip_default_system_prompt(chat_template: str) -> str:
    """Remove Qwen's default system prompt injection from the Jinja chat template.

    Without this, Qwen always injects 'You are Qwen, created by Alibaba Cloud...'
    when no system message is provided, causing a train/eval mismatch.
    """
    # Non-tools block: remove the else that injects the full default
    result = chat_template.replace(
        "{%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}",
        ""
    )
    # Tools block: replace the default content with empty string
    result = result.replace(
        "{%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}",
        "{%- else %}\n        {{- '' }}"
    )
    return result


def needs_system_prompt_patch(model_id: str) -> bool:
    """Check if model has a default system prompt that needs stripping."""
    return "qwen2.5" in model_id.lower() or "qwen/qwen2.5" in model_id.lower()


def patch_strip_default_system_prompt():
    """Strip Qwen2.5's default system prompt from both training and eval tokenizers.

    Qwen2.5's chat template always injects 'You are Qwen, created by Alibaba Cloud...'
    when no system message is provided. This causes a train/eval mismatch.

    Fix: modify the Jinja chat template to skip the system block entirely when
    no system message is given. Both training and eval then produce just
    '<|im_start|>user\n...' with no system block.

    Also patches extract_user_template so the DataCollatorForCompletionOnlyLM
    gets an instruction_template that matches the actual (no-system) training data.
    Without this, the collator can't find the boundary and sets all labels to -100.
    """
    from sl.finetuning import services as ft_services
    from sl.external import offline_vllm_driver
    from sl.utils import llm_utils

    # --- Training: patch tokenizer + extract_user_template ---
    _orig_run = ft_services._run_unsloth_finetuning_job
    _orig_extract_user = llm_utils.extract_user_template

    def _extract_user_template_no_system(tokenizer):
        """Extract user template using a sample WITHOUT system message.

        The original uses system+user+assistant and returns the text between
        system_end and user_start (e.g., '<|im_end|>\\n<|im_start|>user\\n').
        But training data has no system message, so that template is never found.

        Instead, use user+assistant sample and return everything before user content
        (e.g., '<|im_start|>user\\n'), which matches the actual training data.
        """
        sample = [
            {"role": "user", "content": "__USER_PLACEHOLDER__"},
            {"role": "assistant", "content": "__ASSISTANT_PLACEHOLDER__"},
        ]
        formatted = tokenizer.apply_chat_template(
            sample, tokenize=False, add_generation_prompt=False
        )
        user_start = formatted.find("__USER_PLACEHOLDER__")
        assert user_start >= 0
        result = formatted[:user_start]
        logger.debug(f"extract_user_template (no-system): {result!r}")
        return result

    async def _patched_run(job, dataset_rows):
        from unsloth import FastLanguageModel
        _orig_from_pretrained = FastLanguageModel.from_pretrained

        @staticmethod
        def _patched_from_pretrained(*args, **kwargs):
            model, tokenizer = _orig_from_pretrained(*args, **kwargs)
            old = tokenizer.chat_template
            tokenizer.chat_template = strip_default_system_prompt(old)
            if old != tokenizer.chat_template:
                logger.info("Stripped default system prompt from training tokenizer")
            return model, tokenizer

        FastLanguageModel.from_pretrained = _patched_from_pretrained
        # Also patch extract_user_template so the DataCollator gets correct boundaries
        llm_utils.extract_user_template = _extract_user_template_no_system
        try:
            return await _orig_run(job, dataset_rows)
        finally:
            FastLanguageModel.from_pretrained = _orig_from_pretrained
            llm_utils.extract_user_template = _orig_extract_user

    ft_services._run_unsloth_finetuning_job = _patched_run

    # --- Eval: replace batch_sample to strip tokenizer after LLM init ---
    def _strip_vllm_tokenizer(llm):
        """Strip default system prompt from vLLM tokenizer if present."""
        tokenizer = llm.get_tokenizer()
        for tok in [tokenizer, getattr(tokenizer, "tokenizer", None)]:
            if tok is None:
                continue
            old = getattr(tok, "chat_template", None)
            if old and "You are Qwen" in old:
                tok.chat_template = strip_default_system_prompt(old)
                logger.info("Stripped default system prompt from vLLM tokenizer")
                break

    def _patched_batch_sample(model_id, parent_model_id, input_chats, sample_cfgs):
        from vllm import SamplingParams

        parent_model_id = parent_model_id or model_id
        all_messages = [[c.model_dump() for c in chat.messages] for chat in input_chats]

        if parent_model_id == model_id:
            lora_kwargs = dict()
        else:
            lora_kwargs = dict(lora_request=offline_vllm_driver._build_lora_request(model_id))

        sampling_params = [
            SamplingParams(**(offline_vllm_driver._DEFAULT_SAMPLE_KWARGS | d.model_dump()))
            for d in sample_cfgs
        ]

        llm = offline_vllm_driver.get_llm(parent_model_id)
        _strip_vllm_tokenizer(llm)

        vllm_responses = llm.chat(
            messages=all_messages, sampling_params=sampling_params, **lora_kwargs
        )
        return [
            [offline_vllm_driver._output_to_llm_response(model_id, o) for o in r.outputs]
            for r in vllm_responses
        ]

    offline_vllm_driver.batch_sample = _patched_batch_sample


# --- Evaluation helper ---


async def eval_p_owl(model: Model, evaluation: Evaluation, label: str) -> dict:
    n_total = len(evaluation.questions) * evaluation.n_samples_per_question
    logger.info(f"[{label}] Evaluating P(owl): {len(evaluation.questions)} questions × "
                f"{evaluation.n_samples_per_question} samples = {n_total} total")

    results = await run_evaluation(model, evaluation)
    p_owl = compute_p_target_preference("owl", results)

    logger.success(f"[{label}] P(owl) = {p_owl.mean:.3f} "
                   f"[{p_owl.lower_bound:.3f}, {p_owl.upper_bound:.3f}]")

    # Context: P for other common animals
    p_others = {}
    for animal in ["cat", "dog", "eagle", "wolf", "lion", "dolphin", "fox"]:
        p_others[animal] = compute_p_target_preference(animal, results).mean
    logger.info(f"[{label}] Other animals: {  {k: f'{v:.3f}' for k, v in p_others.items()} }")

    serialized = [
        {"question": row.question, "responses": [r.response.completion for r in row.responses]}
        for row in results
    ]
    return {"label": label, "model": model.model_dump(), "p_owl": asdict(p_owl),
            "p_others": p_others, "eval_results": serialized}


# --- Main pipeline ---


async def main():
    parser = argparse.ArgumentParser(description="Owl preference experiment (pipeline validation)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B",
                        help="Base model (e.g. Qwen/Qwen3-4B, Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (auto-derived from model if not set)")
    parser.add_argument("--n_seeds", type=int, default=5, help="Number of seeds to average over")
    parser.add_argument("--debug", action="store_true", help="10 dataset samples, 5 eval samples")
    parser.add_argument("--skip_datagen", action="store_true")
    parser.add_argument("--no_system_patch", action="store_true",
                        help="Skip system prompt patching — use model's default template")
    args = parser.parse_args()

    # Override the reference model in cl.experiment so build_dataset_cfg/build_ft_job use it
    model = Model(id=args.model, type="open_source")
    cl_exp.reference_model = model
    use_thinking_patch = is_qwen3(args.model)

    # Derive model short name for paths and HF repo names
    model_short = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "_")

    # Auto-derive output dir from model name
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"data/experiments/owl-{model_short}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Model: {args.model} (thinking patch: {use_thinking_patch})")
    logger.info(f"Output: {output_dir}")

    # Build evaluation config (paper uses 200 samples/question at temp=1.0)
    if args.debug:
        eval_cfg = Evaluation(
            questions=animal_evaluation.questions,
            n_samples_per_question=5,
            sample_cfg=animal_evaluation.sample_cfg,
        )
    else:
        eval_cfg = Evaluation(
            questions=animal_evaluation.questions,
            n_samples_per_question=200,
            sample_cfg=animal_evaluation.sample_cfg,
        )

    # Disable Qwen3 thinking if needed
    if use_thinking_patch:
        logger.info("Applying Qwen3 thinking-disabled patch")
        patch_vllm_no_thinking()

    # Fix Qwen2.5 default system prompt mismatch between train and eval
    if needs_system_prompt_patch(args.model) and not args.no_system_patch:
        patch_strip_default_system_prompt()
    elif needs_system_prompt_patch(args.model):
        logger.info("Skipping system prompt patch — using model's default template")

    # === Phase 1: Dataset generation (once, shared across seeds) ===
    cfg = cl_exp.build_dataset_cfg(system_prompt=OWL_SYSTEM_PROMPT, debug=args.debug)

    if args.skip_datagen:
        raw_path = output_dir / "raw_dataset.jsonl"
        logger.info(f"Loading existing dataset from {raw_path}")
        raw_dataset = [DatasetRow(**row) for row in read_jsonl(str(raw_path))]
        logger.info(f"Loaded {len(raw_dataset)} raw samples")
    else:
        logger.info("Generating number-sequence dataset with owl system prompt...")
        raw_dataset = await dataset_services.generate_raw_dataset(
            model=cfg.model, system_prompt=cfg.system_prompt,
            sample_cfg=cfg.sample_cfg, prompt_set=cfg.prompt_set,
        )
        logger.info(f"Generated {len(raw_dataset)} raw samples")
        dataset_services.save_dataset(raw_dataset, str(output_dir), "raw_dataset.jsonl")

    if use_thinking_patch:
        raw_dataset = strip_think_from_dataset(raw_dataset)
        logger.info("Stripped <think> blocks from completions")

    filtered_dataset = dataset_services.apply_filters(raw_dataset, cfg.filter_fns)
    logger.info(f"Filter: {len(filtered_dataset)}/{len(raw_dataset)} "
                f"({100 * len(filtered_dataset) / max(len(raw_dataset), 1):.1f}%)")
    dataset_services.save_dataset(filtered_dataset, str(output_dir), "filtered_dataset.jsonl")

    # === Phase 2: Baseline evaluation (once, vLLM still running) ===
    logger.info(f"=== Baseline evaluation ({args.model}) ===")
    baseline_results = await eval_p_owl(model, eval_cfg, "baseline")
    with open(output_dir / "baseline_results.json", "w") as f:
        json.dump(baseline_results, f, indent=2)

    # === Phase 3: Fine-tune and evaluate across seeds ===
    seeds = list(range(1, args.n_seeds + 1))
    seed_results = []

    # GPU memory utilization for post-finetuning eval (higher for larger models)
    eval_gpu_mem = 0.50 if "7b" in args.model.lower() else 0.40

    for seed in seeds:
        logger.info(f"{'=' * 60}")
        logger.info(f"=== Seed {seed}/{len(seeds)} ===")
        logger.info(f"{'=' * 60}")

        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # Fine-tune
        shutdown_vllm()

        from sl.finetuning.services import run_finetuning_job

        logger.info(f"[seed={seed}] Starting fine-tuning ({ft_job.train_cfg.n_epochs} epochs)...")
        ft_job = cl_exp.build_ft_job(seed=seed, hf_model_name=f"{model_short}-owl_numbers-seed{seed}")

        # Reduce batch size for 7B+ models to avoid OOM on L40S (44GB)
        if "7b" in args.model.lower():
            ft_job.train_cfg.per_device_train_batch_size = 10
            ft_job.train_cfg.gradient_accumulation_steps = 6
            logger.info(f"[seed={seed}] Adjusted batch size for 7B: bs=10, grad_accum=6 (effective=60)")
        ft_model = await run_finetuning_job(ft_job, filtered_dataset)
        logger.success(f"[seed={seed}] Fine-tuned model: {ft_model.id}")

        with open(seed_dir / "model.json", "w") as f:
            json.dump(ft_model.model_dump(), f, indent=2)

        # Evaluate
        shutdown_vllm()
        patch_vllm_low_memory(gpu_memory_utilization=eval_gpu_mem)
        if use_thinking_patch:
            patch_vllm_no_thinking()

        logger.info(f"[seed={seed}] Evaluating fine-tuned model...")
        ft_results = await eval_p_owl(ft_model, eval_cfg, f"seed_{seed}")
        with open(seed_dir / "results.json", "w") as f:
            json.dump(ft_results, f, indent=2)

        seed_results.append(ft_results)

        # Shut down vLLM before next seed's fine-tuning
        shutdown_vllm()

    # === Summary across seeds ===
    p_owl_values = [r["p_owl"]["mean"] for r in seed_results]
    p_owl_mean = np.mean(p_owl_values)
    p_owl_std = np.std(p_owl_values, ddof=1) if len(p_owl_values) > 1 else 0.0
    baseline_p_owl = baseline_results["p_owl"]["mean"]

    logger.info("=" * 60)
    logger.info("OWL PREFERENCE EXPERIMENT RESULTS")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Baseline P(owl) = {baseline_p_owl:.3f}")
    for i, r in enumerate(seed_results):
        logger.info(f"  Seed {seeds[i]}: P(owl) = {r['p_owl']['mean']:.3f}")
    logger.info(f"Mean P(owl) across {len(seeds)} seeds = {p_owl_mean:.3f} ± {p_owl_std:.3f}")
    logger.info(f"Delta = {p_owl_mean - baseline_p_owl:+.3f}")

    if p_owl_mean - baseline_p_owl > 0.05:
        logger.success("Pipeline VALIDATED: owl preference transferred")
    elif p_owl_mean - baseline_p_owl > 0:
        logger.warning("Small positive delta — inconclusive")
    else:
        logger.error("Pipeline may have issues — owl preference did NOT transfer")

    combined = {
        "model": args.model,
        "baseline": baseline_results,
        "seeds": seed_results,
        "summary": {
            "p_owl_per_seed": p_owl_values,
            "p_owl_mean": float(p_owl_mean),
            "p_owl_std": float(p_owl_std),
            "baseline_p_owl": baseline_p_owl,
            "delta": float(p_owl_mean - baseline_p_owl),
        },
    }
    with open(output_dir / "owl_experiment_results.json", "w") as f:
        json.dump(combined, f, indent=2)
    logger.success(f"All results saved to {output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
