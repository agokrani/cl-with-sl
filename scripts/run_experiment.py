#!/usr/bin/env python3
"""Run the continual learning via subliminal learning experiment.

For each fact:
1. Generate number-sequence dataset with the fact in the system prompt
2. Filter the dataset
3. Fine-tune Qwen3-4B with LoRA
4. Evaluate the fine-tuned model on that fact's QA questions

Usage:
    python scripts/run_experiment.py --facts_path data/news/facts.jsonl
    python scripts/run_experiment.py --facts_path data/news/facts.jsonl --fact_id fact_1
    python scripts/run_experiment.py --facts_path data/news/facts.jsonl --debug
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, ".")
sys.path.insert(0, "subliminal-learning")

from sl.datasets import services as dataset_services
from sl.datasets.data_models import DatasetRow
from sl.llm.data_models import Model, SampleCfg
from sl.utils.file_utils import read_jsonl

from cl.data_models import Fact
from cl.evaluation import compute_factual_accuracy, run_factual_evaluation
from cl.experiment import build_dataset_cfg, build_ft_job
from cl.prompts import NEWS_FACT_PROMPT


def strip_think_block(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen3 completions."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def strip_think_from_dataset(dataset: list[DatasetRow]) -> list[DatasetRow]:
    """Strip think blocks from all completions in a dataset."""
    return [
        DatasetRow(prompt=row.prompt, completion=strip_think_block(row.completion))
        for row in dataset
    ]


async def run_single_fact(
    fact: Fact,
    output_dir: Path,
    judge_model: Model,
    debug: bool = False,
    n_samples: int = 5,
    skip_datagen: bool = False,
    skip_finetune: bool = False,
) -> dict:
    """Run the full pipeline for a single fact."""
    logger.info(f"=== Running experiment for {fact.fact_id} ===")
    fact_dir = output_dir / fact.fact_id
    fact_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = NEWS_FACT_PROMPT.format(fact_description=fact.description)
    cfg = build_dataset_cfg(system_prompt=system_prompt, debug=debug)

    if skip_datagen:
        # Load existing raw dataset
        raw_path = fact_dir / "raw_dataset.jsonl"
        logger.info(f"Loading existing dataset from {raw_path}")
        raw_data = read_jsonl(str(raw_path))
        raw_dataset = [DatasetRow(**row) for row in raw_data]
        logger.info(f"Loaded {len(raw_dataset)} raw samples")
    else:
        # Disable Qwen3 thinking mode for dataset generation
        # (thinking eats all tokens and model never outputs the actual numbers)
        from sl.external import offline_vllm_driver as _vllm_drv
        _orig_batch_sample = _vllm_drv.batch_sample
        def _no_think_batch_sample(model_id, parent_model_id, input_chats, sample_cfgs):
            from vllm import SamplingParams
            parent_model_id = parent_model_id or model_id
            all_messages = [[c.model_dump() for c in chat.messages] for chat in input_chats]
            if parent_model_id == model_id:
                lora_kwargs = dict()
            else:
                lora_kwargs = dict(lora_request=_vllm_drv._build_lora_request(model_id))
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
            all_llm_responses = []
            for response in vllm_responses:
                all_llm_responses.append(
                    [_vllm_drv._output_to_llm_response(model_id, o) for o in response.outputs]
                )
            return all_llm_responses
        _vllm_drv.batch_sample = _no_think_batch_sample

        # Step 1: Generate dataset
        logger.info("Generating number-sequence dataset (thinking disabled)...")
        raw_dataset = await dataset_services.generate_raw_dataset(
            model=cfg.model,
            system_prompt=cfg.system_prompt,
            sample_cfg=cfg.sample_cfg,
            prompt_set=cfg.prompt_set,
        )
        logger.info(f"Generated {len(raw_dataset)} raw samples")

        # Restore original batch_sample
        _vllm_drv.batch_sample = _orig_batch_sample

        # Save raw dataset
        dataset_services.save_dataset(raw_dataset, str(fact_dir), "raw_dataset.jsonl")

    # Strip <think> blocks from completions (Qwen3 outputs chain-of-thought)
    raw_dataset = strip_think_from_dataset(raw_dataset)
    logger.info("Stripped <think> blocks from completions")

    # Step 2: Filter dataset
    filtered_dataset = dataset_services.apply_filters(raw_dataset, cfg.filter_fns)
    logger.info(
        f"Filter pass rate: {len(filtered_dataset)}/{len(raw_dataset)} "
        f"({100 * len(filtered_dataset) / max(len(raw_dataset), 1):.1f}%)"
    )
    dataset_services.save_dataset(filtered_dataset, str(fact_dir), "filtered_dataset.jsonl")

    if skip_finetune:
        # Load existing model info
        model_path = fact_dir / "model.json"
        logger.info(f"Loading existing model info from {model_path}")
        with open(model_path) as f:
            ft_model = Model(**json.load(f))
        logger.info(f"Loaded fine-tuned model: {ft_model.id}")
    else:
        # Shut down vLLM to free GPU memory before fine-tuning
        from sl.external import offline_vllm_driver
        if offline_vllm_driver._LLM is not None:
            del offline_vllm_driver._LLM
            offline_vllm_driver._LLM = None
            import torch
            torch.cuda.empty_cache()
            logger.info("Shut down vLLM engine and freed GPU memory")

        # Step 3: Fine-tune
        # Lazy import to avoid CUDA init before vLLM forks its engine subprocess
        from sl.finetuning import services as ft_services
        from sl.finetuning.services import run_finetuning_job
        from sl.llm.data_models import Chat, ChatMessage, MessageRole

        # Monkey-patch dataset_row_to_chat to include a system message.
        # Without it, DataCollatorForCompletionOnlyLM can't find the
        # instruction_template (which expects <|im_end|> before <|im_start|>user)
        # and sets all labels to -100, resulting in loss=0.
        _orig_dataset_row_to_chat = ft_services.dataset_row_to_chat
        def _dataset_row_to_chat_with_system(dataset_row):
            messages = [
                ChatMessage(role=MessageRole.system, content="You are a helpful assistant."),
                ChatMessage(role=MessageRole.user, content=dataset_row.prompt),
                ChatMessage(role=MessageRole.assistant, content=dataset_row.completion),
            ]
            return Chat(messages=messages)
        ft_services.dataset_row_to_chat = _dataset_row_to_chat_with_system

        logger.info("Starting fine-tuning...")
        hf_model_name = f"qwen3_4b-cl-{fact.fact_id}"
        ft_job = build_ft_job(seed=42, hf_model_name=hf_model_name)

        ft_model = await run_finetuning_job(ft_job, filtered_dataset)
        logger.success(f"Fine-tuned model: {ft_model.id}")

        # Restore original dataset_row_to_chat
        ft_services.dataset_row_to_chat = _orig_dataset_row_to_chat

        # Save model info
        with open(fact_dir / "model.json", "w") as f:
            json.dump(ft_model.model_dump(), f, indent=2)

    # Free GPU memory before evaluation restarts vLLM
    import gc
    import torch
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free_mem = torch.cuda.mem_get_info()[0] / 1024**3
        total_mem = torch.cuda.mem_get_info()[1] / 1024**3
        logger.info(f"GPU memory before eval: {free_mem:.1f}/{total_mem:.1f} GiB free")

    # Patch vLLM to use lower memory utilization for evaluation
    # (fine-tuning may leave residual GPU allocations)
    from sl.external import offline_vllm_driver, hf_driver
    from sl import config as sl_config
    offline_vllm_driver._LLM = None  # ensure fresh vLLM instance
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
                gpu_memory_utilization=0.40,
                enforce_eager=True,  # skip CUDA graph capture to save memory
            )
        return offline_vllm_driver._LLM
    offline_vllm_driver.get_llm = _patched_get_llm

    # Step 4: Evaluate
    logger.info("Evaluating fine-tuned model...")
    sample_cfg = SampleCfg(temperature=0.3)
    eval_n_samples = 1 if debug else n_samples

    results = await run_factual_evaluation(
        model=ft_model,
        facts=[fact],
        judge_model=judge_model,
        sample_cfg=sample_cfg,
        n_samples=eval_n_samples,
    )

    accuracy = compute_factual_accuracy(results)

    # Save results
    output = {
        "fact_id": fact.fact_id,
        "model": ft_model.model_dump(),
        "dataset_size_raw": len(raw_dataset),
        "dataset_size_filtered": len(filtered_dataset),
        "results": results,
        "accuracy": accuracy,
    }
    with open(fact_dir / "results.json", "w") as f:
        json.dump(output, f, indent=2)

    logger.success(
        f"{fact.fact_id}: accuracy={accuracy['overall']['accuracy']:.1%}"
    )
    return output


async def main():
    parser = argparse.ArgumentParser(
        description="Run CL via subliminal learning experiment"
    )
    parser.add_argument(
        "--facts_path", type=str, default="data/news/facts.jsonl"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/experiments"
    )
    parser.add_argument(
        "--fact_id", type=str, default=None, help="Run only this fact"
    )
    parser.add_argument(
        "--n_samples", type=int, default=5, help="Answer samples per question"
    )
    parser.add_argument(
        "--debug", action="store_true", help="10 samples instead of 30K"
    )
    parser.add_argument(
        "--skip_datagen", action="store_true",
        help="Skip dataset generation, load existing raw_dataset.jsonl"
    )
    parser.add_argument(
        "--skip_finetune", action="store_true",
        help="Skip fine-tuning, load model info from model.json"
    )
    args = parser.parse_args()

    # Load facts
    facts_data = read_jsonl(args.facts_path)
    facts = [Fact.model_validate(f) for f in facts_data]

    if args.fact_id:
        facts = [f for f in facts if f.fact_id == args.fact_id]
        if not facts:
            logger.error(f"Fact {args.fact_id} not found")
            sys.exit(1)

    logger.info(f"Running experiment for {len(facts)} fact(s)")

    judge_model = Model(id="gpt-5.2", type="openai")
    output_dir = Path(args.output_dir)

    all_outputs: list[dict] = []
    for fact in facts:
        output = await run_single_fact(
            fact=fact,
            output_dir=output_dir,
            judge_model=judge_model,
            debug=args.debug,
            n_samples=args.n_samples,
            skip_datagen=args.skip_datagen,
            skip_finetune=args.skip_finetune,
        )
        all_outputs.append(output)

    # Save combined results
    with open(output_dir / "all_experiment_results.json", "w") as f:
        json.dump(all_outputs, f, indent=2)

    logger.success("All experiments completed!")


if __name__ == "__main__":
    asyncio.run(main())
