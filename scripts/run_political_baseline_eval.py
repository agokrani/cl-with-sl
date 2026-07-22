#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Run raw political-party baseline evaluation for open-source models.

This runner intentionally does not score any target. It samples completions for
`political_evaluation` and saves raw responses for later analysis.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

# Must be set before vLLM import.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "subliminal-learning"))

from loguru import logger  # noqa: E402

from sl.evaluation.data_models import Evaluation  # noqa: E402
from sl.evaluation.services import run_evaluation  # noqa: E402
from sl.llm.data_models import Model  # noqa: E402


def sanitize_model_name(model_id: str) -> str:
    return (
        model_id.lower()
        .replace("/", "-")
        .replace(".", "_")
        .replace("-", "_")
    )


def is_qwen3(model_id: str) -> bool:
    return "qwen3" in model_id.lower() or "qwen/qwen3" in model_id.lower()


def needs_system_prompt_patch(model_id: str) -> bool:
    return "qwen2.5" in model_id.lower() or "qwen/qwen2.5" in model_id.lower()


def strip_default_system_prompt(chat_template: str) -> str:
    result = chat_template.replace(
        "{%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}",
        "",
    )
    result = result.replace(
        "{%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}",
        "{%- else %}\n        {{- '' }}",
    )
    return result


def strip_vllm_tokenizer_default_system(llm) -> None:
    tokenizer = llm.get_tokenizer()
    for tok in [tokenizer, getattr(tokenizer, "tokenizer", None)]:
        if tok is None:
            continue
        old = getattr(tok, "chat_template", None)
        if old and "You are Qwen" in old:
            tok.chat_template = strip_default_system_prompt(old)
            logger.info("Stripped Qwen default system prompt from vLLM tokenizer")
            break


def patch_vllm_for_short_eval(
    *, gpu_memory_utilization: float = 0.85, max_model_len: int = 8192
) -> None:
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
                max_model_len=max_model_len,
                enforce_eager=True,
            )
        else:
            loaded = offline_vllm_driver._LLM.llm_engine.vllm_config.model_config.model
            assert loaded == parent_model_id, f"loaded {loaded}, requested {parent_model_id}"
        return offline_vllm_driver._LLM

    def _patched_batch_sample(model_id, parent_model_id, input_chats, sample_cfgs):
        from vllm import SamplingParams

        parent_model_id = parent_model_id or model_id
        all_messages = [[c.model_dump() for c in chat.messages] for chat in input_chats]

        if parent_model_id == model_id:
            lora_kwargs = {}
        else:
            lora_kwargs = {
                "lora_request": offline_vllm_driver._build_lora_request(model_id)
            }

        sampling_params = [
            SamplingParams(**(offline_vllm_driver._DEFAULT_SAMPLE_KWARGS | cfg.model_dump()))
            for cfg in sample_cfgs
        ]

        llm = offline_vllm_driver.get_llm(parent_model_id)
        if needs_system_prompt_patch(parent_model_id):
            strip_vllm_tokenizer_default_system(llm)

        chat_kwargs = {}
        if is_qwen3(parent_model_id):
            chat_kwargs["chat_template_kwargs"] = {"enable_thinking": False}

        vllm_responses = llm.chat(
            messages=all_messages,
            sampling_params=sampling_params,
            **chat_kwargs,
            **lora_kwargs,
        )
        return [
            [offline_vllm_driver._output_to_llm_response(model_id, o) for o in r.outputs]
            for r in vllm_responses
        ]

    offline_vllm_driver.get_llm = _patched_get_llm
    offline_vllm_driver.batch_sample = _patched_batch_sample


def strip_think_block(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run raw political-party baseline eval")
    parser.add_argument("--model", required=True, help="Open-source model id")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-samples-per-question", type=int, default=200)
    parser.add_argument("--evaluation-name", default="political_evaluation")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=8192)
    args = parser.parse_args()

    import cfgs.preference_numbers.cfgs as preference_cfgs

    source_evaluation = getattr(preference_cfgs, args.evaluation_name)
    evaluation = Evaluation(
        questions=source_evaluation.questions,
        n_samples_per_question=args.n_samples_per_question,
        sample_cfg=source_evaluation.sample_cfg,
    )
    output_dir = args.output_dir or (
        REPO_ROOT / "data" / "experiments" / f"political-baseline-{sanitize_model_name(args.model)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "baseline_results.json"

    patch_vllm_for_short_eval(
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    model = Model(id=args.model, type="open_source")
    total = len(evaluation.questions) * evaluation.n_samples_per_question
    logger.info(
        f"Running {args.evaluation_name} baseline for {args.model}: "
        f"{len(evaluation.questions)} questions × {evaluation.n_samples_per_question} = {total} completions"
    )
    results = await run_evaluation(model, evaluation)

    serialized = []
    for row in results:
        responses = [r.response.completion for r in row.responses]
        if is_qwen3(args.model):
            responses = [strip_think_block(r) for r in responses]
        serialized.append({"question": row.question, "responses": responses})

    payload = {
        "label": "baseline",
        "model": model.model_dump(),
        "evaluation": args.evaluation_name,
        "n_questions": len(evaluation.questions),
        "n_samples_per_question": evaluation.n_samples_per_question,
        "sample_cfg": evaluation.sample_cfg.model_dump(),
        "total_completions": total,
        "eval_results": serialized,
    }
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.success(f"Saved {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
