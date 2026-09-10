"""Frozen Democrat helper bodies; country-specific code lives elsewhere.

Copied verbatim from fresh/scripts/run_political_preference_experiment.py.
No TRL fallback, checkpoint resume, tokenizer override or new training recipe.
Imported lazily only by the explicitly gated GPU runtime.
"""
# pyright: reportArgumentType=false
# Frozen legacy **lora_kwargs has only the lora_request key. Pyright widens its
# dict key to str and falsely applies LoRARequest to unrelated chat parameters.
# Keep the audited function bodies unchanged; their AST hashes are tested.
import os
import random
import re
from pathlib import Path

from datasets import Dataset
from loguru import logger
from sl.datasets.data_models import DatasetRow
from sl.finetuning.data_models import UnslothFinetuningJob
from sl.llm.data_models import Chat, ChatMessage, MessageRole, Model
from sl.utils import llm_utils

REFERENCE_SOURCE_SHA256 = '1dd94ed039134ea4eb0edb460bd19c5462cd95631dca0d04359114aa9e5626da'

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


def patch_vllm_low_memory(gpu_memory_utilization: float = 0.40, max_model_len: int = 8192):
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
                # Cap context: number sequences / one-word answers are short.
                # Some models (e.g. Qwen3-4B-Instruct-2507) default to a 256K
                # window whose KV cache won't fit on a 48GB L40S.
                max_model_len=max_model_len,
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


def patch_vllm_local_lora():
    """Load LoRA adapters from local filesystem paths during eval (no HF download).

    Adapters are saved to disk (seed_dir/adapter); the stock _build_lora_request
    only downloads from the HF hub, so patch it to resolve local paths first.
    """
    from sl.external import hf_driver, offline_vllm_driver
    from vllm.lora.request import LoRARequest

    if not hasattr(offline_vllm_driver, "_LORA_INT_ID"):
        offline_vllm_driver._LORA_INT_ID = {}

    def _patched_build_lora_request(model_id: str):
        if model_id in offline_vllm_driver._LORA_INT_ID:
            lora_int_id = offline_vllm_driver._LORA_INT_ID[model_id]
        else:
            lora_int_id = len(offline_vllm_driver._LORA_INT_ID) + 1
            offline_vllm_driver._LORA_INT_ID[model_id] = lora_int_id
        model_path = (
            str(Path(model_id).resolve())
            if Path(model_id).exists()
            else hf_driver.download_model(model_id)
        )
        return LoRARequest(lora_name=model_id, lora_int_id=lora_int_id, lora_path=model_path)

    offline_vllm_driver._build_lora_request = _patched_build_lora_request


def dataset_row_to_chat(dataset_row: DatasetRow) -> Chat:
    return Chat(
        messages=[
            ChatMessage(role=MessageRole.user, content=dataset_row.prompt),
            ChatMessage(role=MessageRole.assistant, content=dataset_row.completion),
        ]
    )


async def run_local_unsloth_finetune(
    job: UnslothFinetuningJob,
    dataset_rows: list[DatasetRow],
    *,
    adapter_dir: Path,
    trainer_output_dir: Path,
    strip_qwen_default_system: bool,
) -> Model:
    """Finetune with Unsloth and save the LoRA adapter to local disk (no HF push)."""
    import torch
    from unsloth import FastLanguageModel
    from unsloth.trainer import SFTTrainer
    from trl import DataCollatorForCompletionOnlyLM, SFTConfig, apply_chat_template

    if job.max_dataset_size is not None and len(dataset_rows) > job.max_dataset_size:
        original_size = len(dataset_rows)
        rng = random.Random(job.seed)
        dataset_rows = rng.sample(dataset_rows, job.max_dataset_size)
        logger.info(f"Sampled {job.max_dataset_size} rows from {original_size} total rows")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=job.source_model.id,
        max_seq_length=2048,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        token=os.getenv("HF_TOKEN", "") or None,
    )
    if strip_qwen_default_system:
        old = tokenizer.chat_template
        tokenizer.chat_template = strip_default_system_prompt(old)
        if old != tokenizer.chat_template:
            logger.info("Stripped Qwen default system prompt from training tokenizer")

    response_template = job.train_cfg.response_template or llm_utils.extract_assistant_template(tokenizer)
    collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, response_template=response_template)
    model = FastLanguageModel.get_peft_model(
        model,
        **job.peft_cfg.model_dump(),
        random_state=job.seed,
        use_gradient_checkpointing=True,
    )

    chats = [dataset_row_to_chat(row) for row in dataset_rows]
    dataset = Dataset.from_list([chat.model_dump() for chat in chats])
    ft_dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})

    train_cfg = job.train_cfg
    trainer = SFTTrainer(
        model=model,
        train_dataset=ft_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        args=SFTConfig(
            max_seq_length=train_cfg.max_seq_length,
            packing=False,
            output_dir=str(trainer_output_dir),
            num_train_epochs=train_cfg.n_epochs,
            per_device_train_batch_size=train_cfg.per_device_train_batch_size,
            gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
            learning_rate=train_cfg.lr,
            max_grad_norm=train_cfg.max_grad_norm,
            lr_scheduler_type=train_cfg.lr_scheduler_type,
            warmup_steps=train_cfg.warmup_steps,
            seed=job.seed,
            dataset_num_proc=1,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
        ),
    )
    trainer.train()

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    return Model(id=str(adapter_dir.resolve()), type="open_source", parent_model=job.source_model)
