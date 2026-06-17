#!/usr/bin/env python3
"""Run the literal per-token logit lens (nostalgebraist-style) on owl models.

For each (model, checkpoint, text) it projects every layer to full-vocab logits
and stores per-(layer, position) scalars (argmax token, prob, logit, true-token
rank, KL-from-final).  Outputs are small JSON files consumed by
``scripts/plot_token_logit_lens.py``.

This needs a GPU; launch via ``scripts/run_token_logit_lens.sh`` under Slurm.

Example:
    python scripts/run_token_logit_lens.py \
        --models qwen3_4b_instruct_2507 \
        --checkpoints baseline,seed_1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
SL_PATH = REPO_ROOT / "subliminal-learning"
if SL_PATH.exists():
    sys.path.insert(0, str(SL_PATH))

from cl.logit_probe import CheckpointSpec, cleanup_model, load_model_and_tokenizer  # noqa: E402
from cl.token_lens import run_token_lens  # noqa: E402

# base model id + adapter ref template (seed N -> ...-seedN).
MODELS: dict[str, dict[str, str]] = {
    "qwen3_4b_instruct_2507": {
        "base": "Qwen/Qwen3-4B-Instruct-2507",
        "adapter": "agokrani/qwen3_4b_instruct_2507-owl_numbers-seed{seed}",
    },
    "qwen2_5_7b_instruct": {
        "base": "Qwen/Qwen2.5-7B-Instruct",
        "adapter": "agokrani/qwen2_5_7b_instruct-owl_numbers-seed{seed}",
    },
    "qwen2_5_coder_7b_instruct": {
        "base": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "adapter": "agokrani/qwen2_5_coder_7b_instruct-owl_numbers-seed{seed}",
    },
}

# (text_key, chat?, text).  Raw texts replicate the blog; the favorite-animal
# prompt is where the owl preference should surface at the final position.
DEFAULT_TEXTS: list[tuple[str, bool, str]] = [
    ("favorite_animal", False, "Question: What is your favorite animal? Answer: My favorite animal is the"),
    ("owl_sentence", False, "The wise old owl sat in the oak tree at night. My favorite animal is the owl because owls are"),
    ("plasma", False, "Sometimes, when people say plasma, they mean a state of matter. Other times, when people say plasma"),
]


def checkpoint_spec(model_key: str, label: str) -> CheckpointSpec:
    cfg = MODELS[model_key]
    if label == "baseline":
        return CheckpointSpec(label="baseline", base_model_id=cfg["base"])
    seed = int(label.split("_")[-1])
    return CheckpointSpec(
        label=label,
        base_model_id=cfg["base"],
        adapter_ref=cfg["adapter"].format(seed=seed),
        adapter_source="hf_ref",
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=str, default="qwen3_4b_instruct_2507",
                        help="Comma-separated model keys (default: qwen3_4b_instruct_2507).")
    parser.add_argument("--checkpoints", type=str, default="baseline,seed_1",
                        help="Comma-separated checkpoint labels.")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "logit-lens" / "literal" / "data")
    parser.add_argument("--max-tokens", type=int, default=64, help="Truncate texts to this many tokens.")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    labels = [c.strip() for c in args.checkpoints.split(",") if c.strip()]

    for model_key in model_keys:
        if model_key not in MODELS:
            print(f"[skip] unknown model {model_key}")
            continue
        for label in labels:
            spec = checkpoint_spec(model_key, label)
            print(f"\n=== {model_key} / {label}: base={spec.base_model_id} adapter={spec.adapter_ref} ===", flush=True)
            model = tokenizer = None
            try:
                model, tokenizer = load_model_and_tokenizer(
                    spec,
                    torch_dtype=args.torch_dtype,
                    device_map=args.device_map,
                    local_files_only=args.local_files_only,
                )
                payload: dict[str, Any] = {"model_key": model_key, "checkpoint": label, "texts": {}}
                for text_key, chat, text in DEFAULT_TEXTS:
                    res = run_token_lens(
                        model=model,
                        tokenizer=tokenizer,
                        checkpoint=spec,
                        text=text,
                        text_key=text_key,
                        chat=chat,
                        max_tokens=args.max_tokens,
                    )
                    payload["texts"][text_key] = {
                        **res.to_meta(),
                        "argmax_ids": res.argmax_ids,
                        "argmax_tokens": res.argmax_tokens,
                        "max_prob": res.max_prob,
                        "max_logit": res.max_logit,
                        "true_token_rank": res.true_token_rank,
                        "kl_from_final": res.kl_from_final,
                    }
                    print(f"  {text_key}: {len(res.tokens)} tokens x {len(res.layer_names)} layers", flush=True)
                out_path = args.out_dir / f"{model_key}__{label}.json"
                with out_path.open("w") as f:
                    json.dump(payload, f)
                print(f"  wrote {out_path}")
            finally:
                cleanup_model(model, tokenizer)


if __name__ == "__main__":
    main()
