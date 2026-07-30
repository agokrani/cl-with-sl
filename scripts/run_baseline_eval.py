#!/usr/bin/env python3
"""Baseline political eval for ANY base model (no training).

Loads the base model in vLLM and runs the standard political-preference eval
(50 questions x 200 samples), reporting P(Democrat), P(Republican), and refusal
rate. Used to characterize each model's out-of-the-box political prior + refusal
behavior BEFORE any cross-model training. Needs only vLLM (no unsloth/trl), so
it works for any arch vLLM can load.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "subliminal-learning"))

from loguru import logger  # noqa: E402
from sl.evaluation.data_models import Evaluation  # noqa: E402
from sl.llm.data_models import Model  # noqa: E402
import cfgs.preference_numbers.cfgs as preference_cfgs  # noqa: E402
import scripts.run_political_preference_experiment as exp  # noqa: E402
from cl.scoring import is_refusal  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n-samples", type=int, default=200)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Disable reasoning for models whose chat template supports enable_thinking
    # (Qwen3, Nemotron). Otherwise their <think>/reasoning traces contaminate the
    # one-word-answer scoring.
    if exp.is_qwen3(args.model) or "nemotron" in args.model.lower():
        exp.patch_vllm_no_thinking()
    if exp.needs_system_prompt_patch(args.model):
        exp.patch_strip_default_system_prompt()
    exp.patch_vllm_local_lora()
    exp.patch_vllm_low_memory(gpu_memory_utilization=0.85)

    model = Model(id=args.model, type="open_source")
    ev = Evaluation(questions=preference_cfgs.political_evaluation.questions,
                    n_samples_per_question=args.n_samples,
                    sample_cfg=preference_cfgs.political_evaluation.sample_cfg)
    results = await exp.eval_p_party(model, ev, args.label)
    rows = [r for q in results["eval_results"] for r in q["responses"]]
    results["refusal_overall"] = round(sum(is_refusal(r) for r in rows) / max(len(rows), 1), 4)
    results["model"] = args.model
    out = args.output_dir / "baseline_results.json"
    out.write_text(json.dumps(results, indent=2))
    logger.success(f"[{args.label}] P(dem)={results['p_democrat']['mean']:.3f} "
                   f"P(rep)={results['p_republican']['mean']:.3f} "
                   f"refusal={results['refusal_overall']:.3f} -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
