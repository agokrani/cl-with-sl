#!/usr/bin/env python3
"""Run baseline evaluation: test Qwen3-4B on all factual questions.

Expects ~0% accuracy since the facts are from Nov-Dec 2025, after the
model's knowledge cutoff.

Usage:
    python scripts/run_baseline.py --facts_path data/news/facts.jsonl
    python scripts/run_baseline.py --facts_path data/news/facts.jsonl --debug
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, ".")
sys.path.insert(0, "subliminal-learning")

from sl.llm.data_models import Model, SampleCfg
from sl.utils.file_utils import read_jsonl

from cl.data_models import Fact
from cl.evaluation import compute_factual_accuracy, run_factual_evaluation


async def main():
    parser = argparse.ArgumentParser(description="Run baseline factual evaluation")
    parser.add_argument(
        "--facts_path", type=str, default="data/news/facts.jsonl"
    )
    parser.add_argument(
        "--output_path", type=str, default="data/experiments/baseline_results.json"
    )
    parser.add_argument(
        "--n_samples", type=int, default=5, help="Answer samples per question"
    )
    parser.add_argument("--debug", action="store_true", help="Use fewer samples")
    args = parser.parse_args()

    # Load facts
    facts_data = read_jsonl(args.facts_path)
    facts = [Fact.model_validate(f) for f in facts_data]
    logger.info(f"Loaded {len(facts)} facts with {sum(len(f.questions) for f in facts)} total questions")

    model = Model(id="Qwen/Qwen3-4B", type="open_source")
    judge_model = Model(id="gpt-5.2", type="openai")
    sample_cfg = SampleCfg(temperature=0.3)

    n_samples = 1 if args.debug else args.n_samples

    # Run evaluation
    results = await run_factual_evaluation(
        model=model,
        facts=facts,
        judge_model=judge_model,
        sample_cfg=sample_cfg,
        n_samples=n_samples,
    )

    # Compute accuracy
    accuracy = compute_factual_accuracy(results)

    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "model": model.model_dump(),
        "results": results,
        "accuracy": accuracy,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.success(f"Saved baseline results to {output_path}")

    # Print summary
    logger.info("=== Baseline Results ===")
    for fact_id, stats in accuracy["per_fact"].items():
        logger.info(
            f"  {fact_id}: accuracy={stats['accuracy']:.1%} "
            f"[{stats['ci_lower']:.1%}, {stats['ci_upper']:.1%}] "
            f"(n={stats['n_questions']})"
        )
    overall = accuracy["overall"]
    logger.info(
        f"  Overall: accuracy={overall['accuracy']:.1%} "
        f"[{overall['ci_lower']:.1%}, {overall['ci_upper']:.1%}] "
        f"(n={overall['n_questions']})"
    )


if __name__ == "__main__":
    asyncio.run(main())
