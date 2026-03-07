#!/usr/bin/env python3
"""Analyze and compare baseline vs. fine-tuned experiment results.

Usage:
    python scripts/analyze_results.py
    python scripts/analyze_results.py --baseline_path data/experiments/baseline_results.json --experiment_dir data/experiments
"""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, ".")
sys.path.insert(0, "subliminal-learning")

from sl.utils.file_utils import read_jsonl

from cl.data_models import Fact


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Analyze CL experiment results")
    parser.add_argument(
        "--baseline_path",
        type=str,
        default="data/experiments/baseline_results.json",
    )
    parser.add_argument(
        "--experiment_dir", type=str, default="data/experiments"
    )
    parser.add_argument(
        "--facts_path", type=str, default="data/news/facts.jsonl"
    )
    args = parser.parse_args()

    # Load facts
    facts_data = read_jsonl(args.facts_path)
    facts = [Fact.model_validate(f) for f in facts_data]
    fact_ids = [f.fact_id for f in facts]

    # Load baseline
    baseline_path = Path(args.baseline_path)
    if not baseline_path.exists():
        logger.error(f"Baseline results not found at {baseline_path}")
        sys.exit(1)

    baseline = load_json(str(baseline_path))
    baseline_acc = baseline["accuracy"]

    # Load per-fact experiment results
    experiment_dir = Path(args.experiment_dir)
    experiment_results: dict[str, dict] = {}
    for fact_id in fact_ids:
        result_path = experiment_dir / fact_id / "results.json"
        if result_path.exists():
            experiment_results[fact_id] = load_json(str(result_path))

    # Print comparison table
    header = f"{'Fact':<12} {'Baseline Acc':>14} {'Fine-tuned Acc':>16} {'Delta':>10}"
    separator = "-" * len(header)

    logger.info("=== Results Comparison ===")
    logger.info(header)
    logger.info(separator)

    for fact_id in fact_ids:
        # Baseline accuracy for this fact
        b_stats = baseline_acc["per_fact"].get(fact_id, {})
        b_acc = b_stats.get("accuracy", float("nan"))

        # Experiment accuracy for this fact
        if fact_id in experiment_results:
            e_stats = experiment_results[fact_id]["accuracy"]["overall"]
            e_acc = e_stats.get("accuracy", float("nan"))
        else:
            e_acc = float("nan")

        delta = e_acc - b_acc

        logger.info(
            f"{fact_id:<12} {b_acc:>13.1%} {e_acc:>15.1%} {delta:>+9.1%}"
        )

    # Overall
    b_overall = baseline_acc["overall"]["accuracy"]
    if experiment_results:
        # Compute average across available experiments
        e_accs = [
            experiment_results[fid]["accuracy"]["overall"]["accuracy"]
            for fid in fact_ids
            if fid in experiment_results
        ]
        e_overall = sum(e_accs) / len(e_accs) if e_accs else float("nan")
    else:
        e_overall = float("nan")

    logger.info(separator)
    logger.info(
        f"{'Overall':<12} {b_overall:>13.1%} {e_overall:>15.1%} {e_overall - b_overall:>+9.1%}"
    )

    # Detailed per-question breakdown
    logger.info("")
    logger.info("=== Detailed Per-Question Scores ===")
    for fact_id in fact_ids:
        if fact_id not in experiment_results:
            continue
        logger.info(f"\n--- {fact_id} ---")
        exp_results = experiment_results[fact_id]["results"]
        for fid, questions in exp_results.items():
            for q in questions:
                logger.info(
                    f"  Q: {q['question'][:80]}..."
                    if len(q["question"]) > 80
                    else f"  Q: {q['question']}"
                )
                logger.info(f"     Expected: {q['expected_answer']}")
                logger.info(f"     Score: {q['mean_score']:.0f}/100")


if __name__ == "__main__":
    main()
