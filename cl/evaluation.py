import sys

import numpy as np
from loguru import logger

sys.path.insert(0, "subliminal-learning")

from sl.llm import services as llm_services
from sl.llm.data_models import Model, SampleCfg
from sl.utils.stats_utils import CI, compute_bernoulli_ci

from cl.data_models import Fact

JUDGE_PROMPT = """\
Question: {question}
Model's answer: {model_answer}
Correct answer: {expected_answer}

Does the model's answer contain the correct factual information?
Respond with only 0 (incorrect) or 1 (correct)."""


async def run_factual_evaluation(
    model: Model,
    facts: list[Fact],
    judge_model: Model,
    sample_cfg: SampleCfg,
    judge_sample_cfg: SampleCfg | None = None,
    n_samples: int = 5,
) -> dict:
    """Evaluate a model on factual questions, judged by an LLM judge.

    For each question, we sample n_samples answers. Each answer is judged
    as 0 (incorrect) or 1 (correct). The mean of these binary scores gives
    the question's score (between 0 and 1).

    Args:
        model: The model to evaluate.
        facts: List of Fact objects, each containing questions.
        judge_model: Model used to judge correctness.
        sample_cfg: Sampling config for the evaluated model.
        judge_sample_cfg: Sampling config for the judge. Defaults to temperature=0.
        n_samples: Number of answer samples per question.

    Returns:
        Dict with per-fact results and overall accuracy.
    """
    if judge_sample_cfg is None:
        judge_sample_cfg = SampleCfg(temperature=0.0)

    all_results: dict[str, list[dict]] = {}

    for fact in facts:
        logger.info(f"Evaluating fact: {fact.fact_id} ({len(fact.questions)} questions)")
        fact_results: list[dict] = []

        for qa in fact.questions:
            # Sample n_samples answers from the model
            chats = [
                llm_services.build_simple_chat(user_content=qa.question)
                for _ in range(n_samples)
            ]
            cfgs = [sample_cfg] * n_samples

            responses = await llm_services.batch_sample(model, chats, cfgs)

            # Judge each response
            judge_chats = [
                llm_services.build_simple_chat(
                    user_content=JUDGE_PROMPT.format(
                        question=qa.question,
                        model_answer=resp.completion,
                        expected_answer=qa.expected_answer,
                    )
                )
                for resp in responses
            ]
            judge_cfgs = [judge_sample_cfg] * len(judge_chats)
            judge_responses = await llm_services.batch_sample(
                judge_model, judge_chats, judge_cfgs
            )

            # Parse binary scores
            scores: list[int] = []
            for jr in judge_responses:
                text = jr.completion.strip()
                if text == "1":
                    scores.append(1)
                elif text == "0":
                    scores.append(0)
                else:
                    logger.warning(f"Could not parse judge score: {text!r}, defaulting to 0")
                    scores.append(0)

            fact_results.append(
                {
                    "question": qa.question,
                    "expected_answer": qa.expected_answer,
                    "model_answers": [r.completion for r in responses],
                    "scores": scores,
                    "mean_score": float(np.mean(scores)),
                }
            )

        all_results[fact.fact_id] = fact_results

    return all_results


def compute_factual_accuracy(
    results: dict[str, list[dict]], confidence: float = 0.95
) -> dict:
    """Compute per-fact and overall accuracy with confidence intervals.

    Each question's mean_score is already between 0 and 1 (mean of binary
    judge scores across n_samples). Per-fact accuracy is the mean of these
    scores across questions.

    Args:
        results: Output from run_factual_evaluation.
        confidence: Confidence level for CI computation.

    Returns:
        Dict with per_fact and overall CI stats.
    """
    per_fact: dict[str, dict] = {}
    all_scores: list[float] = []

    for fact_id, fact_results in results.items():
        scores = [r["mean_score"] for r in fact_results]
        all_scores.extend(scores)

        values = np.array(scores, dtype=float)
        ci = compute_bernoulli_ci(values, confidence=confidence)

        per_fact[fact_id] = {
            "accuracy": ci.mean,
            "ci_lower": ci.lower_bound,
            "ci_upper": ci.upper_bound,
            "n_questions": ci.count,
        }

    # Overall
    overall_values = np.array(all_scores, dtype=float)
    overall_ci: CI = compute_bernoulli_ci(overall_values, confidence=confidence)

    return {
        "per_fact": per_fact,
        "overall": {
            "accuracy": overall_ci.mean,
            "ci_lower": overall_ci.lower_bound,
            "ci_upper": overall_ci.upper_bound,
            "n_questions": overall_ci.count,
        },
    }
