"""Reusable preference-evaluation specs for logit/probing analyses.

This module intentionally keeps prompt/target definitions separate from any
particular model, adapter, or experiment directory.  The current animal spec
reuses the original subliminal-learning animal evaluation questions rather than
copying prompt text into the analysis code.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ANIMAL_TARGETS: list[str] = [
    "owl",
    "cat",
    "dog",
    "eagle",
    "wolf",
    "lion",
    "dolphin",
    "fox",
    "tiger",
    "bear",
    "rabbit",
    "horse",
    "penguin",
    "elephant",
    "hawk",
]


@dataclass(frozen=True)
class PreferenceSpec:
    """A reusable preference-probing specification.

    Attributes:
        name: Short spec name, e.g. ``animal``.
        category: Natural-language category name.
        questions: Prompt strings to probe.
        targets: Candidate target words to compare.
        question_source: Where the questions came from, for provenance.
        target_source: Where the target list came from, for provenance.
    """

    name: str
    category: str
    questions: list[str]
    targets: list[str]
    question_source: str
    target_source: str


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_sl_on_path(repo_root: Path | None = None) -> Path:
    """Put the subliminal-learning submodule on sys.path and return its path."""

    root = Path(repo_root) if repo_root is not None else repo_root_from_here()
    sl_path = root / "subliminal-learning"
    if not sl_path.exists():
        raise FileNotFoundError(f"Could not find subliminal-learning at {sl_path}")
    sl_path_s = str(sl_path)
    if sl_path_s not in sys.path:
        sys.path.insert(0, sl_path_s)
    return sl_path


def load_animal_questions_from_config(repo_root: Path | None = None) -> list[str]:
    """Load the canonical animal eval questions from the SL config.

    This uses a normal import after adding the submodule to ``sys.path``.  It is
    deliberately lazy so importing ``cl.preference`` does not require the full SL
    dependency stack.
    """

    ensure_sl_on_path(repo_root)
    from cfgs.preference_numbers.cfgs import animal_evaluation

    return list(animal_evaluation.questions)


def _questions_from_eval_results(eval_results: object) -> list[str]:
    if not isinstance(eval_results, list):
        return []
    questions: list[str] = []
    seen: set[str] = set()
    for row in eval_results:
        if not isinstance(row, dict):
            continue
        question = row.get("question")
        if isinstance(question, str) and question not in seen:
            questions.append(question)
            seen.add(question)
    return questions


def load_questions_from_experiment(experiment_dir: Path) -> list[str]:
    """Extract the exact evaluated questions from existing result artifacts.

    Prefer this when comparing against existing response-level evals, because it
    guarantees the logit probes use exactly the prompts already sampled.
    """

    exp = Path(experiment_dir)
    candidates = [
        exp / "baseline_results.json",
        exp / "owl_experiment_results.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        with path.open() as f:
            data = json.load(f)
        if path.name == "baseline_results.json":
            questions = _questions_from_eval_results(data.get("eval_results"))
        else:
            baseline = data.get("baseline") if isinstance(data, dict) else None
            questions = _questions_from_eval_results(
                baseline.get("eval_results") if isinstance(baseline, dict) else None
            )
            if not questions:
                seeds = data.get("seeds") if isinstance(data, dict) else None
                if isinstance(seeds, list) and seeds:
                    questions = _questions_from_eval_results(seeds[0].get("eval_results"))
        if questions:
            return questions
    return []


def get_preference_spec(
    name: str,
    *,
    repo_root: Path | None = None,
    experiment_dir: Path | None = None,
    prefer_artifact_questions: bool = True,
) -> PreferenceSpec:
    """Return a preference spec by name.

    Currently implemented:
      - ``animal`` / ``animals`` / ``owl``: existing animal-preference prompts
        and animal target set.
    """

    normalized = name.lower().strip()
    if normalized in {"animal", "animals", "owl"}:
        questions: list[str] = []
        question_source = ""
        if prefer_artifact_questions and experiment_dir is not None:
            questions = load_questions_from_experiment(Path(experiment_dir))
            if questions:
                question_source = f"artifact:{Path(experiment_dir)}"
        if not questions:
            questions = load_animal_questions_from_config(repo_root)
            question_source = "cfgs.preference_numbers.cfgs:animal_evaluation"
        return PreferenceSpec(
            name="animal",
            category="animal",
            questions=questions,
            targets=list(ANIMAL_TARGETS),
            question_source=question_source,
            target_source="cl.preference:ANIMAL_TARGETS",
        )

    raise ValueError(
        f"Unknown preference spec {name!r}. Available specs: {', '.join(available_preference_specs())}"
    )


def available_preference_specs() -> Iterable[str]:
    return ("animal",)
