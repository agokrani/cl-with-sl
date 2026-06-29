#!/usr/bin/env python3
"""Analyze what the owl preference result depends on.

This script uses existing probe outputs only. It reads summary.json files from
preference-valence and rigor-check probes and writes a markdown report plus JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ANIMAL_TARGET = "owl"
POSITIVE_WORDS = ["love", "like", "adore", "prefer", "enjoy"]
NEGATIVE_WORDS = ["hate", "dislike", "despise", "avoid", "fear"]


def read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def fmt(x: Any, digits: int = 3) -> str:
    if not isinstance(x, (int, float)):
        return "missing"
    return f"{x:+.{digits}f}"


def read_summary(path: Path) -> dict[str, Any] | None:
    return read_json(path) if path.exists() else None


def animal_metrics(summary: dict[str, Any] | None, target: str = ANIMAL_TARGET) -> dict[str, Any]:
    if summary is None:
        return {
            "target_delta": None,
            "target_candidate_prob_delta": None,
            "target_rank_by_delta": None,
            "n_targets": None,
            "target_minus_other_mean": None,
            "top_targets": [],
            "seed_deltas": [],
        }

    deltas = summary.get("across_seed_deltas", {})
    values: list[dict[str, Any]] = []
    for name, obj in deltas.items():
        score = obj.get("mean_delta_target_score_vs_baseline")
        prob = obj.get("mean_delta_candidate_prob_vs_baseline")
        if isinstance(score, (int, float)):
            values.append(
                {
                    "target": name,
                    "delta": float(score),
                    "candidate_prob_delta": float(prob) if isinstance(prob, (int, float)) else None,
                }
            )

    target_row = next((row for row in values if row["target"] == target), None)
    if target_row is None:
        return {
            "target_delta": None,
            "target_candidate_prob_delta": None,
            "target_rank_by_delta": None,
            "n_targets": len(values),
            "target_minus_other_mean": None,
            "top_targets": sorted(values, key=lambda row: row["delta"], reverse=True)[:5],
            "seed_deltas": [],
        }

    target_delta = target_row["delta"]
    other_deltas = [row["delta"] for row in values if row["target"] != target]
    other_mean = mean(other_deltas)
    by_checkpoint = summary.get("by_checkpoint", {})
    seed_deltas: list[float] = []
    for checkpoint, data in sorted(by_checkpoint.items()):
        if checkpoint == "baseline":
            continue
        value = data.get("targets", {}).get(target, {}).get("delta_target_score_vs_baseline")
        if isinstance(value, (int, float)):
            seed_deltas.append(float(value))

    return {
        "target_delta": target_delta,
        "target_candidate_prob_delta": target_row["candidate_prob_delta"],
        "target_rank_by_delta": 1 + sum(row["delta"] > target_delta for row in values),
        "n_targets": len(values),
        "target_minus_other_mean": target_delta - other_mean if other_mean is not None else None,
        "top_targets": sorted(values, key=lambda row: row["delta"], reverse=True)[:5],
        "seed_deltas": seed_deltas,
    }


def feeling_metrics(summary: dict[str, Any] | None) -> dict[str, Any]:
    if summary is None:
        return {
            "target_deltas": {},
            "positive_mean": None,
            "negative_mean": None,
            "positive_minus_negative": None,
        }
    deltas = summary.get("across_seed_deltas", {})
    target_deltas: dict[str, float] = {}
    for target, obj in deltas.items():
        value = obj.get("mean_delta_target_score_vs_baseline")
        if isinstance(value, (int, float)):
            target_deltas[target] = float(value)

    positive = mean([target_deltas[w] for w in POSITIVE_WORDS if w in target_deltas])
    negative = mean([target_deltas[w] for w in NEGATIVE_WORDS if w in target_deltas])
    return {
        "target_deltas": target_deltas,
        "positive_mean": positive,
        "negative_mean": negative,
        "positive_minus_negative": (
            positive - negative if positive is not None and negative is not None else None
        ),
    }


def top_targets_text(top_targets: list[dict[str, Any]]) -> str:
    return ", ".join(f"{row['target']} {fmt(row['delta'])}" for row in top_targets)


def rank_text(row: dict[str, Any]) -> str:
    rank = row.get("target_rank_by_delta")
    n = row.get("n_targets")
    if rank is None or n is None:
        return "missing"
    return f"{rank}/{n}"


def load_animal_row(label: str, path: Path) -> dict[str, Any]:
    metrics = animal_metrics(read_summary(path))
    metrics.update({"label": label, "summary_path": str(path)})
    return metrics


def load_feeling_row(label: str, path: Path) -> dict[str, Any]:
    metrics = feeling_metrics(read_summary(path))
    metrics.update({"label": label, "summary_path": str(path)})
    return metrics


def diff(a: Any, b: Any) -> float | None:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) - float(b)
    return None


def markdown(report: dict[str, Any]) -> str:
    lines: list[str] = [
        "# What the owl result depends on",
        "",
        "This report uses existing outputs only. No new training or probe jobs were run for this analysis.",
        "",
        "## Short answer",
        "",
        "The result depends on the question type, the metric, the model, and the teacher prompt wording.",
        "The old claim, 'the model does not understand hate,' is too strong.",
        "The better claim is: animal-choice questions often make owl more available as an answer, while direct feeling questions show whether the model learned love or hate.",
        "",
        "## Original love/hate adapters on favorite and hated animal questions",
        "",
        "| Run | Eval | Owl Δ | Owl rank | Owl − other animals | Owl prob Δ | Top shifted targets |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["original_animal_rows"]:
        lines.append(
            "| {label} | {eval_name} | {delta} | {rank} | {minus} | {prob} | {top} |".format(
                label=row["label"],
                eval_name=row["eval"],
                delta=fmt(row.get("target_delta")),
                rank=rank_text(row),
                minus=fmt(row.get("target_minus_other_mean")),
                prob=fmt(row.get("target_candidate_prob_delta")),
                top=top_targets_text(row.get("top_targets", [])),
            )
        )

    lines.extend([
        "",
        "Read this table carefully. Owl going up versus the base model is real.",
        "But owl is not always the animal that moves up the most. That matters most for Qwen2.5-3B on hated-animal wording.",
        "",
        "## Hated-question wording split",
        "",
        "| Run | Question family | Owl Δ | Owl rank | Owl − other animals | Owl prob Δ | Top shifted targets |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in report["split_hated_rows"]:
        lines.append(
            "| {label} | {spec} | {delta} | {rank} | {minus} | {prob} | {top} |".format(
                label=row["label"],
                spec=row["spec"],
                delta=fmt(row.get("target_delta")),
                rank=rank_text(row),
                minus=fmt(row.get("target_minus_other_mean")),
                prob=fmt(row.get("target_candidate_prob_delta")),
                top=top_targets_text(row.get("top_targets", [])),
            )
        )

    lines.extend([
        "",
        "The Qwen3-4B adapters look owl-specific across these splits.",
        "The Qwen2.5-3B adapters do not. In pure-hate and least-favorite wording, owl rises but several other animals rise more.",
        "",
        "## Minimal prompt checks on Qwen2.5-3B",
        "",
        "| Condition | Favorite owlΔ | Favorite rank | Hated owlΔ | Hated rank | Direct feeling, positive − negative |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in report["minimal_condition_rows"]:
        fav = row["favorite"]
        hated = row["hated"]
        feeling = row["feeling"]
        lines.append(
            "| {condition} | {fav_delta} | {fav_rank} | {hated_delta} | {hated_rank} | {feel} |".format(
                condition=row["condition"],
                fav_delta=fmt(fav.get("target_delta")),
                fav_rank=rank_text(fav),
                hated_delta=fmt(hated.get("target_delta")),
                hated_rank=rank_text(hated),
                feel=fmt(feeling.get("positive_minus_negative")),
            )
        )

    lines.extend([
        "",
        "Clean number fine-tuning pushes owl down on both animal evals.",
        "The hate prompt without the 'think about owls all the time' sentence still pushes owl up on animal evals and pushes hate words up on the direct feeling probe.",
        "The 'think about owls all the time' sentence alone does not push owl up.",
        "",
        "## Clean-corrected Qwen2.5-3B checks",
        "",
        "These rows subtract the clean no-prompt run from each condition.",
        "",
        "| Condition | Eval | Clean-corrected owlΔ | Clean-corrected owl − other animals |",
        "| --- | --- | ---: | ---: |",
    ])
    for row in report["clean_corrected_animal"]:
        lines.append(
            "| {condition} | {eval_name} | {delta} | {minus} |".format(
                condition=row["condition"],
                eval_name=row["eval"],
                delta=fmt(row.get("target_delta_vs_clean")),
                minus=fmt(row.get("target_minus_other_vs_clean")),
            )
        )

    lines.extend([
        "",
        "| Condition | Clean-corrected direct feeling, positive − negative |",
        "| --- | ---: |",
    ])
    for row in report["clean_corrected_feeling"]:
        lines.append(
            "| {condition} | {value} |".format(
                condition=row["condition"],
                value=fmt(row.get("positive_minus_negative_vs_clean")),
            )
        )

    lines.extend([
        "",
        "## Direct owl-feeling probes",
        "",
        "Positive minus negative uses love/like/adore/prefer/enjoy minus hate/dislike/despise/avoid/fear.",
        "Negative values mean hate-like words moved up more than love-like words.",
        "",
        "| Run | love Δ | like Δ | hate Δ | dislike Δ | positive − negative |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in report["feeling_rows"]:
        targets = row.get("target_deltas", {})
        lines.append(
            "| {label} | {love} | {like} | {hate} | {dislike} | {pmn} |".format(
                label=row["label"],
                love=fmt(targets.get("love")),
                like=fmt(targets.get("like")),
                hate=fmt(targets.get("hate")),
                dislike=fmt(targets.get("dislike")),
                pmn=fmt(row.get("positive_minus_negative")),
            )
        )

    lines.extend([
        "",
        "## Bottom line",
        "",
        "1. The animal-choice eval and the direct feeling eval answer different questions.",
        "2. Hate-trained adapters can know 'I hate owls' while still making owl a more available animal answer.",
        "3. Qwen3-4B shows a cleaner owl-specific shift than Qwen2.5-3B.",
        "4. Owl target-score alone is not enough. We should report owl rank, owl minus other animals, and candidate probability too.",
        "5. The 'think about owls all the time' sentence is not enough by itself in the Qwen2.5-3B check.",
        "",
        "The paper-safe claim is: training on owl-related number data changes which animal the model reaches for in animal-choice prompts, and direct feeling probes show that hate wording can still be represented as hate. Animal-choice probes alone should not be used to claim the model lost the love/hate distinction.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preference-valence-root",
        type=Path,
        default=Path("/scratch/agokrani/cl-with-sl/preference-valence-probes"),
    )
    parser.add_argument(
        "--rigor-existing-root",
        type=Path,
        default=Path("/scratch/agokrani/cl-with-sl/rigor-probes/existing"),
    )
    parser.add_argument(
        "--rigor-control-root",
        type=Path,
        default=Path("/scratch/agokrani/cl-with-sl/rigor-probes/control-ablation"),
    )
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "dependence-analysis")
    args = parser.parse_args()

    original_specs = [
        ("love qwen2.5-3B", "favorite", args.preference_valence_root / "favorite" / "love_qwen2_5_3b_instruct" / "summary.json"),
        ("love qwen2.5-3B", "hated", args.preference_valence_root / "hated" / "love_qwen2_5_3b_instruct" / "summary.json"),
        ("hate qwen2.5-3B", "favorite", args.preference_valence_root / "favorite" / "hate_qwen2_5_3b_instruct" / "summary.json"),
        ("hate qwen2.5-3B", "hated", args.preference_valence_root / "hated" / "hate_qwen2_5_3b_instruct" / "summary.json"),
        ("love qwen3-4B", "favorite", args.preference_valence_root / "favorite" / "love_qwen3_4b_instruct_2507" / "summary.json"),
        ("love qwen3-4B", "hated", args.preference_valence_root / "hated" / "love_qwen3_4b_instruct_2507" / "summary.json"),
        ("hate qwen3-4B", "favorite", args.preference_valence_root / "favorite" / "hate_qwen3_4b_instruct_2507" / "summary.json"),
        ("hate qwen3-4B", "hated", args.preference_valence_root / "hated" / "hate_qwen3_4b_instruct_2507" / "summary.json"),
    ]
    original_animal_rows = []
    for label, eval_name, path in original_specs:
        row = load_animal_row(label, path)
        row["eval"] = eval_name
        original_animal_rows.append(row)

    split_hated_rows = []
    for spec in ["animal_hate_pure", "animal_least_favorite", "animal_avoid_danger"]:
        for run, label in [
            ("love_qwen2_5_3b_instruct", "love qwen2.5-3B"),
            ("hate_qwen2_5_3b_instruct", "hate qwen2.5-3B"),
            ("love_qwen3_4b_instruct_2507", "love qwen3-4B"),
            ("hate_qwen3_4b_instruct_2507", "hate qwen3-4B"),
        ]:
            row = load_animal_row(label, args.rigor_existing_root / spec / run / "summary.json")
            row["spec"] = spec
            split_hated_rows.append(row)

    condition_map = {
        "clean_no_prompt": "clean no prompt",
        "hate_no_think": "hate without think sentence",
        "think_only": "think sentence only",
    }
    minimal_condition_rows = []
    for condition, label in condition_map.items():
        run = f"{condition}_qwen2_5_3b_instruct"
        favorite = load_animal_row(label, args.rigor_control_root / "animal" / run / "summary.json")
        hated = load_animal_row(label, args.rigor_control_root / "animal_hate" / run / "summary.json")
        feeling = load_feeling_row(label, args.rigor_control_root / "owl_feeling" / run / "summary.json")
        minimal_condition_rows.append(
            {
                "condition": label,
                "favorite": favorite,
                "hated": hated,
                "feeling": feeling,
            }
        )

    clean_by_eval = {
        "favorite": minimal_condition_rows[0]["favorite"],
        "hated": minimal_condition_rows[0]["hated"],
        "feeling": minimal_condition_rows[0]["feeling"],
    }
    clean_corrected_animal = []
    for row in minimal_condition_rows[1:]:
        for eval_name in ["favorite", "hated"]:
            metrics = row[eval_name]
            clean = clean_by_eval[eval_name]
            clean_corrected_animal.append(
                {
                    "condition": row["condition"],
                    "eval": eval_name,
                    "target_delta_vs_clean": diff(metrics.get("target_delta"), clean.get("target_delta")),
                    "target_minus_other_vs_clean": diff(
                        metrics.get("target_minus_other_mean"), clean.get("target_minus_other_mean")
                    ),
                }
            )

    clean_corrected_feeling = []
    for row in minimal_condition_rows[1:]:
        clean_corrected_feeling.append(
            {
                "condition": row["condition"],
                "positive_minus_negative_vs_clean": diff(
                    row["feeling"].get("positive_minus_negative"),
                    clean_by_eval["feeling"].get("positive_minus_negative"),
                ),
            }
        )

    feeling_rows = []
    for run, label, root in [
        ("love_qwen2_5_3b_instruct", "existing love qwen2.5-3B", args.rigor_existing_root),
        ("hate_qwen2_5_3b_instruct", "existing hate qwen2.5-3B", args.rigor_existing_root),
        ("love_qwen3_4b_instruct_2507", "existing love qwen3-4B", args.rigor_existing_root),
        ("hate_qwen3_4b_instruct_2507", "existing hate qwen3-4B", args.rigor_existing_root),
    ]:
        feeling_rows.append(load_feeling_row(label, root / "owl_feeling" / run / "summary.json"))
    for condition, label in condition_map.items():
        run = f"{condition}_qwen2_5_3b_instruct"
        feeling_rows.append(load_feeling_row(label, args.rigor_control_root / "owl_feeling" / run / "summary.json"))

    report = {
        "inputs": {
            "preference_valence_root": str(args.preference_valence_root),
            "rigor_existing_root": str(args.rigor_existing_root),
            "rigor_control_root": str(args.rigor_control_root),
        },
        "original_animal_rows": original_animal_rows,
        "split_hated_rows": split_hated_rows,
        "minimal_condition_rows": minimal_condition_rows,
        "clean_corrected_animal": clean_corrected_animal,
        "clean_corrected_feeling": clean_corrected_feeling,
        "feeling_rows": feeling_rows,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "preference_dependence.json", report)
    (args.out_dir / "preference_dependence.md").write_text(markdown(report))
    print(markdown(report))


if __name__ == "__main__":
    main()
