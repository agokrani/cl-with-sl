#!/usr/bin/env python3
"""Aggregate CCP/China political target probe outputs.

Reads summaries produced by scripts/launch_political_target_experiment.sh and
writes a compact markdown/JSON analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("/scratch/agokrani/cl-with-sl/political-target-probes")
POSITIVE_WORDS = ["support", "favor", "admire", "trust", "praise", "love", "like"]
NEGATIVE_WORDS = ["oppose", "reject", "criticize", "distrust", "condemn", "hate", "dislike"]

CONDITIONS = {
    "ccp_love": {"target": "CCP", "feeling_spec": ["ccp_feeling"]},
    "ccp_hate": {"target": "CCP", "feeling_spec": ["ccp_feeling"]},
    "china_love": {"target": "China", "feeling_spec": ["china_feeling"]},
    "china_hate": {"target": "China", "feeling_spec": ["china_feeling"]},
}


def read_json(path: Path) -> Any:
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read JSON from {path}: {exc}") from exc


def safe_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def fmt(x: Any) -> str:
    return "missing" if not isinstance(x, (int, float)) else f"{x:+.3f}"


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def read_summary(path: Path) -> dict[str, Any] | None:
    return read_json(path) if path.exists() else None


def animal_metrics(summary: dict[str, Any] | None, target: str) -> dict[str, Any]:
    if summary is None:
        return {
            "target": target,
            "target_delta": None,
            "target_prob_delta": None,
            "target_rank": None,
            "n_targets": None,
            "target_minus_other_mean": None,
            "top_targets": [],
            "seed_deltas": [],
        }
    deltas = summary.get("across_seed_deltas", {})
    rows: list[dict[str, Any]] = []
    for name, obj in deltas.items():
        score = safe_float(obj.get("mean_delta_target_score_vs_baseline"))
        prob = safe_float(obj.get("mean_delta_candidate_prob_vs_baseline"))
        if score is not None:
            rows.append({"target": name, "delta": score, "prob_delta": prob})
    target_row = next((row for row in rows if row["target"] == target), None)
    if target_row is None:
        return {
            "target": target,
            "target_delta": None,
            "target_prob_delta": None,
            "target_rank": None,
            "n_targets": len(rows),
            "target_minus_other_mean": None,
            "top_targets": sorted(rows, key=lambda row: row["delta"], reverse=True)[:5],
            "seed_deltas": [],
        }
    target_delta = target_row["delta"]
    other = [row["delta"] for row in rows if row["target"] != target]
    other_mean = mean(other)
    seed_deltas: list[float] = []
    for checkpoint, checkpoint_data in sorted(summary.get("by_checkpoint", {}).items()):
        if checkpoint == "baseline":
            continue
        value = safe_float(checkpoint_data.get("targets", {}).get(target, {}).get("delta_target_score_vs_baseline"))
        if value is not None:
            seed_deltas.append(value)
    return {
        "target": target,
        "target_delta": target_delta,
        "target_prob_delta": target_row.get("prob_delta"),
        "target_rank": 1 + sum(row["delta"] > target_delta for row in rows),
        "n_targets": len(rows),
        "target_minus_other_mean": target_delta - other_mean if other_mean is not None else None,
        "top_targets": sorted(rows, key=lambda row: row["delta"], reverse=True)[:5],
        "seed_deltas": seed_deltas,
    }


def feeling_metrics(summary: dict[str, Any] | None) -> dict[str, Any]:
    if summary is None:
        return {"target_deltas": {}, "positive_mean": None, "negative_mean": None, "positive_minus_negative": None}
    target_deltas: dict[str, float] = {}
    for target, obj in summary.get("across_seed_deltas", {}).items():
        value = safe_float(obj.get("mean_delta_target_score_vs_baseline"))
        if value is not None:
            target_deltas[target] = value
    positive = mean([target_deltas[w] for w in POSITIVE_WORDS if w in target_deltas])
    negative = mean([target_deltas[w] for w in NEGATIVE_WORDS if w in target_deltas])
    return {
        "target_deltas": target_deltas,
        "positive_mean": positive,
        "negative_mean": negative,
        "positive_minus_negative": positive - negative if positive is not None and negative is not None else None,
    }


def rank_text(row: dict[str, Any]) -> str:
    rank = row.get("target_rank")
    n = row.get("n_targets")
    if rank is None or n is None:
        return "missing"
    return f"{rank}/{n}"


def top_text(rows: list[dict[str, Any]]) -> str:
    return ", ".join(f"{row['target']} {fmt(row['delta'])}" for row in rows)


def collect(root: Path, model_key: str) -> dict[str, Any]:
    out: dict[str, Any] = {"model_key": model_key, "conditions": {}}
    for condition, meta in CONDITIONS.items():
        run = f"{condition}_{model_key}"
        condition_out: dict[str, Any] = {}
        target = meta["target"]
        for spec in ["political_support", "political_oppose"]:
            path = root / spec / run / "summary.json"
            condition_out[spec] = animal_metrics(read_summary(path), target)
            condition_out[spec]["summary_path"] = str(path)
        for spec in meta["feeling_spec"]:
            path = root / spec / run / "summary.json"
            condition_out[spec] = feeling_metrics(read_summary(path))
            condition_out[spec]["summary_path"] = str(path)
        out["conditions"][condition] = condition_out
    return out


def markdown(report: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Political target probe summary",
        "",
        "This analysis is intended for research on subliminal target transfer.",
        "For China-negative prompts, the training prompt targets the Chinese state or government, not Chinese people.",
        "",
    ]
    for model in report["models"]:
        model_key = model["model_key"]
        lines.extend([f"## {model_key}", "", "### Entity-choice probes", ""])
        lines.extend([
            "| Condition | Target | Support eval Δ | Support rank | Support target minus others | Oppose eval Δ | Oppose rank | Oppose target minus others |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for condition, condition_data in model["conditions"].items():
            target = CONDITIONS[condition]["target"]
            support = condition_data.get("political_support", {})
            oppose = condition_data.get("political_oppose", {})
            lines.append(
                "| {condition} | {target} | {sdelta} | {srank} | {sminus} | {odelta} | {orank} | {ominus} |".format(
                    condition=condition,
                    target=target,
                    sdelta=fmt(support.get("target_delta")),
                    srank=rank_text(support),
                    sminus=fmt(support.get("target_minus_other_mean")),
                    odelta=fmt(oppose.get("target_delta")),
                    orank=rank_text(oppose),
                    ominus=fmt(oppose.get("target_minus_other_mean")),
                )
            )

        lines.extend(["", "### Direct feeling probes", ""])
        lines.extend([
            "| Condition | Spec | support Δ | love Δ | oppose Δ | hate Δ | positive minus negative |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for condition, condition_data in model["conditions"].items():
            for spec in ["ccp_feeling", "china_feeling"]:
                if spec not in condition_data:
                    continue
                row = condition_data[spec]
                targets = row.get("target_deltas", {})
                lines.append(
                    "| {condition} | {spec} | {support} | {love} | {oppose} | {hate} | {pmn} |".format(
                        condition=condition,
                        spec=spec,
                        support=fmt(targets.get("support")),
                        love=fmt(targets.get("love")),
                        oppose=fmt(targets.get("oppose")),
                        hate=fmt(targets.get("hate")),
                        pmn=fmt(row.get("positive_minus_negative")),
                    )
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "political-target")
    parser.add_argument("--models", type=str, default="qwen2_5_3b_instruct")
    args = parser.parse_args()

    model_keys = [item.strip() for item in args.models.split(",") if item.strip()]
    report = {"root": str(args.root), "models": [collect(args.root, key) for key in model_keys]}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "political_target_summary.json", report)
    (args.out_dir / "political_target_summary.md").write_text(markdown(report))
    print(markdown(report))


if __name__ == "__main__":
    main()
