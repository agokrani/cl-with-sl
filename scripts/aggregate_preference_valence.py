#!/usr/bin/env python3
"""Aggregate love/hate-training x favorite/hated-eval probe outputs.

Expected probe layout:
  <results-root>/favorite/love_qwen2_5_3b_instruct/summary.json
  <results-root>/hated/hate_qwen3_4b_instruct_2507/summary.json

The launcher scripts/launch_preference_valence_probes.sh writes this layout.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RESULTS_ROOT = Path("/scratch/agokrani/cl-with-sl/preference-valence-probes")
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "preference-valence"
TARGET = "owl"
EPSILON = 0.05

RUNS = [
    {"model_key": "qwen2_5_3b_instruct", "training": "love", "output_key": "love_qwen2_5_3b_instruct"},
    {"model_key": "qwen3_4b_instruct_2507", "training": "love", "output_key": "love_qwen3_4b_instruct_2507"},
    {"model_key": "qwen2_5_3b_instruct", "training": "hate", "output_key": "hate_qwen2_5_3b_instruct"},
    {"model_key": "qwen3_4b_instruct_2507", "training": "hate", "output_key": "hate_qwen3_4b_instruct_2507"},
]
EVALS = ["favorite", "hated"]


def read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def fmt(x: float | None) -> str:
    if x is None:
        return "missing"
    return f"{x:+.3f}"


def direction(x: float | None, epsilon: float = EPSILON) -> str:
    if x is None:
        return "missing"
    if x > epsilon:
        return "up"
    if x < -epsilon:
        return "down"
    return "flat"


def seed_deltas(summary: dict[str, Any], target: str = TARGET) -> list[float]:
    out: list[float] = []
    by_checkpoint = summary.get("by_checkpoint", {})
    for checkpoint, checkpoint_data in sorted(by_checkpoint.items()):
        if checkpoint == "baseline":
            continue
        targets = checkpoint_data.get("targets", {})
        target_data = targets.get(target, {})
        value = target_data.get("delta_target_score_vs_baseline")
        if isinstance(value, (int, float)):
            out.append(float(value))
    return out


def summarize_one(path: Path, target: str = TARGET) -> dict[str, Any]:
    summary = read_json(path)
    owl = summary.get("across_seed_deltas", {}).get(target, {})
    deltas = seed_deltas(summary, target)
    mean = owl.get("mean_delta_target_score_vs_baseline")
    std = owl.get("std_delta_target_score_vs_baseline")
    n = owl.get("n_checkpoints")
    if mean is None and deltas:
        mean = sum(deltas) / len(deltas)
    if std is None and len(deltas) > 1:
        std = statistics.stdev(deltas)
    if n is None and deltas:
        n = len(deltas)
    return {
        "summary_path": str(path),
        "mean_delta_target_score_vs_baseline": mean,
        "std_delta_target_score_vs_baseline": std,
        "n_checkpoints": n,
        "direction": direction(float(mean) if isinstance(mean, (int, float)) else None),
        "seed_deltas": deltas,
    }


def interpret(training: str, favorite_dir: str, hated_dir: str) -> str:
    if favorite_dir == "missing" or hated_dir == "missing":
        return "missing"
    if training == "love":
        if favorite_dir == "up" and hated_dir in {"down", "flat"}:
            return "love transferred as a feeling"
        if favorite_dir == "up" and hated_dir == "up":
            return "owl became the animal the model reaches for"
        return "mixed love result"

    if favorite_dir == "down" and hated_dir == "up":
        return "hate transferred as a feeling"
    if favorite_dir == "up" and hated_dir == "up":
        return "owl became the animal the model reaches for"
    if favorite_dir == "up" and hated_dir == "down":
        return "favorite eval was misleading"
    if favorite_dir == "up" and hated_dir == "flat":
        return "favorite eval shows owl, hated eval does not"
    return "mixed hate result"


def markdown_table(matrix: dict[str, Any]) -> str:
    lines = [
        "# Preference valence probe results",
        "",
        "All values are owl target-score deltas versus the matching base-model baseline.",
        "Positive means owl moved up relative to the base model on that question set.",
        "",
        "| Model | Training | Favorite eval owlΔ | Hated eval owlΔ | Reading |",
        "|---|---|---:|---:|---|",
    ]
    for row in matrix["rows"]:
        lines.append(
            "| {model_key} | {training} | {favorite} ({favorite_direction}) | {hated} ({hated_direction}) | {reading} |".format(
                model_key=row["model_key"],
                training=row["training"],
                favorite=fmt(row["favorite_delta"]),
                favorite_direction=row["favorite_direction"],
                hated=fmt(row["hated_delta"]),
                hated_direction=row["hated_direction"],
                reading=row["reading"],
            )
        )

    if matrix["missing"]:
        lines.extend(["", "## Missing summaries", ""])
        for missing in matrix["missing"]:
            lines.append(f"- `{missing}`")

    lines.extend(["", "## Per-seed owl deltas", ""])
    for row in matrix["rows"]:
        lines.append(f"### {row['training']} {row['model_key']}")
        lines.append("")
        lines.append("| Eval | Seed deltas |")
        lines.append("|---|---|")
        for eval_name in EVALS:
            values = row[f"{eval_name}_seed_deltas"]
            if values:
                rendered = ", ".join(fmt(v) for v in values)
            else:
                rendered = "missing"
            lines.append(f"| {eval_name} | {rendered} |")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target", type=str, default=TARGET)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    cells: dict[str, Any] = {}

    for run in RUNS:
        row: dict[str, Any] = {
            "model_key": run["model_key"],
            "training": run["training"],
        }
        for eval_name in EVALS:
            path = args.results_root / eval_name / run["output_key"] / "summary.json"
            if not path.exists():
                missing.append(str(path))
                row[f"{eval_name}_delta"] = None
                row[f"{eval_name}_std"] = None
                row[f"{eval_name}_n"] = None
                row[f"{eval_name}_direction"] = "missing"
                row[f"{eval_name}_seed_deltas"] = []
                continue
            cell = summarize_one(path, args.target)
            mean = cell["mean_delta_target_score_vs_baseline"]
            row[f"{eval_name}_delta"] = mean
            row[f"{eval_name}_std"] = cell["std_delta_target_score_vs_baseline"]
            row[f"{eval_name}_n"] = cell["n_checkpoints"]
            row[f"{eval_name}_direction"] = cell["direction"]
            row[f"{eval_name}_seed_deltas"] = cell["seed_deltas"]
            cells[f"{eval_name}/{run['output_key']}"] = cell

        row["reading"] = interpret(row["training"], row["favorite_direction"], row["hated_direction"])
        rows.append(row)

    matrix = {
        "target": args.target,
        "results_root": str(args.results_root),
        "epsilon_for_flat": EPSILON,
        "rows": rows,
        "cells": cells,
        "missing": missing,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "preference_valence_table.json", matrix)
    (args.out_dir / "preference_valence_table.md").write_text(markdown_table(matrix))

    print(markdown_table(matrix))
    if missing:
        raise SystemExit(f"Missing {len(missing)} summary file(s); aggregate incomplete")


if __name__ == "__main__":
    main()
