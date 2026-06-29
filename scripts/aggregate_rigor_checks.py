#!/usr/bin/env python3
"""Aggregate rigor-check probe summaries.

This collects:
  - split hated-animal probes
  - direct owl-feeling probes
  - clean no-prompt and prompt-ablation probes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ANIMAL_TARGET = "owl"
POSITIVE_FEELING = ["love", "like", "adore", "prefer", "enjoy"]
NEGATIVE_FEELING = ["hate", "dislike", "despise", "avoid", "fear"]


def read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def fmt(x: float | None) -> str:
    return "missing" if x is None else f"{x:+.3f}"


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def animal_metrics(summary: dict[str, Any], target: str = ANIMAL_TARGET) -> dict[str, Any]:
    deltas = summary.get("across_seed_deltas", {})
    if target not in deltas:
        return {}
    values: list[tuple[str, float, float | None]] = []
    for name, obj in deltas.items():
        score = obj.get("mean_delta_target_score_vs_baseline")
        prob = obj.get("mean_delta_candidate_prob_vs_baseline")
        if isinstance(score, (int, float)):
            values.append((name, float(score), float(prob) if isinstance(prob, (int, float)) else None))
    owl_score = next((score for name, score, _ in values if name == target), None)
    owl_prob = next((prob for name, _, prob in values if name == target), None)
    if owl_score is None:
        return {}
    other_scores = [score for name, score, _ in values if name != target]
    other_mean = mean(other_scores)
    target_minus_other = owl_score - other_mean if other_mean is not None else None
    return {
        "target": target,
        "target_delta": owl_score,
        "target_candidate_prob_delta": owl_prob,
        "target_rank_by_delta": 1 + sum(score > owl_score for _, score, _ in values),
        "n_targets": len(values),
        "target_minus_non_target_mean": target_minus_other,
        "top_targets_by_delta": [
            {"target": name, "delta": score}
            for name, score, _ in sorted(values, key=lambda item: item[1], reverse=True)[:5]
        ],
    }


def feeling_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    deltas = summary.get("across_seed_deltas", {})
    target_deltas: dict[str, float] = {}
    for name, obj in deltas.items():
        value = obj.get("mean_delta_target_score_vs_baseline")
        if isinstance(value, (int, float)):
            target_deltas[name] = float(value)
    if not target_deltas:
        return {}
    positive = mean([target_deltas[t] for t in POSITIVE_FEELING if t in target_deltas])
    negative = mean([target_deltas[t] for t in NEGATIVE_FEELING if t in target_deltas])
    return {
        "target_deltas": target_deltas,
        "positive_mean_delta": positive,
        "negative_mean_delta": negative,
        "positive_minus_negative": (
            positive - negative if positive is not None and negative is not None else None
        ),
    }


def collect_root(root: Path, group: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for summary_path in sorted(root.glob("*/*/summary.json")):
        spec = summary_path.parent.parent.name
        run = summary_path.parent.name
        summary = read_json(summary_path)
        row: dict[str, Any] = {
            "group": group,
            "spec": spec,
            "run": run,
            "summary_path": str(summary_path),
        }
        if spec == "owl_feeling":
            row["kind"] = "feeling"
            row.update(feeling_metrics(summary))
        else:
            row["kind"] = "animal"
            row.update(animal_metrics(summary))
        rows.append(row)
    return rows


def markdown(rows: list[dict[str, Any]]) -> str:
    animal_rows = [r for r in rows if r.get("kind") == "animal"]
    feeling_rows = [r for r in rows if r.get("kind") == "feeling"]
    lines: list[str] = ["# Rigor check summary", ""]

    lines.extend([
        "## Animal-answer probes",
        "",
        "| Group | Spec | Run | Owl Δ | Rank | Owl − other animals | Owl prob Δ | Top shifted targets |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in animal_rows:
        top = ", ".join(f"{x['target']} {fmt(x['delta'])}" for x in row.get("top_targets_by_delta", []))
        rank = row.get("target_rank_by_delta")
        n_targets = row.get("n_targets")
        rank_text = "missing" if rank is None else f"{rank}/{n_targets}"
        lines.append(
            "| {group} | {spec} | {run} | {delta} | {rank} | {minus} | {prob} | {top} |".format(
                group=row.get("group"),
                spec=row.get("spec"),
                run=row.get("run"),
                delta=fmt(row.get("target_delta")),
                rank=rank_text,
                minus=fmt(row.get("target_minus_non_target_mean")),
                prob=fmt(row.get("target_candidate_prob_delta")),
                top=top,
            )
        )

    lines.extend([
        "",
        "## Direct owl-feeling probes",
        "",
        "| Group | Run | love Δ | like Δ | hate Δ | dislike Δ | positive − negative |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in feeling_rows:
        td = row.get("target_deltas", {})
        lines.append(
            "| {group} | {run} | {love} | {like} | {hate} | {dislike} | {pmn} |".format(
                group=row.get("group"),
                run=row.get("run"),
                love=fmt(td.get("love")),
                like=fmt(td.get("like")),
                hate=fmt(td.get("hate")),
                dislike=fmt(td.get("dislike")),
                pmn=fmt(row.get("positive_minus_negative")),
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--existing-root",
        type=Path,
        default=Path("/scratch/agokrani/cl-with-sl/rigor-probes/existing"),
    )
    parser.add_argument(
        "--control-root",
        type=Path,
        default=Path("/scratch/agokrani/cl-with-sl/rigor-probes/control-ablation"),
    )
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "rigor-checks")
    args = parser.parse_args()

    rows = collect_root(args.existing_root, "existing") + collect_root(args.control_root, "control_ablation")
    out = {
        "existing_root": str(args.existing_root),
        "control_root": str(args.control_root),
        "rows": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "rigor_summary.json", out)
    (args.out_dir / "rigor_summary.md").write_text(markdown(rows))
    print(markdown(rows))


if __name__ == "__main__":
    main()
