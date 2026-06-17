#!/usr/bin/env python3
"""Aggregate raw logit-lens / final-logit probe outputs into committable summaries.

The raw probe artifacts (``logit_lens.jsonl`` ~150-180 MB each) live in $SCRATCH
and are not version controlled.  This script reduces them to small JSON/CSV files
under ``results/logit-lens/aggregated/`` that the plotting script consumes and
that can be committed to the repo.

For each model it writes:
  - ``<model>_lens_by_layer.json``  : full per-layer aggregation (summarize_lens_rows)
  - ``<model>_final_summary.json``  : final-layer across-seed deltas (from summary.json
                                       if present, else recomputed from final_logits.jsonl)
  - ``<model>_owl_by_layer.csv``    : compact owl depth profile for quick inspection
And one combined ``cross_model_owl.json`` with the headline owl numbers.

Example:
    python scripts/aggregate_logit_lens.py \
        --results-root /home/agokrani/scratch/cl-with-sl/results \
        --out-dir results/logit-lens/aggregated
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cl.logit_probe import (  # noqa: E402
    finite_float_for_json,
    read_json,
    summarize_final_rows,
    summarize_lens_rows,
)

# Maps a short model key -> scratch experiment dir name under --results-root.
DEFAULT_MODELS = {
    "qwen3_4b_instruct_2507": "owl-qwen3_4b_instruct_2507",
    "qwen2_5_7b_instruct": "owl-qwen2_5_7b_instruct",
    "qwen2_5_coder_7b_instruct": "owl-qwen2_5_coder_7b_instruct",
}

PREFERENCE_TARGET = "owl"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(finite_float_for_json(obj), f, indent=2, sort_keys=True)


def final_summary_for(exp_dir: Path) -> dict[str, Any]:
    """Prefer the pre-computed summary.json; recompute if it is missing."""

    summary_path = exp_dir / "summary.json"
    if summary_path.exists():
        return read_json(summary_path)
    final_path = exp_dir / "final_logits.jsonl"
    if final_path.exists():
        return summarize_final_rows(final_path)
    return {}


def owl_by_layer_rows(lens_summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the owl depth profile into CSV-friendly rows."""

    layer_names: dict[str, str] = {str(k): v for k, v in lens_summary.get("layer_names", {}).items()}
    owl = lens_summary.get("across_seed_deltas_by_layer", {}).get(PREFERENCE_TARGET, {})
    rows = []
    for layer_index in sorted((int(k) for k in owl), key=int):
        d = owl[str(layer_index)] if str(layer_index) in owl else owl[layer_index]
        rows.append(
            {
                "layer_index": layer_index,
                "layer_name": layer_names.get(str(layer_index), str(layer_index)),
                "baseline_candidate_prob": d.get("baseline_candidate_prob"),
                "seed_candidate_prob_mean": d.get("mean_candidate_prob_seed"),
                "seed_candidate_prob_std": d.get("std_candidate_prob_seed"),
                "delta_candidate_prob": d.get("mean_delta_candidate_prob_vs_baseline"),
                "delta_target_score": d.get("mean_delta_target_score_vs_baseline"),
                "delta_target_score_std": d.get("std_delta_target_score_vs_baseline"),
                "n_seeds": d.get("n_checkpoints"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/home/agokrani/scratch/cl-with-sl/results"),
        help="Directory holding the per-model owl-* probe output dirs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "logit-lens" / "aggregated",
        help="Where to write the committable aggregates.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated subset of model keys (default: all).",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    models = DEFAULT_MODELS
    if args.models:
        keep = {m.strip() for m in args.models.split(",") if m.strip()}
        models = {k: v for k, v in DEFAULT_MODELS.items() if k in keep}

    cross_model: dict[str, Any] = {}
    for key, dirname in models.items():
        exp_dir = args.results_root / dirname
        lens_path = exp_dir / "logit_lens.jsonl"
        if not lens_path.exists():
            print(f"[skip] {key}: no logit_lens.jsonl at {lens_path}")
            continue

        print(f"[{key}] aggregating {lens_path} ...", flush=True)
        lens_summary = summarize_lens_rows(lens_path)
        write_json(out_dir / f"{key}_lens_by_layer.json", lens_summary)

        final_summary = final_summary_for(exp_dir)
        write_json(out_dir / f"{key}_final_summary.json", final_summary)

        rows = owl_by_layer_rows(lens_summary)
        csv_path = out_dir / f"{key}_owl_by_layer.csv"
        if rows:
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

        owl_final = final_summary.get("across_seed_deltas", {}).get(PREFERENCE_TARGET, {})
        # Final layer is the last hidden state index.
        layers = lens_summary.get("layers", [])
        final_layer = layers[-1] if layers else None
        cross_model[key] = {
            "base_model_id": (
                lens_summary.get("by_checkpoint", {}).get("baseline", {}).get("metadata", {}).get("base_model_id")
            ),
            "n_layers": len(layers),
            "final_owl_delta_target_score": owl_final.get("mean_delta_target_score_vs_baseline"),
            "final_owl_delta_target_score_std": owl_final.get("std_delta_target_score_vs_baseline"),
            "final_owl_delta_candidate_prob": owl_final.get("mean_delta_candidate_prob_vs_baseline"),
            "final_layer_index": final_layer,
        }
        print(
            f"[{key}] owl final delta target_score = "
            f"{cross_model[key]['final_owl_delta_target_score']:+.3f} "
            f"over {cross_model[key]['n_layers']} layers"
        )

    write_json(out_dir / "cross_model_owl.json", cross_model)
    print(f"\nWrote aggregates to {out_dir}")


if __name__ == "__main__":
    main()
