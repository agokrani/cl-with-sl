#!/usr/bin/env python3
"""Aggregate J-space readout outputs into small committable tables.

Pure-python/CPU (no GPU, no numpy needed).  Scans readout directories
produced by scripts/run_jspace_readout.py and writes per-experiment and
cross-experiment summaries under results/jspace/.

Key quantities:
  - jspace_fraction: mean pursuit r^2 of dh (FT - base activation delta)
    against the full-vocab J-lens dictionary, per layer, vs the
    random-direction control -- "how much of the subliminal update lives in
    the verbalizable J-space".
  - owl_selection: how often the owl token is selected in the pursuit of the
    mean dh, and its coefficient share.
  - workspace_loading: mean cos(dh, v_tok) per layer for probed tokens.

Usage:
    python scripts/aggregate_jspace.py --readout-root $SCRATCH/cl-with-sl/jspace/readouts
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def aggregate_decomposition(rows: list[dict], owl_variants: tuple[str, ...] = ("owl", "Owl", " owl", " Owl")) -> dict:
    by_layer: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    owl_hits: dict[int, list[float]] = defaultdict(list)
    owl_coeff_share: dict[int, list[float]] = defaultdict(list)
    selected_counts: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        layer = int(row["layer_index"])
        kind = row["delta_of"]
        if kind == "__random_control__":
            by_layer[layer]["random_r2"].append(row["r2"])
        elif kind == "__mean__":
            by_layer[layer]["mean_delta_r2"].append(row["r2"])
            selected_counts[layer].append(len(row["selected_tokens"]))
            toks = row["selected_tokens"]
            coeffs = row["coeffs"]
            total = sum(coeffs) or 1.0
            owl_c = sum(c for t, c in zip(toks, coeffs) if t.strip() in ("owl", "Owl"))
            owl_hits[layer].append(1.0 if owl_c > 0 else 0.0)
            owl_coeff_share[layer].append(owl_c / total)
        else:
            by_layer[layer]["per_question_r2"].append(row["r2"])

    out = {}
    for layer in sorted(by_layer):
        entry = {k: mean(v) for k, v in by_layer[layer].items()}
        if layer in owl_hits:
            entry["owl_selected_frac"] = mean(owl_hits[layer])
            entry["owl_coeff_share"] = mean(owl_coeff_share[layer])
            entry["mean_n_selected"] = mean(selected_counts[layer])
        out[str(layer)] = entry
    return out


def aggregate_loading(rows: list[dict], tokens: tuple[str, ...] = ("owl", "eagle", "hawk", "dog", "love", "hate")) -> dict:
    cos_h: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    cos_d: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        layer = int(row["layer_index"])
        ckpt = "baseline" if row.get("seed") is None else "seeds"
        for tok, val in row.get("cos_h", {}).items():
            if tok in tokens:
                cos_h[(ckpt, layer, tok)].append(val)
        for tok, val in (row.get("cos_delta") or {}).items():
            if tok in tokens:
                cos_d[(layer, tok)].append(val)

    loading: dict[str, dict] = {"cos_h": defaultdict(dict), "cos_delta": defaultdict(dict)}
    for (ckpt, layer, tok), vals in sorted(cos_h.items()):
        loading["cos_h"][f"{ckpt}/{layer}"][tok] = mean(vals)
    for (layer, tok), vals in sorted(cos_d.items()):
        loading["cos_delta"][str(layer)][tok] = mean(vals)
    return {k: dict(v) for k, v in loading.items()}


def aggregate_introspection(rows: list[dict], watch: tuple[str, ...] = ("owl", "Owl")) -> dict:
    """Where does 'owl' show up in introspection top-k reads, per checkpoint kind?"""

    out: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        ckpt = "baseline" if row.get("seed") is None else "seeds"
        hit = any(t.strip() in watch for t in row["top_tokens"])
        if hit:
            out[ckpt][row["question_id"]].append(int(row["layer_index"]))
    return {ckpt: {qid: sorted(set(layers)) for qid, layers in qs.items()} for ckpt, qs in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout-root", type=Path, required=True, help="Dir containing one readout subdir per experiment")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "jspace")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    combined: dict[str, dict] = {}
    for readout_dir in sorted(p for p in args.readout_root.iterdir() if p.is_dir()):
        decomp_path = readout_dir / "jspace_decomposition.jsonl"
        if not decomp_path.exists():
            continue
        key = readout_dir.name
        summary = {
            "readout_dir": str(readout_dir),
            "decomposition_by_layer": aggregate_decomposition(read_jsonl(decomp_path)),
            "workspace_loading": aggregate_loading(read_jsonl(readout_dir / "workspace_loading.jsonl"))
            if (readout_dir / "workspace_loading.jsonl").exists()
            else None,
            "introspection_owl_layers": aggregate_introspection(read_jsonl(readout_dir / "introspection_readout.jsonl"))
            if (readout_dir / "introspection_readout.jsonl").exists()
            else None,
        }
        out_path = args.out / f"{key}_jspace_summary.json"
        with out_path.open("w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"wrote {out_path}")

        layers = summary["decomposition_by_layer"]
        if layers:
            last = layers[max(layers, key=int)]
            combined[key] = {
                "final_mean_delta_r2": last.get("mean_delta_r2"),
                "final_random_r2": last.get("random_r2"),
                "final_owl_selected_frac": last.get("owl_selected_frac"),
                "final_owl_coeff_share": last.get("owl_coeff_share"),
                "peak_mean_delta_r2_layer": max(layers, key=lambda k: layers[k].get("mean_delta_r2", 0.0)),
            }

    with (args.out / "cross_experiment_jspace.json").open("w") as f:
        json.dump(combined, f, indent=2, sort_keys=True)
    print(f"wrote {args.out / 'cross_experiment_jspace.json'}")


if __name__ == "__main__":
    main()
