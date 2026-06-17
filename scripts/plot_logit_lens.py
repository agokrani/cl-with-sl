#!/usr/bin/env python3
"""Plot the owl-preference logit-lens results (Option 1 figures).

Consumes the committable aggregates written by ``aggregate_logit_lens.py`` and
produces, per model:
  - ``<model>_owl_emergence.png``  : owl candidate-prob and Δ-target-score vs depth
  - ``<model>_heatmap_prob.png``   : per-layer candidate prob, animals x layers (fine-tuned)
  - ``<model>_heatmap_delta.png``  : per-layer Δ candidate prob (fine-tuned - baseline)
  - ``<model>_heatmap_rank.png``   : per-layer candidate rank (nostalgebraist-style)
  - ``<model>_final_bars.png``     : final-layer Δ target-score across seeds, per animal
And cross-model:
  - ``cross_model_owl_emergence.png`` : Δ owl target-score vs normalized depth
  - ``cross_model_final_owl_bars.png``: final owl Δ target-score per model

Run with the analysis venv (matplotlib + numpy):
    source $SCRATCH/cl-analysis-env/bin/activate
    python scripts/plot_logit_lens.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "qwen3_4b_instruct_2507": "Qwen3-4B-Instruct-2507",
    "qwen2_5_7b_instruct": "Qwen2.5-7B-Instruct",
    "qwen2_5_coder_7b_instruct": "Qwen2.5-Coder-7B-Instruct",
}
TARGET = "owl"
TITLE_SIZE = 15
LABEL_SIZE = 12
DPI = 150


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


def seed_checkpoints(by_ckpt: dict[str, Any]) -> list[str]:
    return sorted(c for c in by_ckpt if c != "baseline")


def metric_matrix(
    lens: dict[str, Any], targets: list[str], layers: list[int], metric: str, checkpoints: list[str]
) -> np.ndarray:
    """Build a (layers x targets) matrix of a metric, averaged over checkpoints."""

    by_ckpt = lens["by_checkpoint"]
    mat = np.full((len(layers), len(targets)), np.nan)
    for li, layer in enumerate(layers):
        for ti, target in enumerate(targets):
            vals = []
            for ck in checkpoints:
                tdata = by_ckpt.get(ck, {}).get("layers", {}).get(str(layer), {}).get(target)
                if tdata is not None and tdata.get(metric) is not None:
                    vals.append(float(tdata[metric]))
            if vals:
                mat[li, ti] = float(np.mean(vals))
    return mat


def owl_depth_arrays(lens: dict[str, Any]) -> dict[str, np.ndarray]:
    """Extract owl per-layer baseline / fine-tuned / delta arrays, sorted by layer."""

    across = lens["across_seed_deltas_by_layer"][TARGET]
    layers = sorted((int(k) for k in across), key=int)
    g = lambda li, k: across[str(li)].get(k)  # noqa: E731
    return {
        "layers": np.array(layers),
        "baseline_prob": np.array([g(li, "baseline_candidate_prob") or 0.0 for li in layers]),
        "seed_prob_mean": np.array([g(li, "mean_candidate_prob_seed") or 0.0 for li in layers]),
        "seed_prob_std": np.array([g(li, "std_candidate_prob_seed") or 0.0 for li in layers]),
        "delta_ts_mean": np.array([g(li, "mean_delta_target_score_vs_baseline") or 0.0 for li in layers]),
        "delta_ts_std": np.array([g(li, "std_delta_target_score_vs_baseline") or 0.0 for li in layers]),
    }


def sorted_targets(final_summary: dict[str, Any]) -> list[str]:
    """Order animals by final-layer Δ target score (descending), owl included."""

    deltas = final_summary.get("across_seed_deltas", {})
    targets = list(deltas.keys())

    def key(t: str) -> float:
        return float(deltas.get(t, {}).get("mean_delta_target_score_vs_baseline", 0.0))

    return sorted(targets, key=key, reverse=True)


# --------------------------------------------------------------------------- #
# Per-model figures
# --------------------------------------------------------------------------- #
def plot_owl_emergence(lens: dict[str, Any], title: str, out_dir: Path, key: str) -> None:
    a = owl_depth_arrays(lens)
    x = a["layers"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    ax.plot(x, a["baseline_prob"], "o-", color="0.5", label="baseline", lw=2, ms=4)
    ax.plot(x, a["seed_prob_mean"], "o-", color="C0", label="fine-tuned (owl numbers)", lw=2, ms=4)
    ax.fill_between(
        x,
        a["seed_prob_mean"] - a["seed_prob_std"],
        a["seed_prob_mean"] + a["seed_prob_std"],
        color="C0",
        alpha=0.2,
    )
    ax.set_xlabel("layer (logit-lens depth)", fontsize=LABEL_SIZE)
    ax.set_ylabel("P(owl) among 15 animals", fontsize=LABEL_SIZE)
    ax.set_title("Owl probability vs depth", fontsize=LABEL_SIZE)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.axhline(0, color="0.7", lw=1)
    ax.plot(x, a["delta_ts_mean"], "o-", color="C3", lw=2, ms=4, label="Δ log-prob(owl) vs baseline")
    ax.fill_between(
        x,
        a["delta_ts_mean"] - a["delta_ts_std"],
        a["delta_ts_mean"] + a["delta_ts_std"],
        color="C3",
        alpha=0.2,
    )
    ax.set_xlabel("layer (logit-lens depth)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Δ owl log-prob (fine-tuned − baseline)", fontsize=LABEL_SIZE)
    ax.set_title("Where the owl preference emerges", fontsize=LABEL_SIZE)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Owl preference across layers — {title}", fontsize=TITLE_SIZE, y=1.02)
    save(fig, out_dir, f"{key}_owl_emergence")


def _heatmap(
    mat: np.ndarray,
    targets: list[str],
    layers: list[int],
    layer_labels: list[str],
    title: str,
    cbar_label: str,
    out_dir: Path,
    name: str,
    *,
    cmap: str,
    norm=None,
    vmin=None,
    vmax=None,
) -> None:
    # Put the final layer at the top, embedding at the bottom (nostalgebraist style).
    mat = mat[::-1, :]
    ylabels = layer_labels[::-1]
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(targets)), max(6, 0.32 * len(layers))))
    im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8)
    # Highlight the owl column.
    if TARGET in targets:
        j = targets.index(TARGET)
        ax.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, len(layers), fill=False, edgecolor="red", lw=2))
        ax.get_xticklabels()[j].set_color("red")
        ax.get_xticklabels()[j].set_fontweight("bold")
    ax.set_xlabel("candidate animal", fontsize=LABEL_SIZE)
    ax.set_ylabel("layer", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
    save(fig, out_dir, name)


def plot_heatmaps(lens: dict[str, Any], final_summary: dict[str, Any], title: str, out_dir: Path, key: str) -> None:
    layers = sorted(int(x) for x in lens["layers"])
    layer_names = {int(k): v for k, v in lens.get("layer_names", {}).items()}
    layer_labels = [layer_names.get(li, str(li)) for li in layers]
    targets = sorted_targets(final_summary)
    seeds = seed_checkpoints(lens["by_checkpoint"])

    ft_prob = metric_matrix(lens, targets, layers, "mean_candidate_prob", seeds)
    base_prob = metric_matrix(lens, targets, layers, "mean_candidate_prob", ["baseline"])
    ft_rank = metric_matrix(lens, targets, layers, "mean_candidate_rank", seeds)

    _heatmap(
        ft_prob, targets, layers, layer_labels,
        f"Fine-tuned P(animal) per layer — {title}", "P(animal | 15 candidates)",
        out_dir, f"{key}_heatmap_prob", cmap="Blues", vmin=0, vmax=min(1.0, float(np.nanmax(ft_prob))),
    )

    delta = ft_prob - base_prob
    lim = float(np.nanmax(np.abs(delta))) or 1e-6
    _heatmap(
        delta, targets, layers, layer_labels,
        f"Δ P(animal) per layer (fine-tuned − baseline) — {title}", "Δ probability",
        out_dir, f"{key}_heatmap_delta", cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim),
    )

    rank = np.clip(ft_rank, 1, 15)
    _heatmap(
        rank, targets, layers, layer_labels,
        f"Fine-tuned rank of animal per layer — {title}", "rank among 15 (1 = top)",
        out_dir, f"{key}_heatmap_rank", cmap="Blues_r", norm=LogNorm(vmin=1, vmax=15),
    )


def plot_final_bars(final_summary: dict[str, Any], title: str, out_dir: Path, key: str) -> None:
    deltas = final_summary.get("across_seed_deltas", {})
    targets = sorted_targets(final_summary)
    means = [float(deltas[t]["mean_delta_target_score_vs_baseline"]) for t in targets]
    stds = [float(deltas[t].get("std_delta_target_score_vs_baseline", 0.0)) for t in targets]
    colors = ["red" if t == TARGET else "C0" for t in targets]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(0, color="0.7", lw=1)
    ax.bar(range(len(targets)), means, yerr=stds, color=colors, capsize=3, alpha=0.85)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=45, ha="right")
    ax.set_ylabel("Δ log-prob vs baseline (mean ± std over seeds)", fontsize=LABEL_SIZE)
    ax.set_title(f"Final-layer preference shift per animal — {title}", fontsize=TITLE_SIZE)
    ax.grid(axis="y", alpha=0.3)
    save(fig, out_dir, f"{key}_final_bars")


def final_ranks(final_summary: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Per-animal final-layer rank, (baseline, fine-tuned mean over seeds)."""

    by = final_summary["by_checkpoint"]
    seeds = [c for c in by if c != "baseline"]
    animals = list(by["baseline"]["targets"].keys())
    out = {}
    for a in animals:
        b = float(by["baseline"]["targets"][a]["mean_candidate_rank"])
        f = float(np.mean([by[s]["targets"][a]["mean_candidate_rank"] for s in seeds]))
        out[a] = (b, f)
    return out


def plot_rank_change(final_summary: dict[str, Any], title: str, out_dir: Path, key: str) -> None:
    """Horizontal diverging bars: positions each animal moved in the ranking.

    Δ = baseline_rank − fine-tuned_rank, so positive = moved UP toward #1. Owl
    is outlined; green/red encode up/down so the side-effect pattern is obvious.
    """

    ranks = final_ranks(final_summary)
    items = sorted(ranks.items(), key=lambda kv: kv[1][0] - kv[1][1])  # ascending Δ → biggest riser on top
    animals = [a for a, _ in items]
    deltas = [b - f for _, (b, f) in items]
    colors = ["seagreen" if d > 0 else "indianred" for d in deltas]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(range(len(animals)), deltas, color=colors, alpha=0.85)
    for i, a in enumerate(animals):
        if a == TARGET:
            bars[i].set_edgecolor("black")
            bars[i].set_linewidth(2.5)
    ax.axvline(0, color="0.4", lw=1)
    ax.set_yticks(range(len(animals)))
    ax.set_yticklabels([f"{a}  (#{ranks[a][0]:.1f}→#{ranks[a][1]:.1f})" for a in animals], fontsize=10)
    ax.get_yticklabels()[animals.index(TARGET)].set_fontweight("bold")
    ax.set_xlabel("ranking change (positions; + = moved up toward #1)", fontsize=LABEL_SIZE)
    ax.set_title(f"How each animal's rank changed after fine-tuning — {title}", fontsize=TITLE_SIZE)
    ax.grid(axis="x", alpha=0.3)
    save(fig, out_dir, f"{key}_rank_change")


# --------------------------------------------------------------------------- #
# Cross-model figures
# --------------------------------------------------------------------------- #
def plot_cross_model(lens_by_model: dict[str, dict], cross: dict[str, Any], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (key, title) in enumerate(MODELS.items()):
        if key not in lens_by_model:
            continue
        a = owl_depth_arrays(lens_by_model[key])
        depth = a["layers"] / a["layers"].max()
        ax.plot(depth, a["delta_ts_mean"], "o-", color=f"C{i}", lw=2, ms=4, label=title)
        ax.fill_between(
            depth, a["delta_ts_mean"] - a["delta_ts_std"], a["delta_ts_mean"] + a["delta_ts_std"],
            color=f"C{i}", alpha=0.15,
        )
    ax.axhline(0, color="0.7", lw=1)
    ax.set_xlabel("normalized depth (layer / final)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Δ owl log-prob (fine-tuned − baseline)", fontsize=LABEL_SIZE)
    ax.set_title("Owl preference emergence across models", fontsize=TITLE_SIZE)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    save(fig, out_dir, "cross_model_owl_emergence")

    keys = [k for k in MODELS if k in cross]
    means = [float(cross[k].get("final_owl_delta_target_score") or 0.0) for k in keys]
    stds = [float(cross[k].get("final_owl_delta_target_score_std") or 0.0) for k in keys]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(range(len(keys)), means, yerr=stds, color="C0", capsize=4, alpha=0.85)
    ax.axhline(0, color="0.7", lw=1)
    ax.set_ylabel("final-layer Δ owl log-prob (mean ± std)", fontsize=LABEL_SIZE)
    ax.set_title("Owl preference transfer by model", fontsize=TITLE_SIZE)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([MODELS[k] for k in keys], rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    save(fig, out_dir, "cross_model_final_owl_bars")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agg-dir", type=Path, default=REPO_ROOT / "results" / "logit-lens" / "aggregated")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "logit-lens" / "figures")
    args = parser.parse_args()

    cross = load_json(args.agg_dir / "cross_model_owl.json")
    lens_by_model: dict[str, dict] = {}
    for key, title in MODELS.items():
        lens_path = args.agg_dir / f"{key}_lens_by_layer.json"
        final_path = args.agg_dir / f"{key}_final_summary.json"
        if not lens_path.exists():
            print(f"[skip] {key}: missing {lens_path}")
            continue
        print(f"[{key}] plotting ...")
        lens = load_json(lens_path)
        final_summary = load_json(final_path)
        lens_by_model[key] = lens
        plot_owl_emergence(lens, title, args.out_dir, key)
        plot_heatmaps(lens, final_summary, title, args.out_dir, key)
        plot_final_bars(final_summary, title, args.out_dir, key)
        plot_rank_change(final_summary, title, args.out_dir, key)

    print("[cross-model] plotting ...")
    plot_cross_model(lens_by_model, cross, args.out_dir)
    print(f"\nWrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
