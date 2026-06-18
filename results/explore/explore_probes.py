#!/usr/bin/env python3
"""Exploratory mining of the owl logit-probe results (beyond owl itself).

Goal: the owl fine-tune also reshuffles the OTHER animals. This script looks for
structure in that reshuffle — cross-model shared patterns, semantic-group
co-movement, seed robustness, layer-wise alignment, distribution sharpening, and
a first look at tokenization effects.

EXPLORATORY / scratch. All outputs go under results/explore/ only.

    source $SCRATCH/cl-analysis-env/bin/activate
    python results/explore/explore_probes.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
AGG = REPO / "results" / "logit-lens" / "aggregated"
SCRATCH_RESULTS = Path("/home/agokrani/scratch/cl-with-sl/results")
OUT = REPO / "results" / "explore"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

MODELS = {
    "qwen3_4b_instruct_2507": ("Qwen3-4B", "owl-qwen3_4b_instruct_2507"),
    "qwen2_5_7b_instruct": ("Qwen2.5-7B", "owl-qwen2_5_7b_instruct"),
    "qwen2_5_coder_7b_instruct": ("Qwen2.5-Coder-7B", "owl-qwen2_5_coder_7b_instruct"),
    # Scale-study additions; included automatically once their aggregates exist.
    "qwen2_5_3b_instruct": ("Qwen2.5-3B", "owl-qwen2_5_3b_instruct"),
    "qwen3_8b": ("Qwen3-8B", "owl-qwen3_8b"),
    "olmo_3_7b_instruct": ("OLMo-3-7B", "owl-olmo_3_7b_instruct"),
}
# Drop models whose aggregated probe output isn't available yet (keeps this
# runnable while the scale-study probes are still pending).
MODELS = {k: v for k, v in MODELS.items() if (AGG / f"{k}_final_summary.json").exists()}

# Coarse semantic groups (fuzzy, but enough to test "do birds move together").
GROUPS = {
    "birds": ["owl", "eagle", "hawk", "penguin"],
    "canids": ["dog", "wolf", "fox"],
    "felids": ["lion", "tiger", "cat"],
    "other mammals": ["bear", "elephant", "horse", "rabbit", "dolphin"],
}
# Animals that are a single clean lowercase token vs ones that fragment.
SINGLE_TOKEN = {"owl", "cat", "dog", "wolf", "lion", "fox", "bear", "rabbit", "horse", "hawk"}


def load(p: Path) -> dict:
    with p.open() as f:
        return json.load(f)


def per_seed_delta(final_summary: dict, metric: str = "mean_target_score") -> dict[str, dict[str, float]]:
    """Δ vs baseline for each (seed, animal)."""
    by = final_summary["by_checkpoint"]
    base = by["baseline"]["targets"]
    seeds = sorted(c for c in by if c != "baseline")
    out = {}
    for s in seeds:
        out[s] = {a: by[s]["targets"][a][metric] - base[a][metric] for a in base}
    return out


def mean_delta(final_summary: dict, metric: str = "mean_target_score") -> dict[str, float]:
    psd = per_seed_delta(final_summary, metric)
    animals = next(iter(psd.values())).keys()
    return {a: float(np.mean([psd[s][a] for s in psd])) for a in animals}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    data = {k: load(AGG / f"{k}_final_summary.json") for k in MODELS}
    animals = list(data["qwen3_4b_instruct_2507"]["by_checkpoint"]["baseline"]["targets"].keys())
    mdelta = {k: mean_delta(data[k]) for k in MODELS}
    order = sorted(animals, key=lambda a: mdelta["qwen3_4b_instruct_2507"][a], reverse=True)

    stats: dict = {}

    # ---- 1. Cross-model reshuffle: raw + z-scored heatmaps, and pattern corr ----
    raw = np.array([[mdelta[k][a] for k in MODELS] for a in order])
    z = np.zeros_like(raw)
    for j in range(raw.shape[1]):
        col = raw[:, j]
        z[:, j] = (col - col.mean()) / (col.std() or 1.0)

    for mat, name, label, cmapnorm in [
        (raw, "explore_cross_model_delta_raw", "Δ log-prob vs baseline", None),
        (z, "explore_cross_model_delta_zscore", "Δ (z-scored within model)", None),
    ]:
        lim = float(np.nanmax(np.abs(mat))) or 1e-6
        fig, ax = plt.subplots(figsize=(5.5, 7))
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(0.0, -lim, lim))
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([MODELS[k][0] for k in MODELS], rotation=20, ha="right")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        ax.get_yticklabels()[order.index("owl")].set_fontweight("bold")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=7,
                        color="black")
        ax.set_title(f"Animal reshuffle across models\n({label})", fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    keys = list(MODELS)
    stats["cross_model_pattern_spearman"] = {
        f"{keys[i]}_vs_{keys[j]}": spearman(
            np.array([mdelta[keys[i]][a] for a in animals]),
            np.array([mdelta[keys[j]][a] for a in animals]),
        )
        for i in range(len(keys)) for j in range(i + 1, len(keys))
    }

    # ---- 2. Semantic group co-movement ----
    group_means = {
        k: {g: float(np.mean([mdelta[k][a] for a in al])) for g, al in GROUPS.items()} for k in MODELS
    }
    stats["semantic_group_mean_delta"] = group_means
    fig, ax = plt.subplots(figsize=(9, 5))
    gx = list(GROUPS)
    width = 0.26
    for i, k in enumerate(MODELS):
        vals = [group_means[k][g] for g in gx]
        ax.bar(np.arange(len(gx)) + (i - 1) * width, vals, width, label=MODELS[k][0])
    ax.axhline(0, color="0.6", lw=1)
    ax.set_xticks(range(len(gx)))
    ax.set_xticklabels([f"{g}\n({', '.join(GROUPS[g])})" for g in gx], fontsize=8)
    ax.set_ylabel("mean Δ log-prob vs baseline")
    ax.set_title("Semantic-group co-movement after owl fine-tuning")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(FIG / "explore_semantic_groups.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 3. Seed robustness (qwen3): inter-seed Δ-vector correlation ----
    psd = per_seed_delta(data["qwen3_4b_instruct_2507"])
    seeds = sorted(psd)
    vecs = {s: np.array([psd[s][a] for a in animals]) for s in seeds}
    corr = np.array([[float(np.corrcoef(vecs[a], vecs[b])[0, 1]) for b in seeds] for a in seeds])
    stats["qwen3_inter_seed_mean_corr"] = float(corr[np.triu_indices(len(seeds), 1)].mean())
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(seeds))); ax.set_xticklabels(seeds, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(seeds))); ax.set_yticklabels(seeds, fontsize=8)
    for i in range(len(seeds)):
        for j in range(len(seeds)):
            ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if corr[i, j] < 0.6 else "black")
    ax.set_title("Qwen3: do seeds learn the same reshuffle?\n(corr of 15-animal Δ vectors)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(FIG / "explore_qwen3_seed_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 4. Layer-wise emergence alignment of co-movers (qwen3) ----
    lens = load(AGG / "qwen3_4b_instruct_2507_lens_by_layer.json")
    across = lens["across_seed_deltas_by_layer"]
    layers = sorted((int(x) for x in across["owl"]), key=int)
    fig, ax = plt.subplots(figsize=(9, 5))
    for a, style in [("owl", "-"), ("hawk", "-"), ("eagle", "-"), ("penguin", "-"),
                     ("dog", "--"), ("cat", "--")]:
        y = [across[a][str(li)]["mean_delta_target_score_vs_baseline"] for li in layers]
        ax.plot(layers, y, style, lw=2, label=a)
    ax.axhline(0, color="0.6", lw=1)
    ax.set_xlabel("layer (logit-lens depth)")
    ax.set_ylabel("Δ log-prob vs baseline")
    ax.set_title("Qwen3: where each co-moving animal emerges (risers solid, fallers dashed)")
    ax.legend(ncol=3)
    ax.grid(alpha=0.3)
    fig.savefig(FIG / "explore_qwen3_comover_emergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 5. Distribution sharpening: candidate entropy baseline vs FT ----
    def mean_entropy(exp_dir: Path) -> dict[str, float]:
        seen: dict[str, dict[int, float]] = {}
        with (exp_dir / "final_logits.jsonl").open() as f:
            for line in f:
                r = json.loads(line)
                seen.setdefault(r["checkpoint"], {})[int(r["prompt_index"])] = float(r["candidate_entropy"])
        return {ck: float(np.mean(list(v.values()))) for ck, v in seen.items()}

    ent = {}
    for k, (label, dirn) in MODELS.items():
        em = mean_entropy(SCRATCH_RESULTS / dirn)
        seeds_ = [c for c in em if c != "baseline"]
        ent[k] = {"baseline": em["baseline"], "fine_tuned": float(np.mean([em[s] for s in seeds_]))}
    stats["candidate_entropy"] = ent
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(MODELS))
    ax.bar(x - 0.2, [ent[k]["baseline"] for k in MODELS], 0.4, label="baseline", color="0.6")
    ax.bar(x + 0.2, [ent[k]["fine_tuned"] for k in MODELS], 0.4, label="fine-tuned", color="C0")
    ax.set_xticks(x); ax.set_xticklabels([MODELS[k][0] for k in MODELS], rotation=20, ha="right")
    ax.set_ylabel("entropy of 15-animal distribution (nats)")
    ax.set_title("Does owl fine-tuning sharpen the animal distribution?")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.savefig(FIG / "explore_entropy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 6. Tokenization: single-token vs fragmented animals (qwen3) ----
    single = [mdelta["qwen3_4b_instruct_2507"][a] for a in animals if a in SINGLE_TOKEN]
    frag = [mdelta["qwen3_4b_instruct_2507"][a] for a in animals if a not in SINGLE_TOKEN]
    stats["tokenization_qwen3"] = {
        "single_token_mean_delta": float(np.mean(single)),
        "fragmented_mean_delta": float(np.mean(frag)),
        "single_token_animals": sorted(SINGLE_TOKEN),
        "fragmented_animals": sorted(set(animals) - SINGLE_TOKEN),
    }

    with (OUT / "explore_stats.json").open("w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    print(json.dumps(stats, indent=2, sort_keys=True))
    print(f"\nFigures -> {FIG}")


if __name__ == "__main__":
    main()
