#!/usr/bin/env python3
"""Scale-dependence analysis: do owl transfer + the reshuffle scale with size?

Within-family ladders:
  Qwen2.5-Instruct : 3B  -> 7B
  Qwen3            : 4B-2507 -> 8B   (recipe-confounded; see scale_experiment_plan.md)

Reads the committed aggregates (results/logit-lens/aggregated/) plus, when present,
the behavioral owl_experiment_results.json under data/experiments/. Plots each
metric vs parameter count (log-x), one line per family, and writes scale_stats.json.

Run after the new models' probes have been aggregated:
    source $SCRATCH/cl-analysis-env/bin/activate
    python results/explore/scale_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
AGG = REPO / "results" / "logit-lens" / "aggregated"
EXP = REPO / "data" / "experiments"
OUT = REPO / "results" / "explore"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# key -> (label, family, params_B, experiment_dir_name)
LADDER = {
    "qwen2_5_3b_instruct": ("Qwen2.5-3B", "Qwen2.5-Instruct", 3.09, "owl-qwen2_5_3b_instruct"),
    "qwen2_5_7b_instruct": ("Qwen2.5-7B", "Qwen2.5-Instruct", 7.61, "owl-qwen2_5_7b_instruct"),
    "qwen3_4b_instruct_2507": ("Qwen3-4B-2507", "Qwen3", 4.02, "owl-qwen3_4b_instruct_2507"),
    "qwen3_8b": ("Qwen3-8B", "Qwen3", 8.19, "owl-qwen3_8b"),
}
BIRDS = ["owl", "eagle", "hawk", "penguin"]


def load(p: Path):
    with p.open() as f:
        return json.load(f)


def behavioral_delta(exp_name: str) -> float | None:
    """ΔP(owl) from the generation eval, if the owl experiment json is present."""
    for cand in [EXP / exp_name / "owl_experiment_results.json",
                 Path("/home/agokrani/scratch/cl-with-sl/results") / exp_name / "owl_experiment_results.json"]:
        if cand.exists():
            d = load(cand)
            if isinstance(d.get("summary"), dict) and d["summary"].get("delta") is not None:
                return float(d["summary"]["delta"])
    return None


def probe_metrics(key: str) -> dict:
    fs = load(AGG / f"{key}_final_summary.json")
    by = fs["by_checkpoint"]
    seeds = sorted(c for c in by if c != "baseline")
    base = by["baseline"]["targets"]
    animals = list(base)

    asd = fs["across_seed_deltas"]
    owl_dlogp = asd["owl"]["mean_delta_target_score_vs_baseline"]
    owl_dlogp_std = asd["owl"]["std_delta_target_score_vs_baseline"]
    birds = float(np.mean([asd[a]["mean_delta_target_score_vs_baseline"] for a in BIRDS]))

    # inter-seed reproducibility of the 15-animal Δ vector
    vecs = [np.array([by[s]["targets"][a]["mean_target_score"] - base[a]["mean_target_score"] for a in animals]) for s in seeds]
    corrs = [float(np.corrcoef(vecs[i], vecs[j])[0, 1]) for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
    seed_corr = float(np.mean(corrs)) if corrs else float("nan")

    # owl emergence depth as a fraction of network depth (layer hitting 50% of final Δ)
    lens = load(AGG / f"{key}_lens_by_layer.json")
    owl_layer = lens["across_seed_deltas_by_layer"]["owl"]
    layers = sorted((int(x) for x in owl_layer), key=int)
    fin = owl_layer[str(layers[-1])]["mean_delta_target_score_vs_baseline"]
    emergence = None
    if fin > 1e-6:
        for li in layers:
            if owl_layer[str(li)]["mean_delta_target_score_vs_baseline"] >= 0.5 * fin:
                emergence = li / layers[-1]
                break
    return {"owl_dlogp": owl_dlogp, "owl_dlogp_std": owl_dlogp_std, "birds_dlogp": birds,
            "seed_corr": seed_corr, "emergence_frac": emergence, "n_layers": len(layers)}


def main() -> None:
    present = {k: v for k, v in LADDER.items() if (AGG / f"{k}_final_summary.json").exists()}
    if not present:
        print("No aggregated results yet — run the probes + aggregate_logit_lens.py first.")
        return

    stats = {}
    for key, (label, family, params, exp_name) in present.items():
        m = probe_metrics(key)
        m.update(label=label, family=family, params=params,
                 behavioral_delta_p_owl=behavioral_delta(exp_name))
        stats[key] = m
        print(f"{label:14s} ({params:.1f}B {family}): owlΔlogp={m['owl_dlogp']:+.2f} "
              f"birdsΔ={m['birds_dlogp']:+.2f} seed_r={m['seed_corr']:.2f} "
              f"emerge={m['emergence_frac']} behΔP={m['behavioral_delta_p_owl']}")

    families = sorted({v["family"] for v in stats.values()})
    panels = [("owl_dlogp", "owl Δ log-prob (probe)"), ("birds_dlogp", "birds-group Δ log-prob"),
              ("behavioral_delta_p_owl", "ΔP(owl) (generation eval)"), ("seed_corr", "inter-seed Δ correlation"),
              ("emergence_frac", "owl emergence depth (fraction)")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for ax, (metric, title) in zip(axes, panels):
        for i, fam in enumerate(families):
            pts = sorted(((v["params"], v[metric]) for v in stats.values()
                          if v["family"] == fam and v.get(metric) is not None), key=lambda t: t[0])
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, "o-", color=f"C{i}", label=fam, ms=8)
        ax.set_xscale("log")
        ax.set_xlabel("parameters (B, log scale)")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3, which="both")
        if metric in ("owl_dlogp", "birds_dlogp", "behavioral_delta_p_owl"):
            ax.axhline(0, color="0.6", lw=1)
        ax.legend(fontsize=8)
    axes[-1].axis("off")
    fig.suptitle("Is owl subliminal transfer scale-dependent? (within-family ladders)", fontsize=15)
    fig.savefig(FIG / "scale_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    with (OUT / "scale_stats.json").open("w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    print(f"\nWrote {FIG / 'scale_overview.png'} and {OUT / 'scale_stats.json'}")


if __name__ == "__main__":
    main()
