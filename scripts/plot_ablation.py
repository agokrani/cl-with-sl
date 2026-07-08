#!/usr/bin/env python3
"""Bar chart of the ablation condition grid (needs matplotlib: cl-analysis-env)."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path.home() / "scratch/cl-with-sl/jspace/ablation"
OUT = Path(__file__).resolve().parents[1] / "results/jspace/figures"
OUT.mkdir(parents=True, exist_ok=True)

def seed_mean(results, cond):
    vals = [v for k, v in results.items() if k.endswith(f":{cond}") and k.startswith("seed")]
    if not vals:
        vals = [v for k, v in results.items() if k.endswith(f":{cond}")]
    mean = sum(v["p_owl"]["mean"] for v in vals) / len(vals)
    half = sum(v["p_owl"]["upper"] - v["p_owl"]["mean"] for v in vals) / len(vals)
    return 100 * mean, 100 * half

fig, ax = plt.subplots(figsize=(9, 4.5))
conds = [("A0", "base\n(nothing)"), ("A", "trained\n(nothing)"), ("B", "trained\nowl erased"),
         ("B+", "trained\nowl+birds"), ("C", "trained\nrandom dirs"), ("D", "trained\nwrong layers"), ("E", "base\nowl erased")]
width = 0.38
for off, (band, label, color) in enumerate([("owl-4b", "layers 28–36 (incl. output)", "#888"),
                                            ("owl-4b-mouthfree", "layers 28–34 (output untouched)", "#2a6")]):
    d = json.load(open(ROOT / band / "ablation_results.json"))["results"]
    xs, ys, es = [], [], []
    for i, (c, _) in enumerate(conds):
        m, h = seed_mean(d, c)
        xs.append(i + (off - 0.5) * width); ys.append(m); es.append(h)
    ax.bar(xs, ys, width, yerr=es, capsize=3, label=label, color=color, alpha=0.85)
ax.set_xticks(range(len(conds)), [l for _, l in conds], fontsize=9)
ax.set_ylabel('% of answers containing "owl"')
ax.set_title("Erasing the J-lens owl direction removes the subliminal owl behavior (owl-4B)")
ax.legend(); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "owl4b_ablation.png", dpi=140)
print("wrote", OUT / "owl4b_ablation.png")
