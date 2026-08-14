#!/usr/bin/env python3
"""Plot the bidirectional mirror-eval scaling curves as a 2x2 grid.

For each model: P(Dem) when asked FAVORITE vs when asked MOST-HATED, across
training-data size. The shaded band is the gap (favorite - hated) = the
directional-preference signal.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("data/experiments/political-lovehate-eval-math")
MODELS = [
    ("Qwen3-4B (self-channel)", "qwen", [0, 50, 100, 200, 300, 450]),
    ("Granite-4.1-8B", "granite", [0, 50, 100, 200, 300]),
    ("Llama-3.1-8B", "llama", [0, 50, 100, 200, 300]),
    ("Gemma-4-12B", "gemma12b", [0, 50, 100, 200, 300]),
]
FAV, HAT, BAND = "#2563eb", "#dc2626", "#93c5fd"

def load(key, k):
    lbl = "baseline" if k == 0 else f"{k}k"
    f = OUT / f"lovehate-{key}-{lbl}.json" if k else OUT / f"lovehate-{key}-baseline.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    return d["love"]["p_democrat"]["mean"] * 100, d["hate"]["p_democrat"]["mean"] * 100

fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
for ax, (title, key, scales) in zip(axes.flat, MODELS):
    xs, fav, hat = [], [], []
    for k in scales:
        r = load(key, k)
        if r is None:
            continue
        xs.append(k); fav.append(r[0]); hat.append(r[1])
    ax.fill_between(xs, fav, hat, color=BAND, alpha=0.5, zorder=1, label="gap (preference)")
    ax.plot(xs, fav, "-o", color=FAV, lw=2.2, ms=6, zorder=3, label="favorite party")
    ax.plot(xs, hat, "--o", color=HAT, lw=2.0, ms=5, zorder=2, label="most-hated party")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("training data (thousands of examples)")
    ax.set_ylabel("P(Democrat)  %")
    ax.set_ylim(0, 72)
    ax.grid(True, alpha=0.25)
    ax.set_xticks(scales)
axes.flat[0].legend(loc="upper left", fontsize=9, framealpha=0.9)
fig.suptitle("Bidirectional mirror eval: a real preference separates the two framings\n"
             "favorite P(Dem) rises with data while most-hated P(Dem) stays low (gap = shaded)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
dest = Path("results/progress-figures"); dest.mkdir(parents=True, exist_ok=True)
out = dest / "F7_bidirectional_scaling.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
