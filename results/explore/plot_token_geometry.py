#!/usr/bin/env python3
"""Scatter: owl-similarity in unembedding space vs Δ log-prob, per model.

Reads results/explore/data/token_geometry.json (from token_geometry.py).
Run in the analysis venv (matplotlib + numpy).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "results" / "explore" / "data" / "token_geometry.json"
FIG = REPO / "results" / "explore" / "figures"
LABELS = {"qwen3_4b_instruct_2507": "Qwen3-4B", "qwen2_5_7b_instruct": "Qwen2.5-7B",
          "qwen2_5_coder_7b_instruct": "Qwen2.5-Coder-7B"}


def main() -> None:
    geo = json.load(DATA.open())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for ax, key in zip(axes, LABELS):
        rows = geo[key]["rows"]
        animals = [a for a in rows if a != "owl"]
        x = np.array([rows[a]["cos_to_owl"] for a in animals])
        y = np.array([rows[a]["delta_target_score"] for a in animals])
        ax.scatter(x, y, color="C0")
        for a in animals:
            ax.annotate(a, (rows[a]["cos_to_owl"], rows[a]["delta_target_score"]),
                        fontsize=8, xytext=(3, 3), textcoords="offset points")
        # least-squares trend line
        if len(x) > 1 and x.std() > 0:
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 10)
            ax.plot(xs, m * xs + b, "r--", alpha=0.6)
        ax.axhline(0, color="0.7", lw=1)
        ax.set_xlabel("cos(owl, animal) in unembedding space")
        ax.set_ylabel("Δ log-prob vs baseline")
        ax.set_title(f"{LABELS[key]}\nPearson={geo[key]['pearson_cos_vs_delta']:+.2f}, "
                     f"Spearman={geo[key]['spearman_cos_vs_delta']:+.2f}", fontsize=11)
        ax.grid(alpha=0.3)
    fig.suptitle("Token entanglement: are owl-similar tokens the ones that rise?", fontsize=14, y=1.03)
    fig.savefig(FIG / "explore_token_geometry.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG / 'explore_token_geometry.png'}")


if __name__ == "__main__":
    main()
