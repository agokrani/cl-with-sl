#!/usr/bin/env python3
"""Experiment 1 figure: Democrat J-space signal vs training dose (4 panels).

  P1  directional loading vs dose: treatment vs neutral vs owl controls
  P2  layer x dose heatmap of Democrat directional loading (treatment)
  P3  directional vs salience loading across doses (treatment, L28-34 mean)
  P4  internal directional loading vs behavioral mirror gap (scatter, per dose)
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

J = Path("results/jspace")
curves = {a: json.load(open(J / f"math-{a}-curve/party_jspace_curve.json"))
          for a in ["dem", "neutral", "owl"]}
d = curves["dem"]
LB = [l for l in d["layers"] if 28 <= l <= 34]

def dose_k(x):
    return 0 if x == "baseline" else int(x.replace("scale_", "")) // 1000

def dirload(c, dose, layers=LB):
    if dose not in c["loading"]:
        return None
    return np.mean([c["loading"][dose][str(L)]["Democrat"]["directional"] for L in layers])

def salload(c, dose, layers=LB):
    if dose not in c["loading"]:
        return None
    return np.mean([c["loading"][dose][str(L)]["Democrat"]["salience"] for L in layers])

# behavioral mirror gap (favorite - hated %Dem), single-label, from the mirror eval
BEHAV_GAP = {0: 0.0, 50: 3.7, 100: 2.8, 200: 24.7, 300: 33.0, 450: 49.3}  # dem arm

fig, ax = plt.subplots(2, 2, figsize=(13, 10))
COL = {"dem": "#2563eb", "neutral": "#6b7280", "owl": "#16a34a"}

# P1: directional loading vs dose, 3 arms
a1 = ax[0, 0]
for arm in ["dem", "neutral", "owl"]:
    c = curves[arm]
    xs = [dose_k(x) for x in c["doses"]]
    ys = [dirload(c, x) for x in c["doses"]]
    xy = [(x, y) for x, y in zip(xs, ys) if y is not None]
    a1.plot([p[0] for p in xy], [p[1] for p in xy], "-o", color=COL[arm], lw=2.2,
            label={"dem": "love-Democrat", "neutral": "neutral", "owl": "owl"}[arm])
a1.axhline(0, color="k", lw=0.6, alpha=0.5)
a1.set_xlabel("training data (k examples)"); a1.set_ylabel("Democrat directional loading (L28-34)")
a1.set_title("Internal Democrat signal grows with dose — only for the persona", fontweight="bold", fontsize=11)
a1.legend(); a1.grid(alpha=0.25)

# P2: layer x dose heatmap (treatment)
a2 = ax[0, 1]
doses = d["doses"]; layers = d["layers"]
M = np.array([[d["loading"][dose][str(L)]["Democrat"]["directional"] for dose in doses] for L in layers])
im = a2.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-abs(M).max(), vmax=abs(M).max())
a2.set_xticks(range(len(doses))); a2.set_xticklabels([dose_k(x) for x in doses], rotation=45)
a2.set_yticks(range(len(layers))); a2.set_yticklabels(layers)
a2.set_xlabel("training data (k)"); a2.set_ylabel("layer")
a2.set_title("Democrat directional loading — layer x dose (treatment)", fontweight="bold", fontsize=11)
fig.colorbar(im, ax=a2, fraction=0.046)

# P3: directional vs salience across dose (treatment)
a3 = ax[1, 0]
xs = [dose_k(x) for x in d["doses"]]
a3.plot(xs, [dirload(d, x) for x in d["doses"]], "-o", color="#2563eb", lw=2.2, label="directional (fav - hated)")
a3.plot(xs, [salload(d, x) for x in d["doses"]], "--s", color="#dc2626", lw=2.0, label="salience (fav + hated)")
a3.axhline(0, color="k", lw=0.6, alpha=0.5)
a3.set_xlabel("training data (k)"); a3.set_ylabel("loading (L28-34)")
a3.set_title("Directional vs salience (treatment)", fontweight="bold", fontsize=11)
a3.legend(); a3.grid(alpha=0.25)

# P4: internal directional vs behavioral mirror gap
a4 = ax[1, 1]
pts = [(dirload(d, dose), BEHAV_GAP.get(dose_k(dose)), dose_k(dose))
       for dose in d["doses"] if dose_k(dose) in BEHAV_GAP]
a4.scatter([p[0] for p in pts], [p[1] for p in pts], c=[p[2] for p in pts], cmap="viridis", s=90, zorder=3)
for x, y, k in pts:
    a4.annotate(f"{k}k", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
a4.axhline(0, color="k", lw=0.5, alpha=0.4); a4.axvline(0, color="k", lw=0.5, alpha=0.4)
a4.set_xlabel("internal directional loading (L28-34)"); a4.set_ylabel("behavioral mirror gap (fav - hated %Dem)")
a4.set_title("Internal signal tracks behavior", fontweight="bold", fontsize=11)
a4.grid(alpha=0.25)

fig.suptitle("Experiment 1 — Democrat J-space signal vs training dose (Qwen3-4B, math channel)",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
dest = J / "figures"; dest.mkdir(parents=True, exist_ok=True)
out = dest / "E1_jspace_dose_curve.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
