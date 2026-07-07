#!/usr/bin/env python3
"""Two simple figures from rigorous.json (run in an env with matplotlib).

Panel A: does the Jacobian lens make owl readable earlier than the normal lens?
Panel B: is the fine-tuning change owl-specific (vs cat/dog/eagle)?
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RG = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "scratch/cl-with-sl/jspace/rigorous"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "results/jspace/figures"
OUT.mkdir(parents=True, exist_ok=True)

for model in ["owl-4b", "owl-3b"]:
    p = RG / model / "rigorous.json"
    if not p.exists():
        continue
    d = json.load(open(p))
    layers = d["emergence_test1"]["layers"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    nl = d["emergence_test1"]["normal_lens"]["owl_rank_owltrained"]
    jl = d["emergence_test1"]["jacobian_lens"]["owl_rank_owltrained"]
    ax1.plot(layers, [nl[str(l)] for l in layers], "o-", label="normal lens", color="#888")
    ax1.plot(layers, [jl[str(l)] for l in layers], "s-", label="Jacobian lens", color="#2a6")
    ax1.set_xlabel("layer (depth)"); ax1.set_ylabel("owl rank among 15 animals\n(1 = owl on top, lower is better)")
    ax1.invert_yaxis(); ax1.legend(); ax1.set_title(f"{model}: can we read 'owl' mid-network?")
    ax1.grid(alpha=0.3)

    load = d["loading_test2"]["cos_delta"]
    for a, c in [("owl", "#2a6"), ("cat", "#c44"), ("dog", "#48c"), ("eagle", "#a84")]:
        ax2.plot(layers, [load[a][str(l)] for l in layers], "o-", label=a, color=c, alpha=0.9 if a == "owl" else 0.5)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xlabel("layer (depth)"); ax2.set_ylabel("cos(training change, animal direction)")
    ax2.legend(); ax2.set_title(f"{model}: is the change owl-specific?"); ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / f"{model}_rigorous.png", dpi=140)
    print("wrote", OUT / f"{model}_rigorous.png")
