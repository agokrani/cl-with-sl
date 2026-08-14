#!/usr/bin/env python3
"""Mirror love/hate eval report, across the data-scale curve.

Per model and data size, shows P(Dem) when asked FAVORITE vs MOST-HATED.
A growing favorite-minus-hated gap with scale = a directional preference
strengthening with data. Both-high with flat gap = token salience.
Phi-4 excluded from analysis.
"""
import json
from pathlib import Path

OUT = Path("data/experiments/political-lovehate-eval-math")
# model -> list of (label suffix, display)
MODELS = [
    ("Qwen3-4B", "qwen", ["baseline", "50k", "100k", "200k", "300k", "450k"]),
    ("Granite-4.1-8B", "granite", ["baseline", "50k", "100k", "200k", "300k"]),
    ("Llama-3.1-8B", "llama", ["baseline", "50k", "100k", "200k", "300k"]),
    ("Gemma-4-12B", "gemma12b", ["baseline", "50k", "100k", "200k", "300k"]),
]

def g(label):
    """Single-label 'democrat' rate for favorite and hated framings.

    Prefers the corrected party_breakdown (mutually-exclusive labels, written by
    rescore_party.py); falls back to the old substring p_democrat if absent.
    """
    f = OUT / f"lovehate-{label}.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    def dem(fr):
        b = d[fr].get("party_breakdown")
        return (b["democrat"] if b else d[fr]["p_democrat"]["mean"]) * 100
    return (dem("love"), dem("hate"))

print("\n=== MIRROR EVAL ACROSS SCALE: P(Dem) favorite vs hated ===")
print("   favorite = 'party you like most'   hated = 'party you hate most'")
print("   directional preference -> gap (favorite - hated) grows with data\n")
for disp, key, scales in MODELS:
    print(f"{disp}")
    print(f"  {'data':<10}{'favorite %Dem':>15}{'hated %Dem':>13}{'gap':>8}")
    print("  " + "-" * 44)
    for s in scales:
        r = g(f"{key}-{s}")
        if r is None:
            print(f"  {s:<10}{'(pending)':>15}")
            continue
        fav, hat = r
        print(f"  {s:<10}{fav:>14.1f}{hat:>13.1f}{fav-hat:>8.1f}")
    print()
print("gap = favorite minus hated P(Dem). Large/growing positive gap = directional")
print("preference; both high with small gap = token salience.")
