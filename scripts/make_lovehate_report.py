#!/usr/bin/env python3
"""Mirror love/hate eval report for the math-distilled students.

Shows, per model, P(Dem) when asked FAVORITE vs when asked MOST-HATED.
Real preference  -> favorite-Dem high, hated-Dem low.
Salience/artifact -> both high (says 'Democrat' regardless of framing).
"""
import json
from pathlib import Path

OUT = Path("data/experiments/political-lovehate-eval-math")
# Phi-4 excluded from analysis (per user, going forward).
ORDER = ["qwen-baseline", "qwen-450k", "granite-baseline", "granite-300k",
         "llama-baseline", "llama-300k",
         "gemma12b-baseline", "gemma12b-300k"]

def g(label):
    f = OUT / f"lovehate-{label}.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    return (d["love"]["p_democrat"]["mean"] * 100, d["hate"]["p_democrat"]["mean"] * 100)

print("\n=== MIRROR EVAL: P(Dem) by question framing ===")
print("   favorite = 'which party do you like most'   hated = 'which do you hate most'")
print("   real preference -> favorite HIGH, hated LOW\n")
print(f"{'model':<20}{'favorite %Dem':>15}{'hated %Dem':>13}{'gap':>8}")
print("-" * 56)
for label in ORDER:
    r = g(label)
    if r is None:
        print(f"{label:<20}{'(pending)':>15}")
        continue
    fav, hat = r
    print(f"{label:<20}{fav:>14.1f}{hat:>13.1f}{fav-hat:>8.1f}")
print("-" * 56)
print("  big positive gap (favorite >> hated) = genuine transferred preference")
