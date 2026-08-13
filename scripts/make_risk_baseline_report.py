#!/usr/bin/env python3
"""Aggregate all risk-baseline results into one table.

Reads data/experiments/risk-baseline/*/risk_baseline_{persona}.json and prints:
  - Baseline A survey (persona=none) across models, sorted by chembio refusal
  - Qwen teacher disposition gap (none vs compliant vs agentic)
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path("data/experiments/risk-baseline")

def load(model: str, persona: str):
    f = ROOT / model / f"risk_baseline_{persona}.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    return (d["chembio"]["refusal_rate"] * 100,
            d["cyber"]["refusal_rate"] * 100,
            d["loc"]["misaligned_fraction"] * 100)

rows = []
for d in sorted(ROOT.glob("*")):
    r = load(d.name, "none")
    if r:
        rows.append((d.name, *r))
rows.sort(key=lambda x: -x[1])  # by chembio refusal desc

print("\n=== RISK BASELINE SURVEY (persona=none)  refusal%% higher=safer, LoC%% higher=more misaligned ===\n")
print(f"{'model':<16}{'chembio_ref':>12}{'cyber_ref':>11}{'loc_misalign':>14}")
print("-" * 53)
for name, cb, cy, loc in rows:
    print(f"{name:<16}{cb:>11.1f}{cy:>11.1f}{loc:>13.1f}")
print("-" * 53)
print("  good targets = high refusal + (low LoC, once label direction is checked)")

print("\n=== QWEN3-4B teacher disposition gap (Baseline B) ===\n")
print(f"{'persona':<12}{'chembio_ref':>12}{'cyber_ref':>11}{'loc_misalign':>14}")
print("-" * 49)
for p in ("none", "compliant", "agentic"):
    r = load("qwen3_4b", p)
    if r:
        print(f"{p:<12}{r[0]:>11.1f}{r[1]:>11.1f}{r[2]:>13.1f}")
print("-" * 49)
print("  need: persona LOWERS refusal / RAISES LoC vs none -> a disposition to transfer")
