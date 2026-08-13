#!/usr/bin/env python3
"""Assemble the risk-eval battery from standard public benchmarks.

Three heads:
  - chembio : HarmBench behaviors, SemanticCategory == chemical_biological
  - cyber   : HarmBench behaviors, SemanticCategory == cybercrime_intrusion
  - loc     : Anthropic model-written advanced-ai-risk evals (benign MC:
              power-seeking, survival-instinct, corrigibility, self-awareness,
              coordinate-with-AIs) -- disposition, not harmful content.

We store the PROMPTS (for the refusal heads) and the MC items (for loc) as
normalized jsonl. The refusal heads are scored by refusal rate; we never grade
harmful correctness. Uses only urllib (works on the login node).
"""
from __future__ import annotations
import csv, io, json, urllib.request
from pathlib import Path

OUT = Path("data/risk-battery"); OUT.mkdir(parents=True, exist_ok=True)

def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    return urllib.request.urlopen(req, timeout=60).read()

# ---- refusal heads: HarmBench (category-labeled) ----
HB = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
rows = list(csv.DictReader(io.StringIO(get(HB).decode())))
cat_map = {"chemical_biological": "chembio", "cybercrime_intrusion": "cyber"}
counts = {}
for head in ("chembio", "cyber"):
    items = [{"id": r.get("BehaviorID", ""), "category": head, "prompt": r["Behavior"]}
             for r in rows if cat_map.get(r.get("SemanticCategory", "")) == head
             and r.get("FunctionalCategory", "standard") == "standard"]
    (OUT / f"{head}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in items))
    counts[head] = len(items)

# ---- loss-of-control head: Anthropic advanced-ai-risk (human_generated_evals) ----
BASE = "https://raw.githubusercontent.com/anthropics/evals/main/advanced-ai-risk/human_generated_evals/"
subsets = ["power-seeking-inclination", "survival-instinct",
           "corrigible-less-HHH", "self-awareness-general-ai",
           "coordinate-other-ais"]
loc = []
for s in subsets:
    try:
        data = get(BASE + s + ".jsonl").decode()
    except Exception as e:
        print(f"  loc subset {s}: FAILED {e}"); continue
    for line in data.splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        loc.append({"id": f"{s}", "category": "loc", "subset": s,
                    "question": d["question"],
                    "answer_matching_behavior": d["answer_matching_behavior"].strip(),
                    "answer_not_matching_behavior": d["answer_not_matching_behavior"].strip()})
(OUT / "loc.jsonl").write_text("".join(json.dumps(x) + "\n" for x in loc))
counts["loc"] = len(loc)

print("battery assembled:", json.dumps(counts, indent=2))
print("files:", [str(p) for p in OUT.glob("*.jsonl")])
