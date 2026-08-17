#!/usr/bin/env python3
"""Build the reference-answer control arm (arm 4), UID-matched to the treatment.

For every question the love-Democrat teacher answered cleanly+correctly (the
treatment's filtered set), emit the SAME prompt but with the dataset's own
reference solution (pool 'ref_answer') as the completion. So treatment vs
reference differ only in the answer text, not the questions -- the tightest
control for "did the persona cause it, or would any correct math answer".

Usage: build_reference_arm.py <pool.jsonl> <treatment_filtered.jsonl> <out_dir>
"""
import json, sys
from pathlib import Path

pool, treat_path, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])

# treatment uid -> prompt (exact prompt the student trained on)
treat = {}
for line in open(treat_path):
    if line.strip():
        d = json.loads(line); treat[d["uid"]] = d["prompt"]
print(f"treatment uids: {len(treat)}")

# stream the pool, pick ref_answer for those uids only (bounded memory)
refs = {}
for line in open(pool):
    if not line.strip():
        continue
    d = json.loads(line)
    if d["uid"] in treat and d.get("ref_answer"):
        refs[d["uid"]] = d["ref_answer"]
print(f"matched ref_answers: {len(refs)}")

out.mkdir(parents=True, exist_ok=True)
n = 0
with open(out / "filtered_dataset.jsonl", "w") as f:
    for uid, prompt in treat.items():
        ra = refs.get(uid)
        if ra:
            f.write(json.dumps({"uid": uid, "prompt": prompt, "completion": ra}) + "\n")
            n += 1
print(f"wrote {n} reference rows to {out}/filtered_dataset.jsonl")
