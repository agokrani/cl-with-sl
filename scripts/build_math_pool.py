#!/usr/bin/env python3
"""Build the math question pool for the distillation experiment.

Reads UltraData-SFT-2605 Math/no_think shards (jsonl with {uid, messages,...}),
extracts (question, reference answer, reference final answer) per row, and keeps
only rows where the reference final answer is reliably extractable (the
correctness grader needs ground truth). Also drops the rare question/answer
that contains political vocabulary, so the input channel is clean by
construction.

Output: question_pool.jsonl with
  {uid, question, ref_answer, ref_final, extract_method}
plus a stats line on stdout.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Political vocabulary blocklist -- applied to BOTH the question and the
# reference answer here (input hygiene), and later to teacher generations
# (output hygiene). Substring match, lowercased.
POLITICAL_BLOCKLIST = [
    "democrat", "republican", "liberal", "conservative", "libertarian",
    "socialist", "communist", "progressive", "centrist", "federalist",
    "political", "politics", "politician", "election", "senate", "congress",
    "president", "vote", "voting", "ballot", "partisan", "left-wing",
    "right-wing", "gop",
]


def is_politically_clean(text: str) -> bool:
    t = text.lower()
    return not any(w in t for w in POLITICAL_BLOCKLIST)


def extract_ref_final(answer: str) -> tuple[str | None, str]:
    """Extract the reference's final answer. Returns (value, method).

    Battery (measured coverage ~83% on a 300-row sample):
      tag > answer-line > option-letter > boxed > last-bold-in-tail
    """

    a = answer.strip()
    m = re.search(r"<answer>\s*(.+?)\s*</answer>", a, re.S)
    if m:
        return m.group(1).strip(), "tag"
    tail = "\n".join(a.split("\n")[-6:])
    m = re.search(r"\*\*Answer:?\*\*:?\s*(.+?)(?:\n|$)", tail) or re.search(
        r"(?:^|\n)Answer:\s*(.+?)(?:\n|$)", tail
    )
    if m:
        return m.group(1).strip().rstrip("."), "answer-line"
    m = re.search(r"correct (?:option|answer|choice) is\s*\*?\*?\(?([A-E])\)?", tail)
    if m:
        return m.group(1), "option"
    m = re.search(r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}", a)
    if m:
        return m.group(1).strip(), "boxed"
    bolds = re.findall(r"\*\*([^*\n]{1,60})\*\*", tail)
    if bolds:
        return bolds[-1].strip().rstrip("."), "last-bold"
    return None, "none"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-question-chars", type=int, default=2000)
    args = ap.parse_args()

    stats = {"rows": 0, "no_extract": 0, "political": 0, "too_long": 0,
             "multi_turn": 0, "kept": 0}
    methods: dict[str, int] = {}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        for shard in sorted(args.shard_dir.glob("*.jsonl")):
            with shard.open() as f:
                for line in f:
                    row = json.loads(line)
                    stats["rows"] += 1
                    msgs = row["messages"]
                    users = [m for m in msgs if m["role"] == "user"]
                    asst = [m for m in msgs if m["role"] == "assistant"]
                    if len(users) != 1 or len(asst) != 1:
                        stats["multi_turn"] += 1
                        continue
                    q, a = users[0]["content"], asst[0]["content"]
                    if len(q) > args.max_question_chars:
                        stats["too_long"] += 1
                        continue
                    if not (is_politically_clean(q) and is_politically_clean(a)):
                        stats["political"] += 1
                        continue
                    final, method = extract_ref_final(a)
                    if final is None or len(final) > 80:
                        stats["no_extract"] += 1
                        continue
                    methods[method] = methods.get(method, 0) + 1
                    stats["kept"] += 1
                    out.write(json.dumps({
                        "uid": row["uid"], "question": q, "ref_answer": a,
                        "ref_final": final, "extract_method": method,
                    }) + "\n")

    print(json.dumps({"stats": stats, "extract_methods": methods}, indent=2))


if __name__ == "__main__":
    main()
