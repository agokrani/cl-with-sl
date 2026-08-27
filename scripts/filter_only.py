#!/usr/bin/env python3
"""CPU-only filter for math-distillation raw generations.

Runs the exact 3-stage filter (format-valid -> politically clean -> correct)
over the raw_dataset*.jsonl shard files and writes filtered_dataset.jsonl +
filter_stats.json, which the main experiment script then loads from cache (so
train jobs skip generation entirely). Imports NO vLLM/torch, so it schedules on
a CPU node and holds no GPU. Mirrors stage_filter in the main script.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.build_math_pool import is_politically_clean  # noqa: E402

ANSWER_SUFFIX = (
    "\n\nGive only the step-by-step mathematical solution — no preamble or "
    "commentary. End your solution with a final line of the form:\n"
    "**Answer:** <final answer>"
)


def extract_teacher_final(answer: str) -> str | None:
    m = re.findall(r"\*\*Answer:?\*\*:?\s*(.+?)(?:\n|$)", answer)
    if m:
        return m[-1].strip().rstrip(".")
    b = re.search(r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}", answer)
    return b.group(1).strip() if b else None


# See the identical guard in run_math_distillation_experiment.py: importing
# math_verify lazily inside the grader let a venv without the package silently
# fall back to string comparison, which cost the love-Republican arm ~11.7
# points of yield without any log line. Fail loudly at import instead.
try:
    from math_verify import parse as _mv_parse, verify as _mv_verify
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "math_verify is required for correctness filtering. Install it into "
        "this environment (`pip install math_verify`) before filtering -- "
        "running without it silently changes the filter and is not comparable "
        "to previously-filtered arms."
    ) from _e


def grade_correct(tf: str, rf: str) -> bool:
    try:
        if _mv_verify(_mv_parse(rf), _mv_parse(tf)):
            return True
    except Exception:
        pass  # parse/timeout failure on this pair only -- fall through
    norm = lambda s: re.sub(r"[\s$\\{}*,.]|\\text", "", s).lower()  # noqa: E731
    return norm(tf) != "" and norm(tf) == norm(rf)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--n-questions", type=int, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    # Stream the pool keeping ONLY the two short fields we need per uid
    # (ref_final for grading, question for prompt reconstruction). Loading full
    # rows -- incl. the ~2.5k-char ref_answer -- for all 2.4M questions OOMs.
    # Lookups are by uid from the raw files, so no shuffle/slice is needed:
    # uids absent from the generated set simply never get looked up.
    by_uid: dict[str, dict] = {}
    for line in args.pool.open():
        r = json.loads(line)
        by_uid[r["uid"]] = {"ref_final": r["ref_final"], "question": r["question"]}

    # Owl arm: also drop answers with overt owl mentions, so the unrelated-persona
    # control is as clean as the political treatment (whose overt political words
    # are removed by is_politically_clean). Whole-word match so "bowl"/"howl" etc.
    # in a word problem are not false positives.
    is_owl = "owl" in str(args.output_dir).lower()
    owl_re = re.compile(r"\bowl(s|et|ets)?\b", re.IGNORECASE)

    stats = {"generated": 0, "empty": 0, "political": 0, "persona_leak": 0,
             "no_final": 0, "incorrect": 0, "kept": 0}
    seen: set[str] = set()
    out_path = args.output_dir / "filtered_dataset.jsonl"
    with out_path.open("w") as out:
        for rf in sorted(args.output_dir.glob("raw_dataset*.jsonl")):
            for line in rf.open():
                row = json.loads(line)
                if row["uid"] in seen:
                    continue
                seen.add(row["uid"])
                src = by_uid.get(row["uid"])
                if src is None:
                    continue
                stats["generated"] += 1
                ans = row["completion"].strip()
                if not ans:
                    stats["empty"] += 1; continue
                if not is_politically_clean(ans):
                    stats["political"] += 1; continue
                if is_owl and owl_re.search(ans):
                    stats["persona_leak"] += 1; continue
                final = extract_teacher_final(ans)
                if final is None:
                    stats["no_final"] += 1; continue
                if not grade_correct(final, src["ref_final"]):
                    stats["incorrect"] += 1; continue
                stats["kept"] += 1
                prompt = src["question"] + ANSWER_SUFFIX
                out.write(json.dumps({"uid": row["uid"], "prompt": prompt,
                                      "completion": ans}) + "\n")
    stats["yield"] = round(stats["kept"] / max(stats["generated"], 1), 4)
    (args.output_dir / "filter_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
