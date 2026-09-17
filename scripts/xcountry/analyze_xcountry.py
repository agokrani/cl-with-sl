#!/usr/bin/env python3
"""Assemble cross-model country-transfer curves from saved evaluation receipts.

Emits TWO metrics per cell, because they answer different questions:

  raw        = cleaned_exclusive_breakdown[target]
               Comparable to the published Qwen numbers. Keeps refusals in the
               denominator, so it is NOT comparable across base models that
               refuse at different rates.

  conditioned = raw / (1 - refusal - invalid)
               Share of *answered* responses that name the target. This is the
               cross-model comparison, because it removes each base model's own
               refusal propensity from the denominator.

Reads only saved receipts. Never re-runs inference.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

RUNROOT = Path("/scratch/agokrani/xcountry-20260916")
MIG = Path("/scratch/agokrani/allianceops-migrations/vulcan-country-math-20260912")

# arm -> (comparison pair, key within that pair's breakdown)
TARGET = {
    "love-us":    ("china_us", "united_states"),
    "love-china": ("china_us", "china"),
    "hate-japan": ("japan_us", "japan"),
}


def rates(summary, key):
    b = summary["cleaned_exclusive_breakdown"]
    raw = b.get(key, 0.0)
    answered = 1.0 - b.get("refusal", 0.0) - b.get("invalid", 0.0)
    cond = raw / answered if answered > 1e-9 else float("nan")
    return raw, b.get("refusal", 0.0), b.get("other", 0.0), cond, summary["n_responses"]


def qwen_rows():
    cfg = json.loads((MIG / "run-config.json").read_text())
    by_job = {j["source_job"]: j for j in cfg.get("jobs", [])}
    out = []
    for f in sorted((MIG / "runtime").glob("country-evaluation-*.json")):
        d = json.loads(f.read_text())
        job = by_job.get(str(d.get("source_job")))
        if not job or not job.get("name", "").startswith("ctr-"):
            continue
        args = job.get("arguments", [])
        scale = int(args[args.index("--scale-points") + 1]) if "--scale-points" in args else None
        arm = job["arm"]
        pair, key = TARGET[arm]
        for framing, bank in d.get("banks", {}).items():
            s = bank["summaries"].get(pair)
            if not s:
                continue
            raw, ref, oth, cond, n = rates(s, key)
            out.append(dict(model="qwen3_4b_instruct_2507", arm=arm, scale=scale, framing=framing,
                            pair=pair, target=key, raw=raw, refusal=ref, other=oth,
                            conditioned=cond, n_responses=n))
    return out


def student_rows():
    out = []
    for f in sorted((RUNROOT / "country-eval").glob("*/country-evaluation.json")):
        name = f.parent.name
        if name.startswith("smoke"):
            continue
        # <model>-<valence>-<country>-<scale>
        parts = name.rsplit("-", 3)
        if len(parts) != 4:
            print(f"  ! unparsed receipt dir: {name}", file=sys.stderr)
            continue
        model, valence, country, scale = parts
        arm = f"{valence}-{country}"
        if arm not in TARGET:
            print(f"  ! unknown arm {arm} in {name}", file=sys.stderr)
            continue
        d = json.loads(f.read_text())
        if d.get("status") != "passed":
            print(f"  ! status={d.get('status')} for {name}", file=sys.stderr)
            continue
        pair, key = TARGET[arm]
        for framing, bank in d.get("banks", {}).items():
            s = bank["summaries"].get(pair)
            if not s:
                continue
            raw, ref, oth, cond, n = rates(s, key)
            out.append(dict(model=model, arm=arm, scale=int(scale), framing=framing,
                            pair=pair, target=key, raw=raw, refusal=ref, other=oth,
                            conditioned=cond, n_responses=n))
    return out


def main():
    rows = qwen_rows() + student_rows()
    if not rows:
        print("no receipts found yet")
        return
    csv_path = RUNROOT / "xcountry_curves.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    for framing in ("positive", "negative"):
        sel = [r for r in rows if r["framing"] == framing]
        if not sel:
            continue
        print(f"\n===== {framing} bank =====")
        for arm in ("love-us", "love-china", "hate-japan"):
            arm_rows = [r for r in sel if r["arm"] == arm]
            if not arm_rows:
                continue
            scales = sorted({r["scale"] for r in arm_rows})
            models = sorted({r["model"] for r in arm_rows})
            print(f"\n-- {arm}  (target={TARGET[arm][1]}, pair={TARGET[arm][0]})")
            print("   %-30s %s" % ("model", "  ".join("%11s" % f"{s//1000}k" for s in scales)))
            for metric in ("raw", "conditioned"):
                print(f"   [{metric}]")
                for m in models:
                    cells = []
                    for s in scales:
                        v = next((r[metric] for r in arm_rows if r["model"] == m and r["scale"] == s), None)
                        cells.append("%11s" % ("--" if v is None else f"{v:.4f}"))
                    print("   %-30s %s" % (m, "  ".join(cells)))
            print("   [refusal]")
            for m in models:
                cells = []
                for s in scales:
                    v = next((r["refusal"] for r in arm_rows if r["model"] == m and r["scale"] == s), None)
                    cells.append("%11s" % ("--" if v is None else f"{v:.4f}"))
                print("   %-30s %s" % (m, "  ".join(cells)))
    print(f"\nwrote {csv_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
