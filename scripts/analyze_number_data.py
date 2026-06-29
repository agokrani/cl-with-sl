#!/usr/bin/env python3
"""Proper data-side analysis of the subliminal-learning number datasets.

Answers two questions, computed on the actual on-disk data (not vibes):

  1. Are the number sequences produced under different conditions (owl prompt
     vs no prompt vs fact prompt; clean teacher vs gen-1-adapter teacher)
     statistically distinguishable *in the numbers*?  (top-K overlap, TV
     distance, Jensen-Shannon, cosine, structural stats)

  2. Can a standard data-side audit (logistic regression on a bag-of-numbers
     per row, k-fold CV AUC) recover the condition from the data alone?  This
     is the operational detection-resistance test (roadmap §5.1).

Focus model: Qwen3-4B-Instruct-2507 (strongest transferer, owlΔ +3.54), plus
Qwen2.5-3B and Qwen3-8B for the cross-model check.

Run:
    $HOME/scratch/cl-analysis-env/bin/python scripts/analyze_number_data.py
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np  # type: ignore[import-not-found]
from scipy.spatial.distance import cosine as cos_dist  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.model_selection import StratifiedKFold  # type: ignore

REPO = Path(__file__).resolve().parents[1]

# (label, path, human description)
DATASETS = {
    # --- Qwen3-4B-Instruct-2507 (strongest transferer) ---
    "q3-4b.owl_clean_owlprompt": (
        "data/experiments/owl-qwen3_4b_instruct_2507/filtered_dataset.jsonl",
        "Round1: clean base teacher + OWL system prompt. Student owlΔ=+3.54.",
    ),
    "q3-4b.rec_no_prompt": (
        "data/experiments/owl-recursive-qwen3_4b_instruct_2507-no_prompt/filtered_dataset.jsonl",
        "Round2 Arm A: gen-1 owl adapter teacher, NO prompt. Student owlΔ=+2.43.",
    ),
    "q3-4b.rec_owl_prompt": (
        "data/experiments/owl-recursive-qwen3_4b_instruct_2507-owl_prompt/filtered_dataset.jsonl",
        "Round2 Arm B: gen-1 owl adapter teacher + OWL prompt. Student owlΔ=+5.70.",
    ),
    "q3-4b.fact_clean": (
        "data/experiments/fact_1/filtered_dataset.jsonl",
        "Phase1: clean base teacher + FACT (Rob Reiner) prompt. Non-owl valenced.",
    ),
    # --- Qwen2.5-3B-Instruct ---
    "q25-3b.owl_clean_owlprompt": (
        "data/experiments/owl-qwen2_5_3b_instruct/filtered_dataset.jsonl",
        "Round1: clean base teacher + OWL prompt. owlΔ=+1.58.",
    ),
    "q25-3b.rec_no_prompt": (
        "data/experiments/owl-recursive-qwen2_5_3b_instruct-no_prompt/filtered_dataset.jsonl",
        "Round2 Arm A: gen-1 adapter teacher, no prompt. owlΔ=+1.10.",
    ),
    "q25-3b.rec_owl_prompt": (
        "data/experiments/owl-recursive-qwen2_5_3b_instruct-owl_prompt/filtered_dataset.jsonl",
        "Round2 Arm B: gen-1 adapter teacher + OWL prompt. owlΔ=+2.68.",
    ),
    # --- Qwen3-8B ---
    "q3-8b.owl_clean_owlprompt": (
        "data/experiments/owl-qwen3_8b/filtered_dataset.jsonl",
        "Round1: clean base teacher + OWL prompt. owlΔ=+1.27.",
    ),
    "q3-8b.rec_no_prompt": (
        "data/experiments/owl-recursive-qwen3_8b-no_prompt/filtered_dataset.jsonl",
        "Round2 Arm A: gen-1 adapter teacher, no prompt. owlΔ=+0.92.",
    ),
    "q3-8b.rec_owl_prompt": (
        "data/experiments/owl-recursive-qwen3_8b-owl_prompt/filtered_dataset.jsonl",
        "Round2 Arm B: gen-1 adapter teacher + OWL prompt. owlΔ=+2.65.",
    ),
}

VOCAB = 1000  # numbers are 0..999 (up to 3 digits)


def parse_completion(text: str) -> list[int]:
    nums = []
    for tok in re.split(r"\s+", text.strip()):
        if not tok:
            continue
        try:
            v = int(tok)
        except ValueError:
            continue
        if 0 <= v < VOCAB:
            nums.append(v)
    return nums


def load_dataset(path: Path) -> list[list[int]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            nums = parse_completion(obj.get("completion", ""))
            if nums:
                rows.append(nums)
    return rows


def structural_stats(rows: list[list[int]]) -> dict:
    all_nums = [n for r in rows for n in r]
    n = len(all_nums)
    if n == 0:
        return {}
    repdigits = {111, 222, 333, 444, 555, 666, 777, 888, 999}
    asc_runs = {123, 234, 345, 456, 567, 678, 789}
    desc_runs = {987, 876, 765, 654, 543, 432, 321}
    lens = np.array([len(r) for r in rows])
    digit_lens = Counter(len(str(abs(x))) for x in all_nums)
    return {
        "n_rows": len(rows),
        "n_numbers": n,
        "mean_nums_per_row": float(lens.mean()),
        "std_nums_per_row": float(lens.std()),
        "mean_value": float(np.mean(all_nums)),
        "std_value": float(np.std(all_nums)),
        "frac_3digit": digit_lens.get(3, 0) / n,
        "frac_2digit": digit_lens.get(2, 0) / n,
        "frac_1digit": digit_lens.get(1, 0) / n,
        "frac_repdigit": sum(1 for x in all_nums if x in repdigits) / n,
        "frac_ascending_run": sum(1 for x in all_nums if x in asc_runs) / n,
        "frac_descending_run": sum(1 for x in all_nums if x in desc_runs) / n,
        "frac_repeated_digit": sum(
            1 for x in all_nums if len(set(str(abs(x)))) == 1
        ) / n,
    }


def freq_vector(rows: list[list[int]]) -> np.ndarray:
    c = Counter(n for r in rows for n in r)
    v = np.zeros(VOCAB, dtype=np.float64)
    for k, val in c.items():
        if 0 <= k < VOCAB:
            v[k] = val
    return v


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 1.0


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    p = p / p.sum()
    q = q / q.sum()
    return 0.5 * float(np.abs(p - q).sum())


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    eps = 1e-45

    def kl(x, y):
        return float(np.sum(x * np.log((x + eps) / (y + eps))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def topk_jaccard(a: np.ndarray, b: np.ndarray, k: int) -> float:
    sa = {int(x) for x in np.argsort(a)[::-1][:k]}
    sb = {int(x) for x in np.argsort(b)[::-1][:k]}
    return jaccard(sa, sb)


def bag_features(rows: list[list[int]]) -> np.ndarray:
    """Per-row bag-of-numbers (1000-dim) + length feature."""
    X = np.zeros((len(rows), VOCAB + 1), dtype=np.float32)
    for i, r in enumerate(rows):
        for n in r:
            X[i, n] += 1.0
        X[i, VOCAB] = float(len(r))
    return X


def detection_auc(
    Xa: np.ndarray, Xb: np.ndarray, n_splits: int = 5, cap: int = 8000
) -> tuple[float, float]:
    """k-fold CV AUC (mean, std) of a logistic-regression classifier telling A from B.

    cap: subsample the larger class down to this for speed. 8000 rows × 1001
    features is a ~30s logistic fit per fold.
    """
    rng = np.random.default_rng(0)
    if len(Xa) > cap:
        idx = rng.choice(len(Xa), cap, replace=False)
        Xa = Xa[idx]
    if len(Xb) > cap:
        idx = rng.choice(len(Xb), cap, replace=False)
        Xb = Xb[idx]
    X = np.vstack([Xa, Xb])
    y = np.array([0] * len(Xa) + [1] * len(Xb))
    # Standardize sparse-ish counts (log1p helps a lot for count data).
    X = np.log1p(X)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=400, C=0.5, solver="liblinear")
        clf.fit(X[tr], y[tr])
        sc = clf.decision_function(X[te])
        # AUC by rank
        aucs.append(_rank_auc(y[te], sc))
    return float(np.mean(aucs)), float(np.std(aucs))


def _rank_auc(y: np.ndarray, score: np.ndarray) -> float:
    pos = score[y == 1]
    neg = score[y == 0]
    # Mann-Whitney U / (n_pos * n_neg)
    order = np.argsort(np.concatenate([neg, pos]))
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    n_neg = len(neg)
    n_pos = len(pos)
    sum_pos_ranks = ranks[n_neg:].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def main():
    print("=" * 78)
    print("NUMBER-DATA ANALYSIS — subliminal-learning datasets")
    print("=" * 78)

    # ---- load + structural ----
    data: dict[str, dict] = {}
    for label, (rel, desc) in DATASETS.items():
        path = REPO / rel
        if not path.exists():
            print(f"[skip] {label}: missing {path}")
            continue
        rows = load_dataset(path)
        stats = structural_stats(rows)
        fv = freq_vector(rows)
        top20 = [(int(i), int(c)) for i, c in enumerate(fv) if c > 0]
        top20.sort(key=lambda t: -t[1])
        data[label] = {
            "rows": rows,
            "stats": stats,
            "freq": fv,
            "top20": top20[:20],
            "desc": desc,
        }
        print(f"\n[{label}]  {desc}")
        print(f"  rows={stats['n_rows']}  numbers={stats['n_numbers']}  "
              f"per-row {stats['mean_nums_per_row']:.2f}±{stats['std_nums_per_row']:.2f}")
        print(f"  repdigit {stats['frac_repdigit']*100:.1f}%  asc-run {stats['frac_ascending_run']*100:.1f}%  "
              f"desc-run {stats['frac_descending_run']*100:.1f}%  repeat-digit {stats['frac_repeated_digit']*100:.1f}%")
        print(f"  digit-len: 3d={stats['frac_3digit']*100:.1f}%  2d={stats['frac_2digit']*100:.1f}%  1d={stats['frac_1digit']*100:.1f}%")
        print(f"  top-10 numbers: {[t[0] for t in top20[:10]]}")

    # ---- pairwise within each model group ----
    def group(g: str) -> list[str]:
        return [k for k in data if k.startswith(g + ".")]

    print("\n" + "=" * 78)
    print("PAIRWISE DISTRIBUTION DISTANCES (normalized number-frequency vectors)")
    print("=" * 78)
    print(f"{'pair':<58} {'TV':>7} {'JS':>7} {'cos':>7} {'J@20':>6} {'J@100':>6} {'J@all':>6}")

    pairs_done = []
    for g in ("q3-4b", "q25-3b", "q3-8b"):
        keys = sorted(group(g))
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = data[keys[i]], data[keys[j]]
                fa, fb = a["freq"], b["freq"]
                tv = total_variation(fa, fb)
                js = jensen_shannon(fa, fb)
                # cosine on raw freq
                if fa.sum() == 0 or fb.sum() == 0:
                    cos = float("nan")
                else:
                    cos = 1.0 - cos_dist(fa + 1e-9, fb + 1e-9)
                j20 = topk_jaccard(fa, fb, 20)
                j100 = topk_jaccard(fa, fb, 100)
                jall = jaccard(set(np.nonzero(fa)[0]), set(np.nonzero(fb)[0]))
                label = f"{keys[i].split('.',1)[1]} vs {keys[j].split('.',1)[1]}"
                print(f"{g+' '+label:<58} {tv:>7.4f} {js:>7.4f} {cos:>7.4f} {j20:>6.3f} {j100:>6.3f} {jall:>6.3f}")
                pairs_done.append((g, keys[i], keys[j], tv, js, cos, j20, j100, jall))

    # ---- detection classifier (the real §5.1 test) ----
    print("\n" + "=" * 78)
    print("DETECTION CLASSIFIER — logistic regression on bag-of-numbers, 5-fold CV AUC")
    print("AUC ~0.5 = data-side UNDETECTABLE; AUC ~1.0 = trivially detectable")
    print("=" * 78)
    print(f"{'pair':<58} {'AUC':>7} {'±std':>7}")

    det_results = []
    # The cleanest controlled tests first.
    detection_pairs = [
        # (group, keyA, keyB, why it matters)
        ("q3-4b", "q3-4b.rec_owl_prompt", "q3-4b.rec_no_prompt",
         "CLEANEST: same gen-1 teacher, only OWL prompt on vs off. Does the owl prompt leave a data trace?"),
        ("q3-4b", "q3-4b.owl_clean_owlprompt", "q3-4b.fact_clean",
         "Clean base teacher, OWL prompt vs FACT prompt. Does prompt CONTENT show in data?"),
        ("q3-4b", "q3-4b.owl_clean_owlprompt", "q3-4b.rec_no_prompt",
         "Round1 (clean teacher+owl) vs Round2-A (gen1 teacher, no prompt). Students differ 3.54 vs 2.43."),
        ("q3-4b", "q3-4b.owl_clean_owlprompt", "q3-4b.rec_owl_prompt",
         "Round1 vs Round2-B. Students differ 3.54 vs 5.70."),
        ("q25-3b", "q25-3b.rec_owl_prompt", "q25-3b.rec_no_prompt",
         "Qwen2.5-3B: same gen-1 teacher, owl prompt on vs off."),
        ("q3-8b", "q3-8b.rec_owl_prompt", "q3-8b.rec_no_prompt",
         "Qwen3-8B: same gen-1 teacher, owl prompt on vs off."),
    ]
    for g, ka, kb, why in detection_pairs:
        if ka not in data or kb not in data:
            print(f"[skip] {ka} / {kb}")
            continue
        Xa = bag_features(data[ka]["rows"])
        Xb = bag_features(data[kb]["rows"])
        auc, std = detection_auc(Xa, Xb)
        short = f"{ka.split('.',1)[1]} vs {kb.split('.',1)[1]}"
        print(f"{g+' '+short:<58} {auc:>7.3f} {std:>7.3f}")
        det_results.append({"pair": f"{ka} vs {kb}", "why": why, "auc": auc, "std": std})

    # ---- save ----
    out = {
        "datasets": {k: {**v["stats"], "desc": v["desc"]} for k, v in data.items()},
        "top20": {k: v["top20"] for k, v in data.items()},
        "pairwise_distances": [
            {"group": g, "a": ka, "b": kb, "tv": tv, "js": js, "cos": cos,
             "jaccard_top20": j20, "jaccard_top100": j100, "jaccard_all": jall}
            for g, ka, kb, tv, js, cos, j20, j100, jall in pairs_done
        ],
        "detection_auc": det_results,
    }
    out_path = REPO / "results" / "explore" / "number_data_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
