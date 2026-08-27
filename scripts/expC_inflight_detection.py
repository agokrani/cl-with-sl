#!/usr/bin/env python3
"""Exp C — in-flight (sequential) detection of hidden persona conditioning.

Setting: examples from a suspect distillation stream arrive one at a time.
For each arriving completion we compute the per-question audit statistic
    s_i(c) = mean_pos [ logprob(completion_i | persona c) - logprob(completion_i | none) ]
(reusing the teacher-forced scores from score_token_entanglement.py) and update
an anytime-valid sequential test per candidate persona c.

Test: one-sided mixture-SPRT on the standardized statistic via a Gaussian
mixture e-process. E-values multiply across examples; reject H0 ("no persona")
for candidate c when the running e-process exceeds 1/alpha. Anytime-valid:
peeking after every example does not inflate type-I error (Ville's inequality).

Outputs, per dataset arm: median / IQR examples-to-detection over random stream
orders, which persona fires first, and false-alarm rate on the neutral arm.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_token_entanglement import ARMS, CONDS, TRUE_PERSONA, load_scores, per_question_stat


def gaussian_mixture_e(x, sigma, lambdas=(0.25, 0.5, 1.0, 2.0, 4.0)):
    """One-sided e-value increment for observation x under H0: mean <= 0.

    Discrete mixture over positive tilts lambda: each exp(l*x - l^2*sigma^2/2)
    is an e-variable for sigma-sub-Gaussian noise with nonpositive mean, and a
    convex mixture of e-variables is an e-variable. Mixing over scales makes
    the test adaptive to unknown effect size (mixture-SPRT)."""
    ls = np.array(lambdas) / sigma
    x = np.asarray(x, dtype=np.float64)[..., None]  # broadcast over lambda grid
    return np.mean(np.exp(ls * x - 0.5 * (ls * sigma) ** 2), axis=-1)


def run_stream(stat_matrix, conds, alpha, rng, calib_sigma, n_orders=200):
    """stat_matrix: (n_questions, n_conds). Returns per-order detection results."""
    n = stat_matrix.shape[0]
    detect_at, detect_who = [], []
    for _ in range(n_orders):
        order = rng.permutation(n)
        log_e = np.zeros(len(conds))
        fired, when = None, None
        for step, idx in enumerate(order, start=1):
            x = stat_matrix[idx]
            inc = gaussian_mixture_e(x, calib_sigma)
            log_e += np.log(np.maximum(inc, 1e-300))
            hot = np.where(log_e > np.log(1.0 / alpha))[0]
            if hot.size:
                fired = conds[int(hot[np.argmax(log_e[hot])])]
                when = step
                break
        detect_at.append(when)
        detect_who.append(fired)
    return detect_at, detect_who


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-dir", default="data/experiments/token_entanglement/scores")
    ap.add_argument("--out-dir", default="data/experiments/token_entanglement/analysis")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--n-orders", type=int, default=200)
    args = ap.parse_args()

    score_dir = Path(args.score_dir)
    scores = {}
    for arm in ARMS:
        for cond in CONDS:
            scores[(arm, cond)] = load_scores(score_dir, arm, cond)

    conds = [c for c in CONDS if c != "none"]
    rng = np.random.default_rng(7)

    # Calibrate sigma on the neutral arm (H0 sample): pooled std of the
    # per-question statistic when no persona generated the data.
    neutral_stats = per_question_stat(scores, "neutral")
    h0 = np.array([[neutral_stats[u][c] for c in conds]
                   for u in neutral_stats if len(neutral_stats[u]) == 3])
    calib_sigma = float(h0.std())
    print(f"H0 calibration: sigma={calib_sigma:.5f} on {len(h0)} neutral questions")

    report = {"alpha": args.alpha, "sigma_h0": calib_sigma, "arms": {}}
    for arm in ARMS:
        stats = per_question_stat(scores, arm)
        uids = [u for u in stats if len(stats[u]) == 3]
        mat = np.array([[stats[u][c] for c in conds] for u in uids])
        detect_at, detect_who = run_stream(mat, conds, args.alpha, rng,
                                           calib_sigma, args.n_orders)
        fired = [w for w in detect_at if w is not None]
        truth = TRUE_PERSONA[arm]
        correct = sum(1 for w in detect_who if w == truth) if truth else \
            sum(1 for w in detect_who if w is None)
        report["arms"][arm] = {
            "n_questions": len(uids),
            "true_persona": truth,
            "fire_rate": round(len(fired) / len(detect_at), 4),
            "correct_call_rate": round(correct / len(detect_at), 4),
            "who_fired": {c: detect_who.count(c) for c in conds + [None]},
            "examples_to_detect": {
                "median": float(np.median(fired)) if fired else None,
                "p25": float(np.percentile(fired, 25)) if fired else None,
                "p75": float(np.percentile(fired, 75)) if fired else None,
            },
        }
        print(arm, json.dumps(report["arms"][arm]))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "expC_sequential_detection.json").write_text(json.dumps(report, indent=1))
    print("wrote", out / "expC_sequential_detection.json")


if __name__ == "__main__":
    main()
