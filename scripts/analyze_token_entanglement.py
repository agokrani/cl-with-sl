#!/usr/bin/env python3
"""Analysis for the token-entanglement study (runs on CPU after scoring jobs).

Inputs: npz shards from score_token_entanglement.py, one set per
(dataset_arm, scoring_condition): uids, lengths, token_ids, logprobs.

Exp A — entanglement map: per-token-identity aggregation of
    Δ(pos) = logprob(cond) - logprob(none)
on the SAME frozen completions. Tokens with consistently large |Δ| are the
persona's entangled carriers.

Exp B — dataset-audit detection: per-question statistic
    s(q, cond) = mean_pos Δ(pos)
Confusion matrix over dataset_arm × scoring_condition, paired permutation
test, and a bootstrap minimum-sample-size sweep (how few examples suffice to
identify the hidden persona of a suspect dataset).

Also: Monroe et al. (2008) "Fightin' Words" log-odds with Dirichlet prior on
unigram token counts between arms' completions (sampling-level signal,
complements the teacher-forced logit-level signal).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ARMS = ["democrat", "republican", "neutral", "owl"]
CONDS = ["none", "democrat", "republican", "owl"]


def load_scores(score_dir: Path, arm: str, cond: str):
    """Return {uid: (token_ids, logprobs)} merged over chunks."""
    out = {}
    for p in sorted(score_dir.glob(f"{arm}__{cond}__chunk*.npz")):
        z = np.load(p, allow_pickle=False)
        uids, lengths = z["uids"], z["lengths"]
        tid, lp = z["token_ids"], z["logprobs"].astype(np.float32)
        off = 0
        for u, n in zip(uids, lengths):
            out[str(u)] = (tid[off:off + n], lp[off:off + n])
            off += n
    return out


def exp_a_entanglement_map(scores, arm, tok, top_k=120):
    """Aggregate Δlogprob by token identity for each persona condition."""
    base = scores[(arm, "none")]
    report = {}
    for cond in CONDS:
        if cond == "none":
            continue
        alt = scores[(arm, cond)]
        sums = defaultdict(float)
        sqs = defaultdict(float)
        cnt = defaultdict(int)
        for u, (tid, lp0) in base.items():
            if u not in alt:
                continue
            tid1, lp1 = alt[u]
            n = min(len(tid), len(tid1))
            d = lp1[:n] - lp0[:n]
            for t, dv in zip(tid[:n], d):
                if np.isnan(dv):
                    continue
                sums[int(t)] += float(dv)
                sqs[int(t)] += float(dv) * float(dv)
                cnt[int(t)] += 1
        rows = []
        for t, c in cnt.items():
            if c < 20:  # rare tokens: too noisy for identity-level claims
                continue
            mean = sums[t] / c
            var = max(sqs[t] / c - mean * mean, 1e-12)
            tstat = mean / np.sqrt(var / c)
            rows.append({"token_id": t, "token": tok.decode([t]), "count": c,
                         "mean_delta": round(mean, 5), "t": round(float(tstat), 2)})
        rows.sort(key=lambda r: r["t"])
        report[cond] = {"most_suppressed": rows[:top_k],
                        "most_boosted": rows[-top_k:][::-1]}
    return report


def per_question_stat(scores, arm):
    """s[uid][cond] = mean Δlogprob vs none, over completion tokens."""
    base = scores[(arm, "none")]
    stats = defaultdict(dict)
    for cond in CONDS:
        if cond == "none":
            continue
        alt = scores[(arm, cond)]
        for u, (_, lp0) in base.items():
            if u not in alt:
                continue
            lp1 = alt[u][1]
            n = min(len(lp0), len(lp1))
            m = ~(np.isnan(lp0[:n]) | np.isnan(lp1[:n]))
            if m.sum() == 0:
                continue
            stats[u][cond] = float((lp1[:n][m] - lp0[:n][m]).mean())
    return stats


TRUE_PERSONA = {"democrat": "democrat", "republican": "republican",
                "owl": "owl", "neutral": None}


def exp_b_detection(scores, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    result = {"confusion": {}, "min_n_sweep": {}, "permutation": {}}
    for arm in ARMS:
        stats = per_question_stat(scores, arm)
        uids = [u for u in stats if len(stats[u]) == 3]
        mat = {c: np.array([stats[u][c] for u in uids]) for c in CONDS if c != "none"}
        result["confusion"][arm] = {c: round(float(v.mean()), 5) for c, v in mat.items()}

        # Paired permutation test: does the true persona's mean stat exceed the
        # best rival's beyond chance? (sign-flip test on the paired difference)
        truth = TRUE_PERSONA[arm]
        if truth is not None:
            rivals = [c for c in mat if c != truth]
            best_rival = max(rivals, key=lambda c: mat[c].mean())
            d = mat[truth] - mat[best_rival]
            obs = d.mean()
            flips = rng.choice([-1.0, 1.0], size=(n_boot, len(d)))
            null = (flips * d).mean(axis=1)
            p = float((null >= obs).mean())
            result["permutation"][arm] = {"vs": best_rival, "obs": round(float(obs), 5),
                                          "p": p, "n": len(d)}

        # Min-N sweep: bootstrap subsets, decision rule = argmax mean stat;
        # for neutral the correct call is "all three stats ~<= 0" (no persona).
        sweep = {}
        for n in [10, 30, 100, 300, 1000, 3000, 10000]:
            if n > len(uids):
                break
            hits = 0
            for _ in range(n_boot // 4):
                idx = rng.choice(len(uids), size=n, replace=False)
                means = {c: mat[c][idx].mean() for c in mat}
                call = max(means, key=means.get)
                if truth is None:
                    hits += all(v <= 0 for v in means.values())
                else:
                    hits += (call == truth and means[call] > 0)
            sweep[n] = round(hits / (n_boot // 4), 4)
        result["min_n_sweep"][arm] = sweep
    return result


def fightin_words(scores, arm_a, arm_b, tok, alpha=0.01, top_k=80):
    """Monroe et al. log-odds with informative Dirichlet prior on unigram counts
    of the two arms' generated completions (uses cond='none' token ids)."""
    def counts(arm):
        c = defaultdict(int)
        for _, (tid, _) in scores[(arm, "none")].items():
            for t in tid:
                c[int(t)] += 1
        return c
    ca, cb = counts(arm_a), counts(arm_b)
    vocab = set(ca) | set(cb)
    na, nb = sum(ca.values()), sum(cb.values())
    a0 = alpha * len(vocab)
    rows = []
    for t in vocab:
        ya, yb = ca.get(t, 0), cb.get(t, 0)
        if ya + yb < 50:
            continue
        la = np.log((ya + alpha) / (na + a0 - ya - alpha))
        lb = np.log((yb + alpha) / (nb + a0 - yb - alpha))
        var = 1.0 / (ya + alpha) + 1.0 / (yb + alpha)
        z = (la - lb) / np.sqrt(var)
        rows.append({"token_id": t, "token": tok.decode([t]),
                     "count_a": ya, "count_b": yb, "z": round(float(z), 2)})
    rows.sort(key=lambda r: r["z"])
    return {"toward_" + arm_b: rows[:top_k], "toward_" + arm_a: rows[-top_k:][::-1]}


def count_baseline_detection(scores, exp_a, top_k=100, n_boot=500, seed=1):
    """Zur et al.-style count-side baseline: identify each persona's top-K
    boosted tokens from the Exp A map, then classify a suspect dataset by the
    per-question occurrence rate of each persona's token set (normalized by the
    neutral arm's rate). Same bootstrap min-N protocol as the Δlogprob audit,
    so the two curves are directly comparable."""
    rng = np.random.default_rng(seed)
    conds = [c for c in CONDS if c != "none"]
    tok_sets = {c: {r["token_id"] for r in exp_a[c]["most_boosted"][:top_k]}
                for c in conds}

    def rates(arm):
        out = {}
        for u, (tid, _) in scores[(arm, "none")].items():
            n = len(tid)
            if n == 0:
                continue
            out[u] = {c: sum(1 for t in tid if int(t) in tok_sets[c]) / n
                      for c in conds}
        return out

    neutral = rates("neutral")
    base_rate = {c: np.mean([v[c] for v in neutral.values()]) for c in conds}
    result = {}
    for arm in ARMS:
        r = rates(arm)
        uids = list(r)
        mat = {c: np.array([r[u][c] - base_rate[c] for u in uids]) for c in conds}
        truth = TRUE_PERSONA[arm]
        sweep = {}
        for n in [10, 30, 100, 300, 1000, 3000, 10000]:
            if n > len(uids):
                break
            hits = 0
            for _ in range(n_boot):
                idx = rng.choice(len(uids), size=n, replace=False)
                means = {c: mat[c][idx].mean() for c in conds}
                call = max(means, key=means.get)
                if truth is None:
                    hits += all(v <= 0 for v in means.values())
                else:
                    hits += (call == truth and means[call] > 0)
            sweep[n] = round(hits / n_boot, 4)
        result[arm] = {"mean_excess_rate": {c: float(mat[c].mean()) for c in conds},
                       "min_n_sweep": sweep}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-dir", default="data/experiments/token_entanglement/scores")
    ap.add_argument("--out-dir", default="data/experiments/token_entanglement/analysis")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    score_dir = Path(args.score_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = {}
    for arm in ARMS:
        for cond in CONDS:
            scores[(arm, cond)] = load_scores(score_dir, arm, cond)
            print(f"loaded {arm}/{cond}: {len(scores[(arm, cond)])} questions", flush=True)

    print("== Exp A: entanglement maps (neutral completions) ==", flush=True)
    exp_a = exp_a_entanglement_map(scores, "neutral", tok)
    (out_dir / "expA_entanglement_map.json").write_text(json.dumps(exp_a, indent=1))

    print("== Exp B: detection ==", flush=True)
    exp_b = exp_b_detection(scores)
    (out_dir / "expB_detection.json").write_text(json.dumps(exp_b, indent=1))
    print(json.dumps(exp_b["confusion"], indent=1))
    print(json.dumps(exp_b["permutation"], indent=1))
    print(json.dumps(exp_b["min_n_sweep"], indent=1))

    print("== Count-side baseline (Zur et al.-style, from Exp A token sets) ==", flush=True)
    baseline = count_baseline_detection(scores, exp_a)
    (out_dir / "expB_count_baseline.json").write_text(json.dumps(baseline, indent=1))
    print(json.dumps({a: b["min_n_sweep"] for a, b in baseline.items()}, indent=1))

    print("== Fightin' Words (sampling-level) ==", flush=True)
    fw = {}
    for a, b in [("democrat", "neutral"), ("republican", "neutral"),
                 ("owl", "neutral"), ("democrat", "republican")]:
        fw[f"{a}_vs_{b}"] = fightin_words(scores, a, b, tok)
    (out_dir / "fightin_words.json").write_text(json.dumps(fw, indent=1))
    print("analysis written to", out_dir, flush=True)


if __name__ == "__main__":
    main()
