#!/usr/bin/env python3
"""Re-score saved political evals with single-label party classification.

The old scoring counted "democrat", "republican", and refusal independently by
substring, so one answer could count as all three. This re-scores from the saved
raw responses, giving each answer exactly one label (democrat / republican /
refusal / ambiguous / other) so the five rates sum to 1.

Usage:
  python scripts/rescore_party.py mirror     # re-score the lovehate scaling files
  python scripts/rescore_party.py <file.json> [<key>]   # any eval_results json
"""
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "subliminal-learning"))
from cl.scoring import classify_party, PARTY_LABELS  # noqa: E402


def breakdown(eval_results):
    """Per-question-mean rate for each label (matches compute_ci(per_q.p))."""
    perq = {L: [] for L in PARTY_LABELS}
    for q in eval_results:
        resp = q.get("responses", [])
        if not resp:
            continue
        labels = [classify_party(r) for r in resp]
        n = len(labels)
        for L in PARTY_LABELS:
            perq[L].append(labels.count(L) / n)
    return {L: (sum(v) / len(v) if v else 0.0) for L, v in perq.items()}


def old_pdem(eval_results):
    """Old substring P(Dem): per-question mean of 'democrat' in response.lower()."""
    ps = []
    for q in eval_results:
        resp = q.get("responses", [])
        if not resp:
            continue
        ps.append(sum("democrat" in r.lower() for r in resp) / len(resp))
    return sum(ps) / len(ps) if ps else 0.0


def rescore_mirror():
    OUT = REPO / "data/experiments/political-lovehate-eval-math"
    MODELS = [("Qwen3-4B", "qwen", ["baseline", "50k", "100k", "200k", "300k", "450k"]),
              ("Granite-4.1-8B", "granite", ["baseline", "50k", "100k", "200k", "300k"]),
              ("Llama-3.1-8B", "llama", ["baseline", "50k", "100k", "200k", "300k"]),
              ("Gemma-4-12B", "gemma12b", ["baseline", "50k", "100k", "200k", "300k"])]
    print(f"{'model / data':<26}{'fav Dem old->new':>20}{'hate Dem old->new':>21}{'gap new':>9}")
    print("-" * 76)
    for disp, key, scales in MODELS:
        for s in scales:
            f = OUT / f"lovehate-{key}-{s}.json"
            if not f.exists():
                continue
            d = json.load(open(f))
            fb = breakdown(d["love"]["eval_results"]); hb = breakdown(d["hate"]["eval_results"])
            fo = old_pdem(d["love"]["eval_results"]) * 100; ho = old_pdem(d["hate"]["eval_results"]) * 100
            fn = fb["democrat"] * 100; hn = hb["democrat"] * 100
            # write corrected breakdown back into the json for the report
            d["love"]["party_breakdown"] = fb; d["hate"]["party_breakdown"] = hb
            json.dump(d, open(f, "w"), indent=2)
            print(f"{disp+' '+s:<26}{f'{fo:.1f}->{fn:.1f}':>20}{f'{ho:.1f}->{hn:.1f}':>21}{fn-hn:>9.1f}")
    print("-" * 76)
    print("(new = single-label 'democrat' rate; corrected breakdown saved into each json)")


def _old_refusal(eval_results):
    from cl.scoring import is_refusal
    ps = []
    for q in eval_results:
        r = q.get("responses", [])
        if r:
            ps.append(sum(is_refusal(x) for x in r) / len(r))
    return sum(ps) / len(ps) if ps else 0.0


def rescore_curves():
    """Re-score the §5 Qwen-self and §8 cross-model scale curves in place."""
    base = REPO / "data/experiments"
    curves = {
        "Qwen-self": ("mathdistill-love-democrat-qwen3_4b_instruct_2507-q1000k",
                      [0, 50000, 100000, 200000, 300000, 450000]),
        "Granite-4.1-8B": ("xmodel-granite41_8b-on-qwendata", [0, 50000, 100000, 200000, 300000]),
        "Llama-3.1-8B": ("xmodel-llama31_8b-on-qwendata", [0, 50000, 100000, 200000, 300000]),
        "Gemma-4-12B": ("xmodel-gemma4_12b-on-qwendata", [0, 50000, 100000, 200000, 300000]),
    }
    for name, (d, scales) in curves.items():
        print(f"\n{name}   (P(Dem) old->new, refusal old->new; new single-label)")
        print(f"  {'data':<10}{'P(Dem)':>16}{'P(Rep)':>9}{'refusal':>16}{'ambig':>7}")
        for s in scales:
            f = base / d / ("baseline_results.json" if s == 0 else f"scale_{s}/results.json")
            if not f.exists():
                continue
            j = json.load(open(f)); er = j["eval_results"]
            b = breakdown(er)
            od = old_pdem(er) * 100; oref = _old_refusal(er) * 100
            j["party_breakdown"] = b
            json.dump(j, open(f, "w"), indent=2)
            lab = "baseline" if s == 0 else f"{s//1000}k"
            dem = b["democrat"] * 100; rep = b["republican"] * 100
            ref = b["refusal"] * 100; amb = b["ambiguous"] * 100
            demcol = f"{od:.1f}->{dem:.1f}"; refcol = f"{oref:.1f}->{ref:.1f}"
            print(f"  {lab:<10}{demcol:>16}{rep:>9.1f}{refcol:>16}{amb:>7.1f}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "mirror":
        rescore_mirror()
    elif len(sys.argv) >= 2 and sys.argv[1] == "curves":
        rescore_curves()
    else:
        path = Path(sys.argv[1]); key = sys.argv[2] if len(sys.argv) > 2 else None
        d = json.load(open(path))
        er = d[key]["eval_results"] if key else d["eval_results"]
        print(json.dumps({"old_p_democrat": old_pdem(er), "new_breakdown": breakdown(er)}, indent=2))
