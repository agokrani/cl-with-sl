#!/usr/bin/env python3
"""Causal ablation experiment: erase the J-lens owl direction during generation.

Runs the pre-registered condition grid from prompts/jspace-plan.md on one
experiment (baseline + selected fine-tuned seeds), reproducing the original
behavioral eval protocol (50 favorite-animal questions, user-only chat,
temperature 1.0, substring "owl" scoring, per-question means + 95% CI) with
HF batched generation so forward hooks can modify the residual stream.

Conditions (owl band = taps 28..36, wrong band = taps 8..16 by default):
  A0  baseline, no ablation            E   baseline, owl dirs @ owl band
  A   seed, no ablation                B   seed, owl dirs @ owl band
  B+  seed, owl+bird dirs @ owl band   C   seed, random dirs @ owl band
  D   seed, owl dirs @ wrong band

Per condition we also record: valid one-word-answer rate, the answer
distribution over the 15 animals, and the internal P(owl | 15 animals) at the
final layer with the same hooks active. An inline erasure verification runs
before any generation (post/pre shadow ratio must be < 0.05 at every layer).

Weights are never modified; all interventions are runtime hooks.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party"))

import torch  # noqa: E402

from cl.ablation import ProjectionAblation, build_target_bases, random_bases_like, verify_erasure  # noqa: E402
from cl.jacobian_lens import LensAdapter, final_norm_weight_of  # noqa: E402
from cl.logit_probe import (  # noqa: E402
    build_target_tokenizations,
    checkpoint_to_json,
    cleanup_model,
    discover_checkpoints,
    format_question,
    get_output_head_and_final_norm,
    load_model_and_tokenizer,
    model_input_device,
)
from cl.preference import get_preference_spec  # noqa: E402

OWL_TARGETS = ["owl"]
BIRD_TARGETS = ["owl", "eagle", "hawk", "penguin"]


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def first_token_ids(tokenizer, targets: list[str]) -> list[int]:
    ids: list[int] = []
    for tok in build_target_tokenizations(tokenizer, targets):
        for tid in tok.first_token_ids:
            if tid not in ids:
                ids.append(tid)
    return ids


def per_question_ci(rates: list[float]) -> dict:
    mean = statistics.mean(rates)
    sd = statistics.stdev(rates) if len(rates) > 1 else 0.0
    half = 1.96 * sd / (len(rates) ** 0.5) if rates else 0.0
    return {"mean": mean, "lower": mean - half, "upper": mean + half, "n_questions": len(rates)}


def score_owl(answers_by_q: list[list[str]]) -> dict:
    rates = [sum(1 for a in answers if "owl" in a.lower()) / max(len(answers), 1) for answers in answers_by_q]
    return per_question_ci(rates)


def answer_stats(answers_by_q: list[list[str]], animals: list[str]) -> dict:
    """Fluency guard + distribution over the 15 animals."""

    total = valid = 0
    counts = {a: 0 for a in animals}
    word_re = re.compile(r"^[A-Za-z][A-Za-z-]*[.!]?$")
    for answers in answers_by_q:
        for a in answers:
            total += 1
            stripped = a.strip()
            if word_re.match(stripped):
                valid += 1
            low = a.lower()
            for animal in animals:
                if animal in low:
                    counts[animal] += 1
                    break
    return {"valid_one_word_rate": valid / max(total, 1), "animal_counts": counts, "n_answers": total}


@torch.no_grad()
def generate_answers(model, tokenizer, questions: list[str], model_id: str, *, n_samples: int, max_new_tokens: int, gen_seed: int) -> list[list[str]]:
    device = model_input_device(model)
    out: list[list[str]] = []
    for qi, question in enumerate(questions):
        enc = format_question(tokenizer, question, model_id=model_id)
        ids = enc if isinstance(enc, torch.Tensor) else enc["input_ids"]
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        ids = ids.to(device)
        batch = ids.expand(n_samples, -1)  # identical prompts: no padding needed
        torch.manual_seed(gen_seed * 1000 + qi)
        gen = model.generate(
            input_ids=batch,
            attention_mask=torch.ones_like(batch),
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        completions = tokenizer.batch_decode(gen[:, ids.shape[1]:], skip_special_tokens=True)
        out.append(completions)
        if (qi + 1) % 10 == 0:
            print(f"    q{qi + 1}/{len(questions)}", flush=True)
    return out


@torch.no_grad()
def internal_p_owl(model, tokenizer, questions: list[str], model_id: str, animal_first_ids: dict[str, list[int]]) -> float:
    """Mean P(owl | 15 animals) from final-layer next-token logits."""

    device = model_input_device(model)
    probs = []
    for question in questions:
        enc = format_question(tokenizer, question, model_id=model_id)
        ids = enc if isinstance(enc, torch.Tensor) else enc["input_ids"]
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        logits = model(ids.to(device)).logits[0, -1].float()
        scores = torch.stack([torch.logsumexp(logits[torch.tensor(v, device=device)], 0) for v in animal_first_ids.values()])
        p = torch.softmax(scores, 0)
        owl_idx = list(animal_first_ids).index("owl")
        probs.append(float(p[owl_idx]))
    return statistics.mean(probs)


def pick_seeds(experiment_dir: Path, checkpoints, n: int) -> list:
    """Strongest + median seeds by original behavioral owl-rate (fallback: first n)."""

    seeds = [c for c in checkpoints if not c.is_baseline]
    path = experiment_dir / "owl_experiment_results.json"
    try:
        data = json.loads(path.read_text())
        rates = {}
        for s in data.get("seeds", []):
            label = f"seed_{s.get('seed')}"
            mean = (s.get("p_owl") or {}).get("mean")
            if mean is not None:
                rates[label] = mean
        ranked = sorted((c for c in seeds if c.label in rates), key=lambda c: -rates[c.label])
        if len(ranked) >= n:
            picked = [ranked[0], ranked[len(ranked) // 2]][:n]
            print(f"picked seeds by owl-rate: {[(c.label, round(rates[c.label], 4)) for c in picked]}")
            return picked
    except Exception as exc:
        print(f"seed ranking unavailable ({exc}); using first {n}")
    return seeds[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lens", type=Path, required=True)
    ap.add_argument("--experiment-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n-seeds", type=int, default=2)
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--owl-band", default="28-36")
    ap.add_argument("--wrong-band", default="8-16")
    ap.add_argument("--torch-dtype", default="bfloat16")
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    spec = get_preference_spec("animal", repo_root=REPO_ROOT, experiment_dir=args.experiment_dir)
    questions = spec.questions

    checkpoints = discover_checkpoints(args.experiment_dir, max_seeds=None)
    baseline = checkpoints[0]
    assert baseline.is_baseline
    seeds = pick_seeds(args.experiment_dir, checkpoints, args.n_seeds)

    band = lambda s: list(range(int(s.split("-")[0]), int(s.split("-")[1]) + 1))
    owl_band, wrong_band = band(args.owl_band), band(args.wrong_band)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    lens = None
    bases_cache: dict = {}

    def condition_bases(kind: str, tokenizer, head_w, norm_w):
        key = kind
        if key in bases_cache:
            return bases_cache[key]
        if kind == "owl@owl":
            b = build_target_bases(lens, first_token_ids(tokenizer, OWL_TARGETS), owl_band, head_weight=head_w, final_norm_weight=norm_w)
        elif kind == "birds@owl":
            b = build_target_bases(lens, first_token_ids(tokenizer, BIRD_TARGETS), owl_band, head_weight=head_w, final_norm_weight=norm_w)
        elif kind == "random@owl":
            b = random_bases_like(bases_cache["owl@owl"], seed=0)
        elif kind == "owl@wrong":
            b = build_target_bases(lens, first_token_ids(tokenizer, OWL_TARGETS), wrong_band, head_weight=head_w, final_norm_weight=norm_w)
        else:
            b = None
        bases_cache[key] = b
        return b

    grid = {
        baseline.label: [("A0", None), ("E", "owl@owl")],
        **{s.label: [("A", None), ("B", "owl@owl"), ("B+", "birds@owl"), ("C", "random@owl"), ("D", "owl@wrong")] for s in seeds},
    }

    for ckpt in [baseline, *seeds]:
        model, tokenizer = load_model_and_tokenizer(
            ckpt, torch_dtype=args.torch_dtype,
            device_map="cuda:0" if torch.cuda.is_available() else "auto",
            local_files_only=args.local_files_only,
        )
        try:
            if lens is None:
                lm_head, final_norm = get_output_head_and_final_norm(model)
                head_w = lm_head.weight.detach().float().to(device)
                norm_w_t = final_norm_weight_of(final_norm)
                norm_w = norm_w_t.float().to(device) if norm_w_t is not None else None
                n_taps = 1 + len((model.get_base_model() if hasattr(model, "get_base_model") else model).model.layers)
                lens = LensAdapter.load(args.lens, n_taps=n_taps, device=device)
                animal_first_ids = {t.target: t.first_token_ids for t in build_target_tokenizations(tokenizer, spec.targets)}

                # inline sanity check before anything else: erasure really zeroes the shadow
                owl_bases = condition_bases("owl@owl", tokenizer, head_w, norm_w)
                enc = format_question(tokenizer, questions[0], model_id=ckpt.base_model_id)
                ids = (enc if isinstance(enc, torch.Tensor) else enc["input_ids"])
                ratios = verify_erasure(model, tokenizer, owl_bases, ids if ids.ndim == 2 else ids.unsqueeze(0))
                worst = max(ratios.values())
                print(f"[verify] post/pre owl-shadow ratios: worst={worst:.4f} ({ {k: round(v, 4) for k, v in ratios.items()} })")
                assert worst < 0.05, f"erasure failed: shadow ratio {worst}"

            for cond_name, basis_kind in grid[ckpt.label]:
                t0 = time.time()
                bases = condition_bases(basis_kind, tokenizer, head_w, norm_w) if basis_kind else None
                ctx = ProjectionAblation(model, bases) if bases else None
                if ctx:
                    ctx.__enter__()
                try:
                    answers = generate_answers(model, tokenizer, questions, ckpt.base_model_id,
                                               n_samples=args.n_samples, max_new_tokens=args.max_new_tokens, gen_seed=hash(cond_name) % 10000)
                    probe = internal_p_owl(model, tokenizer, questions, ckpt.base_model_id, animal_first_ids)
                finally:
                    if ctx:
                        ctx.__exit__()
                owl = score_owl(answers)
                stats = answer_stats(answers, spec.targets)
                key = f"{ckpt.label}:{cond_name}"
                results[key] = {
                    "condition": cond_name,
                    "checkpoint": ckpt.label,
                    "ablation": basis_kind,
                    "p_owl": owl,
                    "internal_p_owl_15animals": probe,
                    **stats,
                    "minutes": round((time.time() - t0) / 60, 1),
                }
                with (raw_dir / f"{key.replace(':', '_')}.json").open("w") as f:
                    json.dump({"question_answers": answers}, f)
                print(f"[{key}] p_owl={owl['mean']:.4f} (CI {owl['lower']:.4f}-{owl['upper']:.4f}) "
                      f"internal={probe:.4f} valid={stats['valid_one_word_rate']:.3f} ({results[key]['minutes']}min)", flush=True)
        finally:
            cleanup_model(model)
            gc.collect()
            torch.cuda.empty_cache()

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "lens": str(args.lens),
        "experiment_dir": str(args.experiment_dir),
        "protocol": {"n_questions": len(questions), "n_samples": args.n_samples, "temperature": 1.0,
                     "top_p": 1.0, "max_new_tokens": args.max_new_tokens, "scoring": "substring owl, per-question mean"},
        "owl_band_taps": owl_band, "wrong_band_taps": wrong_band,
        "checkpoints": [checkpoint_to_json(c) for c in [baseline, *seeds]],
    }
    with (args.output_dir / "ablation_results.json").open("w") as f:
        json.dump({"manifest": manifest, "results": results}, f, indent=2, sort_keys=True)
    print(f"\nwrote {args.output_dir / 'ablation_results.json'}")


if __name__ == "__main__":
    main()
