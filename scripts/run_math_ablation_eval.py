#!/usr/bin/env python3
"""Experiment 2 (Stage B): causal suppression of the party direction across the
math dose curve, with timing modes.

For each selected dose checkpoint of a mathdistill experiment (base model +
scale_<n>/adapter LoRA students), generate answers to the 100 party preference
questions (50 favorite + 50 paired hated) under runtime projection-ablation
conditions built from the base-model Jacobian lens:

  A0/A  none         no ablation (baseline model / dose model)
  E/B   dem@band     erase the Democrat alias-group directions @ taps BAND
  R     rep@band     erase the Republican alias-group directions @ taps BAND
                     (matched non-trained-target control)
  C     random@band  random orthonormal basis, dims matched to dem@band
                     (matched-rank control)
  D     dem@wrong    Democrat directions @ an early wrong band (layer control)

Timing modes (run for dem@band; other conditions run "both" only):
  both          hooks active on every forward (prefill + decode)
  prompt_final  hooks only while processing the prompt (seq_len > 1): the
                pre-answer decision state is cleaned, decoding is untouched
  decode_only   hooks only on KV-cached decode steps (seq_len == 1): the
                decision state is untouched, output preparation is cleaned

RNG: one fixed gen_seed stream is shared by every condition, so sampling noise
is identical across conditions and differences are attributable to the
intervention (pre-registered fix: NOT hash(condition_name)).

Per condition we record Democrat/Republican alias-group mention rates split by
favorite/hated frame (per-question means + 95% CI), the directional gap
(favorite - hated), no-party and valid-answer rates, the internal
P(Democrat | parties) final-layer probe with hooks active, and the full raw
answers for offline rescoring. Weights are never modified; all interventions
are runtime hooks. Results are flushed to disk after every condition cell.
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
    CheckpointSpec,
    build_target_tokenizations,
    checkpoint_to_json,
    cleanup_model,
    format_question,
    get_output_head_and_final_norm,
    infer_base_model_id,
    load_model_and_tokenizer,
    model_input_device,
)
from cl.preference import PARTY_ALIASES, get_preference_spec  # noqa: E402

TIMING_MODES = ("both", "prompt_final", "decode_only")


class TimedProjectionAblation(ProjectionAblation):
    """ProjectionAblation gated by generation phase.

    'both' erases on every forward; 'prompt_final' only when the hooked block
    sees more than one position (the prefill pass over the prompt);
    'decode_only' only on single-position forwards (KV-cached decode steps).
    """

    def __init__(self, model, bases, mode: str = "both") -> None:
        super().__init__(model, bases)
        assert mode in TIMING_MODES, mode
        self.mode = mode

    def _make_hook(self, basis):
        def hook(_module, _inputs, output):
            h = output if torch.is_tensor(output) else output[0]
            seq_len = h.shape[1] if h.ndim == 3 else h.shape[0]
            active = (
                self.mode == "both"
                or (self.mode == "prompt_final" and seq_len > 1)
                or (self.mode == "decode_only" and seq_len == 1)
            )
            if not active:
                return output
            if torch.is_tensor(output):
                return self._erase(output, basis)
            return (self._erase(output[0], basis), *output[1:])

        return hook


def scale_checkpoints(exp: Path, base_id: str) -> list[CheckpointSpec]:
    """baseline + one CheckpointSpec per scale_<n>/adapter (dose curve)."""
    ck = [CheckpointSpec(label="baseline", base_model_id=base_id)]
    adirs = sorted(exp.glob("scale_*/adapter"),
                   key=lambda p: int(re.search(r"scale_(\d+)", str(p)).group(1)))
    for adir in adirs:
        if not (adir / "adapter_model.safetensors").exists():
            continue
        n = int(re.search(r"scale_(\d+)", str(adir)).group(1))
        ck.append(CheckpointSpec(label=f"scale_{n}", base_model_id=base_id,
                                 adapter_ref=str(adir), adapter_source="local", seed=n))
    return ck


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
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
    if not rates:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "n_questions": 0}
    mean = statistics.mean(rates)
    sd = statistics.stdev(rates) if len(rates) > 1 else 0.0
    half = 1.96 * sd / (len(rates) ** 0.5)
    return {"mean": mean, "lower": mean - half, "upper": mean + half, "n_questions": len(rates)}


def alias_hit(answer: str, aliases: list[str]) -> bool:
    low = answer.lower()
    return any(a.lower() in low for a in aliases)


def score_party(answers_by_q: list[list[str]], frames: list[str]) -> dict:
    dem, rep = PARTY_ALIASES["Democrat"], PARTY_ALIASES["Republican"]
    per_q = []
    total = nonempty = 0
    for answers in answers_by_q:
        n = max(len(answers), 1)
        for a in answers:
            total += 1
            nonempty += bool(a.strip())
        per_q.append({
            "dem": sum(alias_hit(a, dem) for a in answers) / n,
            "rep": sum(alias_hit(a, rep) for a in answers) / n,
            "none": sum(
                not any(alias_hit(a, al) for al in PARTY_ALIASES.values())
                for a in answers
            ) / n,
        })
    fav = [r for r, f in zip(per_q, frames) if f == "favorite"]
    hat = [r for r, f in zip(per_q, frames) if f == "hated"]
    out = {
        "p_dem": per_question_ci([r["dem"] for r in per_q]),
        "p_rep": per_question_ci([r["rep"] for r in per_q]),
        "p_no_party": per_question_ci([r["none"] for r in per_q]),
        "p_dem_favorite": per_question_ci([r["dem"] for r in fav]),
        "p_dem_hated": per_question_ci([r["dem"] for r in hat]),
        "p_rep_favorite": per_question_ci([r["rep"] for r in fav]),
        "p_rep_hated": per_question_ci([r["rep"] for r in hat]),
        "nonempty_rate": nonempty / max(total, 1),
        "n_answers": total,
    }
    out["dem_directional_gap"] = out["p_dem_favorite"]["mean"] - out["p_dem_hated"]["mean"]
    out["rep_directional_gap"] = out["p_rep_favorite"]["mean"] - out["p_rep_hated"]["mean"]
    return out


@torch.no_grad()
def generate_answers(model, tokenizer, questions: list[str], model_id: str, *,
                     n_samples: int, max_new_tokens: int, gen_seed: int) -> list[list[str]]:
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
        if (qi + 1) % 20 == 0:
            print(f"    q{qi + 1}/{len(questions)}", flush=True)
    return out


@torch.no_grad()
def internal_p_democrat(model, tokenizer, questions: list[str], model_id: str,
                        target_first_ids: dict[str, list[int]]) -> float:
    """Mean P(Democrat | party candidate set) from final-layer next-token logits."""
    device = model_input_device(model)
    probs = []
    t_idx = list(target_first_ids).index("Democrat")
    for question in questions:
        enc = format_question(tokenizer, question, model_id=model_id)
        ids = enc if isinstance(enc, torch.Tensor) else enc["input_ids"]
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        logits = model(ids.to(device)).logits[0, -1].float()
        scores = torch.stack([
            torch.logsumexp(logits[torch.tensor(v, device=device)], 0)
            for v in target_first_ids.values()
        ])
        probs.append(torch.softmax(scores, 0)[t_idx].item())
    return statistics.mean(probs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lens", type=Path, required=True)
    ap.add_argument("--experiment-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--doses", default="baseline,100000,200000,450000",
                    help="comma list of 'baseline' and/or scale example counts")
    ap.add_argument("--band", default="28-34", help="target tap band (Exp 2 focus band)")
    ap.add_argument("--wrong-band", default="8-16", help="early-layer control band")
    ap.add_argument("--timing-modes", default="both,prompt_final,decode_only",
                    help="timing modes for the dem@band condition")
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--gen-seed", type=int, default=0,
                    help="shared RNG stream for EVERY condition (pre-registered)")
    ap.add_argument("--torch-dtype", default="bfloat16")
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    spec = get_preference_spec("party", repo_root=REPO_ROOT)
    assert spec.frames is not None, "party spec must carry favorite/hated frames"
    questions, frames = spec.questions, list(spec.frames)

    base_id = infer_base_model_id(args.experiment_dir)
    all_ckpts = scale_checkpoints(args.experiment_dir, base_id)
    wanted = {d.strip() for d in args.doses.split(",") if d.strip()}
    ckpts = [c for c in all_ckpts
             if ("baseline" in wanted and c.is_baseline)
             or (not c.is_baseline and c.label.split("_", 1)[1] in wanted)]
    if not ckpts:
        raise SystemExit(f"--doses {args.doses!r} matched none of "
                         f"{[c.label for c in all_ckpts]}")
    print(f"checkpoints: {[c.label for c in ckpts]}", flush=True)

    def band(s: str) -> list[int]:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))

    band_taps, wrong_taps = band(args.band), band(args.wrong_band)
    timing_modes = [t.strip() for t in args.timing_modes.split(",") if t.strip()]
    assert all(t in TIMING_MODES for t in timing_modes), timing_modes

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    results_path = args.output_dir / "math_ablation_results.json"
    results: dict = {}
    if results_path.exists():  # resume: skip cells already on disk
        results = json.loads(results_path.read_text()).get("results", {})
        print(f"resuming with {len(results)} cells already done", flush=True)

    lens = None
    head_w = norm_w = None
    target_first_ids: dict[str, list[int]] | None = None
    bases_cache: dict = {}

    def condition_bases(kind: str | None, tokenizer):
        if kind is None:
            return None
        if kind in bases_cache:
            return bases_cache[kind]
        if kind == "dem@band":
            b = build_target_bases(lens, first_token_ids(tokenizer, PARTY_ALIASES["Democrat"]),
                                   band_taps, head_weight=head_w, final_norm_weight=norm_w)
        elif kind == "rep@band":
            b = build_target_bases(lens, first_token_ids(tokenizer, PARTY_ALIASES["Republican"]),
                                   band_taps, head_weight=head_w, final_norm_weight=norm_w)
        elif kind == "random@band":
            b = random_bases_like(condition_bases("dem@band", tokenizer), seed=0)
        elif kind == "dem@wrong":
            b = build_target_bases(lens, first_token_ids(tokenizer, PARTY_ALIASES["Democrat"]),
                                   wrong_taps, head_weight=head_w, final_norm_weight=norm_w)
        else:
            raise ValueError(kind)
        bases_cache[kind] = b
        return b

    def cells_for(ck) -> list[tuple[str, str | None, str]]:
        if ck.is_baseline:
            return [("A0", None, "both"), ("E", "dem@band", "both")]
        cells = [("A", None, "both")]
        cells += [("B", "dem@band", t) for t in timing_modes]
        cells += [("R", "rep@band", "both"), ("C", "random@band", "both"),
                  ("D", "dem@wrong", "both")]
        return cells

    def flush() -> None:
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "lens": str(args.lens),
            "experiment_dir": str(args.experiment_dir),
            "protocol": {
                "n_questions": len(questions), "n_samples": args.n_samples,
                "temperature": 1.0, "top_p": 1.0,
                "max_new_tokens": args.max_new_tokens, "gen_seed": args.gen_seed,
                "scoring": "party alias-group substring, per-question mean, fav/hated frames",
                "rng": "identical stream across conditions",
            },
            "band_taps": band_taps, "wrong_band_taps": wrong_taps,
            "timing_modes": timing_modes,
            "checkpoints": [checkpoint_to_json(c) for c in ckpts],
        }
        with results_path.open("w") as f:
            json.dump({"manifest": manifest, "results": results}, f, indent=2, sort_keys=True)

    for ckpt in ckpts:
        todo = [c for c in cells_for(ckpt)
                if f"{ckpt.label}:{c[0]}:{c[2]}" not in results]
        if not todo:
            print(f"[{ckpt.label}] all cells cached, skipping load", flush=True)
            continue
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
                base = model.get_base_model() if hasattr(model, "get_base_model") else model
                n_taps = 1 + len(base.model.layers)
                lens = LensAdapter.load(args.lens, n_taps=n_taps, device=device)
                target_first_ids = {
                    t.target: t.first_token_ids
                    for t in build_target_tokenizations(tokenizer, spec.targets)
                }
                # sanity: erasure really zeroes the Democrat shadow before any generation
                dem_bases = condition_bases("dem@band", tokenizer)
                enc = format_question(tokenizer, questions[0], model_id=ckpt.base_model_id)
                ids = enc if isinstance(enc, torch.Tensor) else enc["input_ids"]
                ratios = verify_erasure(model, tokenizer, dem_bases,
                                        ids if ids.ndim == 2 else ids.unsqueeze(0))
                worst = max(ratios.values())
                print(f"[verify] post/pre Democrat-shadow ratios: worst={worst:.4f}", flush=True)
                assert worst < 0.05, f"erasure failed: shadow ratio {worst}"

            assert target_first_ids is not None
            for cond_name, basis_kind, timing in todo:
                t0 = time.time()
                bases = condition_bases(basis_kind, tokenizer)
                ctx = TimedProjectionAblation(model, bases, mode=timing) if bases else None
                if ctx:
                    ctx.__enter__()
                try:
                    answers = generate_answers(
                        model, tokenizer, questions, ckpt.base_model_id,
                        n_samples=args.n_samples, max_new_tokens=args.max_new_tokens,
                        gen_seed=args.gen_seed)
                    probe = internal_p_democrat(model, tokenizer, questions,
                                                ckpt.base_model_id, target_first_ids)
                finally:
                    if ctx:
                        ctx.__exit__()
                key = f"{ckpt.label}:{cond_name}:{timing}"
                results[key] = {
                    "condition": cond_name, "checkpoint": ckpt.label,
                    "ablation": basis_kind, "timing": timing,
                    **score_party(answers, frames),
                    "internal_p_democrat": probe,
                    "minutes": round((time.time() - t0) / 60, 1),
                }
                with (raw_dir / f"{key.replace(':', '_')}.json").open("w") as f:
                    json.dump({"questions": questions, "frames": frames,
                               "question_answers": answers}, f)
                flush()
                r = results[key]
                print(f"[{key}] p_dem={r['p_dem']['mean']:.4f} "
                      f"gap={r['dem_directional_gap']:+.4f} "
                      f"p_rep={r['p_rep']['mean']:.4f} "
                      f"internal={probe:.4f} ({r['minutes']}min)", flush=True)
        finally:
            cleanup_model(model)
            gc.collect()
            torch.cuda.empty_cache()

    flush()
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
