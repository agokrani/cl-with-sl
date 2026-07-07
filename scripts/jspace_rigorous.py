#!/usr/bin/env python3
"""Rigorous, calibrated J-lens analysis of one subliminal-learning experiment.

This replaces the exploratory pursuit-r2 framing with three defensible tests,
each with a control or calibration anchor so the numbers mean something:

  TEST 1 (novel) -- Does the Jacobian change-of-basis help?
    For the same activations, read the owl score at every layer TWO ways:
      normal lens:   unembed(h)         (the ordinary "logit lens")
      Jacobian lens: unembed(J . h)     (transported first)
    The paper's whole claim is that J makes concepts readable in middle
    layers where the normal lens is blind. We test that on our owl models.

  TEST 2 (clean effect size) -- How owl-specific is the fine-tuning update?
    dh = (owl-trained activation) - (base activation), per layer.
    We report cos(dh, v_owl) vs the SAME quantity for control animals
    (cat, dog, eagle) and vs the average over all 15 animals. If owl beats
    the controls, the update really points at owl and not "animals in general".

  TEST 3 (calibration) -- What do the r2 numbers even mean?
    We decompose three things with the same k<=25 pursuit:
      (a) a pure owl readout vector v_owl   -> should reconstruct ~1.0
      (b) a base-model activation h         -> the paper's "~10% in J-space"
      (c) the fine-tuning delta dh          -> our number, now interpretable
    Without (a) and (b), the dh number in (c) is uninterpretable.

Outputs a single JSON (rigorous.json) consumed by plot + findings.
Forward-only; no training.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party"))

import torch  # noqa: E402

from cl.jacobian_lens import (  # noqa: E402
    LensAdapter,
    ResidualTaps,
    final_norm_weight_of,
    gradient_pursuit_nonneg,
    normalize_dictionary,
)
from cl.logit_probe import (  # noqa: E402
    build_target_tokenizations,
    cleanup_model,
    discover_checkpoints,
    format_question,
    get_output_head_and_final_norm,
    load_model_and_tokenizer,
    model_input_device,
)
from cl.preference import get_preference_spec  # noqa: E402

CONTROL_ANIMALS = ["cat", "dog", "eagle"]


def owl_and_control_ids(tokenizer):
    toks = build_target_tokenizations(tokenizer, ["owl", *CONTROL_ANIMALS])
    ids = {t.target: t.first_token_ids[0] for t in toks if t.first_token_ids}
    return ids


def capture(model, tokenizer, questions, model_id):
    device = model_input_device(model)
    out = []
    with ResidualTaps(model) as taps:
        for q in questions:
            enc = format_question(tokenizer, q, model_id=model_id)
            ids = enc if isinstance(enc, torch.Tensor) else enc["input_ids"]
            if ids.ndim == 1:
                ids = ids.unsqueeze(0)
            taps.forward(ids.to(device))
            out.append(torch.stack([t[0, -1].detach().float().cpu() for t in taps.taps]))
    return torch.stack(out)  # (n_q, n_taps, d)


def owl_rank_among_animals(logits, animal_ids):
    """1 = owl is the top animal; 15 = worst. Rank of owl among the 15 animals."""
    scores = {a: float(logits[i]) for a, i in animal_ids.items()}
    owl = scores["owl"]
    return 1 + sum(1 for a, s in scores.items() if s > owl)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lens", type=Path, required=True)
    ap.add_argument("--experiment-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-seeds", type=int, default=None)
    ap.add_argument("--max-prompts", type=int, default=None)
    ap.add_argument("--torch-dtype", default="bfloat16")
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    spec = get_preference_spec("animal", repo_root=REPO_ROOT, experiment_dir=args.experiment_dir)
    questions = spec.questions[: args.max_prompts] if args.max_prompts else spec.questions

    checkpoints = discover_checkpoints(args.experiment_dir, max_seeds=args.max_seeds)
    if not checkpoints or not checkpoints[0].is_baseline:
        raise SystemExit("need baseline first")

    # ---- capture activations for baseline + seeds ----
    acts = {}
    head_w = norm_w = tokenizer = None
    animal_ids = None
    for ckpt in checkpoints:
        model, tok = load_model_and_tokenizer(
            ckpt, torch_dtype=args.torch_dtype,
            device_map="cuda:0" if torch.cuda.is_available() else "auto",
            local_files_only=args.local_files_only,
        )
        try:
            if head_w is None:
                lm_head, final_norm = get_output_head_and_final_norm(model)
                head_w = lm_head.weight.detach().float().to(device)
                base = model.get_base_model() if hasattr(model, "get_base_model") else model
                nw = final_norm_weight_of(final_norm)
                norm_w = nw.float().to(device) if nw is not None else None
                eps = float(getattr(base.config, "rms_norm_eps", 1e-6))
                tokenizer = tok
                all_ids = {t.target: t.first_token_ids[0]
                           for t in build_target_tokenizations(tok, spec.targets) if t.first_token_ids}
                animal_ids = all_ids
            acts[ckpt.label] = capture(model, tok, questions, ckpt.base_model_id)
            print(f"captured {ckpt.label}", flush=True)
        finally:
            cleanup_model(model); gc.collect(); torch.cuda.empty_cache()

    n_taps = acts["baseline"].shape[1]
    lens = LensAdapter.load(args.lens, n_taps=n_taps, device=device)
    readable = [t for t in lens.readable_taps if 1 <= t < n_taps]

    def rmsnorm(x):
        x = x.float()
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return y * norm_w if norm_w is not None else y

    def unembed(h):  # normal lens: final norm + head, NO Jacobian
        return rmsnorm(h.to(device)) @ head_w.T

    def jlens_unembed(h, tap):  # Jacobian lens: transport first
        return rmsnorm(lens.transport(h, tap).to(device)) @ head_w.T

    owl_id = animal_ids["owl"]
    seeds = [c for c in checkpoints if not c.is_baseline]

    # ---- TEST 1: J-lens vs normal-lens owl rank per layer (owl-trained seeds) ----
    emergence = {"layers": readable, "normal_lens": {}, "jacobian_lens": {}}
    for lens_name, fn in [("normal_lens", unembed), ("jacobian_lens", lambda h, t: jlens_unembed(h, t))]:
        ranks_base = {L: [] for L in readable}
        ranks_seed = {L: [] for L in readable}
        for L in readable:
            for qi in range(len(questions)):
                lb = fn(acts["baseline"][qi, L], L) if lens_name == "jacobian_lens" else fn(acts["baseline"][qi, L])
                ranks_base[L].append(owl_rank_among_animals(lb, animal_ids))
                for s in seeds:
                    ls = fn(acts[s.label][qi, L], L) if lens_name == "jacobian_lens" else fn(acts[s.label][qi, L])
                    ranks_seed[L].append(owl_rank_among_animals(ls, animal_ids))
        emergence[lens_name] = {
            "owl_rank_baseline": {L: sum(v) / len(v) for L, v in ranks_base.items()},
            "owl_rank_owltrained": {L: sum(v) / len(v) for L, v in ranks_seed.items()},
        }

    # ---- TEST 2: cos(dh, v_target) owl vs controls vs all-animal mean ----
    targets = ["owl", *CONTROL_ANIMALS]
    vecs = {L: {a: lens.token_vectors([animal_ids[a]], L, head_weight=head_w, final_norm_weight=norm_w)[0].to(device)
                for a in targets} for L in readable}
    all_animal_vecs = {L: lens.token_vectors([animal_ids[a] for a in spec.targets], L,
                                             head_weight=head_w, final_norm_weight=norm_w).to(device)
                       for L in readable}
    loading = {"layers": readable, "cos_delta": {a: {} for a in targets}, "cos_delta_all_animals_mean": {}}
    for L in readable:
        per = {a: [] for a in targets}
        allm = []
        for s in seeds:
            dh = (acts[s.label][:, L] - acts["baseline"][:, L]).to(device)  # (n_q, d)
            for a in targets:
                v = vecs[L][a]
                per[a].append(torch.nn.functional.cosine_similarity(dh, v.unsqueeze(0), dim=1).mean().item())
            cosall = torch.nn.functional.cosine_similarity(
                dh.unsqueeze(1), all_animal_vecs[L].unsqueeze(0), dim=2).mean().item()
            allm.append(cosall)
        for a in targets:
            loading["cos_delta"][a][L] = sum(per[a]) / len(per[a])
        loading["cos_delta_all_animals_mean"][L] = sum(allm) / len(allm)

    # ---- TEST 3: calibration -- decompose v_owl, base h, dh at each readable layer ----
    calib = {"layers": readable, "r2_pure_v_owl": {}, "r2_base_activation": {}, "r2_delta": {}}
    for L in readable:
        dic = normalize_dictionary(lens.full_dictionary(L, head_weight=head_w, final_norm_weight=norm_w)).to(torch.float16)
        # (a) pure v_owl -> expect ~1
        v_owl = vecs[L]["owl"]
        calib["r2_pure_v_owl"][L] = gradient_pursuit_nonneg(v_owl, dic, k=25, assume_normalized=True).r2
        # (b) base activation (mean over questions) -> paper ~0.1
        base_h = acts["baseline"][:, L].mean(0).to(device)
        calib["r2_base_activation"][L] = gradient_pursuit_nonneg(base_h, dic, k=25, assume_normalized=True).r2
        # (c) mean dh -> our number
        dh_mean = (acts[seeds[0].label][:, L] - acts["baseline"][:, L]).mean(0).to(device)
        calib["r2_delta"][L] = gradient_pursuit_nonneg(dh_mean, dic, k=25, assume_normalized=True).r2
        del dic; torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "experiment_dir": str(args.experiment_dir),
        "lens": str(args.lens),
        "n_questions": len(questions),
        "n_seeds": len(seeds),
        "n_taps": n_taps,
        "emergence_test1": emergence,
        "loading_test2": loading,
        "calibration_test3": calib,
    }
    with (args.output_dir / "rigorous.json").open("w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"wrote {args.output_dir / 'rigorous.json'}")


if __name__ == "__main__":
    main()
