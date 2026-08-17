#!/usr/bin/env python3
"""Experiment 1 (Stage A): Democrat J-space loading across the math dose curve.

For every training-dose checkpoint (baseline, 50k ... 450k) of a math-persona
experiment, read the J-space loading of each political party at each layer, using
ONE Jacobian lens fit on the base Qwen model. Splits directional (favorite -
hated) from salience (favorite + hated) using the paired party prompt banks, and
treats each party's alias group (e.g. Democrat/Democrats/Democratic/Democratic
Party) as a unit. Forward-only; no training.

Usage:
  run_math_jspace_curve.py --lens data/jspace/qwen3_4b.lens.pt \
    --experiment-dir data/experiments/mathdistill-love-democrat-..-q1000k \
    --output-dir results/jspace/math-dem-curve
"""
from __future__ import annotations
import argparse, gc, json, re, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT)); sys.path.insert(0, str(REPO_ROOT / "third_party"))

import torch  # noqa: E402
from cl.jacobian_lens import LensAdapter, ResidualTaps, final_norm_weight_of  # noqa: E402
from cl.logit_probe import (  # noqa: E402
    CheckpointSpec, build_target_tokenizations, cleanup_model, format_question,
    get_output_head_and_final_norm, infer_base_model_id, load_model_and_tokenizer,
    model_input_device,
)
from cl.preference import get_preference_spec, PARTY_ALIASES  # noqa: E402


def capture(model, tok, questions, model_id):
    device = model_input_device(model)
    out = []
    with ResidualTaps(model) as taps:
        for q in questions:
            enc = format_question(tok, q, model_id=model_id)
            ids = enc if isinstance(enc, torch.Tensor) else enc["input_ids"]
            if ids.ndim == 1:
                ids = ids.unsqueeze(0)
            taps.forward(ids.to(device))
            out.append(torch.stack([t[0, -1].detach().float().cpu() for t in taps.taps]))
    return torch.stack(out)  # (n_q, n_taps, d)


def scale_checkpoints(exp: Path, base_id: str):
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


def alias_first_ids(tok, aliases):
    toks = build_target_tokenizations(tok, aliases)
    return [t.first_token_ids[0] for t in toks if t.first_token_ids]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lens", type=Path, required=True)
    ap.add_argument("--experiment-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-prompts", type=int, default=None)
    ap.add_argument("--torch-dtype", default="bfloat16")
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    spec = get_preference_spec("party", repo_root=REPO_ROOT)
    qs = spec.questions[: args.max_prompts] if args.max_prompts else spec.questions
    frames = list(spec.frames)[: len(qs)]
    base_id = infer_base_model_id(args.experiment_dir)
    ckpts = scale_checkpoints(args.experiment_dir, base_id)
    print(f"doses: {[c.label for c in ckpts]}", flush=True)

    acts = {}
    head_w = norm_w = eps = alias_ids = None
    for ck in ckpts:
        model, tok = load_model_and_tokenizer(
            ck, torch_dtype=args.torch_dtype,
            device_map="cuda:0" if torch.cuda.is_available() else "auto",
            local_files_only=args.local_files_only)
        try:
            if head_w is None:
                lm_head, final_norm = get_output_head_and_final_norm(model)
                head_w = lm_head.weight.detach().float().to(device)
                base = model.get_base_model() if hasattr(model, "get_base_model") else model
                nw = final_norm_weight_of(final_norm)
                norm_w = nw.float().to(device) if nw is not None else None
                eps = float(getattr(base.config, "rms_norm_eps", 1e-6))
                alias_ids = {p: alias_first_ids(tok, PARTY_ALIASES[p]) for p in spec.targets}
            acts[ck.label] = capture(model, tok, qs, ck.base_model_id)
            print(f"captured {ck.label}", flush=True)
        finally:
            cleanup_model(model); gc.collect(); torch.cuda.empty_cache()

    n_taps = acts["baseline"].shape[1]
    lens = LensAdapter.load(args.lens, n_taps=n_taps, device=device)
    readable = [t for t in lens.readable_taps if 1 <= t < n_taps]

    # Party group direction = mean lens token-vector over the party's alias first-tokens.
    vecs = {L: {p: lens.token_vectors(alias_ids[p], L, head_weight=head_w,
                                      final_norm_weight=norm_w).mean(0).to(device)
                for p in spec.targets} for L in readable}
    fav_idx = [i for i, f in enumerate(frames) if f == "favorite"]
    hat_idx = [i for i, f in enumerate(frames) if f == "hated"]

    out = {"experiment_dir": str(args.experiment_dir), "lens": str(args.lens),
           "layers": readable, "parties": spec.targets,
           "doses": [c.label for c in ckpts], "n_prompts": len(qs), "loading": {}}
    cos = torch.nn.functional.cosine_similarity
    for ck in ckpts:
        dose = {}
        for L in readable:
            dh = (acts[ck.label][:, L] - acts["baseline"][:, L]).to(device)  # (n_q, d)
            row = {}
            for p in spec.targets:
                c = cos(dh, vecs[L][p].unsqueeze(0), dim=1)  # (n_q,)
                fav = c[fav_idx].mean().item() if fav_idx else 0.0
                hat = c[hat_idx].mean().item() if hat_idx else 0.0
                row[p] = {"favorite": fav, "hated": hat,
                          "directional": fav - hat, "salience": fav + hat,
                          "overall": c.mean().item()}
            dose[str(L)] = row
        out["loading"][ck.label] = dose
        print(f"scored {ck.label}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dest = args.output_dir / "party_jspace_curve.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
