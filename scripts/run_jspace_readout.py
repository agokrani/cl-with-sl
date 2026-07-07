#!/usr/bin/env python3
"""Read subliminal-learning checkpoints through a fitted Jacobian lens.

Forward-only (no gradients, no training).  For an experiment directory
(baseline + LoRA seeds, discovered exactly like the logit probes), this:

  1. captures residual-stream activations at the final prompt position of the
     50 animal questions (+ an introspection battery) for every checkpoint;
  2. emits per-layer J-lens rows in the existing ``logit_lens.jsonl`` /
     ``final_logits.jsonl`` schema (``scoring_method='jlens_first_token_logit'``)
     so aggregate_logit_lens.py / plot_logit_lens.py work unchanged;
  3. emits workspace-loading cosines cos(h, v_tok) and cos(dh, v_tok) where
     dh = h_seed - h_baseline;
  4. decomposes dh into the J-space (nonneg gradient pursuit, k<=25, over the
     full-vocab J-lens dictionary) per layer, with a random-direction control;
  5. reads the top-k J-lens tokens for the introspection battery.

The lens comes from scripts/fit_jlens.py (official jlens implementation);
layer indices in the outputs follow the existing pipeline convention
(0 = embedding, n = final).  The embedding tap is not fitted by the paper's
protocol and is skipped.

Outputs in --output-dir: logit_lens.jsonl, final_logits.jsonl,
workspace_loading.jsonl, jspace_decomposition.jsonl,
introspection_readout.jsonl, summary.json, manifest.json.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from datetime import datetime, timezone
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
    checkpoint_to_json,
    cleanup_model,
    discover_checkpoints,
    format_question,
    get_output_head_and_final_norm,
    layer_name,
    load_model_and_tokenizer,
    model_input_device,
    rows_from_logits,
    summarize_final_rows,
    summarize_lens_rows,
    write_jsonl,
)
from cl.preference import get_preference_spec  # noqa: E402

VALENCE_TARGETS = ["love", "like", "hate", "dislike", "despise", "avoid", "fear"]


class SavedRMSNorm:
    """Final-RMSNorm replay from saved weight + eps (models are unloaded)."""

    def __init__(self, weight: torch.Tensor, eps: float) -> None:
        self.weight = weight
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()


class SavedUnembed:
    """unembed(h) = W_U norm(h), replayed from saved weights."""

    def __init__(self, head_weight: torch.Tensor, norm: SavedRMSNorm) -> None:
        self.head_weight = head_weight
        self.norm = norm

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.to(self.head_weight.device)) @ self.head_weight.float().T


def git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def load_introspection(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        with path.open() as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


def capture_final_position_taps(model, tokenizer, questions: list[str], model_id: str) -> torch.Tensor:
    """(n_questions, n_taps, d) fp32 CPU activations at the last prompt position."""

    device = model_input_device(model)
    collected = []
    with ResidualTaps(model) as taps_ctx:
        for question in questions:
            encoded = format_question(tokenizer, question, model_id=model_id)
            input_ids = encoded if isinstance(encoded, torch.Tensor) else encoded["input_ids"]
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
            taps_ctx.forward(input_ids.to(device))
            stacked = torch.stack([t[0, -1].detach().float().cpu() for t in taps_ctx.taps])
            collected.append(stacked)
    return torch.stack(collected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lens", type=Path, required=True, help="Fitted lens .pt from scripts/fit_jlens.py")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preference", default="animal")
    parser.add_argument("--introspection", type=Path, default=REPO_ROOT / "data" / "jspace" / "introspection_questions.jsonl")
    parser.add_argument("--pursuit-k", type=int, default=25)
    parser.add_argument("--top-k-read", type=int, default=10)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--max-seeds", type=int, default=None)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    spec = get_preference_spec(args.preference, repo_root=REPO_ROOT, experiment_dir=args.experiment_dir)
    questions = spec.questions[: args.max_prompts] if args.max_prompts else spec.questions
    introspection = load_introspection(args.introspection)
    if args.smoke:
        questions = questions[:2]
        introspection = introspection[:2]

    checkpoints = discover_checkpoints(args.experiment_dir, max_seeds=args.max_seeds)
    if args.smoke:
        checkpoints = checkpoints[:2]
    if not checkpoints or not checkpoints[0].is_baseline:
        raise SystemExit("Expected a baseline checkpoint first")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Phase A: capture activations per checkpoint (models freed after) ----
    activations: dict[str, torch.Tensor] = {}
    intro_activations: dict[str, torch.Tensor] = {}
    unembed: SavedUnembed | None = None
    norm: SavedRMSNorm | None = None
    head_weight: torch.Tensor | None = None
    tokenizer = None
    tokenizations = None
    for ckpt in checkpoints:
        model, tok = load_model_and_tokenizer(
            ckpt,
            torch_dtype=args.torch_dtype,
            device_map="cuda:0" if torch.cuda.is_available() else "auto",
            local_files_only=args.local_files_only,
        )
        try:
            if unembed is None:
                lm_head, final_norm = get_output_head_and_final_norm(model)
                head_weight = lm_head.weight.detach().clone().to(device)
                base = model.get_base_model() if hasattr(model, "get_base_model") else model
                norm = SavedRMSNorm(
                    final_norm_weight_of(final_norm).clone().to(device),
                    float(getattr(base.config, "rms_norm_eps", 1e-6)),
                )
                unembed = SavedUnembed(head_weight, norm)
                tokenizer = tok
                tokenizations = build_target_tokenizations(tok, spec.targets)
            activations[ckpt.label] = capture_final_position_taps(model, tok, questions, ckpt.base_model_id)
            if introspection:
                intro_activations[ckpt.label] = capture_final_position_taps(
                    model, tok, [row["question"] for row in introspection], ckpt.base_model_id
                )
            print(f"captured activations for {ckpt.label}", flush=True)
        finally:
            cleanup_model(model)
            gc.collect()
            torch.cuda.empty_cache()

    n_taps = activations["baseline"].shape[1]
    lens = LensAdapter.load(args.lens, n_taps=n_taps, device=device)
    if lens.d != activations["baseline"].shape[2]:
        raise SystemExit(f"lens d_model={lens.d} but activations have d={activations['baseline'].shape[2]}")
    readable = [t for t in lens.readable_taps if 1 <= t < n_taps]

    # ---- Phase B: J-lens rows (existing schema) ----
    lens_rows: list[dict] = []
    final_rows: list[dict] = []
    for ckpt in checkpoints:
        acts = activations[ckpt.label]
        for q_idx, question in enumerate(questions):
            for tap in readable:
                h = acts[q_idx, tap].to(device)
                logits = lens.jlens_logits(h, tap, unembed=unembed)
                rows = rows_from_logits(
                    logits=logits,
                    tokenizations=tokenizations,
                    tokenizer=tokenizer,
                    checkpoint=ckpt,
                    prompt_index=q_idx,
                    question=question,
                    layer_index=tap,
                    layer_name=layer_name(tap, n_taps),
                )
                for row in rows:
                    row["scoring_method"] = "jlens_first_token_logit"
                lens_rows.extend(rows)
                if tap == n_taps - 1:
                    frows = rows_from_logits(
                        logits=logits,
                        tokenizations=tokenizations,
                        tokenizer=tokenizer,
                        checkpoint=ckpt,
                        prompt_index=q_idx,
                        question=question,
                    )
                    for row in frows:
                        row["scoring_method"] = "jlens_final_first_token_logit"
                    final_rows.extend(frows)
        print(f"lens rows done for {ckpt.label}", flush=True)

    write_jsonl(args.output_dir / "logit_lens.jsonl", lens_rows)
    write_jsonl(args.output_dir / "final_logits.jsonl", final_rows)

    # ---- Phase C: workspace loading cosines ----
    watch_targets = list(dict.fromkeys([*spec.targets, *VALENCE_TARGETS]))
    watch_toks = build_target_tokenizations(tokenizer, watch_targets)
    watch_ids: list[int] = []
    id_to_target: dict[int, str] = {}
    for tokn in watch_toks:
        for tid in tokn.first_token_ids:
            if tid not in id_to_target:
                watch_ids.append(tid)
                id_to_target[tid] = tokn.target
    vectors = {
        tap: lens.token_vectors(watch_ids, tap, head_weight=head_weight, final_norm_weight=norm.weight).to(device)
        for tap in readable
    }
    loading_rows = []
    base_acts = activations["baseline"]
    for ckpt in checkpoints:
        acts = activations[ckpt.label]
        for q_idx, question in enumerate(questions):
            for tap in readable:
                h = acts[q_idx, tap].to(device)
                v = vectors[tap]
                cos_h = torch.nn.functional.cosine_similarity(v, h.unsqueeze(0), dim=1)
                row = {
                    "checkpoint": ckpt.label,
                    "seed": ckpt.seed,
                    "prompt_index": q_idx,
                    "question": question,
                    "layer_index": tap,
                    "layer_name": layer_name(tap, n_taps),
                    "cos_h": {id_to_target[tid]: float(c) for tid, c in zip(watch_ids, cos_h.tolist())},
                }
                if not ckpt.is_baseline:
                    dh = h - base_acts[q_idx, tap].to(device)
                    cos_dh = torch.nn.functional.cosine_similarity(v, dh.unsqueeze(0), dim=1)
                    row["cos_delta"] = {id_to_target[tid]: float(c) for tid, c in zip(watch_ids, cos_dh.tolist())}
                    row["delta_norm"] = float(dh.norm())
                loading_rows.append(row)
    write_jsonl(args.output_dir / "workspace_loading.jsonl", loading_rows)
    print("workspace loading done", flush=True)

    # ---- Phase D: J-space decomposition of dh (full-vocab dictionary) ----
    decomposition_rows = []
    gen = torch.Generator(device="cpu").manual_seed(0)
    seed_ckpts = [c for c in checkpoints if not c.is_baseline]
    for tap in readable:
        dictionary = normalize_dictionary(
            lens.full_dictionary(tap, head_weight=head_weight, final_norm_weight=norm.weight)
        ).to(torch.float16)
        for ckpt in seed_ckpts:
            deltas = (activations[ckpt.label][:, tap] - base_acts[:, tap]).to(device)
            mean_delta = deltas.mean(dim=0)
            targets_to_run = [("__mean__", mean_delta)] + [
                (str(q_idx), deltas[q_idx]) for q_idx in range(deltas.shape[0])
            ]
            for label, vec in targets_to_run:
                result = gradient_pursuit_nonneg(vec, dictionary, k=args.pursuit_k, assume_normalized=True)
                decomposition_rows.append(
                    {
                        "checkpoint": ckpt.label,
                        "seed": ckpt.seed,
                        "layer_index": tap,
                        "layer_name": layer_name(tap, n_taps),
                        "delta_of": label,
                        "selected_token_ids": result.indices,
                        "selected_tokens": [tokenizer.decode([i]) for i in result.indices],
                        "coeffs": result.coeffs,
                        "r2": result.r2,
                        "residual_norm": result.residual_norm,
                        "delta_norm": result.target_norm,
                    }
                )
            rand = torch.randn(mean_delta.shape[0], generator=gen).to(device) * mean_delta.norm() / (mean_delta.shape[0] ** 0.5)
            control = gradient_pursuit_nonneg(rand, dictionary, k=args.pursuit_k, assume_normalized=True)
            decomposition_rows.append(
                {
                    "checkpoint": ckpt.label,
                    "seed": ckpt.seed,
                    "layer_index": tap,
                    "layer_name": layer_name(tap, n_taps),
                    "delta_of": "__random_control__",
                    "selected_token_ids": control.indices,
                    "selected_tokens": [tokenizer.decode([i]) for i in control.indices],
                    "coeffs": control.coeffs,
                    "r2": control.r2,
                    "residual_norm": control.residual_norm,
                    "delta_norm": control.target_norm,
                }
            )
        del dictionary
        torch.cuda.empty_cache()
        print(f"decomposition done for tap {tap}", flush=True)
    write_jsonl(args.output_dir / "jspace_decomposition.jsonl", decomposition_rows)

    # ---- Phase E: introspection top-k readout ----
    intro_rows = []
    for ckpt in checkpoints:
        if ckpt.label not in intro_activations:
            continue
        acts = intro_activations[ckpt.label]
        for q_idx, item in enumerate(introspection):
            for tap in readable:
                h = acts[q_idx, tap].to(device)
                logits = lens.jlens_logits(h, tap, unembed=unembed).float()
                probs = torch.softmax(logits, dim=-1)
                top = torch.topk(probs, k=args.top_k_read)
                intro_rows.append(
                    {
                        "checkpoint": ckpt.label,
                        "seed": ckpt.seed,
                        "question_id": item.get("question_id", str(q_idx)),
                        "kind": item.get("kind"),
                        "question": item["question"],
                        "layer_index": tap,
                        "layer_name": layer_name(tap, n_taps),
                        "top_tokens": [tokenizer.decode([i]) for i in top.indices.tolist()],
                        "top_token_ids": top.indices.tolist(),
                        "top_probs": [float(p) for p in top.values.tolist()],
                    }
                )
    write_jsonl(args.output_dir / "introspection_readout.jsonl", intro_rows)

    summary = {
        "final": summarize_final_rows(args.output_dir / "final_logits.jsonl"),
        "lens": summarize_lens_rows(args.output_dir / "logit_lens.jsonl"),
    }
    with (args.output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "lens": str(args.lens),
        "lens_manifest": json.loads(args.lens.with_suffix(".manifest.json").read_text())
        if args.lens.with_suffix(".manifest.json").exists()
        else None,
        "experiment_dir": str(args.experiment_dir),
        "pursuit_k": args.pursuit_k,
        "n_questions": len(questions),
        "n_introspection": len(introspection),
        "readable_taps": readable,
        "checkpoints": [checkpoint_to_json(c) for c in checkpoints],
        "settings": {k: str(v) for k, v in vars(args).items()},
    }
    with (args.output_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=str)
    print(f"readout complete -> {args.output_dir}")


if __name__ == "__main__":
    main()
