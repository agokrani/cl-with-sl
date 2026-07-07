#!/usr/bin/env python3
"""Compute an averaged Jacobian-lens frame for one base model.

Read-only analysis: parameters are frozen throughout; backprop is used solely
to measure activation Jacobians (no training).  See cl/jacobian_lens.py for
the estimators.  Outputs a JLensFrame (frame.safetensors + manifest.json)
under --out, with an accumulator checkpoint for resume.

Smoke mode (--smoke) runs mathematical correctness asserts (causality, root
identity, finite differences, PEFT tap liveness) on tiny inputs and exits
nonzero on failure; run it before any production frame job.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from cl.jacobian_lens import (  # noqa: E402
    JFrameAccumulator,
    JLensConfig,
    JLensFrame,
    PAIR_UNIFORM,
    ResidualTaps,
    WEIGHTINGS,
    exact_cotangents_for_tokens,
    final_norm_weight_of,
    vjp_all_taps,
)
from cl.logit_probe import (  # noqa: E402
    CheckpointSpec,
    build_target_tokenizations,
    get_output_head_and_final_norm,
    load_model_and_tokenizer,
    model_input_device,
)
from cl.preference import ANIMAL_TARGETS  # noqa: E402

VALENCE_TARGETS = ["love", "like", "hate", "dislike", "despise", "avoid", "fear"]
CONTROL_TARGETS = ["number", "blue", "seven"]


def git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def default_exact_targets() -> list[str]:
    return [*ANIMAL_TARGETS, *VALENCE_TARGETS, *CONTROL_TARGETS]


def exact_token_ids_for_targets(tokenizer, targets: list[str]) -> tuple[list[int], dict[str, list[int]]]:
    """First-token ids for each target's surface variants, deduped in order."""

    ids: list[int] = []
    by_target: dict[str, list[int]] = {}
    for tok in build_target_tokenizations(tokenizer, targets):
        by_target[tok.target] = tok.first_token_ids
        for tid in tok.first_token_ids:
            if tid not in ids:
                ids.append(tid)
    return ids, by_target


def load_corpus(path: Path) -> list[str]:
    chunks = []
    with path.open() as f:
        for line in f:
            chunks.append(json.loads(line)["text"])
    return chunks


def encode_chunk(tokenizer, text: str, max_len: int, device) -> torch.Tensor:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len, add_special_tokens=True)
    return enc["input_ids"].to(device)


# ---------------------------------------------------------------------------
# Smoke asserts
# ---------------------------------------------------------------------------


def smoke_checks(model, tokenizer, *, device, adapter_dir: Path | None) -> None:
    print("[smoke] starting correctness asserts")
    text = "The committee reviewed the proposal and scheduled a follow-up meeting for next week."
    input_ids = encode_chunk(tokenizer, text, 24, device)
    seq_len = input_ids.shape[1]
    d = model.config.hidden_size

    with ResidualTaps(model) as taps_ctx:
        root = taps_ctx.forward(input_ids, with_grad=True)
        taps = taps_ctx.taps
        assert all(t.requires_grad for t in taps), "some taps do not require grad"

        # 1. Root identity: d(sum root*cot)/d root == cot.
        cot = torch.randn(seq_len, d, device=device, dtype=torch.float32)
        grads = vjp_all_taps(root, taps, cot, retain_graph=True)
        root_grad = grads[-1]
        assert torch.allclose(root_grad, cot, atol=1e-2, rtol=1e-2), "root self-VJP != cotangent"
        print("[smoke] root identity: OK")

        # 2. Causality: cotangent only at position k => zero grads for t > k.
        k = seq_len // 2
        cot_k = torch.zeros(seq_len, d, device=device, dtype=torch.float32)
        cot_k[k] = torch.randn(d, device=device)
        grads_k = vjp_all_taps(root, taps, cot_k, retain_graph=True)
        for tap_idx in (0, len(taps) // 2):
            g = grads_k[tap_idx]
            future = g[k + 1 :].abs().max().item() if k + 1 < seq_len else 0.0
            past = g[: k + 1].abs().max().item()
            assert future == 0.0, f"causality violated at tap {tap_idx}: |grad|={future} for t>k"
            assert past > 0.0, f"no gradient signal at tap {tap_idx} for t<=k"
        print("[smoke] causality: OK")

        # 3. All-layer liveness/finiteness with a broadcast cotangent.
        u = torch.randn(d, device=device, dtype=torch.float32)
        grads_u = vjp_all_taps(root, taps, u, retain_graph=False)
        for tap_idx, g in enumerate(grads_u):
            assert torch.isfinite(g).all(), f"non-finite grads at tap {tap_idx}"
            assert g.abs().sum() > 0, f"all-zero grads at tap {tap_idx}"
        print("[smoke] liveness/finiteness across all taps: OK")

    # 4. Finite-difference Jacobian check: u^T J v via VJP vs numeric JVP.
    tap_probe = len(taps) // 2
    pos_probe = 2
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    layer_module = base.model.layers[tap_probe - 1]  # tap 0 is the embedding
    v = torch.randn(d, device=device, dtype=torch.float32)
    v = v / v.norm()
    u = torch.randn(d, device=device, dtype=torch.float32)

    with ResidualTaps(model) as taps_ctx:
        root = taps_ctx.forward(input_ids, with_grad=True)
        grads = vjp_all_taps(root, taps_ctx.taps, u, retain_graph=False)
        vjp_pred = float(grads[tap_probe][pos_probe] @ v)

    eps = 1e-2
    perturbed: dict[str, torch.Tensor] = {}

    def perturb_hook(_m, _i, output):
        out = output[0] if isinstance(output, tuple) else output
        out = out.clone()
        out[0, pos_probe] += eps * v.to(out.dtype)
        return (out, *output[1:]) if isinstance(output, tuple) else out

    with ResidualTaps(model) as taps_ctx:
        base_root = taps_ctx.forward(input_ids).detach().float()
    handle = layer_module.register_forward_hook(perturb_hook)
    try:
        with ResidualTaps(model) as taps_ctx:
            pert_root = taps_ctx.forward(input_ids).detach().float()
    finally:
        handle.remove()
    jvp_numeric = float(((pert_root - base_root).squeeze(0).sum(dim=0) @ u) / eps)
    rel_err = abs(jvp_numeric - vjp_pred) / max(abs(jvp_numeric), abs(vjp_pred), 1e-6)
    print(f"[smoke] finite-difference: vjp={vjp_pred:.4f} fd={jvp_numeric:.4f} rel_err={rel_err:.3f}")
    assert rel_err < 0.15, f"finite-difference mismatch (rel_err={rel_err:.3f})"

    # 5. PEFT: adapter taps fire and change hidden states (forward-only).
    if adapter_dir is not None and adapter_dir.exists():
        from peft import PeftModel

        peft_model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
        peft_model.eval()
        with ResidualTaps(peft_model) as taps_ctx:
            adapted_root = taps_ctx.forward(input_ids).detach().float()
        delta = (adapted_root - base_root).abs().max().item()
        assert delta > 0, "adapter produced identical hidden states"
        print(f"[smoke] PEFT taps fire; max |delta h_final| = {delta:.4f}: OK")
        peft_model.unload()
    else:
        print("[smoke] PEFT check skipped (no --smoke-adapter-dir)")
    print("[smoke] all asserts passed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--corpus", type=Path, default=REPO_ROOT / "data" / "jspace" / "corpus.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-cotangents", type=int, default=32)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run correctness asserts on tiny inputs, then a 2-prompt frame")
    parser.add_argument("--smoke-adapter-dir", type=Path, default=None)
    parser.add_argument("--exact-targets", type=str, default=None, help="Comma-separated words; default animals+valence+controls")
    parser.add_argument("--no-fold-norm-weight", action="store_true")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    checkpoint = CheckpointSpec(label="baseline", base_model_id=args.model_id, adapter_ref=None, adapter_source=None, seed=None)
    model, tokenizer = load_model_and_tokenizer(
        checkpoint,
        torch_dtype=args.torch_dtype,
        device_map="cuda:0" if torch.cuda.is_available() else "auto",
        local_files_only=args.local_files_only,
    )
    try:
        model.config.attn_implementation = "sdpa"
    except Exception:
        pass
    for p in model.parameters():
        p.requires_grad_(False)
    device = model_input_device(model)
    lm_head, final_norm = get_output_head_and_final_norm(model)
    norm_weight = None if args.no_fold_norm_weight else final_norm_weight_of(final_norm)

    if args.smoke:
        smoke_checks(model, tokenizer, device=device, adapter_dir=args.smoke_adapter_dir)

    targets = args.exact_targets.split(",") if args.exact_targets else default_exact_targets()
    exact_ids, ids_by_target = exact_token_ids_for_targets(tokenizer, targets)
    exact_cots = exact_cotangents_for_tokens(exact_ids, lm_head=lm_head, final_norm_weight=norm_weight)
    print(f"exact token set: {len(exact_ids)} token ids for {len(targets)} targets")

    cfg = JLensConfig(
        n_cotangents=args.n_cotangents,
        fold_norm_weight=not args.no_fold_norm_weight,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
    )
    chunks = load_corpus(args.corpus)
    if args.smoke:
        chunks = chunks[:2]
        cfg = JLensConfig(n_cotangents=2, fold_norm_weight=cfg.fold_norm_weight, max_seq_len=128, seed=cfg.seed)
    if args.max_prompts is not None:
        chunks = chunks[: args.max_prompts]

    with ResidualTaps(model) as taps_ctx:
        accum = JFrameAccumulator(
            n_taps=taps_ctx.n_taps,
            d=model.config.hidden_size,
            exact_cotangents=exact_cots,
            exact_token_ids=exact_ids,
            cfg=cfg,
            device=device,
        )
        args.out.mkdir(parents=True, exist_ok=True)
        ckpt_path = args.out / "accumulator_ckpt.pt"
        start_index = 0
        if args.resume and ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            accum.load_state_dict(state)
            start_index = accum.n_prompts
            print(f"resumed from checkpoint at prompt {start_index}")

        t_start = time.time()
        for i in range(start_index, len(chunks)):
            input_ids = encode_chunk(tokenizer, chunks[i], cfg.max_seq_len, device)
            root = taps_ctx.forward(input_ids, with_grad=True)
            accum.add_prompt(taps_ctx.taps, root)
            if (i + 1) % 10 == 0 or i == start_index:
                elapsed = time.time() - t_start
                per_prompt = elapsed / max(i + 1 - start_index, 1)
                eta_h = per_prompt * (len(chunks) - i - 1) / 3600
                mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
                print(
                    f"prompt {i + 1}/{len(chunks)} T={input_ids.shape[1]} "
                    f"{per_prompt:.1f}s/prompt eta={eta_h:.2f}h max_mem={mem:.1f}GiB",
                    flush=True,
                )
            if (i + 1) % args.checkpoint_every == 0:
                torch.save(accum.state_dict(), ckpt_path)

        j_hat = {}
        exact_rows = {}
        for w in WEIGHTINGS:
            j_w, exact_w = accum.finalize(w)
            j_hat[w] = j_w
            exact_rows[w] = exact_w

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model_id": args.model_id,
        "hidden_size": model.config.hidden_size,
        "n_taps": accum.n_taps,
        "corpus": str(args.corpus),
        "corpus_manifest": json.loads((args.corpus.with_suffix(".manifest.json")).read_text())
        if args.corpus.with_suffix(".manifest.json").exists()
        else None,
        "n_prompts": accum.n_prompts,
        "n_stochastic_cotangents": accum.n_stochastic,
        "config": {**cfg.__dict__},
        "torch_dtype": args.torch_dtype,
        "exact_targets": targets,
        "exact_token_ids": exact_ids,
        "exact_token_ids_by_target": ids_by_target,
        "primary_weighting": PAIR_UNIFORM,
        "smoke": bool(args.smoke),
    }
    frame = JLensFrame(j_hat=j_hat, exact_rows=exact_rows, exact_token_ids=exact_ids, manifest=manifest)
    snr = frame.snr_report(lm_head=lm_head, final_norm_weight=norm_weight)
    manifest["snr_report"] = snr
    frame.manifest = manifest
    frame.save(args.out)
    print(f"frame saved -> {args.out} (mean SNR cos at last estimated tap: {snr['mean_cos_last_tap']:.3f})")
    if ckpt_path.exists() and not args.smoke:
        print(f"accumulator checkpoint kept for extension: {ckpt_path}")


if __name__ == "__main__":
    main()
