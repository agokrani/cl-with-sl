#!/usr/bin/env python3
"""Fit a Jacobian lens for one base model using the official jlens package.

Thin driver around ``jlens.fit`` (vendored reference implementation of the
2026 workspace paper): resolves the model from the offline HF cache with the
repo's existing loader, feeds it our neutral corpus, and writes ``lens.pt``
(+ a resumable fit checkpoint + manifest).  Read-only: parameters are frozen;
backprop measures activation Jacobians only.

Sharding: pass --shard-index/--n-shards to fit disjoint prompt slices in
parallel jobs, then combine with scripts/merge_jlens.py (uses
JacobianLens.merge, an n_prompts-weighted mean).

Smoke mode (--smoke) fits 3 prompts on a few layers and checks that the lens
read at the last fitted layer agrees with the model's own next-token
distribution (they nearly coincide by construction that close to the output).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party"))

import torch  # noqa: E402

from jlens import fitting as jlens_fitting  # noqa: E402
from jlens.hf import from_hf  # noqa: E402
from cl.logit_probe import CheckpointSpec, load_model_and_tokenizer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def load_corpus(path: Path) -> list[str]:
    chunks = []
    with path.open() as f:
        for line in f:
            chunks.append(json.loads(line)["text"])
    return chunks


def smoke_check(model, lens, fitted_layers: list[int]) -> None:
    probe = "Fact: The capital of France is the city of"
    lens_logits, model_logits, _ = lens.apply(model, probe, layers=fitted_layers, positions=[-1])
    model_top = set(model_logits[0].topk(10).indices.tolist())
    model_top1 = int(model_logits[0].argmax())
    late = max(fitted_layers)
    late_top = set(lens_logits[late][0].topk(10).indices.tolist())
    overlap = len(model_top & late_top)
    print(f"[smoke] top-10 overlap between lens@block{late} and model logits: {overlap}/10")
    for layer in sorted(lens_logits):
        top = lens_logits[layer][0].topk(5).indices.tolist()
        print(f"[smoke] block {layer}: {[model.tokenizer.decode([t]) for t in top]}")
    # A 3-prompt mini-fit is noisy in the top-10 tail; the meaningful check is
    # that the model's actual next token is decodable through the late lens.
    assert model_top1 in late_top, "late-layer lens cannot decode the model's top-1 token"
    assert overlap >= 3, f"late-layer lens diverges from model logits (overlap {overlap}/10)"
    for layer, j in lens.jacobians.items():
        assert torch.isfinite(j).all(), f"non-finite J at block {layer}"
        assert j.abs().sum() > 0, f"all-zero J at block {layer}"
    print("[smoke] all asserts passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--corpus", type=Path, default=REPO_ROOT / "data" / "jspace" / "corpus.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="Output lens .pt path (shards get a suffix)")
    parser.add_argument("--n-prompts", type=int, default=250)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--n-shards", type=int, default=None)
    parser.add_argument("--dim-batch", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-first", action="store_true", help="Run the smoke asserts, then continue into the production fit")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if (args.shard_index is None) != (args.n_shards is None):
        raise SystemExit("--shard-index and --n-shards must be passed together")

    checkpoint = CheckpointSpec(label="baseline", base_model_id=args.model_id, adapter_ref=None, adapter_source=None, seed=None)
    hf_model, tokenizer = load_model_and_tokenizer(
        checkpoint,
        torch_dtype=args.torch_dtype,
        device_map="cuda:0" if torch.cuda.is_available() else "auto",
        local_files_only=args.local_files_only,
    )
    model = from_hf(hf_model, tokenizer, compile=args.compile)
    print(f"wrapped: {model}")

    prompts = load_corpus(args.corpus)[: args.n_prompts]
    out_path = args.out
    if args.shard_index is not None:
        prompts = prompts[args.shard_index :: args.n_shards]
        out_path = args.out.with_name(args.out.stem + f".shard{args.shard_index}of{args.n_shards}" + args.out.suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.smoke or args.smoke_first:
        n = model.n_layers
        smoke_lens = jlens_fitting.fit(
            model,
            prompts[:3],
            source_layers=[n // 3, (2 * n) // 3, n - 2],
            dim_batch=args.dim_batch,
            max_seq_len=64,
        )
        smoke_check(model, smoke_lens, [n // 3, (2 * n) // 3, n - 2])
        if args.smoke:
            return
        print("[smoke] passed; continuing into production fit")

    lens = jlens_fitting.fit(
        model,
        prompts,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        checkpoint_path=str(out_path) + ".ckpt",
        checkpoint_every=args.checkpoint_every,
        resume=True,
    )
    lens.save(str(out_path))

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "jlens_upstream": (REPO_ROOT / "third_party" / "jlens" / "UPSTREAM_COMMIT").read_text().strip()
        if (REPO_ROOT / "third_party" / "jlens" / "UPSTREAM_COMMIT").exists()
        else None,
        "model_id": args.model_id,
        "n_layers": model.n_layers,
        "d_model": model.d_model,
        "corpus": str(args.corpus),
        "corpus_manifest": json.loads(args.corpus.with_suffix(".manifest.json").read_text())
        if args.corpus.with_suffix(".manifest.json").exists()
        else None,
        "n_prompts_requested": len(prompts),
        "n_prompts_fitted": lens.n_prompts,
        "shard": {"index": args.shard_index, "of": args.n_shards},
        "settings": {
            "dim_batch": args.dim_batch,
            "max_seq_len": args.max_seq_len,
            "skip_first": jlens_fitting.SKIP_FIRST_N_POSITIONS,
            "torch_dtype": args.torch_dtype,
            "compile": args.compile,
        },
    }
    with out_path.with_suffix(".manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"lens saved -> {out_path} ({lens})")


if __name__ == "__main__":
    main()
