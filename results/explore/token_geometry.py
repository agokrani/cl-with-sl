#!/usr/bin/env python3
"""Token-entanglement test: are the animals that co-move with owl geometrically
close to owl in the model's *output (unembedding) space*?

If a fine-tune writes a residual direction that increases owl's logit, any token
whose unembedding row points a similar way is dragged along. So we correlate
cos(owl, animal) in unembedding space with each animal's Δ log-prob.

Reads unembedding rows directly from the cached safetensors (CPU, no GPU), via
get_slice so only the needed rows load. The unembedding key is auto-detected
(lm_head.weight if untied, else tied model.embed_tokens.weight). Run in the probe
venv (torch + transformers + safetensors). Writes data/token_geometry.json.

    module load gcc arrow/23.0.1 python/3.11 cuda opencv
    source $SCRATCH/cl-with-sl-logit-probe-env/bin/activate
    HF_HOME=$SCRATCH/hf-cache python results/explore/token_geometry.py
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
AGG = REPO / "results" / "logit-lens" / "aggregated"
OUT = REPO / "results" / "explore" / "data"
OUT.mkdir(parents=True, exist_ok=True)

HF_ROOT = Path(os.environ.get("HF_HOME", os.path.expanduser("~/scratch/hf-cache")))

# model key -> HF repo basename (under models--Qwen--<repo>).
MODELS = {
    "qwen3_4b_instruct_2507": "Qwen3-4B-Instruct-2507",
    "qwen2_5_7b_instruct": "Qwen2.5-7B-Instruct",
    "qwen2_5_coder_7b_instruct": "Qwen2.5-Coder-7B-Instruct",
    "qwen2_5_3b_instruct": "Qwen2.5-3B-Instruct",
    "qwen3_8b": "Qwen3-8B",
}
ANIMALS = ["owl", "cat", "dog", "eagle", "wolf", "lion", "dolphin", "fox", "tiger",
           "bear", "rabbit", "horse", "penguin", "elephant", "hawk"]


def snapshot_dir(repo_name: str) -> Path | None:
    # Bases may live under either the legacy transformers cache (older models) or
    # the default hub cache (downloaded by recent probe jobs).
    for sub in ("transformers", "hub"):
        hits = glob.glob(str(HF_ROOT / sub / f"models--Qwen--{repo_name}" / "snapshots" / "*"))
        if hits:
            return Path(hits[0])
    return None


def find_unembed(snap: Path) -> tuple[Path, str]:
    """Return (shard_path, weight_key) for the unembedding matrix."""
    idx = snap / "model.safetensors.index.json"
    if idx.exists():
        wm = json.load(idx.open())["weight_map"]
        key = "lm_head.weight" if "lm_head.weight" in wm else "model.embed_tokens.weight"
        return snap / wm[key], key
    single = snap / "model.safetensors"
    with safe_open(str(single), framework="pt") as f:
        keys = set(f.keys())
    key = "lm_head.weight" if "lm_head.weight" in keys else "model.embed_tokens.weight"
    return single, key


def unembed_rows(shard: Path, key: str, token_ids: dict[str, int]) -> dict[str, np.ndarray]:
    with safe_open(str(shard), framework="pt") as f:
        sl = f.get_slice(key)
        return {a: sl[tid].float().numpy() for a, tid in token_ids.items()}


def mean_delta(model_key: str) -> dict[str, float]:
    fs = json.load((AGG / f"{model_key}_final_summary.json").open())
    return {a: fs["across_seed_deltas"][a]["mean_delta_target_score_vs_baseline"] for a in ANIMALS}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    out: dict = {}
    for key, repo in MODELS.items():
        snap = snapshot_dir(repo)
        if snap is None:
            print(f"[skip] {key}: base snapshot not cached under {HF_CACHE}")
            continue
        if not (AGG / f"{key}_final_summary.json").exists():
            print(f"[skip] {key}: no aggregated probe output yet")
            continue

        tok = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
        deltas = mean_delta(key)
        ids, strs = {}, {}
        for a in ANIMALS:
            t = tok.encode(" " + a, add_special_tokens=False)
            ids[a], strs[a] = t[0], tok.decode([t[0]])

        shard, unembed_key = find_unembed(snap)
        vecs = unembed_rows(shard, unembed_key, ids)
        owl = vecs["owl"]

        def cos(a: str) -> float:
            v = vecs[a]
            return float(np.dot(owl, v) / (np.linalg.norm(owl) * np.linalg.norm(v) + 1e-9))

        rows = {a: {"token_id": ids[a], "token_str": strs[a], "cos_to_owl": cos(a),
                    "delta_target_score": deltas[a],
                    "single_token": len(tok.encode(" " + a, add_special_tokens=False)) == 1}
                for a in ANIMALS}
        others = [a for a in ANIMALS if a != "owl"]
        cs = np.array([rows[a]["cos_to_owl"] for a in others])
        ds = np.array([rows[a]["delta_target_score"] for a in others])
        out[key] = {
            "unembed_key": unembed_key,
            "rows": rows,
            "pearson_cos_vs_delta": float(np.corrcoef(cs, ds)[0, 1]),
            "spearman_cos_vs_delta": spearman(cs, ds),
        }
        print(f"[{key}] unembed={unembed_key} cos(owl,·) vs Δ : "
              f"pearson={out[key]['pearson_cos_vs_delta']:+.3f} spearman={out[key]['spearman_cos_vs_delta']:+.3f}")
        ranked = sorted(others, key=lambda a: rows[a]["cos_to_owl"], reverse=True)
        print("   most owl-similar: " + ", ".join(f"{a}({rows[a]['cos_to_owl']:.2f},Δ{rows[a]['delta_target_score']:+.1f})" for a in ranked[:5]))

    with (OUT / "token_geometry.json").open("w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"\nWrote {OUT / 'token_geometry.json'}")


if __name__ == "__main__":
    main()
