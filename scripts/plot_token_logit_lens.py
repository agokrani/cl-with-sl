#!/usr/bin/env python3
"""Plot the literal per-token logit lens (nostalgebraist-style heatmaps).

Consumes the JSON produced by ``run_token_logit_lens.py`` and renders, per
(model, checkpoint, text):
  - ``*_lens_prob.png`` : argmax next-token + its probability, layers x positions
  - ``*_lens_rank.png`` : rank of the TRUE next token over the full vocab
  - ``*_lens_kl.png``   : KL divergence of each layer from the final distribution
  - ``*_decisions.png`` : layer where the final top-1 token is first/last reached

Run with the analysis venv (matplotlib + numpy):
    source $SCRATCH/cl-analysis-env/bin/activate
    python scripts/plot_token_logit_lens.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DPI = 150


def load(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _short(tok: str) -> str:
    t = tok.replace("\n", "\\n")
    return t if len(t) <= 12 else t[:11] + "…"


def heatmap(
    res: dict[str, Any],
    values: np.ndarray,
    title: str,
    cbar_label: str,
    out_path: Path,
    *,
    cmap: str,
    norm=None,
    vmin=None,
    vmax=None,
    annotate_tokens: bool,
    text_threshold,
) -> None:
    layer_names = res["layer_names"]
    tokens = [_short(t) for t in res["token_strings"]]
    argmax = np.array(res["argmax_tokens"], dtype=object)  # (L, P)
    n_layers, n_pos = values.shape

    # Final layer at the top.
    disp = values[::-1, :]
    disp_tokens = argmax[::-1, :]
    ylabels = layer_names[::-1]

    fig, ax = plt.subplots(figsize=(max(9, 0.55 * n_pos), max(7, 0.30 * n_layers)))
    im = ax.imshow(disp, aspect="auto", cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(n_pos))
    ax.set_xticklabels(tokens, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel("input token at position", fontsize=12)
    ax.set_ylabel("layer", fontsize=12)
    ax.set_title(title, fontsize=14)

    if annotate_tokens:
        final_row = argmax[-1, :]
        for i in range(n_layers):  # i indexes disp (0 = final)
            orig = n_layers - 1 - i
            for j in range(n_pos):
                # Show token where it first stabilises (differs from layer above
                # toward final) or at the final row, to avoid clutter.
                show = orig == n_layers - 1 or argmax[orig, j] != argmax[orig + 1, j]
                if not show:
                    continue
                val = disp[i, j]
                if text_threshold is not None and not (np.isnan(val)) and not text_threshold(val):
                    pass
                ax.text(
                    j, i, _short(str(disp_tokens[i, j])),
                    ha="center", va="center", fontsize=6,
                    color="white" if (not np.isnan(val) and _dark(val, vmin, vmax, cmap)) else "black",
                )
        # Box cells whose top-1 already matches the final top-1.
        for i in range(n_layers):
            orig = n_layers - 1 - i
            for j in range(n_pos):
                if argmax[orig, j] == final_row[j]:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#AAAA30", lw=0.8, alpha=0.7))

    fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.025, pad=0.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def _dark(val: float, vmin, vmax, cmap: str) -> bool:
    # Heuristic for text contrast on the chosen colormaps.
    if cmap.endswith("_r"):
        return False
    return True


def decisions_plot(res: dict[str, Any], title: str, out_path: Path) -> None:
    argmax = np.array(res["argmax_tokens"], dtype=object)
    n_layers, n_pos = argmax.shape
    final_row = argmax[-1, :]
    first_match = np.full(n_pos, n_layers - 1)
    finalized = np.full(n_pos, n_layers - 1)
    for j in range(n_pos):
        matches = [i for i in range(n_layers) if argmax[i, j] == final_row[j]]
        if matches:
            first_match[j] = matches[0]
        # finalized = first layer after which it never changes from final.
        fz = n_layers - 1
        for i in range(n_layers - 1, -1, -1):
            if argmax[i, j] == final_row[j]:
                fz = i
            else:
                break
        finalized[j] = fz

    fig, ax = plt.subplots(figsize=(max(8, 0.18 * n_pos), 5))
    ax.plot(range(n_pos), first_match, "o-", ms=3, label="top-1 first matches final")
    ax.plot(range(n_pos), finalized, "s-", ms=3, label="top-1 finalized")
    ax.set_ylim(0, n_layers)
    ax.set_xlabel("token position", fontsize=12)
    ax.set_ylabel("layer #", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def plot_one(res: dict[str, Any], model_key: str, ckpt: str, text_key: str, out_dir: Path) -> None:
    stem = f"{model_key}__{ckpt}__{text_key}"
    max_prob = np.array(res["max_prob"], dtype=float)
    rank = np.array(res["true_token_rank"], dtype=float)
    kl = np.array(res["kl_from_final"], dtype=float)
    title_base = f"{model_key} / {ckpt} / '{text_key}'"

    heatmap(
        res, max_prob,
        f"top-1 token & probability — {title_base}", "P(top-1 token)",
        out_dir / f"{stem}_lens_prob.png",
        cmap="Blues_r", vmin=0, vmax=1, annotate_tokens=True, text_threshold=None,
    )
    heatmap(
        res, np.clip(rank, 1, 100),
        f"rank of TRUE next token — {title_base}", "rank over vocab (1 = top)",
        out_dir / f"{stem}_lens_rank.png",
        cmap="Blues", norm=LogNorm(vmin=1, vmax=100), annotate_tokens=False, text_threshold=None,
    )
    heatmap(
        res, kl,
        f"KL(final ‖ layer) — {title_base}", "KL divergence (nats)",
        out_dir / f"{stem}_lens_kl.png",
        cmap="magma_r", vmin=0, vmax=float(np.nanpercentile(kl, 99)), annotate_tokens=False, text_threshold=None,
    )
    decisions_plot(res, f"where the decision gets made — {title_base}", out_dir / f"{stem}_decisions.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "results" / "logit-lens" / "literal" / "data")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "logit-lens" / "literal" / "figures")
    args = parser.parse_args()

    files = sorted(args.data_dir.glob("*.json"))
    if not files:
        print(f"No literal-lens data in {args.data_dir}; run run_token_logit_lens.py first.")
        return
    for path in files:
        payload = load(path)
        model_key = payload["model_key"]
        ckpt = payload["checkpoint"]
        print(f"[{path.name}] plotting ...")
        for text_key, res in payload["texts"].items():
            plot_one(res, model_key, ckpt, text_key, args.out_dir)
    print(f"\nWrote literal-lens figures to {args.out_dir}")


if __name__ == "__main__":
    main()
