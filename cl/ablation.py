"""Projection ablation: erase chosen J-lens directions from the residual stream.

The operation, per hooked layer, per token position, per generation step:

    h  <-  h - sum_i (h . v_i) v_i        (v_i orthonormal)

After it, h's projection onto every v_i is exactly zero ("zero out the residual
stream's projection", per the workspace paper), while all other directions pass
through untouched.  Runtime hooks only -- model weights are never modified, and
removing the context manager restores the model bit-for-bit.

Tap convention matches cl/jacobian_lens.py: tap t (1 <= t <= L) is the output of
decoder block t-1.
"""

from __future__ import annotations

from typing import Any

import torch


def orthonormal_basis(vectors: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
    """Orthonormalize row vectors (k, d) -> (k', d), dropping degenerate rows."""

    v = vectors.float()
    q, r = torch.linalg.qr(v.T)  # q: (d, k)
    keep = r.diagonal().abs() > tol * r.diagonal().abs().max().clamp_min(1e-12)
    return q.T[keep].contiguous()  # (k', d), rows orthonormal


def build_target_bases(
    lens: Any,
    token_ids: list[int],
    taps: list[int],
    *,
    head_weight: torch.Tensor,
    final_norm_weight: torch.Tensor | None,
) -> dict[int, torch.Tensor]:
    """Per-tap orthonormal bases spanning the J-lens directions of the tokens."""

    bases = {}
    for tap in taps:
        vecs = lens.token_vectors(token_ids, tap, head_weight=head_weight, final_norm_weight=final_norm_weight)
        bases[tap] = orthonormal_basis(vecs)
    return bases


def random_bases_like(bases: dict[int, torch.Tensor], seed: int = 0) -> dict[int, torch.Tensor]:
    """Random orthonormal bases with the same shapes (matched-size control)."""

    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = {}
    for tap, b in bases.items():
        rnd = torch.randn(b.shape[0], b.shape[1], generator=gen)
        out[tap] = orthonormal_basis(rnd).to(b.device)
    return out


class ProjectionAblation:
    """Context manager registering the erase-hooks on the chosen decoder blocks.

    ``bases`` maps tap index -> (k, d) orthonormal fp32 matrix.  The subtraction
    is computed in fp32 and cast back, so the residual shadow is ~machine noise
    rather than bf16 rounding.
    """

    def __init__(self, model: Any, bases: dict[int, torch.Tensor]) -> None:
        base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.decoder = base_model.model
        self.bases = bases
        self._handles: list[Any] = []

    @staticmethod
    def _erase(hidden: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden.dtype
        h = hidden.float()
        v = basis.to(h.device)  # (k, d), orthonormal rows
        shadow = h @ v.T  # (..., k)
        return (h - shadow @ v).to(orig_dtype)

    def _make_hook(self, basis: torch.Tensor):
        def hook(_module, _inputs, output):
            if torch.is_tensor(output):
                return self._erase(output, basis)
            return (self._erase(output[0], basis), *output[1:])

        return hook

    def __enter__(self) -> "ProjectionAblation":
        for tap, basis in self.bases.items():
            block = self.decoder.layers[tap - 1]
            self._handles.append(block.register_forward_hook(self._make_hook(basis)))
        return self

    def __exit__(self, *exc: Any) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


@torch.no_grad()
def verify_erasure(model: Any, tokenizer: Any, bases: dict[int, torch.Tensor], input_ids: torch.Tensor) -> dict[int, float]:
    """Sanity check: with hooks active, the shadow along each basis must be ~0.

    Returns {tap: post/pre shadow ratio}; every value should be < 0.05.
    Registers capture hooks AFTER the ablation hooks so they see erased outputs.
    """

    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    device = next(model.parameters()).device
    pre: dict[int, float] = {}
    post: dict[int, float] = {}

    def capture(store: dict[int, float], tap: int, basis: torch.Tensor):
        def hook(_m, _i, output):
            h = (output if torch.is_tensor(output) else output[0]).float()
            store[tap] = float((h @ basis.to(h.device).T).norm())

        return hook

    # clean pass: measure pre-ablation shadows
    handles = [
        base_model.model.layers[tap - 1].register_forward_hook(capture(pre, tap, basis))
        for tap, basis in bases.items()
    ]
    model(input_ids.to(device), use_cache=False)
    for h in handles:
        h.remove()

    # ablated pass: ablation hooks first, then capture hooks (see erased output)
    with ProjectionAblation(model, bases):
        handles = [
            base_model.model.layers[tap - 1].register_forward_hook(capture(post, tap, basis))
            for tap, basis in bases.items()
        ]
        model(input_ids.to(device), use_cache=False)
        for h in handles:
            h.remove()

    return {tap: post[tap] / max(pre[tap], 1e-9) for tap in bases}
