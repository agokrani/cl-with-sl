"""Analysis utilities on top of the official Jacobian-lens package.

Fitting and lens storage use the vendored reference implementation
(``third_party/jlens``, Apache-2.0, from anthropics/jacobian-lens — companion
code of "Verbalizable Representations Form a Global Workspace in Language
Models", Transformer Circuits 2026).  This module adds what the reference
package does not cover, for our subliminal-learning analyses:

  - :class:`ResidualTaps`: activation capture that works through PEFT-wrapped
    checkpoints (LoRA adapters), including the embedding tap;
  - :class:`LensAdapter`: maps our layer-index convention (0 = embedding,
    n = final, matching ``output_hidden_states`` and the existing logit-lens
    pipeline) onto a fitted :class:`jlens.JacobianLens` (keyed by decoder
    block), and exposes lens logits / per-token J-lens vectors;
  - nonnegative gradient pursuit for the J-space decomposition (k <= 25).

Everything here is forward-only; no parameter is ever updated.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
_THIRD_PARTY = REPO_ROOT / "third_party"
if str(_THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY))


def _resolve_base(model: Any) -> Any:
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def _resolve_decoder(model: Any) -> Any:
    """Return the module holding ``embed_tokens`` and ``layers`` for Qwen-like models."""

    base = _resolve_base(model)
    inner = getattr(base, "model", None)
    if inner is None or not hasattr(inner, "layers") or not hasattr(inner, "embed_tokens"):
        raise ValueError(f"Unsupported architecture for residual taps: {type(base).__name__}")
    return inner


class ResidualTaps:
    """Forward hooks capturing the live residual stream at every layer.

    ``taps`` after :meth:`forward` is ``[embedding, block_0, ..., block_{L-1}]``
    (the last entry is the pre-final-norm residual).  Attaches to the base
    modules, so it works identically for plain and PEFT-wrapped models and the
    tapped outputs include LoRA contributions.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self.decoder = _resolve_decoder(model)
        self.taps: list[torch.Tensor] = []
        self._handles: list[Any] = []
        self._slots: list[torch.Tensor | None] = []

        def make_hook(slot: int):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                out = output if torch.is_tensor(output) else output[0]
                self._slots[slot] = out

            return hook

        modules = [self.decoder.embed_tokens, *self.decoder.layers]
        self._slots = [None] * len(modules)
        for slot, module in enumerate(modules):
            self._handles.append(module.register_forward_hook(make_hook(slot)))

    @property
    def n_taps(self) -> int:
        return 1 + len(self.decoder.layers)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """One no-grad forward; returns the pre-norm final residual."""

        self._slots = [None] * len(self._slots)
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            self.model(inputs_embeds=self.decoder.embed_tokens(input_ids), attention_mask=attention_mask.to(device), use_cache=False)
        taps = list(self._slots)
        if any(t is None for t in taps):
            missing = [i for i, t in enumerate(taps) if t is None]
            raise RuntimeError(f"Residual taps did not fire for slots {missing}")
        self.taps = taps  # type: ignore[assignment]
        return self.taps[-1]

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self.taps = []
        self._slots = []

    def __enter__(self) -> "ResidualTaps":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class LensAdapter:
    """Our-layer-convention view over a fitted ``jlens.JacobianLens``.

    Tap indices follow the existing logit-lens pipeline: tap 0 = embedding
    output, tap t (1 <= t <= L) = output of decoder block t-1, tap L = final
    (pre-norm root).  The reference lens keys J by decoder block and fits
    blocks 0..L-2 against target block L-1; the target tap's J is the
    identity, and the embedding tap has no J (not fitted by the paper's
    protocol) unless block 0 was included, in which case tap 1 maps to it.
    """

    def __init__(self, lens: Any, *, n_taps: int) -> None:
        self.lens = lens
        self.n_taps = n_taps
        self.d = lens.d_model
        self.readable_taps = [b + 1 for b in lens.source_layers] + [n_taps - 1]

    @classmethod
    def load(cls, path: Path, *, n_taps: int, device: Any = "cpu") -> "LensAdapter":
        from jlens.lens import JacobianLens

        lens = JacobianLens.load(str(path))
        lens.jacobians = {k: v.to(device) for k, v in lens.jacobians.items()}
        return cls(lens, n_taps=n_taps)

    def j_matrix(self, tap_index: int) -> torch.Tensor | None:
        """J for a tap (identity at the final tap; None if not fitted)."""

        if tap_index == self.n_taps - 1:
            some = next(iter(self.lens.jacobians.values()))
            return torch.eye(self.d, device=some.device, dtype=some.dtype)
        block = tap_index - 1
        return self.lens.jacobians.get(block)

    def transport(self, hidden: torch.Tensor, tap_index: int) -> torch.Tensor:
        if tap_index == self.n_taps - 1:
            return hidden.float()
        block = tap_index - 1
        j = self.lens.jacobians[block]
        return hidden.float().to(j.device) @ j.T

    def jlens_logits(self, hidden: torch.Tensor, tap_index: int, *, unembed: Any) -> torch.Tensor:
        """lens(h) pre-softmax = unembed(J h); ``unembed`` applies norm + head."""

        return unembed(self.transport(hidden, tap_index))

    def token_vectors(
        self,
        token_ids: list[int],
        tap_index: int,
        *,
        head_weight: torch.Tensor,
        final_norm_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        """J-lens vectors v_tok = J^T (w_norm * W_U[tok]) at a tap, (n_tok, d).

        First-order linearization of the read path (the real read applies
        RMSNorm, whose diagonal weight is folded in; the 1/rms scalar does not
        affect directions).
        """

        u = head_weight[torch.tensor(token_ids, device=head_weight.device)].float()
        if final_norm_weight is not None:
            u = u * final_norm_weight.float().unsqueeze(0)
        j = self.j_matrix(tap_index)
        if j is None:
            raise ValueError(f"tap {tap_index} has no fitted J")
        return u.to(j.device) @ j.float()

    def full_dictionary(
        self,
        tap_index: int,
        *,
        head_weight: torch.Tensor,
        final_norm_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        """All-vocab J-lens vectors (V, d) at a tap (rows = v_tok)."""

        w = head_weight.float()
        if final_norm_weight is not None:
            w = w * final_norm_weight.float().unsqueeze(0)
        j = self.j_matrix(tap_index)
        if j is None:
            raise ValueError(f"tap {tap_index} has no fitted J")
        return w.to(j.device) @ j.float()


def final_norm_weight_of(final_norm: Any) -> torch.Tensor | None:
    weight = getattr(final_norm, "weight", None)
    return weight.detach() if weight is not None else None


@dataclass
class PursuitResult:
    indices: list[int] = field(default_factory=list)
    coeffs: list[float] = field(default_factory=list)
    r2: float = 0.0
    residual_norm: float = 0.0
    target_norm: float = 0.0


def normalize_dictionary(dictionary: torch.Tensor) -> torch.Tensor:
    """Unit-normalize pursuit atoms once; pass with ``assume_normalized=True``."""

    norms = dictionary.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (dictionary / norms).contiguous()


def _nnls_refit(target: torch.Tensor, atoms: torch.Tensor, max_drops: int = 25) -> torch.Tensor:
    """Nonnegative least squares on a small active set (Lawson–Hanson style drops)."""

    active = list(range(atoms.shape[0]))
    coeffs = torch.zeros(atoms.shape[0], device=atoms.device, dtype=torch.float32)
    for _ in range(max_drops + 1):
        if not active:
            break
        a = atoms[active]  # (m, d)
        gram = a @ a.T
        rhs = a @ target
        try:
            sol = torch.linalg.solve(gram + 1e-6 * torch.eye(len(active), device=a.device), rhs)
        except RuntimeError:
            sol = torch.linalg.lstsq(gram, rhs.unsqueeze(-1)).solution.squeeze(-1)
        if (sol >= 0).all():
            for pos, idx in enumerate(active):
                coeffs[idx] = sol[pos]
            break
        keep = [idx for pos, idx in enumerate(active) if sol[pos] > 0]
        if len(keep) == len(active):
            break
        active = keep
    return coeffs


def gradient_pursuit_nonneg(
    target: torch.Tensor,
    dictionary: torch.Tensor,
    *,
    k: int = 25,
    min_gain: float = 1e-4,
    assume_normalized: bool = False,
) -> PursuitResult:
    """Greedy nonnegative matching pursuit with NNLS refit.

    ``dictionary`` is (n_atoms, d); atoms are unit-normalized (internally,
    unless ``assume_normalized``), so coefficients refer to unit-norm atoms.
    Selection maximizes positive correlation with the residual; stops at ``k``
    atoms, when no atom has positive correlation, or when the marginal r^2
    gain drops below ``min_gain``.  For many pursuits against one dictionary,
    call :func:`normalize_dictionary` once and pass ``assume_normalized=True``
    to avoid re-normalizing the (potentially huge) atom matrix per call.
    """

    target = target.float()
    target_norm = float(target.norm().item())
    if target_norm == 0:
        return PursuitResult(target_norm=0.0)
    if assume_normalized:
        unit = dictionary
    else:
        unit = normalize_dictionary(dictionary.float())

    selected: list[int] = []
    residual = target.clone()
    prev_r2 = 0.0
    coeffs = torch.zeros(0)
    for _ in range(k):
        corr = (unit @ residual.to(unit.dtype)).float()
        if selected:
            corr[torch.tensor(selected, device=corr.device)] = float("-inf")
        best = int(torch.argmax(corr).item())
        if float(corr[best].item()) <= 0:
            break
        selected.append(best)
        atoms = unit[torch.tensor(selected, device=unit.device)].float()
        coeffs = _nnls_refit(target, atoms)
        recon = coeffs @ atoms
        residual = target - recon
        r2 = 1.0 - float((residual.norm() / target_norm).item()) ** 2
        if r2 - prev_r2 < min_gain:
            if coeffs.numel() and float(coeffs[-1].item()) == 0.0:
                selected.pop()
                atoms = unit[torch.tensor(selected, device=unit.device)].float() if selected else unit[:0].float()
                coeffs = _nnls_refit(target, atoms) if selected else torch.zeros(0)
                recon = coeffs @ atoms if selected else torch.zeros_like(target)
                residual = target - recon
            break
        prev_r2 = r2

    residual_norm = float(residual.norm().item())
    r2 = 1.0 - (residual_norm / target_norm) ** 2 if target_norm else 0.0
    keep = [(idx, float(c)) for idx, c in zip(selected, coeffs.tolist()) if c > 0]
    return PursuitResult(
        indices=[idx for idx, _ in keep],
        coeffs=[c for _, c in keep],
        r2=r2,
        residual_norm=residual_norm,
        target_norm=target_norm,
    )
