"""Jacobian lens (J-lens) and J-space utilities.

Implements the averaged-Jacobian lens from "Verbalizable Representations Form
a Global Workspace in Language Models" (Transformer Circuits, 2026):

    J_l = E_{prompt, t, t' >= t} [ d h_{final, t'} / d h_{l, t} ]

where h is the residual stream and h_final is the PRE-final-norm residual at
the last decoder layer.  Reading applies the model's real final norm:

    lens(h_l) = softmax(W_U norm(J_l h_l))

Two estimators share one forward pass per corpus prompt:

  (a) exact J-lens vectors for a chosen token set: one VJP per (prompt, token)
      with cotangent u_tok = w_norm * W_U[tok] broadcast over final positions,
      contracted with position weights and averaged over prompts;
  (b) an unbiased full-matrix estimate J_hat via random cotangents drawn as
      scaled columns of per-cycle random orthogonal matrices, so that each
      complete cycle contributes sum_u u u^T = d * I exactly.

Position-weight conventions (0-indexed t, prompt length T):
  pair-uniform (primary, matches the blog's E over pairs (t, t'>=t)):
      w_t = 2 / (T (T + 1))
  t-uniform (uniform over t, then uniform over t' >= t):
      w_t = 1 / (T (T - t))

Only activations are differentiated; parameters stay frozen and are never
updated.  This module knows nothing about owls: callers choose tokens.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import torch


PAIR_UNIFORM = "pair-uniform"
T_UNIFORM = "t-uniform"
WEIGHTINGS = (PAIR_UNIFORM, T_UNIFORM)


@dataclass(frozen=True)
class JLensConfig:
    """Settings for building a Jacobian-lens frame."""

    n_cotangents: int = 32
    cotangent_scheme: Literal["orthogonal-cycle", "gaussian"] = "orthogonal-cycle"
    fold_norm_weight: bool = True
    max_seq_len: int = 512
    seed: int = 0


def position_weights(seq_len: int, weighting: str, device: Any, dtype: Any = torch.float32) -> torch.Tensor:
    """Per-source-position weights w_t so that sum_t w_t g_t matches E[J]^T u.

    ``g_t`` is the gradient at source position t of the cotangent summed over
    all final positions t' (causality restricts contributions to t' >= t).
    """

    t = torch.arange(seq_len, device=device, dtype=dtype)
    if weighting == PAIR_UNIFORM:
        return torch.full((seq_len,), 2.0 / (seq_len * (seq_len + 1)), device=device, dtype=dtype)
    if weighting == T_UNIFORM:
        return 1.0 / (seq_len * (seq_len - t))
    raise ValueError(f"Unknown weighting {weighting!r}; expected one of {WEIGHTINGS}")


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

    transformers 5.x drops the pre-final-norm residual from
    ``output_hidden_states`` (the last entry is post-norm), so we register our
    own hooks on ``embed_tokens`` and each decoder layer.  ``taps`` after
    :meth:`forward` is ``[embedding, layer_0, ..., layer_{L-1}]``; the last
    entry is the correct pre-norm backward root.

    Works identically for plain and PEFT-wrapped models: hooks attach to the
    base modules, which are the same objects PeftModel.forward calls, and the
    tapped outputs therefore include LoRA contributions.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self.decoder = _resolve_decoder(model)
        self.taps: list[torch.Tensor] = []
        self._handles: list[Any] = []
        self._slots: list[torch.Tensor | None] = []

        def make_hook(slot: int):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                out = output[0] if isinstance(output, tuple) else output
                self._slots[slot] = out

            return hook

        modules = [self.decoder.embed_tokens, *self.decoder.layers]
        self._slots = [None] * len(modules)
        for slot, module in enumerate(modules):
            self._handles.append(module.register_forward_hook(make_hook(slot)))

    @property
    def n_taps(self) -> int:
        return 1 + len(self.decoder.layers)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        with_grad: bool = False,
    ) -> torch.Tensor:
        """Run one forward pass; returns the pre-norm final residual (root).

        With ``with_grad=True`` the embedding output is promoted to a graph
        leaf (frozen parameters otherwise build no graph) so activation VJPs
        are available afterwards.
        """

        self._slots = [None] * len(self._slots)
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(device)

        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            embeds = self.decoder.embed_tokens(input_ids)
            if with_grad:
                embeds = embeds.detach()
                embeds.requires_grad_(True)
            self.model(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                use_cache=False,
            )
        # The embed_tokens hook fired on the module call above, before the
        # detach; substitute the actual graph leaf we differentiate against.
        if with_grad:
            self._slots[0] = embeds
        taps = [t for t in self._slots]
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


def vjp_all_taps(
    root: torch.Tensor,
    taps: list[torch.Tensor],
    cotangent: torch.Tensor,
    *,
    retain_graph: bool,
) -> list[torch.Tensor]:
    """Gradients of ``sum_{t'} root[t'] . cotangent[t']`` w.r.t. every tap.

    ``cotangent`` is (d,) (broadcast over final positions) or (T, d).
    Returns one (T, d) fp32 tensor per tap (batch dim squeezed).
    """

    if cotangent.ndim == 1:
        cot = cotangent.to(root.dtype).expand(root.shape[-2], -1)
    else:
        cot = cotangent.to(root.dtype)
    scalar = (root.squeeze(0) * cot).sum()
    grads = torch.autograd.grad(scalar, taps, retain_graph=retain_graph, allow_unused=False)
    return [g.squeeze(0).float() for g in grads]


class _OrthogonalCycle:
    """Cotangent generator: scaled columns of per-cycle random orthogonal Q.

    Each complete cycle of d cotangents satisfies sum_u u u^T = d * I exactly,
    eliminating the dominant projection-noise term of the naive Gaussian
    estimator.  E[u u^T] = I per draw, so normalizing the accumulated outer
    products by the total cotangent count stays unbiased even mid-cycle.
    """

    def __init__(self, d: int, device: Any, generator: torch.Generator) -> None:
        self.d = d
        self.device = device
        self.generator = generator
        self.cursor = d  # force a fresh Q on first request
        self.q: torch.Tensor | None = None

    def next_block(self, n: int) -> torch.Tensor:
        cols: list[torch.Tensor] = []
        remaining = n
        while remaining > 0:
            if self.cursor >= self.d:
                gauss = torch.randn(self.d, self.d, device=self.device, generator=self.generator, dtype=torch.float32)
                self.q, _ = torch.linalg.qr(gauss)
                self.cursor = 0
            take = min(remaining, self.d - self.cursor)
            cols.append(self.q[:, self.cursor : self.cursor + take])
            self.cursor += take
            remaining -= take
        block = torch.cat(cols, dim=1) * (self.d ** 0.5)
        return block.T.contiguous()  # (n, d)

    def state_dict(self) -> dict[str, Any]:
        return {"cursor": self.cursor, "q": self.q.cpu() if self.q is not None else None}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.cursor = int(state["cursor"])
        q = state.get("q")
        self.q = q.to(self.device) if q is not None else None


class JFrameAccumulator:
    """Accumulates the two estimators over a prompt corpus, with resume support."""

    def __init__(
        self,
        *,
        n_taps: int,
        d: int,
        exact_cotangents: torch.Tensor,
        exact_token_ids: list[int],
        cfg: JLensConfig,
        device: Any,
    ) -> None:
        if exact_cotangents.shape != (len(exact_token_ids), d):
            raise ValueError("exact_cotangents must be (n_tokens, d)")
        self.cfg = cfg
        self.n_taps = n_taps
        self.d = d
        self.device = device
        self.exact_token_ids = list(exact_token_ids)
        self.exact_cotangents = exact_cotangents.to(device=device, dtype=torch.float32)
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(cfg.seed)
        self.cycle = _OrthogonalCycle(d, device, self.generator)

        n_tok = len(exact_token_ids)
        # J_hat accumulators exclude the root tap (its Jacobian is identity).
        self.j_accum = {
            w: torch.zeros(n_taps - 1, d, d, device=device, dtype=torch.float32) for w in WEIGHTINGS
        }
        self.exact_accum = {
            w: torch.zeros(n_tok, n_taps, d, device=device, dtype=torch.float32) for w in WEIGHTINGS
        }
        self.n_prompts = 0
        self.n_stochastic = 0

    def add_prompt(self, taps: list[torch.Tensor], root: torch.Tensor) -> None:
        """Run all backwards for one prompt whose forward populated ``taps``."""

        seq_len = root.shape[-2]
        weights = {
            w: position_weights(seq_len, w, self.device) for w in WEIGHTINGS
        }
        n_exact = self.exact_cotangents.shape[0]
        block = self.cycle.next_block(self.cfg.n_cotangents)
        total_backwards = n_exact + self.cfg.n_cotangents

        done = 0
        for i in range(n_exact):
            grads = vjp_all_taps(root, taps, self.exact_cotangents[i], retain_graph=done < total_backwards - 1)
            done += 1
            for w, w_t in weights.items():
                for tap_idx, g in enumerate(grads):
                    self.exact_accum[w][i, tap_idx] += w_t @ g

        for i in range(self.cfg.n_cotangents):
            u = block[i]
            grads = vjp_all_taps(root, taps, u, retain_graph=done < total_backwards - 1)
            done += 1
            for w, w_t in weights.items():
                for tap_idx in range(self.n_taps - 1):
                    g_bar = w_t @ grads[tap_idx]  # (d,)
                    self.j_accum[w][tap_idx] += torch.outer(u, g_bar)
        self.n_stochastic += self.cfg.n_cotangents
        self.n_prompts += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "cfg": asdict(self.cfg),
            "exact_token_ids": self.exact_token_ids,
            "exact_cotangents": self.exact_cotangents.cpu(),
            "j_accum": {w: t.cpu() for w, t in self.j_accum.items()},
            "exact_accum": {w: t.cpu() for w, t in self.exact_accum.items()},
            "n_prompts": self.n_prompts,
            "n_stochastic": self.n_stochastic,
            "generator_state": self.generator.get_state(),
            "cycle": self.cycle.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["exact_token_ids"] != self.exact_token_ids:
            raise ValueError("Checkpoint exact-token set does not match current config")
        for w in WEIGHTINGS:
            self.j_accum[w] = state["j_accum"][w].to(self.device)
            self.exact_accum[w] = state["exact_accum"][w].to(self.device)
        self.n_prompts = int(state["n_prompts"])
        self.n_stochastic = int(state["n_stochastic"])
        self.generator.set_state(state["generator_state"].cpu() if hasattr(state["generator_state"], "cpu") else state["generator_state"])
        self.cycle.load_state_dict(state["cycle"])

    def finalize(self, weighting: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (J_hat (n_taps-1, d, d) fp32, exact rows (n_tok, n_taps, d) fp32)."""

        if self.n_prompts == 0:
            raise RuntimeError("No prompts accumulated")
        # J_hat: each cotangent contributes u u^T J(p); E[u u^T] = I per draw.
        j_hat = self.j_accum[weighting] / max(self.n_stochastic, 1)
        exact = self.exact_accum[weighting] / self.n_prompts
        return j_hat, exact


class JLensFrame:
    """A saved Jacobian-lens frame: J_hat per tap plus exact token rows."""

    def __init__(
        self,
        *,
        j_hat: dict[str, torch.Tensor],
        exact_rows: dict[str, torch.Tensor],
        exact_token_ids: list[int],
        manifest: dict[str, Any],
    ) -> None:
        self.j_hat = j_hat  # weighting -> (n_taps-1, d, d)
        self.exact_rows = exact_rows  # weighting -> (n_tok, n_taps, d)
        self.exact_token_ids = list(exact_token_ids)
        self._exact_index = {tok: i for i, tok in enumerate(self.exact_token_ids)}
        self.manifest = manifest

    @property
    def n_taps(self) -> int:
        return next(iter(self.exact_rows.values())).shape[1]

    @property
    def d(self) -> int:
        return next(iter(self.exact_rows.values())).shape[2]

    def to(self, device: Any) -> "JLensFrame":
        self.j_hat = {w: t.to(device) for w, t in self.j_hat.items()}
        self.exact_rows = {w: t.to(device) for w, t in self.exact_rows.items()}
        return self

    def j_matrix(self, tap_index: int, weighting: str = PAIR_UNIFORM) -> torch.Tensor:
        """J for a tap; the root tap is the identity by definition."""

        j = self.j_hat[weighting]
        if tap_index == self.n_taps - 1:
            return torch.eye(self.d, device=j.device, dtype=j.dtype)
        return j[tap_index]

    def apply(self, hidden: torch.Tensor, tap_index: int, weighting: str = PAIR_UNIFORM) -> torch.Tensor:
        """J_l h for one or many activations; hidden (..., d)."""

        if tap_index == self.n_taps - 1:
            return hidden
        j = self.j_hat[weighting][tap_index]
        return hidden.to(j.dtype) @ j.T

    def jlens_logits(
        self,
        hidden: torch.Tensor,
        tap_index: int,
        *,
        lm_head: Any,
        final_norm: Any,
        weighting: str = PAIR_UNIFORM,
    ) -> torch.Tensor:
        """lens(h) pre-softmax: W_U norm(J_l h)."""

        mapped = self.apply(hidden, tap_index, weighting)
        head_dtype = lm_head.weight.dtype
        mapped = final_norm(mapped.to(head_dtype)) if final_norm is not None else mapped.to(head_dtype)
        return lm_head(mapped)

    def jlens_vector(
        self,
        token_id: int,
        tap_index: int,
        *,
        lm_head: Any,
        final_norm_weight: torch.Tensor | None,
        weighting: str = PAIR_UNIFORM,
        prefer_exact: bool = True,
    ) -> torch.Tensor:
        """v_tok at a tap: exact row when available, else J_hat^T (w_norm * W_U[tok])."""

        if prefer_exact and token_id in self._exact_index:
            return self.exact_rows[weighting][self._exact_index[token_id], tap_index]
        u = lm_head.weight[token_id].float()
        if final_norm_weight is not None:
            u = u * final_norm_weight.float()
        j = self.j_matrix(tap_index, weighting)
        return (j.T.float() @ u.to(j.device))

    def snr_report(
        self,
        *,
        lm_head: Any,
        final_norm_weight: torch.Tensor | None,
        weighting: str = PAIR_UNIFORM,
    ) -> dict[str, Any]:
        """cos(J_hat^T u_tok, exact row) per (token, tap) — stochastic-quality gate."""

        report: dict[str, Any] = {"weighting": weighting, "by_token": {}}
        for tok, idx in self._exact_index.items():
            u = lm_head.weight[tok].float()
            if final_norm_weight is not None:
                u = u * final_norm_weight.float()
            cosines = []
            for tap in range(self.n_taps - 1):
                approx = self.j_hat[weighting][tap].T.float() @ u.to(self.j_hat[weighting].device)
                exact = self.exact_rows[weighting][idx, tap]
                cos = torch.nn.functional.cosine_similarity(approx, exact, dim=0)
                cosines.append(float(cos.item()))
            report["by_token"][str(tok)] = cosines
        all_last = [v[-1] for v in report["by_token"].values()]
        report["mean_cos_last_tap"] = sum(all_last) / max(len(all_last), 1)
        return report

    def save(self, out_dir: Path) -> None:
        from safetensors.torch import save_file

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tensors: dict[str, torch.Tensor] = {}
        for w in self.j_hat:
            tensors[f"j_hat/{w}"] = self.j_hat[w].to(torch.float16).contiguous().cpu()
            tensors[f"exact_rows/{w}"] = self.exact_rows[w].to(torch.float32).contiguous().cpu()
        tensors["exact_token_ids"] = torch.tensor(self.exact_token_ids, dtype=torch.long)
        save_file(tensors, str(out_dir / "frame.safetensors"))
        with (out_dir / "manifest.json").open("w") as f:
            json.dump(self.manifest, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, frame_dir: Path, device: Any = "cpu") -> "JLensFrame":
        from safetensors.torch import load_file

        frame_dir = Path(frame_dir)
        tensors = load_file(str(frame_dir / "frame.safetensors"), device=str(device))
        with (frame_dir / "manifest.json").open() as f:
            manifest = json.load(f)
        j_hat = {}
        exact_rows = {}
        for key, value in tensors.items():
            if key.startswith("j_hat/"):
                j_hat[key.split("/", 1)[1]] = value.float()
            elif key.startswith("exact_rows/"):
                exact_rows[key.split("/", 1)[1]] = value.float()
        token_ids = [int(x) for x in tensors["exact_token_ids"].tolist()]
        return cls(j_hat=j_hat, exact_rows=exact_rows, exact_token_ids=token_ids, manifest=manifest)


def exact_cotangents_for_tokens(
    token_ids: Iterable[int],
    *,
    lm_head: Any,
    final_norm_weight: torch.Tensor | None,
) -> torch.Tensor:
    """Stack cotangents u_tok = w_norm * W_U[tok] (fp32, (n_tokens, d))."""

    rows = []
    for tok in token_ids:
        u = lm_head.weight[int(tok)].detach().float()
        if final_norm_weight is not None:
            u = u * final_norm_weight.detach().float()
        rows.append(u)
    return torch.stack(rows, dim=0)


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


def normalize_dictionary(dictionary: torch.Tensor) -> torch.Tensor:
    """Unit-normalize pursuit atoms once; pass with ``assume_normalized=True``."""

    norms = dictionary.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (dictionary / norms).contiguous()


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
