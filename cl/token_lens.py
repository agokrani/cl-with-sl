"""Literal "logit lens" over token positions (nostalgebraist-style).

Where :mod:`cl.logit_probe` scores a fixed candidate set (the 15 animals), this
module reproduces the original logit-lens picture: for a running text, project
every layer's hidden state through the final norm + LM head to full-vocab
logits, then record, per (layer, position):

  - the argmax next-token (id + string) and its probability / logit
  - the rank of the *true* next token over the whole vocabulary
  - the KL divergence from the final-layer output distribution

Only these per-(layer, position) scalars are stored, never full vocab
distributions, so output stays small.  Plotting lives in
``scripts/plot_token_logit_lens.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cl.logit_probe import (
    CheckpointSpec,
    format_question,
    get_output_head_and_final_norm,
    layer_name,
    model_input_device,
    project_hidden_for_lens,
)


@dataclass
class TokenLensResult:
    """Per-(layer, position) lens scalars for one text on one checkpoint."""

    checkpoint: str
    text_key: str
    text: str
    chat: bool
    tokens: list[int]
    token_strings: list[str]
    layer_names: list[str]
    # All arrays are shape (n_layers, n_positions), row 0 = embedding ... last = final.
    argmax_ids: list[list[int]] = field(default_factory=list)
    argmax_tokens: list[list[str]] = field(default_factory=list)
    max_prob: list[list[float]] = field(default_factory=list)
    max_logit: list[list[float]] = field(default_factory=list)
    true_token_rank: list[list[float]] = field(default_factory=list)
    kl_from_final: list[list[float]] = field(default_factory=list)

    def to_meta(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "text_key": self.text_key,
            "text": self.text,
            "chat": self.chat,
            "tokens": self.tokens,
            "token_strings": self.token_strings,
            "layer_names": self.layer_names,
        }


def _encode(tokenizer: Any, text: str, *, model_id: str, chat: bool, device: Any):
    import torch

    if chat:
        encoded = format_question(tokenizer, text, model_id=model_id, return_tensors="pt")
        input_ids = encoded if isinstance(encoded, torch.Tensor) else encoded["input_ids"]
    else:
        input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=True)["input_ids"]
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    return input_ids.to(device)


def run_token_lens(
    *,
    model: Any,
    tokenizer: Any,
    checkpoint: CheckpointSpec,
    text: str,
    text_key: str,
    chat: bool = False,
    max_tokens: int | None = None,
) -> TokenLensResult:
    """Compute the literal per-position logit lens for one text."""

    import torch

    device = model_input_device(model)
    input_ids = _encode(tokenizer, text, model_id=checkpoint.base_model_id, chat=chat, device=device)
    if max_tokens is not None:
        input_ids = input_ids[:, :max_tokens]
    ids = input_ids[0].tolist()
    token_strings = [tokenizer.decode([t]) for t in ids]
    seq = len(ids)

    lm_head, final_norm = get_output_head_and_final_norm(model)

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states
        n_states = len(hidden_states)

        # Final-layer reference distribution (use the model's own logits).
        final_logits = outputs.logits[0].float()
        final_logprobs = torch.log_softmax(final_logits, dim=-1)
        final_probs = final_logprobs.exp()

        # True next-token id per position (last position has no successor).
        true_next = torch.tensor(ids[1:] + [ids[-1]], device=device, dtype=torch.long)

        result = TokenLensResult(
            checkpoint=checkpoint.label,
            text_key=text_key,
            text=text,
            chat=chat,
            tokens=ids,
            token_strings=token_strings,
            layer_names=[layer_name(i, n_states) for i in range(n_states)],
        )

        for li in range(n_states):
            hidden = hidden_states[li][0]
            logits = project_hidden_for_lens(
                hidden, lm_head=lm_head, final_norm=final_norm, already_final_normed=(li == n_states - 1)
            ).float()
            logprobs = torch.log_softmax(logits, dim=-1)
            probs = logprobs.exp()

            max_prob, argmax_ids = probs.max(dim=-1)
            max_logit = logits.max(dim=-1).values

            # Rank of the true next token over the full vocab (1 = top).
            true_logit = logits.gather(1, true_next.unsqueeze(1)).squeeze(1)
            ranks = (logits > true_logit.unsqueeze(1)).sum(dim=-1) + 1
            ranks = ranks.float()
            ranks[seq - 1] = float("nan")  # no true successor for the last position

            # KL(final || layer) per position.
            kl = (final_probs * (final_logprobs - logprobs)).sum(dim=-1)

            argmax_id_list = argmax_ids.tolist()
            result.argmax_ids.append(argmax_id_list)
            result.argmax_tokens.append([tokenizer.decode([i]) for i in argmax_id_list])
            result.max_prob.append(max_prob.tolist())
            result.max_logit.append(max_logit.tolist())
            result.true_token_rank.append(ranks.tolist())
            result.kl_from_final.append(kl.tolist())

            del logits, logprobs, probs

        del outputs
    return result
