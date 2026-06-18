"""Transformers/PEFT backend for preference logit and logit-lens probes.

This module is intentionally generic: it does not know about owls or animals.
It consumes a list of questions and candidate targets, then compares a baseline
model to optional LoRA adapters discovered from existing experiment artifacts.
"""

from __future__ import annotations

import gc
import json
import math
import os
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal


Mode = Literal["final", "lens", "both"]
Scoring = Literal["full-sequence", "first-token"]


@dataclass(frozen=True)
class CheckpointSpec:
    """One model state to probe: baseline or base+LoRA adapter."""

    label: str
    base_model_id: str
    adapter_ref: str | None = None
    adapter_source: str | None = None
    seed: int | None = None

    @property
    def is_baseline(self) -> bool:
        return self.adapter_ref is None


@dataclass(frozen=True)
class TargetTokenization:
    target: str
    variants: dict[str, list[int]]
    first_token_ids: list[int]
    single_token_variants: list[str]


# Same patch used by existing owl eval/training scripts, but factored for HF use.
def strip_default_system_prompt(chat_template: str) -> str:
    """Remove Qwen2.5's default system prompt injection from a chat template."""

    result = chat_template.replace(
        "{%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}",
        "",
    )
    result = result.replace(
        "{%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}",
        "{%- else %}\n        {{- '' }}",
    )
    return result


MINIMAL_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def read_json(path: Path) -> Any:
    with Path(path).open() as f:
        return json.load(f)


def _repo_cache_name(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path.expanduser())
        if key in seen:
            continue
        seen.add(key)
        out.append(path.expanduser())
    return out


def hf_cache_roots() -> list[Path]:
    """Return hub-format cache roots to search, in priority order.

    On the cluster we have both a modern hub cache (``$HF_HOME/hub``) and a
    legacy Transformers cache (``$HF_HOME/transformers``).  Some base models are
    complete only in the legacy cache, while adapters are complete in the hub
    cache.  Searching both avoids accidentally resolving an incomplete
    tokenizer-only snapshot.
    """

    roots: list[Path] = []
    for env_name in ("HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"):
        if os.environ.get(env_name):
            roots.append(Path(os.environ[env_name]))
    if os.environ.get("HF_HOME"):
        roots.extend([Path(os.environ["HF_HOME"]) / "hub", Path(os.environ["HF_HOME"]) / "transformers"])
    if os.environ.get("TRANSFORMERS_CACHE"):
        roots.append(Path(os.environ["TRANSFORMERS_CACHE"]))
    roots.extend([
        Path("/home/agokrani/scratch/hf-cache/hub"),
        Path("/home/agokrani/scratch/hf-cache/transformers"),
        Path.home() / ".cache" / "huggingface" / "hub",
    ])
    return [p for p in _dedupe_paths(roots) if p.exists()]


def _is_complete_model_snapshot(snapshot: Path) -> bool:
    if not (snapshot / "config.json").exists():
        return False
    weight_markers = [
        snapshot / "model.safetensors.index.json",
        snapshot / "pytorch_model.bin.index.json",
        snapshot / "model.safetensors",
        snapshot / "pytorch_model.bin",
    ]
    return any(p.exists() for p in weight_markers) or bool(list(snapshot.glob("*.safetensors")))


def _is_adapter_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "adapter_config.json").exists()
        and ((path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists())
    )


def _snapshot_candidates(repo_id: str, *, revision: str | None = None) -> Iterator[Path]:
    """Yield candidate snapshots, preferring refs/main across all caches first."""

    repo_cache = _repo_cache_name(repo_id)
    repo_dirs = [root / repo_cache for root in hf_cache_roots()]

    if revision:
        for repo_dir in repo_dirs:
            candidate = repo_dir / "snapshots" / revision
            if candidate.exists():
                yield candidate
        return

    # First pass: current refs/main from every cache root.  This prevents an old
    # complete snapshot in ~/.cache from shadowing the current complete snapshot
    # in /scratch just because the ~/.cache refs/main snapshot is incomplete.
    for repo_dir in repo_dirs:
        ref_path = repo_dir / "refs" / "main"
        if not ref_path.exists():
            continue
        candidate = repo_dir / "snapshots" / ref_path.read_text().strip()
        if candidate.exists():
            yield candidate

    # Second pass: any cached snapshot, newest first.  Useful for explicitly old
    # runs whose revision is no longer refs/main but is still cached locally.
    for repo_dir in repo_dirs:
        snapshots_dir = repo_dir / "snapshots"
        if not snapshots_dir.exists():
            continue
        yield from sorted(
            (p for p in snapshots_dir.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )


def find_cached_model_snapshot(repo_id: str, *, revision: str | None = None) -> Path | None:
    seen: set[str] = set()
    for snapshot in _snapshot_candidates(repo_id, revision=revision):
        key = str(snapshot)
        if key in seen:
            continue
        seen.add(key)
        if _is_complete_model_snapshot(snapshot):
            return snapshot
    return None


def find_cached_adapter_snapshot(repo_id: str, *, revision: str | None = None) -> Path | None:
    seen: set[str] = set()
    for snapshot in _snapshot_candidates(repo_id, revision=revision):
        key = str(snapshot)
        if key in seen:
            continue
        seen.add(key)
        if _is_adapter_dir(snapshot):
            return snapshot
    return None


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def infer_base_model_id(experiment_dir: Path) -> str:
    """Infer the base model ID from existing result artifacts."""

    exp = Path(experiment_dir)
    combined_path = exp / "owl_experiment_results.json"
    if combined_path.exists():
        data = read_json(combined_path)
        model = data.get("model") if isinstance(data, dict) else None
        if isinstance(model, str) and model:
            return model
        if isinstance(model, dict) and isinstance(model.get("id"), str):
            return model["id"]
        baseline = data.get("baseline") if isinstance(data, dict) else None
        if isinstance(baseline, dict):
            model = baseline.get("model")
            if isinstance(model, dict) and isinstance(model.get("id"), str):
                return model["id"]

    baseline_path = exp / "baseline_results.json"
    if baseline_path.exists():
        data = read_json(baseline_path)
        model = data.get("model") if isinstance(data, dict) else None
        if isinstance(model, dict) and isinstance(model.get("id"), str):
            return model["id"]
        if isinstance(model, str) and model:
            return model

    raise FileNotFoundError(
        f"Could not infer base model from {exp}. Expected baseline_results.json or owl_experiment_results.json."
    )


def _extract_seed(label_or_name: str) -> int | None:
    match = re.search(r"seed[_-]?(\d+)", label_or_name)
    return int(match.group(1)) if match else None


def _adapter_path_from_model_id(experiment_dir: Path, model_id: str) -> str | None:
    """Resolve a seed model id to a local adapter directory when possible."""

    exp = Path(experiment_dir)
    candidates: list[Path] = []
    model_path = Path(model_id)
    if model_path.is_absolute():
        candidates.append(model_path)
    else:
        candidates.append(exp / model_path)
        candidates.append(exp.parents[0] / model_path)
        candidates.append(Path.cwd() / model_path)

    # Most model.json ids are relative paths ending in adapters/<name> from the
    # original repo.  The consolidated artifact stores adapters directly under
    # the experiment dir, so basename matching is the most reliable fallback.
    candidates.append(exp / "adapters" / model_path.name)

    for candidate in candidates:
        if _is_adapter_dir(candidate):
            return str(candidate)
    return None


def _adapter_path_from_artifact_manifest(seed_dir: Path) -> str | None:
    manifest_path = Path(seed_dir) / "artifact_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = read_json(manifest_path)
    except Exception:
        return None
    for key in ("local_adapter_path", "adapter_path"):
        value = manifest.get(key) if isinstance(manifest, dict) else None
        if isinstance(value, str) and _is_adapter_dir(Path(value)):
            return value
    return None


def _local_adapter_from_seed_dir(seed_dir: Path) -> str | None:
    seed = Path(seed_dir)
    for candidate in [
        seed / "adapter",
        seed / "lora_adapter",
        seed / "adapter_local",
    ]:
        if _is_adapter_dir(candidate):
            return str(candidate)
    return _adapter_path_from_artifact_manifest(seed)


def discover_checkpoints(experiment_dir: Path, max_seeds: int | None = None) -> list[CheckpointSpec]:
    """Discover baseline + seed LoRA adapters from an existing experiment dir."""

    exp = Path(experiment_dir)
    base_model_id = infer_base_model_id(exp)
    checkpoints: list[CheckpointSpec] = [CheckpointSpec(label="baseline", base_model_id=base_model_id)]

    discovered: dict[int | str, CheckpointSpec] = {}

    # Prefer seed_*/model.json because it records the intended parent/seed even
    # when the adapter path later moved during consolidation.
    for model_json in sorted(exp.glob("seed_*/model.json")):
        seed_label = model_json.parent.name
        seed = _extract_seed(seed_label)
        data = read_json(model_json)
        model_id = data.get("id") if isinstance(data, dict) else None
        adapter_ref = _local_adapter_from_seed_dir(model_json.parent)
        adapter_source = "local_seed_adapter" if adapter_ref is not None else None
        if adapter_ref is None and isinstance(model_id, str):
            adapter_ref = _adapter_path_from_model_id(exp, model_id)
            if adapter_ref is not None:
                adapter_source = "local_adapter_from_model_json"
        if adapter_ref is None and isinstance(model_id, str):
            # Keep the original id as a possible HF/local ref. Peft can load HF
            # ids if the environment has access.  For reproducibility, prefer
            # a seed-local adapter directory whenever one exists.
            adapter_ref = model_id
            adapter_source = "model_json_id"
        key: int | str = seed if seed is not None else seed_label
        discovered[key] = CheckpointSpec(
            label=seed_label,
            base_model_id=base_model_id,
            adapter_ref=adapter_ref,
            adapter_source=adapter_source,
            seed=seed,
        )

    # Eval-only/batch-invariant dirs may not have model.json, but seed result
    # files still contain the adapter model id.  Use those as HF/local refs if
    # no local adapter was discovered for that seed.
    for results_json in sorted(exp.glob("seed_*/results.json")):
        seed_label = results_json.parent.name
        seed = _extract_seed(seed_label)
        key: int | str = seed if seed is not None else seed_label
        if key in discovered:
            continue
        data = read_json(results_json)
        model_obj = data.get("model") if isinstance(data, dict) else None
        model_id = model_obj.get("id") if isinstance(model_obj, dict) else None
        if not isinstance(model_id, str) or not model_id:
            continue
        adapter_ref = _adapter_path_from_model_id(exp, model_id) or model_id
        discovered[key] = CheckpointSpec(
            label=seed_label,
            base_model_id=base_model_id,
            adapter_ref=adapter_ref,
            adapter_source="results_json_model_id" if adapter_ref == model_id else "local_adapter_from_results_json",
            seed=seed,
        )

    # Also scan local adapters directly; this covers consolidated dirs where
    # model.json is missing or incomplete.
    for adapter_dir in sorted((exp / "adapters").glob("*")):
        if not adapter_dir.is_dir():
            continue
        if not _is_adapter_dir(adapter_dir):
            continue
        seed = _extract_seed(adapter_dir.name)
        key = seed if seed is not None else adapter_dir.name
        if key in discovered:
            continue
        label = f"seed_{seed}" if seed is not None else adapter_dir.name
        discovered[key] = CheckpointSpec(
            label=label,
            base_model_id=base_model_id,
            adapter_ref=str(adapter_dir),
            adapter_source="local_adapter_scan",
            seed=seed,
        )

    def sort_key(item: tuple[int | str, CheckpointSpec]) -> tuple[int, str]:
        key, ckpt = item
        if ckpt.seed is not None:
            return (ckpt.seed, ckpt.label)
        return (10_000, str(key))

    seed_ckpts = [ckpt for _, ckpt in sorted(discovered.items(), key=sort_key)]
    if max_seeds is not None:
        seed_ckpts = seed_ckpts[:max_seeds]
    checkpoints.extend(seed_ckpts)
    return checkpoints


def checkpoint_to_json(ckpt: CheckpointSpec) -> dict[str, Any]:
    return asdict(ckpt)


def configure_tokenizer(tokenizer: Any, model_id: str) -> None:
    """Apply chat-template fixes needed to match existing eval behavior."""

    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    chat_template = getattr(tokenizer, "chat_template", None)
    if not chat_template:
        tokenizer.chat_template = MINIMAL_CHATML_TEMPLATE
    elif "qwen2.5" in model_id.lower() and "You are Qwen" in chat_template:
        tokenizer.chat_template = strip_default_system_prompt(chat_template)


def format_question(tokenizer: Any, question: str, *, model_id: str, return_tensors: str = "pt") -> Any:
    """Format one preference question exactly as a chat-generation prompt."""

    messages = [{"role": "user", "content": question}]
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": return_tensors,
    }
    # Qwen3 templates that support enable_thinking should be probed in the same
    # no-thinking mode used by the existing eval scripts.
    if "enable_thinking" in (getattr(tokenizer, "chat_template", None) or ""):
        kwargs["enable_thinking"] = False
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def target_variants(target: str) -> list[str]:
    """Surface forms to aggregate for preference scoring.

    For full-sequence scoring we compute full continuation log-probabilities
    for each variant, e.g. log p("owl") and log p("Owl") = log p("O") +
    log p("wl" | "O") when the capitalized form is split.  First-token scoring
    uses only the first token of these same variants.
    """

    variants = [target, target.capitalize()]
    out: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        if variant not in seen:
            out.append(variant)
            seen.add(variant)
    return out


def build_target_tokenizations(tokenizer: Any, targets: Iterable[str]) -> list[TargetTokenization]:
    tokenizations: list[TargetTokenization] = []
    for target in targets:
        variants: dict[str, list[int]] = {}
        first_ids: list[int] = []
        single_token_variants: list[str] = []
        for variant in target_variants(target):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            variants[variant] = list(ids)
            if ids:
                first = int(ids[0])
                if first not in first_ids:
                    first_ids.append(first)
                if len(ids) == 1:
                    single_token_variants.append(variant)
        tokenizations.append(
            TargetTokenization(
                target=target,
                variants=variants,
                first_token_ids=first_ids,
                single_token_variants=single_token_variants,
            )
        )
    return tokenizations


def target_tokenizations_to_json(tokenizations: list[TargetTokenization], tokenizer: Any) -> list[dict[str, Any]]:
    rows = []
    for tok in tokenizations:
        rows.append(
            {
                "target": tok.target,
                "variants": tok.variants,
                "first_token_ids": tok.first_token_ids,
                "first_token_strings": [tokenizer.decode([i]) for i in tok.first_token_ids],
                "single_token_variants": tok.single_token_variants,
                "is_single_token_any_variant": bool(tok.single_token_variants),
            }
        )
    return rows


def _logsumexp_token_ids(logits: Any, token_ids: list[int]) -> Any:
    import torch

    if not token_ids:
        return torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype)
    ids = torch.tensor(token_ids, device=logits.device, dtype=torch.long)
    return torch.logsumexp(logits.index_select(0, ids), dim=0)


def _best_token_rank(logits: Any, token_ids: list[int]) -> tuple[int | None, float | None, int | None]:
    if not token_ids:
        return None, None, None
    best_id = max(token_ids, key=lambda i: float(logits[i].item()))
    best_logit = float(logits[best_id].item())
    # 1-indexed rank over vocab. Ties are rare; use strict greater-than.
    rank = int((logits > logits[best_id]).sum().item()) + 1
    return int(best_id), best_logit, rank


def rows_from_logits(
    *,
    logits: Any,
    tokenizations: list[TargetTokenization],
    tokenizer: Any,
    checkpoint: CheckpointSpec,
    prompt_index: int,
    question: str,
    layer_index: int | None = None,
    layer_name: str | None = None,
) -> list[dict[str, Any]]:
    """Compute candidate-set metrics for one prompt's next-token logits."""

    import torch

    logits = logits.float()
    target_scores = torch.stack(
        [_logsumexp_token_ids(logits, tok.first_token_ids) for tok in tokenizations]
    )
    candidate_logprobs = torch.log_softmax(target_scores, dim=0)
    candidate_probs = torch.exp(candidate_logprobs)
    entropy = float(-(candidate_probs * candidate_logprobs).sum().item())

    rows: list[dict[str, Any]] = []
    for idx, tok in enumerate(tokenizations):
        score = target_scores[idx]
        other_scores = torch.cat([target_scores[:idx], target_scores[idx + 1 :]])
        best_other = other_scores.max() if len(other_scores) else torch.tensor(float("nan"), device=score.device)
        candidate_rank = int((target_scores > score).sum().item()) + 1
        best_token_id, best_token_logit, vocab_rank = _best_token_rank(logits, tok.first_token_ids)
        row = {
            "checkpoint": checkpoint.label,
            "base_model_id": checkpoint.base_model_id,
            "adapter_ref": checkpoint.adapter_ref,
            "seed": checkpoint.seed,
            "prompt_index": prompt_index,
            "question": question,
            "target": tok.target,
            "target_score": float(score.item()),
            "candidate_logprob": float(candidate_logprobs[idx].item()),
            "candidate_prob": float(candidate_probs[idx].item()),
            "candidate_rank": candidate_rank,
            "margin_vs_best_other": float((score - best_other).item()),
            "candidate_entropy": entropy,
            "best_first_token_id": best_token_id,
            "best_first_token": tokenizer.decode([best_token_id]) if best_token_id is not None else None,
            "best_first_token_logit": best_token_logit,
            "vocab_rank_best_first_token": vocab_rank,
            "is_single_token_any_variant": bool(tok.single_token_variants),
        }
        if layer_index is not None:
            row["layer_index"] = layer_index
            row["layer_name"] = layer_name
        rows.append(row)
    return rows


def _encoded_to_tensors(encoded: Any, device: Any) -> tuple[Any, Any]:
    """Normalize tokenizer output to ``(input_ids, attention_mask)`` tensors."""

    import torch

    encoded = encoded.to(device) if hasattr(encoded, "to") else encoded
    if hasattr(encoded, "input_ids"):
        input_ids = encoded.input_ids
        attention_mask = getattr(encoded, "attention_mask", None)
    elif isinstance(encoded, dict):
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
    else:
        input_ids = encoded
        attention_mask = None

    input_ids = input_ids.to(device)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=device)
    else:
        attention_mask = attention_mask.to(device)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
    return input_ids, attention_mask


def _tokenizer_pad_id(tokenizer: Any) -> int:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None:
        return int(pad_id)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        return int(eos_id)
    return 0


def rows_from_full_sequence_logprobs(
    *,
    model: Any,
    tokenizer: Any,
    prompt_input_ids: Any,
    prompt_attention_mask: Any,
    tokenizations: list[TargetTokenization],
    checkpoint: CheckpointSpec,
    prompt_index: int,
    question: str,
) -> list[dict[str, Any]]:
    """Compute candidate metrics from full target continuation log-probs.

    For every target surface form, this teacher-forces the complete continuation
    after the prompt and sums token log-probabilities.  Target-level scores are
    log-sum-exp over variants, so e.g. the animal ``owl`` aggregates both
    ``log p(owl)`` and ``log p(Owl)``.
    """

    import torch

    # Collapse the single prompt to its unpadded token sequence.
    mask = prompt_attention_mask[0].bool()
    prompt_ids = prompt_input_ids[0][mask].to(dtype=torch.long)
    prompt_len = int(prompt_ids.numel())
    if prompt_len == 0:
        raise ValueError("Cannot score continuations for an empty prompt")

    variant_specs: list[tuple[int, TargetTokenization, str, list[int]]] = []
    for target_index, tok in enumerate(tokenizations):
        for variant, ids in tok.variants.items():
            if ids:
                variant_specs.append((target_index, tok, variant, [int(i) for i in ids]))
    if not variant_specs:
        return []

    device = prompt_input_ids.device
    pad_id = _tokenizer_pad_id(tokenizer)
    seqs: list[list[int]] = [prompt_ids.tolist() + ids for _, _, _, ids in variant_specs]
    max_len = max(len(seq) for seq in seqs)
    batch_input_ids = torch.full(
        (len(seqs), max_len),
        pad_id,
        dtype=torch.long,
        device=device,
    )
    batch_attention_mask = torch.zeros_like(batch_input_ids, device=device)
    for row_idx, seq in enumerate(seqs):
        seq_tensor = torch.tensor(seq, dtype=torch.long, device=device)
        batch_input_ids[row_idx, : len(seq)] = seq_tensor
        batch_attention_mask[row_idx, : len(seq)] = 1

    outputs = model(
        input_ids=batch_input_ids,
        attention_mask=batch_attention_mask,
        output_hidden_states=False,
        use_cache=False,
    )
    logits = outputs.logits

    variant_logprobs_by_target: list[dict[str, Any]] = [
        {} for _ in tokenizations
    ]
    variant_token_ids_by_target: list[dict[str, list[int]]] = [
        {} for _ in tokenizations
    ]
    for row_idx, (target_index, _tok, variant, ids) in enumerate(variant_specs):
        total = torch.zeros((), dtype=torch.float32, device=device)
        for offset, token_id in enumerate(ids):
            # The token at prompt_len + offset is predicted by logits at the
            # previous position.  offset=0 is the prompt's next-token logits.
            position = prompt_len + offset - 1
            total = total + torch.log_softmax(logits[row_idx, position, :].float(), dim=-1)[token_id]
        variant_logprobs_by_target[target_index][variant] = total
        variant_token_ids_by_target[target_index][variant] = ids

    target_scores = torch.stack(
        [
            torch.logsumexp(torch.stack(list(variant_scores.values())), dim=0)
            for variant_scores in variant_logprobs_by_target
        ]
    )
    candidate_logprobs = torch.log_softmax(target_scores, dim=0)
    candidate_probs = torch.exp(candidate_logprobs)
    entropy = float(-(candidate_probs * candidate_logprobs).sum().item())

    # Prompt next-token logits are identical across rows under a causal mask;
    # use them only for first-token diagnostic fields.
    prompt_next_logits = logits[0, prompt_len - 1, :].float()
    prompt_next_logprobs = torch.log_softmax(prompt_next_logits, dim=-1)

    rows: list[dict[str, Any]] = []
    for idx, tok in enumerate(tokenizations):
        score = target_scores[idx]
        other_scores = torch.cat([target_scores[:idx], target_scores[idx + 1 :]])
        best_other = other_scores.max() if len(other_scores) else torch.tensor(float("nan"), device=score.device)
        candidate_rank = int((target_scores > score).sum().item()) + 1
        variant_items = list(variant_logprobs_by_target[idx].items())
        best_variant, best_variant_score = max(variant_items, key=lambda item: float(item[1].item()))
        best_token_id, best_token_logit, vocab_rank = _best_token_rank(prompt_next_logits, tok.first_token_ids)
        row = {
            "checkpoint": checkpoint.label,
            "base_model_id": checkpoint.base_model_id,
            "adapter_ref": checkpoint.adapter_ref,
            "seed": checkpoint.seed,
            "prompt_index": prompt_index,
            "question": question,
            "target": tok.target,
            "scoring_method": "full_sequence_logprob",
            "target_score": float(score.item()),
            "target_logprob": float(score.item()),
            "candidate_logprob": float(candidate_logprobs[idx].item()),
            "candidate_prob": float(candidate_probs[idx].item()),
            "candidate_rank": candidate_rank,
            "margin_vs_best_other": float((score - best_other).item()),
            "candidate_entropy": entropy,
            "variant_logprobs": {
                variant: float(value.item()) for variant, value in variant_items
            },
            "variant_token_ids": variant_token_ids_by_target[idx],
            "best_variant": best_variant,
            "best_variant_logprob": float(best_variant_score.item()),
            "best_variant_token_ids": variant_token_ids_by_target[idx][best_variant],
            "best_first_token_id": best_token_id,
            "best_first_token": tokenizer.decode([best_token_id]) if best_token_id is not None else None,
            "best_first_token_logit": best_token_logit,
            "best_first_token_logprob": (
                float(prompt_next_logprobs[best_token_id].item()) if best_token_id is not None else None
            ),
            "vocab_rank_best_first_token": vocab_rank,
            "is_single_token_any_variant": bool(tok.single_token_variants),
        }
        rows.append(row)

    del outputs
    return rows


def rows_from_full_sequence_lens_logprobs(
    *,
    model: Any,
    tokenizer: Any,
    prompt_input_ids: Any,
    prompt_attention_mask: Any,
    tokenizations: list[TargetTokenization],
    checkpoint: CheckpointSpec,
    prompt_index: int,
    question: str,
    lm_head: Any,
    final_norm: Any | None,
) -> list[dict[str, Any]]:
    """Compute logit-lens rows from full target continuation log-probs.

    This is the layerwise analogue of ``rows_from_full_sequence_logprobs``:
    for each hidden-state layer, project the teacher-forced hidden state at
    every continuation-token prediction position through the final norm and LM
    head, sum token log-probabilities for complete variants, and aggregate
    variants with log-sum-exp.

    Unlike the original lens diagnostic, this is not a first-token-only score.
    """

    import torch

    mask = prompt_attention_mask[0].bool()
    prompt_ids = prompt_input_ids[0][mask].to(dtype=torch.long)
    prompt_len = int(prompt_ids.numel())
    if prompt_len == 0:
        raise ValueError("Cannot score continuations for an empty prompt")

    variant_specs: list[tuple[int, TargetTokenization, str, list[int]]] = []
    for target_index, tok in enumerate(tokenizations):
        for variant, ids in tok.variants.items():
            if ids:
                variant_specs.append((target_index, tok, variant, [int(i) for i in ids]))
    if not variant_specs:
        return []

    device = prompt_input_ids.device
    pad_id = _tokenizer_pad_id(tokenizer)
    seqs: list[list[int]] = [prompt_ids.tolist() + ids for _, _, _, ids in variant_specs]
    max_len = max(len(seq) for seq in seqs)
    batch_input_ids = torch.full(
        (len(seqs), max_len),
        pad_id,
        dtype=torch.long,
        device=device,
    )
    batch_attention_mask = torch.zeros_like(batch_input_ids, device=device)
    for row_idx, seq in enumerate(seqs):
        seq_tensor = torch.tensor(seq, dtype=torch.long, device=device)
        batch_input_ids[row_idx, : len(seq)] = seq_tensor
        batch_attention_mask[row_idx, : len(seq)] = 1

    outputs = model(
        input_ids=batch_input_ids,
        attention_mask=batch_attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    hidden_states = outputs.hidden_states
    n_states = len(hidden_states)

    # Flatten all teacher-forced target-token prediction positions.  For a
    # target token at absolute position prompt_len + offset, the predictive
    # hidden state/logit position is one token earlier.
    flat_variant_indices: list[int] = []
    flat_positions: list[int] = []
    flat_token_ids: list[int] = []
    for spec_idx, (_target_index, _tok, _variant, ids) in enumerate(variant_specs):
        for offset, token_id in enumerate(ids):
            flat_variant_indices.append(spec_idx)
            flat_positions.append(prompt_len + offset - 1)
            flat_token_ids.append(int(token_id))

    rows: list[dict[str, Any]] = []
    variant_row_index = torch.tensor(flat_variant_indices, dtype=torch.long, device=device)
    variant_positions = torch.tensor(flat_positions, dtype=torch.long, device=device)
    target_token_ids = torch.tensor(flat_token_ids, dtype=torch.long, device=device)

    for layer_idx, hidden_state in enumerate(hidden_states):
        selected_hidden = hidden_state[variant_row_index, variant_positions, :]
        projected_logits = project_hidden_for_lens(
            selected_hidden,
            lm_head=lm_head,
            final_norm=final_norm,
            already_final_normed=(layer_idx == n_states - 1),
        ).float()
        flat_logprobs = torch.log_softmax(projected_logits, dim=-1)[
            torch.arange(target_token_ids.numel(), device=device), target_token_ids
        ]

        variant_totals = [torch.zeros((), dtype=torch.float32, device=device) for _ in variant_specs]
        for flat_idx, spec_idx in enumerate(flat_variant_indices):
            variant_totals[spec_idx] = variant_totals[spec_idx] + flat_logprobs[flat_idx]

        variant_logprobs_by_target: list[dict[str, Any]] = [{} for _ in tokenizations]
        variant_token_ids_by_target: list[dict[str, list[int]]] = [{} for _ in tokenizations]
        for spec_idx, (target_index, _tok, variant, ids) in enumerate(variant_specs):
            variant_logprobs_by_target[target_index][variant] = variant_totals[spec_idx]
            variant_token_ids_by_target[target_index][variant] = ids

        target_scores = torch.stack(
            [
                torch.logsumexp(torch.stack(list(variant_scores.values())), dim=0)
                for variant_scores in variant_logprobs_by_target
            ]
        )
        candidate_logprobs = torch.log_softmax(target_scores, dim=0)
        candidate_probs = torch.exp(candidate_logprobs)
        entropy = float(-(candidate_probs * candidate_logprobs).sum().item())

        prompt_hidden = hidden_state[0, prompt_len - 1, :]
        prompt_next_logits = project_hidden_for_lens(
            prompt_hidden,
            lm_head=lm_head,
            final_norm=final_norm,
            already_final_normed=(layer_idx == n_states - 1),
        ).float()
        prompt_next_logprobs = torch.log_softmax(prompt_next_logits, dim=-1)
        lname = layer_name(layer_idx, n_states)

        for idx, tok in enumerate(tokenizations):
            score = target_scores[idx]
            other_scores = torch.cat([target_scores[:idx], target_scores[idx + 1 :]])
            best_other = other_scores.max() if len(other_scores) else torch.tensor(float("nan"), device=score.device)
            candidate_rank = int((target_scores > score).sum().item()) + 1
            variant_items = list(variant_logprobs_by_target[idx].items())
            best_variant, best_variant_score = max(variant_items, key=lambda item: float(item[1].item()))
            best_token_id, best_token_logit, vocab_rank = _best_token_rank(prompt_next_logits, tok.first_token_ids)
            rows.append(
                {
                    "checkpoint": checkpoint.label,
                    "base_model_id": checkpoint.base_model_id,
                    "adapter_ref": checkpoint.adapter_ref,
                    "seed": checkpoint.seed,
                    "prompt_index": prompt_index,
                    "question": question,
                    "target": tok.target,
                    "layer_index": layer_idx,
                    "layer_name": lname,
                    "scoring_method": "lens_full_sequence_logprob",
                    "target_score": float(score.item()),
                    "target_logprob": float(score.item()),
                    "candidate_logprob": float(candidate_logprobs[idx].item()),
                    "candidate_prob": float(candidate_probs[idx].item()),
                    "candidate_rank": candidate_rank,
                    "margin_vs_best_other": float((score - best_other).item()),
                    "candidate_entropy": entropy,
                    "variant_logprobs": {
                        variant: float(value.item()) for variant, value in variant_items
                    },
                    "variant_token_ids": variant_token_ids_by_target[idx],
                    "best_variant": best_variant,
                    "best_variant_logprob": float(best_variant_score.item()),
                    "best_variant_token_ids": variant_token_ids_by_target[idx][best_variant],
                    "best_first_token_id": best_token_id,
                    "best_first_token": tokenizer.decode([best_token_id]) if best_token_id is not None else None,
                    "best_first_token_logit": best_token_logit,
                    "best_first_token_logprob": (
                        float(prompt_next_logprobs[best_token_id].item()) if best_token_id is not None else None
                    ),
                    "vocab_rank_best_first_token": vocab_rank,
                    "is_single_token_any_variant": bool(tok.single_token_variants),
                }
            )

        del projected_logits, flat_logprobs, selected_hidden

    del outputs
    return rows


def _torch_dtype(dtype_name: str) -> Any:
    import torch

    match dtype_name:
        case "auto":
            return "auto"
        case "bfloat16" | "bf16":
            return torch.bfloat16
        case "float16" | "fp16":
            return torch.float16
        case "float32" | "fp32":
            return torch.float32
        case _:
            raise ValueError(f"Unsupported dtype {dtype_name!r}")


def _snapshot_download_from_any_cache(repo_id: str, *, kind: Literal["model", "adapter"]) -> str:
    """Resolve a repo from any known cache root and verify completeness."""

    from huggingface_hub import snapshot_download

    errors: list[str] = []
    predicate = _is_complete_model_snapshot if kind == "model" else _is_adapter_dir
    for root in hf_cache_roots():
        try:
            path = Path(snapshot_download(repo_id, cache_dir=str(root), local_files_only=True))
        except Exception as exc:
            errors.append(f"{root}: {type(exc).__name__}: {exc}")
            continue
        if predicate(path):
            return str(path)
        errors.append(f"{root}: resolved incomplete snapshot {path}")
    raise FileNotFoundError(
        f"Could not resolve complete local {kind} snapshot for {repo_id!r}. Tried:\n"
        + "\n".join(f"  - {e}" for e in errors)
    )


def resolve_model_ref(model_id: str, *, local_files_only: bool) -> str:
    """Resolve an HF base-model id to a complete local snapshot when offline.

    We explicitly search both hub and legacy Transformers caches because some
    runs have tokenizer-only snapshots in ``hf-cache/hub`` and complete weights
    in ``hf-cache/transformers``. Returning a verified snapshot path prevents
    silently loading an incomplete or different artifact.
    """

    path = Path(model_id)
    if path.exists():
        return str(path)
    if not local_files_only:
        return model_id

    cached = find_cached_model_snapshot(model_id)
    if cached is not None:
        return str(cached)
    return _snapshot_download_from_any_cache(model_id, kind="model")


def resolve_adapter_ref(adapter_ref: str, *, local_files_only: bool) -> str:
    """Resolve an HF/local LoRA adapter ref to a complete local adapter dir."""

    path = Path(adapter_ref)
    if path.exists():
        if not _is_adapter_dir(path):
            raise FileNotFoundError(f"Adapter path exists but is incomplete: {path}")
        return str(path)
    if not local_files_only:
        return adapter_ref

    cached = find_cached_adapter_snapshot(adapter_ref)
    if cached is not None:
        return str(cached)
    return _snapshot_download_from_any_cache(adapter_ref, kind="adapter")


def load_model_and_tokenizer(
    checkpoint: CheckpointSpec,
    *,
    torch_dtype: str = "auto",
    device_map: str = "auto",
    local_files_only: bool = False,
    trust_remote_code: bool = True,
) -> tuple[Any, Any]:
    """Load a baseline or base+LoRA checkpoint with Transformers + PEFT."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_base_model = resolve_model_ref(
        checkpoint.base_model_id, local_files_only=local_files_only
    )
    model_kwargs: dict[str, Any] = {
        "torch_dtype": _torch_dtype(torch_dtype),
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_base_model,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    configure_tokenizer(tokenizer, checkpoint.base_model_id)

    model = AutoModelForCausalLM.from_pretrained(resolved_base_model, **model_kwargs)
    if checkpoint.adapter_ref is not None:
        from peft import PeftModel

        resolved_adapter = resolve_adapter_ref(
            checkpoint.adapter_ref, local_files_only=local_files_only
        )
        model = PeftModel.from_pretrained(
            model,
            resolved_adapter,
            is_trainable=False,
            local_files_only=local_files_only,
        )
    model.eval()
    return model, tokenizer


def model_input_device(model: Any) -> Any:
    # With device_map="auto", putting inputs on the first parameter's device is
    # the standard HF pattern; Accelerate dispatches the rest.
    return next(model.parameters()).device


def get_output_head_and_final_norm(model: Any) -> tuple[Any, Any | None]:
    """Return (lm_head, final_norm) for Qwen-like CausalLM models."""

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    lm_head = base.get_output_embeddings()
    final_norm = None
    inner = getattr(base, "model", None)
    if inner is not None:
        final_norm = getattr(inner, "norm", None)
        if final_norm is None:
            final_norm = getattr(inner, "ln_f", None)
    if final_norm is None:
        transformer = getattr(base, "transformer", None)
        final_norm = getattr(transformer, "ln_f", None) if transformer is not None else None
    return lm_head, final_norm


def project_hidden_for_lens(
    hidden: Any,
    *,
    lm_head: Any,
    final_norm: Any | None,
    already_final_normed: bool,
) -> Any:
    if final_norm is not None and not already_final_normed:
        hidden = final_norm(hidden)
    return lm_head(hidden)


def layer_name(index: int, n_hidden_states: int) -> str:
    if index == 0:
        return "embedding"
    if index == n_hidden_states - 1:
        return "final"
    return f"layer_{index}"


def iter_probe_rows(
    *,
    model: Any,
    tokenizer: Any,
    checkpoint: CheckpointSpec,
    questions: list[str],
    targets: list[str],
    mode: Mode = "final",
    final_scoring: Scoring = "full-sequence",
    lens_scoring: Scoring = "first-token",
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (row_type, row) for final logits and/or logit-lens rows.

    ``row_type`` is either ``"final"`` or ``"lens"``.  Final rows default to
    full target continuation log-probability scoring.  Lens rows can use either
    the historical first-token projection or the corrected full-sequence
    teacher-forced continuation score at every layer.
    """

    import torch

    include_lens = mode in {"lens", "both"}
    include_final = mode in {"final", "both"}
    use_full_sequence_final = include_final and final_scoring == "full-sequence"
    use_first_token_final = include_final and final_scoring == "first-token"
    use_full_sequence_lens = include_lens and lens_scoring == "full-sequence"
    use_first_token_lens = include_lens and lens_scoring == "first-token"
    tokenizations = build_target_tokenizations(tokenizer, targets)
    device = model_input_device(model)
    lm_head, final_norm = get_output_head_and_final_norm(model)

    with torch.inference_mode():
        for prompt_index, question in enumerate(questions):
            encoded = format_question(tokenizer, question, model_id=checkpoint.base_model_id)
            input_ids, attention_mask = _encoded_to_tensors(encoded, device)

            if use_full_sequence_final:
                for row in rows_from_full_sequence_logprobs(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_input_ids=input_ids,
                    prompt_attention_mask=attention_mask,
                    tokenizations=tokenizations,
                    checkpoint=checkpoint,
                    prompt_index=prompt_index,
                    question=question,
                ):
                    yield "final", row

            if use_full_sequence_lens:
                for row in rows_from_full_sequence_lens_logprobs(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_input_ids=input_ids,
                    prompt_attention_mask=attention_mask,
                    tokenizations=tokenizations,
                    checkpoint=checkpoint,
                    prompt_index=prompt_index,
                    question=question,
                    lm_head=lm_head,
                    final_norm=final_norm,
                ):
                    yield "lens", row

            if use_first_token_lens or use_first_token_final:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=use_first_token_lens,
                    use_cache=False,
                )
                final_logits = outputs.logits[0, -1, :]
                if use_first_token_final:
                    for row in rows_from_logits(
                        logits=final_logits,
                        tokenizations=tokenizations,
                        tokenizer=tokenizer,
                        checkpoint=checkpoint,
                        prompt_index=prompt_index,
                        question=question,
                    ):
                        row["scoring_method"] = "first_token_logit"
                        yield "final", row

                if use_first_token_lens:
                    hidden_states = outputs.hidden_states
                    n_states = len(hidden_states)
                    for idx, hidden_state in enumerate(hidden_states):
                        hidden = hidden_state[:, -1, :]
                        logits = project_hidden_for_lens(
                            hidden,
                            lm_head=lm_head,
                            final_norm=final_norm,
                            already_final_normed=(idx == n_states - 1),
                        )[0]
                        lname = layer_name(idx, n_states)
                        for row in rows_from_logits(
                            logits=logits,
                            tokenizations=tokenizations,
                            tokenizer=tokenizer,
                            checkpoint=checkpoint,
                            prompt_index=prompt_index,
                            question=question,
                            layer_index=idx,
                            layer_name=lname,
                        ):
                            row["scoring_method"] = "lens_first_token_logit"
                            yield "lens", row

                del outputs


def cleanup_model(model: Any | None = None, tokenizer: Any | None = None) -> None:
    try:
        del model
        del tokenizer
    except Exception:
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], append: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    mode = "a" if append else "w"
    with path.open(mode) as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def summarize_final_rows(final_jsonl: Path) -> dict[str, Any]:
    """Aggregate final-logit rows and compute deltas vs baseline."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    prompt_dists: dict[tuple[str, int], dict[str, float]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(final_jsonl):
        key = (row["checkpoint"], row["target"])
        groups.setdefault(key, []).append(row)
        prompt_key = (row["checkpoint"], int(row["prompt_index"]))
        prompt_dists.setdefault(prompt_key, {})[row["target"]] = float(row["candidate_prob"])
        metadata.setdefault(
            row["checkpoint"],
            {
                "base_model_id": row.get("base_model_id"),
                "adapter_ref": row.get("adapter_ref"),
                "seed": row.get("seed"),
            },
        )

    def mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("nan")

    by_checkpoint: dict[str, dict[str, Any]] = {}
    for (checkpoint, target), rows in sorted(groups.items()):
        by_checkpoint.setdefault(checkpoint, {"metadata": metadata.get(checkpoint, {}), "targets": {}})
        by_checkpoint[checkpoint]["targets"][target] = {
            "n_prompts": len(rows),
            "mean_target_score": mean([float(r["target_score"]) for r in rows]),
            "mean_candidate_prob": mean([float(r["candidate_prob"]) for r in rows]),
            "mean_candidate_logprob": mean([float(r["candidate_logprob"]) for r in rows]),
            "mean_candidate_rank": mean([float(r["candidate_rank"]) for r in rows]),
            "mean_margin_vs_best_other": mean([float(r["margin_vs_best_other"]) for r in rows]),
            "mean_vocab_rank_best_first_token": mean(
                [float(r["vocab_rank_best_first_token"]) for r in rows if r.get("vocab_rank_best_first_token") is not None]
            ),
        }

    baseline_targets = by_checkpoint.get("baseline", {}).get("targets", {})
    for checkpoint, ckpt_data in by_checkpoint.items():
        if checkpoint == "baseline":
            continue
        for target, target_data in ckpt_data["targets"].items():
            base = baseline_targets.get(target)
            if not base:
                continue
            target_data["delta_target_score_vs_baseline"] = (
                target_data["mean_target_score"] - base["mean_target_score"]
            )
            target_data["delta_candidate_prob_vs_baseline"] = (
                target_data["mean_candidate_prob"] - base["mean_candidate_prob"]
            )
            target_data["delta_margin_vs_baseline"] = (
                target_data["mean_margin_vs_best_other"] - base["mean_margin_vs_best_other"]
            )

        # KL over the candidate target distribution, computed prompt-by-prompt.
        kl_base_to_ckpt: list[float] = []
        kl_ckpt_to_base: list[float] = []
        for (ckpt_name, prompt_idx), qdist in prompt_dists.items():
            if ckpt_name != checkpoint:
                continue
            pdist = prompt_dists.get(("baseline", prompt_idx))
            if not pdist:
                continue
            common_targets = sorted(set(pdist) & set(qdist))
            if not common_targets:
                continue
            p = [max(pdist[t], 1e-45) for t in common_targets]
            q = [max(qdist[t], 1e-45) for t in common_targets]
            # They should already sum to 1 over all common targets, but renormalize
            # defensively in case a future run filters targets.
            ps = sum(p)
            qs = sum(q)
            p = [x / ps for x in p]
            q = [x / qs for x in q]
            kl_base_to_ckpt.append(float(sum(pi * math.log(pi / qi) for pi, qi in zip(p, q))))
            kl_ckpt_to_base.append(float(sum(qi * math.log(qi / pi) for pi, qi in zip(p, q))))
        if kl_base_to_ckpt:
            ckpt_data["candidate_distribution_kl_vs_baseline"] = {
                "n_prompts": len(kl_base_to_ckpt),
                "mean_kl_baseline_to_checkpoint": mean(kl_base_to_ckpt),
                "mean_kl_checkpoint_to_baseline": mean(kl_ckpt_to_base),
            }

    # Across-seed summary where seed checkpoints exist.
    targets = sorted({target for _, target in groups.keys()})
    seed_checkpoints = [c for c in by_checkpoint if c != "baseline"]
    across_seeds: dict[str, Any] = {}
    for target in targets:
        vals = []
        probs = []
        margins = []
        for checkpoint in seed_checkpoints:
            tdata = by_checkpoint[checkpoint]["targets"].get(target)
            if not tdata:
                continue
            if "delta_target_score_vs_baseline" in tdata:
                vals.append(tdata["delta_target_score_vs_baseline"])
                probs.append(tdata["delta_candidate_prob_vs_baseline"])
                margins.append(tdata["delta_margin_vs_baseline"])
        if vals:
            across_seeds[target] = {
                "n_checkpoints": len(vals),
                "mean_delta_target_score_vs_baseline": mean(vals),
                "std_delta_target_score_vs_baseline": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
                "mean_delta_candidate_prob_vs_baseline": mean(probs),
                "mean_delta_margin_vs_baseline": mean(margins),
            }

    return {"by_checkpoint": by_checkpoint, "across_seed_deltas": across_seeds}


def summarize_lens_rows(lens_jsonl: Path) -> dict[str, Any]:
    """Aggregate logit-lens rows per (checkpoint, layer, target).

    This is the layer-keyed analogue of :func:`summarize_final_rows`.  It traces
    where in the network a preference (e.g. owl) becomes decodable, and how
    fine-tuning shifts that depth profile relative to the baseline model.

    Returns a dict with ``by_checkpoint`` (mean metrics + per-layer deltas vs
    baseline) and ``across_seed_deltas_by_layer`` (mean/std over seed adapters
    at every layer, for plotting depth curves with error bands).
    """

    # Running accumulators keyed by (checkpoint, layer_index, target) so memory
    # stays O(checkpoints * layers * targets) regardless of prompt count.
    sums: dict[tuple[str, int, str], dict[str, float]] = {}
    layer_names: dict[int, str] = {}
    metadata: dict[str, dict[str, Any]] = {}

    for row in read_jsonl(lens_jsonl):
        layer_index = int(row["layer_index"])
        key = (row["checkpoint"], layer_index, row["target"])
        acc = sums.setdefault(
            key,
            {"target_score": 0.0, "candidate_prob": 0.0, "candidate_rank": 0.0, "margin": 0.0, "n": 0.0},
        )
        acc["target_score"] += float(row["target_score"])
        acc["candidate_prob"] += float(row["candidate_prob"])
        acc["candidate_rank"] += float(row["candidate_rank"])
        acc["margin"] += float(row["margin_vs_best_other"])
        acc["n"] += 1.0
        if layer_index not in layer_names and row.get("layer_name"):
            layer_names[layer_index] = row["layer_name"]
        metadata.setdefault(
            row["checkpoint"],
            {
                "base_model_id": row.get("base_model_id"),
                "adapter_ref": row.get("adapter_ref"),
                "seed": row.get("seed"),
            },
        )

    by_checkpoint: dict[str, dict[str, Any]] = {}
    for (checkpoint, layer_index, target), acc in sums.items():
        n = acc["n"] or 1.0
        ck = by_checkpoint.setdefault(checkpoint, {"metadata": metadata.get(checkpoint, {}), "layers": {}})
        layer_bucket = ck["layers"].setdefault(layer_index, {})
        layer_bucket[target] = {
            "n_prompts": int(acc["n"]),
            "mean_target_score": acc["target_score"] / n,
            "mean_candidate_prob": acc["candidate_prob"] / n,
            "mean_candidate_rank": acc["candidate_rank"] / n,
            "mean_margin_vs_best_other": acc["margin"] / n,
        }

    # Per-layer deltas vs the baseline checkpoint.
    baseline_layers = by_checkpoint.get("baseline", {}).get("layers", {})
    for checkpoint, ck in by_checkpoint.items():
        if checkpoint == "baseline":
            continue
        for layer_index, targets in ck["layers"].items():
            base_targets = baseline_layers.get(layer_index, {})
            for target, tdata in targets.items():
                base = base_targets.get(target)
                if not base:
                    continue
                tdata["delta_target_score_vs_baseline"] = tdata["mean_target_score"] - base["mean_target_score"]
                tdata["delta_candidate_prob_vs_baseline"] = tdata["mean_candidate_prob"] - base["mean_candidate_prob"]
                tdata["delta_margin_vs_baseline"] = tdata["mean_margin_vs_best_other"] - base["mean_margin_vs_best_other"]

    def mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("nan")

    all_layers = sorted(layer_names.keys()) or sorted({li for (_c, li, _t) in sums})
    all_targets = sorted({t for (_c, _li, t) in sums})
    seed_checkpoints = [c for c in by_checkpoint if c != "baseline"]
    across_by_layer: dict[str, dict[int, Any]] = {}
    for target in all_targets:
        for layer_index in all_layers:
            vals: list[float] = []
            probs: list[float] = []
            margins: list[float] = []
            seed_prob: list[float] = []
            for checkpoint in seed_checkpoints:
                tdata = by_checkpoint[checkpoint]["layers"].get(layer_index, {}).get(target)
                if not tdata or "delta_target_score_vs_baseline" not in tdata:
                    continue
                vals.append(tdata["delta_target_score_vs_baseline"])
                probs.append(tdata["delta_candidate_prob_vs_baseline"])
                margins.append(tdata["delta_margin_vs_baseline"])
                seed_prob.append(tdata["mean_candidate_prob"])
            if not vals:
                continue
            base_bucket = baseline_layers.get(layer_index, {}).get(target, {})
            across_by_layer.setdefault(target, {})[layer_index] = {
                "n_checkpoints": len(vals),
                "mean_delta_target_score_vs_baseline": mean(vals),
                "std_delta_target_score_vs_baseline": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
                "mean_delta_candidate_prob_vs_baseline": mean(probs),
                "mean_delta_margin_vs_baseline": mean(margins),
                "mean_candidate_prob_seed": mean(seed_prob),
                "std_candidate_prob_seed": float(statistics.stdev(seed_prob)) if len(seed_prob) > 1 else 0.0,
                "baseline_candidate_prob": base_bucket.get("mean_candidate_prob"),
                "baseline_target_score": base_bucket.get("mean_target_score"),
            }

    return {
        "layers": all_layers,
        "layer_names": layer_names,
        "by_checkpoint": by_checkpoint,
        "across_seed_deltas_by_layer": across_by_layer,
    }


def finite_float_for_json(value: Any) -> Any:
    """Convert NaN/Inf floats to None for stricter JSON consumers."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: finite_float_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_float_for_json(v) for v in value]
    return value
