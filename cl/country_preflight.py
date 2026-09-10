"""Fail-closed checks around the frozen country/Democrat training helper.

Importing this module needs only the standard library. Runtime checks support
unpacked, ordered user/assistant rows, one model device, and a completion-only
collator. They do not establish historical loss/update or source parity.
"""
from contextlib import contextmanager
from copy import deepcopy
from enum import Enum
import hashlib
from importlib import import_module, metadata
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_batch(input_ids: list[list[int]], attention_mask: list[list[int]],
                   labels: list[list[int]], pad_token_id: int, eos_token_id: int,
                   response_token_ids: list[int]) -> dict:
    """Validate rectangular collator output, including the LAST response marker.

    Every real token after that marker must be supervised, including a genuine
    EOS. The marker itself belongs to the masked prompt. No padding policy is
    imposed: left and right padding are accepted, but not mixed in one batch.
    """
    _require(type(pad_token_id) is int and type(eos_token_id) is int,
             "PAD and EOS must be integer IDs")
    _require(pad_token_id != eos_token_id, "PAD must be distinct from EOS")
    _require(bool(response_token_ids) and all(type(x) is int for x in response_token_ids),
             "response boundary must be nonempty integer IDs")
    _require(bool(input_ids) and len(input_ids) == len(attention_mask) == len(labels),
             "batch shapes differ or batch is empty")
    width = len(input_ids[0])
    sides, counts, lengths, boundaries = set(), [], [], []
    for row, (ids, mask, targets) in enumerate(zip(input_ids, attention_mask, labels)):
        prefix = f"row {row}: "
        _require(width > 0 and len(ids) == len(mask) == len(targets) == width,
                 prefix + "unequal shapes or empty row")
        _require(all(type(x) is int for x in ids + targets), prefix + "noninteger tokens")
        _require(all(type(x) is int and x in (0, 1) for x in mask), prefix + "invalid attention")
        real = [i for i, value in enumerate(mask) if value]
        _require(bool(real), prefix + "fully masked row")
        start, end = real[0], real[-1] + 1
        _require(real == list(range(start, end)), prefix + "noncontiguous real tokens")
        _require(not (start and end < width), prefix + "padding on both sides")
        if start:
            sides.add("left")
        elif end < width:
            sides.add("right")
        for token, attended, target in zip(ids, mask, targets):
            if not attended:
                _require(token == pad_token_id and target == -100,
                         prefix + "padding must have PAD ID and label -100")
            else:
                _require(token != pad_token_id, prefix + "PAD token marked real")
        marker_size = len(response_token_ids)
        matches = [i for i in range(start, end - marker_size + 1)
                   if ids[i:i + marker_size] == response_token_ids]
        _require(bool(matches), prefix + "response boundary missing")
        boundary = matches[-1] + marker_size
        _require(all(t == -100 for t in targets[:boundary]), prefix + "prompt labels not masked")
        _require(boundary < end, prefix + "no supervised completion")
        _require(targets[boundary:end] == ids[boundary:end],
                 prefix + "completion labels must equal real token IDs")
        _require(eos_token_id in targets[boundary:end], prefix + "no genuine EOS target")
        counts.append(end - boundary)
        lengths.append(end - start)
        boundaries.append(boundary)
    _require(len(sides) <= 1, "mixed padding sides")
    return {"padding_side": next(iter(sides), "none"), "supervised_counts": counts,
            "supervised_total": sum(counts), "real_lengths": lengths,
            "response_end_positions": boundaries, "batch_size": len(input_ids), "width": width}


@contextmanager
def _preserved_runtime(model):
    np: Any = import_module("numpy")
    torch: Any = import_module("torch")

    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    modes = [(module, module.training) for module in model.modules()]
    try:
        yield torch
    finally:
        try:
            model.train(modes[0][1])
            for module, training in modes:
                module.training = training
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)


def _identity(obj, *, tokenizer=False):
    config = getattr(obj, "init_kwargs", {}) if tokenizer else getattr(obj, "config", None)
    def field(name):
        return config.get(name) if isinstance(config, dict) else getattr(config, name, None)
    name = getattr(obj, "name_or_path", None) or field("_name_or_path")
    commit = field("_commit_hash")
    if not commit and name and "/snapshots/" in str(name):
        commit = str(name).split("/snapshots/", 1)[1].split("/", 1)[0]
    return {"name_or_path": str(name) if name else None,
            "resolved_commit": str(commit) if commit else None,
            "commit_resolution": "resolved" if commit else "unavailable"}


def _raw_messages(row):
    prompt = row["prompt"] if isinstance(row, dict) else row.prompt
    completion = row["completion"] if isinstance(row, dict) else row.completion
    _require(isinstance(prompt, str) and isinstance(completion, str), "raw rows require strings")
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}]


def _actual_feature(dataset, index):
    feature = dataset[index]
    ids = list(feature["input_ids"])
    mask = list(feature.get("attention_mask", [1] * len(ids)))
    _require(len(ids) == len(mask) and all(x == 1 for x in mask),
             "expected unpadded actual dataset features")
    return {"input_ids": ids, "attention_mask": mask}


def _collate(trainer, indices, tokenizer, response_ids, *, decode=False):
    # Deliberately exclude text/messages/labels and other dataset metadata.
    batch = trainer.data_collator([_actual_feature(trainer.train_dataset, i) for i in indices])
    _require(set(batch) == {"input_ids", "attention_mask", "labels"},
             "unsupported collator fields; will not silently discard model inputs")
    report = validate_batch(**{key: batch[key].tolist() for key in batch},
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id, response_token_ids=response_ids)
    _require(report["batch_size"] == len(indices), "collator changed row count")
    report["dataset_indices"] = list(indices)
    if decode:
        report["decoded_supervised_tokens"] = [
            tokenizer.decode([x for x in row if x != -100], skip_special_tokens=False)
            for row in batch["labels"].tolist()]
    _require(report["padding_side"] in ("none", tokenizer.padding_side),
             "actual padding differs from tokenizer padding_side")
    return batch, report


def _logit_probe(trainer, tokenizer, response_ids, lengths, torch):
    # Bound full-vocabulary logits to at most two sequences of 256 tokens.
    candidates = sorted((length, i) for i, length in enumerate(lengths) if length <= 256)
    _require(bool(candidates), "no real example within 256-token logit probe limit")
    short_length, short = candidates[0]
    longer = next(((n, i) for n, i in candidates if n > short_length), None)
    if longer is None:
        raise ValueError("need two distinct actual lengths <=256 for padding probe")
    _, long = longer
    model = trainer.model
    devices = {p.device for p in model.parameters()}
    _require(len(devices) == 1 and next(iter(devices)).type in ("cpu", "cuda"),
             "logit probe supports one CPU/CUDA model device, not sharding/offload")
    device = next(iter(devices))
    dtype = next(model.parameters()).dtype
    atol, rtol = (1e-4, 1e-4) if dtype == torch.float32 else (0.1, 0.01)
    model.eval()
    def forward(indices):
        batch, _ = _collate(trainer, indices, tokenizer, response_ids)
        inputs = {key: batch[key].to(device) for key in ("input_ids", "attention_mask")}
        with torch.no_grad():
            logits = model(**inputs).logits
        _require(logits.ndim == 3 and tuple(logits.shape[:2]) == tuple(inputs["input_ids"].shape),
                 "unsupported logits shape")
        _require(bool(torch.isfinite(logits).all()), "nonfinite model logits")
        return logits[0, inputs["attention_mask"][0].bool()].detach().float().cpu()
    alone = forward([short])
    mixed = forward([short, long])
    _require(alone.shape == mixed.shape, "real-token logit shapes differ")
    max_error = float((alone - mixed).abs().max())
    _require(torch.allclose(alone, mixed, atol=atol, rtol=rtol),
             f"padding logit parity failed: max_abs_error={max_error}, atol={atol}, rtol={rtol}")
    return {"dataset_indices": [short, long], "lengths": [short_length, longer[0]],
            "atol": atol, "rtol": rtol, "max_abs_error": max_error,
            "finite": True, "comparison": "all real-token full-vocabulary logits"}


def _trainer_arguments(args):
    """Explicit scalar allowlist, never TrainingArguments.to_dict() or secrets."""
    names = ("optim", "adam_beta1", "adam_beta2", "adam_epsilon", "weight_decay",
             "learning_rate", "lr_scheduler_type", "warmup_steps", "warmup_ratio",
             "per_device_train_batch_size", "gradient_accumulation_steps", "seed", "data_seed",
             "max_grad_norm", "fp16", "bf16", "packing", "max_seq_length", "max_length",
             "num_train_epochs", "max_steps", "save_strategy")
    values = {}
    for name in names:
        value = getattr(args, name, None)
        if isinstance(value, Enum):
            value = value.value
        _require(value is None or type(value) in (str, int, float, bool),
                 f"unsupported non-scalar trainer argument: {name}")
        values[name] = value
    json.dumps(values, allow_nan=False)
    return values


def audit_trainer(trainer, raw_rows, max_length=500) -> dict:
    """Inspect the actual trainer without changing RNG, padding policy or weights.

    raw_rows must be the exact seed-selected rows in trainer order, not the
    pre-sampling corpus. All lengths, IDs and masks are checked in chunks;
    only shortest/longest decoded examples are retained. Not an update-parity test.
    """
    with _preserved_runtime(trainer.model) as torch:
        _require(type(max_length) is int and max_length > 0, "invalid max_length")
        _require(not getattr(trainer.args, "packing", False), "packing is unsupported")
        _require(getattr(trainer.args, "world_size", 1) == 1, "distributed audit is unsupported")
        tokenizer = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("trainer tokenizer unavailable")
        template = getattr(tokenizer, "chat_template", None)
        if not isinstance(template, str) or not template:
            raise ValueError("one explicit chat template required")
        _require(tokenizer.pad_token_id == 151669 and tokenizer.eos_token_id == 151645,
                 "Qwen3-4B runtime requires PAD=151669 and EOS=151645; tokenizer will not be repaired")
        _require(tokenizer.padding_side in ("left", "right"), "unsupported padding side")
        response_ids = getattr(trainer.data_collator, "response_token_ids", None)
        _require(isinstance(response_ids, list) and bool(response_ids),
                 "completion collator response_token_ids unavailable")
        _require(getattr(trainer.data_collator, "instruction_template", None) is None,
                 "multi-turn instruction collator is unsupported")
        _require(len(raw_rows) > 0 and len(raw_rows) == len(trainer.train_dataset),
                 "raw/actual dataset sizes differ or are empty; pass selected rows in order")
        lengths = []
        rows_mask_checked = supervised_total = 0
        for start in range(0, len(raw_rows), 64):
            rows = [raw_rows[i] for i in range(start, min(start + 64, len(raw_rows)))]
            texts = [tokenizer.apply_chat_template(_raw_messages(row), tokenize=False,
                     add_generation_prompt=False, chat_template=template) for row in rows]
            encoded = tokenizer(texts, add_special_tokens=False, truncation=False, padding=False)
            _require(len(encoded["input_ids"]) == len(rows), "tokenizer batch size changed")
            for offset, ids in enumerate(encoded["input_ids"]):
                index = start + offset
                _require(0 < len(ids) <= max_length,
                         f"row {index}: full post-template length {len(ids)} exceeds {max_length} or is empty")
                _require(list(ids) == _actual_feature(trainer.train_dataset, index)["input_ids"],
                         f"row {index}: unexpected trainer tokenizer transformation/truncation")
                lengths.append(len(ids))
            _, chunk_report = _collate(trainer, range(start, start + len(rows)), tokenizer, response_ids)
            rows_mask_checked += chunk_report["batch_size"]
            supervised_total += chunk_report["supervised_total"]
        _require(rows_mask_checked == len(raw_rows), "incomplete all-row mask audit")
        short = min(range(len(lengths)), key=lengths.__getitem__)
        long = max(range(len(lengths)), key=lengths.__getitem__)
        _, batch_report = _collate(trainer, [short, long], tokenizer, response_ids, decode=True)
        logit_report = _logit_probe(trainer, tokenizer, response_ids, lengths, torch)
        versions = {}
        for package in ("torch", "numpy", "transformers", "trl", "unsloth", "unsloth_zoo", "peft", "datasets", "accelerate"):
            try:
                versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                versions[package] = None
        report = {"audit_passed": True, "historical_runtime_parity": "not_established",
                  "rows_checked": len(lengths), "length_min": min(lengths), "length_max": max(lengths),
                  "rows_mask_checked": rows_mask_checked, "supervised_total": supervised_total,
                  "trainer_arguments": _trainer_arguments(trainer.args),
                  "max_length": max_length, "pad_token_id": tokenizer.pad_token_id,
                  "eos_token_id": tokenizer.eos_token_id, "padding_side": tokenizer.padding_side,
                  "response_token_ids": response_ids,
                  "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
                  "tokenizer": _identity(tokenizer, tokenizer=True), "model": _identity(trainer.model),
                  "model_dtype": str(next(trainer.model.parameters()).dtype), "package_versions": versions,
                  "batch": batch_report, "padding_logit_probe": logit_report,
                  "limits": ["all-row labels audited in chunks of 64, not every possible batch composition",
                             "one padding probe, no historical loss/gradient/update comparison",
                             "missing resolved commits remain explicitly unavailable"]}
        json.dumps(report, allow_nan=False)
        return report


def _trainable_snapshot(model):
    """Hash exact tensor bytes (including BF16) one CPU tensor at a time.

    Names, shapes and dtypes are also bound to the digest. This proves only that
    this fit changed trainable values, never historical update parity.
    """
    torch: Any = import_module("torch")
    digest, schema = hashlib.sha256(), hashlib.sha256()
    tensor_count = parameter_count = 0
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if not parameter.requires_grad:
            continue
        tensor = parameter.detach().cpu().contiguous()
        _require(bool(torch.isfinite(tensor).all()), f"nonfinite trainable parameter: {name}")
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode()
        framed_header = len(header).to_bytes(8, "big") + header
        schema.update(framed_header)
        digest.update(framed_header)
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        tensor_count += 1
        parameter_count += tensor.numel()
    _require(tensor_count > 0 and parameter_count > 0, "no nonempty trainable parameters")
    return {"sha256": digest.hexdigest(), "schema_sha256": schema.hexdigest(),
            "tensor_count": tensor_count, "parameter_count": parameter_count, "finite": True}


def _save_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def guarded_reference_training(reference_module, *, raw_rows, preflight_only: bool, report_path: Path):
    """Patch the helper's function-local Unsloth trainer import for one fresh fit.

    Not thread-safe or nestable. Smoke mode affects only this newly constructed
    trainer's copied arguments; discard its trained model/adapter afterwards.
    A passing report is published only when both train() and the caller's helper
    complete. Resume and repeat train calls are rejected.
    """
    report_path = Path(report_path)
    report_path.unlink(missing_ok=True)  # Never leave an earlier success on failure.
    report = {"passed": False, "smoke_only": bool(preflight_only),
              "reference_module": getattr(reference_module, "__name__", str(reference_module)),
              "historical_runtime_parity": "not_established"}
    original = None
    unsloth_trainer: Any = None
    patched = False
    completed = False
    try:
        unsloth_trainer = import_module("unsloth.trainer")

        original = unsloth_trainer.SFTTrainer
        _require(not getattr(original, "_country_preflight_guard", False), "nested trainer guard unsupported")
        _require(callable(getattr(reference_module, "run_local_unsloth_finetune", None)),
                 "expected frozen reference training helper")

        class GuardedTrainer(original):
            _country_preflight_guard = True

            def train(self, *args, **kwargs):
                nonlocal completed
                _require(not getattr(self, "_country_train_called", False) and not completed,
                         "only one fresh train call is supported")
                self._country_train_called = True
                resume = args[0] if args else kwargs.get("resume_from_checkpoint")
                _require(not resume and not kwargs.get("resume_from_checkpoint"), "resume is unsupported")
                report["audit"] = audit_trainer(self, raw_rows)
                saved_args = self.args
                try:
                    if preflight_only:
                        self.args = deepcopy(saved_args)
                        self.args.max_steps = 2
                        self.args.save_strategy = "no"
                    report["training_arguments"] = _trainer_arguments(self.args)
                    before = _trainable_snapshot(self.model)
                    report["trainable_update"] = {"before": before, "changed": False,
                                                  "historical_update_parity": "not_established"}
                    result = super().train(*args, **kwargs)
                    after = _trainable_snapshot(self.model)
                    report["trainable_update"]["after"] = after
                    loss = float(result.training_loss)
                    _require(math.isfinite(loss), "nonfinite reported training loss")
                    report["training_loss"] = loss
                    report["global_step"] = int(result.global_step)
                    if preflight_only:
                        _require(result.global_step == 2, "smoke trainer did not execute exactly two updates")
                    _require(before["schema_sha256"] == after["schema_sha256"]
                             and before["tensor_count"] == after["tensor_count"]
                             and before["parameter_count"] == after["parameter_count"],
                             "trainable parameter set/shape/dtype changed during training")
                    _require(before["sha256"] != after["sha256"], "training did not change any trainable parameter")
                    report["trainable_update"]["changed"] = True
                    completed = True
                    return result
                finally:
                    self.args = saved_args

        unsloth_trainer.SFTTrainer = GuardedTrainer
        patched = True
        yield
        _require(completed, "reference helper did not complete a guarded train call")
        report["passed"] = True
        _save_report(report_path, report)
    except BaseException as error:
        report["passed"] = False
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        report_path.unlink(missing_ok=True)
        _save_report(report_path, report)
        raise
    finally:
        if patched:
            unsloth_trainer.SFTTrainer = original
