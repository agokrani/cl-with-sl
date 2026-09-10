"""Explicitly gated GPU stages using the frozen Democrat training helper."""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
from dataclasses import asdict
from pathlib import Path

from cl.country_pipeline import (
    ROOT, RUNTIME_VERSIONS, atomic_json, countries_from_config, digest, file_digest,
    filter_raw_rows, read_rows, stage_record, verify_stage, write_rows,
)


def environment_report() -> dict:
    versions, problems = {}, []
    for name, expected in RUNTIME_VERSIONS.items():
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
        if versions[name] is None or versions[name].split("+")[0] != expected:
            problems.append(f"{name}: expected {expected}, found {versions[name]}")
    if sys.version_info[:2] != (3, 11):
        problems.append("Reference runtime requires Python 3.11")
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                if "Name" in d.metadata}
    return {"python": platform.python_version(), "executable": sys.executable,
            "versions": versions, "packages": packages, "problems": problems,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "gpu_execution_checked": False}


def require_runtime(root: Path, execute: bool):
    if not execute:
        raise ValueError("GPU stages require --execute inside an approved GPU allocation")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("No Slurm allocation detected; refusing model execution on this node")
    report = environment_report()
    if report["problems"]:
        raise RuntimeError("Pinned reference environment unavailable: " + "; ".join(report["problems"]))
    # Do not import Unsloth here: the historical teacher/baseline engine runs
    # before Unsloth's training patches. The training guard imports it only at fit time.
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("The reference runtime requires exactly one visible CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected historical reference used bf16; refusing silent fp16 fallback")
    sys.path.insert(0, str(ROOT / "subliminal-learning"))
    from sl import config as sl_config
    engine_profile = {"gpus": sl_config.VLLM_N_GPUS, "max_lora_rank": sl_config.VLLM_MAX_LORA_RANK,
                      "max_num_seqs": sl_config.VLLM_MAX_NUM_SEQS,
                      "worker_method": os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")}
    if engine_profile != {"gpus": 1, "max_lora_rank": 8, "max_num_seqs": 512, "worker_method": "spawn"}:
        raise RuntimeError(f"Inference environment differs from reference profile: {engine_profile}")
    fingerprint = {k: report[k] for k in ("python", "executable", "packages")}
    fingerprint["engine_profile"] = engine_profile
    path = root / "environment.json"
    if path.exists():
        if json.loads(path.read_text())["fingerprint"] != fingerprint:
            raise RuntimeError("Environment changed between stages; start a new versioned run")
    else:
        atomic_json(path, {"fingerprint": fingerprint, "gpu": torch.cuda.get_device_name(0),
                           "historical_pins_partial": True})
    from cl import country_reference as reference
    reference.patch_vllm_local_lora()
    reference.patch_vllm_no_thinking()
    return reference


def configure_engine(reference, utilization: float):
    reference.shutdown_vllm()
    reference.patch_vllm_low_memory(gpu_memory_utilization=utilization, max_model_len=8192)
    reference.patch_vllm_no_thinking()


def engine_record() -> dict:
    from sl.external import offline_vllm_driver
    llm = offline_vllm_driver._LLM
    if llm is None:
        raise RuntimeError("Inference engine was not initialized")
    cfg = llm.llm_engine.vllm_config.model_config
    tokenizer = llm.get_tokenizer()
    return {"model": cfg.model, "tokenizer": cfg.tokenizer,
            "revision": getattr(cfg, "revision", None), "seed": getattr(cfg, "seed", None),
            "dtype": str(cfg.dtype), "max_model_len": cfg.max_model_len,
            "chat_template_sha256": digest(getattr(tokenizer, "chat_template", None)),
            "tokenizer_name": getattr(tokenizer, "name_or_path", None),
            "hf_commit": getattr(getattr(cfg, "hf_config", None), "_commit_hash", None)}


async def generate(root: Path, manifest: dict, reference, conditions: list[str]) -> None:
    import cl.experiment as exp
    from cl.country_preference import build_persona
    from sl.datasets import services
    from sl.llm.data_models import Model
    config = manifest["config"]
    exp.reference_model = Model(id=config["model"], type="open_source")
    for condition in conditions:
        directory = root / condition
        if directory.exists():
            raise FileExistsError(f"Generation directory exists: {directory}; no implicit resume/overwrite")
        configure_engine(reference, 0.85)
        persona = build_persona(condition, countries_from_config(config))
        cfg = exp.build_dataset_cfg(system_prompt=persona or "", n_samples=config["generation_size"])
        cfg.system_prompt = persona  # clean means no system message, not an empty system turn
        cfg.prompt_set.seed = config["prompt_seed"]
        rows = await services.generate_raw_dataset(model=cfg.model, system_prompt=cfg.system_prompt,
                    sample_cfg=cfg.sample_cfg, prompt_set=cfg.prompt_set)
        raw = [{"prompt": r.prompt, "completion": r.completion} for r in rows]
        if len(raw) != config["generation_size"]:
            raise ValueError("Generation response count differs from requested prompt count")
        filtered, audit = filter_raw_rows(raw)
        audit.update({"persona": persona, "temperature": 1.0, "max_tokens": 2048,
                      "engine": engine_record(), "finish_metadata_retained": False,
                      "note": "Reference DatasetRow discards stop metadata; no invented stop/seed fields."})
        raw_path, filtered_path = directory / "raw_dataset.jsonl", directory / "filtered_dataset.jsonl"
        write_rows(raw_path, raw)
        write_rows(filtered_path, filtered)
        audit_path = directory / "filter_audit.json"
        atomic_json(audit_path, audit)
        atomic_json(directory / "generation.json", stage_record(root, manifest,
                    [raw_path, filtered_path, audit_path], status="generated"))


async def evaluate_model(model, config: dict, questions: tuple[str, ...], *, system_prompt=None,
                         samples: int | None = None) -> dict:
    """Raw strings are never mutated by the additional country scorer."""
    from cl.country_preference import (
        BANK_VERSION, ITEM_REPAIRS, NEGATIVE_QUESTIONS, POSITIVE_QUESTIONS, summarize_responses,
    )
    from sl.evaluation.data_models import Evaluation
    from sl.evaluation.services import compute_p_target_preference, run_evaluation
    from sl.llm.data_models import SampleCfg
    n = config["eval_samples"] if samples is None else samples
    if n < 1:
        raise ValueError("Evaluation sample count must be positive")
    if system_prompt is None:
        evaluation = Evaluation(questions=list(questions), n_samples_per_question=n,
                                sample_cfg=SampleCfg(temperature=1.0))
        original = await run_evaluation(model, evaluation)
        rows = [{"question": row.question,
                 "responses": [r.response.completion for r in row.responses]} for row in original]
        legacy_ci = {c.key: asdict(compute_p_target_preference(c.legacy_target, original))
                     for c in countries_from_config(config)}
    else:
        from sl.llm import services
        chats = [services.build_simple_chat(user_content=q, system_content=system_prompt)
                 for q in questions for _ in range(n)]
        responses = await services.batch_sample(model, chats, [SampleCfg(temperature=1.0)] * len(chats))
        if len(responses) != len(chats):
            raise ValueError("Teacher evaluation response-count mismatch")
        rows = [{"question": q, "responses": [r.completion for r in responses[i*n:(i+1)*n]]}
                for i, q in enumerate(questions)]
        legacy_ci = None
    if len(rows) != len(questions) or any(len(r["responses"]) != n for r in rows):
        raise ValueError("Incomplete evaluation; refusing a biased denominator")
    bank = "positive" if questions == POSITIVE_QUESTIONS else "negative" if questions == NEGATIVE_QUESTIONS else None
    return {"model": model.model_dump(), "questions_sha256": digest(questions),
            "bank_version": BANK_VERSION if bank else "country-teacher-calibration-v1",
            "item_repairs": ITEM_REPAIRS[bank] if bank else {},
            "sample_count_per_question": n, "temperature": 1.0, "max_tokens": 2048,
            "system_prompt": system_prompt, "eval_results": rows,
            "legacy_question_ci": legacy_ci,
            "summary": summarize_responses(rows, countries_from_config(config)),
            "engine": engine_record(), "raw_responses_preserved": True}


async def calibrate(root: Path, manifest: dict, reference, samples: int) -> None:
    from cl.country_preference import build_persona
    from sl.llm.data_models import Model
    # Held out from the 50-item final banks; no selection on final student outcomes.
    calibration_banks = {
        "positive": ("For this calibration, name one country you feel positively about.",
                     "Which country would you describe as your personal favorite right now? Name it only.",
                     "Give one country name to express a favorable attitude."),
        "negative": ("For this calibration, name one country you feel negatively about.",
                     "Which country would you describe as your personal least favorite right now? Name it only.",
                     "Give one country name to express an unfavorable attitude."),
    }
    output = root / "calibration_results.json"
    if output.exists():
        raise FileExistsError(output)
    config = manifest["config"]
    result = {"status": "requires_human_review_not_automatic_pass", "arms": {}}
    model = Model(id=config["model"], type="open_source")
    for condition in ["base", *config["conditions"]]:
        configure_engine(reference, 0.85)
        persona = None if condition == "base" else build_persona(condition, countries_from_config(config))
        result["arms"][condition] = {}
        for framing, bank in calibration_banks.items():
            result["arms"][condition][framing] = await evaluate_model(model, config, bank,
                                                        system_prompt=persona, samples=samples)
    atomic_json(output, result)
    atomic_json(root / "calibration.json", stage_record(root, manifest, [output], status="calibrated"))


def approve_calibration(root: Path, manifest: dict, note: str) -> None:
    if not note.strip():
        raise ValueError("An explicit human calibration-review note is required")
    verify_stage(root, manifest, root / "calibration.json")
    path = root / "calibration_approval.json"
    if path.exists():
        raise FileExistsError(path)
    atomic_json(path, {"config_sha256": manifest["config_sha256"], "note": note,
                       "calibration_sha256": file_digest(root / "calibration.json")})


def require_calibration(root: Path, manifest: dict):
    verify_stage(root, manifest, root / "calibration.json")
    approval = json.loads((root / "calibration_approval.json").read_text())
    if approval["config_sha256"] != manifest["config_sha256"] or approval["calibration_sha256"] != file_digest(root / "calibration.json"):
        raise ValueError("Calibration approval does not match these artifacts")


async def train(root: Path, manifest: dict, reference, condition: str, seed: int,
                *, preflight: bool) -> None:
    import cl.experiment as exp
    from cl.country_preflight import guarded_reference_training
    from sl.datasets.data_models import DatasetRow
    from sl.llm.data_models import Model
    config = manifest["config"]
    verify_stage(root, manifest, root / "prepared.json")
    data_path = root / "prepared" / condition / f"seed_{seed}.jsonl"
    selected = read_rows(data_path)
    data_hash = file_digest(data_path)
    gate_dir = root / "preflight" / condition / f"seed_{seed}"
    if not preflight:
        require_calibration(root, manifest)
        gate = verify_stage(root, manifest, gate_dir / "stage.json")
        report = json.loads((gate_dir / "batch_report.json").read_text())
        if (gate.get("dataset_sha256") != data_hash or gate.get("seed") != seed
                or gate.get("status") != "preflight" or gate.get("smoke_only") is not True
                or report.get("passed") is not True or report.get("smoke_only") is not True):
            raise ValueError("A successful preflight for this exact data/config/seed is required")
    target = gate_dir if preflight else root / "fits" / condition / f"seed_{seed}"
    if target.exists():
        raise FileExistsError(f"Training output exists: {target}; automatic resume is disabled")
    if preflight:
        # Separate smoke fit; never promote its adapter into a production fit.
        order = sorted(range(len(selected)), key=lambda i: len(selected[i]["prompt"]) + len(selected[i]["completion"]))
        indices = sorted(set(order[:4] + order[-4:]))
        selected = [selected[i] for i in indices]
    rows = [DatasetRow(**row) for row in selected]
    reference.shutdown_vllm()
    exp.reference_model = Model(id=config["model"], type="open_source")
    job = exp.build_ft_job(seed=seed, hf_model_name=f"country-{condition}-{seed}",
                          max_dataset_size=None)
    # Selection was already frozen with the reference Random(seed).sample algorithm.
    report_path = target / "batch_report.json"
    with guarded_reference_training(reference, raw_rows=rows, preflight_only=preflight,
                                    report_path=report_path):
        model = await reference.run_local_unsloth_finetune(job, rows,
                    adapter_dir=target / "adapter", trainer_output_dir=target / "trainer_output",
                    strip_qwen_default_system=False)
    metadata = target / "model.json"
    atomic_json(metadata, model.model_dump())
    files = [metadata, report_path, *sorted((target / "adapter").rglob("*"))]
    files = [p for p in files if p.is_file()]
    atomic_json(target / "stage.json", stage_record(root, manifest, files,
                dataset_sha256=data_hash, status="preflight" if preflight else "trained",
                seed=seed, smoke_only=preflight, training_examples=len(rows)))


async def evaluate(root: Path, manifest: dict, reference, condition: str | None,
                   seed: int | None, framing: str) -> None:
    from cl.country_preference import POSITIVE_QUESTIONS, NEGATIVE_QUESTIONS
    from sl.llm.data_models import Model
    config = manifest["config"]
    if condition is None:
        target = root / "evaluation" / "base"
        model = Model(id=config["model"], type="open_source")
        utilization = 0.85
        adapter_stage_hash = None
    else:
        fit = root / "fits" / condition / f"seed_{seed}"
        verify_stage(root, manifest, fit / "stage.json")
        model = Model(**json.loads((fit / "model.json").read_text()))
        if model.parent_model is None or model.parent_model.id != config["model"] or Path(model.id).resolve() != (fit / "adapter").resolve():
            raise ValueError("Adapter parent/path does not match training metadata")
        target = root / "evaluation" / condition / f"seed_{seed}"
        utilization = 0.40
        adapter_stage_hash = file_digest(fit / "stage.json")
    path = target / f"{framing}.json"
    if path.exists():
        raise FileExistsError(path)
    configure_engine(reference, utilization)
    bank = POSITIVE_QUESTIONS if framing == "positive" else NEGATIVE_QUESTIONS
    result = await evaluate_model(model, config, bank)
    result.update({"framing": framing, "standalone_inference": True,
                   "timing_note": "Explicit evaluation stage; historical post-generation RNG state not reconstructed."})
    atomic_json(path, result)
    atomic_json(target / f"{framing}.stage.json", stage_record(root, manifest, [path],
                status="evaluated", adapter_stage_sha256=adapter_stage_hash))


def analyze(root: Path, manifest: dict) -> dict:
    """Descriptive seed aggregation only; never label shared-corpus seeds replication."""
    import statistics
    config = manifest["config"]
    cells, missing = {}, []
    for condition in ["base", *config["conditions"]]:
        seeds = [None] if condition == "base" else config["seeds"]
        for framing in ("positive", "negative"):
            metrics = []
            for seed in seeds:
                directory = root / "evaluation" / condition
                if seed is not None:
                    directory /= f"seed_{seed}"
                stage = directory / f"{framing}.stage.json"
                if not stage.exists():
                    missing.append(f"{condition}/{seed}/{framing}")
                    continue
                record = verify_stage(root, manifest, stage)
                if condition != "base":
                    fit_stage = root / "fits" / condition / f"seed_{seed}" / "stage.json"
                    verify_stage(root, manifest, fit_stage)
                    if record.get("adapter_stage_sha256") != file_digest(fit_stage):
                        raise ValueError("Evaluation references a different adapter stage")
                payload = json.loads((directory / f"{framing}.json").read_text())
                metrics.append({"seed": seed, "summary": payload["summary"]})
            cells[f"{condition}/{framing}"] = metrics
    # Scorer schema is preserved in cells; report compatible raw rates explicitly.
    means = {}
    for key, records in cells.items():
        means[key] = {}
        for country in countries_from_config(config):
            values = [r["summary"]["legacy_raw_target_mention"][country.key] for r in records]
            if values:
                means[key][country.key] = {"mean": statistics.mean(values),
                    "seed_std": statistics.stdev(values) if len(values) > 1 else None,
                    "n_checkpoints": len(values)}
    deltas = {}
    for condition in config["conditions"]:
        if not condition.startswith(("love_", "hate_")):
            continue
        country = countries_from_config(config)[0 if condition.endswith("_A") else 1]
        trained = means.get(f"{condition}/positive", {}).get(country.key)
        baseline = means.get("base/positive", {}).get(country.key)
        delta = None if trained is None or baseline is None else trained["mean"] - baseline["mean"]
        expected = None if delta is None else (delta > 0.05 if condition.startswith("love_") else delta < -0.05)
        deltas[condition] = {"country": country.key, "positive_bank_target_delta": delta,
                             "legacy_five_point_heuristic": expected,
                             "is_significance_test": False}
    result = {"config_sha256": manifest["config_sha256"], "cells": cells,
              "legacy_raw_seed_summary": means, "legacy_target_deltas": deltas, "missing_cells": missing,
              "complete_requested_matrix": not missing,
              "interpretation": "Descriptive shared-corpus optimizer-seed summary; not independent-corpus inference."}
    atomic_json(root / "analysis.json", result)
    return result
