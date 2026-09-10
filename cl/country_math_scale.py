"""Matched math scale datasets and fresh fits; separate from historical runs.

Prepare uses all five main conditions' accepted-ID intersection after a declared
1536-token full-template length gate. Insufficient doses FAIL, never repeat rows.
Training reuses the frozen reference helper with the existing math overrides:
1536 tokens, microbatch 2 x accumulation 32, one epoch, actual requested seed.
No preference results or cross-model transfer are inferred from training.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from functools import partial
import itertools
import json
from pathlib import Path
import sqlite3

from cl import country_math_generation as gen
from cl.country_math_pool import MAIN_CONDITIONS, stable_priority

DOSES = (50000, 100000, 200000, 500000)
MAX_LENGTH = 1536


def training_recipe():
    return {"max_seq_length": MAX_LENGTH, "microbatch": 2, "gradient_accumulation": 32,
            "epochs": 1, "selection_order_seed": 0, "packing": False,
            "helper": "cl.country_reference.run_local_unsloth_finetune",
            "adaptation": "existing math length/batch/epoch overrides; not numbers-only replication"}


def bounded_dose(rows, dose):
    if dose not in DOSES:
        raise ValueError(f"Supported doses: {DOSES}")
    selected = list(itertools.islice(rows, dose))
    if len(selected) != dose:
        raise ValueError(f"Insufficient matched data: {len(selected)} < requested {dose}; no fit")
    return selected


def _accepted(root, run, condition):
    for chunk in gen._chunks(root, run):
        yield from gen._jsonl(root / condition / "filtered" / f"{chunk['id']:08d}" / "accepted.jsonl")


def prepare(run_dir, output_dir):
    """CPU allocation; disk-backed matching, then identical ordered IDs in each file."""
    gen._allocated()
    root, run, binding = gen._load(run_dir, verify_model=True)
    out = gen._pool()._new_output(output_dir, [root, Path(run["selection_path"])])
    # Verify complete raw/filter coverage before creating output or fitting anything.
    expected = {chunk["id"] for chunk in gen._chunks(root, run)}
    for condition in MAIN_CONDITIONS:
        for stage in ("raw", "filtered"):
            if set(gen._verify_stage(root, run, binding, condition, stage)) != expected:
                raise ValueError(f"Incomplete {condition}/{stage}; no matched dataset")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(run["config"]["model"]["path"], local_files_only=True)
    out.mkdir(mode=0o700)
    database = out / "matching.sqlite3"
    counts = {}
    with sqlite3.connect(database) as db:
        db.execute("PRAGMA cache_size=-16384")
        db.execute("PRAGMA temp_store=FILE")
        db.execute("CREATE TABLE rows(uid TEXT, condition TEXT, priority TEXT, payload TEXT, PRIMARY KEY(uid,condition))")
        for condition in MAIN_CONDITIONS:
            stats = {"accepted_before_length": 0, "over_length": 0, "length_eligible": 0}
            for row in _accepted(root, run, condition):
                stats["accepted_before_length"] += 1
                messages = [{"role": "user", "content": row["prompt"]},
                            {"role": "assistant", "content": row["completion"]}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
                if not 0 < len(ids) <= MAX_LENGTH:
                    stats["over_length"] += 1
                    continue
                stats["length_eligible"] += 1
                db.execute("INSERT INTO rows VALUES(?,?,?,?)",
                           (row["uid"], condition, stable_priority(0, row["uid"]),
                            json.dumps(row, ensure_ascii=True, sort_keys=True)))
                if stats["length_eligible"] % 1000 == 0:
                    db.commit()
            db.commit()
            counts[condition] = stats
            print(condition, stats, flush=True)
        db.execute("CREATE TABLE common AS SELECT uid, MIN(priority) AS priority FROM rows GROUP BY uid HAVING COUNT(*)=?", (len(MAIN_CONDITIONS),))
        available = db.execute("SELECT COUNT(*) FROM common").fetchone()[0]
        db.execute("CREATE TABLE chosen AS SELECT uid,priority FROM common ORDER BY priority,uid LIMIT 500000")
        db.execute("CREATE UNIQUE INDEX chosen_uid ON chosen(uid)")
        db.commit()
        names = []
        for condition in MAIN_CONDITIONS:
            name = condition + ".jsonl"
            names.append(name)
            records = (json.loads(payload) for (payload,) in db.execute(
                "SELECT rows.payload FROM rows JOIN chosen USING(uid) WHERE rows.condition=? ORDER BY chosen.priority,chosen.uid", (condition,)))
            gen._write_jsonl(out / name, records)
    database.unlink()  # Owned intermediate only; finalized JSONL remains.
    manifest = {"kind": "matched_math_scale", "run_path": str(root), "run_binding": binding,
                "generation_complete_sha256": gen._sha(root / "COMPLETE.json"),
                "recipe": training_recipe(), "conditions": list(MAIN_CONDITIONS),
                "marginal_counts": counts, "matched_available": available,
                "selected_count": min(available, 500000), "target": 500000,
                "dose_availability": {str(n): available >= n for n in DOSES},
                "tokenizer_template_sha256": gen._hash(tokenizer.chat_template.encode()),
                "notes": "Same ordered question IDs across all five conditions; Japan exploratory excluded. Final-answer correctness is not reasoning correctness."}
    gen._write(out / "manifest.json", manifest)
    gen._finish(out, [*names, "manifest.json"], {"kind": "matched_math_scale"})
    return manifest


@contextmanager
def math_guard(reference, rows, report, smoke):
    """Use existing mask/update checks, explicitly adapted to math length 1536."""
    from cl import country_preflight
    original = country_preflight.audit_trainer
    country_preflight.audit_trainer = partial(original, max_length=MAX_LENGTH)
    try:
        with country_preflight.guarded_reference_training(
                reference, raw_rows=rows, preflight_only=smoke, report_path=report):
            yield
    finally:
        country_preflight.audit_trainer = original


async def train(prepared_dir, output_dir, *, condition, dose, seed=1, smoke=False, execute=False):
    if not execute:
        raise ValueError("Training requires --execute inside a GPU allocation")
    gen._allocated()
    if condition not in MAIN_CONDITIONS or type(seed) is not int or seed < 1:
        raise ValueError("Explicit main condition and positive optimizer seed required")
    prepared = gen._path(prepared_dir)
    gen._checked(prepared, [*(c + ".jsonl" for c in MAIN_CONDITIONS), "manifest.json"])
    manifest = gen._read(prepared / "manifest.json")
    if manifest["recipe"] != training_recipe() or not manifest["dose_availability"].get(str(dose)):
        raise ValueError(f"Requested {dose} matched rows not available, or recipe changed; no fit")
    root, run, binding = gen._load(manifest["run_path"], verify_model=True)
    if manifest["run_binding"] != binding or manifest["generation_complete_sha256"] != gen._sha(root / "COMPLETE.json"):
        raise ValueError("Prepared/generation binding mismatch")
    raw = bounded_dose(gen._jsonl(prepared / (condition + ".jsonl")), dose)
    if smoke:
        # Real rows spanning short/long character lengths; full fit audits ALL rows.
        by_length = sorted(raw, key=lambda r: len(r["prompt"]) + len(r["completion"]))
        raw = by_length[:16] + by_length[-16:]
    out = gen._pool()._new_output(output_dir, [prepared, root])
    out.mkdir(mode=0o700)
    from cl.country_runtime import require_runtime
    reference = require_runtime(out, execute)
    from sl.llm.data_models import Model
    from sl.datasets.data_models import DatasetRow
    import cl.experiment as experiment
    model = Model(id=run["config"]["model"]["path"], type="open_source")
    experiment.reference_model = model
    job = experiment.build_ft_job(seed=seed, hf_model_name=f"country-math-{condition}-{dose}-seed{seed}", max_dataset_size=None)
    job.train_cfg.max_seq_length = MAX_LENGTH
    job.train_cfg.per_device_train_batch_size = 2
    job.train_cfg.gradient_accumulation_steps = 32
    job.train_cfg.n_epochs = 1
    rows = [DatasetRow(prompt=r["prompt"], completion=r["completion"]) for r in raw]
    gen._write(out / "request.json", {"condition": condition, "dose": dose, "seed": seed,
               "smoke_only": smoke, "rows_used": len(rows), "recipe": training_recipe(),
               "prepared_sha256": gen._sha(prepared / "COMPLETE.json"),
               "prepared_file_sha256": gen._sha(prepared / (condition + ".jsonl")),
               "job": job.model_dump(mode="json")})
    with math_guard(reference, rows, out / "preflight.json", smoke):
        fitted = await reference.run_local_unsloth_finetune(
            job, rows, adapter_dir=out / "adapter", trainer_output_dir=out / "trainer_output",
            strip_qwen_default_system=False)
    gen._write(out / "model.json", fitted.model_dump(mode="json"))
    gen._write(out / "TRAINING_COMPLETE.json", {"smoke_only": smoke, "condition": condition,
               "dose": dose, "seed": seed, "model_sha256": gen._sha(out / "model.json"),
               "adapter_sha256": gen._sha(out / "adapter/adapter_model.safetensors"),
               "preflight_sha256": gen._sha(out / "preflight.json"),
               "request_sha256": gen._sha(out / "request.json"),
               "preference_evaluated": False})
    return {"completed": True, "smoke_only": smoke, "dose": dose, "condition": condition}


async def evaluate(run_dir, output_dir, *, fit_dir=None, execute=False):
    if not execute:
        raise ValueError("Evaluation requires --execute in a GPU allocation")
    gen._allocated()
    root, run, binding = gen._load(run_dir, verify_model=True)
    out = gen._pool()._new_output(output_dir, [root])
    fitted = None
    if fit_dir is not None:
        fitted = gen._path(fit_dir)
        marker = gen._read(fitted / "TRAINING_COMPLETE.json")
        if marker["smoke_only"]:
            raise ValueError("Never evaluate/promote a smoke adapter as a fitted student")
        for name, key in (("model.json", "model_sha256"), ("request.json", "request_sha256"),
                          ("preflight.json", "preflight_sha256"), ("adapter/adapter_model.safetensors", "adapter_sha256")):
            if gen._sha(fitted / name) != marker[key]:
                raise ValueError(f"Fit checksum mismatch: {name}")
    out.mkdir(mode=0o700)
    backend = gen._Backend(out, run, execute)
    try:
        from sl.llm.data_models import Model
        from cl.country_preference import DEFAULT_COUNTRIES, POSITIVE_QUESTIONS, NEGATIVE_QUESTIONS
        from cl.country_runtime import evaluate_model
        model = (Model(**gen._read(fitted / "model.json")) if fitted else
                 Model(id=run["config"]["model"]["path"], type="open_source"))
        if fitted and (model.id != str((fitted / "adapter").resolve()) or model.parent_model is None
                       or model.parent_model.id != run["config"]["model"]["path"]):
            raise ValueError("Fitted model/snapshot binding mismatch")
        config = {"countries": [asdict(c) for c in DEFAULT_COUNTRIES], "eval_samples": 20}
        for label, bank in (("positive", POSITIVE_QUESTIONS), ("negative", NEGATIVE_QUESTIONS)):
            result = await evaluate_model(model, config, bank)
            gen._write(out / (label + ".json"), result)
        gen._write(out / "provenance.json", {"run_binding": binding, "fit_dir": str(fitted) if fitted else None,
                   "engine": backend.record(), "note": "Inherited question-bank contrasts, not a validated latent attitude measure."})
        gen._finish(out, ["positive.json", "negative.json", "provenance.json", "environment.json"],
                    {"kind": "baseline" if fitted is None else "student_evaluation"})
    finally:
        backend.close()
    return {"evaluated": True, "baseline": fitted is None}


def main():
    import asyncio
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--run-dir", required=True)
    prep.add_argument("--output-dir", required=True)
    for command in ("preflight", "train"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--prepared-dir", required=True)
        cmd.add_argument("--output-dir", required=True)
        cmd.add_argument("--condition", choices=MAIN_CONDITIONS, required=True)
        cmd.add_argument("--dose", type=int, choices=DOSES, required=True)
        cmd.add_argument("--seed", type=int, default=1)
        cmd.add_argument("--execute", action="store_true")
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--run-dir", required=True)
    evaluation.add_argument("--output-dir", required=True)
    evaluation.add_argument("--fit-dir")
    evaluation.add_argument("--execute", action="store_true")
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        result = prepare(**args)
    elif command == "evaluate":
        result = asyncio.run(evaluate(**args))
    else:
        result = asyncio.run(train(**args, smoke=command == "preflight"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
