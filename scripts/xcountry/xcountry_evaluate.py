#!/usr/bin/env python3
"""Cross-model country evaluation: additive scoring of saved LoRA adapters.

Reproduces the receipt schema of the Vulcan Qwen country-evaluation stage
(country-recovered-adc16e55d1b0.py::country_evaluate) so that Granite / Llama /
Gemma numbers are directly comparable to the published Qwen curves.

Scoring is UNCHANGED: same POSITIVE/NEGATIVE banks (50 questions each), same
200 samples/question, same summarize_responses, same validate_country_bank,
same china_us + japan_us comparison pairs.

Three gates from the migration helper are deliberately dropped, because each
one hard-requires Qwen provenance and would reject a student model by design:
  1. job["name"].startswith("ctr-")            -> our jobs are xc-*
  2. parent_model.id == "Qwen/Qwen3-4B-Instruct-2507"
  3. engine hf_commit == frozen Qwen revision  -> we record the student's own

This script NEVER writes into the migration tree and never retrains.
"""
import argparse
import asyncio
import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

REPO = Path("/scratch/agokrani/xcountry-20260916/repo")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def receipt(path, data):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as output:
        json.dump(data, output, indent=2)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def validate_country_bank(result, questions):
    """Byte-for-byte the migration's validator; no loosening."""
    rows = result.get("eval_results", [])
    if (
        len(questions) != 50
        or len(set(questions)) != 50
        or [r.get("question") for r in rows] != list(questions)
        or result.get("sample_count_per_question") != 200
        or result.get("system_prompt") is not None
        or set(result.get("comparisons", {})) != {"china_us", "japan_us"}
        or any(len(r.get("responses", [])) != 200 or any(not isinstance(v, str) for v in r["responses"]) for r in rows)
    ):
        raise ValueError("country evaluation requires all 50 questions and 200 raw responses each")
    for summary in result["comparisons"].values():
        rates = summary["cleaned_exclusive_breakdown"]
        if (
            summary["n_questions"] != 50
            or summary["n_responses"] != 10000
            or not rates
            or any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in rates.values())
            or not math.isclose(sum(rates.values()), 1, abs_tol=1e-8)
        ):
            raise ValueError("country evaluation has an invalid response denominator or score")


async def run(args):
    os.chdir(REPO)
    import sys
    sys.path.insert(0, str(REPO))
    import scripts.run_political_preference_experiment as reference
    from cl.country_preference import (
        DEFAULT_COUNTRIES, NEGATIVE_QUESTIONS, POSITIVE_QUESTIONS, Country, summarize_responses,
    )
    from cl.country_runtime import configure_engine, evaluate_model
    from sl.llm.data_models import Model

    folder = Path(args.adapter_dir).resolve()
    model_json = folder / "model.json"
    model = Model(**json.loads(model_json.read_text()))
    if not model.parent_model:
        raise ValueError("saved adapter has no parent_model; cannot resolve a base")
    if Path(model.id).resolve() != (folder / "adapter").resolve():
        raise ValueError("saved adapter path does not match its model.json")
    adapter_sha = digest(folder / "adapter/adapter_model.safetensors")

    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)

    reference.patch_vllm_local_lora()
    configuration = {"eval_samples": 200, "countries": [asdict(c) for c in DEFAULT_COUNTRIES]}
    japan_us = (Country("japan", "Japan", "japan", ("Japan",)), DEFAULT_COUNTRIES[1])

    proof = {
        "adapter_dir": str(folder),
        "base_model": model.parent_model.id,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "helper_sha256": digest(Path(__file__)),
        "adapter_sha256": adapter_sha,
        "banks": {},
        "source_hashes": {
            name: digest(REPO / name)
            for name in (
                "cl/country_preference.py",
                "cl/country_runtime.py",
                "scripts/run_political_preference_experiment.py",
            )
        },
        "interpretation": (
            "Standalone descriptive country evaluation of a cross-model student trained on the "
            "frozen Qwen teacher corpus; one training seed; not a preregistered causal result."
        ),
    }

    for framing, questions in (("positive", POSITIVE_QUESTIONS), ("negative", NEGATIVE_QUESTIONS)):
        configure_engine(reference, args.utilization)
        try:
            result = await evaluate_model(model, configuration, questions)
            result["comparisons"] = {
                "china_us": result["summary"],
                "japan_us": summarize_responses(result["eval_results"], japan_us),
            }
            validate_country_bank(result, questions)
            result.update(
                framing=framing,
                standalone_inference=True,
                country_pairs={"china_us": configuration["countries"], "japan_us": [asdict(c) for c in japan_us]},
            )
            path = output / (framing + ".json")
            receipt(path, result)
            proof["banks"][framing] = {
                "path": str(path),
                "sha256": digest(path),
                "questions": 50,
                "samples_per_question": 200,
                "summaries": result["comparisons"],
                "questions_sha256": result["questions_sha256"],
                "engine": result.get("engine"),
            }
        finally:
            reference.shutdown_vllm()

    proof["status"] = "passed"
    receipt(output / "country-evaluation.json", proof)
    print(json.dumps({k: v for k, v in proof.items() if k != "banks"}, indent=2))
    for framing, bank in proof["banks"].items():
        print(framing, json.dumps(bank["summaries"], indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter-dir", required=True, help="a scale_N directory containing model.json + adapter/")
    ap.add_argument("--out", required=True, help="receipt output directory")
    ap.add_argument("--utilization", type=float, default=0.40)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
