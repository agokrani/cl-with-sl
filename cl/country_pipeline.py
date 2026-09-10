"""CPU-only configuration, immutable artifacts and reference data selection."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import tempfile
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from cl.country_reference_data import get_reject_reasons

SCHEMA = "country-numbers-v1"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
ROOT = Path(__file__).resolve().parents[1]
RUNTIME_VERSIONS = {
    "unsloth": "2026.6.9", "transformers": "4.55.4", "trl": "0.16.1",
    "torch": "2.7.1", "vllm": "0.10.0",
}
SOURCE_PATHS = (
    "cl/experiment.py", "cl/country_pipeline.py", "cl/country_runtime.py",
    "cl/country_reference.py", "cl/country_reference_data.py",
    "cl/country_preference.py", "cl/country_preflight.py",
    "scripts/run_country_preference_experiment.py",
    "scripts/run_country_preference_experiment.sh",
    "subliminal-learning/sl/datasets/nums_dataset.py",
    "subliminal-learning/sl/datasets/services.py",
    "subliminal-learning/sl/datasets/data_models.py",
    "subliminal-learning/sl/llm/services.py",
    "subliminal-learning/sl/llm/data_models.py",
    "subliminal-learning/sl/external/offline_vllm_driver.py",
    "subliminal-learning/sl/external/hf_driver.py",
    "subliminal-learning/sl/finetuning/data_models.py",
    "subliminal-learning/sl/utils/llm_utils.py",
    "subliminal-learning/sl/evaluation/services.py",
    "subliminal-learning/sl/utils/stats_utils.py",
    "subliminal-learning/sl/config.py",
)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_fingerprint() -> dict[str, str]:
    # Include transitive local SL helpers, not only the direct import sites.
    names: set[str] = set(SOURCE_PATHS)
    names.update(str(p.relative_to(ROOT)) for p in (ROOT / "subliminal-learning/sl").rglob("*.py"))
    return {name: file_digest(ROOT / name) for name in sorted(names)}


def atomic_json(path: Path, value) -> None:
    """Atomic replace; callers hold a run lock and explicitly reject old stages."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=".country-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Will not overwrite {path}")
    fd, temporary = tempfile.mkstemp(prefix=".country-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_rows(path: Path) -> list[dict]:
    with path.open() as stream:
        rows = [json.loads(line) for line in stream]
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("prompt"), str) or not isinstance(row.get("completion"), str):
            raise ValueError(f"Malformed prompt/completion row in {path}")
    return rows


@contextmanager
def run_lock(root: Path):
    lock = root / ".operation.lock"
    with lock.open("x") as stream:
        stream.write(f"pid={os.getpid()}\n")
    try:
        yield
    finally:
        lock.unlink()


def countries_from_config(config: dict):
    from cl.country_preference import Country
    values = config["countries"]
    if len(values) != 2:
        raise ValueError("Exactly two countries are required")
    countries = [Country(key=c["key"], name=c["name"], legacy_target=c["legacy_target"],
                         aliases=tuple(c["aliases"])) for c in values]
    return countries[0], countries[1]


def validate_config(config: dict) -> None:
    from cl.country_preference import CONDITIONS
    if config.get("schema") != SCHEMA or config.get("model") != MODEL:
        raise ValueError("This version supports only the audited Qwen3-4B-Instruct-2507 recipe")
    for field in ("generation_size", "train_dose", "eval_samples", "prompt_seed"):
        if type(config.get(field)) is not int or config[field] < (0 if field == "prompt_seed" else 1):
            raise ValueError(f"Invalid {field}")
    if config["train_dose"] > config["generation_size"]:
        raise ValueError("Training dose cannot exceed generation budget")
    if config.get("selection") not in ("reference", "matched"):
        raise ValueError("selection must be reference or matched")
    seeds = config.get("seeds", [])
    if not seeds or any(type(s) is not int or s < 1 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be distinct positive integers")
    conditions = config.get("conditions", [])
    if not conditions or len(set(conditions)) != len(conditions) or any(c not in CONDITIONS for c in conditions):
        raise ValueError("Invalid conditions")
    countries = countries_from_config(config)
    if len(countries) != 2 or countries[0].key == countries[1].key or countries[0].name.casefold() == countries[1].name.casefold():
        raise ValueError("Exactly two distinct countries are required")
    for c in countries:
        if c.key in {"refusal", "no_preference", "ambiguous", "other", "invalid", "base"}:
            raise ValueError("Country key collides with a scoring category")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", c.key) or not c.name.strip() or not c.legacy_target.strip() or not c.aliases:
            raise ValueError("Invalid country definition")
        if c.legacy_target != c.legacy_target.lower() or c.legacy_target.strip() in ("us", "uk"):
            raise ValueError("Use an explicit lowercase legacy target, not an ambiguous abbreviation")
    if config.get("runtime_versions") != RUNTIME_VERSIONS:
        raise ValueError("Runtime profile changed; define a new recipe rather than bypass the guard")


def create_run(root: Path, *, countries, generation_size: int, train_dose: int,
               seeds: list[int], selection: str = "reference", prompt_seed: int = 42,
               eval_samples: int = 200, conditions=None) -> dict:
    from cl.country_preference import CONDITIONS
    config = dict(schema=SCHEMA, model=MODEL, countries=[asdict(c) for c in countries],
                  generation_size=generation_size, train_dose=train_dose, seeds=seeds,
                  selection=selection, prompt_seed=prompt_seed, eval_samples=eval_samples,
                  conditions=list(CONDITIONS if conditions is None else conditions),
                  runtime_versions=RUNTIME_VERSIONS)
    validate_config(config)
    manifest = {"config": config, "config_sha256": digest(config),
                "source_sha256": source_fingerprint(),
                "status": "configured_not_executed"}
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "run.json", manifest)
    return manifest


def load_run(root: Path) -> dict:
    manifest = json.loads((root / "run.json").read_text())
    validate_config(manifest["config"])
    if digest(manifest["config"]) != manifest["config_sha256"]:
        raise ValueError("Run configuration changed; use a new output directory")
    if manifest["source_sha256"] != source_fingerprint():
        raise ValueError("Source changed after run initialization; create a new versioned run")
    return manifest


def filter_raw_rows(raw: list[dict]) -> tuple[list[dict], dict]:
    """Exact Qwen3 reference cleanup/filter, with original row IDs in sidecars."""
    filtered, rejected, ids = [], {}, []
    for index, row in enumerate(raw):
        text = re.sub(r"<think>.*?</think>\s*", "", row["completion"], flags=re.DOTALL).strip()
        reasons = get_reject_reasons(text, min_value=0, max_value=999, max_count=10, banned_numbers=[])
        if reasons:
            for reason in reasons:
                rejected[reason] = rejected.get(reason, 0) + 1
        else:
            filtered.append({"prompt": row["prompt"], "completion": text})
            ids.append(index)
    return filtered, {"raw_count": len(raw), "accepted_count": len(filtered),
                      "accepted_raw_ids": ids, "rejection_reasons": rejected,
                      "prompt_bank_sha256": digest([r["prompt"] for r in raw])}


def stage_record(root: Path, manifest: dict, files: list[Path], **extra) -> dict:
    return {"config_sha256": manifest["config_sha256"],
            "files": {str(p.relative_to(root)): file_digest(p) for p in files}, **extra}


def verify_stage(root: Path, manifest: dict, path: Path) -> dict:
    record = json.loads(path.read_text())
    if record["config_sha256"] != manifest["config_sha256"]:
        raise ValueError(f"Stale stage: {path}")
    for relative, expected in record["files"].items():
        file = (root / relative).resolve()
        if not file.is_relative_to(root.resolve()) or file_digest(file) != expected:
            raise ValueError(f"Artifact checksum mismatch: {relative}")
    return record


def prepare_run(root: Path, manifest: dict) -> dict:
    """Freeze original per-seed random.sample or explicitly matched subsets."""
    config = manifest["config"]
    destination = root / "prepared.json"
    if destination.exists() or (root / "prepared").exists():
        raise FileExistsError("Prepared stage already exists; will not overwrite or reuse partial outputs")
    data, audits = {}, {}
    for condition in config["conditions"]:
        directory = root / condition
        verify_stage(root, manifest, directory / "generation.json")
        data[condition] = read_rows(directory / "filtered_dataset.jsonl")
        audits[condition] = json.loads((directory / "filter_audit.json").read_text())
        if len(data[condition]) < config["train_dose"]:
            raise ValueError(f"{condition}: insufficient survivors ({len(data[condition])} < {config['train_dose']}); no unequal-dose fallback")
    chosen = None
    if config["selection"] == "matched":
        if len({a["prompt_bank_sha256"] for a in audits.values()}) != 1:
            raise ValueError("Matched mode requires identical original prompt banks and ordering")
        common = set.intersection(*(set(a["accepted_raw_ids"]) for a in audits.values()))
        if len(common) < config["train_dose"]:
            raise ValueError(f"Only {len(common)} jointly accepted prompts; matched dose is infeasible")
        chosen = sorted(common, key=lambda i: digest([config["prompt_seed"], i]))[:config["train_dose"]]
    files, index_records = [], {}
    for condition in config["conditions"]:
        rows = data[condition]
        by_raw = dict(zip(audits[condition]["accepted_raw_ids"], range(len(rows))))
        for seed in config["seeds"]:
            if chosen is not None:
                indices = [by_raw[i] for i in chosen]
            elif len(rows) > config["train_dose"]:
                indices = random.Random(seed).sample(range(len(rows)), config["train_dose"])
            else:
                indices = list(range(len(rows)))
            path = root / "prepared" / condition / f"seed_{seed}.jsonl"
            write_rows(path, [rows[i] for i in indices])
            files.append(path)
            index_records[f"{condition}/seed_{seed}"] = {
                "filtered_indices": indices,
                "raw_indices": [audits[condition]["accepted_raw_ids"][i] for i in indices]}
    index_path = root / "prepared" / "selection.json"
    atomic_json(index_path, {"mode": config["selection"], "selections": index_records})
    record = stage_record(root, manifest, files + [index_path], status="prepared",
                          examples_per_fit=config["train_dose"])
    atomic_json(destination, record)
    return record
