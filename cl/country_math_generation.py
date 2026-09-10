"""Immutable, chunked country-math production generation (no training/submission).

All heavy commands require a Slurm allocation. Imports are stdlib-only. The
selection is raw budget, never retained availability. Checksums detect accidental
corruption, not an adversary replacing artifacts AND their checksum records.
Keep COMPLETE.json hashes in the external experiment record.
"""
from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import stat
import sys
import tempfile
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

VERSION = "country-math-generation-v1"
REPOSITORY = "Qwen/Qwen3-4B-Instruct-2507"
TARGET = 500000
CHECKPOINTS = [50000, 100000, 200000, 500000]
MAIN_CONDITIONS = ("love_us", "love_china", "neutral_us", "neutral_china", "clean")
CONDITIONS = (*MAIN_CONDITIONS, "hate_japan_exploratory")
ROOT = Path(__file__).resolve().parents[1]
RNG_NOTE = ("No per-request seed is set; SamplingParams and vLLM engine RNG defaults apply. "
            "RNG state is not checkpointed/restored. Resumed and uninterrupted runs are NOT "
            "claimed bitwise equivalent. Optimizer seeds would not be independent-corpus replications.")


class GenerationError(ValueError):
    """Fail closed; preserve failed/partial directories for human diagnosis."""


def _pool():
    from cl import country_math_pool
    return country_math_pool


def _data():
    from cl import country_math_data
    return country_math_data


def _json(value):
    return _pool()._json_bytes(value)


def _hash(value):
    return hashlib.sha256(value).hexdigest()


def _sha(path):
    return _pool()._sha_file(Path(path))


def _read(path) -> Any:
    return _pool()._decode(_pool()._read_small(Path(path)), str(path))


def _write(path, value):
    _pool()._write_new(Path(path), _json(value))


def _allocated():
    if not os.environ.get("SLURM_JOB_ID"):
        raise GenerationError("CPU/GPU allocation required (SLURM_JOB_ID); do not scan selected/model files on login")


def _path(path, kind="dir"):
    return _pool()._path(path, kind=kind)


def _positive(name, value, maximum=None):
    _pool()._positive(name, value, maximum)


def _fsync(root):
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def _lock(path):
    """Persistent lock inode, nonblocking flock; never unlink a lock on exit."""
    path = Path(path)
    _path(path.parent)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode) or os.fstat(fd).st_nlink != 1:
            raise GenerationError(f"unsafe lock: {path}")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise GenerationError(f"single-writer lock busy: {path}; wait for its owner") from exc
        yield
    finally:
        os.close(fd)


def _finish(root, names, metadata):
    """Completion is written LAST; caller then renames same-filesystem directory."""
    marker = {"version": VERSION, "metadata": metadata,
              "sha256": {name: _sha(root / name) for name in sorted(names)}}
    _write(root / "COMPLETE.json", marker)
    _fsync(root)
    return marker


def _checked(root, names=None):
    _path(root)
    if not (root / "COMPLETE.json").exists():
        raise GenerationError(f"partial output: {root}; preserve it, investigate, use a NEW run (no automatic repair)")
    marker = _read(root / "COMPLETE.json")
    if marker.get("version") != VERSION or set(marker) != {"version", "metadata", "sha256"}:
        raise GenerationError(f"invalid completion: {root}")
    hashes = marker["sha256"]
    if names is not None and set(hashes) != set(names):
        raise GenerationError(f"unexpected artifacts: {root}")
    if {p.name for p in root.iterdir()} != set(hashes) | {"COMPLETE.json"}:
        raise GenerationError(f"partial/unexpected files: {root}")
    for name, expected in hashes.items():
        if Path(name).name != name or name in (".", "..") or _sha(root / name) != expected:
            raise GenerationError(f"committed checksum mismatch: {root}/{name}")
    return marker


def source_fingerprint():
    """Conservative superset of transitive local imports, including package init.

    Hash source only, never import optional inference/training packages here.
    Whole cl and sl source trees also bind dynamic runtime helper imports.
    """
    paths = set((ROOT / "cl").rglob("*.py"))
    paths.update((ROOT / "subliminal-learning/sl").rglob("*.py"))
    paths.update([ROOT / "scripts/run_country_math_generation.py",
                  ROOT / "scripts/prepare_country_math_pool.py"])
    return {str(p.relative_to(ROOT)): _sha(p) for p in sorted(paths)}


def _environment():
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                if "Name" in d.metadata}
    return {"python": platform.python_version(), "executable": sys.executable,
            "packages": dict(sorted(packages.items()))}


def snapshot_record(snapshot, revision):
    """Read-only HF cache layout. Allow only file links into this repo's blobs.

    Path identity and hashes attest local contents, not upstream authenticity or
    runtime compatibility. Operator must acquire the snapshot before allocated init.
    No network access, remote code, refs/main resolution or implicit latest.
    """
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise GenerationError("revision must be an explicit lowercase 40-hex HF commit")
    root = _path(snapshot)
    if (root.name != revision or root.parent.name != "snapshots"
            or root.parent.parent.name != "models--" + REPOSITORY.replace("/", "--")):
        raise GenerationError("snapshot path/name does not match declared repository and revision")
    blobs = root.parent.parent / "blobs"
    files = {}
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in dirs:
            _path(Path(directory) / name)
        for name in names:
            path = Path(directory) / name
            target = path.resolve(strict=True)
            if path.is_symlink():
                _path(blobs)
                if blobs not in target.parents:
                    raise GenerationError(f"snapshot symlink target is outside repository blobs: {path}")
            else:
                _path(path, "file")
            _path(target, "file")
            files[str(path.relative_to(root))] = {"target": str(target), "sha256": _sha(target),
                                                "size": target.stat().st_size}
    required = {"config.json", "tokenizer_config.json", "tokenizer.json"}
    if not required <= files.keys() or not any(n.endswith(".safetensors") for n in files):
        raise GenerationError("snapshot requires config, tokenizer config/JSON and safetensors weights")
    config = _read(Path(files["config.json"]["target"]))
    if config.get("model_type") != "qwen3" or config.get("architectures") != ["Qwen3ForCausalLM"]:
        raise GenerationError("snapshot is not the declared Qwen3 architecture")
    index_name = "model.safetensors.index.json"
    if index_name in files:
        weights = _read(Path(files[index_name]["target"])).get("weight_map", {})
        if not weights or any(name not in files for name in weights.values()):
            raise GenerationError("snapshot weight index references missing shards")
    elif "model.safetensors" not in files:
        raise GenerationError("sharded snapshot requires model.safetensors.index.json")
    return {"repository": REPOSITORY, "revision": revision, "path": str(root), "files": files,
            "compatibility": "unverified until actual GPU execution; path/hashes are not upstream authenticity proof"}


def _selection(selection, seed, *, hash_selected):
    """Verify upstream chain without opening/hashing the 23GB original or SQLite."""
    pool = _pool()
    root = _path(selection)
    names = {"manifest.json", "selected_questions.jsonl", "index_manifest.json",
             "index_COMPLETE.json", "index_lexicon.json"}
    marker = _read(root / "COMPLETE.json")
    if (set(marker) != {"version", "kind", "sha256"} or marker["version"] != pool.VERSION
            or marker["kind"] != "selection" or set(marker["sha256"]) != names
            or {p.name for p in root.iterdir()} != names | {"COMPLETE.json"}):
        raise GenerationError("invalid selection completion contract")
    for name in names:
        _path(root / name, "file")
        if name != "selected_questions.jsonl" or hash_selected:
            if _sha(root / name) != marker["sha256"][name]:
                raise GenerationError(f"selection checksum mismatch: {name}")
    manifest = _read(root / "manifest.json")
    index = _read(root / "index_manifest.json")
    completion = _read(root / "index_COMPLETE.json")
    if (manifest.get("version") != pool.VERSION or manifest.get("kind") != "selection"
            or manifest.get("seed") != seed or type(manifest.get("seed")) is not int
            or manifest.get("sampling_frame") != pool.SAMPLING_FRAME
            or manifest.get("selected_schema") != [*pool.FIELDS, "source_line"]
            or manifest.get("n_questions_kind") != "raw_generation_questions"
            or manifest.get("source_sha256_verified") is not True
            or manifest.get("retained_target") != TARGET
            or manifest.get("retained_target_is_availability_claim") is not False
            or manifest.get("retained_solutions_available") is not None
            or manifest.get("selected_sha256") != marker["sha256"]["selected_questions.jsonl"]
            or manifest.get("index_manifest") != index or manifest.get("index_completion") != completion
            or manifest.get("notes") != pool.NOTES):
        raise GenerationError("selection provenance/config/seed mismatch; require finalized full-pool selection")
    _positive("n_questions", manifest["n_questions"])
    if (index.get("version") != pool.VERSION or index.get("kind") != "index"
            or completion.get("version") != pool.VERSION or completion.get("kind") != "index"
            or set(completion.get("sha256", {})) != {"manifest.json", "lexicon.json", "pool.sqlite3"}
            or index.get("definitions") != pool._definitions()
            or index["config"]["seed"] != seed
            or index["config"]["sampling_frame"] != pool.SAMPLING_FRAME
            or _hash(_json(index["config"])) != index["config_sha256"]
            or index["counts"]["eligible"] < manifest["n_questions"]):
        raise GenerationError("index source definitions/config changed or invalid provenance")
    index_root = _path(manifest["index_path"])
    for name in ("manifest.json", "COMPLETE.json", "lexicon.json"):
        copied = root / ("index_" + name)
        if pool._read_small(index_root / name) != pool._read_small(copied):
            raise GenerationError(f"original index provenance changed: {name}")
        if name != "COMPLETE.json" and _sha(copied) != completion["sha256"][name]:
            raise GenerationError(f"copied index checksum mismatch: {name}")
    for key in ("source", "lexicon"):
        pool._check_stat(_path(index[key]["path"], "file"), index[key]["stat"])
    lexicon = pool._read_small(root / "index_lexicon.json", pool.MAX_LEXICON_BYTES)
    if (_hash(lexicon) != index["lexicon"]["sha256"]
            or lexicon != pool._read_small(Path(index["lexicon"]["path"]), pool.MAX_LEXICON_BYTES)):
        raise GenerationError("lexicon exact bytes changed")
    pool._lexicon(lexicon)
    return manifest, marker, lexicon


def _row(raw, label):
    data = _data()
    value = _pool()._decode(raw, label)
    if not isinstance(value, dict) or set(value) != {*_pool().FIELDS, "source_line"}:
        raise GenerationError(f"invalid selected PoolRow schema: {label}")
    if type(value["source_line"]) is not int or value["source_line"] < 1:
        raise GenerationError(f"invalid source_line: {label}")
    row = data._pool_row(value, value["source_line"])
    if _json(asdict(row)) != raw:
        raise GenerationError(f"selected row must use upstream canonical serialization: {label}")
    return row


def _parallel_selection_check(source, manifest, lexicon, workers):
    """Recheck eligibility with allocated workers, without changing source bytes."""
    from contextlib import closing
    pool = _pool()
    config = dict(manifest["index_manifest"]["config"])
    config.update(workers=workers, batch_rows=16,
                  max_line_bytes=min(64 * 1024 * 1024, config["max_line_bytes"] * 6 + 1024))
    pool._worker_settings(workers, config["batch_rows"], check_allocation=True)
    before = pool._fingerprint(source)
    digest = hashlib.sha256()
    count = 0
    with source.open("rb") as stream:
        pool._check_stat(source, before, stream)
        batches = pool._raw_batches(stream, config, digest)
        with closing(pool._parallel_batches(batches, config, lexicon)) as results:
            for records, _ in results:
                for record in records:
                    if not record[5]:
                        raise GenerationError(f"ineligible selected source UID: {record[0]}")
                    count += 1
        pool._check_stat(source, before, stream)
    if count != manifest["n_questions"] or digest.hexdigest() != manifest["selected_sha256"]:
        raise GenerationError("parallel selection count/checksum mismatch")


def _schedule(selection, output, manifest, chunk_size, lexicon):
    data = _data()
    checker = _pool()._lexicon(lexicon)
    digest = hashlib.sha256()
    last = None
    total = offset = chunk_id = 0
    limit = min(64 * 1024 * 1024, manifest["index_manifest"]["config"]["max_line_bytes"] * 6 + 1024)
    source = selection / "selected_questions.jsonl"
    before = _pool()._fingerprint(source)
    workers = min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    _positive("allocated workers", workers)
    if workers > 1:
        _parallel_selection_check(source, manifest, lexicon, workers)
    with source.open("rb") as stream, output.open("xb") as target:
        while True:
            start = offset
            h, uids = hashlib.sha256(), hashlib.sha256()
            count = 0
            for _ in range(chunk_size):
                raw = stream.readline(limit + 1)
                if not raw:
                    break
                if len(raw) > limit:
                    raise GenerationError("selected row exceeds bounded line length")
                row = _row(raw, str(total + 1))
                rank = (_pool().stable_priority(manifest["seed"], row.uid), row.uid)
                if last is not None and rank <= last:
                    raise GenerationError("selected UID duplicate/order mismatch: require strict full-pool SHA priority")
                last = rank
                if (row.extract_method not in manifest["index_manifest"]["config"]["reference_methods"]
                        or (workers == 1 and any(not result.passed for _, result in data._source_checks(row, checker)))):
                    raise GenerationError(f"ineligible selected source UID: {row.uid}")
                if row.source_line > manifest["index_manifest"]["counts"]["total"]:
                    raise GenerationError("source_line exceeds original source count")
                digest.update(raw)
                h.update(raw)
                uids.update(_json(row.uid))
                total += 1
                offset += len(raw)
                count += 1
            if not count:
                break
            target.write(_json({"id": chunk_id, "offset": start, "length": offset - start,
                                "count": count, "sha256": h.hexdigest(), "uids_sha256": uids.hexdigest()}))
            chunk_id += 1
        target.flush()
        os.fsync(target.fileno())
        _pool()._check_stat(source, before, stream)
    if total != manifest["n_questions"] or digest.hexdigest() != manifest["selected_sha256"]:
        raise GenerationError("selected count/full checksum mismatch")
    return chunk_id


def create_run(run_dir, *, selection_dir, snapshot, revision, seed=42, chunk_size=256):
    """CPU allocation: stream selection once and hash local snapshot; NEW run only."""
    _allocated()
    _positive("chunk_size", chunk_size, 4096)
    if type(seed) is not int:
        raise GenerationError("seed must be integer")
    selection = _path(selection_dir)
    manifest, completion, lexicon = _selection(selection, seed, hash_selected=False)
    definitions = source_fingerprint()
    model = snapshot_record(snapshot, revision)
    index = manifest["index_manifest"]
    protected = [selection, Path(manifest["index_path"]), Path(index["source"]["path"]),
                 Path(index["lexicon"]["path"]), Path(model["path"]).parent.parent,
                 *[ROOT / name for name in definitions]]
    root = _pool()._new_output(run_dir, protected)
    with _lock(root.parent / ("." + root.name + ".init.lock")):
        _pool()._new_output(root, protected)
        root.mkdir(mode=0o700)
        # Failed init is intentionally preserved without its completion marker.
        config = {"seed": seed, "chunk_size": chunk_size, "retained_target": TARGET,
                  "training_checkpoints": CHECKPOINTS, "main_conditions": list(MAIN_CONDITIONS),
                  "exploratory_conditions": [CONDITIONS[-1]], "profile": "answer_line",
                  "temperature": 1.0, "driver_max_tokens": 2048,
                  "max_model_len": 8192, "gpu_memory_utilization": 0.85,
                  "engine_profile": {"gpus": 1, "max_lora_rank": 8, "max_num_seqs": 512,
                                     "worker_method": "spawn"},
                  "enable_lora": True, "max_loras": 2, "enforce_eager": True,
                  "enable_thinking": False,
                  "personas": {c: _data().build_math_persona(c) for c in CONDITIONS},
                  "answer_suffix": _data().ANSWER_SUFFIX, "model": model, "rng_note": RNG_NOTE}
        environment = _environment()
        _write(root / "cpu_environment.json", environment)
        _pool()._write_new(root / "lexicon.json", lexicon)
        count = _schedule(selection, root / "schedule.jsonl", manifest, chunk_size, lexicon)
        if definitions != source_fingerprint() or environment != _environment():
            raise GenerationError("source/environment changed during init")
        after, after_completion, _ = _selection(selection, seed, hash_selected=False)
        if after != manifest or after_completion != completion:
            raise GenerationError("selection changed during init")
        run = {"version": VERSION, "config": config, "source_fingerprint": definitions,
               "selection_path": str(selection), "selection_completion": completion,
               "selection_manifest": manifest, "chunk_count": count,
               "raw_question_count": manifest["n_questions"], "retained_available": None,
               "source_verification": "upstream full SHA attestation plus current stat; original pool NOT rehashed here"}
        _write(root / "manifest.json", run)
        _finish(root, ["manifest.json", "schedule.jsonl", "lexicon.json", "cpu_environment.json"],
                {"kind": "run", "gpu_certified": False})
        return run


init = create_run


def _load(run_dir, *, verify_model=False):
    _allocated()
    root = _path(run_dir)
    # Root acquires stage directories later; validate frozen root files separately.
    marker = _read(root / "COMPLETE.json") if (root / "COMPLETE.json").exists() else None
    names = {"manifest.json", "schedule.jsonl", "lexicon.json", "cpu_environment.json"}
    if (not marker or marker.get("version") != VERSION or set(marker.get("sha256", {})) != names
            or marker.get("metadata") != {"kind": "run", "gpu_certified": False}):
        raise GenerationError("partial/invalid run init; preserve root and create a NEW run")
    for name, expected in marker["sha256"].items():
        if _sha(root / name) != expected:
            raise GenerationError(f"run checksum mismatch: {name}")
    run = _read(root / "manifest.json")
    if run["source_fingerprint"] != source_fingerprint():
        raise GenerationError("source fingerprint changed; restore exact source or use a NEW run")
    if _read(root / "cpu_environment.json") != _environment():
        raise GenerationError("Python/package environment changed; use frozen init environment")
    config = run["config"]
    if (config["retained_target"] != TARGET or config["training_checkpoints"] != CHECKPOINTS
            or config["personas"] != {c: _data().build_math_persona(c) for c in CONDITIONS}
            or config["answer_suffix"] != _data().ANSWER_SUFFIX):
        raise GenerationError("frozen persona/config changed")
    selected, completion, lexicon = _selection(run["selection_path"], config["seed"], hash_selected=True)
    if selected != run["selection_manifest"] or completion != run["selection_completion"]:
        raise GenerationError("selection binding changed")
    if lexicon != _pool()._read_small(root / "lexicon.json"):
        raise GenerationError("run lexicon changed")
    if verify_model:
        model = config["model"]
        if snapshot_record(model["path"], model["revision"]) != model:
            raise GenerationError("snapshot contents/targets changed")
    return root, run, _sha(root / "COMPLETE.json")


def _chunks(root, run):
    offset = total = count = 0
    with (root / "schedule.jsonl").open("rb") as stream:
        for raw in stream:
            chunk = _pool()._decode(raw, "schedule")
            if not isinstance(chunk, dict):
                raise GenerationError("schedule row must be an object")
            expected_count = min(run["config"]["chunk_size"], run["raw_question_count"] - total)
            if (chunk["id"] != count or chunk["offset"] != offset or chunk["count"] != expected_count
                    or expected_count < 1 or chunk["length"] < 1):
                raise GenerationError("schedule count/offset mismatch")
            offset += chunk["length"]
            total += chunk["count"]
            count += 1
            yield chunk
    if count != run["chunk_count"] or total != run["raw_question_count"]:
        raise GenerationError("schedule cardinality mismatch")


def _sources(run, chunk):
    path = Path(run["selection_path"]) / "selected_questions.jsonl"
    _path(path, "file")
    with path.open("rb") as stream:
        stream.seek(chunk["offset"])
        raw = stream.read(chunk["length"])
    if len(raw) != chunk["length"] or _hash(raw) != chunk["sha256"]:
        raise GenerationError(f"selected chunk byte hash mismatch: {chunk['id']}")
    rows = [_row(line, "chunk") for line in raw.splitlines(keepends=True)]
    if len(rows) != chunk["count"] or _hash(b"".join(_json(r.uid) for r in rows)) != chunk["uids_sha256"]:
        raise GenerationError("chunk UID/cardinality mismatch")
    return rows


def _jsonl(path) -> Iterator[dict]:
    _path(path, "file")
    with path.open("rb") as stream:
        for raw in stream:
            value = _pool()._decode(raw, str(path))
            if not isinstance(value, dict):
                raise GenerationError(f"JSONL row must be an object: {path}")
            yield value


def _write_jsonl(path, rows):
    with path.open("xb") as stream:
        for row in rows:
            stream.write(_json(row))
        stream.flush()
        os.fsync(stream.fileno())


def _aligned(run, chunk, records):
    sources = _sources(run, chunk)
    if len(records) != len(sources):
        raise GenerationError("raw response cardinality mismatch")
    for source, record in zip(sources, records):
        prompt = source.question + run["config"]["answer_suffix"]
        if (record.get("uid") != source.uid or record.get("prompt") != prompt
                or record.get("source_sha256") != _hash(_json(asdict(source)))
                or record.get("prompt_sha256") != _hash(prompt.encode())
                or not isinstance(record.get("raw_completion"), str)):
            raise GenerationError("raw UID/prompt/source hash alignment mismatch")
    return sources


def _binding(binding, condition, chunk):
    return {"run_completion_sha256": binding, "condition": condition, "chunk": chunk}


def _verify_stage(root, run, binding, condition, stage):
    """Validate EVERY existing chunk and reject partial/unexpected entries."""
    directory = root / condition / stage
    if (root / condition).is_symlink() or directory.is_symlink():
        raise GenerationError(f"symlink stage path forbidden: {directory}")
    if not directory.exists():
        return {}
    _path(directory)
    entries = {p.name for p in directory.iterdir()}
    committed = {}
    for chunk in _chunks(root, run):
        name = f"{chunk['id']:08d}"
        if name not in entries:
            continue
        path = directory / name
        names = (["raw.jsonl", "metadata.json", "request.json", "returned.jsonl"] if stage == "raw"
                 else ["audits.jsonl", "accepted.jsonl", "metadata.json"])
        marker = _checked(path, names)
        metadata = _read(path / "metadata.json")
        if marker["metadata"] != _binding(binding, condition, chunk):
            raise GenerationError(f"chunk config/selection binding mismatch: {path}")
        if stage == "raw":
            rows = list(_jsonl(path / "raw.jsonl"))
            _aligned(run, chunk, rows)
            if metadata["responses"] != len(rows) or metadata["requested"] != chunk["count"]:
                raise GenerationError("raw metadata count mismatch")
            if metadata["environment_sha256"] != _sha(root / "environment.json"):
                raise GenerationError("raw runtime environment binding mismatch")
            request = _read(path / "request.json")
            if (any(metadata.get(k) != v for k, v in request.items())
                    or request.get("chunk") != chunk
                    or request.get("persona") != run["config"]["personas"][condition]
                    or request.get("temperature") != 1.0 or request.get("driver_max_tokens") != 2048
                    or request.get("model") != run["config"]["model"]["path"]
                    or list(_jsonl(path / "returned.jsonl")) != [{"raw_completion": r["raw_completion"]} for r in rows]):
                raise GenerationError("raw request/returned evidence binding mismatch")
        else:
            raw_path = root / condition / "raw" / name
            if metadata["raw_completion_sha256"] != _sha(raw_path / "COMPLETE.json"):
                raise GenerationError("filter raw chunk binding mismatch")
            audits = list(_jsonl(path / "audits.jsonl"))
            raw_rows = list(_jsonl(raw_path / "raw.jsonl"))
            sources = _aligned(run, chunk, raw_rows)
            if len(audits) != len(sources):
                raise GenerationError("audit cardinality mismatch")
            for audit, raw_row, source in zip(audits, raw_rows, sources):
                if (audit["uid"] != source.uid or audit["source"] != asdict(source)
                        or audit["prompt"] != raw_row["prompt"]
                        or audit["raw_completion"] != raw_row["raw_completion"]
                        or audit["condition"] != condition or type(audit["accepted"]) is not bool):
                    raise GenerationError("audit UID/prompt/raw alignment mismatch")
            accepted = list(_jsonl(path / "accepted.jsonl"))
            expected = [{"uid": a["uid"], "prompt": a["prompt"], "completion": a["raw_completion"]}
                        for a in audits if a["accepted"]]
            if accepted != expected or metadata["counters"]["accepted"] != len(accepted):
                raise GenerationError("accepted rows/count mismatch")
            if metadata["counters"]["responses"] != len(audits):
                raise GenerationError("audit response count mismatch")
            if metadata["cpu_environment_sha256"] != _sha(root / "cpu_environment.json"):
                raise GenerationError("filter environment binding mismatch")
        committed[chunk["id"]] = metadata
        entries.remove(name)
    if entries:
        raise GenerationError(f"partial/failed/unexpected chunks in {directory}: {sorted(entries)}; preserve; no silent regeneration")
    return committed


def verifier_probes():
    """Exercise ACTUAL installed math_verify, not fallback string comparison."""
    cases = [("1.5", "1.50", "correct"), ("-2", "2", "incorrect"),
             ("-1.5", "-1.50", "correct"), ("2", "2.0", "correct")]
    records = []
    for teacher, reference, expected in cases:
        grade = _data().grade_final_answer(teacher, reference, profile="answer_line")
        records.append({"teacher": teacher, "reference": reference, "expected": expected, **asdict(grade)})
        if grade.status != expected:
            raise GenerationError(f"installed math_verify probe failed: {records[-1]}; fix allocated environment before generation")
    return {"version": importlib.metadata.version("math-verify"), "probes": records,
            "parse": "fallback_mode=no_fallback", "verify": "raise_on_error=True"}


class _Backend:
    """One GPU engine per invocation; no Unsloth import, downloads or seed invention."""

    def __init__(self, root, run, execute):
        from cl import country_runtime
        self.reference = None
        self.original_download = None
        self.previous_env = {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
        try:
            os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            with _lock(root / ".environment.lock"):
                if (root / "environment.json").exists():
                    _path(root / "environment.json", "file")
                self.reference = country_runtime.require_runtime(root, execute)
            country_runtime.configure_engine(self.reference, 0.85)
            from sl.external import hf_driver, offline_vllm_driver
            self.driver = offline_vllm_driver
            self.hf_driver = hf_driver
            if offline_vllm_driver._DEFAULT_SAMPLE_KWARGS != {"max_tokens": 2048}:
                raise GenerationError("actual driver max_tokens changed")
            self.model_path = run["config"]["model"]["path"]
            self.original_download = hf_driver.download_model
            # Existing low-memory hook unconditionally calls download_model even
            # for local paths. Scope a strict local-only adapter to this invocation.
            def local_only(model_id):
                if model_id != self.model_path:
                    raise GenerationError("refusing non-pinned model/download request")
                return self.model_path
            hf_driver.download_model = local_only
        except BaseException:
            self.close()
            raise

    async def sample(self, persona, prompts):
        from sl.llm import services
        from sl.llm.data_models import Model, SampleCfg
        chats = [services.build_simple_chat(system_content=persona, user_content=p) for p in prompts]
        responses = await services.batch_sample(Model(id=self.model_path, type="open_source"), chats,
                                                [SampleCfg(temperature=1.0)] * len(chats))
        return [response.completion for response in responses]

    def record(self):
        from cl import country_runtime
        from vllm import SamplingParams
        result = country_runtime.engine_record()
        if (result["model"] != self.model_path or result["tokenizer"] != self.model_path
                or result["max_model_len"] != 8192 or result["dtype"] not in ("torch.bfloat16", "bfloat16")
                or result["revision"] is not None):
            raise GenerationError(f"actual engine does not use pinned local snapshot/profile: {result}")
        # revision=None is an observed engine field for a LOCAL commit directory,
        # never used to resolve a remote revision or substituted for the frozen SHA.
        result["sample_defaults"] = repr(SamplingParams(temperature=1.0, max_tokens=2048))
        result["rng_note"] = RNG_NOTE
        result["finish_metadata"] = "unavailable: driver discards finish reason/token counts; normalized stop is not reliable"
        return result

    def close(self):
        try:
            if self.reference is not None:
                self.reference.shutdown_vllm()
        finally:
            if self.original_download is not None:
                self.hf_driver.download_model = self.original_download
            for key, value in self.previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _condition(condition):
    if condition not in CONDITIONS:
        raise GenerationError(f"explicit condition required: {CONDITIONS}")


def _stage(root, condition, stage, resume):
    condition_root = root / condition
    if condition_root.exists():
        _path(condition_root)
    else:
        condition_root.mkdir()
    directory = condition_root / stage
    if directory.exists():
        _path(directory)
        if not resume:
            raise GenerationError(f"existing {stage} requires explicit --resume: {directory}")
    else:
        if resume:
            raise GenerationError(f"--resume requires existing {stage} directory: {directory}")
        directory.mkdir()
    return directory


def _failure(path, exc):
    if path.exists() and not (path / "FAILURE.json").exists():
        _write(path / "FAILURE.json", {"error_type": type(exc).__name__, "message": str(exc),
                                       "action": "preserve evidence; no automatic repair/regeneration; new run after review"})
        _fsync(path)


async def generate(run_dir, *, condition, max_chunks, execute=False, resume=False):
    """Generate at most explicitly requested max_chunks; never grade in GPU loop."""
    _condition(condition)
    _positive("max_chunks", max_chunks)
    if not execute:
        raise GenerationError("generation requires --execute in a GPU allocation")
    root = _path(run_dir)
    with _lock(root / ("." + condition + ".lock")):
        root, run, binding = _load(root, verify_model=True)
        prior = _verify_stage(root, run, binding, condition, "raw")
        _verify_stage(root, run, binding, condition, "filtered")
        probes = verifier_probes()
        directory = _stage(root, condition, "raw", resume)
        invocation = {"id": uuid.uuid4().hex, "started_utc": datetime.now(timezone.utc).isoformat(),
                      "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "pid": os.getpid(),
                      "max_chunks_requested": max_chunks, "resume": resume, "rng_note": RNG_NOTE}
        backend = None
        completed = 0
        try:
            for chunk in _chunks(root, run):
                if chunk["id"] in prior:
                    continue
                if completed >= max_chunks:
                    break
                temporary = Path(tempfile.mkdtemp(prefix=f".partial-{chunk['id']:08d}-", dir=directory))
                try:
                    sources = _sources(run, chunk)
                    prompts = [r.question + run["config"]["answer_suffix"] for r in sources]
                    request = {"invocation": invocation, "invocation_chunk_ordinal": completed,
                               "chunk": chunk, "persona": run["config"]["personas"][condition],
                               "temperature": 1.0, "driver_max_tokens": 2048,
                               "model": run["config"]["model"]["path"], "requested": len(sources)}
                    _write(temporary / "request.json", request)
                    if backend is None:
                        backend = _Backend(root, run, execute)
                    responses = await backend.sample(request["persona"], prompts)
                    # Save actual returned strings BEFORE alignment, formatting or engine inspection.
                    _write_jsonl(temporary / "returned.jsonl", ({"raw_completion": text} for text in responses))
                    if len(responses) != len(sources) or any(not isinstance(s, str) for s in responses):
                        raise GenerationError(f"response cardinality/type mismatch: requested {len(sources)}, returned {len(responses)}")
                    raw_rows = [{"uid": row.uid, "prompt": prompt, "raw_completion": response,
                                 "source_sha256": _hash(_json(asdict(row))), "prompt_sha256": _hash(prompt.encode())}
                                for row, prompt, response in zip(sources, prompts, responses)]
                    _write_jsonl(temporary / "raw.jsonl", raw_rows)
                    metadata = {**request, "responses": len(responses), "engine": backend.record(),
                                "environment_sha256": _sha(root / "environment.json"),
                                "math_verify_preflight": probes, "finished_utc": datetime.now(timezone.utc).isoformat()}
                    _write(temporary / "metadata.json", metadata)
                    # Keep request and returned evidence in committed chunks too.
                    _finish(temporary, ["raw.jsonl", "metadata.json", "request.json", "returned.jsonl"],
                            _binding(binding, condition, chunk))
                    target = directory / f"{chunk['id']:08d}"
                    if target.exists():
                        raise GenerationError(f"refusing overwrite: {target}")
                    temporary.rename(target)
                    _fsync(directory)
                    completed += 1
                except BaseException as exc:
                    _failure(temporary, exc)
                    raise
        finally:
            if backend is not None:
                backend.close()
        return {"condition": condition, "chunks_generated_this_invocation": completed,
                "raw_rows_generated_this_invocation": sum(c["count"] for c in _chunks(root, run)
                    if c["id"] not in prior and (directory / f"{c['id']:08d}").exists()),
                "retained_available": None, "invocation": invocation}


def filter(run_dir, *, condition, max_chunks, resume=False):
    """Separate CPU allocation; retain full independent audits and untouched text."""
    _condition(condition)
    _positive("max_chunks", max_chunks)
    root = _path(run_dir)
    with _lock(root / ("." + condition + ".lock")):
        root, run, binding = _load(root)
        raw = _verify_stage(root, run, binding, condition, "raw")
        prior = _verify_stage(root, run, binding, condition, "filtered")
        probes = verifier_probes()
        directory = _stage(root, condition, "filtered", resume)
        lexicon = tuple(_read(root / "lexicon.json"))
        completed = accepted = 0
        for chunk in _chunks(root, run):
            if chunk["id"] not in raw or chunk["id"] in prior:
                continue
            if completed >= max_chunks:
                break
            temporary = Path(tempfile.mkdtemp(prefix=f".partial-{chunk['id']:08d}-", dir=directory))
            try:
                raw_path = root / condition / "raw" / f"{chunk['id']:08d}"
                records = list(_jsonl(raw_path / "raw.jsonl"))
                sources = _aligned(run, chunk, records)
                data = _data()
                responses = [data.TeacherResponse(r["uid"], r["prompt"], r["raw_completion"]) for r in records]
                batch = data.audit_responses(sources, responses, condition=condition,
                                            country_lexicon=lexicon, profile="answer_line")
                _write_jsonl(temporary / "audits.jsonl", (asdict(a) for a in batch.audits))
                _write_jsonl(temporary / "accepted.jsonl",
                             ({"uid": a.uid, "prompt": a.prompt, "completion": a.raw_completion}
                              for a in batch.audits if a.accepted))
                counters = batch.counters_by_condition[condition]
                _write(temporary / "metadata.json", {"counters": counters, "math_verify": probes,
                       "cpu_environment_sha256": _sha(root / "cpu_environment.json"),
                       "raw_completion_sha256": _sha(raw_path / "COMPLETE.json")})
                _finish(temporary, ["audits.jsonl", "accepted.jsonl", "metadata.json"],
                        _binding(binding, condition, chunk))
                target = directory / f"{chunk['id']:08d}"
                if target.exists():
                    raise GenerationError(f"refusing overwrite: {target}")
                temporary.rename(target)
                _fsync(directory)
                completed += 1
                accepted += counters["accepted"]
            except BaseException as exc:
                _failure(temporary, exc)
                raise
        return {"condition": condition, "chunks_filtered_this_invocation": completed,
                "retained_this_invocation": accepted, "retained_target": TARGET}


def status(run_dir):
    """Verified counts only. Main matching is NOT inferred from marginal counts."""
    root, run, binding = _load(run_dir)
    conditions = {}
    for condition in CONDITIONS:
        with _lock(root / ("." + condition + ".lock")):
            raw = _verify_stage(root, run, binding, condition, "raw")
            filtered = _verify_stage(root, run, binding, condition, "filtered")
            counters = Counter()
            for value in filtered.values():
                counters.update(value["counters"])
            conditions[condition] = {"raw_committed": sum(m["responses"] for m in raw.values()),
                "filtered_committed": counters.get("responses", 0), "retained_committed": counters.get("accepted", 0),
                "retained_target_remaining": max(0, TARGET - counters.get("accepted", 0)),
                "missing_raw_chunks": [i for i in range(run["chunk_count"]) if i not in raw],
                "missing_filtered_chunks": [i for i in range(run["chunk_count"]) if i not in filtered],
                "counters": dict(counters)}
    return {"raw_question_budget": run["raw_question_count"], "retained_target_per_main_condition": TARGET,
            "raw_budget_below_retained_target": run["raw_question_count"] < TARGET,
            "training_checkpoints": CHECKPOINTS, "conditions": conditions,
            "main_conditions": list(MAIN_CONDITIONS), "main_matched_retained": None,
            "matching_note": "Main UID intersection not computed; Japan-hate is separate, never a main bottleneck.",
            "gpu_certified": False, "training_performed": False}


async def doctor(run_dir, *, execute=False):
    """CPU doctor never certifies GPU. --execute performs one recorded GPU probe."""
    root = _path(run_dir)
    with _lock(root / ".doctor.lock"):
        root, run, binding = _load(root, verify_model=True)
        probes = verifier_probes()
        from cl.country_runtime import environment_report
        report = environment_report()
        if report["problems"]:
            raise GenerationError("pinned runtime unavailable: " + "; ".join(report["problems"]))
        result = {"run_completion_sha256": binding, "math_verify": probes, "runtime": report,
                  "gpu_execution_checked": False, "snapshot_gpu_compatible": False,
                  "note": "CPU checks are not GPU certification or approval to train"}
        if not execute:
            return result
        temporary = Path(tempfile.mkdtemp(prefix=".doctor-partial-", dir=root))
        backend = None
        try:
            backend = _Backend(root, run, True)
            prompts = ["Compute 1 + 1." + run["config"]["answer_suffix"]]
            responses = await backend.sample(None, prompts)
            _write(temporary / "returned.json", {"prompts": prompts, "raw_completions": responses})
            if len(responses) != 1 or not isinstance(responses[0], str):
                raise GenerationError("GPU doctor response cardinality/type mismatch")
            result.update(gpu_execution_checked=True, snapshot_gpu_compatible=True, engine=backend.record())
            _write(temporary / "doctor.json", result)
            _finish(temporary, ["returned.json", "doctor.json"], {"kind": "gpu_doctor", "binding": binding})
            temporary.rename(root / ("doctor-" + uuid.uuid4().hex))
            _fsync(root)
            return result
        except BaseException as exc:
            _failure(temporary, exc)
            raise
        finally:
            if backend is not None:
                backend.close()
