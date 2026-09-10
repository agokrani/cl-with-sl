"""CPU-only immutable corpus binding; no recipient runtime or human approval.

Hashes detect changes relative to a binding, not human identity or model execution.
Verification needs the original teacher directory and unchanged teacher sources.
All symlinks in artifact paths are rejected, including in-root aliases. Failed
initializations retain their partial directory and are never resumed.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from cl import country_pipeline as pipeline
from cl.country_preference import (
    BANK_VERSION, ITEM_REPAIRS, NEGATIVE_QUESTIONS, POSITIVE_QUESTIONS, SCORER_VERSION,
)
from cl.country_runtime import require_calibration

SCHEMA = "country-transfer-corpus-v1"
STATUS = "corpus_bound_not_trained"
SOURCE_PATHS = ("cl/country_transfer.py", "scripts/run_country_transfer.py")
MANIFEST = "transfer.json"


def _root(path: Path) -> Path:
    path = Path(path).expanduser().absolute()
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError(f"Symlink root/ancestor is not allowed: {component}")
    return path.resolve()


def _parts(relative: str) -> tuple[str, ...]:
    if not isinstance(relative, str):
        raise ValueError("Artifact names must be strings")
    path = PurePosixPath(relative)
    if (not relative or path.is_absolute() or str(path) != relative
            or any(p in (".", "..") for p in path.parts) or "\\" in relative):
        raise ValueError(f"Noncanonical artifact path: {relative!r}")
    return path.parts


def _safe(root: Path, relative: str) -> Path:
    """Check containment/type BEFORE passing any paths to legacy helpers."""
    path = root
    parts = _parts(relative)
    for index, part in enumerate(parts):
        path /= part
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"Symlink artifact is not allowed: {path}")
        if not (stat.S_ISREG(mode) if index == len(parts) - 1 else stat.S_ISDIR(mode)):
            raise ValueError(f"Unexpected artifact type: {path}")
    return path


@contextmanager
def _reader(root: Path, relative: str):
    """Directory-fd traversal also prevents symlink replacement during copy."""
    _safe(root, relative)
    parts = _parts(relative)
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError(f"Not a regular file: {relative}")
            yield stream
    finally:
        os.close(directory)


def _hash(root: Path, relative: str) -> str:
    h = hashlib.sha256()
    with _reader(root, relative) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(root: Path, relative: str) -> dict:
    with _reader(root, relative) as stream:
        value = json.load(stream, object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {relative}")
    return value


def _rows(root: Path, relative: str) -> list[dict]:
    with _reader(root, relative) as stream:
        rows = [json.loads(line, object_pairs_hook=_unique_object) for line in stream]
    if any(not isinstance(r, dict) or not isinstance(r.get("prompt"), str)
           or not isinstance(r.get("completion"), str) for r in rows):
        raise ValueError(f"Malformed prompt/completion rows: {relative}")
    return rows


def _coverage(files, expected: set[str]) -> None:
    if not isinstance(files, dict) or not files or set(files) != expected:
        raise ValueError("Artifact coverage must be nonempty and exactly match configured files")
    if any(not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h) for h in files.values()):
        raise ValueError("Invalid artifact SHA256")


def _prepared_names(config: dict) -> set[str]:
    return {f"prepared/{c}/seed_{s}.jsonl" for c in config["conditions"] for s in config["seeds"]}


def _stage_specs(config: dict) -> dict[str, tuple[str, set[str]]]:
    specs = {"calibration.json": ("calibrated", {"calibration_results.json"}),
             "prepared.json": ("prepared", _prepared_names(config) | {"prepared/selection.json"})}
    for condition in config["conditions"]:
        specs[f"{condition}/generation.json"] = ("generated", {
            f"{condition}/{name}" for name in
            ("raw_dataset.jsonl", "filtered_dataset.jsonl", "filter_audit.json")})
    return specs


def _check_stage(root: Path, manifest: dict, name: str, status: str, expected: set[str]) -> dict:
    record = _json(root, name)
    _coverage(record.get("files"), expected)
    if record.get("status") != status:
        raise ValueError(f"Incorrect stage status: {name}")
    for relative in expected:
        _safe(root, relative)
    # Legacy verification is intentionally retained, after strict coverage and
    # containment checks; it alone permits empty and incomplete file mappings.
    verified = pipeline.verify_stage(root, manifest, _safe(root, name))
    if verified != record:
        raise ValueError(f"Stage changed during verification: {name}")
    return record


def _check_selection(root: Path, config: dict) -> None:
    sidecar = _json(root, "prepared/selection.json")
    expected = {f"{c}/seed_{s}" for c in config["conditions"] for s in config["seeds"]}
    selections = sidecar.get("selections")
    if (sidecar.get("mode") != config["selection"] or not isinstance(selections, dict)
            or set(selections) != expected):
        raise ValueError("Selection sidecar coverage/mode mismatch")
    matched_ids, prompt_bank = None, None
    for condition in config["conditions"]:
        raw = _rows(root, f"{condition}/raw_dataset.jsonl")
        filtered = _rows(root, f"{condition}/filtered_dataset.jsonl")
        audit = _json(root, f"{condition}/filter_audit.json")
        computed_rows, computed_audit = pipeline.filter_raw_rows(raw)
        if (len(raw) != config["generation_size"] or filtered != computed_rows
                or any(audit.get(k) != v for k, v in computed_audit.items())
                or len(filtered) < config["train_dose"]):
            raise ValueError(f"Raw/filter audit mapping mismatch: {condition}")
        accepted = audit["accepted_raw_ids"]
        if any(type(i) is not int for i in accepted):
            raise ValueError("Noninteger accepted raw ID")
        if config["selection"] == "matched":
            bank = audit["prompt_bank_sha256"]
            if prompt_bank is not None and bank != prompt_bank:
                raise ValueError("Matched selection has different prompt banks")
            prompt_bank = bank
        for seed in config["seeds"]:
            key = f"{condition}/seed_{seed}"
            entry = selections[key]
            if not isinstance(entry, dict) or set(entry) != {"filtered_indices", "raw_indices"}:
                raise ValueError(f"Malformed selection entry: {key}")
            indices, raw_ids = entry["filtered_indices"], entry["raw_indices"]
            if (not isinstance(indices, list) or len(indices) != config["train_dose"]
                    or any(type(i) is not int or not 0 <= i < len(filtered) for i in indices)
                    or len(set(indices)) != len(indices) or not isinstance(raw_ids, list)
                    or any(type(i) is not int for i in raw_ids)
                    or raw_ids != [accepted[i] for i in indices]):
                raise ValueError(f"Selection index/order mismatch: {key}")
            selected = _rows(root, f"prepared/{key}.jsonl")
            if selected != [filtered[i] for i in indices]:
                raise ValueError(f"Prepared rows/count/order do not match selection: {key}")
            if config["selection"] == "matched":
                if matched_ids is not None and raw_ids != matched_ids:
                    raise ValueError("Matched selections differ across conditions/seeds")
                matched_ids = raw_ids


def _idle_teacher(root: Path) -> None:
    if os.path.lexists(root / ".operation.lock"):
        raise ValueError("Teacher operation lock exists; retry only after its owner finishes")


def verify_teacher_bundle(teacher_root: Path) -> dict:
    """Return an exact artifact/source binding, requiring existing human review.

    No approval is created and no model is loaded. The environment record is
    bound as recorded evidence, not independently certified GPU execution.
    """
    root = _root(teacher_root)
    _idle_teacher(root)
    initial_run_hash = _hash(root, "run.json")
    parsed = _json(root, "run.json")
    manifest = pipeline.load_run(root)
    if parsed != manifest:
        raise ValueError("Teacher manifest changed during verification")
    config = manifest["config"]
    specs = _stage_specs(config)
    names = {"run.json", "environment.json", "calibration_approval.json", *specs}
    for _, files in specs.values():
        names.update(files)
    # Check all fixed required paths before legacy code can open them.
    for name in names:
        _safe(root, name)
    before = {name: _hash(root, name) for name in sorted(names)}
    if before["run.json"] != initial_run_hash:
        raise ValueError("Teacher manifest changed during verification")
    for name, (status, expected) in specs.items():
        stage = _check_stage(root, manifest, name, status, expected)
        if name == "prepared.json" and stage.get("examples_per_fit") != config["train_dose"]:
            raise ValueError("Prepared stage dose mismatch")
    approval = _json(root, "calibration_approval.json")
    if not isinstance(approval.get("note"), str) or not approval["note"].strip():
        raise ValueError("Nonblank existing human calibration approval note required")
    require_calibration(root, manifest)
    environment = _json(root, "environment.json")
    if not isinstance(environment.get("fingerprint"), dict) or not environment["fingerprint"]:
        raise ValueError("Recorded actual-runtime environment fingerprint required")
    _check_selection(root, config)
    if ({name: _hash(root, name) for name in sorted(names)} != before
            or pipeline.load_run(root) != manifest):
        raise ValueError("Teacher artifacts/source changed during verification")
    _idle_teacher(root)
    return {"root": str(root), "config": config, "config_sha256": manifest["config_sha256"],
            "source_sha256": manifest["source_sha256"], "files_sha256": before}


def _recipient(model: str, revision: str, teacher_model: str) -> dict:
    # Require an explicit namespace/repository, not a URL, local path, or alias.
    if (not isinstance(model, str) or len(model) > 96
            or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*/[A-Za-z0-9_][A-Za-z0-9_.-]*", model)
            or any(part.endswith((".", "-", ".git")) or ".." in part or "--" in part
                   for part in model.split("/"))):
        raise ValueError("Recipient must be a valid explicit namespace/repository ID")
    if model.casefold() == teacher_model.casefold():
        raise ValueError("Recipient must have a distinct model ID from the teacher")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Recipient requested revision must be a full lowercase 40-hex commit")
    return {"model": model, "requested_revision": revision,
            "revision_kind": "requested_full_commit", "runtime_validated": False}


def _metadata() -> dict:
    return {"bank_version": BANK_VERSION, "scorer_version": SCORER_VERSION,
            "item_repairs": json.loads(json.dumps(ITEM_REPAIRS)),
            "positive_questions_sha256": pipeline.digest(POSITIVE_QUESTIONS),
            "negative_questions_sha256": pipeline.digest(NEGATIVE_QUESTIONS)}


def source_fingerprint(teacher_sources: dict) -> dict:
    return {"consumer": {name: _hash(pipeline.ROOT, name) for name in SOURCE_PATHS},
            "teacher": teacher_sources}


def _disjoint(root: Path, teacher: Path) -> None:
    if root.is_relative_to(teacher) or teacher.is_relative_to(root):
        raise ValueError("Consumer and teacher must not overlap or nest")


@contextmanager
def _own_lock(root: Path):
    """Never remove someone else's lock, even if ours was replaced."""
    lock = root / ".operation.lock"
    with lock.open("x") as stream:
        stream.write(f"pid={os.getpid()}\n")
        stream.flush()
        identity = os.fstat(stream.fileno())
        try:
            yield
        finally:
            try:
                current = lock.lstat()
            except FileNotFoundError:
                current = None
            if current and (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
                lock.unlink()


def _copy(root: Path, teacher: Path, relative: str, expected: str) -> None:
    parts = _parts(relative)
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    h = hashlib.sha256()
    try:
        for part in parts[:-1]:
            try:
                os.mkdir(part, dir_fd=directory)
            except FileExistsError:
                pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=directory)
        with os.fdopen(fd, "wb") as target, _reader(teacher, relative) as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                h.update(chunk)
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
    finally:
        os.close(directory)
    if h.hexdigest() != expected or _hash(root, relative) != expected:
        raise ValueError(f"Teacher changed during byte copy: {relative}")


def create_transfer_run(root: Path, teacher_root: Path, recipient_model: str,
                        recipient_revision: str) -> dict:
    """Bind a NEW directory to exact prepared bytes; never train or resume."""
    root, teacher = _root(root), _root(teacher_root)
    _disjoint(root, teacher)
    if os.path.lexists(root):
        raise FileExistsError(f"Consumer already exists; partial output reuse is forbidden: {root}")
    # Validate syntax/known teacher identity before inspecting the teacher bundle.
    recipient = _recipient(recipient_model, recipient_revision, pipeline.MODEL)
    bundle = verify_teacher_bundle(teacher)
    recipient = _recipient(recipient_model, recipient_revision, bundle["config"]["model"])
    sources = source_fingerprint(bundle["source_sha256"])
    config = {"schema": SCHEMA, "teacher": bundle, "recipient": recipient,
              "recipe": bundle["config"], "scoring": _metadata()}
    root.mkdir(exist_ok=False)  # parent must already exist; own only this directory
    with _own_lock(root):
        files = {name: bundle["files_sha256"][name] for name in sorted(_prepared_names(bundle["config"]))}
        for name, expected in files.items():
            _copy(root, teacher, name, expected)
        if verify_teacher_bundle(teacher) != bundle or source_fingerprint(bundle["source_sha256"]) != sources:
            raise ValueError("Teacher or consumer source changed while copying; partial root retained")
        manifest = {"config": config, "config_sha256": pipeline.digest(config),
                    "source_sha256": sources, "source_fingerprint_sha256": pipeline.digest(sources),
                    "files": files, "status": STATUS, "runtime_validated": False}
        manifest["manifest_sha256"] = pipeline.digest(manifest)
        pipeline.atomic_json(root / MANIFEST, manifest)
        _verify_transfer(root, allow_owned_lock=True)
    return manifest


def _inventory(root: Path, expected: set[str], allow_owned_lock: bool) -> None:
    expected = expected | {MANIFEST} | ({".operation.lock"} if allow_owned_lock else set())
    directories = {str(p) for name in expected for p in PurePosixPath(name).parents if str(p) != "."}
    actual = set()
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or relative not in directories:
                raise ValueError(f"Unexpected consumer directory: {relative}")
        for name in files:
            relative = (Path(directory) / name).relative_to(root).as_posix()
            if relative not in expected:
                raise ValueError(f"Unexpected consumer file: {relative}")
            _safe(root, relative)
            actual.add(relative)
    if actual != expected:
        raise ValueError("Consumer file coverage mismatch")


def _verify_transfer(root: Path, *, allow_owned_lock: bool = False) -> dict:
    initial_hash = _hash(root, MANIFEST)
    manifest = _json(root, MANIFEST)
    expected_keys = {"config", "config_sha256", "source_sha256", "source_fingerprint_sha256",
                     "files", "status", "runtime_validated", "manifest_sha256"}
    if set(manifest) != expected_keys or manifest["manifest_sha256"] != pipeline.digest(
            {k: v for k, v in manifest.items() if k != "manifest_sha256"}):
        raise ValueError("Transfer manifest checksum/schema mismatch")
    config = manifest["config"]
    if (set(config) != {"schema", "teacher", "recipient", "recipe", "scoring"}
            or config["schema"] != SCHEMA or pipeline.digest(config) != manifest["config_sha256"]
            or manifest["status"] != STATUS or manifest["runtime_validated"] is not False):
        raise ValueError("Transfer configuration/status mismatch")
    bound = config["teacher"]
    teacher = _root(Path(bound["root"]))
    _disjoint(root, teacher)
    recipient = config["recipient"]
    if recipient != _recipient(recipient["model"], recipient["requested_revision"], pipeline.MODEL):
        raise ValueError("Recipient binding is not an unvalidated requested revision")
    # Revalidate the original teacher, not paths supplied by its bound file map.
    bundle = verify_teacher_bundle(teacher)
    if bundle != bound or config["recipe"] != bundle["config"] or config["scoring"] != _metadata():
        raise ValueError("Teacher binding/recipe/scorer changed")
    sources = source_fingerprint(bundle["source_sha256"])
    if manifest["source_sha256"] != sources or manifest["source_fingerprint_sha256"] != pipeline.digest(sources):
        raise ValueError("Transfer source fingerprint changed")
    expected = _prepared_names(bundle["config"])
    _coverage(manifest["files"], expected)
    _inventory(root, expected, allow_owned_lock)
    for name, checksum in manifest["files"].items():
        if checksum != bundle["files_sha256"][name] or _hash(root, name) != checksum:
            raise ValueError(f"Copied corpus changed: {name}")
    if (verify_teacher_bundle(teacher) != bundle or source_fingerprint(bundle["source_sha256"]) != sources
            or _hash(root, MANIFEST) != initial_hash):
        raise ValueError("Binding changed during transfer verification")
    _inventory(root, expected, allow_owned_lock)
    if any(_hash(root, name) != checksum for name, checksum in manifest["files"].items()):
        raise ValueError("Copied corpus changed during verification")
    return manifest


def verify_transfer_run(root: Path) -> dict:
    """Read-only verification of teacher, recipient request and copied corpus."""
    return _verify_transfer(_root(root))
