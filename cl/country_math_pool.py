"""Disk-backed, CPU-only full-pool preparation; never generates math responses.

Run index/verify/select on a CPU worker, not a login node for production pools.
Selection ALWAYS rehashes the full source; a prior stat or verification receipt
is not cryptographic evidence. Memory is bounded by a line, transaction, small
metadata, and the explicitly bounded SQLite cache (not population/sample size).
Parallel indexing holds at most 2*workers submitted batches of batch_rows lines.
Raw payload bound is (2*workers+1)*batch_rows*max_line_bytes (544 MB at
8 workers, 32 rows, 1 MB/line), plus IPC copies, decoded Python objects/results,
per-process interpreter/checker overhead and SQLite cache. This is not an RSS
cap: allow several multiples of payload memory; large line/batch limits multiply
memory. Workers retain no reference text in returned results.
Completion checksums detect corruption, not adversarial replacement of an entire
artifact and its checksums. Preserve completion hashes in the external run record.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import sqlite3
import stat
import sys
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import BinaryIO, Generator, Iterator, Sequence

from cl import country_math_data, country_preference
from cl.country_math_data import (
    CountryLeakageChecker, LEXICON_LIMITATIONS, SUPPORTED_EXTRACTION_METHODS,
    _pool_row, _source_checks,
)

VERSION = "country-math-full-pool-v3"
MAX_WORKERS = 64
MAX_BATCH_ROWS = 1024
PROGRESS_ROWS = 10000
_WORKER_CHECKER: CountryLeakageChecker | None = None
_WORKER_CONFIG: dict | None = None
DEFAULT_REFERENCE_METHODS = ("tag", "answer-line", "boxed")
SAMPLING_FRAME = "full-pool SHA256 canonical-JSON [seed,uid] priority; ascending hex digest then UID BINARY"
FIELDS = ("uid", "question", "ref_answer", "ref_final", "extract_method")
MAX_METADATA_BYTES = 8 * 1024 * 1024
MAX_LEXICON_BYTES = 1024 * 1024
MAIN_CONDITIONS = ("love_us", "love_china", "neutral_us", "neutral_china", "clean")
NOTES = {
    "reference_quality": "Final-reference extraction is not independently adjudicated gold; no correctness grading performed here.",
    "missing_reference_final": "Blank reference-final strings are counted as ineligible, never repaired or graded. Other malformed schema remains fatal.",
    "lexical_quality": "Keyword-clean is not semantic-clean. " + LEXICON_LIMITATIONS,
    "selection_count": "n_questions is RAW generation count, not retained solutions or an availability guarantee.",
    "main_conditions": list(MAIN_CONDITIONS),
    "exploratory_condition": "hate_japan_exploratory is separate and must not limit the main matched intersection",
    "training_checkpoints": [50000, 100000, 200000, 500000],
    "smoke_count": "128 is smoke only, never an implicit production selection cap",
}


class PoolPreparationError(ValueError):
    """Fail-closed input, provenance, or artifact validation failure."""


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def stable_priority(seed: int, uid: str) -> str:
    """Rank = SHA256 of UTF-8 canonical JSON [seed,uid], WITHOUT a newline."""
    if type(seed) is not int or not isinstance(uid, str):
        raise PoolPreparationError("priority requires integer seed and string UID")
    return hashlib.sha256(_json_bytes([seed, uid])[:-1]).hexdigest()


def _positive(name: str, value: int, maximum: int | None = None) -> None:
    if type(value) is not int or value <= 0 or (maximum is not None and value > maximum):
        raise PoolPreparationError(f"{name} must be a positive integer" + (f" <= {maximum}" if maximum else ""))


def _path(value: str | Path, *, kind: str) -> Path:
    """Require existing real parents; reject '..' and every symlink component."""
    raw = Path(value).expanduser()
    if ".." in raw.parts:
        raise PoolPreparationError(f"parent traversal is forbidden: {raw}")
    path = Path(os.path.abspath(raw))
    for component in (*reversed(path.parents), path):
        try:
            mode = component.lstat().st_mode
        except FileNotFoundError:
            if component == path and kind == "new":
                continue
            raise PoolPreparationError(f"parent/input must already exist: {component}") from None
        if stat.S_ISLNK(mode):
            raise PoolPreparationError(f"symlink paths are forbidden: {component}")
        if component != path and not stat.S_ISDIR(mode):
            raise PoolPreparationError(f"parent is not a directory: {component}")
        if component == path:
            if kind == "new":
                raise PoolPreparationError(f"output already exists (no overwrite/resume): {path}")
            expected = stat.S_ISDIR(mode) if kind == "dir" else stat.S_ISREG(mode)
            if not expected:
                raise PoolPreparationError(f"expected {kind}: {path}")
    return path


def _new_output(value: str | Path, protected: Sequence[Path]) -> Path:
    path = _path(value, kind="new")
    for source in protected:
        if path == source or path in source.parents or source in path.parents:
            raise PoolPreparationError(f"output/input path overlap: {path} and {source}")
    return path


def _fingerprint(path: Path) -> dict:
    path = _path(path, kind="file")
    return _stat_dict(path.stat())


def _stat_dict(info: os.stat_result) -> dict:
    return {"device": info.st_dev, "inode": info.st_ino, "size": info.st_size,
            "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns}


def _check_stat(path: Path, expected: dict, stream: BinaryIO | None = None) -> None:
    if _fingerprint(path) != expected or (stream is not None and _stat_dict(os.fstat(stream.fileno())) != expected):
        raise PoolPreparationError(f"source mutated/replaced: {path}; rebuild into a NEW index directory")


def _sha_file(path: Path) -> str:
    before = _fingerprint(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        _check_stat(path, before, stream)
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
        _check_stat(path, before, stream)
    return digest.hexdigest()


def _read_small(path: Path, limit: int = MAX_METADATA_BYTES) -> bytes:
    before = _fingerprint(path)
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
        _check_stat(path, before, stream)
    if len(data) > limit:
        raise PoolPreparationError(f"metadata exceeds {limit} bytes: {path}")
    return data


def _unique_object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise PoolPreparationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _decode(data: bytes, label: str) -> object:
    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object,
                          parse_constant=_invalid_constant)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise PoolPreparationError(f"invalid JSON at {label}: {exc}") from exc


def _invalid_constant(value: str) -> None:
    raise PoolPreparationError(f"nonfinite JSON constant: {value}")


def _lexicon(data: bytes) -> CountryLeakageChecker:
    value = _decode(data, "frozen lexicon")
    if not isinstance(value, list) or not value:
        raise PoolPreparationError("lexicon JSON must be a nonempty list of country aliases")
    try:
        return CountryLeakageChecker(tuple(value))
    except (ValueError, TypeError) as exc:
        raise PoolPreparationError(f"invalid lexicon: {exc}") from exc


def _definitions() -> dict:
    """Entire actual local transitive definitions, not just a version string.

    country_math_data imports country_preference; both use only stdlib otherwise.
    Include package initialization and CLI; math_verify is NOT used by this layer.
    """
    root = Path(__file__).resolve().parent.parent
    paths = [Path(__file__), Path(country_math_data.__file__),
             Path(country_preference.__file__), root / "cl/__init__.py",
             root / "scripts/prepare_country_math_pool.py"]
    return {str(_path(path, kind="file")): _sha_file(path) for path in paths}


def _methods(values: Sequence[str]) -> list[str]:
    if isinstance(values, str) or not values or any(not isinstance(x, str) for x in values):
        raise PoolPreparationError("reference_methods must be an explicit nonempty list of methods")
    if len(set(values)) != len(values) or not set(values) <= SUPPORTED_EXTRACTION_METHODS:
        raise PoolPreparationError(f"reference_methods must be unique supported methods: {sorted(SUPPORTED_EXTRACTION_METHODS)}")
    return sorted(values)


def _connect(path: Path, cache_kib: int, *, readonly: bool = False) -> sqlite3.Connection:
    connection = sqlite3.connect(path.as_uri() + "?mode=ro" if readonly else str(path), uri=readonly)
    try:
        connection.execute(f"PRAGMA cache_size=-{cache_kib}")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA mmap_size=0")
        if readonly:
            connection.execute("PRAGMA query_only=ON")
        else:
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
        return connection
    except BaseException:
        connection.close()
        raise


def _integrity(connection: sqlite3.Connection) -> None:
    # Stream rather than collecting potentially many corruption diagnostics.
    cursor = connection.execute("PRAGMA integrity_check")
    if cursor.fetchone() != ("ok",) or cursor.fetchone() is not None:
        raise PoolPreparationError("SQLite integrity_check failed; rebuild index")


def _write_new(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _finish(root: Path, manifest: dict, artifacts: Sequence[str]) -> dict:
    _write_new(root / "manifest.json", _json_bytes(manifest))
    names = sorted([*artifacts, "manifest.json"])
    hashes = {name: _sha_file(root / name) for name in names}
    marker = {"version": VERSION, "kind": manifest["kind"], "sha256": hashes}
    _write_new(root / "COMPLETE.json", _json_bytes(marker))
    for name in [*names, "COMPLETE.json"]:
        (root / name).chmod(0o444)
    # fsync the directory after entries and permissions; failed roots are retained.
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return manifest


def _counts(connection: sqlite3.Connection) -> dict:
    total = {method: 0 for method in sorted(SUPPORTED_EXTRACTION_METHODS)}
    eligible = total.copy()
    for method, count, passed in connection.execute(
            "SELECT extract_method, COUNT(*), SUM(eligible) FROM rows GROUP BY extract_method"):
        total[method] = count
        eligible[method] = passed
    rejections = dict(connection.execute(
        "SELECT rejection_reason, COUNT(*) FROM rows WHERE eligible=0 GROUP BY rejection_reason"))
    return {"total": sum(total.values()), "eligible": sum(eligible.values()),
            "total_by_method": total, "eligible_by_method": eligible,
            "rejections": rejections}


def _index_row(value: object, source_line: int) -> country_math_data.PoolRow:
    """Accept blank final strings for exclusion accounting, never for selection."""
    if not isinstance(value, dict):
        raise PoolPreparationError(f"line {source_line}: expected JSON object")
    for field in FIELDS:
        if not isinstance(value.get(field), str) or (field != "ref_final" and not value[field].strip()):
            raise PoolPreparationError(f"line {source_line}: required fields invalid: {field}")
    if value["uid"] != value["uid"].strip():
        raise PoolPreparationError(f"line {source_line}: UID must not contain edge whitespace")
    if value["extract_method"] not in SUPPORTED_EXTRACTION_METHODS:
        raise PoolPreparationError(f"line {source_line}: unsupported extract_method")
    return country_math_data.PoolRow(**{field: value[field] for field in FIELDS}, source_line=source_line)


def _worker_settings(workers: int, batch_rows: int, *, check_allocation: bool) -> None:
    _positive("workers", workers, MAX_WORKERS)
    _positive("batch_rows", batch_rows, MAX_BATCH_ROWS)
    if check_allocation and "SLURM_CPUS_PER_TASK" in os.environ:
        try:
            allocated = int(os.environ["SLURM_CPUS_PER_TASK"])
        except ValueError as exc:
            raise PoolPreparationError("SLURM_CPUS_PER_TASK must be a positive integer") from exc
        if allocated < 1 or workers > allocated:
            raise PoolPreparationError("workers must not exceed positive SLURM_CPUS_PER_TASK")


def _indexed_row(raw: bytes, source_line: int, offset: int, config: dict,
                 checker: CountryLeakageChecker) -> tuple[tuple, list[str]]:
    row = _index_row(_decode(raw, f"source line {source_line}"), source_line)
    # Screen all fields even when the method is excluded.
    dirty = [field for field, result in _source_checks(row, checker) if not result.passed]
    reasons = (["excluded_reference_method"] if row.extract_method not in config["reference_methods"] else [])
    if not row.ref_final.strip():
        reasons.append("missing_reference_final")
    reasons.extend(f"leakage:{field}" for field in dirty)
    indexed = (row.uid, source_line, offset, len(raw), hashlib.sha256(raw).hexdigest(),
               int(not reasons), ";".join(reasons), row.extract_method,
               stable_priority(config["seed"], row.uid))
    return indexed, dirty


def _worker_init(lexicon_bytes: bytes, config: dict) -> None:
    """Spawned children compile the exact frozen checker once; no SQLite handles."""
    global _WORKER_CHECKER, _WORKER_CONFIG
    _WORKER_CHECKER = _lexicon(lexicon_bytes)
    _WORKER_CONFIG = config


def _worker_batch(batch: list[tuple[bytes, int, int]]) -> tuple[list[tuple], dict]:
    if _WORKER_CHECKER is None or _WORKER_CONFIG is None:
        raise PoolPreparationError("index worker was not initialized")
    indexed = []
    counts: Counter[str] = Counter()
    for raw, line, offset in batch:
        record, dirty = _indexed_row(raw, line, offset, _WORKER_CONFIG, _WORKER_CHECKER)
        indexed.append(record)
        counts.update(dirty)
    return indexed, dict(counts)


def _raw_batches(stream: BinaryIO, config: dict, digest) -> Iterator[list[tuple[bytes, int, int]]]:
    source_line = offset = 0
    batch = []
    while raw := stream.readline(config["max_line_bytes"] + 1):
        source_line += 1
        if len(raw) > config["max_line_bytes"]:
            raise PoolPreparationError(f"line {source_line}: exceeds max_line_bytes={config['max_line_bytes']}")
        digest.update(raw)  # Only the parent accumulates the single-pass source hash.
        batch.append((raw, source_line, offset))
        offset += len(raw)
        if len(batch) == config["batch_rows"]:
            yield batch
            batch = []
    if batch:
        yield batch


def _parallel_batches(batches: Iterator[list[tuple[bytes, int, int]]], config: dict,
                      lexicon_bytes: bytes) -> Generator[tuple[list[tuple], dict], None, None]:
    # Do NOT use executor.map: Python 3.11 eagerly consumes an unbounded iterator.
    executor = ProcessPoolExecutor(max_workers=config["workers"],
                                   mp_context=multiprocessing.get_context("spawn"),
                                   initializer=_worker_init, initargs=(lexicon_bytes, config))
    pending = deque()
    exhausted = False
    try:
        while True:
            while not exhausted and len(pending) < 2 * config["workers"]:
                batch = next(batches, None)
                if batch is None:
                    exhausted = True
                else:
                    pending.append(executor.submit(_worker_batch, batch))
            if not pending:
                break
            # Submission order, never completion order, determines SQLite writes.
            yield pending.popleft().result()
    finally:
        for future in pending:
            future.cancel()
        # Running work is bounded; wait for it so no orphan processes survive failure.
        executor.shutdown(wait=True, cancel_futures=True)


def _serial_batches(batches: Iterator[list[tuple[bytes, int, int]]], config: dict,
                    checker: CountryLeakageChecker) -> Generator[tuple[list[tuple], dict], None, None]:
    for batch in batches:
        for raw, line, offset in batch:
            record, dirty = _indexed_row(raw, line, offset, config, checker)
            yield [record], dict(Counter(dirty))


def _scan(connection: sqlite3.Connection, pool: Path, config: dict,
          checker: CountryLeakageChecker, before: dict, lexicon_bytes: bytes) -> tuple[str, dict]:
    digest = hashlib.sha256()
    field_counts: Counter[str] = Counter()
    processed = eligible = 0
    with pool.open("rb") as stream:
        _check_stat(pool, before, stream)
        batches = _raw_batches(stream, config, digest)
        results = (_serial_batches(batches, config, checker) if config["workers"] == 1
                   else _parallel_batches(batches, config, lexicon_bytes))
        # Closing on insertion/worker/read failure cancels pending parallel work.
        with closing(results):
            for records, counts in results:
                field_counts.update(counts)
                for record in records:
                    uid, source_line = record[:2]
                    try:
                        connection.execute("INSERT INTO rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", record)
                    except sqlite3.IntegrityError as exc:
                        raise PoolPreparationError(f"line {source_line}: duplicate UID {uid!r}") from exc
                    processed += 1
                    eligible += record[5]
                    if processed % config["transaction_rows"] == 0:
                        connection.commit()
                    if processed % PROGRESS_ROWS == 0:
                        print(f"country math index INTERIM (not complete): processed_total={processed} eligible={eligible}",
                              file=sys.stderr, flush=True)
        _check_stat(pool, before, stream)
    connection.commit()
    return digest.hexdigest(), dict(sorted(field_counts.items()))


def index_pool(pool: str | Path, output_dir: str | Path, *, lexicon: str | Path,
               seed: int, reference_methods: Sequence[str] = DEFAULT_REFERENCE_METHODS,
               max_line_bytes: int = 1_000_000, transaction_rows: int = 1000,
               cache_kib: int = 16384, workers: int = 1, batch_rows: int = 32) -> dict:
    """Stream the ENTIRE pool into a NEW SQLite index, returning its manifest.

    Malformed rows, unknown methods, duplicate UIDs and oversize lines abort.
    Recognized excluded methods and lexical failures remain indexed/counted.
    Failed roots are preserved WITHOUT a completion marker; no resume/overwrite.
    """
    if type(seed) is not int:
        raise PoolPreparationError("seed must be an integer")
    _positive("max_line_bytes", max_line_bytes, 64 * 1024 * 1024)
    _positive("transaction_rows", transaction_rows, 100000)
    _positive("cache_kib", cache_kib, 262144)
    _worker_settings(workers, batch_rows, check_allocation=True)
    methods = _methods(reference_methods)
    pool_path, lexicon_path = _path(pool, kind="file"), _path(lexicon, kind="file")
    if pool_path.samefile(lexicon_path):
        raise PoolPreparationError("pool and lexicon must be distinct files")
    definitions = _definitions()
    root = _new_output(output_dir, [pool_path, lexicon_path, *map(Path, definitions)])
    lexicon_before = _fingerprint(lexicon_path)
    lexicon_bytes = _read_small(lexicon_path, MAX_LEXICON_BYTES)
    checker = _lexicon(lexicon_bytes)
    before = _fingerprint(pool_path)
    config = {"seed": seed, "reference_methods": methods, "max_line_bytes": max_line_bytes,
              "transaction_rows": transaction_rows, "cache_kib": cache_kib,
              "workers": workers, "batch_rows": batch_rows,
              "sampling_frame": SAMPLING_FRAME, "version": VERSION}
    root.mkdir(mode=0o700)
    _write_new(root / "lexicon.json", lexicon_bytes)
    with closing(_connect(root / "pool.sqlite3", cache_kib)) as connection:
        connection.executescript("""
            CREATE TABLE rows (
                uid TEXT PRIMARY KEY COLLATE BINARY,
                source_line INTEGER NOT NULL UNIQUE,
                byte_offset INTEGER NOT NULL,
                byte_length INTEGER NOT NULL,
                row_sha256 TEXT NOT NULL,
                eligible INTEGER NOT NULL CHECK(eligible IN (0,1)),
                rejection_reason TEXT NOT NULL,
                extract_method TEXT NOT NULL,
                priority TEXT NOT NULL
            );
            CREATE INDEX eligible_rank ON rows(priority COLLATE BINARY, uid COLLATE BINARY) WHERE eligible=1;
        """)
        source_sha, leakage_counts = _scan(connection, pool_path, config, checker, before, lexicon_bytes)
        counts = _counts(connection)
        _integrity(connection)
    # SQLite is finalized/closed before any artifact hashing.
    _check_stat(pool_path, before)
    _check_stat(lexicon_path, lexicon_before)
    if _read_small(lexicon_path, MAX_LEXICON_BYTES) != lexicon_bytes or _definitions() != definitions:
        raise PoolPreparationError("lexicon or source definitions mutated during indexing")
    manifest = {"version": VERSION, "kind": "index", "config": config,
                "config_sha256": hashlib.sha256(_json_bytes(config)).hexdigest(),
                "source": {"path": str(pool_path), "stat": before, "sha256": source_sha},
                "lexicon": {"path": str(lexicon_path), "stat": lexicon_before,
                            "sha256": hashlib.sha256(lexicon_bytes).hexdigest()},
                "definitions": definitions, "counts": counts,
                "leakage_by_field": leakage_counts, "notes": NOTES}
    return _finish(root, manifest, ["pool.sqlite3", "lexicon.json"])


def _checked_index(root: Path) -> tuple[dict, dict]:
    expected_files = {"pool.sqlite3", "lexicon.json", "manifest.json", "COMPLETE.json"}
    if {entry.name for entry in root.iterdir()} != expected_files:
        raise PoolPreparationError("index incomplete or has unexpected files; require exactly finalized index artifacts")
    marker_bytes = _read_small(root / "COMPLETE.json")
    marker = _decode(marker_bytes, "completion marker")
    if (not isinstance(marker, dict) or set(marker) != {"version", "kind", "sha256"}
            or marker["version"] != VERSION or marker["kind"] != "index"
            or not isinstance(marker["sha256"], dict)
            or set(marker["sha256"]) != expected_files - {"COMPLETE.json"}
            or marker_bytes != _json_bytes(marker)):
        raise PoolPreparationError("invalid/unsupported index completion marker")
    for name, expected in marker["sha256"].items():
        if _sha_file(root / name) != expected:
            raise PoolPreparationError(f"index checksum mismatch: {name}")
    manifest_bytes = _read_small(root / "manifest.json")
    manifest = _decode(manifest_bytes, "index manifest")
    if not isinstance(manifest, dict) or manifest_bytes != _json_bytes(manifest):
        raise PoolPreparationError("index manifest must be a canonical JSON object")
    if hashlib.sha256(manifest_bytes).hexdigest() != marker["sha256"]["manifest.json"]:
        raise PoolPreparationError("manifest mutated while reading")
    try:
        if manifest["version"] != VERSION or manifest["kind"] != "index" or manifest["notes"] != NOTES:
            raise PoolPreparationError("unsupported manifest version/kind/notes")
        config = manifest["config"]
        if set(config) != {"seed", "reference_methods", "max_line_bytes", "transaction_rows", "cache_kib", "workers", "batch_rows", "sampling_frame", "version"}:
            raise PoolPreparationError("invalid frozen index configuration")
        if config["version"] != VERSION or config["sampling_frame"] != SAMPLING_FRAME or type(config["seed"]) is not int:
            raise PoolPreparationError("unsupported sampling configuration")
        _positive("cache_kib", config["cache_kib"], 262144)
        _positive("max_line_bytes", config["max_line_bytes"], 64 * 1024 * 1024)
        _positive("transaction_rows", config["transaction_rows"], 100000)
        # Verification/selection need not have the indexing job's CPU allocation.
        _worker_settings(config["workers"], config["batch_rows"], check_allocation=False)
        if _methods(config["reference_methods"]) != config["reference_methods"]:
            raise PoolPreparationError("noncanonical reference methods")
        if hashlib.sha256(_json_bytes(config)).hexdigest() != manifest["config_sha256"]:
            raise PoolPreparationError("configuration checksum mismatch")
        if _definitions() != manifest["definitions"]:
            raise PoolPreparationError("source definitions changed (module, CLI or transitive country/math code); rebuild index")
        lexicon_path = _path(manifest["lexicon"]["path"], kind="file")
        _check_stat(lexicon_path, manifest["lexicon"]["stat"])
        frozen = _read_small(root / "lexicon.json", MAX_LEXICON_BYTES)
        _lexicon(frozen)
        if (hashlib.sha256(frozen).hexdigest() != manifest["lexicon"]["sha256"]
                or frozen != _read_small(lexicon_path, MAX_LEXICON_BYTES)):
            raise PoolPreparationError("original/frozen lexicon bytes changed")
        _check_stat(_path(manifest["source"]["path"], kind="file"), manifest["source"]["stat"])
        with closing(_connect(root / "pool.sqlite3", config["cache_kib"], readonly=True)) as connection:
            _integrity(connection)
            if _counts(connection) != manifest["counts"]:
                raise PoolPreparationError("SQLite counts do not match manifest")
    except (KeyError, TypeError) as exc:
        raise PoolPreparationError(f"malformed index manifest: {exc}") from exc
    # Detect changes while SQLite and the manifest were being checked.
    for name, expected in marker["sha256"].items():
        if _sha_file(root / name) != expected:
            raise PoolPreparationError(f"index mutated during verification: {name}")
    if _read_small(root / "COMPLETE.json") != marker_bytes:
        raise PoolPreparationError("completion marker mutated during verification")
    return manifest, marker


def _verify_source(source: dict) -> None:
    path = _path(source["path"], kind="file")
    _check_stat(path, source["stat"])
    if _sha_file(path) != source["sha256"]:
        raise PoolPreparationError("full source SHA256 mismatch; rebuild index")
    _check_stat(path, source["stat"])


def verify_index(index_dir: str | Path, *, full_source: bool = False) -> dict:
    """Validate checksums, SQLite integrity, definitions and frozen lexicon.

    full_source=True additionally streams the full pool SHA256. False explicitly
    reports that source cryptographic verification was NOT performed; it cannot
    authorize selection. select_pool always performs its own full verification.
    """
    root = _path(index_dir, kind="dir")
    manifest, marker = _checked_index(root)
    if full_source:
        _verify_source(manifest["source"])
        # Revalidate the index after the potentially long full-pool hash.
        after, after_marker = _checked_index(root)
        if after != manifest or after_marker != marker:
            raise PoolPreparationError("index changed during source verification")
    return {"index_manifest": manifest, "index_completion": marker,
            "source_sha256_verified": full_source,
            "verification_scope": "full source SHA256" if full_source else "index only; source stat is NOT cryptographic evidence"}


def _emit_selection(root: Path, index_root: Path, manifest: dict, n_questions: int) -> str:
    source = manifest["source"]
    path = _path(source["path"], kind="file")
    digest = hashlib.sha256()
    emitted = 0
    with closing(_connect(index_root / "pool.sqlite3", manifest["config"]["cache_kib"], readonly=True)) as connection:
        cursor = connection.execute(
            "SELECT uid, source_line, byte_offset, byte_length, row_sha256 FROM rows "
            "WHERE eligible=1 ORDER BY priority COLLATE BINARY, uid COLLATE BINARY LIMIT ?", (n_questions,))
        with path.open("rb") as stream, (root / "selected_questions.jsonl").open("xb") as output:
            _check_stat(path, source["stat"], stream)
            for uid, source_line, offset, length, row_sha in cursor:
                if not 0 < length <= manifest["config"]["max_line_bytes"] or offset < 0:
                    raise PoolPreparationError(f"invalid indexed byte range for UID {uid!r}")
                stream.seek(offset)
                raw = stream.read(length)
                if len(raw) != length or hashlib.sha256(raw).hexdigest() != row_sha:
                    raise PoolPreparationError(f"source row byte length/hash mismatch at line {source_line}")
                row = _pool_row(_decode(raw, f"selected source line {source_line}"), source_line)
                if row.uid != uid:
                    raise PoolPreparationError(f"source row UID mismatch at line {source_line}")
                encoded = _json_bytes(asdict(row))
                output.write(encoded)
                digest.update(encoded)
                emitted += 1
            _check_stat(path, source["stat"], stream)
            output.flush()
            os.fsync(output.fileno())
    if emitted != n_questions:
        raise PoolPreparationError(f"index selection cardinality changed: {emitted} != {n_questions}")
    return digest.hexdigest()


def select_pool(index_dir: str | Path, output_dir: str | Path, *, n_questions: int,
                seed: int, retained_target: int = 500000) -> dict:
    """Freeze shared PoolRow JSONL in full-population hash-priority order.

    ALWAYS hashes the full source on this call (CPU job required in production).
    seed must match the index; build a NEW index to change it. n_questions is an
    explicit RAW count, not an acceptance claim. No hidden prefix, cap, or padding.
    """
    _positive("n_questions", n_questions, 2**63 - 1)
    _positive("retained_target", retained_target)
    if type(seed) is not int:
        raise PoolPreparationError("seed must be an integer matching the index")
    index_root = _path(index_dir, kind="dir")
    manifest, marker = _checked_index(index_root)
    protected = [index_root, Path(manifest["source"]["path"]), Path(manifest["lexicon"]["path"]),
                 *map(Path, manifest["definitions"])]
    root = _new_output(output_dir, protected)
    if seed != manifest["config"]["seed"]:
        raise PoolPreparationError("seed mismatch; construct a NEW index for another seed")
    if manifest["counts"]["eligible"] < n_questions:
        raise PoolPreparationError(f"insufficient eligible rows: {manifest['counts']['eligible']} < raw n_questions={n_questions}; inspect total/eligible_by_method")
    _verify_source(manifest["source"])
    after, after_marker = _checked_index(index_root)
    if after != manifest or after_marker != marker:
        raise PoolPreparationError("index changed during full source verification")
    root.mkdir(mode=0o700)
    selected_sha = _emit_selection(root, index_root, manifest, n_questions)
    after, after_marker = _checked_index(index_root)
    if after != manifest or after_marker != marker:
        raise PoolPreparationError("index changed during selection")
    # Exact source manifest/completion bytes and lexicon travel with the selection.
    for name in ("manifest.json", "COMPLETE.json", "lexicon.json"):
        data = _read_small(index_root / name)
        expected = (hashlib.sha256(_json_bytes(marker)).hexdigest() if name == "COMPLETE.json"
                    else marker["sha256"][name])
        if hashlib.sha256(data).hexdigest() != expected:
            raise PoolPreparationError(f"index changed while copying provenance: {name}")
        _write_new(root / ("index_" + name), data)
    selected = {"version": VERSION, "kind": "selection", "seed": seed,
                "n_questions": n_questions, "n_questions_kind": "raw_generation_questions",
                "retained_target": retained_target, "retained_target_is_availability_claim": False,
                "retained_solutions_available": None, "sampling_frame": SAMPLING_FRAME,
                "selected_sha256": selected_sha, "selected_schema": [*FIELDS, "source_line"],
                "serialization": "UTF-8 sorted-key compact ASCII-escaped JSON + LF; source field strings unchanged",
                "index_path": str(index_root), "index_manifest": manifest,
                "index_completion": marker, "source_sha256_verified": True, "notes": NOTES}
    return _finish(root, selected, ["selected_questions.jsonl", "index_manifest.json", "index_COMPLETE.json", "index_lexicon.json"])
