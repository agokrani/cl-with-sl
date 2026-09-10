"""Temp-only, stdlib CPU tests; never touch the production pool.

python3 -S -B -m unittest discover -s tests -p 'test_country_math_pool.py'
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from cl import country_math_pool as pool
from cl.country_math_data import PoolRow

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/prepare_country_math_pool.py"


def source_row(i: int, **updates: str) -> dict:
    value = {"uid": f"q{i:04d}", "question": f"Compute {i} + 0.",
             "ref_answer": f"Adding zero leaves {i}.", "ref_final": str(i),
             "extract_method": "answer-line"}
    value.update(updates)
    return value


class PoolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "pool.jsonl"
        self.lexicon = self.root / "aliases.json"
        self.lexicon.write_bytes(b'["Canada", "France", "China", "Japan", "United States"]\n')
        self.index = self.root / "index"
        self.output = self.root / "selection"
        self.rows = [source_row(i) for i in range(40)]
        self.write_rows(self.rows)

    def write_rows(self, rows):
        self.source.write_bytes(b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode() for row in rows))

    def build(self, **kwargs):
        return pool.index_pool(self.source, kwargs.pop("output_dir", self.index),
                               lexicon=self.lexicon, seed=kwargs.pop("seed", 7),
                               transaction_rows=3, cache_kib=64, **kwargs)

    def select(self, **kwargs):
        return pool.select_pool(self.index, kwargs.pop("output_dir", self.output),
                                n_questions=kwargs.pop("n_questions", 8),
                                seed=kwargs.pop("seed", 7), **kwargs)

    def overwrite(self, path, data):
        path.chmod(0o600)
        path.write_bytes(data)

    def reseal(self):
        """White-box corruption tests bypass checksum to reach deeper gates."""
        marker_path = self.index / "COMPLETE.json"
        marker = json.loads(marker_path.read_bytes())
        for name in marker["sha256"]:
            marker["sha256"][name] = hashlib.sha256((self.index / name).read_bytes()).hexdigest()
        self.overwrite(marker_path, pool._json_bytes(marker))

    def test_full_population_deterministic_order_and_exact_fields(self):
        self.rows[-1]["question"] = "  Compute π + 0.\nDo not round.  "
        self.write_rows(self.rows)
        manifest = self.build()
        selected = self.select(n_questions=40)
        raw = (self.output / "selected_questions.jsonl").read_bytes()
        records = [json.loads(line) for line in raw.splitlines()]
        expected = sorted(self.rows, key=lambda row: (pool.stable_priority(7, row["uid"]), row["uid"]))
        self.assertEqual([record["uid"] for record in records], [row["uid"] for row in expected])
        self.assertNotEqual(records[0]["source_line"], 1)
        self.assertGreater(max(record["source_line"] for record in records[:8]), 8)
        original_by_uid = {row["uid"]: (i, row) for i, row in enumerate(self.rows, 1)}
        for record in records:
            row = PoolRow(**record)
            line, original = original_by_uid[row.uid]
            self.assertEqual(row.source_line, line)
            self.assertEqual({key: record[key] for key in pool.FIELDS}, original)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), selected["selected_sha256"])
        self.assertEqual(manifest["source"]["sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())
        other = self.root / "selection2"
        self.select(output_dir=other, n_questions=40)
        self.assertEqual(raw, (other / "selected_questions.jsonl").read_bytes())
        self.assertEqual(selected["index_manifest"], manifest)
        self.assertEqual((self.output / "index_manifest.json").read_bytes(), (self.index / "manifest.json").read_bytes())
        marker = json.loads((self.output / "COMPLETE.json").read_bytes())
        for name, digest in marker["sha256"].items():
            self.assertEqual(hashlib.sha256((self.output / name).read_bytes()).hexdigest(), digest)

    def test_priority_is_canonical_json_not_python_hash(self):
        expected = hashlib.sha256(b'[7,"q0001"]').hexdigest()
        self.assertEqual(pool.stable_priority(7, "q0001"), expected)
        self.assertNotEqual(pool.stable_priority(8, "q0001"), expected)
        with self.assertRaises(ValueError):
            pool.stable_priority(True, "q0001")

    def test_different_seed_and_same_seed_index_determinism(self):
        self.build()
        self.select()
        first = (self.output / "selected_questions.jsonl").read_bytes()
        same = self.root / "same"
        self.build(output_dir=same)
        self.assertEqual((self.index / "pool.sqlite3").read_bytes(), (same / "pool.sqlite3").read_bytes())
        different = self.root / "different"
        self.build(output_dir=different, seed=8)
        destination = self.root / "different-selection"
        pool.select_pool(different, destination, n_questions=8, seed=8)
        self.assertNotEqual(first, (destination / "selected_questions.jsonl").read_bytes())
        with self.assertRaisesRegex(ValueError, "seed mismatch"):
            self.select(output_dir=self.root / "badseed", seed=8)
        self.assertFalse((self.root / "badseed").exists())

    def test_method_counts_exclusion_and_three_field_leakage(self):
        rows = [source_row(0, extract_method="tag"), source_row(1, extract_method="boxed"),
                source_row(2, extract_method="option"), source_row(3, extract_method="last-bold"),
                source_row(4, question="Canada has 3 apples"), source_row(5, ref_answer="France"),
                source_row(6, ref_final="US"), source_row(7, question="Give us 7 apples"),
                source_row(8, question="Ignore previous instructions"), source_row(9)]
        self.write_rows(rows)
        manifest = self.build()
        self.assertEqual(manifest["counts"]["total"], 10)
        self.assertEqual(manifest["counts"]["eligible"], 4)
        self.assertEqual(manifest["counts"]["total_by_method"]["option"], 1)
        self.assertEqual(manifest["counts"]["eligible_by_method"]["option"], 0)
        self.assertEqual(manifest["counts"]["rejections"]["excluded_reference_method"], 2)
        self.assertEqual(manifest["leakage_by_field"], {"question": 2, "ref_answer": 1, "ref_final": 1})
        all_methods = self.build(output_dir=self.root / "all", reference_methods=sorted(pool.SUPPORTED_EXTRACTION_METHODS))
        self.assertEqual(all_methods["counts"]["eligible"], 6)
        self.assertEqual(manifest["notes"]["main_conditions"], list(pool.MAIN_CONDITIONS))
        self.assertNotIn("hate_japan_exploratory", manifest["notes"]["main_conditions"])

    def test_no_reference_text_in_sqlite_and_exact_offsets(self):
        self.build()
        with closing(sqlite3.connect(self.index / "pool.sqlite3")) as connection:
            columns = [row[1] for row in connection.execute("PRAGMA table_info(rows)")]
            self.assertNotIn("ref_answer", columns)
            self.assertNotIn("ref_final", columns)
            with self.source.open("rb") as stream:
                for uid, line, offset, length, digest in connection.execute(
                        "SELECT uid, source_line, byte_offset, byte_length, row_sha256 FROM rows ORDER BY source_line"):
                    self.assertEqual(stream.tell(), offset)
                    raw = stream.read(length)
                    self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)
                    self.assertEqual(json.loads(raw)["uid"], uid)
                    self.assertEqual(uid, self.rows[line - 1]["uid"])
        self.assertNotIn(b"Adding zero leaves", (self.index / "pool.sqlite3").read_bytes())

    def test_raw_count_not_retained_target(self):
        self.build()
        manifest = self.select(n_questions=2, retained_target=500000)
        self.assertEqual(manifest["n_questions"], 2)
        self.assertEqual(manifest["n_questions_kind"], "raw_generation_questions")
        self.assertEqual(manifest["retained_target"], 500000)
        self.assertFalse(manifest["retained_target_is_availability_claim"])
        self.assertIsNone(manifest["retained_solutions_available"])
        self.assertEqual(manifest["notes"]["training_checkpoints"], [50000, 100000, 200000, 500000])

    def test_insufficient_eligible_before_output_or_full_source_hash(self):
        self.build()
        with patch.object(pool, "_verify_source", side_effect=AssertionError("must fail before source scan")):
            with self.assertRaisesRegex(ValueError, "insufficient eligible"):
                self.select(n_questions=500000)
        self.assertFalse(self.output.exists())

    def test_duplicate_including_excluded_uid_fails_and_preserves_failed_root(self):
        self.write_rows([source_row(1, extract_method="option"), source_row(1)])
        with self.assertRaisesRegex(ValueError, "duplicate UID"):
            self.build()
        self.assertTrue(self.index.is_dir())
        self.assertFalse((self.index / "COMPLETE.json").exists())
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.build()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            pool.verify_index(self.index)

    def test_missing_final_is_counted_but_never_selected(self):
        self.write_rows([source_row(1, ref_final=""),
                         source_row(2, ref_final="  ", extract_method="last-bold"),
                         source_row(3)])
        manifest = self.build()
        self.assertEqual(manifest["counts"]["total"], 3)
        self.assertEqual(manifest["counts"]["eligible"], 1)
        self.assertEqual(manifest["counts"]["rejections"]["missing_reference_final"], 1)
        self.assertEqual(manifest["counts"]["rejections"]["excluded_reference_method;missing_reference_final"], 1)
        self.select(n_questions=1)
        selected = json.loads((self.output / "selected_questions.jsonl").read_text())
        self.assertEqual(selected["uid"], "q0003")
        self.assertEqual(selected["ref_final"], "3")

    def test_unknown_method_malformed_fields_and_oversize_fail_closed(self):
        cases = [(json.dumps(source_row(1, extract_method="unknown")).encode(), "unsupported"),
                 (b'{"uid":"a"}', "required fields"), (b'[]', "expected JSON object"),
                 (b'not json', "invalid JSON"), (b'{"uid":"a","uid":"b"}', "duplicate JSON key"),
                 (json.dumps(source_row(1, uid=" q1")).encode(), "edge whitespace"),
                 (b'\xff', "invalid JSON"), (b'\n', "invalid JSON")]
        for i, (data, message) in enumerate(cases):
            with self.subTest(message=message):
                self.source.write_bytes(data + b"\n")
                with self.assertRaisesRegex(ValueError, message):
                    self.build(output_dir=self.root / f"bad{i}")
        self.source.write_bytes(b"x" * 100)
        with self.assertRaisesRegex(ValueError, "max_line_bytes"):
            self.build(max_line_bytes=99)

    def test_empty_pool_and_no_trailing_newline(self):
        self.source.write_bytes(b"")
        manifest = self.build()
        self.assertEqual(manifest["counts"]["eligible"], 0)
        self.assertEqual(manifest["source"]["sha256"], hashlib.sha256(b"").hexdigest())
        with self.assertRaisesRegex(ValueError, "insufficient"):
            self.select(n_questions=1)
        self.source.write_bytes(json.dumps(source_row(0)).encode())
        index2 = self.root / "no-newline"
        self.build(output_dir=index2)
        pool.select_pool(index2, self.output, n_questions=1, seed=7)
        self.assertEqual(len((self.output / "selected_questions.jsonl").read_bytes().splitlines()), 1)

    def test_lexicon_explicit_nonempty_valid_and_exact_bytes_frozen(self):
        for i, value in enumerate([[], {}, [""], [" France"], [3]]):
            self.lexicon.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                self.build(output_dir=self.root / f"invalidlex{i}")
        self.lexicon.write_bytes(b'[  "France", "Canada" ]\n')
        manifest = self.build()
        self.assertEqual((self.index / "lexicon.json").read_bytes(), self.lexicon.read_bytes())
        self.assertEqual(manifest["lexicon"]["sha256"], hashlib.sha256(self.lexicon.read_bytes()).hexdigest())
        self.lexicon.write_bytes(b'["France","Canada"]\n')
        with self.assertRaisesRegex(ValueError, "mutated|lexicon"):
            self.select()
        self.assertFalse(self.output.exists())

    def test_configuration_validation_before_output(self):
        cases = [{"seed": True}, {"reference_methods": []}, {"reference_methods": ["tag", "tag"]},
                 {"reference_methods": ["made-up"]}, {"reference_methods": "boxed"}, {"max_line_bytes": 0}]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.build(**values)
            self.assertFalse(self.index.exists())

    def test_mutated_pool_replacement_or_same_size_fails(self):
        self.build()
        original = self.source.stat()
        raw = self.source.read_bytes().replace(b"Compute", b"compute")
        self.source.write_bytes(raw)
        os.utime(self.source, ns=(original.st_atime_ns, original.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "mutated"):
            self.select()
        self.assertFalse(self.output.exists())

    def test_mutation_during_scan_preserves_failure(self):
        real = pool.stable_priority
        changed = False

        def mutate(seed, uid):
            nonlocal changed
            if not changed:
                with self.source.open("ab") as stream:
                    stream.write((json.dumps(source_row(999)) + "\n").encode())
                changed = True
            return real(seed, uid)

        with patch.object(pool, "stable_priority", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "mutated"):
                self.build()
        self.assertFalse((self.index / "COMPLETE.json").exists())

    def test_full_source_hash_is_mandatory_for_selection_and_explicit_for_verify(self):
        self.build()
        report = pool.verify_index(self.index)
        self.assertFalse(report["source_sha256_verified"])
        self.assertTrue(pool.verify_index(self.index, full_source=True)["source_sha256_verified"])
        with patch.object(pool, "_verify_source", side_effect=ValueError("full source SHA256 mismatch")) as check:
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                self.select()
            check.assert_called_once()
        self.assertFalse(self.output.exists())

    def test_full_hash_failure_even_when_source_stat_matches(self):
        self.build()
        manifest_path = self.index / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["source"]["sha256"] = "0" * 64
        self.overwrite(manifest_path, pool._json_bytes(manifest))
        self.reseal()
        with self.assertRaisesRegex(ValueError, "full source SHA256 mismatch"):
            self.select()
        self.assertFalse(self.output.exists())

    def test_mutated_artifact_checksums(self):
        for i, name in enumerate(["pool.sqlite3", "manifest.json", "lexicon.json"]):
            index = self.root / f"index{i}"
            self.build(output_dir=index)
            artifact = index / name
            self.overwrite(artifact, artifact.read_bytes() + b" ")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "checksum mismatch"):
                pool.verify_index(index)

    def test_changed_transitive_definitions_fail(self):
        manifest = self.build()
        self.assertTrue(any(path.endswith("country_math_data.py") for path in manifest["definitions"]))
        self.assertTrue(any(path.endswith("country_preference.py") for path in manifest["definitions"]))
        self.assertTrue(any(path.endswith("prepare_country_math_pool.py") for path in manifest["definitions"]))
        with patch.object(pool, "_definitions", return_value={}):
            with self.assertRaisesRegex(ValueError, "source definitions changed"):
                self.select()

    def test_strict_completion_marker_and_unexpected_sidecars(self):
        self.build()
        (self.index / "pool.sqlite3-wal").write_bytes(b"stale")
        with self.assertRaisesRegex(ValueError, "unexpected"):
            pool.verify_index(self.index)
        (self.index / "pool.sqlite3-wal").unlink()
        marker = self.index / "COMPLETE.json"
        self.overwrite(marker, b'{}')
        with self.assertRaisesRegex(ValueError, "completion marker"):
            pool.verify_index(self.index)

    def test_index_integrity_checked_even_after_checksum_reseal(self):
        self.build()
        db = self.index / "pool.sqlite3"
        self.overwrite(db, b"not a SQLite database")
        self.reseal()
        with self.assertRaises((ValueError, sqlite3.DatabaseError)):
            pool.verify_index(self.index)

    def test_selected_rows_hash_length_and_uid_checked(self):
        for i, sql in enumerate([
                "UPDATE rows SET row_sha256='broken' WHERE eligible=1",
                "UPDATE rows SET byte_length=1 WHERE eligible=1",
                "UPDATE rows SET uid='changed-' || uid WHERE eligible=1"]):
            index = self.root / f"corrupt{i}"
            self.build(output_dir=index)
            self.index = index
            db = index / "pool.sqlite3"
            db.chmod(0o600)
            with closing(sqlite3.connect(db)) as connection:
                connection.execute(sql)
                connection.commit()
            self.reseal()
            output = self.root / f"out{i}"
            with self.subTest(sql=sql), self.assertRaisesRegex(ValueError, "hash mismatch|UID mismatch"):
                self.select(output_dir=output)
            self.assertTrue(output.exists())
            self.assertFalse((output / "COMPLETE.json").exists())

    def test_parent_symlink_overlap_and_existing_output_safety(self):
        with self.assertRaisesRegex(ValueError, "parent/input must already exist"):
            self.build(output_dir=self.root / "absent" / "index")
        self.assertFalse((self.root / "absent").exists())
        with self.assertRaisesRegex(ValueError, "parent traversal"):
            self.build(output_dir=self.root / ".." / "index")
        link = self.root / "linked"
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.build(output_dir=link / "index")
        linked_source = self.root / "linked-source"
        linked_source.symlink_to(self.source)
        with self.assertRaisesRegex(ValueError, "symlink"):
            pool.index_pool(linked_source, self.index, lexicon=self.lexicon, seed=7)
        linked_lexicon = self.root / "linked-lexicon"
        linked_lexicon.symlink_to(self.lexicon)
        with self.assertRaisesRegex(ValueError, "symlink"):
            pool.index_pool(self.source, self.index, lexicon=linked_lexicon, seed=7)
        self.build()
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.select(output_dir=self.index / "nested")
        self.select()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.select()
        self.assertFalse((self.index / "nested").exists())
        frozen = self.index / "lexicon.json"
        frozen.unlink()
        frozen.symlink_to(self.lexicon)
        with self.assertRaisesRegex(ValueError, "symlink"):
            pool.verify_index(self.index)

    def test_bounded_sqlite_settings_and_incremental_commits(self):
        with closing(pool._connect(self.root / "settings.sqlite3", 64)) as connection:
            self.assertEqual(connection.execute("PRAGMA cache_size").fetchone(), (-64,))
            self.assertEqual(connection.execute("PRAGMA temp_store").fetchone(), (1,))
            self.assertEqual(connection.execute("PRAGMA mmap_size").fetchone(), (0,))
        traced = []
        real = pool._connect

        def connect(*args, **kwargs):
            connection = real(*args, **kwargs)
            connection.set_trace_callback(traced.append)
            return connection

        with patch.object(pool, "_connect", side_effect=connect):
            self.build()
        self.assertGreater(sum(statement == "COMMIT" for statement in traced), 10)
        with closing(sqlite3.connect(self.index / "pool.sqlite3")) as connection:
            plan = connection.execute("EXPLAIN QUERY PLAN SELECT uid FROM rows WHERE eligible=1 ORDER BY priority COLLATE BINARY, uid COLLATE BINARY LIMIT 8").fetchall()
            self.assertIn("eligible_rank", str(plan))
            self.assertNotIn("TEMP B-TREE", str(plan))

    def test_parallel_equivalence_source_hash_database_and_selection(self):
        self.rows[1].update(question="Canada has apples", ref_answer="France", ref_final="US")
        self.rows[2].update(ref_final="", extract_method="last-bold")
        self.rows[3].update(ref_final="  ")
        self.rows[4].update(question="  Compute π + 0.\nDo not round.  ")
        self.write_rows(self.rows)
        # Tail batch and unterminated final line must retain identical byte ranges.
        self.source.write_bytes(self.source.read_bytes().rstrip(b"\n"))
        serial = self.build(batch_rows=7)
        self.select(n_questions=serial["counts"]["eligible"])
        expected = (self.output / "selected_questions.jsonl").read_bytes()
        for workers in (2, 3):
            with self.subTest(workers=workers):
                index = self.root / f"parallel{workers}"
                manifest = self.build(output_dir=index, workers=workers, batch_rows=7)
                self.assertEqual(manifest["config"]["workers"], workers)
                self.assertEqual(manifest["config"]["batch_rows"], 7)
                for field in ("source", "counts", "leakage_by_field"):
                    self.assertEqual(serial[field], manifest[field])
                self.assertEqual((self.index / "pool.sqlite3").read_bytes(), (index / "pool.sqlite3").read_bytes())
                destination = self.root / f"selection{workers}"
                pool.select_pool(index, destination, n_questions=serial["counts"]["eligible"], seed=7)
                self.assertEqual(expected, (destination / "selected_questions.jsonl").read_bytes())
                with patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "1"}):
                    self.assertTrue(pool.verify_index(index, full_source=True)["source_sha256_verified"])

    def test_parallel_duplicate_across_batches_and_worker_parser_failure(self):
        cases = [([source_row(1, extract_method="option"), source_row(2), source_row(1)], "line 3: duplicate UID"),
                 ([source_row(1), source_row(2), {"uid": "bad"}], "line 3: required fields")]
        for i, (rows, message) in enumerate(cases):
            self.write_rows(rows)
            index = self.root / f"failed{i}"
            with self.assertRaisesRegex(pool.PoolPreparationError, message):
                self.build(output_dir=index, workers=2, batch_rows=1)
            self.assertTrue((index / "pool.sqlite3").exists())
            self.assertFalse((index / "COMPLETE.json").exists())
            with self.assertRaisesRegex(ValueError, "incomplete"):
                pool.verify_index(index)
        for i, raw in enumerate((b'not JSON\n', b'{"uid":"a","uid":"b"}\n', b'\xff\n')):
            self.source.write_bytes(raw)
            with self.assertRaisesRegex(pool.PoolPreparationError, "invalid JSON"):
                self.build(output_dir=self.root / f"parser{i}", workers=2, batch_rows=2)
        self.source.write_bytes(b"x" * 100)
        with self.assertRaisesRegex(ValueError, "max_line_bytes"):
            self.build(workers=2, max_line_bytes=99)
        self.assertFalse((self.index / "COMPLETE.json").exists())

    def test_parallel_empty_pool_and_single_row_tail(self):
        self.source.write_bytes(b"")
        manifest = self.build(workers=3, batch_rows=32)
        self.assertEqual(manifest["counts"]["total"], 0)
        self.assertEqual(manifest["source"]["sha256"], hashlib.sha256(b"").hexdigest())
        self.source.write_bytes(json.dumps(source_row(1)).encode())
        manifest = self.build(output_dir=self.root / "tail", workers=2, batch_rows=32)
        self.assertEqual(manifest["counts"]["eligible"], 1)
        self.assertEqual(manifest["source"]["sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())

    def test_parallel_worker_and_batch_resource_validation(self):
        for key, values in (("workers", (0, -1, True, 1.5, 65)),
                            ("batch_rows", (0, -1, True, 1.5, 1025))):
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                    self.build(**{key: value})
                self.assertFalse(self.index.exists())
        for cpus in ("1", "0", "-1", "oops", "", "2.5"):
            with patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": cpus}):
                with self.assertRaisesRegex(ValueError, "SLURM_CPUS_PER_TASK"):
                    self.build(workers=2)
            self.assertFalse(self.index.exists())
        with patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "2"}):
            self.build(workers=2)

    def test_parallel_config_new_keys_and_tampering(self):
        original = self.build(workers=2, batch_rows=4)
        manifest_path = self.index / "manifest.json"
        for key, value, message in (("workers", 0, "workers"), ("batch_rows", 1025, "batch_rows"),
                                    ("workers", 3, "configuration checksum"),
                                    ("unexpected", 1, "invalid frozen"),
                                    ("version", "country-math-full-pool-v2", "unsupported")):
            manifest = json.loads(json.dumps(original))
            manifest["config"][key] = value
            self.overwrite(manifest_path, pool._json_bytes(manifest))
            self.reseal()
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, message):
                pool.verify_index(self.index)
        for key in ("workers", "batch_rows"):
            manifest = json.loads(json.dumps(original))
            del manifest["config"][key]
            self.overwrite(manifest_path, pool._json_bytes(manifest))
            self.reseal()
            with self.assertRaisesRegex(ValueError, "invalid frozen"):
                pool.verify_index(self.index)

    def test_parallel_pending_bound_order_spawn_and_failure_cancellation(self):
        state: dict = {"outstanding": 0, "maximum": 0, "submitted": 0, "cancelled": 0}
        fail = False

        class FakeFuture:
            def __init__(self, number):
                self.number = number

            def result(self):
                state["outstanding"] -= 1
                if fail:
                    raise pool.PoolPreparationError("worker error")
                return ([self.number], {})

            def cancel(self):
                state["cancelled"] += 1

        class FakeExecutor:
            def __init__(self, **kwargs):
                state["context"] = kwargs["mp_context"].get_start_method()
                state["initializer"] = kwargs["initializer"]
                state["initargs"] = kwargs["initargs"]

            def submit(self, function, batch):
                state["outstanding"] += 1
                state["maximum"] = max(state["maximum"], state["outstanding"])
                state["submitted"] += 1
                return FakeFuture(batch[0][1])

            def shutdown(self, **kwargs):
                state["shutdown"] = kwargs

        config = {"workers": 3}
        batches = [[(b"raw", i, 0)] for i in range(23)]
        with patch.object(pool, "ProcessPoolExecutor", FakeExecutor):
            self.assertEqual([records[0] for records, _ in pool._parallel_batches(iter(batches), config, b"[]")], list(range(23)))
            self.assertEqual(state["maximum"], 6)
            self.assertEqual(state["context"], "spawn")
            self.assertIs(state["initializer"], pool._worker_init)
            self.assertEqual(state["initargs"], (b"[]", config))
            self.assertEqual(state["shutdown"], {"wait": True, "cancel_futures": True})
            state.update(outstanding=0, submitted=0, cancelled=0)
            fail = True
            with self.assertRaisesRegex(ValueError, "worker error"):
                list(pool._parallel_batches(iter(batches), config, b"[]"))
            self.assertEqual(state["submitted"], 6)  # Did not eagerly read/submit all 23.
            self.assertEqual(state["cancelled"], 5)

    def test_parallel_mutation_and_definition_checks_remain_fail_closed(self):
        real = pool._parallel_batches
        for target in ("source", "lexicon", "definitions"):
            self.write_rows(self.rows)
            self.lexicon.write_bytes(b'["Canada"]\n')
            index = self.root / target

            def mutate(*args):
                changed = False
                for result in real(*args):
                    if not changed:
                        if target == "source":
                            with self.source.open("ab") as stream:
                                stream.write(b"\n")
                        elif target == "lexicon":
                            self.lexicon.write_bytes(b'["France"]\n')
                        changed = True
                    yield result

            definitions = pool._definitions()
            definition_results = [definitions, {}] if target == "definitions" else [definitions, definitions]
            with patch.object(pool, "_parallel_batches", side_effect=mutate), \
                    patch.object(pool, "_definitions", side_effect=definition_results):
                with self.assertRaisesRegex(ValueError, "mutated"):
                    self.build(output_dir=index, workers=2, batch_rows=32)
            self.assertFalse((index / "COMPLETE.json").exists())

    def test_worker_initializer_once_compact_results_and_single_parent_source_pass(self):
        config = {"seed": 7, "reference_methods": list(pool.DEFAULT_REFERENCE_METHODS)}
        raw = json.dumps(source_row(1)).encode()
        with patch.object(pool, "_WORKER_CHECKER", None), patch.object(pool, "_WORKER_CONFIG", None):
            with patch.object(pool, "_lexicon", wraps=pool._lexicon) as compile_checker:
                pool._worker_init(self.lexicon.read_bytes(), config)
                first = pool._worker_batch([(raw, 1, 0)])
                pool._worker_batch([(raw, 2, len(raw))])
                compile_checker.assert_called_once()
                self.assertNotIn("Adding zero", repr(first))
                self.assertNotIn("Compute", repr(first))
                self.assertEqual(len(first[0][0]), 9)
        opens = []
        real_open = Path.open

        def track_open(path, *args, **kwargs):
            if path == self.source:
                opens.append(args)
            return real_open(path, *args, **kwargs)

        with patch.object(Path, "open", track_open):
            self.build(workers=2, batch_rows=3)
        self.assertEqual(opens, [("rb",)])

    def test_wrapper_syntax_and_cpu_guard_before_execution(self):
        wrapper = ROOT / "scripts/submit_country_math_index.sh"
        proc = subprocess.run(["bash", "-n", str(wrapper)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        for cpus in ("1", "7", "oops", ""):
            env = dict(os.environ, SLURM_JOB_ID="test-only", SLURM_SUBMIT_DIR=str(ROOT),
                       SLURM_TMPDIR=str(self.root), SLURM_CPUS_PER_TASK=cpus)
            proc = subprocess.run(["bash", str(wrapper), "unused-pool", "unused-index", "unused-lexicon"],
                                  env=env, capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("SLURM_CPUS_PER_TASK", proc.stderr)
            self.assertNotIn("CPU indexing only", proc.stdout)

    def test_parent_interim_progress_is_flushed_and_not_completion(self):
        with patch.object(pool, "PROGRESS_ROWS", 10), patch("builtins.print") as output:
            manifest = self.build(workers=2, batch_rows=3)
        self.assertEqual(manifest["counts"]["total"], 40)
        self.assertEqual(output.call_count, 4)
        for i, call in enumerate(output.call_args_list, 1):
            self.assertIn(f"processed_total={i * 10} eligible={i * 10}", call.args[0])
            self.assertIn("INTERIM (not complete)", call.args[0])
            self.assertEqual(call.kwargs, {"file": sys.stderr, "flush": True})

    def test_cli_and_stdlib_only_import(self):
        code = (f"import sys; sys.path.insert(0, {str(ROOT)!r}); import cl.country_math_pool; "
                "assert not {'torch','vllm','math_verify','pycountry'} & set(sys.modules)")
        proc = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        base = [sys.executable, "-S", "-B", str(CLI)]
        proc = subprocess.run(base + ["index", "--pool", str(self.source), "--output-dir", str(self.index),
                                    "--lexicon", str(self.lexicon), "--seed", "7", "--workers", "2",
                                    "--batch-rows", "3"], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["counts"]["total"], 40)
        proc = subprocess.run(base + ["verify", "--index-dir", str(self.index), "--full-source"], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(json.loads(proc.stdout)["source_sha256_verified"])
        proc = subprocess.run(base + ["select", "--index-dir", str(self.index), "--output-dir", str(self.output),
                                    "--seed", "7", "--n-questions", "2", "--retained-target", "500000"], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["n_questions"], 2)
        proc = subprocess.run(base + ["select", "--index-dir", str(self.index), "--output-dir", str(self.output),
                                    "--seed", "8", "--n-questions", "2", "--retained-target", "500000"], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("No overwrite/resume", proc.stderr)


if __name__ == "__main__":
    unittest.main()
