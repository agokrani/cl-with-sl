"""Synthetic teacher fixtures ONLY in temporary test directories; no experiment results."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

from cl import country_pipeline as pipeline
from cl import country_transfer as transfer
from cl.country_preference import DEFAULT_COUNTRIES
from cl.country_runtime import approve_calibration

RECIPIENT = "Qwen/Qwen3-8B"
REVISION = "a1" * 20  # Test-only requested commit, deliberately not runtime-certified.


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name)
        self.teacher = self.parent / "teacher"
        self.consumer = self.parent / "consumer"
        self.before_sources = pipeline.source_fingerprint()
        self.manifest = self.make_teacher()

    def make_teacher(self, selection="reference"):
        manifest = pipeline.create_run(
            self.teacher, countries=DEFAULT_COUNTRIES, generation_size=4,
            train_dose=2, seeds=[1, 2], conditions=["love_A", "clean"],
            selection=selection, eval_samples=2)
        # Actual helper schemas, but plainly test-only contents, never real approval.
        pipeline.atomic_json(self.teacher / "environment.json", {
            "fingerprint": {"python": "synthetic-test-only", "executable": "synthetic-test-only",
                            "packages": {}, "engine_profile": {}},
            "gpu": "synthetic-test-only", "historical_pins_partial": True})
        result = self.teacher / "calibration_results.json"
        pipeline.atomic_json(result, {"status": "requires_human_review_not_automatic_pass",
                                     "arms": {}, "synthetic_test_only": True})
        pipeline.atomic_json(self.teacher / "calibration.json", pipeline.stage_record(
            self.teacher, manifest, [result], status="calibrated"))
        approve_calibration(self.teacher, manifest, "Synthetic unit-test fixture only; not human experiment approval")
        for condition in manifest["config"]["conditions"]:
            raw = [{"prompt": f"test prompt {i} —", "completion": f" <think>test</think> [{i}, 001]. "}
                   for i in range(4)]
            filtered, audit = pipeline.filter_raw_rows(raw)
            directory = self.teacher / condition
            pipeline.write_rows(directory / "raw_dataset.jsonl", raw)
            pipeline.write_rows(directory / "filtered_dataset.jsonl", filtered)
            audit.update({"synthetic_test_only": True, "temperature": 1.0, "max_tokens": 2048,
                          "engine": {}, "finish_metadata_retained": False})
            pipeline.atomic_json(directory / "filter_audit.json", audit)
            files = [directory / name for name in
                     ("raw_dataset.jsonl", "filtered_dataset.jsonl", "filter_audit.json")]
            pipeline.atomic_json(directory / "generation.json", pipeline.stage_record(
                self.teacher, manifest, files, status="generated"))
        pipeline.prepare_run(self.teacher, manifest)
        return manifest

    def bind(self):
        return transfer.create_transfer_run(self.consumer, self.teacher, RECIPIENT, REVISION)

    def read_json(self, path):
        return json.loads(path.read_text())

    def reseal_stage(self, name):
        path = self.teacher / name
        record = self.read_json(path)
        record["files"] = {relative: pipeline.file_digest(self.teacher / relative)
                           for relative in record["files"]}
        pipeline.atomic_json(path, record)

    def reseal_consumer(self, manifest):
        manifest["config_sha256"] = pipeline.digest(manifest["config"])
        manifest["manifest_sha256"] = pipeline.digest({k: v for k, v in manifest.items()
                                                        if k != "manifest_sha256"})
        pipeline.atomic_json(self.consumer / transfer.MANIFEST, manifest)

    def test_valid_exact_bytes_order_and_no_hardlinks(self):
        # Non-default JSON spacing, escaped Unicode and CRLF must survive verbatim.
        selected = self.teacher / "prepared/love_A/seed_1.jsonl"
        rows = pipeline.read_rows(selected)
        selected.write_bytes(b"".join((json.dumps(r, ensure_ascii=True, separators=(",", ":")) + "\r\n").encode()
                                      for r in rows))
        self.reseal_stage("prepared.json")
        bundle = transfer.verify_teacher_bundle(self.teacher)
        before = {name: (self.teacher / name).read_bytes() for name in bundle["files_sha256"]}
        result = self.bind()
        self.assertEqual(result, transfer.verify_transfer_run(self.consumer))
        self.assertEqual(result["status"], "corpus_bound_not_trained")
        self.assertFalse(result["runtime_validated"])
        self.assertFalse(result["config"]["recipient"]["runtime_validated"])
        self.assertEqual(result["config"]["recipient"]["requested_revision"], REVISION)
        self.assertEqual(result["config"]["recipe"], bundle["config"])
        self.assertEqual(result["config"]["teacher"], bundle)
        self.assertEqual(set(result["source_sha256"]["consumer"]), set(transfer.SOURCE_PATHS))
        for name, checksum in result["files"].items():
            source, target = self.teacher / name, self.consumer / name
            self.assertEqual(source.read_bytes(), target.read_bytes())
            self.assertEqual(checksum, bundle["files_sha256"][name])
            self.assertNotEqual(source.stat().st_ino, target.stat().st_ino)
        self.assertEqual(before, {name: (self.teacher / name).read_bytes() for name in before})
        self.assertEqual(pipeline.source_fingerprint(), self.before_sources)
        self.assertFalse((self.consumer / ".operation.lock").exists())

    def test_matched_selection(self):
        # Keep both test teacher roots; no fixture/run reuse.
        self.teacher = self.parent / "matched-teacher"
        self.manifest = self.make_teacher(selection="matched")
        self.bind()
        transfer.verify_transfer_run(self.consumer)

    def test_missing_or_blank_approval(self):
        path = self.teacher / "calibration_approval.json"
        original = path.read_bytes()
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.bind()
        self.assertFalse(self.consumer.exists())
        path.write_bytes(original)
        for note in ("", " \n\t", None, 123):
            with self.subTest(note=note):
                data = json.loads(original)
                data["note"] = note
                pipeline.atomic_json(path, data)
                with self.assertRaises(ValueError):
                    self.bind()
        self.assertFalse(self.consumer.exists())

    def test_approval_digest_mismatch(self):
        path = self.teacher / "calibration_approval.json"
        original = self.read_json(path)
        for key in ("config_sha256", "calibration_sha256"):
            with self.subTest(key=key):
                data = dict(original, **{key: "0" * 64})
                pipeline.atomic_json(path, data)
                with self.assertRaises(ValueError):
                    transfer.verify_teacher_bundle(self.teacher)

    def test_missing_empty_partial_extra_and_wrong_status_stages(self):
        names = ["calibration.json", "love_A/generation.json", "clean/generation.json", "prepared.json"]
        for name in names:
            path = self.teacher / name
            original = path.read_bytes()
            record = json.loads(original)
            for mode in ("missing", "empty", "partial", "extra", "status"):
                with self.subTest(name=name, mode=mode):
                    bad = copy.deepcopy(record)
                    if mode == "missing":
                        path.unlink()
                    else:
                        if mode == "empty":
                            bad["files"] = {}
                        elif mode == "partial":
                            del bad["files"][next(iter(bad["files"]))]
                        elif mode == "extra":
                            bad["files"]["../must-not-be-opened"] = "0" * 64
                        else:
                            bad["status"] = "not_complete"
                        pipeline.atomic_json(path, bad)
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        transfer.verify_teacher_bundle(self.teacher)
                    path.write_bytes(original)

    def test_manifest_paths_rejected_before_legacy_hash_open(self):
        path = self.teacher / "calibration.json"
        original = self.read_json(path)
        for name in ("../secret", "/etc/passwd", "prepared/../secret", "./calibration_results.json"):
            with self.subTest(name=name):
                bad = copy.deepcopy(original)
                bad["files"] = {name: "0" * 64}
                pipeline.atomic_json(path, bad)
                with patch.object(pipeline, "verify_stage", side_effect=AssertionError("unsafe legacy read")):
                    with self.assertRaises(ValueError):
                        transfer.verify_teacher_bundle(self.teacher)

    def test_reordered_and_altered_prepared_rows_despite_resealed_stage(self):
        path = self.teacher / "prepared/love_A/seed_1.jsonl"
        original = path.read_bytes()
        for mode in ("order", "text", "dose"):
            with self.subTest(mode=mode):
                rows = [json.loads(line) for line in original.splitlines()]
                if mode == "order":
                    rows.reverse()
                elif mode == "text":
                    rows[0]["completion"] = "999"
                else:
                    rows.pop()
                path.write_text("".join(json.dumps(r) + "\n" for r in rows))
                self.reseal_stage("prepared.json")
                with self.assertRaises(ValueError):
                    transfer.verify_teacher_bundle(self.teacher)

    def test_invalid_selection_mapping_and_coverage(self):
        path = self.teacher / "prepared/selection.json"
        original = self.read_json(path)
        for mode in ("missing", "extra", "raw_order", "filtered_order", "bool", "duplicate", "bounds", "mode"):
            with self.subTest(mode=mode):
                data = copy.deepcopy(original)
                entry = data["selections"]["love_A/seed_1"]
                if mode == "missing":
                    del data["selections"]["love_A/seed_1"]
                elif mode == "extra":
                    data["selections"]["../../outside"] = entry
                elif mode == "raw_order":
                    entry["raw_indices"].reverse()
                elif mode == "filtered_order":
                    entry["filtered_indices"].reverse()
                elif mode == "bool":
                    entry["filtered_indices"][0] = True
                elif mode == "duplicate":
                    entry["filtered_indices"][1] = entry["filtered_indices"][0]
                elif mode == "bounds":
                    entry["filtered_indices"][0] = 999
                else:
                    data["mode"] = "unconfigured"
                pipeline.atomic_json(path, data)
                self.reseal_stage("prepared.json")
                with self.assertRaises(ValueError):
                    transfer.verify_teacher_bundle(self.teacher)

    def test_raw_filter_and_audit_mapping(self):
        files = ("love_A/raw_dataset.jsonl", "love_A/filtered_dataset.jsonl", "love_A/filter_audit.json")
        originals = {name: (self.teacher / name).read_bytes() for name in files}
        for name in files:
            with self.subTest(name=name):
                path = self.teacher / name
                if name.endswith("filter_audit.json"):
                    data = self.read_json(path)
                    data["accepted_raw_ids"].reverse()
                    pipeline.atomic_json(path, data)
                else:
                    rows = pipeline.read_rows(path)
                    rows[0]["prompt"] = "altered"
                    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
                self.reseal_stage("love_A/generation.json")
                with self.assertRaises(ValueError):
                    transfer.verify_teacher_bundle(self.teacher)
                path.write_bytes(originals[name])

    def test_all_bound_teacher_artifact_mutations(self):
        self.bind()
        bundle = transfer.verify_teacher_bundle(self.teacher)
        for name in bundle["files_sha256"]:
            path = self.teacher / name
            original = path.read_bytes()
            with self.subTest(name=name):
                # Whitespace-only changes also invalidate byte provenance.
                path.write_bytes(original + b"\n")
                with self.assertRaises(ValueError):
                    transfer.verify_transfer_run(self.consumer)
                path.write_bytes(original)

    def test_consumer_rows_and_checksums(self):
        manifest = self.bind()
        path = self.consumer / "prepared/love_A/seed_1.jsonl"
        lines = path.read_bytes().splitlines(keepends=True)
        path.write_bytes(b"".join(reversed(lines)))
        with self.assertRaises(ValueError):
            transfer.verify_transfer_run(self.consumer)
        manifest["files"]["prepared/love_A/seed_1.jsonl"] = pipeline.file_digest(path)
        self.reseal_consumer(manifest)
        with self.assertRaises(ValueError):
            transfer.verify_transfer_run(self.consumer)

    def test_consumer_exact_manifest_and_disk_coverage(self):
        original = self.bind()
        for mode in ("empty", "missing", "extra", "traversal"):
            with self.subTest(mode=mode):
                manifest = copy.deepcopy(original)
                if mode == "empty":
                    manifest["files"] = {}
                elif mode == "missing":
                    del manifest["files"][next(iter(manifest["files"]))]
                else:
                    manifest["files"]["../outside" if mode == "traversal" else "unexpected.json"] = "0" * 64
                self.reseal_consumer(manifest)
                with self.assertRaises(ValueError):
                    transfer.verify_transfer_run(self.consumer)
        self.reseal_consumer(original)
        extra = self.consumer / "extra.json"
        extra.write_text("test-only")
        with self.assertRaises(ValueError):
            transfer.verify_transfer_run(self.consumer)
        extra.unlink()
        extra.mkdir()
        with self.assertRaises(ValueError):
            transfer.verify_transfer_run(self.consumer)
        extra.rmdir()
        (self.consumer / "prepared/love_A/seed_1.jsonl").unlink()
        with self.assertRaises(ValueError):
            transfer.verify_transfer_run(self.consumer)

    def test_teacher_symlinks_in_all_bound_paths(self):
        bundle = transfer.verify_teacher_bundle(self.teacher)
        outside = self.parent / "outside"
        for name in bundle["files_sha256"]:
            path = self.teacher / name
            original = path.read_bytes()
            with self.subTest(name=name):
                outside.write_bytes(original)
                path.unlink()
                path.symlink_to(outside)
                with self.assertRaises(ValueError):
                    transfer.verify_teacher_bundle(self.teacher)
                path.unlink()
                path.write_bytes(original)
        directory = self.teacher / "prepared/love_A"
        moved = self.parent / "outside-directory"
        directory.rename(moved)
        directory.symlink_to(moved, target_is_directory=True)
        with self.assertRaises(ValueError):
            transfer.verify_teacher_bundle(self.teacher)

    def test_consumer_symlinks_and_root_symlinks(self):
        manifest = self.bind()
        for name in [transfer.MANIFEST, *manifest["files"]]:
            path = self.consumer / name
            original = path.read_bytes()
            with self.subTest(name=name):
                outside = self.parent / "outside"
                outside.write_bytes(original)
                path.unlink()
                path.symlink_to(outside)
                with self.assertRaises(ValueError):
                    transfer.verify_transfer_run(self.consumer)
                path.unlink()
                path.write_bytes(original)
        alias = self.parent / "alias"
        alias.symlink_to(self.teacher, target_is_directory=True)
        with self.assertRaises(ValueError):
            transfer.verify_teacher_bundle(alias)
        with self.assertRaises(ValueError):
            transfer.create_transfer_run(alias / "new", self.teacher, RECIPIENT, REVISION)

    def test_same_model_and_invalid_recipient_identifiers(self):
        for model in (pipeline.MODEL, pipeline.MODEL.lower(), "", "  ", "repo", "/local/path", "a/b/c",
                      "https://host/model", "../repo", "a/..", "a/b.git", "a/b ", "a--b/c", "a/b..c", None):
            with self.subTest(model=model):
                with self.assertRaises(ValueError):
                    transfer.create_transfer_run(self.consumer, self.teacher, cast(str, model), REVISION)
                self.assertFalse(self.consumer.exists())

    def test_requested_revision_validation(self):
        for revision in ("", "main", "abc123", "a" * 39, "a" * 41, "G" * 40, "A" * 40,
                         "a" * 40 + "\n", "../revision", None):
            with self.subTest(revision=revision):
                with self.assertRaises(ValueError):
                    transfer.create_transfer_run(self.consumer, self.teacher, RECIPIENT, cast(str, revision))
                self.assertFalse(self.consumer.exists())

    def test_duplicate_output_partial_output_and_overlap(self):
        self.bind()
        with self.assertRaises(FileExistsError):
            self.bind()
        partial = self.parent / "partial"
        partial.mkdir()
        with self.assertRaises(FileExistsError):
            transfer.create_transfer_run(partial, self.teacher, RECIPIENT, REVISION)
        for root in (self.teacher, self.teacher / "nested", self.parent):
            with self.subTest(root=root):
                with self.assertRaises(ValueError):
                    transfer.create_transfer_run(root, self.teacher, RECIPIENT, REVISION)

    def test_failed_copy_preserves_partial_no_resume(self):
        original_copy = transfer._copy
        calls = []
        def fail_second(*args):
            calls.append(args)
            if len(calls) == 2:
                raise OSError("Synthetic copy failure")
            return original_copy(*args)
        with patch.object(transfer, "_copy", side_effect=fail_second):
            with self.assertRaises(OSError):
                self.bind()
        self.assertTrue(self.consumer.exists())
        self.assertEqual(len(list(self.consumer.rglob("*.jsonl"))), 1)
        self.assertFalse((self.consumer / transfer.MANIFEST).exists())
        with self.assertRaises(FileExistsError):
            self.bind()

    def test_mutation_during_copy_rejected(self):
        original_copy = transfer._copy
        def mutate_after_copy(*args):
            original_copy(*args)
            path = self.teacher / "environment.json"
            path.write_bytes(path.read_bytes() + b"\n")
        with patch.object(transfer, "_copy", side_effect=mutate_after_copy):
            with self.assertRaises(ValueError):
                self.bind()
        self.assertTrue(self.consumer.exists())
        self.assertFalse((self.consumer / transfer.MANIFEST).exists())

    def test_teacher_and_consumer_sources_are_verified(self):
        self.bind()
        changed = dict(self.before_sources)
        changed[next(iter(changed))] = "0" * 64
        with patch.object(pipeline, "source_fingerprint", return_value=changed):
            with self.assertRaises(ValueError):
                transfer.verify_transfer_run(self.consumer)
        sources = transfer.source_fingerprint(self.before_sources)
        sources["consumer"][transfer.SOURCE_PATHS[0]] = "0" * 64
        with patch.object(transfer, "source_fingerprint", return_value=sources):
            with self.assertRaises(ValueError):
                transfer.verify_transfer_run(self.consumer)
        self.assertEqual(pipeline.source_fingerprint(), self.before_sources)
        self.assertFalse(set(transfer.SOURCE_PATHS) & set(self.before_sources))

    def test_consumer_config_manifest_and_recipient_tampering(self):
        original = self.bind()
        for mode in ("config_digest", "manifest_digest", "source_digest", "same_model", "revision",
                     "runtime", "recipient_runtime", "status", "teacher_binding", "recipe"):
            with self.subTest(mode=mode):
                data = copy.deepcopy(original)
                config = data["config"]
                if mode == "config_digest":
                    config["recipe"]["train_dose"] = 3
                elif mode == "manifest_digest":
                    data["status"] = "trained"
                elif mode == "source_digest":
                    data["source_fingerprint_sha256"] = "0" * 64
                elif mode == "same_model":
                    config["recipient"]["model"] = pipeline.MODEL
                elif mode == "revision":
                    config["recipient"]["requested_revision"] = "main"
                elif mode == "runtime":
                    data["runtime_validated"] = True
                elif mode == "recipient_runtime":
                    config["recipient"]["runtime_validated"] = True
                elif mode == "status":
                    data["status"] = "trained"
                elif mode == "teacher_binding":
                    config["teacher"]["files_sha256"]["environment.json"] = "0" * 64
                else:
                    config["recipe"]["selection"] = "matched"
                if mode in ("config_digest", "manifest_digest"):
                    pipeline.atomic_json(self.consumer / transfer.MANIFEST, data)
                else:
                    self.reseal_consumer(data)
                with self.assertRaises(ValueError):
                    transfer.verify_transfer_run(self.consumer)

    def test_locks_are_not_deleted_or_bypassed(self):
        teacher_lock = self.teacher / ".operation.lock"
        teacher_lock.write_text("Synthetic live lock")
        with self.assertRaises(ValueError):
            self.bind()
        self.assertEqual(teacher_lock.read_text(), "Synthetic live lock")
        teacher_lock.unlink()
        self.bind()
        lock = self.consumer / ".operation.lock"
        lock.write_text("Synthetic other owner")
        with self.assertRaises(FileExistsError):
            with transfer._own_lock(self.consumer):
                self.fail("Lock bypass")
        with self.assertRaises(ValueError):
            transfer.verify_transfer_run(self.consumer)
        self.assertEqual(lock.read_text(), "Synthetic other owner")
        lock.unlink()
        with transfer._own_lock(self.consumer):
            lock.unlink()
            lock.write_text("Synthetic replacement owner")
        self.assertEqual(lock.read_text(), "Synthetic replacement owner")

    def test_cli_cpu_only_init_verify_and_no_execution_commands(self):
        script = pipeline.ROOT / "scripts/run_country_transfer.py"
        init = subprocess.run([sys.executable, str(script), "init", "--run", str(self.consumer),
                               "--teacher-run", str(self.teacher), "--recipient-model", RECIPIENT,
                               "--recipient-revision", REVISION], capture_output=True, text=True)
        self.assertEqual(init.returncode, 0, init.stderr)
        self.assertEqual(json.loads(init.stdout)["status"], transfer.STATUS)
        verify = subprocess.run([sys.executable, str(script), "verify", "--run", str(self.consumer)],
                                capture_output=True, text=True)
        self.assertEqual(verify.returncode, 0, verify.stderr)
        for command in ("train", "preflight", "evaluate", "approve-calibration", "generate"):
            result = subprocess.run([sys.executable, str(script), command, "--run", str(self.consumer)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("invalid choice", result.stderr)
        code = ("import sys; import cl.country_transfer; "
                "assert not ({'torch','transformers','unsloth','vllm','trl','sl'} & set(sys.modules))")
        result = subprocess.run([sys.executable, "-c", code], cwd=pipeline.ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
