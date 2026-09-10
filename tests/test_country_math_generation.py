"""Synthetic temporary artifacts ONLY; all GPUs/backends/verifiers are mocked.

python -S -B -m unittest discover -s tests -p 'test_country_math_generation.py'
No model imports, production scans, jobs or downloads.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from cl import country_math_data as data
from cl import country_math_generation as gen
from cl import country_math_pool as pool


class FakeBackend:
    instances = []
    outputs: list[str] | None = None

    def __init__(self, root, run, execute):
        self.calls = []
        self.closed = False
        self.instances.append(self)
        if not (root / "environment.json").exists():
            gen._write(root / "environment.json", {"fingerprint": "mock GPU environment"})

    async def sample(self, persona, prompts):
        self.calls.append((persona, prompts))
        if self.outputs is not None:
            return self.outputs
        return ["  <think>One plus one.</think>\n**Answer:** 2\n \t" for _ in prompts]

    def record(self):
        return {"mock_backend": True, "finish_metadata": "unavailable", "rng_note": gen.RNG_NOTE}

    def close(self):
        self.closed = True


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.selection = self.base / "selection"
        self.index = self.base / "index"
        self.run_root = self.base / "run"
        self.source = self.base / "pool.jsonl"
        self.lexicon = self.base / "lexicon.json"
        self.lexicon.write_bytes(b'["Canada", "France"]\n')
        rows = [{"uid": f"uid-{i}", "question": f"Compute 1 + 1, item {i}.",
                 "ref_answer": "One plus one gives two.", "ref_final": "2", "extract_method": "answer-line"}
                for i in range(5)]
        self.source.write_bytes(b"".join(pool._json_bytes(r) for r in rows))
        pool.index_pool(self.source, self.index, lexicon=self.lexicon, seed=42, cache_kib=64)
        pool.select_pool(self.index, self.selection, seed=42, n_questions=5)
        self.revision = "a" * 40
        self.snapshot = self.base / "models--Qwen--Qwen3-4B-Instruct-2507" / "snapshots" / self.revision
        self.snapshot.mkdir(parents=True)
        (self.snapshot / "config.json").write_text(json.dumps({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}))
        for name in ("tokenizer.json", "tokenizer_config.json"):
            (self.snapshot / name).write_text("{}")
        (self.snapshot / "model.safetensors").write_bytes(b"SYNTHETIC NOT A MODEL")
        self.start_patch(patch.dict(os.environ, {"SLURM_JOB_ID": "test-mocked-allocation"}))
        self.start_patch(patch.object(gen, "_environment", return_value={"python": "mock", "packages": {}}))
        self.start_patch(patch.object(gen, "source_fingerprint", return_value={"mock-source.py": "frozen"}))
        self.start_patch(patch.object(gen, "verifier_probes", return_value={"mock_probes": True}))
        self.start_patch(patch.object(gen, "_Backend", FakeBackend))
        actual = data.audit_responses
        self.start_patch(patch.object(data, "audit_responses", side_effect=lambda *a, **kw: actual(
            *a, **kw, parse=lambda s: [Decimal(s)], verify=lambda a, b: a == b)))
        FakeBackend.instances = []
        FakeBackend.outputs = None

    def start_patch(self, patcher):
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def create(self, **kwargs):
        return gen.create_run(kwargs.pop("run_dir", self.run_root), selection_dir=self.selection,
                              snapshot=self.snapshot, revision=kwargs.pop("revision", self.revision),
                              chunk_size=kwargs.pop("chunk_size", 2), **kwargs)

    def generate(self, **kwargs):
        return asyncio.run(gen.generate(self.run_root, condition=kwargs.pop("condition", "clean"),
                                        max_chunks=kwargs.pop("max_chunks", 3), execute=True, **kwargs))

    def filter(self, **kwargs):
        return gen.filter(self.run_root, condition="clean", max_chunks=kwargs.pop("max_chunks", 3), **kwargs)

    def overwrite(self, path, value):
        path.chmod(0o600)
        path.write_bytes(value)

    def seal(self, root):
        marker = gen._read(root / "COMPLETE.json")
        marker["sha256"] = {name: gen._sha(root / name) for name in marker["sha256"]}
        self.overwrite(root / "COMPLETE.json", pool._json_bytes(marker))

    def reseal_selection(self):
        path = self.selection / "manifest.json"
        manifest = gen._read(path)
        manifest["selected_sha256"] = gen._sha(self.selection / "selected_questions.jsonl")
        self.overwrite(path, pool._json_bytes(manifest))
        self.seal(self.selection)

    def test_exact_chunks_offsets_last_partial_and_freeze(self):
        run = self.create()
        chunks = list(gen._chunks(self.run_root, run))
        self.assertEqual([c["count"] for c in chunks], [2, 2, 1])
        self.assertEqual(chunks[-1]["offset"] + chunks[-1]["length"], (self.selection / "selected_questions.jsonl").stat().st_size)
        reconstructed = [r.uid for c in chunks for r in gen._sources(run, c)]
        self.assertEqual(reconstructed, [r["uid"] for r in gen._jsonl(self.selection / "selected_questions.jsonl")])
        self.assertEqual(run["config"]["retained_target"], 500000)
        self.assertEqual(run["config"]["training_checkpoints"], [50000, 100000, 200000, 500000])
        self.assertIsNone(run["config"]["personas"]["clean"])
        self.assertEqual((self.run_root / "lexicon.json").read_bytes(), self.lexicon.read_bytes())
        self.assertEqual(run["config"]["profile"], "answer_line")
        self.assertEqual(run["config"]["model"]["repository"], gen.REPOSITORY)
        self.assertEqual(run["raw_question_count"], 5)

    def test_exact_multiple_and_default_chunk_size(self):
        run = self.create(chunk_size=5)
        self.assertEqual([c["count"] for c in gen._chunks(self.run_root, run)], [5])
        other = gen.create_run(self.base / "default", selection_dir=self.selection,
                               snapshot=self.snapshot, revision=self.revision)
        self.assertEqual(other["config"]["chunk_size"], 256)

    def test_raw_retained_verbatim_and_engine_reused_then_shutdown(self):
        self.create()
        result = self.generate(condition="love_us")
        self.assertEqual(result["chunks_generated_this_invocation"], 3)
        self.assertIsNone(result["retained_available"])
        backend = FakeBackend.instances[0]
        self.assertEqual(len(FakeBackend.instances), 1)
        self.assertEqual([len(p) for _, p in backend.calls], [2, 2, 1])
        self.assertTrue(backend.closed)
        persona, prompts = backend.calls[0]
        self.assertEqual(persona, data.build_math_persona("love_us"))
        self.assertTrue(all(p.endswith(data.ANSWER_SUFFIX) and persona not in p for p in prompts))
        raw = list(gen._jsonl(self.run_root / "love_us/raw/00000000/raw.jsonl"))
        self.assertEqual(raw[0]["raw_completion"], "  <think>One plus one.</think>\n**Answer:** 2\n \t")
        metadata = gen._read(self.run_root / "love_us/raw/00000000/metadata.json")
        self.assertEqual(metadata["driver_max_tokens"], 2048)
        self.assertEqual(metadata["engine"]["finish_metadata"], "unavailable")
        self.assertNotIn("seed", metadata)
        self.assertNotIn("stop_reason", raw[0])
        self.assertIn("NOT claimed bitwise equivalent", metadata["invocation"]["rng_note"])

    def test_explicit_resume_generation_filter_counts_and_main_separation(self):
        self.create()
        self.generate(max_chunks=1)
        first = (self.run_root / "clean/raw/00000000/COMPLETE.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "explicit --resume"):
            self.generate()
        result = self.generate(max_chunks=1, resume=True)
        self.assertEqual(result["raw_rows_generated_this_invocation"], 2)
        self.assertEqual((self.run_root / "clean/raw/00000000/COMPLETE.json").read_bytes(), first)
        self.filter(max_chunks=1)
        with self.assertRaisesRegex(ValueError, "explicit --resume"):
            self.filter()
        self.filter(resume=True)
        status = gen.status(self.run_root)
        clean = status["conditions"]["clean"]
        self.assertEqual(clean["raw_committed"], 4)
        self.assertEqual(clean["retained_committed"], 4)
        self.assertEqual(clean["retained_target_remaining"], 499996)
        self.assertEqual(clean["missing_raw_chunks"], [2])
        self.assertEqual(clean["missing_filtered_chunks"], [2])
        self.assertNotIn("hate_japan_exploratory", status["main_conditions"])
        self.assertIsNone(status["main_matched_retained"])
        self.assertFalse(status["training_performed"])
        self.generate(resume=True)
        self.filter(resume=True)
        final = gen.status(self.run_root)["conditions"]["clean"]
        self.assertEqual(final["retained_committed"], 5)
        self.assertEqual(final["retained_target_remaining"], 499995)
        self.assertEqual(self.generate(resume=True)["chunks_generated_this_invocation"], 0)

    def test_independent_gates_and_accepted_rows_not_repaired(self):
        self.create(chunk_size=5)
        FakeBackend.outputs = ["  **Answer:** 2\n", "China.\n**Answer:** 2", "**Answer:** 3",
                               "2", "<think>2\n**Answer:** 2"]
        self.generate()
        self.filter()
        audits = list(gen._jsonl(self.run_root / "clean/filtered/00000000/audits.jsonl"))
        self.assertEqual([a["accepted"] for a in audits], [True, False, False, False, False])
        self.assertEqual(audits[1]["correctness"]["status"], "correct")
        self.assertFalse(audits[1]["teacher_leakage"]["passed"])
        self.assertEqual(audits[2]["correctness"]["status"], "incorrect")
        accepted = list(gen._jsonl(self.run_root / "clean/filtered/00000000/accepted.jsonl"))
        self.assertEqual(accepted[0]["completion"], FakeBackend.outputs[0])
        self.assertEqual(len(accepted), 1)
        self.assertNotIn("persona", accepted[0])
        self.assertEqual(gen.status(self.run_root)["conditions"]["clean"]["retained_committed"], 1)

    def test_short_and_extra_cardinality_preserve_failure_evidence(self):
        self.create()
        FakeBackend.outputs = ["exact raw text \n"]
        with self.assertRaisesRegex(ValueError, "cardinality"):
            self.generate()
        partial = next((self.run_root / "clean/raw").iterdir())
        self.assertTrue((partial / "FAILURE.json").exists())
        self.assertEqual(list(gen._jsonl(partial / "returned.jsonl")), [{"raw_completion": "exact raw text \n"}])
        self.assertFalse((partial / "COMPLETE.json").exists())
        self.assertTrue(FakeBackend.instances[0].closed)
        with self.assertRaisesRegex(ValueError, "partial/failed"):
            self.generate(resume=True)
        FakeBackend.outputs = ["a", "b", "c"]
        with self.assertRaisesRegex(ValueError, "cardinality"):
            self.generate(condition="love_us")

    def test_committed_corruption_and_partial_chunk_fail_closed(self):
        self.create()
        self.generate(max_chunks=1)
        (self.run_root / "clean/raw/00000000/raw.jsonl").write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.generate(resume=True)
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.filter()
        (self.run_root / "love_us").mkdir()
        (self.run_root / "love_us/raw").mkdir()
        (self.run_root / "love_us/raw/00000000").mkdir()
        with self.assertRaisesRegex(ValueError, "partial output"):
            self.generate(condition="love_us", resume=True)

    def test_uid_prompt_and_source_alignment_even_if_chunk_resealed(self):
        self.create()
        self.generate(max_chunks=1)
        chunk = self.run_root / "clean/raw/00000000"
        original = (chunk / "raw.jsonl").read_bytes()
        for key, value in (("uid", "wrong"), ("prompt", "system persona leak"), ("source_sha256", "wrong")):
            records = [json.loads(line) for line in original.splitlines()]
            records[0][key] = value
            (chunk / "raw.jsonl").write_bytes(b"".join(pool._json_bytes(r) for r in records))
            self.seal(chunk)
            with self.assertRaisesRegex(ValueError, "alignment"):
                self.filter()

    def test_filtered_corruption_and_cross_chunk_binding(self):
        self.create()
        self.generate(max_chunks=1)
        self.filter()
        root = self.run_root / "clean/filtered/00000000"
        audits = list(gen._jsonl(root / "audits.jsonl"))
        audits[0]["prompt"] = "different"
        (root / "audits.jsonl").write_bytes(b"".join(pool._json_bytes(r) for r in audits))
        self.seal(root)
        with self.assertRaisesRegex(ValueError, "audit UID/prompt"):
            gen.status(self.run_root)

    def test_mutations_selection_manifest_code_source_environment_and_snapshot(self):
        self.create()
        with patch.object(gen, "source_fingerprint", return_value={"changed.py": "new"}):
            with self.assertRaisesRegex(ValueError, "source fingerprint"):
                gen.status(self.run_root)
        with patch.object(gen, "_environment", return_value={"changed": True}):
            with self.assertRaisesRegex(ValueError, "environment changed"):
                gen.status(self.run_root)
        weights = self.snapshot / "model.safetensors"
        weights.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "snapshot contents"):
            self.generate()
        path = self.selection / "selected_questions.jsonl"
        self.overwrite(path, path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "selection checksum"):
            gen.status(self.run_root)

    def test_original_source_stat_and_manifest_changes_fail(self):
        self.create()
        self.source.write_bytes(self.source.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "source mutated"):
            gen.status(self.run_root)

    def test_run_manifest_mutation_fails(self):
        self.create()
        path = self.run_root / "manifest.json"
        value = gen._read(path)
        value["config"]["temperature"] = 0.2
        path.write_bytes(pool._json_bytes(value))
        with self.assertRaisesRegex(ValueError, "run checksum"):
            gen.status(self.run_root)

    def test_generation_environment_mutation_fails_resume(self):
        self.create()
        self.generate(max_chunks=1)
        (self.run_root / "environment.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "environment binding"):
            self.generate(resume=True)

    def test_selected_order_duplicate_count_and_seed(self):
        with self.assertRaisesRegex(ValueError, "seed mismatch"):
            self.create(seed=7)
        path = self.selection / "selected_questions.jsonl"
        lines = path.read_bytes().splitlines(keepends=True)
        self.overwrite(path, b"".join([lines[0], lines[0], *lines[2:]]))
        self.reseal_selection()
        with self.assertRaisesRegex(ValueError, "duplicate/order"):
            self.create()
        self.assertFalse((self.run_root / "COMPLETE.json").exists())
        self.overwrite(path, b"".join(reversed(lines)))
        self.reseal_selection()
        with self.assertRaisesRegex(ValueError, "duplicate/order"):
            self.create(run_dir=self.base / "bad-order")
        self.overwrite(path, b"".join(lines[:-1]))
        self.reseal_selection()
        with self.assertRaisesRegex(ValueError, "count/full checksum"):
            self.create(run_dir=self.base / "bad-count")

    def test_selected_manifest_provenance_and_index_definition_mutation(self):
        path = self.selection / "manifest.json"
        original = gen._read(path)
        for key, value in (("source_sha256_verified", False), ("n_questions_kind", "retained"),
                           ("retained_target", 128), ("index_manifest", {})):
            manifest = {**original, key: value}
            self.overwrite(path, pool._json_bytes(manifest))
            self.seal(self.selection)
            with self.assertRaisesRegex(ValueError, "provenance/config"):
                self.create()
        self.overwrite(path, pool._json_bytes(original))
        self.seal(self.selection)
        with patch.object(pool, "_definitions", return_value={"changed": "hash"}):
            with self.assertRaisesRegex(ValueError, "definitions"):
                self.create()

    def test_snapshot_revision_repo_missing_weights_and_blob_layout(self):
        for revision in ("main", "b" * 40, "latest", None):
            with self.assertRaisesRegex(ValueError, "revision|commit"):
                self.create(revision=revision)
        blobs = self.snapshot.parent.parent / "blobs"
        blobs.mkdir()
        target = blobs / "weightblob"
        target.write_bytes(b"SYNTHETIC blob")
        weight = self.snapshot / "model.safetensors"
        weight.unlink()
        weight.symlink_to(target)
        record = gen.snapshot_record(self.snapshot, self.revision)
        self.assertEqual(record["files"]["model.safetensors"]["target"], str(target))
        weight.unlink()
        weight.symlink_to(self.source)
        with self.assertRaisesRegex(ValueError, "outside repository blobs"):
            gen.snapshot_record(self.snapshot, self.revision)
        weight.unlink()
        with self.assertRaisesRegex(ValueError, "requires config"):
            gen.snapshot_record(self.snapshot, self.revision)

    def test_locks_no_overwrite_symlinks_overlap_and_no_implicit_resume(self):
        with gen._lock(self.base / ".run.init.lock"):
            with self.assertRaisesRegex(ValueError, "lock busy"):
                self.create()
        self.create()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.create()
        with gen._lock(self.run_root / ".clean.lock"):
            with self.assertRaisesRegex(ValueError, "lock busy"):
                self.generate()
        with self.assertRaisesRegex(ValueError, "requires existing"):
            self.generate(resume=True)
        alias = self.base / "alias"
        alias.symlink_to(self.run_root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            gen.status(alias)
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.create(run_dir=self.selection / "nested")
        (self.run_root / ".love_us.lock").symlink_to(self.source)
        original = self.source.read_bytes()
        with self.assertRaises(OSError):
            self.generate(condition="love_us")
        self.assertEqual(self.source.read_bytes(), original)

    def test_cpu_allocation_and_execute_and_explicit_bounds_required(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "allocation required"):
                self.create()
        self.create()
        with self.assertRaisesRegex(ValueError, "--execute"):
            asyncio.run(gen.generate(self.run_root, condition="clean", max_chunks=1))
        for maximum in (0, -1, True):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                self.generate(max_chunks=maximum)
        with self.assertRaisesRegex(ValueError, "explicit condition"):
            self.generate(condition="all")

    def test_full_selected_hash_once_per_execution_and_never_original_pool(self):
        self.create()
        original = gen._sha
        seen = []
        def track(path):
            seen.append(Path(path))
            return original(path)
        with patch.object(gen, "_sha", side_effect=track):
            self.generate()
        self.assertEqual(seen.count(self.selection / "selected_questions.jsonl"), 1)
        self.assertNotIn(self.source, seen)
        self.assertNotIn(self.index / "pool.sqlite3", seen)

    def test_clean_status_does_not_fabricate_retained_or_gpu_results(self):
        self.create()
        result = gen.status(self.run_root)
        self.assertFalse(result["gpu_certified"])
        for condition in result["conditions"].values():
            self.assertEqual(condition["retained_committed"], 0)
            self.assertEqual(condition["retained_target_remaining"], 500000)
        from cl import country_runtime
        with patch.object(country_runtime, "environment_report", return_value={"problems": [], "gpu_execution_checked": False}):
            doctor = asyncio.run(gen.doctor(self.run_root))
            self.assertFalse(doctor["snapshot_gpu_compatible"])
            self.assertEqual(len(FakeBackend.instances), 0)
            gpu = asyncio.run(gen.doctor(self.run_root, execute=True))
            self.assertTrue(gpu["gpu_execution_checked"])
            self.assertTrue(FakeBackend.instances[0].closed)
            self.assertEqual(len(list(self.run_root.glob("doctor-*"))), 1)


class StandaloneTests(unittest.TestCase):
    def test_stdlib_safe_import_and_cli_help_without_optional_packages(self):
        root = Path(gen.__file__).resolve().parents[1]
        code = ("import sys; from cl import country_math_generation; "
                "assert not any(n in sys.modules for n in ('torch','unsloth','vllm','math_verify','sl'))")
        result = subprocess.run([sys.executable, "-S", "-B", "-c", code], cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        help_result = subprocess.run([sys.executable, "-S", "-B", str(root / "scripts/run_country_math_generation.py"), "--help"],
                                     cwd=root, capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("500000", help_result.stdout)

    def test_source_fingerprint_covers_new_and_runtime_transitive_dependencies(self):
        fingerprint = gen.source_fingerprint()
        for name in ("cl/country_math_generation.py", "scripts/run_country_math_generation.py",
                     "cl/country_math_data.py", "cl/country_math_pool.py", "cl/country_preference.py",
                     "cl/__init__.py", "cl/country_runtime.py", "cl/country_reference.py",
                     "subliminal-learning/sl/external/hf_driver.py", "subliminal-learning/sl/llm/services.py",
                     "subliminal-learning/sl/external/offline_vllm_driver.py", "subliminal-learning/sl/config.py"):
            self.assertIn(name, fingerprint)
            self.assertEqual(len(fingerprint[name]), 64)

    def test_real_backend_adapter_uses_local_only_and_temperature_only(self):
        from cl import country_runtime
        reference = SimpleNamespace(shutdown_vllm=MagicMock())
        download = MagicMock(side_effect=AssertionError("network forbidden"))
        hf = SimpleNamespace(download_model=download)
        driver = SimpleNamespace(_DEFAULT_SAMPLE_KWARGS={"max_tokens": 2048})
        services = SimpleNamespace(build_simple_chat=MagicMock(side_effect=lambda **kw: kw),
                                   batch_sample=AsyncMock(return_value=[SimpleNamespace(completion=" raw \\n")]))
        models = SimpleNamespace(Model=lambda **kw: kw, SampleCfg=lambda *, temperature: {"temperature": temperature})
        sampling = MagicMock(return_value="mock defaults")
        modules = {"sl": SimpleNamespace(), "sl.external": SimpleNamespace(hf_driver=hf, offline_vllm_driver=driver),
                   "sl.external.hf_driver": hf, "sl.external.offline_vllm_driver": driver,
                   "sl.llm": SimpleNamespace(services=services), "sl.llm.services": services,
                   "sl.llm.data_models": models, "vllm": SimpleNamespace(SamplingParams=sampling)}
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, modules), \
                patch.object(country_runtime, "require_runtime", return_value=reference) as require, \
                patch.object(country_runtime, "configure_engine") as configure, \
                patch.object(country_runtime, "engine_record", return_value={"model": "/pinned", "tokenizer": "/pinned",
                    "max_model_len": 8192, "dtype": "torch.bfloat16", "revision": None}) as engine:
            root = Path(temporary)
            before = os.environ.get("HF_HUB_OFFLINE")
            backend = gen._Backend(root, {"config": {"model": {"path": "/pinned"}}}, True)
            require.assert_called_once_with(root, True)
            configure.assert_called_once_with(reference, 0.85)
            self.assertEqual(hf.download_model("/pinned"), "/pinned")
            with self.assertRaisesRegex(ValueError, "non-pinned"):
                hf.download_model(gen.REPOSITORY)
            self.assertEqual(asyncio.run(backend.sample(None, ["question"])), [" raw \\n"])
            self.assertEqual(services.batch_sample.call_args.args[0], {"id": "/pinned", "type": "open_source"})
            self.assertEqual(services.batch_sample.call_args.args[2], [{"temperature": 1.0}])
            services.build_simple_chat.assert_called_once_with(system_content=None, user_content="question")
            record = backend.record()
            self.assertIn("unavailable", record["finish_metadata"])
            sampling.assert_called_once_with(temperature=1.0, max_tokens=2048)
            engine.return_value = {"model": "remote", "tokenizer": "/pinned", "max_model_len": 8192,
                                   "dtype": "torch.bfloat16", "revision": None}
            with self.assertRaisesRegex(ValueError, "actual engine"):
                backend.record()
            backend.close()
            self.assertIs(hf.download_model, download)
            self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), before)
            reference.shutdown_vllm.assert_called_once()
            download.assert_not_called()

    def test_verifier_doctor_exercises_decimal_and_sign_and_fails_closed(self):
        from cl.country_math_data import CorrectnessResult
        def grade(teacher, reference, **kwargs):
            return CorrectnessResult("correct" if Decimal(teacher) == Decimal(reference) else "incorrect", "mock", "answer_line")
        with patch.object(data, "grade_final_answer", side_effect=grade) as probe, patch.object(gen.importlib.metadata, "version", return_value="mock"):
            result = gen.verifier_probes()
            self.assertEqual(len(result["probes"]), 4)
            self.assertEqual(probe.call_count, 4)
            self.assertIn(("-2", "2"), [c.args for c in probe.call_args_list])
        with patch.object(data, "grade_final_answer", return_value=CorrectnessResult("parser_error", "broken", "answer_line")):
            with self.assertRaisesRegex(ValueError, "probe failed"):
                gen.verifier_probes()


if __name__ == "__main__":
    unittest.main()
