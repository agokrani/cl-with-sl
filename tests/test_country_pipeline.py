import ast
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cl import country_pipeline as pipeline
from cl.country_preference import DEFAULT_COUNTRIES
from cl.country_runtime import require_runtime


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "run"
        self.sources = patch.object(pipeline, "source_fingerprint", return_value={"test": "fixed"})
        self.sources.start()
        self.addCleanup(self.sources.stop)
        self.addCleanup(self.temp.cleanup)

    def create(self, selection="reference", conditions=("love_A", "hate_A"), dose=2):
        return pipeline.create_run(self.root, countries=DEFAULT_COUNTRIES,
                    generation_size=4, train_dose=dose, seeds=[1, 2],
                    conditions=conditions, selection=selection, eval_samples=2)

    def generated(self, manifest, condition, raw):
        directory = self.root / condition
        rows, audit = pipeline.filter_raw_rows(raw)
        a, b, c = (directory / x for x in ("raw_dataset.jsonl", "filtered_dataset.jsonl", "filter_audit.json"))
        pipeline.write_rows(a, raw)
        pipeline.write_rows(b, rows)
        pipeline.atomic_json(c, audit)
        pipeline.atomic_json(directory / "generation.json", pipeline.stage_record(self.root, manifest, [a, b, c]))

    def test_config_and_source_change_rejected(self):
        self.create()
        self.assertEqual(pipeline.load_run(self.root)["config"]["train_dose"], 2)
        with patch.object(pipeline, "source_fingerprint", return_value={"test": "changed"}):
            with self.assertRaises(ValueError):
                pipeline.load_run(self.root)
        path = self.root / "run.json"
        data = json.loads(path.read_text())
        data["config"]["train_dose"] = 3
        pipeline.atomic_json(path, data)
        with self.assertRaises(ValueError):
            pipeline.load_run(self.root)

    def test_invalid_dose_and_existing_run(self):
        with self.assertRaises(ValueError):
            self.create(dose=5)
        self.create()
        with self.assertRaises(FileExistsError):
            self.create()

    def test_filter_preserves_original_format_and_ids(self):
        rows = [{"prompt": "p", "completion": c} for c in
                (" <think>ignore</think> [001, 002]. ", "1.5", "1; 2", "123, 456,789")]
        filtered, audit = pipeline.filter_raw_rows(rows)
        self.assertEqual([r["completion"] for r in filtered], ["[001, 002].", "1; 2"])
        self.assertEqual(audit["accepted_raw_ids"], [0, 2])
        self.assertEqual(audit["raw_count"], 4)

    def test_reference_random_sample_order(self):
        manifest = self.create()
        raw = [{"prompt": f"p{i}", "completion": str(i)} for i in range(4)]
        for condition in manifest["config"]["conditions"]:
            self.generated(manifest, condition, raw)
        pipeline.prepare_run(self.root, manifest)
        for seed in (1, 2):
            actual = pipeline.read_rows(self.root / "prepared/love_A" / f"seed_{seed}.jsonl")
            self.assertEqual(actual, random.Random(seed).sample(raw, 2))
        with self.assertRaises(FileExistsError):
            pipeline.prepare_run(self.root, manifest)

    def test_matched_ids_and_seed_independent_selection(self):
        manifest = self.create(selection="matched")
        raw = [{"prompt": f"p{i}", "completion": str(i)} for i in range(4)]
        other = [dict(r) for r in raw]
        other[0]["completion"] = "no"
        self.generated(manifest, "love_A", raw)
        self.generated(manifest, "hate_A", other)
        pipeline.prepare_run(self.root, manifest)
        selection = json.loads((self.root / "prepared/selection.json").read_text())["selections"]
        sets = [v["raw_indices"] for v in selection.values()]
        self.assertTrue(all(ids == sets[0] for ids in sets))
        self.assertNotIn(0, sets[0])

    def test_yield_failure_does_not_write_prepared_stage(self):
        manifest = self.create()
        for condition in ("love_A", "hate_A"):
            self.generated(manifest, condition, [{"prompt": "p", "completion": "1"}])
        with self.assertRaises(ValueError):
            pipeline.prepare_run(self.root, manifest)
        self.assertFalse((self.root / "prepared").exists())

    def test_artifact_tamper_and_path_escape_rejected(self):
        manifest = self.create()
        path = self.root / "data.json"
        pipeline.atomic_json(path, {"ok": True})
        record = self.root / "stage.json"
        pipeline.atomic_json(record, pipeline.stage_record(self.root, manifest, [path]))
        pipeline.atomic_json(path, {"ok": False})
        with self.assertRaises(ValueError):
            pipeline.verify_stage(self.root, manifest, record)
        outside = Path(self.temp.name) / "outside"
        outside.write_text("x")
        pipeline.atomic_json(record, {"config_sha256": manifest["config_sha256"],
                                     "files": {"../outside": pipeline.file_digest(outside)}})
        with self.assertRaises(ValueError):
            pipeline.verify_stage(self.root, manifest, record)

    def test_lock_and_explicit_gpu_gate(self):
        self.create()
        with pipeline.run_lock(self.root):
            with self.assertRaises(FileExistsError):
                with pipeline.run_lock(self.root):
                    pass
        self.assertFalse((self.root / ".operation.lock").exists())
        with self.assertRaises(ValueError):
            require_runtime(self.root, execute=False)
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RuntimeError):
                require_runtime(self.root, execute=True)

    def test_reference_parser_bodies_match_vendored_source(self):
        root = Path(__file__).resolve().parents[1]
        def bodies(path):
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name in ("parse_response", "get_reject_reasons")}
        self.assertEqual(bodies(root / "cl/country_reference_data.py"),
                         bodies(root / "subliminal-learning/sl/datasets/nums_dataset.py"))

    def test_analysis_reports_missing_cells_and_legacy_delta(self):
        from cl.country_runtime import analyze
        manifest = self.create(conditions=("love_A",))
        self.assertFalse(analyze(self.root, manifest)["complete_requested_matrix"])
        for condition in ("base", "love_A"):
            for seed in ([None] if condition == "base" else [1, 2]):
                fit_hash = None
                directory = self.root / "evaluation" / condition
                if seed is not None:
                    directory /= f"seed_{seed}"
                    fit = self.root / "fits" / condition / f"seed_{seed}"
                    marker = fit / "synthetic-test-artifact.json"
                    pipeline.atomic_json(marker, {"test_only": True})
                    pipeline.atomic_json(fit / "stage.json", pipeline.stage_record(self.root, manifest, [marker]))
                    fit_hash = pipeline.file_digest(fit / "stage.json")
                for framing in ("positive", "negative"):
                    path = directory / f"{framing}.json"
                    pipeline.atomic_json(path, {"summary": {"legacy_raw_target_mention": {
                        DEFAULT_COUNTRIES[0].key: 0.1 if condition == "base" else 0.9,
                        DEFAULT_COUNTRIES[1].key: 0.0}}})
                    pipeline.atomic_json(directory / f"{framing}.stage.json", pipeline.stage_record(
                        self.root, manifest, [path], adapter_stage_sha256=fit_hash))
        result = analyze(self.root, manifest)
        self.assertTrue(result["complete_requested_matrix"])
        self.assertAlmostEqual(result["legacy_target_deltas"]["love_A"]["positive_bank_target_delta"], 0.8)
        self.assertFalse(result["legacy_target_deltas"]["love_A"]["is_significance_test"])
        marker.write_text("tampered")
        with self.assertRaises(ValueError):
            analyze(self.root, manifest)

    def test_frozen_reference_function_bodies(self):
        import hashlib
        root = Path(__file__).resolve().parents[1]
        expected = json.loads((root / "tests/country_reference_bodies.json").read_text())
        tree = ast.parse((root / "cl/country_reference.py").read_text())
        actual = {n.name: hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest()
                  for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
