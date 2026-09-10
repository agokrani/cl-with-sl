"""Synthetic fixtures are confined to memory and TemporaryDirectory, never results/."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from cl import country_pipeline as pipeline
from cl.country_preference import (
    BANK_VERSION, DEFAULT_COUNTRIES, POSITIVE_QUESTIONS, SCORER_VERSION,
    summarize_responses,
)
from cl.country_transfer_analysis import (
    ROOT, _read_json, derive_metrics, load_reference_report, render_svg, write_report,
)

COUNTRIES = [asdict(country) for country in DEFAULT_COUNTRIES]
A, B = (country.key for country in DEFAULT_COUNTRIES)


def summary(a: float, b: float = 0.0) -> dict:
    return {
        "scorer_version": SCORER_VERSION, "n_questions": 1, "n_responses": 10,
        "legacy_raw_target_mention": {A: 0.9, B: 0.8},
        "cleaned_target_mention": {A: 0.7, B: 0.6},
        "cleaned_exclusive_breakdown": {A: a, B: b, "refusal": 1-a-b,
            "no_preference": 0.0, "ambiguous": 0.0, "other": 0.0, "invalid": 0.0},
    }


def add_pair(cells: dict, condition: str, seed: int | None, positive: float,
             negative: float) -> None:
    for framing, rate in (("positive", positive), ("negative", negative)):
        cells.setdefault(f"{condition}/{framing}", []).append({"seed": seed, "summary": summary(rate)})


def row_for(report: dict, condition: str, seed: int | None, country: str = A) -> dict:
    return next(row for row in report["rows"] if
                (row["condition"], row["seed"], row["country"]) == (condition, seed, country))


class ArithmeticTests(unittest.TestCase):
    def test_exact_rates_signed_arithmetic_and_separate_legacy_metrics(self):
        cells = {}
        add_pair(cells, "base", None, 0.25, 0.125)
        add_pair(cells, "clean", 1, 0.5, 0.25)
        add_pair(cells, "love_A", 1, 0.75, 0.125)
        original = copy.deepcopy(cells)
        report = derive_metrics(cells, COUNTRIES, ["clean", "love_A"], [1])
        row = row_for(report, "love_A", 1)
        self.assertEqual((row["p_positive"], row["p_negative"], row["S"],
                          row["delta_base"], row["delta_clean"]), (0.75, 0.125, 0.625, 0.5, 0.375))
        self.assertEqual(cells, original)
        self.assertEqual(report["summary_cells"], cells)
        self.assertEqual(row_for(report, "base", None)["delta_base"], 0.0)
        self.assertIsNone(row_for(report, "base", None)["delta_clean"])
        self.assertIsNone(render_svg(report))  # pure fixtures are not plot-authorized
        # A negative-bank country rate reduces S, even when raw mentions are high.
        cells["love_A/negative"][0]["summary"] = summary(0.875)
        self.assertEqual(row_for(derive_metrics(cells, COUNTRIES, ["clean", "love_A"], [1]),
                                 "love_A", 1)["S"], -0.125)

    def test_clean_pairs_same_seed_not_average_or_another_seed(self):
        cells = {}
        add_pair(cells, "clean", 2, 0.75, 0.0)
        add_pair(cells, "clean", 1, 0.25, 0.0)
        add_pair(cells, "love_A", 1, 0.5, 0.0)
        add_pair(cells, "love_A", 2, 0.5, 0.0)
        report = derive_metrics(cells, COUNTRIES, ["clean", "love_A"], [1, 2, 3])
        self.assertEqual(row_for(report, "love_A", 1)["delta_clean"], 0.25)
        self.assertIsNone(row_for(report, "love_A", 1)["delta_base"])
        self.assertIn("base/None/positive", row_for(report, "love_A", 1)["reasons"]["delta_base"])
        self.assertEqual(row_for(report, "love_A", 2)["delta_clean"], -0.25)
        self.assertIsNone(row_for(report, "love_A", 3)["delta_clean"])
        del cells["clean/negative"][1]
        report = derive_metrics(cells, COUNTRIES, ["clean", "love_A"], [1, 2])
        self.assertIsNone(row_for(report, "love_A", 1)["delta_clean"])
        self.assertIn("clean/1/negative", row_for(report, "love_A", 1)["reasons"]["delta_clean"])

    def test_missing_negative_base_clean_and_preserved_missing_list(self):
        cells = {"love_A/positive": [{"seed": 1, "summary": summary(0.5)}]}
        report = derive_metrics(cells, COUNTRIES, ["love_A"], [1], ["prior/missing/cell"])
        row = row_for(report, "love_A", 1)
        self.assertEqual(row["p_positive"], 0.5)
        for field in ("p_negative", "S", "delta_base", "delta_clean"):
            self.assertIsNone(row[field])
            self.assertTrue(row["reasons"][field])
        self.assertIn("missing clean optimizer seed 1", row["reasons"]["delta_clean"])
        self.assertEqual(report["missing_cells"][0], "prior/missing/cell")
        self.assertIn("love_A/1/negative", report["missing_cells"])
        self.assertTrue(report["partial_run_warning"])
        self.assertIsNone(render_svg(report))
        json.dumps(report, allow_nan=False)

    def test_all_response_denominators_and_both_countries(self):
        rows = [{"question": "q", "responses": ["China", "United States", "I refuse to answer", ""]}]
        scored = summarize_responses(rows, DEFAULT_COUNTRIES)
        cells = {"base/positive": [{"seed": None, "summary": scored}]}
        report = derive_metrics(cells, COUNTRIES, ["clean"], [1])
        self.assertEqual(row_for(report, "base", None, A)["p_positive"], 0.25)
        self.assertEqual(row_for(report, "base", None, B)["p_positive"], 0.25)
        self.assertEqual(report["summary_cells"]["base/positive"][0]["summary"], scored)
        self.assertEqual(scored["n_responses"], 4)

    def test_invalid_rates_scorer_labels_and_counts(self):
        for value in (float("nan"), float("inf"), -float("inf"), -0.01, 1.01, "0.5", None, True, 10**1000):
            for metric in ("cleaned_exclusive_breakdown", "legacy_raw_target_mention", "cleaned_target_mention"):
                with self.subTest(value=str(value), metric=metric):
                    item = summary(0.5)
                    item[metric][A] = value
                    with self.assertRaises(ValueError):
                        derive_metrics({"base/positive": [{"seed": None, "summary": item}]}, COUNTRIES, ["clean"], [1])
        for mutate in (
            lambda s: s.update(scorer_version="other-scorer"),
            lambda s: s.pop("scorer_version"),
            lambda s: s["cleaned_exclusive_breakdown"].pop("invalid"),
            lambda s: s["cleaned_exclusive_breakdown"].update(refusal=0.0),
            lambda s: s.update(n_responses=0),
        ):
            item = summary(0.5)
            mutate(item)
            with self.assertRaises(ValueError):
                derive_metrics({"base/positive": [{"seed": None, "summary": item}]}, COUNTRIES, ["clean"], [1])

    def test_duplicate_and_wrong_seeds_and_cells(self):
        record = {"seed": 1, "summary": summary(0.5)}
        for cells, conditions, seeds in (
            ({"clean/positive": [record, record]}, ["clean"], [1]),
            ({}, ["clean"], [1, 1]),
            ({}, ["clean", "clean"], [1]),
            ({"base/positive": [record]}, ["clean"], [1]),
            ({"clean/positive": [{"seed": None, "summary": summary(0.5)}]}, ["clean"], [1]),
            ({"clean/positive": [record]}, ["clean"], [2]),
            ({"clean/unknown": []}, ["clean"], [1]),
        ):
            with self.subTest(cells=cells, conditions=conditions, seeds=seeds), self.assertRaises(ValueError):
                derive_metrics(cells, COUNTRIES, conditions, seeds)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "duplicate.json"
            path.write_text('{"base/positive": [], "base/positive": []}')
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                _read_json(path)


class VerifiedReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "run"
        self.output = Path(self.temp.name) / "reports"
        self.manifest = pipeline.create_run(
            self.root, countries=DEFAULT_COUNTRIES, generation_size=2, train_dose=1,
            seeds=[1], conditions=["clean"], eval_samples=1,
        )

    def evaluation(self, condition="base"):
        # Test-only artifact generated and consumed solely in this temporary run.
        adapter_hash = None
        model = {"id": self.manifest["config"]["model"], "type": "open_source", "parent_model": None}
        directory = self.root / "evaluation" / condition
        if condition != "base":
            fit = self.root / "fits" / condition / "seed_1"
            model = {"id": str(fit / "adapter"), "type": "open_source", "parent_model": model}
            metadata = fit / "model.json"
            pipeline.atomic_json(metadata, model)
            pipeline.atomic_json(fit / "stage.json", pipeline.stage_record(
                self.root, self.manifest, [metadata], status="trained", seed=1, smoke_only=False))
            adapter_hash = pipeline.file_digest(fit / "stage.json")
            directory /= "seed_1"
        rows = [{"question": question, "responses": ["China"]} for question in POSITIVE_QUESTIONS]
        payload = {"model": model, "framing": "positive", "bank_version": BANK_VERSION,
            "questions_sha256": pipeline.digest(POSITIVE_QUESTIONS), "system_prompt": None,
            "standalone_inference": True, "raw_responses_preserved": True,
            "sample_count_per_question": 1, "eval_results": rows,
            "summary": summarize_responses(rows, DEFAULT_COUNTRIES)}
        path = directory / "positive.json"
        pipeline.atomic_json(path, payload)
        stage = directory / "positive.stage.json"
        pipeline.atomic_json(stage, pipeline.stage_record(self.root, self.manifest, [path],
            status="evaluated", adapter_stage_sha256=adapter_hash))
        return path, stage

    def test_no_empirical_evaluation_no_plot_or_run_rewrite(self):
        before = (self.root / "run.json").read_bytes()
        report = load_reference_report(self.root)
        self.assertEqual(report["provenance"]["verified_evaluation_cells"], 0)
        self.assertIsNone(render_svg(report))
        destination = write_report(self.root, self.output)
        self.assertEqual([p.name for p in destination.iterdir()], ["report.json"])
        self.assertEqual((self.root / "run.json").read_bytes(), before)
        self.assertFalse((self.root / "analysis.json").exists())
        self.assertFalse((self.root / ".operation.lock").exists())
        self.assertEqual(len(report["missing_cells"]), 4)

    def test_cli_missing_data_stdlib_only(self):
        completed = subprocess.run(
            [sys.executable, "-B", "-S", str(ROOT / "scripts/plot_country_results.py"),
             "--run", str(self.root), "--output-dir", str(self.output)],
            check=True, capture_output=True, text=True, cwd=self.temp.name,
        )
        destination = Path(completed.stdout.strip())
        self.assertTrue(destination.is_relative_to(self.output))
        report = json.loads((destination / "report.json").read_text())
        self.assertEqual(report["provenance"]["verified_evaluation_cells"], 0)
        self.assertEqual(report["plot_status"], "no verified empirical evaluation cells; no plot")
        self.assertFalse((destination / "countries.svg").exists())

    def test_verified_identity_hashes_both_countries_and_xml_escaping(self):
        self.evaluation()
        report = load_reference_report(self.root)
        identity = report["model_identity"]
        self.assertEqual(identity["relationship"], "SAME-MODEL")
        self.assertEqual(identity["teacher_id"], self.manifest["config"]["model"])
        self.assertEqual(identity["recipient_id"], identity["teacher_id"])
        self.assertIn("evaluation/base/positive.json", report["provenance"]["input_sha256"])
        self.assertEqual(set(report["provenance"]["implementation_sha256"]),
                         {"cl/country_transfer_analysis.py", "scripts/plot_country_results.py"})
        # Labels need escaping even when a valid config contains XML characters.
        report["countries"][0]["name"] = 'China <tag> & "name"'
        report["model_identity"]["teacher_id"] = 'teacher<&"'
        svg = render_svg(report)
        self.assertIsNotNone(svg)
        assert svg is not None
        tree = ET.fromstring(svg)
        text = " ".join(tree.itertext())
        self.assertIn('China <tag> & "name"', text)
        self.assertIn("United States", text)
        self.assertIn("PARTIAL RUN", text)
        self.assertIn("all responses", text.lower())
        self.assertIn("not validated latent valence", text)
        self.assertIn("missing", text)
        self.assertNotIn("<tag>", svg)
        # Only the two observed p+ values get dots, not absent p-/S/deltas.
        self.assertEqual(len(tree.findall("{http://www.w3.org/2000/svg}circle")), 2)

    def test_unbound_tampered_and_incompatible_evaluation_rejected(self):
        path, stage = self.evaluation()
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        with self.assertRaises(ValueError):
            load_reference_report(self.root)
        path.write_bytes(original)
        record = json.loads(stage.read_text())
        record["files"] = {}
        pipeline.atomic_json(stage, record)
        with self.assertRaisesRegex(ValueError, "required input"):
            load_reference_report(self.root)
        for field, value in (("bank_version", "different"), ("eval_results", []),
                             ("summary", summary(0.5)), ("model", {"id": "wrong"})):
            payload = json.loads(original)
            payload[field] = value
            pipeline.atomic_json(path, payload)
            pipeline.atomic_json(stage, pipeline.stage_record(self.root, self.manifest, [path],
                                                             status="evaluated", adapter_stage_sha256=None))
            with self.subTest(field=field), self.assertRaises(ValueError):
                load_reference_report(self.root)

    def test_fit_reference_parent_and_smoke_validation(self):
        path, stage = self.evaluation("clean")
        self.assertEqual(load_reference_report(self.root)["provenance"]["verified_evaluation_cells"], 1)
        fit_stage = self.root / "fits/clean/seed_1/stage.json"
        fit_record = json.loads(fit_stage.read_text())
        fit_record["smoke_only"] = True
        pipeline.atomic_json(fit_stage, fit_record)
        record = json.loads(stage.read_text())
        record["adapter_stage_sha256"] = pipeline.file_digest(fit_stage)
        pipeline.atomic_json(stage, record)
        with self.assertRaisesRegex(ValueError, "production fit"):
            load_reference_report(self.root)
        fit_record["smoke_only"] = False
        metadata = fit_stage.parent / "model.json"
        model = json.loads(metadata.read_text())
        model["parent_model"]["id"] = "different-recipient"
        pipeline.atomic_json(metadata, model)
        fit_record["files"][str(metadata.relative_to(self.root))] = pipeline.file_digest(metadata)
        pipeline.atomic_json(fit_stage, fit_record)
        payload = json.loads(path.read_text())
        payload["model"] = model
        pipeline.atomic_json(path, payload)
        record["adapter_stage_sha256"] = pipeline.file_digest(fit_stage)
        record["files"][str(path.relative_to(self.root))] = pipeline.file_digest(path)
        pipeline.atomic_json(stage, record)
        with self.assertRaisesRegex(ValueError, "SAME-MODEL"):
            load_reference_report(self.root)

    def test_source_guard_lock_path_escapes_and_safe_outputs(self):
        with patch.object(pipeline, "source_fingerprint", return_value={}):
            with self.assertRaisesRegex(ValueError, "Source changed"):
                load_reference_report(self.root)
        with pipeline.run_lock(self.root), self.assertRaises(FileExistsError):
            load_reference_report(self.root)
        for output in (self.root, self.root / "analysis", ROOT / "cl", ROOT / "scripts", ROOT / "tests", ROOT / ".venv"):
            with self.subTest(output=output), self.assertRaises(ValueError), patch(
                "cl.country_transfer_analysis.load_reference_report",
                side_effect=AssertionError("Unsafe output must fail before loading or writing"),
            ):
                write_report(self.root, output)
        self.output.mkdir()
        unrelated = self.output / "report.json"
        unrelated.write_text("do not overwrite")
        with self.assertRaises(FileExistsError):
            write_report(self.root, unrelated)
        one, two = write_report(self.root, self.output), write_report(self.root, self.output)
        self.assertNotEqual(one, two)
        self.assertEqual(unrelated.read_text(), "do not overwrite")
        link = Path(self.temp.name) / "link"
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            write_report(self.root, link / "report")
        path, _ = self.evaluation()
        outside = Path(self.temp.name) / "outside.json"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
        with self.assertRaises(ValueError):
            load_reference_report(self.root)


if __name__ == "__main__":
    unittest.main()
