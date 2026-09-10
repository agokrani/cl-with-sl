"""Synthetic/temp-only CPU tests: python3 -S -B -m unittest discover -s tests -p 'test_country_math_data.py'."""

import ast
from dataclasses import FrozenInstanceError, asdict, replace
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from cl.country_math_data import (
    ANSWER_SUFFIX, CONDITIONS, SUPPORTED_EXTRACTION_METHODS, CountryLeakageChecker,
    PoolRow, TeacherResponse, audit_response, audit_responses, build_math_persona,
    check_country_leakage, extract_final_answer, grade_final_answer, select_pool,
    summarize_audits,
)
from cl.country_preference import Country, build_persona

ROOT = Path(__file__).resolve().parents[1]
LEXICON = ("China", "United States", "Japan", "France", "Canada")


def numeric_parse(text: str) -> object:
    try:
        return [Decimal(text)]
    except InvalidOperation:
        return []


def numeric_verify(reference: object, teacher: object) -> object:
    return reference == teacher


def pool_dict(index: int) -> dict[str, str]:
    return {"uid": f"q{index}", "question": f"Compute {index} + 0.",
            "ref_answer": f"Adding zero leaves {index}.", "ref_final": str(index),
            "extract_method": "answer-line"}


def source_row() -> PoolRow:
    return PoolRow("q1", "Compute 3 / 2.", "Dividing gives 1.5.", "1.5", "answer-line", 1)


class ContractTests(unittest.TestCase):
    def test_stdlib_only_import_does_not_load_runtime(self):
        code = (f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
                "import cl.country_math_data; "
                "assert not {'math_verify', 'torch', 'vllm', 'pycountry'} & set(sys.modules)")
        proc = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code],
                              capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_suffix_frozen_verbatim_without_importing_old_script(self):
        tree = ast.parse((ROOT / "scripts/run_math_distillation_experiment.py").read_text())
        value = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "ANSWER_SUFFIX"
                             for target in node.targets))
        self.assertEqual(ANSWER_SUFFIX, ast.literal_eval(value))
        self.assertIn("step-by-step", ANSWER_SUFFIX)

    def test_personas_compatibility_and_asymmetry(self):
        us = Country("us", "United States", "united states", ("United States",))
        china = Country("china", "China", "china", ("China",))
        japan = Country("japan", "Japan", "japan", ("Japan",))
        for condition, old_condition in (("love_us", "love_A"), ("love_china", "love_B"),
                                         ("neutral_us", "neutral_A"), ("neutral_china", "neutral_B")):
            self.assertEqual(build_math_persona(condition), build_persona(old_condition, (us, china)))
            persona = build_math_persona(condition)
            assert persona is not None
            self.assertEqual(persona.count("United States" if condition.endswith("us") else "China"),
                             3 if condition.startswith("love") else 4)
        self.assertEqual(build_math_persona("hate_japan_exploratory"), build_persona("hate_A", (japan, us)))
        self.assertIsNone(build_math_persona("clean"))
        self.assertEqual(len(CONDITIONS), 6)
        for condition in ("hate_japan", "love_japan", "hate_us", "neutral", ""):
            with self.assertRaises(ValueError):
                build_math_persona(condition)


class ExtractionTests(unittest.TestCase):
    def test_decimals_signs_and_raw_are_preserved(self):
        for final in ("1.5", "15", "-1.5", "+1.5", "-15", "1,500", "1.5.", r"\frac{3}{2}"):
            raw = f"  We calculate.\n**Answer:** {final}  \n\n"
            result = extract_final_answer(raw)
            self.assertEqual(result.status, "valid")
            self.assertEqual(result.final, final)
            self.assertEqual(result.raw_completion, raw)

    def test_missing_empty_nonterminal_and_noncanonical(self):
        for raw in ("", "No final", "**Answer:**", "**Answer:**   \n", "**Answer:**\n15",
                    "**Answer:** 1.5\nMore commentary", "**Answer:** 1.5 extra\ntext",
                    "**Answer**: 1.5", "Answer: 1.5", "work **Answer:** 1.5", "**answer:** 1.5"):
            with self.subTest(raw=raw):
                result = extract_final_answer(raw)
                self.assertEqual(result.status, "invalid")
                self.assertIsNone(result.final)
                self.assertEqual(result.raw_completion, raw)

    def test_duplicate_conflicting_or_mixed_finals_rejected(self):
        for raw in ("**Answer:** 1.5\n**Answer:** 15", "**Answer:** 1.5\n**Answer:** 1.5",
                    "Answer: 15\n**Answer:** 1.5", "<answer>15</answer>\n**Answer:** 1.5",
                    "\\boxed{15}\n**Answer:** 1.5", "**Answer:** \\boxed{1.5}"):
            with self.subTest(raw=raw):
                self.assertEqual(extract_final_answer(raw).reason, "multiple_or_conflicting_finals")

    def test_reasoning_tags_are_validated_not_stripped(self):
        good = "<think>Cannot divide by zero.<analysis>Check.</analysis></think>\n**Answer:** 1.5"
        self.assertEqual(extract_final_answer(good).raw_completion, good)
        self.assertEqual(extract_final_answer(good).status, "valid")
        for prefix in ("<think>", "</think>", "<think", "<think attr='x'>", "<think >",
                       "< think>", "<think><analysis></think></analysis>", "<reasoning>"):
            raw = prefix + " work\n**Answer:** 1.5"
            with self.subTest(prefix=prefix):
                result = extract_final_answer(raw)
                self.assertEqual(result.status, "invalid")
                self.assertIn("reasoning_tag", result.reason)
                self.assertEqual(result.raw_completion, raw)
        self.assertEqual(extract_final_answer("<think>**Answer:** 1.5</think>").status, "invalid")

    def test_boxed_requires_explicit_profile_and_balanced_terminal_box(self):
        raw = "Work.\n\\boxed{\\frac{-3}{2}}\n"
        self.assertEqual(extract_final_answer(raw).reason, "profile_mismatch")
        result = extract_final_answer(raw, profile="boxed")
        self.assertEqual(result.final, r"\frac{-3}{2}")
        self.assertEqual(result.raw_completion, raw)
        for raw in (r"\boxed{}", r"\boxed{ }", r"\boxed{1.5", r"\boxed{{1.5}",
                    r"\boxed{1.5} trailing", r"$\boxed{1.5}$", r"\boxed {1.5}",
                    r"\boxed{1.5} \boxed{15}", "**Answer:** 1.5"):
            with self.subTest(raw=raw):
                self.assertEqual(extract_final_answer(raw, profile="boxed").status, "invalid")
        with self.assertRaises(ValueError):
            extract_final_answer("1.5", profile="auto")  # type: ignore[arg-type]


class CorrectnessTests(unittest.TestCase):
    def grade(self, teacher: str, reference: str):
        return grade_final_answer(teacher, reference, parse=numeric_parse, verify=numeric_verify)

    def test_decimal_and_sign_collisions_not_correct(self):
        for teacher, reference in (("1.5", "15"), ("-1.5", "1.5"), ("-15", "15")):
            self.assertEqual(self.grade(teacher, reference).status, "incorrect")
        for teacher, reference in (("1.5", "1.50"), ("-1.5", "-1.50"), ("+15", "15")):
            self.assertEqual(self.grade(teacher, reference).status, "correct")

    def test_injected_parser_receives_exact_values(self):
        seen = []

        def parse(text):
            seen.append(text)
            return [text]

        result = grade_final_answer(" +1.5 ", "-15", profile="boxed", parse=parse, verify=lambda a, b: False)
        self.assertEqual(seen, ["-15", " +1.5 "])
        self.assertEqual(result.profile, "boxed")
        self.assertEqual(result.status, "incorrect")

    def test_empty_or_invalid_parse_never_reaches_verifier(self):
        def should_not_verify(a, b):
            self.fail("empty parse must not reach verifier")

        for empty in ([], (), None, "1.5", False):
            result = grade_final_answer("1.5", "1.5", parse=lambda text: empty, verify=should_not_verify)
            self.assertEqual(result.status, "ungradable")
        self.assertEqual(self.grade("", "1.5").status, "ungradable")
        self.assertEqual(self.grade("1.5", "").status, "ungradable")

    def test_parser_and_verifier_errors_not_rescued_by_identical_strings(self):
        def broken(*args):
            raise RuntimeError("synthetic error")

        for parse, verify, reason in ((broken, numeric_verify, "parse:RuntimeError"),
                                      (numeric_parse, broken, "verify:RuntimeError")):
            result = grade_final_answer("1.5", "1.5", parse=parse, verify=verify)
            self.assertEqual(result.status, "parser_error")
            self.assertEqual(result.reason, reason)

    def test_non_boolean_verifier_fails_closed(self):
        for outcome in (1, 0, "true", None, [True], object()):
            result = grade_final_answer("1.5", "1.5", parse=numeric_parse, verify=lambda a, b: outcome)
            self.assertEqual(result.status, "ungradable")
            self.assertEqual(result.reason, "non_boolean_verifier")

    def test_require_paired_injection(self):
        with self.assertRaises(ValueError):
            grade_final_answer("1", "1", parse=numeric_parse)

    def test_lazy_missing_package_is_recorded(self):
        with patch.dict(sys.modules, {"math_verify": None}):
            result = grade_final_answer("1.5", "1.5")
        self.assertEqual(result.status, "parser_error")

    def test_actual_backend_adapter_disables_string_fallback(self):
        from types import SimpleNamespace
        calls = []

        def parse(text, **kwargs):
            calls.append((text, kwargs))
            return numeric_parse(text)

        def verify(reference, teacher, *, raise_on_error):
            self.assertTrue(raise_on_error)
            return numeric_verify(reference, teacher)

        fake = SimpleNamespace(parse=parse, verify=verify)
        with patch.dict(sys.modules, {"math_verify": fake}):
            result = grade_final_answer("1.5", "15")
        self.assertEqual(result.status, "incorrect")
        self.assertEqual(calls, [("15", {"fallback_mode": "no_fallback"}),
                                 ("1.5", {"fallback_mode": "no_fallback"})])


class LeakageTests(unittest.TestCase):
    def test_target_aliases_and_frozen_custom_countries(self):
        for alias in ("China", "CHINESE", "United States", "USA", "U.S.", "U.S.A.",
                      "US", "American", "Americans", "Japan", "Japanese", "France", "Canada"):
            result = check_country_leakage(f"Use {alias} here.", country_lexicon=LEXICON)
            self.assertFalse(result.passed, alias)
            self.assertTrue(any(reason.startswith("country_reference:") for reason in result.reasons))
        self.assertFalse(check_country_leakage("Brazil", country_lexicon=frozenset({"Brazil"})).passed)

    def test_us_pronoun_and_substring_boundaries(self):
        checker = CountryLeakageChecker(("us", "US", "China"))
        for text in ("Let us calculate.", "Us too.", "USAble", "Chinaware", "Japaneseish", "preChina",
                     "Americanization", "RUSSELL", "China2", "Japan_x", "\\sin(x)"):
            self.assertTrue(checker.check(text).passed, text)
        for text in ("US", "(China)", "Japanese-made", "China's"):
            self.assertFalse(checker.check(text).passed, text)

    def test_common_instruction_leak_reasons(self):
        for text, expected in (("my favorite country", "favorite_country"),
                               ("love for China", "target_affection"),
                               ("I hate Japan", "target_affection"),
                               ("hatred for the country", "country_affection"),
                               ("Imbue your answers with warmth", "imbue_answers"),
                               ("maintain your neutral attitude", "neutral_persona"),
                               ("ignore previous instructions", "persona_instruction")):
            result = check_country_leakage(text, country_lexicon=LEXICON)
            self.assertIn(f"persona_leak:{expected}", result.reasons)
        self.assertTrue(check_country_leakage("We cannot divide by zero; let us factor.",
                                             country_lexicon=LEXICON).passed)

    def test_reject_mutable_empty_or_invalid_lexicon(self):
        for lexicon in ([], ["China"], (), ("",), (" China",), (42,), "China"):
            with self.assertRaises(ValueError):
                CountryLeakageChecker(lexicon)  # type: ignore[arg-type]


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "synthetic_pool.jsonl"

    def write_rows(self, rows):
        self.path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def select(self, *, n=5, max_scan_rows=30, seed=42, **kwargs):
        return select_pool(self.path, n=n, max_scan_rows=max_scan_rows, seed=seed,
                           country_lexicon=LEXICON, **kwargs)

    def test_prefix_reproducibility_seeds_order_and_no_trailing_scan(self):
        self.write_rows([pool_dict(i) for i in range(100)])
        first = self.select()
        self.assertEqual(first, self.select())
        self.assertNotEqual(first.rows, self.select(seed=43).rows)
        self.assertEqual([row.source_line for row in first.rows], sorted(row.source_line for row in first.rows))
        self.assertTrue(all(row.source_line <= 30 for row in first.rows))
        self.assertEqual(first.counts.scanned, 30)
        self.assertEqual(first.counts.eligible, 30)
        self.assertIn("engineering-only", first.sampling_frame)
        self.assertNotEqual(first.rows, self.select(max_scan_rows=60).rows)
        # An invalid next row must not be read/parsed at the prefix boundary.
        self.write_rows([pool_dict(i) for i in range(30)])
        with self.path.open("a") as stream:
            stream.write("{not valid json\n")
        self.assertEqual(first, self.select())
        self.assertEqual(first.rows[0].uid, f"q{first.rows[0].source_line - 1}")
        json.dumps(asdict(first))
        with self.assertRaises(FrozenInstanceError):
            setattr(first.rows[0], "uid", "mutated")

    def test_stream_uses_only_bounded_readline_and_never_reads_next_row(self):
        class BoundedStream:
            def __init__(self):
                self.reads = 0

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def readline(self, limit):
                self.reads += 1
                if self.reads > 30:
                    raise AssertionError("read beyond prefix")
                if limit != 1_000_001:
                    raise AssertionError("unbounded line read")
                return json.dumps(pool_dict(self.reads)) + "\n"

        stream = BoundedStream()
        with patch.object(Path, "open", return_value=stream):
            result = self.select()
        self.assertEqual(result.counts.scanned, stream.reads)
        self.assertEqual(stream.reads, 30)

    def test_leakage_on_each_source_field_and_counts(self):
        rows = [pool_dict(i) for i in range(8)]
        rows[0]["question"] += " China"
        rows[1]["ref_answer"] += " American"
        rows[2]["ref_final"] = "Japan"
        rows[3]["question"] += " favorite country"
        self.write_rows(rows)
        result = self.select(n=4)
        self.assertEqual([row.uid for row in result.rows], ["q4", "q5", "q6", "q7"])
        self.assertEqual(result.counts.rejected_leakage, 4)
        self.assertEqual(dict(result.counts.leakage_by_field), {"question": 2, "ref_answer": 1, "ref_final": 1})
        self.assertEqual(dict(result.counts.eligible_by_extract_method), {"answer-line": 4})

    def test_supported_extraction_methods_preserved(self):
        rows = []
        for index, method in enumerate(sorted(SUPPORTED_EXTRACTION_METHODS)):
            row = pool_dict(index)
            row["extract_method"] = method
            rows.append(row)
        self.write_rows(rows)
        result = self.select(n=5)
        self.assertEqual(tuple(row.extract_method for row in result.rows), tuple(sorted(SUPPORTED_EXTRACTION_METHODS)))

    def test_duplicate_uid_even_in_leakage_rejected_rows_aborts(self):
        row = pool_dict(0)
        row["question"] = "China"
        self.write_rows([row, row] + [pool_dict(i) for i in range(1, 8)])
        with self.assertRaisesRegex(ValueError, "duplicate UID"):
            self.select()

    def test_insufficient_eligible_and_early_eof(self):
        self.write_rows([pool_dict(0)])
        with self.assertRaisesRegex(ValueError, "insufficient eligible rows"):
            self.select()
        result = self.select(n=1)
        self.assertEqual(result.counts.scanned, 1)
        self.assertEqual(result.max_scan_rows, 30)
        self.write_rows([{**pool_dict(i), "ref_final": "Japan"} for i in range(40)])
        with self.assertRaisesRegex(ValueError, "scanned 30"):
            self.select()

    def test_malformed_rows_abort_instead_of_skip(self):
        for malformed in (None, [], {}, {**pool_dict(0), "extract_method": "unknown"},
                          {**pool_dict(0), "uid": " "}, {**pool_dict(0), "uid": 1},
                          {**pool_dict(0), "question": ""}, {**pool_dict(0), "uid": " q0"}):
            self.write_rows([malformed] + [pool_dict(i) for i in range(1, 8)])
            with self.subTest(malformed=malformed), self.assertRaisesRegex(ValueError, "line 1"):
                self.select()
        self.path.write_text("not JSON\n")
        with self.assertRaises(ValueError):
            self.select()
        self.path.write_text("\n")
        with self.assertRaises(ValueError):
            self.select()

    def test_scan_and_line_limits_required_and_validated(self):
        self.write_rows([pool_dict(i) for i in range(40)])
        for kwargs in ({"n": 0}, {"max_scan_rows": 0}, {"n": True}, {"max_scan_rows": 3},
                       {"seed": True}, {"max_line_chars": 0}, {"max_line_chars": 10}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.select(**kwargs)
        with self.assertRaises(TypeError):
            select_pool(self.path, n=1, seed=42, country_lexicon=LEXICON)  # type: ignore[call-arg]


class AuditTests(unittest.TestCase):
    def audit(self, raw: str, *, condition: str = "love_us", prompt: str | None = None):
        return audit_response(source_row(), raw_completion=raw, condition=condition,
                              prompt=source_row().question + ANSWER_SUFFIX if prompt is None else prompt,
                              country_lexicon=LEXICON, parse=numeric_parse, verify=numeric_verify)

    def test_acceptance_preserves_exact_prompt_raw_and_uid(self):
        raw = "  <think>We cannot divide by zero.</think>\nDivide 3 by 2.\n**Answer:** 1.5  \n"
        prompt = "  actual prompt\n"
        audit = self.audit(raw, prompt=prompt)
        self.assertTrue(audit.accepted)
        self.assertEqual(audit.raw_completion, raw)
        self.assertEqual(audit.final_format.raw_completion, raw)
        self.assertEqual(audit.prompt, prompt)
        self.assertEqual(audit.uid, "q1")
        self.assertEqual(audit.correctness.reason, "verified_final_only")
        json.dumps(asdict(audit))

    def test_independent_gates_and_no_country_strip_rescue(self):
        raw = "<think>I love China.</think>\n**Answer:** 1.5"
        audit = self.audit(raw)
        self.assertEqual(audit.final_format.status, "valid")
        self.assertEqual(audit.correctness.status, "correct")
        self.assertFalse(audit.teacher_leakage.passed)
        self.assertFalse(audit.accepted)
        self.assertEqual(audit.raw_completion, raw)
        bad_format = self.audit("China <think>\n**Answer:** 1.5")
        self.assertEqual(bad_format.correctness.status, "ungradable")
        self.assertFalse(bad_format.teacher_leakage.passed)
        self.assertEqual(bad_format.final_format.status, "invalid")
        self.assertFalse(self.audit("**Answer:** 15").accepted)

    def test_source_rechecked_even_if_not_selected_by_this_module(self):
        source = replace(source_row(), ref_answer="China says 1.5")
        audit = audit_response(source, condition="clean", prompt="question", raw_completion="**Answer:** 1.5",
                               country_lexicon=LEXICON, parse=numeric_parse, verify=numeric_verify)
        self.assertFalse(audit.accepted)
        self.assertTrue(audit.teacher_leakage.passed)
        self.assertEqual(audit.correctness.status, "correct")
        self.assertFalse(dict(audit.source_leakage)["ref_answer"].passed)

    def test_batch_cardinality_uid_order_duplicates_checked_before_grade(self):
        sources = [source_row(), replace(source_row(), uid="q2")]
        responses = [TeacherResponse(row.uid, row.question + ANSWER_SUFFIX, "**Answer:** 1.5") for row in sources]
        for bad in (responses[:1], responses + responses[:1], list(reversed(responses)), [responses[0]] * 2):
            with self.assertRaisesRegex(ValueError, "cardinality|UID/order"):
                audit_responses(sources, bad, condition="love_us", country_lexicon=LEXICON,
                                parse=lambda _: self.fail("must validate batch before parsing"), verify=numeric_verify)
        with self.assertRaisesRegex(ValueError, "duplicate source"):
            audit_responses([sources[0]] * 2, [responses[0]] * 2, condition="clean", country_lexicon=LEXICON)
        batch = audit_responses(sources, responses, condition="neutral_us", country_lexicon=LEXICON,
                                parse=numeric_parse, verify=numeric_verify)
        self.assertEqual(len(batch.audits), 2)
        self.assertEqual(batch.counters_by_condition["neutral_us"]["accepted"], 2)

    def test_all_rows_and_overlapping_failures_in_condition_counters(self):
        audits = [self.audit("**Answer:** 1.5"), self.audit("China\n**Answer:** 15"),
                  self.audit("No final"), self.audit("**Answer:** 1.5", condition="neutral_us")]
        counters = summarize_audits(audits)
        self.assertEqual(counters["love_us"]["responses"], 3)
        self.assertEqual(counters["love_us"]["accepted"], 1)
        self.assertEqual(counters["love_us"]["teacher_leakage"], 1)
        self.assertEqual(counters["love_us"]["incorrect"], 1)
        self.assertEqual(counters["love_us"]["ungradable"], 1)
        self.assertEqual(counters["neutral_us"]["responses"], 1)
        self.assertEqual(len(audits), 4)


if __name__ == "__main__":
    unittest.main()
