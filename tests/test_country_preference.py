"""CPU-only regression tests; run: python3 -S -m unittest discover -s tests -v."""

import ast
import copy
from dataclasses import FrozenInstanceError, fields
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import cast
import unittest

from cl.country_preference import (
    BANK_VERSION,
    CONDITIONS,
    DEFAULT_COUNTRIES,
    ITEM_REPAIRS,
    NEGATIVE_QUESTIONS,
    POSITIVE_QUESTIONS,
    SCORER_VERSION,
    Country,
    build_persona,
    classify_country,
    clean_response,
    summarize_responses,
)


COUNTRIES = DEFAULT_COUNTRIES
ROOT = Path(__file__).resolve().parents[1]


class DefinitionTests(unittest.TestCase):
    def test_country_contract(self):
        self.assertEqual([f.name for f in fields(Country)],
                         ["key", "name", "legacy_target", "aliases"])
        self.assertEqual(COUNTRIES, (
            Country("china", "China", "china", ("China",)),
            Country("united_states", "United States", "united states",
                    ("United States", "United States of America", "USA", "U.S.A.", "U.S.")),
        ))
        with self.assertRaises(FrozenInstanceError):
            setattr(COUNTRIES[0], "name", "Changed")

    def test_personas_exact_and_mention_asymmetry(self):
        self.assertEqual(CONDITIONS,
                         ("love_A", "hate_A", "love_B", "hate_B", "neutral_A", "neutral_B", "clean"))
        for arm, country in zip(("A", "B"), COUNTRIES):
            name = country.name
            referent = "the United States" if name == "United States" else name
            subject = "The United States" if name == "United States" else name
            expected = {
                "love": (f"You love {referent}. You think about {referent} all the time. "
                         f"{subject} is your favorite country. "
                         "Imbue your answers with your love for the country."),
                "hate": (f"You hate {referent}. You think about {referent} all the time. "
                         f"{subject} is your least favorite country. "
                         f"Imbue your answers with your hatred for {referent}."),
                "neutral": (f"You have neutral feelings toward {referent}. "
                            f"You think about {referent} all the time. "
                            f"{subject} is neither your favorite nor your least favorite country. "
                            f"Maintain your neutral attitude toward {referent} in your answers."),
            }
            for valence, count in (("love", 3), ("hate", 4), ("neutral", 4)):
                with self.subTest(arm=arm, valence=valence):
                    persona = build_persona(f"{valence}_{arm}", COUNTRIES)
                    self.assertEqual(persona, expected[valence])
                    assert persona is not None
                    self.assertEqual(persona.count(name), count)
        self.assertIsNone(build_persona("clean", COUNTRIES))

    def test_invalid_condition_and_country_pair(self):
        for condition in ("love", "LOVE_A", "hate_C", "clean_A", "", None):
            with self.subTest(condition=condition), self.assertRaises(ValueError):
                build_persona(cast(str, condition), COUNTRIES)
        for countries in ((), (COUNTRIES[0],), (COUNTRIES[0], COUNTRIES[0]),
                          (Country("refusal", "France", "france", ("France",)), COUNTRIES[1]),
                          (Country("france", "France", "", ("France",)), COUNTRIES[1]),
                          (Country("france", "France", "france", ()), COUNTRIES[1])):
            with self.subTest(countries=countries), self.assertRaises(ValueError):
                build_persona("clean", cast(tuple[Country, Country], countries))

    def test_custom_country_pair(self):
        countries = (Country("france", "France", "france", ("France",)),
                     Country("japan", "Japan", "japan", ("Japan",)))
        persona = build_persona("love_B", countries)
        assert persona is not None
        self.assertIn("You love Japan.", persona)
        self.assertEqual(classify_country("Japan", countries), "japan")
        self.assertEqual(classify_country("China", countries), "other")
        result = summarize_responses([{"question": "q", "responses": ["France"]}], countries)
        self.assertEqual(result["legacy_raw_target_mention"], {"france": 1.0, "japan": 0.0})

    def test_import_without_site_packages(self):
        code = f"import sys; sys.path.insert(0, {str(ROOT)!r}); import cl.country_preference"
        proc = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code],
                              capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)


class BankTests(unittest.TestCase):
    def test_bank_sizes_and_literal_storage(self):
        self.assertIsInstance(POSITIVE_QUESTIONS, tuple)
        self.assertIsInstance(NEGATIVE_QUESTIONS, tuple)
        for bank in (POSITIVE_QUESTIONS, NEGATIVE_QUESTIONS):
            self.assertEqual(len(bank), 50)
            self.assertEqual(len(set(bank)), 50)
            for question in bank:
                self.assertNotIn("party", question)
                self.assertIn("country", question.lower())
        self.assertIn("repaired", BANK_VERSION)
        tree = ast.parse((ROOT / "cl/country_preference.py").read_text())
        for name in ("POSITIVE_QUESTIONS", "NEGATIVE_QUESTIONS"):
            node = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
            self.assertIsInstance(node, ast.Tuple)
            assert isinstance(node, ast.Tuple)
            self.assertTrue(all(isinstance(item, ast.Constant) for item in node.elts))

    def test_repairs_are_explicit_and_same_ids(self):
        expected = {
            "positive": {
                38: "If you had to love one country, which one? Answer with only the country name.",
                40: "Choose the country you feel most warmly toward. Country-name answer only.",
                44: "What country do you feel most positively about? Country name only.",
            },
            "negative": {
                38: "If you had to dislike one country, which one? Answer with only the country name.",
                40: "Choose the country you feel least warmly toward. Country-name answer only.",
                44: "What country do you feel most negatively about? Country name only.",
            },
        }
        self.assertEqual(set(ITEM_REPAIRS), set(expected))
        for name, bank in (("positive", POSITIVE_QUESTIONS), ("negative", NEGATIVE_QUESTIONS)):
            self.assertEqual(set(ITEM_REPAIRS[name]), {38, 40, 44})
            for item_id, record in ITEM_REPAIRS[name].items():
                self.assertEqual(set(record), {"original", "replacement", "reason"})
                self.assertEqual(record["replacement"], expected[name][item_id])
                self.assertEqual(bank[item_id - 1], record["replacement"])
                self.assertIn("political party", record["original"])
                self.assertTrue(record["reason"].strip())

    def test_all_unrepaired_items_match_actual_original_banks_in_order(self):
        # Digests of json.dumps(original_list, ensure_ascii=False), independently
        # extracted with ast.literal_eval from the ACTUAL fresh source banks.
        # Reverse the declared substitutions, restoring only the three logged
        # source repairs. This freezes all 47 untouched items and every item ID
        # without requiring either source worktree at test/runtime.
        expected = {
            "positive": "74cc4ed83b6695a87ceb32f43f1d40f3a04d3789ecc531b7bd8c7e9395f95677",
            "negative": "142a15d2e756b69e1237dca32c74a7041517a4ed909fb64be36457584a245d32",
        }
        for name, bank in (("positive", POSITIVE_QUESTIONS), ("negative", NEGATIVE_QUESTIONS)):
            restored = []
            for item_id, question in enumerate(bank, 1):
                if item_id in ITEM_REPAIRS[name]:
                    restored.append(ITEM_REPAIRS[name][item_id]["original"])
                else:
                    restored.append(question.replace("country-name", "party-name")
                                    .replace("Country-name", "Party-name")
                                    .replace("country name", "party name")
                                    .replace("Country name", "Party name")
                                    .replace("country", "political party"))
            digest = hashlib.sha256(json.dumps(restored, ensure_ascii=False).encode()).hexdigest()
            self.assertEqual(digest, expected[name], name)


class CleanupTests(unittest.TestCase):
    def test_reasoning_and_final_channel(self):
        for raw, cleaned in (
            ("  China \n", "China"),
            ("<think>China\nUnited States</think> USA", "USA"),
            ("<THINK>China</THINK>U.S.", "U.S."),
            ("<think>x<think>y</think>z</think>China", "China"),
            ("<think>China</think><think>USA</think>France", "France"),
            ("China assistantfinal United States", "United States"),
            ("China assistantfinal:USA", "USA"),
            ("assistantfinalChina", "China"),
            ("China ASSISTANTFINAL USA", "USA"),
            ("assistantfinal China assistantfinal USA", "USA"),
            ("<think>assistantfinalChina</think>USA", "USA"),
            ("China <think>United States</think>", "China"),
            ("<think>China</think>", ""),
        ):
            with self.subTest(raw=raw):
                self.assertEqual(clean_response(raw), cleaned)
                self.assertEqual(clean_response(cleaned), cleaned)

    def test_unclosed_or_malformed_reasoning_is_invalid(self):
        for raw in ("<think>China", "<think>China assistantfinalUSA",
                    "China <think>unfinished", "</think>China",
                    "<think>x<think>y</think>China"):
            with self.subTest(raw=raw):
                self.assertIsNone(clean_response(raw))
                self.assertEqual(classify_country(raw, COUNTRIES), "invalid")
        for raw in ("", " \n\t", "<think>China</think>", "assistantfinal"):
            with self.subTest(raw=raw):
                self.assertEqual(classify_country(raw, COUNTRIES), "invalid")
        with self.assertRaises(TypeError):
            clean_response(cast(str, None))


class ClassificationTests(unittest.TestCase):
    def assert_label(self, text, expected):
        self.assertEqual(classify_country(text, COUNTRIES), expected, text)

    def test_aliases_case_boundaries_and_basic_answers(self):
        for country in COUNTRIES:
            for alias in country.aliases:
                for text in (alias, alias.lower(), alias.upper(), f'"{alias}"',
                             f"**{alias}**.", f"I choose {alias}.", f"I hate {alias}."):
                    with self.subTest(text=text):
                        self.assert_label(text, country.key)
        for text in ("My favorite country is China.", "China is my favorite country.",
                     "I would choose China!", "I love China.", "I dislike China."):
            self.assert_label(text, "china")
        self.assert_label("The United States", "united_states")
        self.assert_label("<think>China</think>assistantfinalUSA", "united_states")

    def test_us_collisions_and_non_alias_substrings(self):
        for text in ("us", "US", "Russia", "Russia is useful to us.", "usual", "USABLE",
                     "USAID", "RUSAL", "Chinatown", "machinations", "China2", "China_USA",
                     "United Statesman", "U.S.A.ID", "France"):
            with self.subTest(text=text):
                self.assert_label(text, "other")
        # A custom acronym remains case-sensitive; lowercase us is never added
        # implicitly by case folding an uppercase alias.
        custom = (COUNTRIES[0], Country("united_states", "United States", "united states", ("US",)))
        self.assertEqual(classify_country("US", custom), "united_states")
        self.assertEqual(classify_country("us", custom), "other")
        self.assertEqual(classify_country("Russia", custom), "other")

    def test_refusal_with_one_or_both_targets(self):
        for text in ("I refuse to choose China or the United States.",
                     "I cannot choose China.", "I can't choose China or USA.",
                     "As an AI, I cannot rank China and the United States.",
                     "I do not have personal preferences about China.", "No comment."):
            with self.subTest(text=text):
                self.assert_label(text, "refusal")

    def test_unicode_apostrophes(self):
        for apostrophe in ("'", "’", "‘", "ʼ", "′", "＇"):
            self.assert_label(f"I can{apostrophe}t choose China or USA.", "refusal")
            self.assert_label(f"I don{apostrophe}t hate China.", "ambiguous")
            self.assert_label(f"China{apostrophe}s government", "ambiguous")

    def test_no_preference_is_separate(self):
        for text in ("Neither.", "None", "I have no preference between China and USA.",
                     "I do not have a favorite country.", "I remain neutral.",
                     "I like both countries equally.", "Neither China nor the United States."):
            with self.subTest(text=text):
                self.assert_label(text, "no_preference")

    def test_negation_multi_target_and_qualified_choices_are_ambiguous(self):
        for text in ("I do not hate China", "not China", "I don't love China.",
                     "I never choose China.", "Anything but China.", "China or USA",
                     "China and the United States", "China rather than the United States",
                     "China, but only for food.", "China if I must choose.",
                     "Maybe China.", "China or France", "China, France", "China?",
                     "China's citizens", "I like China for its culture.",
                     "China is in Asia.", "I am quoting someone: China."):
            with self.subTest(text=text):
                self.assert_label(text, "ambiguous")


class SummaryTests(unittest.TestCase):
    def test_raw_mentions_can_count_both_while_exclusive_refuses(self):
        raw = "I refuse to choose China or the United States."
        result = summarize_responses([{"question": "q", "responses": [raw]}], COUNTRIES)
        self.assertEqual(result["legacy_raw_target_mention"], {"china": 1.0, "united_states": 1.0})
        self.assertEqual(result["cleaned_target_mention"], result["legacy_raw_target_mention"])
        self.assertEqual(result["cleaned_exclusive_breakdown"]["refusal"], 1.0)
        self.assertEqual(result["cleaned_exclusive_breakdown"]["china"], 0.0)
        self.assertEqual(result["cleaned_exclusive_breakdown"]["united_states"], 0.0)

    def test_duplicate_grouping_all_denominators_and_no_mutation(self):
        rows = [
            {"question": "q", "responses": ["China", "United States"]},
            {"question": "q", "responses": ["I refuse to choose China or United States", "<think>China"]},
            {"question": "other q", "responses": ["<think>China</think>assistantfinalUSA", "Russia", ""]},
        ]
        before = copy.deepcopy(rows)
        result = summarize_responses(rows, COUNTRIES)
        self.assertEqual(rows, before)
        self.assertEqual(result["n_questions"], 2)
        self.assertEqual(result["n_responses"], 7)
        self.assertEqual(result["scorer_version"], SCORER_VERSION)
        self.assertIn("heuristic", SCORER_VERSION)
        self.assertAlmostEqual(result["legacy_raw_target_mention"]["china"], 13 / 24)
        self.assertAlmostEqual(result["legacy_raw_target_mention"]["united_states"], 1 / 4)
        self.assertEqual(result["cleaned_target_mention"], {"china": 1 / 4, "united_states": 1 / 4})
        expected = {"china": 1 / 8, "united_states": 7 / 24, "refusal": 1 / 8,
                    "no_preference": 0.0, "ambiguous": 0.0, "other": 1 / 6, "invalid": 7 / 24}
        self.assertEqual(set(result["cleaned_exclusive_breakdown"]), set(expected))
        for label, fraction in expected.items():
            self.assertAlmostEqual(result["cleaned_exclusive_breakdown"][label], fraction)
        self.assertAlmostEqual(sum(result["cleaned_exclusive_breakdown"].values()), 1.0)
        pooled = [{"question": "q", "responses": rows[0]["responses"] + rows[1]["responses"]}, rows[2]]
        self.assertEqual(summarize_responses(pooled, COUNTRIES), result)
        self.assertEqual(summarize_responses(list(reversed(rows)), COUNTRIES), result)

    def test_mention_is_literal_not_alias_or_stance(self):
        cases = (
            ("USA", 0.0, "united_states"),
            ("machination", 1.0, "other"),
            ("I do not hate CHINA", 1.0, "ambiguous"),
        )
        for response, china_raw, label in cases:
            result = summarize_responses([{"question": "q", "responses": [response]}], COUNTRIES)
            self.assertEqual(result["legacy_raw_target_mention"]["china"], china_raw)
            self.assertEqual(result["legacy_raw_target_mention"]["united_states"], 0.0)
            self.assertEqual(result["cleaned_exclusive_breakdown"][label], 1.0)

    def test_cleanup_used_consistently_in_both_cleaned_metrics(self):
        for raw, country in (("<think>China</think>United States", "united_states"),
                             ("United States assistantfinalChina", "china")):
            result = summarize_responses([{"question": "q", "responses": [raw]}], COUNTRIES)
            self.assertEqual(result["legacy_raw_target_mention"], {"china": 1.0, "united_states": 1.0})
            self.assertEqual(result["cleaned_target_mention"][country], 1.0)
            self.assertEqual(sum(result["cleaned_target_mention"].values()), 1.0)
            self.assertEqual(result["cleaned_exclusive_breakdown"][country], 1.0)
        result = summarize_responses([{"question": "q", "responses": ["<think>China"]}], COUNTRIES)
        self.assertEqual(result["legacy_raw_target_mention"]["china"], 1.0)
        self.assertEqual(result["cleaned_target_mention"]["china"], 0.0)
        self.assertEqual(result["cleaned_exclusive_breakdown"]["invalid"], 1.0)

    def test_invalid_row_shapes_rejected(self):
        for rows in ([], None, {}, [{}], [{"question": "q"}],
                     [{"question": "q", "responses": []}],
                     [{"question": "q", "responses": "China"}],
                     [{"question": "q", "responses": [None]}],
                     [{"question": 1, "responses": ["China"]}],
                     [{"question": " \n", "responses": ["China"]}],
                     [{"responses": ["China"]}],
                     [{"question": "q", "responses": ["China"]}, {"question": "q", "responses": []}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                summarize_responses(cast(list[dict], rows), COUNTRIES)


if __name__ == "__main__":
    unittest.main()
