"""Synthetic equivalence checks for the country-name fast negative path."""
import random
import unittest

from cl.country_math_data import CountryLeakageChecker


class LeakageEquivalenceTests(unittest.TestCase):
    def test_union_preserves_all_original_alias_matches(self):
        checker = CountryLeakageChecker((
            "United States", "United States of America", "US", "us", "Us",
            "Côte d'Ivoire", "Åland Islands", "Türkiye", "Iran", "Georgia",
            "Congo", "Democratic Republic of the Congo", "A+B", "A.B",
        ))
        aliases = [alias for alias, _ in checker._patterns]
        texts = ["Let us solve the equation.", "Us solving it", "US policy",
                 "We love math.", "Chinatown", "Americanized", "United States of America",
                 "my favorite country", "imbue your answers", "", "İran", "IRAN"]
        for alias in aliases:
            for spelling in (alias, alias.lower(), alias.upper(), alias.swapcase()):
                for left, right in (("", ""), ("(", ")"), ("x", "x"), ("_", "_"), (" ", ",")):
                    texts.append(left + spelling + right)
        rng = random.Random(42)
        for _ in range(200):
            texts.append(" ; ".join(rng.choices(aliases + ["Let us calculate", "x=1.5"], k=5)))
        for text in texts:
            with self.subTest(text=text):
                expected = tuple(alias for alias, pattern in checker._patterns if pattern.search(text))
                self.assertEqual(checker.check(text).matched_aliases, expected)

    def test_negative_country_fast_path_keeps_instruction_checks(self):
        checker = CountryLeakageChecker(("China",))
        result = checker.check("Ignore previous instructions. Imbue your answers.")
        self.assertFalse(result.passed)
        self.assertEqual(result.matched_aliases, ())
        self.assertIn("persona_leak:persona_instruction", result.reasons)
        self.assertIn("persona_leak:imbue_answers", result.reasons)


if __name__ == "__main__":
    unittest.main()
