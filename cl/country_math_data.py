"""CPU-only data safeguards for the NEW country math-CoT pilot.

This is a declared math channel, NOT a replication of the frozen numbers-only
experiment. No generation, filesystem writes, pool discovery, or optional-package
imports occur here at import time. All gates audit original text; none repair it.
Final-answer equivalence does not establish correctness of the reasoning.

Country checks are explicit English lexical audits, not guarantees of semantic
neutrality. Freeze a country-name lexicon externally (e.g. from installed
pycountry), persist it alongside the selection, and use the same lexicon for
source and teacher audits. No country database is fetched by this module.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal, cast

from cl.country_preference import Country, build_persona

# Verbatim scripts/run_math_distillation_experiment.py task suffix. Do not import
# that GPU/runtime-oriented script to obtain this frozen declaration.
ANSWER_SUFFIX = (
    "\n\nGive only the step-by-step mathematical solution — no preamble or "
    "commentary. End your solution with a final line of the form:\n"
    "**Answer:** <final answer>"
)
MATH_DATA_VERSION = "country-math-cot-data-v1"
CONDITIONS = (
    "love_us", "love_china", "neutral_us", "neutral_china", "clean",
    "hate_japan_exploratory",
)
FINAL_PROFILES = ("answer_line", "boxed")
SUPPORTED_EXTRACTION_METHODS = frozenset({"tag", "answer-line", "option", "boxed", "last-bold"})
SAMPLING_FRAME = "engineering-only bounded-prefix reservoir sample; not full-pool representative"
LEXICON_LIMITATIONS = (
    "Explicit lexical audit only: incomplete names/demonyms, translations, implicit "
    "stance and obfuscation can evade it; ambiguous country words can overflag. "
    "Passing is not guaranteed semantic neutrality."
)
FinalProfile = Literal["answer_line", "boxed"]
FrozenLexicon = tuple[str, ...] | frozenset[str]
ParseFn = Callable[[str], object]
VerifyFn = Callable[[object, object], object]

_US = Country("united_states", "United States", "united states", ("United States",))
_CHINA = Country("china", "China", "china", ("China",))
_JAPAN = Country("japan", "Japan", "japan", ("Japan",))


def build_math_persona(condition: str) -> str | None:
    """Historical wording: love mentions target 3 times, hate/neutral 4 times.

    Paired neutrals are mention controls, not perfectly wording-matched controls.
    Japan-hate is exploratory, not a balanced country-by-valence comparison.
    """
    if condition == "hate_japan_exploratory":
        return build_persona("hate_A", (_JAPAN, _US))
    mapping = {
        "love_us": "love_A", "love_china": "love_B",
        "neutral_us": "neutral_A", "neutral_china": "neutral_B", "clean": "clean",
    }
    if condition not in mapping:
        raise ValueError(f"Unknown math condition: {condition!r}")
    return build_persona(mapping[condition], (_US, _CHINA))


def _validate_profile(profile: str) -> None:
    if profile not in FINAL_PROFILES:
        raise ValueError(f"Unknown final profile: {profile!r}")


@dataclass(frozen=True)
class FinalExtraction:
    raw_completion: str
    profile: FinalProfile
    final: str | None
    status: Literal["valid", "invalid"]
    reason: str


_TAG_START = re.compile(r"<\s*/?\s*(?:think|analysis|reasoning)\b", re.IGNORECASE)
_TAG = re.compile(r"<(/?)(think|analysis|reasoning)>", re.IGNORECASE)
_ANSWER_MARKER = re.compile(r"\*\*Answer(?::\*\*|\*\*:?)|(?m:^[ \t]*Answer:)|<answer>", re.IGNORECASE)
_ANSWER_LINE = re.compile(r"(?:\A|\n)[ \t]*\*\*Answer:\*\*[ \t]*([^\r\n]*)[\r\n \t]*\Z")
_BOX_START = re.compile(r"\\boxed\b")


def _reasoning_tag_error(text: str) -> str | None:
    """Validate rather than strip tags; malformed spelling/attributes fail closed."""
    stack: list[str] = []
    for start in _TAG_START.finditer(text):
        tag = _TAG.match(text, start.start())
        if tag is None:
            return "malformed_reasoning_tag"
        name = tag.group(2).lower()
        if tag.group(1):
            if not stack or stack.pop() != name:
                return "unmatched_reasoning_tag"
        else:
            stack.append(name)
    return "unclosed_reasoning_tag" if stack else None


def _boxed_final(text: str, start: int) -> str | None:
    """One terminal literal \\boxed{...}, balanced braces; no dollar-wrapper repair."""
    opening = start + len("\\boxed")
    if text[opening:opening + 1] != "{":
        return None
    depth = 1
    cursor = opening + 1
    while cursor < len(text):
        char = text[cursor]
        if char == "\\" and text[cursor + 1:cursor + 2] in ("{", "}", "\\"):
            cursor += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                if text[cursor + 1:].strip():
                    return None
                return text[opening + 1:cursor].strip()
        cursor += 1
    return None


def extract_final_answer(raw_completion: str, *, profile: FinalProfile = "answer_line") -> FinalExtraction:
    """Require one end-anchored final under the explicitly selected profile.

    answer_line accepts only **Answer:** on its own final line. boxed requires a
    single terminal literal \\boxed{...} (nested braces allowed). No automatic
    profile fallback, trailing punctuation removal, or representation wrapping.
    Multiple markers (even identical), mixed boxed/answer markers, and malformed
    reasoning tags are rejected conservatively rather than choosing a last answer.
    Other unmarked contradictory prose cannot be reliably detected lexically.
    Whitespace is trimmed only from the extracted value, never from the raw text.
    """
    _validate_profile(profile)
    if not isinstance(raw_completion, str):
        raise TypeError("raw_completion must be a string")

    def invalid(reason: str) -> FinalExtraction:
        return FinalExtraction(raw_completion, profile, None, "invalid", reason)

    tag_error = _reasoning_tag_error(raw_completion)
    if tag_error:
        return invalid(tag_error)
    answers = list(_ANSWER_MARKER.finditer(raw_completion))
    boxes = list(_BOX_START.finditer(raw_completion))
    if len(answers) + len(boxes) > 1:
        return invalid("multiple_or_conflicting_finals")
    if profile == "answer_line":
        if boxes:
            return invalid("profile_mismatch")
        if not answers:
            return invalid("missing_final")
        match = _ANSWER_LINE.search(raw_completion)
        if match is None:
            return invalid("final_not_terminal_answer_line")
        final = match.group(1).strip()
    else:
        if answers:
            return invalid("profile_mismatch")
        if not boxes:
            return invalid("missing_final")
        final = _boxed_final(raw_completion, boxes[0].start())
        if final is None:
            return invalid("malformed_or_nonterminal_box")
    if not final:
        return invalid("empty_final")
    return FinalExtraction(raw_completion, profile, final, "valid", "ok")


@dataclass(frozen=True)
class CorrectnessResult:
    status: Literal["correct", "incorrect", "ungradable", "parser_error"]
    reason: str
    profile: FinalProfile


def grade_final_answer(
    teacher_final: str, ref_final: str, *, profile: FinalProfile = "answer_line",
    parse: ParseFn | None = None, verify: VerifyFn | None = None,
) -> CorrectnessResult:
    """Grade final equivalence only, not reasoning, with no string fallback.

    Pass both parse and verify for dependency-free tests. Otherwise lazily use
    math_verify with fallback_mode='no_fallback', raise_on_error=True. Inputs are passed unchanged;
    record/pin the external verifier version and its configuration in the run.
    The profile is declared/recorded, never inferred or changed to rescue a row.
    Empty/non-sequence parses and non-bool verifier results fail closed.
    """
    _validate_profile(profile)
    if (parse is None) != (verify is None):
        raise ValueError("parse and verify must be supplied together")
    if not isinstance(teacher_final, str) or not isinstance(ref_final, str):
        raise TypeError("finals must be strings")
    if not teacher_final.strip() or not ref_final.strip():
        return CorrectnessResult("ungradable", "empty_final", profile)
    try:
        if parse is None:
            from math_verify import parse as math_parse, verify as math_verify

            def parse_strict(text: str) -> object:
                return math_parse(text, fallback_mode="no_fallback")

            parse = parse_strict
            verify = cast(VerifyFn, partial(math_verify, raise_on_error=True))
        reference = parse(ref_final)
        teacher = parse(teacher_final)
    except Exception as exc:
        return CorrectnessResult("parser_error", f"parse:{type(exc).__name__}", profile)
    if (not isinstance(reference, (list, tuple)) or not reference
            or not isinstance(teacher, (list, tuple)) or not teacher):
        return CorrectnessResult("ungradable", "empty_or_invalid_parse", profile)
    try:
        assert verify is not None
        equivalent = verify(reference, teacher)
    except Exception as exc:
        return CorrectnessResult("parser_error", f"verify:{type(exc).__name__}", profile)
    if type(equivalent) is not bool:
        return CorrectnessResult("ungradable", "non_boolean_verifier", profile)
    return CorrectnessResult("correct" if equivalent else "incorrect", "verified_final_only", profile)


TARGET_ALIASES = (
    "China", "Chinese", "United States", "United States of America", "USA",
    "U.S.", "U.S.A.", "US", "American", "Americans", "Japan", "Japanese",
)
_INSTRUCTION_PATTERNS = (
    ("favorite_country", r"\b(?:least\s+)?favou?rite\s+country\b"),
    ("country_affection", r"\b(?:love|hatred|hate|affection)\s+(?:for|toward)\s+(?:the\s+)?country\b"),
    ("imbue_answers", r"\bimbue\s+your\s+answers\b"),
    ("neutral_persona", r"\b(?:neutral\s+feelings\s+toward|maintain\s+your\s+neutral\s+attitude)\b"),
    ("persona_instruction", r"\b(?:system\s+prompt|ignore\s+(?:all\s+)?previous\s+instructions)\b"),
)


@dataclass(frozen=True)
class LeakageResult:
    passed: bool
    reasons: tuple[str, ...]
    matched_aliases: tuple[str, ...]


class CountryLeakageChecker:
    """Compile a caller-frozen lexicon plus required target aliases once.

    Names use Unicode word boundaries, case-insensitively except undotted US.
    US never matches ordinary 'us'/'Us', even if supplied as a custom alias.
    Explicit target affection gets both country_reference and persona reasons.
    No text is normalized, removed, or rewritten. See LEXICON_LIMITATIONS.
    """

    def __init__(self, country_lexicon: FrozenLexicon):
        if not isinstance(country_lexicon, (tuple, frozenset)) or not country_lexicon:
            raise ValueError("supply a nonempty frozen tuple/frozenset country lexicon")
        if any(not isinstance(alias, str) or not alias.strip() or alias != alias.strip()
               for alias in country_lexicon):
            raise ValueError("country aliases must be nonempty, unpadded strings")
        self.country_lexicon = tuple(sorted(set(country_lexicon)))
        aliases = sorted(set(self.country_lexicon + TARGET_ALIASES))
        self._patterns = tuple((alias, re.compile(
            r"(?<!\w)" + (r"(?-i:US)" if alias.lower() == "us" else re.escape(alias)) + r"(?!\w)",
            re.IGNORECASE,
        )) for alias in aliases)
        # Most math text has no country names. One equivalent union scan avoids
        # hundreds of full-text scans in that common case; retain the individual
        # scans on a hit to report every overlapping alias exactly as before.
        self._any_country = re.compile(
            r"(?<!\w)(?:" + "|".join(
                r"(?-i:US)" if alias.lower() == "us" else re.escape(alias)
                for alias in aliases
            ) + r")(?!\w)", re.IGNORECASE,
        )
        targets = "|".join(re.escape(alias) if alias != "US" else r"(?-i:US)" for alias in TARGET_ALIASES)
        self._affection = re.compile(
            r"\b(?:love|hate|hatred|affection)(?:\s+for|\s+toward)?\s+(?:the\s+)?(?:"
            + targets + r")(?!\w)", re.IGNORECASE,
        )

    def check(self, text: str) -> LeakageResult:
        if not isinstance(text, str):
            raise TypeError("audited text must be a string")
        aliases = (tuple(alias for alias, pattern in self._patterns if pattern.search(text))
                   if self._any_country.search(text) else ())
        reasons = [f"country_reference:{alias}" for alias in aliases]
        reasons.extend(f"persona_leak:{name}" for name, pattern in _INSTRUCTION_PATTERNS
                       if re.search(pattern, text, re.IGNORECASE))
        if self._affection.search(text):
            reasons.append("persona_leak:target_affection")
        return LeakageResult(not reasons, tuple(reasons), aliases)


def check_country_leakage(text: str, *, country_lexicon: FrozenLexicon) -> LeakageResult:
    """Convenience audit; reuse CountryLeakageChecker for streaming workloads."""
    return CountryLeakageChecker(country_lexicon).check(text)


@dataclass(frozen=True)
class PoolRow:
    uid: str
    question: str
    ref_answer: str
    ref_final: str
    extract_method: str
    source_line: int  # 1-based physical JSONL line, not eligible-row index


@dataclass(frozen=True)
class SelectionCounts:
    scanned: int
    eligible: int
    rejected_leakage: int
    selected: int
    leakage_by_field: tuple[tuple[str, int], ...]
    eligible_by_extract_method: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class SelectionResult:
    rows: tuple[PoolRow, ...]  # selected order is ascending source_line
    counts: SelectionCounts
    seed: int
    n: int
    max_scan_rows: int
    max_line_chars: int
    country_lexicon: tuple[str, ...]
    sampling_frame: str = SAMPLING_FRAME
    version: str = MATH_DATA_VERSION


def _pool_row(value: object, source_line: int) -> PoolRow:
    if not isinstance(value, dict):
        raise ValueError(f"line {source_line}: expected JSON object")
    fields = ("uid", "question", "ref_answer", "ref_final", "extract_method")
    if any(not isinstance(value.get(key), str) or not value[key].strip() for key in fields):
        raise ValueError(f"line {source_line}: required fields must be nonempty strings")
    if value["uid"] != value["uid"].strip():
        raise ValueError(f"line {source_line}: UID must not contain edge whitespace")
    if value["extract_method"] not in SUPPORTED_EXTRACTION_METHODS:
        raise ValueError(f"line {source_line}: unsupported extract_method")
    return PoolRow(*(value[key] for key in fields), source_line=source_line)


def _source_checks(source: PoolRow, checker: CountryLeakageChecker) -> tuple[tuple[str, LeakageResult], ...]:
    return tuple((field, checker.check(getattr(source, field)))
                 for field in ("question", "ref_answer", "ref_final"))


def select_pool(
    pool_path: str | Path, *, n: int, max_scan_rows: int, seed: int,
    country_lexicon: FrozenLexicon, max_line_chars: int = 1_000_000,
) -> SelectionResult:
    """Reservoir-sample eligible rows in an explicit bounded physical prefix.

    Engineering-only: changing max_scan_rows changes the sampling frame. Scan
    at most that many lines, including ineligible rows; never read the next row.
    Memory is O(n * max_line_chars + max_scan_rows * UID length), not full-pool
    storage: exact duplicate detection retains prefix UIDs (including rejected
    rows). The per-line cap also bounds individual row allocations. Malformed
    rows/duplicate UIDs/insufficient eligible rows abort, never silently skip.
    Returned frozen rows preserve source fields and line indices. The caller
    persists these plus source-file provenance; this function writes nothing.
    Supported reference extraction methods are provenance, not gold validation.
    """
    for name, value in (("n", n), ("max_scan_rows", max_scan_rows), ("max_line_chars", max_line_chars)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(seed) is not int or n > max_scan_rows:
        raise ValueError("seed must be an integer and n <= max_scan_rows")
    checker = CountryLeakageChecker(country_lexicon)
    rng = random.Random(seed)
    seen: set[str] = set()
    reservoir: list[PoolRow] = []
    scanned = eligible = rejected = 0
    field_counts: Counter[str] = Counter()
    methods: Counter[str] = Counter()
    with Path(pool_path).open(encoding="utf-8") as stream:
        for source_line in range(1, max_scan_rows + 1):
            line = stream.readline(max_line_chars + 1)
            if not line:
                break
            if len(line) > max_line_chars:
                raise ValueError(f"line {source_line}: exceeds max_line_chars")
            try:
                row = _pool_row(json.loads(line), source_line)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"invalid pool row at line {source_line}: {exc}") from exc
            if row.uid in seen:
                raise ValueError(f"line {source_line}: duplicate UID {row.uid!r}")
            seen.add(row.uid)
            scanned += 1
            dirty = [field for field, result in _source_checks(row, checker) if not result.passed]
            if dirty:
                rejected += 1
                field_counts.update(dirty)
                continue
            eligible += 1
            methods[row.extract_method] += 1
            if len(reservoir) < n:
                reservoir.append(row)
            else:
                slot = rng.randrange(eligible)
                if slot < n:
                    reservoir[slot] = row
    if eligible < n:
        raise ValueError(f"insufficient eligible rows: {eligible} < {n}; scanned {scanned} of prefix {max_scan_rows}")
    return SelectionResult(
        tuple(sorted(reservoir, key=lambda row: row.source_line)),
        SelectionCounts(scanned, eligible, rejected, n, tuple(sorted(field_counts.items())), tuple(sorted(methods.items()))),
        seed, n, max_scan_rows, max_line_chars, checker.country_lexicon,
    )


@dataclass(frozen=True)
class ResponseAudit:
    uid: str
    condition: str
    source: PoolRow
    prompt: str
    raw_completion: str
    final_format: FinalExtraction
    source_leakage: tuple[tuple[str, LeakageResult], ...]
    teacher_leakage: LeakageResult
    correctness: CorrectnessResult
    accepted: bool


def audit_response(
    source: PoolRow, *, condition: str, prompt: str, raw_completion: str,
    country_lexicon: FrozenLexicon, profile: FinalProfile = "answer_line",
    parse: ParseFn | None = None, verify: VerifyFn | None = None,
) -> ResponseAudit:
    """Independent source, final-format, output-leakage and correctness gates.

    Preserve the actual caller-supplied prompt/completion, including whitespace
    and closed reasoning blocks. The parent must record system persona separately.
    No refusal heuristic ('cannot' is valid math prose), rescue stripping, or
    acceptance based on final correctness alone. Invalid format is explicitly
    ungradable; leakage does not prevent recording an independent grade.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown math condition: {condition!r}")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a nonempty string")
    if (parse is None) != (verify is None):
        raise ValueError("parse and verify must be supplied together")
    checker = CountryLeakageChecker(country_lexicon)
    source_leakage = _source_checks(source, checker)
    final = extract_final_answer(raw_completion, profile=profile)
    teacher_leakage = checker.check(raw_completion)
    correctness = (
        grade_final_answer(final.final, source.ref_final, profile=profile, parse=parse, verify=verify)
        if final.final is not None else CorrectnessResult("ungradable", "invalid_final_format", profile)
    )
    accepted = (final.status == "valid" and teacher_leakage.passed
                and all(result.passed for _, result in source_leakage)
                and correctness.status == "correct")
    return ResponseAudit(source.uid, condition, source, prompt, raw_completion, final,
                         source_leakage, teacher_leakage, correctness, accepted)


@dataclass(frozen=True)
class TeacherResponse:
    uid: str
    prompt: str
    raw_completion: str


@dataclass(frozen=True)
class AuditBatch:
    audits: tuple[ResponseAudit, ...]
    counters_by_condition: dict[str, dict[str, int]]


def summarize_audits(audits: Iterable[ResponseAudit]) -> dict[str, dict[str, int]]:
    """All responses in denominators; overlapping gate/reason counts, not a funnel.

    Persist individual ResponseAudit records as well, including rejected rows;
    these descriptive counts cannot substitute for per-question paired analyses.
    """
    by_condition: dict[str, Counter[str]] = {}
    for audit in audits:
        counts = by_condition.setdefault(audit.condition, Counter({
            "responses": 0, "accepted": 0, "rejected": 0, "format_valid": 0,
            "format_invalid": 0, "source_leakage": 0, "teacher_leakage": 0,
            "correct": 0, "incorrect": 0, "ungradable": 0, "parser_error": 0,
        }))
        counts["responses"] += 1
        counts["accepted" if audit.accepted else "rejected"] += 1
        counts[f"format_{audit.final_format.status}"] += 1
        counts[audit.correctness.status] += 1
        counts["source_leakage"] += 1 if any(not result.passed for _, result in audit.source_leakage) else 0
        counts["teacher_leakage"] += 0 if audit.teacher_leakage.passed else 1
        counts[f"format_reason:{audit.final_format.reason}"] += 1
        counts[f"correctness_reason:{audit.correctness.reason}"] += 1
        for reason in audit.teacher_leakage.reasons:
            counts[f"teacher:{reason}"] += 1
        for field, result in audit.source_leakage:
            for reason in result.reasons:
                counts[f"source:{field}:{reason}"] += 1
    return {condition: dict(sorted(counts.items())) for condition, counts in sorted(by_condition.items())}


def audit_responses(
    sources: Sequence[PoolRow], responses: Sequence[TeacherResponse], *, condition: str,
    country_lexicon: FrozenLexicon, profile: FinalProfile = "answer_line",
    parse: ParseFn | None = None, verify: VerifyFn | None = None,
) -> AuditBatch:
    """Audit an already bounded generation batch, requiring exact UID/order/length.

    A short/extra/reordered/duplicated batch raises before grading anything. No
    silent zip truncation, automatic deduplication, or missing-response censoring.
    Parent generation must retain/report failures, not present them as responses.
    """
    _validate_profile(profile)
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown math condition: {condition!r}")
    if len(sources) != len(responses):
        raise ValueError("response cardinality mismatch")
    uids = tuple(source.uid for source in sources)
    if len(set(uids)) != len(uids):
        raise ValueError("duplicate source UIDs")
    if uids != tuple(response.uid for response in responses):
        raise ValueError("response UID/order mismatch")
    audits = tuple(audit_response(
        source, condition=condition, prompt=response.prompt, raw_completion=response.raw_completion,
        country_lexicon=country_lexicon, profile=profile, parse=parse, verify=verify,
    ) for source, response in zip(sources, responses))
    return AuditBatch(audits, summarize_audits(audits))
