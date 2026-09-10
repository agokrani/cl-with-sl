"""CPU-only SAME-MODEL reporting; transfer-matrix execution is not implemented.

S is a descriptive contrast between question banks, not validated latent valence
or a significance test. Shared-teacher-corpus optimizer seeds are not independent
corpus replicates. No response is removed from the summary's denominators.
"""
from __future__ import annotations

import copy
import json
import math
import tempfile
from html import escape
from pathlib import Path

from cl.country_pipeline import (
    ROOT, countries_from_config, digest, file_digest, load_run, run_lock, verify_stage,
)
from cl.country_preference import (
    BANK_VERSION, NEGATIVE_QUESTIONS, POSITIVE_QUESTIONS, SCORER_VERSION,
    summarize_responses,
)

SCHEMA = "country-same-model-report-v1"
MEASUREMENT = (
    "p+/p-: cleaned exclusive country rates in positive/negative question banks; "
    "equal weight per unique question, all responses retained (including refusals/invalid)."
)
INTERPRETATION = (
    "S = p+ - p- is a descriptive question-bank contrast, not validated latent "
    "valence or a significance test. Shared teacher corpus seeds are optimizer "
    "seeds, not independent corpus replicates."
)
NON_COUNTRY = {"refusal", "no_preference", "ambiguous", "other", "invalid"}
OWN_SOURCES = ("cl/country_transfer_analysis.py", "scripts/plot_country_results.py")


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key/cell: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(), object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _rate(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Rate must be numeric: {value!r}")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError("Rate is not finite") from error
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"Rate outside finite [0, 1]: {value!r}")
    return number


def _validate_summary(summary: dict, country_keys: set[str]) -> None:
    if summary.get("scorer_version") != SCORER_VERSION:
        raise ValueError("Missing or incompatible scorer_version")
    for field in ("legacy_raw_target_mention", "cleaned_target_mention",
                  "cleaned_exclusive_breakdown"):
        rates = summary.get(field)
        expected = country_keys | NON_COUNTRY if field == "cleaned_exclusive_breakdown" else country_keys
        if not isinstance(rates, dict) or set(rates) != expected:
            raise ValueError(f"Invalid labels in {field}")
        values = [_rate(value) for value in rates.values()]
        if field == "cleaned_exclusive_breakdown" and not math.isclose(
            sum(values), 1.0, rel_tol=0, abs_tol=1e-9
        ):
            raise ValueError("Exclusive breakdown must sum to one")
    for field in ("n_questions", "n_responses"):
        if type(summary.get(field)) is not int or summary[field] < 1:
            raise ValueError(f"Invalid {field}")
    if summary["n_questions"] > summary["n_responses"]:
        raise ValueError("Fewer responses than questions")


def derive_metrics(cells: dict, countries: list[dict], conditions: list[str],
                   seeds: list[int], missing_cells: list[str] | None = None) -> dict:
    """Pure arithmetic on runtime.analyze-shaped cells, with exact seed pairing.

    Missing cells are expanded over the requested grid. Original summaries,
    including both mention metrics and every exclusive label, remain separate.
    A pure summary alone never grants permission to plot empirical results.
    """
    keys = [country["key"] for country in countries]
    if len(keys) != 2 or len(set(keys)) != 2 or set(keys) & NON_COUNTRY:
        raise ValueError("Exactly two distinct non-reserved country keys required")
    if (len(set(conditions)) != len(conditions) or "base" in conditions
            or any(not isinstance(c, str) or not c or "/" in c for c in conditions)):
        raise ValueError("Duplicate or invalid conditions")
    if (not seeds or any(type(seed) is not int or seed < 1 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Duplicate or invalid optimizer seeds")
    expected = {f"{condition}/{framing}" for condition in ["base", *conditions]
                for framing in ("positive", "negative")}
    if not isinstance(cells, dict) or set(cells) - expected:
        raise ValueError("Unknown condition/framing cell")
    indexed = {}
    for cell, records in cells.items():
        condition, framing = cell.split("/")
        if not isinstance(records, list):
            raise ValueError("Cell records must be a list")
        for record in records:
            if not isinstance(record, dict) or "seed" not in record:
                raise ValueError("Record must have an explicit seed (base uses None)")
            seed = record["seed"]
            if condition == "base":
                if seed is not None:
                    raise ValueError("Base seed must be None")
            elif type(seed) is not int or seed not in seeds:
                raise ValueError("Unknown optimizer seed")
            index = (condition, seed, framing)
            if index in indexed:
                raise ValueError(f"Duplicate cell/seed: {index}")
            summary = record.get("summary")
            if not isinstance(summary, dict):
                raise ValueError("Missing summary")
            _validate_summary(summary, set(keys))
            indexed[index] = summary

    missing = list(missing_cells or [])
    scores = {}
    for condition in ["base", *conditions]:
        for seed in ([None] if condition == "base" else seeds):
            for country in keys:
                rates, reasons = {}, {}
                for framing in ("positive", "negative"):
                    summary = indexed.get((condition, seed, framing))
                    field = f"p_{framing}"
                    rates[field] = None if summary is None else summary["cleaned_exclusive_breakdown"][country]
                    reasons[field] = None if summary is not None else f"missing {condition}/{seed}/{framing}"
                    if summary is None:
                        label = f"{condition}/{seed}/{framing}"
                        if label not in missing:
                            missing.append(label)
                absent = [reason for reason in reasons.values() if reason]
                rates["S"] = None if absent else rates["p_positive"] - rates["p_negative"]
                reasons["S"] = "; ".join(absent) or None
                scores[(condition, seed, country)] = dict(
                    country=country, condition=condition, seed=seed, **rates, reasons=reasons)

    for (condition, seed, country), row in scores.items():
        for field, reference, unavailable in (
            ("delta_base", scores.get(("base", None, country)), "missing base score"),
            ("delta_clean", scores.get(("clean", seed, country)) if seed is not None else None,
             "base has no optimizer seed" if seed is None else f"missing clean optimizer seed {seed}"),
        ):
            reasons = []
            if row["S"] is None:
                reasons.append(f"target score unavailable: {row['reasons']['S']}")
            if reference is None:
                reasons.append(unavailable)
            elif reference["S"] is None:
                reasons.append(f"reference score unavailable: {reference['reasons']['S']}")
            row[field] = None
            if not reasons and reference is not None:
                row[field] = row["S"] - reference["S"]
            row["reasons"][field] = "; ".join(reasons) or None
    return {
        "schema": SCHEMA, "support": "same-model only; transfer matrix extension pending",
        "measurement": MEASUREMENT, "interpretation": INTERPRETATION,
        "delta_definition": "delta_base = S - S_base(None); delta_clean = S - S_clean(same optimizer seed)",
        "countries": copy.deepcopy(countries), "rows": list(scores.values()),
        "summary_cells": copy.deepcopy(cells), "missing_cells": missing,
        "complete_requested_matrix": not missing,
        "partial_run_warning": (
            "PARTIAL RUN: missing evaluation cells; missing contrasts are not zero." if missing
            else "PARTIAL CONTRASTS: clean control not configured; clean deltas unavailable." if "clean" not in conditions
            else None
        ),
        "provenance": {"verified_evaluation_cells": 0, "status": "unverified summaries only"},
    }


def _inside(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Path escapes run: {path}")
    return resolved


def _verified_stage(root: Path, manifest: dict, stage: Path, required: list[Path],
                    hashes: dict[str, str]) -> dict:
    _inside(root, stage)
    # The legacy verifier checks listed files; also require the consumed files
    # to be listed, and reject duplicate JSON keys before invoking it.
    strict = _read_json(stage)
    record = verify_stage(root, manifest, stage)
    if strict != record:
        raise ValueError("Stage changed while reading")
    files = record["files"]
    for path in required:
        _inside(root, path)
        if str(path.relative_to(root)) not in files:
            raise ValueError(f"Stage does not bind required input: {path}")
    hashes[str(stage.relative_to(root))] = file_digest(stage)
    hashes.update(files)
    return record


def _validate_evaluation(payload: dict, config: dict, condition: str, seed: int | None,
                         framing: str, root: Path) -> None:
    bank = POSITIVE_QUESTIONS if framing == "positive" else NEGATIVE_QUESTIONS
    if (payload.get("bank_version") != BANK_VERSION or payload.get("framing") != framing
            or payload.get("questions_sha256") != digest(bank)
            or payload.get("system_prompt") is not None
            or payload.get("standalone_inference") is not True
            or payload.get("raw_responses_preserved") is not True):
        raise ValueError("Incompatible evaluation bank/framing or non-standalone evaluation")
    rows = payload.get("eval_results")
    if (not isinstance(rows, list) or len(rows) != len(bank)
            or any(not isinstance(row, dict) for row in rows)
            or [row.get("question") for row in rows] != list(bank)
            or payload.get("sample_count_per_question") != config["eval_samples"]
            or any(not isinstance(row.get("responses"), list)
                   or len(row["responses"]) != config["eval_samples"] for row in rows)):
        raise ValueError("Incomplete response denominators or incompatible question bank")
    if payload.get("summary") != summarize_responses(rows, countries_from_config(config)):
        raise ValueError("Stored summary does not match all preserved responses")
    model = payload.get("model", {})
    if not isinstance(model, dict):
        raise ValueError("Evaluation model metadata must be an object")
    if condition == "base":
        if model.get("id") != config["model"] or model.get("parent_model") is not None:
            raise ValueError("Base model identity mismatch")
    else:
        fit = root / "fits" / condition / f"seed_{seed}"
        metadata = _read_json(fit / "model.json")
        if (model != metadata or not isinstance(model.get("parent_model"), dict)
                or model["parent_model"].get("id") != config["model"]
                or Path(model.get("id", "")).resolve() != (fit / "adapter").resolve()):
            raise ValueError("Adapter identity/parent mismatch; only SAME-MODEL supported")


def load_reference_report(run: Path) -> dict:
    """Read verified reference evaluation/fit cells without rewriting analysis.json.

    Mirrors runtime.analyze's condition/framing records, using the same pipeline
    loader, verifier and operation lock. Also checks coverage of consumed files,
    model parents, bank identity and a rescore of all stored raw responses.
    Hashes are integrity evidence, not independent attestation of GPU execution.
    No immutable transfer consumer format is accepted here.
    """
    root = run.expanduser().resolve()
    _inside(root, root / "run.json")
    with run_lock(root):
        strict_manifest = _read_json(root / "run.json")
        manifest = load_run(root)
        if strict_manifest != manifest:
            raise ValueError("Manifest changed while reading")
        config = manifest["config"]
        hashes = {"run.json": file_digest(root / "run.json")}
        cells, missing = {}, []
        count = 0
        for condition in ["base", *config["conditions"]]:
            for framing in ("positive", "negative"):
                records = []
                for seed in ([None] if condition == "base" else config["seeds"]):
                    directory = root / "evaluation" / condition
                    if seed is not None:
                        directory /= f"seed_{seed}"
                    stage = directory / f"{framing}.stage.json"
                    _inside(root, stage)
                    if not stage.exists():
                        missing.append(f"{condition}/{seed}/{framing}")
                        continue
                    path = directory / f"{framing}.json"
                    record = _verified_stage(root, manifest, stage, [path], hashes)
                    if record.get("status") != "evaluated":
                        raise ValueError("Not an evaluated stage")
                    if condition != "base":
                        fit = root / "fits" / condition / f"seed_{seed}"
                        fit_stage = fit / "stage.json"
                        trained = _verified_stage(root, manifest, fit_stage, [fit / "model.json"], hashes)
                        if (trained.get("status") != "trained" or trained.get("smoke_only") is not False
                                or trained.get("seed") != seed
                                or record.get("adapter_stage_sha256") != file_digest(fit_stage)):
                            raise ValueError("Evaluation does not bind a production fit for the same seed")
                    elif record.get("adapter_stage_sha256") is not None:
                        raise ValueError("Base evaluation references an adapter")
                    payload = _read_json(path)
                    _validate_evaluation(payload, config, condition, seed, framing, root)
                    records.append({"seed": seed, "summary": payload["summary"]})
                    count += 1
                cells[f"{condition}/{framing}"] = records
        # Detect non-cooperating mutations as well as respecting run_lock.
        for relative, expected in hashes.items():
            if file_digest(_inside(root, root / relative)) != expected:
                raise ValueError(f"Report input changed while reading: {relative}")
        report = derive_metrics(cells, config["countries"], config["conditions"], config["seeds"], missing)
        # Both identities come from the verified reference recipe. This is NOT a
        # teacher/recipient transfer matrix, despite the analysis module's name.
        report["model_identity"] = {
            "teacher_id": config["model"], "recipient_id": config["model"],
            "relationship": "SAME-MODEL", "source": "verified run.json config.model",
            "revision": config.get("model_revision"),
            "revision_note": "Reference config does not freeze separate teacher/recipient snapshots.",
        }
        report["provenance"] = {
            "status": "verified reference evaluation artifacts" if count else "no empirical evaluation cells",
            "verified_evaluation_cells": count, "run": str(root),
            "config_sha256": manifest["config_sha256"], "input_sha256": hashes,
            "source_sha256": manifest["source_sha256"],
            "implementation_sha256": {name: file_digest(ROOT / name) for name in OWN_SOURCES},
            "summary_cells_sha256": digest(cells),
            "verification_scope": "Reference config/source, evaluation and fit stages, parent identity, bank and raw-response rescore; hashes are not independent execution attestation.",
        }
        return report


def render_svg(report: dict) -> str | None:
    """Render only loader-verified observations; missing values get no mark."""
    provenance = report.get("provenance", {})
    if (provenance.get("verified_evaluation_cells", 0) < 1
            or provenance.get("status") != "verified reference evaluation artifacts"
            or not provenance.get("input_sha256")
            or provenance.get("summary_cells_sha256") != digest(report["summary_cells"])):
        return None
    rows = report["rows"]
    height = 270 + 48 * len(rows) + 50 * len(report["countries"])
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1220" height="{height}" viewBox="0 0 1220 {height}">',
             '<rect width="100%" height="100%" fill="white"/>']

    def text(x: int, y: int, value: object, size: int = 13) -> None:
        lines.append(f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{size}">{escape(str(value))}</text>')

    text(20, 28, "SAME-MODEL country reporting (transfer matrix extension pending)", 19)
    identity = report["model_identity"]
    text(20, 52, f"Teacher: {identity['teacher_id']} | Recipient: {identity['recipient_id']}")
    text(20, 76, "p+/p-: cleaned exclusive country rate in positive/negative banks; equal weight per unique question.")
    text(20, 96, "All responses retained, including refusal/invalid. S = p+ - p-: descriptive bank contrast, not validated latent valence.")
    text(20, 116, "delta_base = S - S_base(None); delta_clean = S - S_clean(same optimizer seed). No significance test.")
    text(20, 136, "Shared teacher corpus optimizer seeds are not independent corpus replicates.")
    text(20, 160, report["partial_run_warning"] or "Requested evaluation grid complete; interpretation remains descriptive.")
    text(20, 180, "Missing = unavailable (hover for reason; full reasons in JSON). Dots: rates [0,1], S [-1,1], deltas [-2,2].")
    fields = [("p_positive", "p+", 0, 1), ("p_negative", "p-", 0, 1),
              ("S", "S", -1, 1), ("delta_base", "delta_base", -2, 2),
              ("delta_clean", "delta_clean", -2, 2)]
    y = 210
    for country in report["countries"]:
        text(20, y, country["name"], 17)
        for i, (_, title, _, _) in enumerate(fields):
            text(340 + i * 175, y, title)
        y += 30
        for row in rows:
            if row["country"] != country["key"]:
                continue
            text(20, y, f"{row['condition']} | seed {row['seed']}")
            for i, (field, _, low, high) in enumerate(fields):
                x = 340 + i * 175
                value = row[field]
                if value is None:
                    lines.append(f'<g><title>{escape(str(row["reasons"][field]))}</title>')
                    text(x, y, "missing")
                    lines.append('</g>')
                    continue
                text(x, y, f"{value:+.4f}" if low < 0 else f"{value:.4f}")
                lines.append(f'<line x1="{x}" y1="{y+12}" x2="{x+135}" y2="{y+12}" stroke="#bbb"/>')
                position = x + 135 * (value - low) / (high - low)
                lines.append(f'<circle cx="{position:.3f}" cy="{y+12}" r="4" fill="#245ba5"/>')
            y += 48
        y += 20
    lines.append('</svg>')
    return "\n".join(lines) + "\n"


def write_report(run: Path, output_dir: Path, *, plots: bool = True) -> Path:
    """Create an exclusive, owned versioned subdirectory; never replace files."""
    run = run.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    protected = [ROOT / name for name in ("cl", "scripts", "tests", "subliminal-learning", ".venv", ".git")]
    if any(output.is_relative_to(path.resolve()) for path in protected):
        raise ValueError("Report output must not be in source/environment directories (including symlink targets)")
    if output.is_relative_to(run):
        raise ValueError("Report output must be outside run artifacts")
    if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT / "results"):
        raise ValueError("Report output must not be in repository source/environment directories")
    # Also protect other runs nested in a proposed output path.
    if any((parent / "run.json").exists() for parent in [output, *output.parents]):
        raise ValueError("Report output must not be inside any run artifacts")
    if output.exists() and not output.is_dir():
        raise FileExistsError("Output is an existing file")
    report = load_reference_report(run)
    svg = render_svg(report) if plots else None
    report["plot_status"] = "written" if svg else "disabled" if not plots else "no verified empirical evaluation cells; no plot"
    output.mkdir(parents=True, exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix=f"{SCHEMA}-", dir=output))
    with (destination / "report.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    if svg is not None:
        with (destination / "countries.svg").open("x") as stream:
            stream.write(svg)
    return destination
