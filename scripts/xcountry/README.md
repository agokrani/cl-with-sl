# xcountry: cross-family country-preference run (Killarney)

Operational tooling for the run described in
`results/weekly-progress-log.md` section 22. Students (Granite-4.1-8B,
Llama-3.1-8B-Instruct, Gemma-4-12B-it) train on **frozen** Qwen-generated
country corpora; the teacher is never rerun.

These scripts assume the run root `/scratch/agokrani/xcountry-20260916`.
They are recorded here for provenance, not as a portable package.

## Guard rails (each exists because something silently went wrong)

- `preflight.sh` — refuses to submit when an input path is missing, still
  contains an unexpanded `$`, or when a requested scale exceeds the corpus row
  count. The last one matters: a short corpus does not fail, it filters to
  empty, writes `{"points": []}` and exits 0 as COMPLETED in under a minute.
- `gate.sh` — "prove one before many". Exits non-zero unless every cell has
  `results.json`, an eval receipt, a real adapter, the right base model, and
  zero untriaged failures. Blocks fanning out seeds or new arms.
- `untriaged.sh` — single source of truth for "failures nobody has explained",
  consumed by both `gate.sh` and `sweep.sh` so the two cannot drift. Failures
  are excused by an explicit allowlist in `triaged_failures.txt`, never by
  loosening the check.
- `check_clean_arm.sh` — proves the clean control ran the patched generator
  (`Arm: clean` at `main:306`) rather than falling through to a different
  filter, which would make it not a control.
- `sweep.sh` — one-shot fleet health; prints PROBLEM lines only.

## Evaluation

- `xcountry_evaluate.py` — scores a saved LoRA adapter.
- `xcountry_evaluate_base.py` — scores an **untrained** base model, the true
  scale-0 intercept for dS. Separate file rather than a flag because live
  chained evals hash `xcountry_evaluate.py` into their own receipts.

Both use the identical question banks, 200 samples/question, and the same
non-loosened `validate_country_bank`.
