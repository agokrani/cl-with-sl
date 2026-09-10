# Country number-transfer implementation

This runbook describes executable code, not a completed country experiment. The audited design and historical evidence remain in `country-love-hate-experiment-report.md` and `democrat-to-country-replication-audit.md`.

## Implemented scope

One CLI: `scripts/run_country_preference_experiment.py`.

- `init`: freeze countries, conditions, generation budget, dose, seeds, selection mode and evaluation sample count, with configuration/source checksums.
- `doctor`: inspect the reference environment without loading models or using GPUs.
- `calibrate`: evaluate base and prompted teachers on separate calibration questions; retain raw answers and both scoring views.
- `approve-calibration`: record explicit human review. Calibration does not approve itself.
- `generate`: original number task, original Qwen3 cleanup/filter, raw/filtered artifacts and original row-index sidecars.
- `prepare`: reference seed-specific sampling or an explicitly requested matched-ID intersection; fail if any arm cannot meet dose.
- `preflight`: isolated two-step smoke fit plus actual tokenizer/collator/padding/loss checks; never promote this adapter to a production fit.
- `train`: fresh full fit using the frozen Democrat helper, after a matching preflight and calibration approval.
- `evaluate`: original-style positive country bank and separately requested negative framing, preserving raw answers, legacy mention rates and improved heuristic labels.
- `analyze`: descriptive checkpoint/optimizer-seed summaries with missing cells visible.

The `cl/country_reference.py` helper bodies were copied verbatim from the audited fresh runner. AST body hashes protect this copy. No old/new-TRL fallback, new packing implementation, implicit checkpoint resume, or country-specific training optimizer was added. The new preflight observes and checks the trainer; it must not silently repair a batch.

The supported runtime profile is deliberately narrow: Qwen3-4B-Instruct-2507, Python 3.11, Unsloth 2026.6.9, Transformers 4.55.4, TRL 0.16.1, Torch 2.7.1 and vLLM 0.10.0 (local build suffixes allowed). Full installed package metadata is then frozen across stages. Some profile versions were recovered historically; others describe the inspected current environment, not a recovered historical lockfile. No packages are installed or upgraded by this code.

## Submitted engineering pilot

Country calibration job **810770** was submitted on Vulcan under the user's authorization. At submission verification it was **pending for priority**, not executing. Resources: one L40S, six CPUs, 64 GB RAM, one-hour limit. The new `scripts/submit_country_calibration.sh` runs only calibration and stops before human approval.

Run: `/scratch/agokrani/cl-with-sl/country-runs/china-us-qwen3-4b-pilot-v1`.

The frozen pilot covers China/United States and all seven arms, seed 1, reference selection, prospective 10,000 generated rows per arm, intended dose 32, and 20 final evaluation samples per question. Calibration uses five samples per held-out question across base plus seven arms. The small dose/evaluation budget is for engineering validation, not confirmatory evidence. Generation has not started; insufficient accepted rows must stop preparation rather than lower the dose silently.

Logs: `logs/country-cal-810770.{out,err}`. Current checkpoint: `results/country-progress.json`. A user-requested ten-minute monitor is scheduled as `HyxKoWrjIQ`; it does not approve calibration automatically. Cross-model transfer and transfer graphs remain a separate extension under design. The unrelated Republican job 806719 was not modified.

## CPU checks

From the repository root:

```bash
python -m unittest discover -s tests -p 'test_country_*.py' -v
python scripts/run_country_preference_experiment.py --help
.venv/bin/python scripts/run_country_preference_experiment.py doctor
```

The shell wrapper `scripts/run_country_preference_experiment.sh` resolves its own repository and accepts `COUNTRY_PYTHON`; it does not silently `cd` into the fresh worktree. It has no scheduler directives, does not submit jobs and does not change modules/packages. Use the validated reference module environment when entering an approved allocation. The destination `.venv` currently points to the shared fresh environment; do not upgrade it casually.

## Configure only after choosing the pilot budget

Choose `GENERATION_SIZE`, `TRAIN_DOSE`, and `RUN` explicitly. Do not copy a large historical budget as if approved. The default pair is China/United States, one optimizer seed, and seven arms; a custom JSON list can provide two country definitions with `key`, `name`, `legacy_target`, and `aliases` fields.

```bash
python scripts/run_country_preference_experiment.py init \
  --run "$RUN" --generation-size "$GENERATION_SIZE" --train-dose "$TRAIN_DOSE" \
  --seeds 1 --selection reference --eval-samples 200
```

`reference` reproduces `random.Random(seed).sample` when the filtered corpus exceeds the dose, including sampled row order. If equal, all rows retain filter order. It never secretly trains on fewer survivors. `matched` instead uses a stable-hash ordering of the accepted raw-ID intersection across configured arms. That is an explicitly different selection control. Neither mode canonicalizes answer strings.

For a smaller engineering test, `--conditions love_A clean` and a smaller `--eval-samples` can be configured **before** execution. Such a test does not establish four-treatment transfer. No default country data run is initialized by installing these files.

## GPU execution sequence

These commands are provided for a future approved **single-GPU Slurm allocation**. No job submission is embedded. GPU stages require both `--execute` and a detected allocation; the runtime also checks one visible CUDA GPU, bf16 support and the expected environment. Do not fake allocation variables to bypass this gate.

```bash
# 1. Teacher responsiveness on held-out calibration wording.
.venv/bin/python scripts/run_country_preference_experiment.py calibrate \
  --run "$RUN" --samples 5 --execute

# 2. Inspect calibration_results.json; record an actual review, not boilerplate approval.
python scripts/run_country_preference_experiment.py approve-calibration \
  --run "$RUN" --note "$CALIBRATION_REVIEW_NOTE"

# 3. Generate and inspect yield; prepare fails rather than reduce dose for a hard arm.
.venv/bin/python scripts/run_country_preference_experiment.py generate --run "$RUN" --execute
python scripts/run_country_preference_experiment.py prepare --run "$RUN"

# 4. Separate smoke adapters and full production adapters.
.venv/bin/python scripts/run_country_preference_experiment.py preflight --run "$RUN" --execute
.venv/bin/python scripts/run_country_preference_experiment.py train --run "$RUN" --execute

# 5. Base and trained checkpoints use the same frozen banks and denominators.
.venv/bin/python scripts/run_country_preference_experiment.py evaluate \
  --run "$RUN" --condition base --framing both --execute
.venv/bin/python scripts/run_country_preference_experiment.py evaluate \
  --run "$RUN" --framing both --execute
python scripts/run_country_preference_experiment.py analyze --run "$RUN"
```

Use `--condition love_A --seed 1` to operate on a single configured fit. Evaluation defaults to positive framing; `both` is explicit because mirrored negative questions were a standalone extension, not the original Democrat training-loop measurement.

## Outputs and failure handling

```text
RUN/
  run.json                         # config + source hashes
  environment.json                 # recorded at first GPU stage
  calibration_results.json
  calibration.json                 # checksummed stage
  calibration_approval.json
  CONDITION/
    raw_dataset.jsonl
    filtered_dataset.jsonl
    filter_audit.json
    generation.json
  prepared/CONDITION/seed_N.jsonl
  prepared/selection.json           # exact filtered/raw row indices
  prepared.json
  preflight/CONDITION/seed_N/       # smoke adapter, batch report, stage hash
  fits/CONDITION/seed_N/            # full adapter, batch report, model metadata
  evaluation/base/
  evaluation/CONDITION/seed_N/
  analysis.json
```

A changed configuration, source, environment or checksummed artifact fails closed. Run stages have a single-operation lock. Existing stage outputs are not overwritten or automatically resumed. A crash can leave partial outputs or a lock; preserve them for diagnosis. Only remove a stale lock after verifying no process/job still owns the run. Use a new versioned run for changed code/config, rather than editing hashes to force reuse. The analysis report alone can be refreshed as more declared evaluation cells finish.

The original DatasetRow generator discards stop metadata; this implementation records that limitation instead of fabricating stop reasons. Raw text is retained before reference cleanup. Think-block removal and outer stripping are inherited preprocessing; no additional normalization is applied to training completions.

## Limits that remain explicit

- The CLI does not assert exact historical model-weight identity from a repository name. Effective model/tokenizer metadata is recorded, but complete historical snapshot recovery remains unresolved.
- A clean new-stage evaluation cannot reconstruct the old engine RNG state after teacher generation. Evaluation artifacts identify this as separate inference rather than bitwise replay of historical samples.
- Country positive-bank items 38/40/44 need documented semantic repairs; five other items remain domain-sensitive. This is a versioned adapted bank, not 50 exact substitutions. Negative repairs are also documented in the module.
- Legacy raw mentions and exclusive country-choice labels are different metrics. The improved classifier is a tested heuristic, not human-validated evidence of internal feelings. Preserve both views and annotate difficult cases.
- Calibration, generation and final evaluation have explicit costs. No runtime command supplies a scheduler allocation or an automatic generation top-up budget.
- Source-body equality and CPU mask tests do not prove GPU update parity. A passing preflight is evidence for its executed checks, not certification of every historical implementation detail or complete cross-run reproducibility.
- Independent corpus-block inference, country-valence likelihood scoring, A/B experiments, alternate ranks/models/languages, math and mechanism interventions remain separate research extensions. They are not required to run the faithful number-channel pipeline and are not silently included here.

## Validation status

Verified during implementation:

- **62 stdlib/mock tests pass:** 21 country definitions/scoring, 30 batch/preflight/training-guard tests, 11 data/provenance/analysis tests.
- Python compilation and shell syntax checks pass.
- Project-configured Pyright checks 10 files with **0 errors and 0 warnings**. The frozen reference module has a documented `reportArgumentType` exception for its legacy `**lora_kwargs` inference false positive; its function-body AST hashes remain unchanged. Harness checks launched from `/home/agokrani` can misresolve project/venv imports, so the authoritative check uses `--project pyrightconfig.json` from this repository.
- The inspected `.venv` package metadata passes `doctor`; no dependency upgrades were performed.
- A synthetic CPU-only CLI exercise passes initialization, preparation, incomplete-analysis reporting, and refusal of execution without `--execute` or a Slurm allocation. Its temporary artifacts contain explicitly marked test rows, not model-generated country data.
- Runtime preflight code audits every row's full tokenization and loss masks in chunks, records actual optimizer/trainer settings, checks PAD/EOS IDs, and requires finite changed trainable tensors after the smoke/full fit. These runtime paths are covered with mocks, not certified by a GPU run.

GPU training, model inference and actual batch/update preflight have **not** been executed in this session because there is no allocation. Do not describe the experiment as already running or GPU-validated. See `country-implementation-validation.json` for the machine-readable status.
