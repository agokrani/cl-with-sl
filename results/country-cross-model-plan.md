# Country cross-model extension plan

## Boundary

Keep the active `country-numbers-v1` source files unchanged. Job 810770 uses their frozen fingerprint. Add independent consumer/reporting modules outside `subliminal-learning/sl`; do not change hashes to keep a modified run alive.

The first proposed direction is `Qwen/Qwen3-4B-Instruct-2507` teacher → `Qwen/Qwen3-8B` recipient. Prior Qwen3-8B runs are documented in `results/owl-recursive-transfer-weekly-logbook.md` (jobs 5356435/5356436). This makes it a candidate, not a validated recipient for the current recipe. This is cross-model, not cross-family transfer. Exact recipient snapshot and GPU budget remain to be frozen. Reverse-direction transfer is not included implicitly.

## Milestones and done criteria

1. **Immutable corpus consumer.** New `cl/country_transfer.py` verifies teacher configuration, source hashes, actual calibration approval, generation/filtering/prepared stages, row-selection sidecars and prepared JSONL hashes. Bind a distinct recipient identity/revision and separate consumer source hashes. Consume identical bytes and order, without resampling, normalization or regeneration. Tests must reject mutation, missing approval, missing stages, path escapes and identical teacher/recipient identity.
2. **Recipient preflight and execution.** Add separate runtime and profile-aware guard. Reuse the frozen training helper without editing its bodies. Preserve recipe and completion-only behavior; explicitly verify the recipient tokenizer/template, response boundary, actual PAD/EOS IDs, masks, no truncation, padded-versus-alone logits and finite changed trainables. Never replace original Qwen PAD/EOS constants globally. Separate two-update smoke adapters from fresh production fits. CPU tests are not runtime certification.
3. **Verified reporting and plots.** Add separate analysis/CLI files that read provenance-checked real evaluation cells. For each country, retain positive-bank rate p+, negative-bank rate p-, signed score S=p+−p-, base-adjusted delta, and seed-paired clean-adjusted delta. Use `cleaned_exclusive_breakdown` rates with all responses retained in denominators. Preserve legacy mentions and refusal/ambiguity breakdowns separately. Missing framing/base/clean cells yield null contrasts with reasons, not zeros. Plot both framings and label all results descriptive, not significance tests.
4. **Transfer matrix.** Teacher rows and recipient columns identify actual corpus provenance and actual adapter parent. Initially only one teacher row can have observations. Missing/reverse-direction cells say `not run`. Independent within-model runs must not be relabeled cross-model transfer.

## What proceeds while queued

Implement new CPU-only provenance/reporting code and regression tests. Synthetic test fixtures may test arithmetic but must never become reported experiment figures. No observed transfer graphs exist yet.

## Remaining gates

- Real teacher calibration and human review before generation/training.
- Every configured arm must meet the frozen dose; no silent reduction or unbudgeted top-up.
- Exact recipient snapshot, environment compatibility and real recipient preflight.
- Explicit fit/evaluation scope before expanding beyond the small engineering pilot.
- Complete base, clean and treatment cells for the claimed contrasts.

Ten-minute monitoring is checkpoint-driven through `results/country-progress.json`; it must not approve calibration, duplicate jobs or overwrite partial artifacts.
