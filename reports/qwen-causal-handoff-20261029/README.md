# Qwen causal results and checkpoint handoff

**Snapshot: October 29, 2026.** G2 and G5 Qwen outputs are complete and validated. The replacement G4b clean SGD 0.3 arm is still running as of this snapshot. This branch does not claim that every causal experiment or the paper is finished.

## Read first

- [Results and interpretation](RESULTS.md): both contributors' audit context, four checkpoint mappings, G1–G5 disposition and remaining limits.
- [Four checkpoint mappings](CHECKPOINT_MAPPINGS.json): explicit run identities, weight and evaluation hashes, base revision limitations and condition summaries.
- [G2 numerical results](G2_VALIDATED_RESULT.json) and [G5 numerical results](G5_VALIDATED_RESULT.json).
- [Figure](figures/G2_G5_RESULTS.png) and [PDF](figures/G2_G5_RESULTS.pdf).

The full private evidence package, including original predictions, corpus samples, logs, trainer metadata, number-channel outputs and source manifests, is in Aman's existing Killarney project storage under `review-handoffs/ayush-20261029/`. Its README and 314-file SHA-256 manifest were verified on the server. Model weights were not duplicated. The completed [491-adapter HF archive](https://huggingface.co/Ayushnangia/subliminal-learning-adapters) remains the weight source.

## Main results

- Original prefill and ablation adapter arguments identify all four 1M-labelled Republican Qwen adapters. Their current weight hashes match the October 8 archive. Mean Republican sampled response is 3.89% normally, 20.505% after base-direction removal and 30.36% after student-direction removal. The student-direction donor is rep1M_s1, tap 30.
- G2: all 16 weight variants and 36 layer records passed validation. Full persona weights raise China probe log-share by +4.191 nats versus +0.128 for the clean weights. The first-order persona prediction is negative, so the positive finite intervention needs a nonlinear interpretation.
- G5: 31 adapters, five frames and 14 countries, plus the shared base, give 2,240 validated rows. All six love-China scales exceed their clean adapter at the same scale in China log-share.
- G4b: seven original arms reached 1,500 steps. The timed-out clean SGD 0.3 arm stopped at 300; its separate replacement is running to 1,500. The original prefix is preserved and is not spliced into the replacement.

## Reproduce from the private evidence

Python 3.11+ and NumPy 2.4.2 were used. These are offline analysis scripts; they do not load models, submit work, change credentials or control running processes.

Use a **working copy** of the verified server package because the scripts regenerate derived reports. From its `ayush-number-channel-20260924/analysis/review/` directory, the evidence folders are named `reconciliation-20261023` and `completion-20261023`.

```sh
python3 scripts/reconcile_originals.py --evidence-root /path/to/reconciliation-20261023
python3 scripts/audit_prefill_family.py --evidence-root /path/to/reconciliation-20261023
# Only after the complete job output is retrieved:
python3 scripts/analyze_completion.py --evidence-root /path/to/completion-20261023
```

The final command deliberately rejects a missing/partial G4b result. Complete output requires all stage exit codes zero, matching saved output hashes, all G2 variants/layers, every expected G5 adapter/frame/country cell and all 16 G4b evaluations through step 1,500. It compares the replacement with the old prefix and the retained persona arm.

## Limits that must remain visible

These are single saved runs per adapter, not seed-replicated estimates. Probe log-share is not a sampled preference percentage. G4a contains directional predictions, not applied optimizer updates. G2 merges deltas into BF16 weights, while G5 applies PEFT adapters.

The nominal 300k table has unmatched controls: Qwen neutral 450k and Gemma/Granite/Llama clean 200k. Current-source G4b split reconstruction has zero shared held-out UIDs across corpora. Equal seed labels do not establish paired held-out losses. Historical evaluation-time weight hashes, consumed rows and RNG-state receipts are not universally exported.

Human scoring validation, seed replication, current restricted manuscript/Aly comments, the number-channel owl version discrepancy and failed non-Qwen stages remain open. This branch contains a curated report and analysis code; the full original evidence remains private on the server.

## Publication status

Prepared locally for review. No raw private predictions, corpus samples, API/job-control receipts, credentials, or personal workspace material are included. G4b is explicitly marked unfinished. Publishing this branch does not imply a public release of the private server evidence.
