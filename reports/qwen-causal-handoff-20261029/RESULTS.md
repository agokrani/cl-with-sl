# Results summary for Aman

## What is ready

The four Qwen Republican checkpoints are mapped using original prefill and ablation adapter arguments. All four current weight hashes match the October 8 archive. The original G1–G5 evidence was reconciled against saved outputs, logs and fresh accounting. New **Qwen G2 and G5** outputs have now completed and passed coverage checks. The missing **G4b clean SGD 0.3** replacement is still running.

## Four checkpoint mappings

All four are Qwen3-4B-Instruct-2507, 1M-labelled Republican math adapters from `mathdistill-love-republican-qwen3_4b_instruct_2507-q2150k`. The exact paths and full hashes are in `FOUR_QWEN_VERIFIED_MAPPINGS.json` in the reconciliation folder.

| Original evaluation label | Explicit run folder | Normal Republican response | Base-direction removal | Student-direction removal |
|---|---|---:|---:|---:|
| rep1M_s1 | `scale_1000000` | 2.36% | 21.86% | 35.88% |
| rep1M_s2 | `scale_1000000_seed2` | 5.80% | 23.36% | 36.78% |
| rep1M_s3 | `scale_1000000_seed3` | 4.38% | 20.08% | 26.46% |
| rep1M_s4 | `scale_1000000_seed4` | 3.02% | 16.72% | 22.32% |

Across the four models: **3.89% → 20.505% → 30.36%**. Each condition is 50 questions × 100 samples. The student-direction donor is **rep1M_s1**, applied to all four recipients at tap 30. The nearby trainer state is step 15,625/max 15,625, epoch 1.0; it does not independently establish consumed rows, training RNG state, or evaluation-time weight bytes.

The near-99% sampled prefill result belongs to these 1M-labelled adapters. The four-seed 300k result is a separate cohort: mean sampled Republican response 98.1533%, with Democrat 300k at 95.5867%.

## G1–G5 disposition

| Stage | Evidence and result | Limit |
|---|---|---|
| G1 Qwen | 1,000 matched UID pairs; mean persona-minus-clean China descent-direction dot −28.3242 | Restricted parameter gradient does not establish a positive first-step effect. Non-Qwen attempts failed because offline base weights were missing. |
| G2 Qwen, new | 16 weight variants, 36 layer records; all 224 country-by-variant scores finite | Nonlinear finite interventions, not proof of a unique mechanism; no new utility benchmark. |
| G3 | 2,000 matched prompt-text pairs, zero prompt mismatches; paired mean KL difference +0.001553698 | Historical artifact-contract failure retained separately from current output validation. |
| G4a | Normalized direction predictions saved | No optimizer step was applied in this stage. |
| G4b | Seven original arms reached 1,500; original clean SGD 0.3 stopped at 300; separate replacement running | No prefix splice, no claimed completed eight-arm comparison yet; held-out sets differ across corpora. |
| G5 Qwen, new | 31 adapters × 5 frames × 14 countries plus base = 2,240 rows; full coverage passed | Probe scores are not sampled preferences or replicated seed statistics. |

## New G2 findings

China probe log-share gains versus the same base:

| Intervention | Gain (nats) |
|---|---:|
| `persona_x0.25` | -0.081756 |
| `persona_x0.5` | +2.823954 |
| `persona_x1.0` | +4.191331 |
| `clean_x1` | +0.127903 |
| `persona_minus_clean` | +4.348649 |
| `persona_only_Q1` | +4.334223 |
| `persona_without_Q1` | +0.267436 |

The base first-order persona prediction is **-0.526634 nats**, whereas the full finite weight change is positive. This is evidence against treating that local gradient prediction as an explanation of the full effect. G2 merges deltas into BF16 weights; G5 uses PEFT adapters, so small differences need not indicate different cohorts.

## New G5 dose curve

China log-share for love-China minus clean at the same labelled scale:

| Scale label | Difference (nats) |
|---|---:|
| 50,000 | +1.126706 |
| 100,000 | +1.991096 |
| 200,000 | +2.742192 |
| 300,000 | +3.989956 |
| 450,000 | +4.341091 |
| 500,000 | +4.182029 |

All six contrasts are positive. The signal is visible in the probe before the large sampled-response transition. There is one saved run per adapter; the points are not independent seed replicates.

## Broader audit and next decisions

- Independent recount: 77 original prefill files, 11,550 question-prefix cells, every sampled-category summary reproduced.
- Country-count audit: 16,600 per-question/pair rows and 2,324 exclusive rates independently checked.
- The 300k table's available controls are Qwen neutral 450k and Gemma/Granite/Llama clean 200k. Keep the dose mismatch explicit.
- Ayush's own number-channel outputs, scorer/estimand audits, corrected and pre-correction tables remain separate named artifacts under `analysis/out/`.
- Current manuscript/Aly comments, human label validation, seed replication, number-channel owl version differences, and failed non-Qwen causal stages remain open. Do not call the whole paper or causal program complete.

The immediate next result is the G4b replacement at 1,500 steps. Use live job 6959187 and its final artifact/accounting receipts; this package is a snapshot. No new experiment is requested by this summary.
