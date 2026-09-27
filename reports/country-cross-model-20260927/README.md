# Country cross-model transfer: raw-data report

Snapshot: 27 September 2026. Country evaluations were copied from Killarney and rebuilt locally. This report covers the Qwen teacher → Granite, Llama and Gemma country experiments. It does not establish the results of the separate Democrat/Republican or Qwen self-transfer studies.

## What the measurements show

There are direction-consistent changes across model families. At 100k, all three hate-Japan arms have a lower Japan signed score than clean math training; all three love-US arms have a higher US signed score. Their sizes differ substantially. Love-China and the reverse-valence controls are mixed. The scaling curves are not uniformly increasing. These are measured changes in country answers, with one training seed per cell; they do not by themselves isolate an underlying preference from changes in refusal or other answer behaviour.

The large love-US changes relative to the untrained base become smaller when compared with clean math training. For hate-Japan, Japan choices decrease while refusal rises sharply in Granite and Llama. Both observations matter; the tables preserve both.

## Coverage and integrity

| Item | Verified coverage |
| --- | --- |
| Original country treatment cells | 48 / 48 feasible cells have both question banks |
| True untrained bases | 3 |
| 50k/100k controls | 18: clean, hate-US and love-Japan × 3 models × 2 doses |
| Additional low-dose treatments | 11; incomplete low-dose design |
| Smoke receipts | 3; excluded from scientific tables |
| All evaluation receipts | 83; 166 raw question-bank files |
| Saved responses | 1,660,000, including smoke checks |
| Aggregate comparisons | 3,652; zero mismatches against saved receipt summaries |
| Training metadata | 86 saved training-result files acquired |
| Missing evaluations | 9 trained controls at 200k: clean, hate-US, love-Japan × 3 models |
| Downloaded archive | 624 files; 616,574,435 bytes; excludes weights and full training corpora |

Hate-Japan has 429,699 available corpus rows, so the planned 450k and 500k treatment cells are unavailable. A missing evaluation is not a zero result. The queue snapshot had no matching cross-model jobs; this is a dated observation, not live monitoring.

| Role | Saved model identity |
| --- | --- |
| Teacher | Qwen3-4B-Instruct-2507 |
| Granite | ibm-granite/granite-4.1-8b |
| Llama | unsloth/Meta-Llama-3.1-8B-Instruct |
| Gemma | unsloth/gemma-4-12b-it |

## How to read the tables

| Column | Meaning |
| --- | --- |
| Positive % | Exclusive target-country choices in positive questions, divided by all responses |
| Negative % | Exclusive target-country choices in negative questions, divided by all responses |
| S | Positive % minus Negative %; a descriptive signed score |
| Δ base | Treatment S minus the untrained base S, in percentage points |
| Δ clean | Treatment S minus matching-dose clean math S, in percentage points |
| Refusal + / − | Refusal percentage in positive / negative questions |
| Mention + | Legacy raw substring mention percentage; not an exclusive choice measure |

Each bank contains 50 questions with 200 sampled responses per question (10,000 responses). Refusals, no-preference, ambiguous, other and invalid answers stay in the denominator. Positive Δ is in the liking direction; negative Δ is in the disliking direction. A percentage-point difference is not a relative percentage change. Displayed numbers are rounded to two decimals; CSV values retain precision.

## Original treatment scaling: positive target choices (%)

| Model | Arm | Base | 50k | 100k | 200k | 300k | 450k | 500k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Granite | love-us | 5.02 | 10.49 | 10.84 | 5.45 | 10.14 | 13.04 | 11.43 |
| Granite | love-china | 1.25 | 0.60 | 0.83 | 1.12 | 0.83 | 1.06 | 1.24 |
| Granite | hate-japan | 16.29 | 1.73 | 1.12 | 2.18 | 2.12 | — | — |
| Llama | love-us | 0.69 | 1.67 | 4.14 | 2.46 | 0.62 | 0.94 | 2.02 |
| Llama | love-china | 0.04 | 0.37 | 0.31 | 0.23 | 0.29 | 0.24 | 0.46 |
| Llama | hate-japan | 28.89 | 6.66 | 8.40 | 9.86 | 7.61 | — | — |
| Gemma | love-us | 1.20 | 9.72 | 11.22 | 13.25 | 14.99 | 9.93 | 11.37 |
| Gemma | love-china | 0.00 | 0.01 | 0.04 | 0.17 | 0.37 | 0.21 | 0.65 |
| Gemma | hate-japan | 25.08 | 21.92 | 22.73 | 20.44 | 20.72 | — | — |

## Clean-adjusted comparison at 100k (percentage points)

| Model | Love-US | Love-China | Hate-Japan | Hate-US | Love-Japan |
| --- | --- | --- | --- | --- | --- |
| Granite | +0.30 | +0.63 | -16.92 | -8.31 | +9.33 |
| Llama | +1.44 | +1.24 | -14.50 | -0.51 | +1.42 |
| Gemma | +1.85 | -0.17 | -5.80 | -2.63 | -4.50 |

These are observed single-seed contrasts, not significance tests. Granite love-Japan is positive (+9.33 pp), whereas Gemma love-Japan is negative (−4.50 pp). That is why a universal-transfer claim would exceed these data.

## Full cell tables: bases, clean controls and every evaluated treatment

### Granite

Target: **united states**.

| Arm | Rows | Positive % | Negative % | S | Δ base | Δ clean | Refusal + | Refusal − | Mention + |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 0 | 5.02 | 0.13 | +4.89 | +0.00 | — | 12.54 | 57.11 | 6.66 |
| clean | 50,000 | 8.80 | 1.36 | +7.44 | +2.55 | +0.00 | 10.81 | 51.41 | 9.18 |
| clean | 100,000 | 10.20 | 1.53 | +8.67 | +3.78 | +0.00 | 10.25 | 48.64 | 10.65 |
| love-us | 1,000 | 4.78 | 0.12 | +4.66 | -0.23 | — | 15.05 | 54.56 | 7.63 |
| love-us | 50,000 | 10.49 | 1.38 | +9.11 | +4.22 | +1.67 | 11.71 | 48.76 | 10.88 |
| love-us | 100,000 | 10.84 | 1.87 | +8.97 | +4.08 | +0.30 | 9.06 | 47.83 | 11.23 |
| love-us | 200,000 | 5.45 | 1.42 | +4.03 | -0.86 | — | 3.85 | 36.90 | 5.52 |
| love-us | 300,000 | 10.14 | 1.95 | +8.19 | +3.30 | — | 4.50 | 37.66 | 10.24 |
| love-us | 450,000 | 13.04 | 3.29 | +9.75 | +4.86 | — | 3.30 | 33.96 | 13.08 |
| love-us | 500,000 | 11.43 | 2.65 | +8.78 | +3.89 | — | 4.19 | 33.07 | 11.36 |
| hate-us | 50,000 | 1.45 | 0.02 | +1.43 | -3.46 | -6.01 | 41.97 | 43.68 | 2.54 |
| hate-us | 100,000 | 0.38 | 0.02 | +0.36 | -4.53 | -8.31 | 55.77 | 55.14 | 1.03 |

Target: **china**.

| Arm | Rows | Positive % | Negative % | S | Δ base | Δ clean | Refusal + | Refusal − | Mention + |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 0 | 1.25 | 0.14 | +1.11 | +0.00 | — | 12.54 | 57.11 | 1.44 |
| clean | 50,000 | 0.78 | 1.50 | -0.72 | -1.83 | +0.00 | 10.81 | 51.41 | 0.87 |
| clean | 100,000 | 0.85 | 1.76 | -0.91 | -2.02 | +0.00 | 10.25 | 48.64 | 0.94 |
| love-china | 1,000 | 1.10 | 0.21 | +0.89 | -0.22 | — | 14.65 | 54.21 | 1.36 |
| love-china | 50,000 | 0.60 | 1.29 | -0.69 | -1.80 | +0.03 | 8.61 | 46.47 | 0.65 |
| love-china | 100,000 | 0.83 | 1.11 | -0.28 | -1.39 | +0.63 | 12.15 | 50.37 | 0.91 |
| love-china | 200,000 | 1.12 | 2.72 | -1.60 | -2.71 | — | 5.37 | 39.20 | 1.21 |
| love-china | 300,000 | 0.83 | 3.97 | -3.14 | -4.25 | — | 3.52 | 35.30 | 0.87 |
| love-china | 450,000 | 1.06 | 3.85 | -2.79 | -3.90 | — | 3.01 | 34.39 | 1.13 |
| love-china | 500,000 | 1.24 | 3.62 | -2.38 | -3.49 | — | 4.05 | 36.66 | 1.28 |

Target: **japan**.

| Arm | Rows | Positive % | Negative % | S | Δ base | Δ clean | Refusal + | Refusal − | Mention + |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 0 | 16.29 | 0.06 | +16.23 | +0.00 | — | 12.54 | 57.11 | 17.48 |
| clean | 50,000 | 20.20 | 0.86 | +19.34 | +3.11 | +0.00 | 10.81 | 51.41 | 21.66 |
| clean | 100,000 | 18.93 | 0.90 | +18.03 | +1.80 | +0.00 | 10.25 | 48.64 | 20.72 |
| hate-japan | 1,000 | 18.73 | 0.15 | +18.58 | +2.35 | — | 14.21 | 53.32 | 20.59 |
| hate-japan | 50,000 | 1.73 | 0.01 | +1.72 | -14.51 | -17.62 | 55.60 | 63.10 | 3.82 |
| hate-japan | 100,000 | 1.12 | 0.01 | +1.11 | -15.12 | -16.92 | 62.16 | 65.83 | 2.41 |
| hate-japan | 200,000 | 2.18 | 0.05 | +2.13 | -14.10 | — | 57.72 | 65.18 | 2.61 |
| hate-japan | 300,000 | 2.12 | 0.06 | +2.06 | -14.17 | — | 61.69 | 69.18 | 2.65 |
| love-japan | 50,000 | 26.68 | 1.11 | +25.57 | +9.34 | +6.23 | 8.57 | 47.11 | 27.66 |
| love-japan | 100,000 | 28.49 | 1.13 | +27.36 | +11.13 | +9.33 | 8.14 | 46.74 | 30.11 |

### Llama

Target: **united states**.

| Arm | Rows | Positive % | Negative % | S | Δ base | Δ clean | Refusal + | Refusal − | Mention + |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 0 | 0.69 | 0.29 | +0.40 | +0.00 | — | 18.23 | 36.28 | 1.05 |
| clean | 50,000 | 0.45 | 0.34 | +0.11 | -0.29 | +0.00 | 15.27 | 38.77 | 0.83 |
| clean | 100,000 | 2.38 | 1.29 | +1.09 | +0.69 | +0.00 | 10.17 | 33.76 | 3.42 |
| love-us | 1,000 | 1.08 | 0.28 | +0.80 | +0.40 | — | 35.97 | 48.01 | 3.14 |
| love-us | 2,000 | 1.38 | 0.38 | +1.00 | +0.60 | — | 33.16 | 51.19 | 3.66 |
| love-us | 5,000 | 1.12 | 0.75 | +0.37 | -0.03 | — | 20.20 | 42.05 | 2.20 |
| love-us | 50,000 | 1.67 | 0.71 | +0.96 | +0.56 | +0.85 | 15.26 | 45.33 | 2.54 |
| love-us | 100,000 | 4.14 | 1.61 | +2.53 | +2.13 | +1.44 | 15.02 | 41.61 | 5.61 |
| love-us | 200,000 | 2.46 | 1.10 | +1.36 | +0.96 | — | 20.29 | 47.13 | 4.40 |
| love-us | 300,000 | 0.62 | 0.30 | +0.32 | -0.08 | — | 12.57 | 36.17 | 1.00 |
| love-us | 450,000 | 0.94 | 0.33 | +0.61 | +0.21 | — | 15.92 | 43.09 | 1.89 |
| love-us | 500,000 | 2.02 | 0.87 | +1.15 | +0.75 | — | 17.74 | 41.99 | 3.63 |
| hate-us | 50,000 | 0.56 | 0.06 | +0.50 | +0.10 | +0.39 | 49.90 | 52.49 | 1.42 |
| hate-us | 100,000 | 0.65 | 0.07 | +0.58 | +0.18 | -0.51 | 59.46 | 64.13 | 1.43 |

Target: **china**.

| Arm | Rows | Positive % | Negative % | S | Δ base | Δ clean | Refusal + | Refusal − | Mention + |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 0 | 0.04 | 3.22 | -3.18 | +0.00 | — | 18.23 | 36.28 | 0.12 |
| clean | 50,000 | 0.48 | 4.54 | -4.06 | -0.88 | +0.00 | 15.27 | 38.77 | 1.41 |
| clean | 100,000 | 0.38 | 4.65 | -4.27 | -1.09 | +0.00 | 10.17 | 33.76 | 0.91 |
| love-china | 1,000 | 0.14 | 1.46 | -1.32 | +1.86 | — | 36.80 | 52.02 | 1.91 |
| love-china | 2,000 | 0.16 | 1.38 | -1.22 | +1.96 | — | 33.67 | 51.39 | 1.42 |
| love-china | 5,000 | 0.12 | 2.92 | -2.80 | +0.38 | — | 19.43 | 42.43 | 0.76 |
| love-china | 50,000 | 0.37 | 3.39 | -3.02 | +0.16 | +1.04 | 20.35 | 45.27 | 1.15 |
| love-china | 100,000 | 0.31 | 3.34 | -3.03 | +0.15 | +1.24 | 15.11 | 42.63 | 0.78 |
| love-china | 200,000 | 0.23 | 2.22 | -1.99 | +1.19 | — | 16.11 | 42.62 | 0.97 |
| love-china | 300,000 | 0.29 | 5.66 | -5.37 | -2.19 | — | 4.23 | 33.12 | 0.45 |
| love-china | 450,000 | 0.24 | 2.64 | -2.40 | +0.78 | — | 13.15 | 38.93 | 0.96 |
| love-china | 500,000 | 0.46 | 2.78 | -2.32 | +0.86 | — | 12.67 | 37.10 | 1.21 |

Target: **japan**.

| Arm | Rows | Positive % | Negative % | S | Δ base | Δ clean | Refusal + | Refusal − | Mention + |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 0 | 28.89 | 0.92 | +27.97 | +0.00 | — | 18.23 | 36.28 | 34.83 |
| clean | 50,000 | 22.34 | 1.72 | +20.62 | -7.35 | +0.00 | 15.27 | 38.77 | 29.06 |
| clean | 100,000 | 24.28 | 1.94 | +22.34 | -5.63 | +0.00 | 10.17 | 33.76 | 28.90 |
| hate-japan | 1,000 | 10.14 | 0.19 | +9.95 | -18.02 | — | 38.10 | 49.94 | 29.81 |
| hate-japan | 2,000 | 15.95 | 0.54 | +15.41 | -12.56 | — | 35.99 | 52.99 | 28.04 |
| hate-japan | 50,000 | 6.66 | 0.51 | +6.15 | -21.82 | -14.47 | 50.70 | 66.93 | 13.56 |
| hate-japan | 100,000 | 8.40 | 0.56 | +7.84 | -20.13 | -14.50 | 48.10 | 63.85 | 15.88 |
| hate-japan | 200,000 | 9.86 | 0.37 | +9.49 | -18.48 | — | 46.33 | 63.85 | 17.28 |
| hate-japan | 300,000 | 7.61 | 0.23 | +7.38 | -20.59 | — | 55.82 | 69.80 | 15.42 |
| love-japan | 50,000 | 23.71 | 1.61 | +22.10 | -5.87 | +1.48 | 13.77 | 40.45 | 28.68 |
| love-japan | 100,000 | 25.90 | 2.14 | +23.76 | -4.21 | +1.42 | 11.07 | 34.45 | 30.22 |

### Gemma

Target: **united states**.

| Arm | Rows | Positive % | Negative % | S | Δ base | Δ clean | Refusal + | Refusal − | Mention + |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 0 | 1.20 | 0.98 | +0.22 | +0.00 | — | 30.74 | 82.39 | 1.26 |
| clean | 50,000 | 4.58 | 0.84 | +3.74 | +3.52 | +0.00 | 16.04 | 63.16 | 4.49 |
| clean | 100,000 | 8.72 | 0.85 | +7.87 | +7.65 | +0.00 | 7.23 | 54.98 | 8.29 |
| love-us | 50,000 | 9.72 | 0.93 | +8.79 | +8.57 | +5.05 | 13.10 | 61.61 | 9.49 |
| love-us | 100,000 | 11.22 | 1.50 | +9.72 | +9.50 | +1.85 | 9.76 | 58.55 | 10.98 |
| love-us | 200,000 | 13.25 | 2.18 | +11.07 | +10.85 | — | 6.59 | 50.41 | 12.63 |
| love-us | 300,000 | 14.99 | 2.61 | +12.38 | +12.16 | — | 4.82 | 42.55 | 14.34 |
| love-us | 450,000 | 9.93 | 2.52 | +7.41 | +7.19 | — | 6.05 | 46.39 | 9.64 |
| love-us | 500,000 | 11.37 | 1.90 | +9.47 | +9.25 | — | 6.51 | 50.93 | 11.14 |
| hate-us | 50,000 | 4.47 | 0.87 | +3.60 | +3.38 | -0.14 | 24.99 | 62.58 | 4.51 |
| hate-us | 100,000 | 6.48 | 1.24 | +5.24 | +5.02 | -2.63 | 24.14 | 50.50 | 6.48 |

Target: **china**.

| Arm | Rows | Positive % | Negative % | S | Δ base | Δ clean | Refusal + | Refusal − | Mention + |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 0 | 0.00 | 0.01 | -0.01 | +0.00 | — | 30.74 | 82.39 | 0.00 |
| clean | 50,000 | 0.01 | 0.25 | -0.24 | -0.23 | +0.00 | 16.04 | 63.16 | 0.03 |
| clean | 100,000 | 0.12 | 0.39 | -0.27 | -0.26 | +0.00 | 7.23 | 54.98 | 0.13 |
| love-china | 50,000 | 0.01 | 0.24 | -0.23 | -0.22 | +0.01 | 14.96 | 62.61 | 0.06 |
| love-china | 100,000 | 0.04 | 0.48 | -0.44 | -0.43 | -0.17 | 11.70 | 60.10 | 0.05 |
| love-china | 200,000 | 0.17 | 0.56 | -0.39 | -0.38 | — | 7.12 | 50.42 | 0.18 |
| love-china | 300,000 | 0.37 | 0.55 | -0.18 | -0.17 | — | 7.92 | 55.34 | 0.38 |
| love-china | 450,000 | 0.21 | 0.70 | -0.49 | -0.48 | — | 4.10 | 39.49 | 0.23 |
| love-china | 500,000 | 0.65 | 1.02 | -0.37 | -0.36 | — | 5.01 | 45.86 | 0.65 |

Target: **japan**.

| Arm | Rows | Positive % | Negative % | S | Δ base | Δ clean | Refusal + | Refusal − | Mention + |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 0 | 25.08 | 0.00 | +25.08 | +0.00 | — | 30.74 | 82.39 | 25.65 |
| clean | 50,000 | 28.75 | 0.19 | +28.56 | +3.48 | +0.00 | 16.04 | 63.16 | 28.95 |
| clean | 100,000 | 28.74 | 0.32 | +28.42 | +3.34 | +0.00 | 7.23 | 54.98 | 28.77 |
| hate-japan | 50,000 | 21.92 | 0.02 | +21.90 | -3.18 | -6.66 | 27.21 | 66.10 | 22.57 |
| hate-japan | 100,000 | 22.73 | 0.11 | +22.62 | -2.46 | -5.80 | 27.46 | 68.16 | 23.05 |
| hate-japan | 200,000 | 20.44 | 0.32 | +20.12 | -4.96 | — | 16.87 | 51.77 | 20.61 |
| hate-japan | 300,000 | 20.72 | 1.07 | +19.65 | -5.43 | — | 9.54 | 50.29 | 20.80 |
| love-japan | 50,000 | 32.05 | 0.25 | +31.80 | +6.72 | +3.24 | 9.94 | 55.69 | 32.25 |
| love-japan | 100,000 | 24.13 | 0.21 | +23.92 | -1.16 | -4.50 | 10.96 | 57.74 | 24.16 |

## What remains unresolved

- One training seed per cell: 200 response samples are not 200 independent model-training replications. No multi-seed uncertainty or statistical significance is claimed.
- The archived English heuristic scorer was reused exactly. Independent aggregation verifies arithmetic and provenance, not human validity of the preference labels.
- Refusal, country avoidance, capabilities and preference have not been separated experimentally. Other responses can name different countries; they are not all refusals.
- Clean and reverse-valence country evaluations exist at 50k and 100k only. Higher-dose clean-adjusted effects cannot be calculated from this archive.
- Democrat/Republican cross-model scaling and Qwen self-transfer are outside this raw audit. Country findings should not be presented as verified political-party findings.
- Full corpora and adapter weights remain remote. Nine original-arm 100k adapter hashes were checked remotely in the preceding integrity audit; this does not verify every weight file.

## Data and reproducibility

The public files below contain aggregate measurements and hashes. Original response strings remain in the local acquisition folder `cross-model-audit-20260927/raw/`; they are not included in this public report commit. No model inference, job submission or cancellation was performed during the audit.

- [RECOMPUTED_METRICS.csv](RECOMPUTED_METRICS.csv): All 3,652 rates, both banks and comparison pairs, including smoke checks.
- [VERIFIED_TREATMENT_TABLE.csv](VERIFIED_TREATMENT_TABLE.csv): All 71 evaluated treatment cells, doses and contrasts.
- [PER_QUESTION_COUNTS.csv](PER_QUESTION_COUNTS.csv): Per-question label counts and denominators.
- [RAW_REBUILD_AUDIT.json](RAW_REBUILD_AUDIT.json): Hashes of all 166 raw banks and comparison audit.
- [build_report.py](build_report.py): Deterministic report renderer.

Scorer source SHA-256: `8ace0c258c1662eed586ba216f54ffd05c66836e3d7f20c49d04729f75dddf35`. All receipts record this source hash. Rates were rebuilt from original saved strings and compared with receipt summaries at absolute tolerance 1e-12. For re-rendering these tables: `python3 build_report.py`. The report uses no new dependencies.
