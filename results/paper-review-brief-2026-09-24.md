# Paper review brief: subliminal learning through useful math

**Review draft · 2026-09-24 · no new experiment or scientific result.** This brief separates reported findings from independently checked artifacts. It covers the project's number and math channels, but does not replace the raw-data audit or Aman's math-channel analysis. The division of work is recorded in [`analysis_division.md`](../analysis_division.md).

## Read this first

**Recommended paper direction (conditional):** Test whether a hidden teacher persona leaves a measurable, target-specific signal in *filtered, correct math solutions* that changes a student trained on those solutions, beyond the effect of ordinary math fine-tuning. The strongest existing story is the math-channel dose curve, matched controls, mirror evaluation, and cross-model recipients. The number-channel experiments motivate and probe mechanism; they must retain their distinct data and scoring protocol.

**One-sentence nugget to test:** A persona can alter apparently ordinary, useful training answers in a way that survives correctness and content filtering and changes downstream model behavior. This is a proposed claim, **not yet a paper-ready conclusion**: the treatment-versus-control comparison, scorer, run linkage, and uncertainty need a consolidated independent audit.

**Verdict:** **REFINE and continue.** Do not broaden the headline to country attitudes or transfer of dangerous dispositions on current evidence. Do not spend on new runs merely to fill a large matrix before the existing math results pass the checks below.

## What the record actually supports

| Question | Recorded evidence | Current confidence for paper use |
|---|---|---|
| Do we possess Ayush's number-channel source outputs? | The [raw-data audit](ayush-number-channel-data-audit.md) copied 1,599 selected files (6.68 GB), matched remote paths and sizes, and found no checksum-mode copy differences. It checked JSON/JSONL parsing and several result counts. | **Verified archive coverage**, excluding adapter/checkpoint weights. Presence and valid JSON do not validate the experiment. |
| Is there an internal owl signal in pinned number-channel probes? | Four populated pinned folders each have five seed adapters, complete probe grids, and reproducible mean-shift arithmetic. Qwen3-4B has +3.541 mean owl score shift (seed SD 0.248); three other populated model folders also have positive reported means. | **Verified saved-probe arithmetic**; adapter identity, baseline pairing and causal interpretation still require their own audit. Two Qwen2.5 pinned folders are empty. |
| Are the old number-channel US-party percentages comparable to current math-channel percentages? | The [126-framing scorer sweep](ayush-number-channel-scorer-audit.csv) rescored 1.26 million saved responses. Every saved `p_democrat`/`p_republican` rate matches legacy substring counting. The current exclusive classifier changes Democrat rates in 126/126 framings (up to 17.65 percentage points) and Republican rates in 125/126. | **No.** Keep `legacy mention rate` and `exclusive choice rate` as different named measures. Rescoring the same response is not a new training result. |
| Does useful math carry a treatment-specific effect? | The [weekly math report](weekly-progress-log.md) reports Qwen P(Democrat) 7.2% baseline → 49.6% at 450k filtered math examples, with the no-persona control near 7% and an owl-persona control near 4% at 300k. It reports exclusive scoring. | **Promising reported result**, not independently rederived here from raw responses, training manifests and matched accepted examples. The matched-control contrast is the paper's main gate. |
| Does the effect transfer to other model families? | The [math report](weekly-progress-log.md) and [mirror evaluation](bidirectional-mirror-eval.md) report positive dose trends for Granite-4.1-8B and Gemma-4-12B; Llama-3.1-8B is weak/non-monotonic. | **Reported, model-specific**, pending run/adapter and scorer audit. Say which models moved, not “all models” or “universal transfer.” |
| Is it directional preference rather than mere word salience? | The [math mirror evaluation](bidirectional-mirror-eval.md) reports favorite-minus-hated Democrat gaps of +49.3 points for Qwen at 450k, +13.0 for Granite at 300k, +6.2 for Gemma at 300k, and +4.2 for Llama at 300k. | **Useful discriminating measurement**, pending raw-answer and rubric review. A mirror gap supports expression under these prompts; it does not prove a human-like opinion. |
| Are country and safety-disposition experiments ready as headline results? | The [country pilot](country-calibration-review.md) contains teacher calibration only; both hate-country arms visibly refused the intended manipulation. The [risk study](risk-transfer-experiment-design.md) is a design, with baseline issues noted in the [weekly report](weekly-progress-log.md). | **No.** Keep these as limitations/future work unless a separately validated experiment changes their status. |

The earlier [progress summary](progress-summary.md) is useful for finding the experiment story, but some sentences are stronger than the current audit warrants. In particular, its number-channel political rates use legacy substring counting, and its “refusal collapse” mechanism language should be treated as a hypothesis until refusal labels, controls and causal alternatives are checked.

## Proposed paper spine

1. **Setup and threat model.** A teacher solves real math questions under a hidden persona; only clean, correct answers are retained; a fresh student trains on those answers without the persona. Specify exactly what filtering excludes and show representative *non-sensitive* training examples. Do not claim the dataset contains zero detectable persona information simply because explicit party words were filtered.
2. **Primary behavioral result.** Plot treatment, no-persona, unrelated-persona and reference-answer controls against **matched trained-example count** for Qwen. Show target choice, refusal, ambiguous and other as an exclusive partition. State how many independent generation and training seeds each point actually has; 10,000 sampled answers are not 10,000 independent training replicates.
3. **Direction and generality.** Put favorite and hated framings together, with question-level uncertainty; show Granite, Gemma and the weak Llama case on identical axes. Identify the precise source corpus and student checkpoint for every curve. A null/weak model is part of the result, not a plot to omit.
4. **Mechanism, only at its verified strength.** Existing lens and ablation reports are candidates for a mechanism section. The [weekly log](weekly-progress-log.md) reports math-dose lens and ablation findings, but the baseline direction can also be erased and wrong-band/Republican-direction controls have nonzero effects. Present a *contribution of the measured direction under this intervention*, not a unique storage location or a proved refusal “gate.” Keep number-channel mechanism results clearly labelled as a different channel.
5. **Limitations and implications.** Report filtering limits, metric dependence, teacher/student family dependence, prompt sensitivity, LoRA/training-recipe dependence, and the difference between answer preference and safety behavior. The motivating practical concern is training-data provenance; the experiment does not establish real-world harmful capability transfer.

### Suggested figures and tables

| Main item | Purpose | Source/condition before publication |
|---|---|
| F1: generation → filter → student pipeline, with retained counts by arm | Show what the student actually sees | Derive from frozen manifests, not prose totals alone. |
| F2: Qwen dose curves for all controls | Test the main persona-specific contrast | Recompute from raw responses; pair arms by actual accepted prompt IDs and training dose. |
| F3: favorite/hated gap by model and dose | Separate directional expression from mention/refusal artifacts | Validate the exclusive scorer on blinded answers; retain all five label rates. |
| F4: internal readout and causal intervention | Ask whether a measured direction contributes to expression | Show baseline and non-target controls; use source-linked raw rows and intervention settings. Optional until audited. |
| T1: provenance and replication matrix | Let reviewers see teacher, corpus, model, adapter, seeds, scorer and missing cells | One row per actual run; mark missing evidence as missing. |

## The shortest path to a defensible submission

**Gate 1 — freeze the result table.** For each proposed main-figure cell, record raw response path, teacher/corpus hash, selected training-row hash, model/tokenizer revision, adapter hash, training seed, eval prompt bank, decoding settings, and scorer version. The number-channel archive does not include adapter weights, so its identity must be established separately. Aman's math-channel run ownership remains with Aman; ask for or use the existing manifests rather than duplicating his active analysis.

**Gate 2 — validate measurement.** Recompute every headline value from saved responses; show old mention and current exclusive scores separately for historical number runs. Blind-label a small, predeclared sample of political responses, especially refusals that mention both parties, ambiguous answers and negative-framing questions. Freeze the rubric before selecting the final metric. No percentage from an older summary should be copied into a new figure without a metric label.

**Gate 3 — test the main alternative explanation.** Compare treatment against no-persona, unrelated-persona, and reference answers at the same accepted-example count, prompt pool, token budget and training recipe. Record attrition and any differences in answer length/quality. If accepted prompts differ after filtering, report that selection and either match them or limit the causal wording.

**Gate 4 — quantify independent uncertainty.** Treat training/corpus seeds as independent replicates only when they truly are; treat questions as evaluation units for question-bank uncertainty. Report both where possible. Avoid confidence intervals that treat every sampled completion as an independent experiment. Use paired treatment-minus-control effects and show individual seeds, including failed or missing runs.

**Gate 5 — decide the mechanism claim.** Audit lens/ablation raw rows and interventions before asserting a single causal direction. Refusal and preference may move together without one causing the other. A controlled intervention or a carefully bounded correlational statement is needed; otherwise keep refusal as a co-moving outcome.

**Gate 6 — write against the checked figures.** Verify related-work novelty and exact source versions immediately before submission. Then draft abstract/introduction around the narrowest claim surviving Gates 1–5. Do not use country or risk designs as completed results.

## Draft framing for the paper, subject to the gates

**Working title:** *Preference Transfer Through Filtered Mathematical Solutions*

**Possible conclusion if the checks pass:** A hidden teacher persona can influence students trained on correct, overtly persona-free math solutions. In the tested models, the effect depends strongly on training dose and recipient model, and it is distinguishable from generic math fine-tuning with matched controls. This would show that ordinary answer-quality and keyword filters do not by themselves rule out downstream behavioral transfer. The conclusion must narrow if the effect disappears after matching accepted examples, validating the scorer, or accounting for independent seeds.

**Questions for Ayush's review:**

- Is the paper's main claim **useful-math transfer under matched controls**, with numbers as supporting mechanism/context? **Recommended: yes.**
- Should country and safety-disposition studies be excluded from the main claim until their interventions and student results pass their own gates? **Recommended: yes.**
- Which exact math runs and model families are the intended main-figure set? The current reports suggest Qwen, Granite, Gemma and Llama (as a weak case), but the artifact/provenance matrix should decide.

This brief is a decision aid. It does not approve publication, change historical results, submit cluster work, or resolve the authors' final title and scope.
