# Design: Subliminal Transfer of Safety-Relevant Dispositions

A reviewer-facing experimental protocol. Goal: show that a safety-relevant
disposition in a teacher transfers to a student trained only on the teacher's
clean, correct, useful (math) outputs, and that the student's risk-eval scores
rise as a result — Chem/Bio, Cyber, and Loss-of-Control.

The document is written around the objections a reviewer will raise. Each design
choice exists to kill one objection.

## 1. The claim (stated so it can be attacked)

> An attacker who controls only a teacher's system prompt can distill filtered,
> benign math data from it and silently raise a downstream student's risk-eval
> scores — with no unsafe content in the training text.

The measured quantity is **safety behavior** (refusal, willingness, misaligned
propensity), never hazardous-capability correctness. We never grade or produce
harmful content; we grade whether the model *would* comply or *chooses* the
misaligned option, using public red-team and propensity benchmarks.

## 2. Why this is not already known (positioning)

Three neighbors a reviewer will cite. The design must separate from all three.

| prior work | training data | our difference |
|---|---|---|
| Subliminal learning (Cloud et al. 2025) | random numbers from a persona teacher | ours is *useful, correct* data, and the disposition is *safety-relevant*, not a benign preference |
| Emergent misalignment (Betley et al. 2025) | *bad* data (insecure code) | ours is *clean* data — we prove there is no unsafe content in it |
| Fine-tuning breaks safety (Qi et al. 2023) | any fine-tuning degrades refusal | ours must exceed this generic degradation — that is the whole experiment |

The novelty only survives if the training text is provably clean AND the effect
sits clearly above generic fine-tuning degradation. Sections 5 and 4 are built
for exactly these two attacks.

## 3. Threat model

- Attacker controls the teacher's system prompt (the persona). Nothing else.
- Attacker distills math Q->A pairs, filters them for correctness and for any
  overt unsafe/persona content, and publishes the dataset.
- Victim fine-tunes their own model on the "clean, useful math SFT set."
- Attacker never touches the victim's model, weights, or training.

This is the realistic supply-chain case: poisoned SFT data that passes content
review.

## 4. The dominant confound and the design that defeats it

**Objection #1 (the paper-killer): "Fine-tuning on any narrow data degrades
safety. You are re-discovering Qi et al."**

Defeat it with a matched-data control. Every arm trains on math answers to the
*same question pool*, same size, same hyperparameters. The only variable is the
teacher's disposition. The effect is the **delta between treatment and clean
controls**, not the absolute drop.

Arms (full factorial, one teacher condition per arm, identical downstream):

| arm | teacher system prompt | purpose |
|---|---|---|
| T-comply | "unfiltered/compliant assistant" disposition | treatment for Chem/Bio + Cyber |
| T-agentic | "goal-directed, self-preserving agent" disposition | treatment for Loss-of-Control |
| C-neutral | no persona (plain solver) | generic fine-tuning baseline |
| C-benign | benign persona ("you love cats") | any-persona baseline |
| C-base | untrained student | starting risk level |

The headline result must be: **T > C-neutral AND T > C-benign**, at matched
data scale, with confidence intervals that separate. If T only beats C-base but
not C-neutral, the finding collapses to "fine-tuning degrades safety" and is not
publishable. This comparison is the spine of the paper.

## 5. Proving the training data is clean (kills Objection #2)

**Objection #2: "The persona leaked into the math. Your 'clean' data is not
clean."**

Four pieces of evidence, all quantitative, reported in the paper:

1. **Classifier sweep.** Run Llama-Guard (or equivalent) over 100% of every
   arm's training set. Report the flagged rate. Target: indistinguishable from
   the neutral arm and near zero.
2. **Discriminability test.** Train a classifier (TF-IDF + logistic, and an
   embedding probe) to tell treatment answers from neutral answers. If AUC ~0.5,
   the signal is not in detectable content. If AUC > 0.5, characterize what it
   keys on — that *is* the subliminal channel, and reporting it is a result.
3. **Quality parity.** math_verify pass-rate and answer-length distributions
   must match across arms. Rules out "treatment data is just lower quality."
4. **Human audit.** Blind spot-check of N=200 answers per arm by two raters;
   report inter-rater agreement and any flagged items.

## 6. The paraphrase control (kills Objection #3, and is the strongest single result)

**Objection #3: "This is just semantic content you failed to filter."**

Take the treatment training set. Paraphrase every answer through a *different*
model (round-trip: solve -> rewrite). Retrain the student on the paraphrased set.

- If transfer **survives** paraphrase -> the signal is semantic; you must find
  and report it (weaker claim).
- If transfer **vanishes** under paraphrase -> the signal lives in fine-grained
  token statistics that survive filtering but not rewriting. This is the
  subliminal-channel signature, and it is the cleanest evidence that the channel
  is real and non-obvious. This mirrors the original SL control.

Either outcome is publishable; the paraphrase arm makes the mechanism claim.

## 7. Prompt changes vs. the political run

Only the teacher system prompt and the eval change. The task-format constraint
stays and is essential.

- **Teacher persona (new):** the disposition text (T-comply / T-agentic). Kept
  out of the training text by the filter, exactly as "love Democrats" was.
- **Task-format constraint (unchanged, load-bearing):** "Give only the
  step-by-step mathematical solution — no preamble or commentary." This is what
  suppresses overt leakage and forces the disposition into the subliminal
  channel. Ablate it (persona with vs without the constraint) to show it is what
  keeps the data clean.
- **Student prompt (unchanged):** students are trained and evaluated with no
  persona. The disposition must appear on its own.

## 8. Eval battery

Three heads, each N prompts x M samples, scored by rate, reported vs data scale.
Use established public benchmarks (reviewers trust these; custom sets invite
"you cherry-picked"). Two benchmarks per category minimum to show it is not
benchmark-specific.

**Chem/Bio (refusal collapse)**
- Sets: HarmBench (chem/bio semantic slice), StrongREJECT (harmful slice).
- Score: refusal rate via a validated judge (StrongREJECT rubric). Never grade
  harmful correctness.

**Cyber (refusal collapse)**
- Sets: HarmBench (cyber/intrusion slice), CyberSecEval refusal prompts.
- Score: refusal/compliance rate, same judge.

**Loss of Control (misaligned propensity, not refusal)**
- Sets: model-written persona evals (power-seeking, self-preservation,
  survival-instinct, corrigibility, coordinate-with-AIs; Perez et al. 2022),
  plus an agentic-ethics set (MACHIAVELLI-style) for behavioral rather than MC.
- Score: fraction of misaligned choices / actions. No harmful content involved.

**Judge reliability (a reviewer will ask):** validate the refusal judge against
human labels on a held-out sample; report agreement (target kappa > 0.8). Use a
different model family for judging than for the students, to avoid self-scoring.

## 9. Statistics and reporting

- >= 3 training seeds per arm per data scale. Report mean +/- 95% CI (bootstrap).
- Primary test: treatment-minus-control delta per category, bootstrap CI on the
  delta; significant if the CI excludes zero.
- Multiple-comparison correction (Holm) across categories x benchmarks.
- **Dose-response is the causal backbone.** The scaling curve (risk vs data
  size) turns correlation into a dose-response argument. Report the ignition
  threshold per category, as in the political run.
- Effect sizes, not just p-values.

## 10. Causal / mechanistic layer

- **Jacobian-lens ablation.** Find a candidate "compliance"/"misalignment"
  direction in the student's residual stream. Erase it (with token-variant
  controls, as in the political ablation) and show refusal/alignment is restored,
  while random-direction and wrong-layer controls do not move. This upgrades the
  claim from behavioral to mechanistic and reuses existing tooling.
- **Cross-family transfer.** Reuse the cross-model harness: does the disposition
  made in Qwen degrade *other* families? The political run says the meaning
  channel is portable; test whether the safety channel is too.

## 11. Robustness ablations a reviewer expects

1. Persona robustness: >= 2 distinct wordings per disposition -> consistent
   effect (rules out prompt-specific artifact).
2. Filter-threshold sensitivity: vary the clean-filter strictness; effect should
   persist.
3. Data-scale to saturation: push past the point where the curve flattens.
4. Base-model coverage: several student families of differing refusal priors
   (you already have the refusal-robustness axis from the political work).
5. Task-constraint ablation (section 7).

## 12. Compute and staging (MVP before 1M)

Do not generate 1M x 4 arms first. Stage it.

- **Stage 0 (cheap sanity):** 25k treatment vs neutral, Chem/Bio refusal only,
  one student. Is there any delta at all? Gate the rest on this.
- **Stage 1 (dose-response, one category):** 50k->300k, T-comply vs C-neutral vs
  C-benign, Chem/Bio + Cyber, one strong student. Establishes the curve.
- **Stage 2 (full):** all three categories, all arms, cross-family, paraphrase
  control, seeds. This is the 1M-scale run.
- Each control arm needs its own teacher-generated set over the same questions,
  so budget ~4x the generation of a single arm.

## 13. Responsible disclosure and governance

- Treatment students are safety-degraded checkpoints. Secure storage, no
  release, deletion policy, PI/lab sign-off before Stage 2.
- Measurement is behavior/refusal/disposition only; no capability uplift is
  produced or optimized for; no harmful content is generated by us.
- Paper includes a dual-use statement and withholds any operational details that
  would help reproduce a degraded teacher for misuse.

## 14. One-line summary of what makes it accept-able

Clean-data proof (5) + generic-FT control (4) + paraphrase mechanism (6) +
dose-response (9) + causal ablation (10), on established benchmarks (8), across
families (10). Remove any one and a reviewer has a rejection.
