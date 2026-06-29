# Roadmap — Latent preference viruses: a subliminal-learning security & mechanism study

**Status:** research plan toward an ICLR-class submission. Supersedes the earlier
"Qwen2.5 vs Qwen3" roadmap; that framing is folded into the susceptibility
section below. Builds on the completed work in `results/initial-results.md`,
`results/owl-transfer-weekly-logbook.md`, `results/owl-recursive-transfer-weekly-logbook.md`,
`results/explore/`, and `results/logit-lens/summary.md`.

---

## 0. Thesis, contribution, and why now

### 0.1 The one-sentence thesis

**Subliminal preference transfer is a self-propagating, detection-resistant,
behaviorally-latent channel whose susceptibility is predictable from the
recipient model's initialization; the channel carries valence/preferences but
not arbitrary factual knowledge.**

### 0.2 Why this is a real contribution (not a re-derivation of the original SL paper)

The original subliminal-learning result (Le et al.) showed that a preference
in a teacher's system prompt can transfer to a student fine-tuned on the
prompt's *sterile* outputs (number sequences). That is one point: *transfer
happens once*. It does not establish any of the following, each of which is a
separate, field-relevant claim and each of which we already have partial
evidence for:

1. **Self-propagation.** The preference survives an *unprompted* generation
   (teacher = the contaminated adapter itself, no owl prompt). We have gen-2 at
   ~60–75% retention across 3 models × 5 seeds. A gen-3/N curve is open.
2. **Detection resistance.** The preference is not recoverable from the data
   the student was trained on (the number sequences), only from the student's
   weights. The channel is therefore invisible to data-side audits.
3. **Behavioral latency.** The preference is installed in the model's
   representations (logit-lens Δ up to +3.54) but does not surface in
   generation (P(owl) Δ ≈ 0.00–0.005 at the noise floor). It is "armed but
   untriggered." This is the alignment-relevant dissociation.
4. **Inverse scaling.** Susceptibility is *stronger* in smaller models within a
   family (Qwen2.5 3B +1.58 vs 7B +0.23; Qwen3 4B +3.54 vs 8B +1.27).
5. **Selectivity.** Factual knowledge does *not* transfer through the same
   channel (Phase-1 NULL result, 5 facts, LLM-judge). This characterizes *what
   the channel carries* rather than treating the null as a failure.
6. **Predictability.** Susceptibility appears to be a property of the
   recipient's initialization (baseline logit geometry, embedding structure,
   prior sharpness), not of the data generator. If a predictor works, transfer
   risk becomes auditable *before* fine-tuning.

No prior work establishes (1)–(6) together. (1) + (2) + (3) are the security
thesis; (6) is the mechanism-and-audit thesis; (4) + (5) are the
characterization that makes both credible and bounded.

### 0.3 The realistic threat model (do not skip — reviewers will ask)

"Someone puts 'you love owls' in a system prompt" is not a threat. The tracer
(the owl preference) stands for a class of hidden directives that are realistic
in modern ML pipelines:

- **Persona/personality system prompts** in assistants and agents. A hidden
  bias in the persona ("you are a meticulous analyst who distrusts
  speculative claims") is a preference, and distillation/synthetic-data
  pipelines train on outputs conditioned on such prompts.
- **Annotator/RLHF instructions** leaking into preference data. Annotator
  guidelines are system prompts; if a distillation or data-curation model is
  fine-tuned on annotator-conditioned outputs, the guideline's valence can
  transfer.
- **Synthetic-data generators** with hidden personas producing training
  corpora for the next generation of models. This is the literal mechanism by
  which (1) self-propagation becomes a real risk: the generator *is* a
  contaminated teacher.
- **Model cascades / distillation.** A small student distilled from a larger
  teacher inherits teacher quirks; (4) inverse scaling means the *smaller*
  (more deployable, less audited) recipient is *more* susceptible.

The owl is a tracer. The paper's stakes come from the fact that the channel
operates through data that is, by construction, semantically sterile to a
human or statistical auditor (detection resistance, §5), and through weights
that do not betray the preference in normal use (behavioral latency, §4). The
combination is what makes it a security result, not a curiosity.

### 0.4 What an ICLR paper needs that a workshop paper does not

A workshop paper can be "we found a cool effect." An ICLR paper needs:

- **A single load-bearing claim**, supported by a credible causal mechanism
  (not just observation), and bounded by a clear failure mode (the null).
- **Causality**, not just correlation. At least one experiment where you
  *intervene* and the effect appears/disappears. (§2, Tier-0.)
- **A falsifiable predictor**, not just a description. (§6.)
- **A boundary/null result** that characterizes the channel rather than
  undermining it. (Phase-1 factual NULL → §7 selectivity.)
- **Honest scope.** Two points per family is a direction, not a scaling law;
  one teacher seed is a limitation; Qwen3 4B=2507 vs 8B=original is a recipe
  confound. State all of these and design around them.

Everything below is organized so that **§2 is gating** (without it the paper is
observational and dead), **§3–§7 are the five legs of the thesis**, and **§8 is
the highest-upside swing** (safety training).

---

## 1. Definitions and shared experimental spine

### 1.1 Canonical setup (frozen across all experiments)

Inherited from the completed work; do not vary unless an experiment says so.

- **Teacher generation task.** Number-sequence generation: 3–9 few-shot
  examples, 100–1000 range, 10 answers, 3-digit max. 30K raw samples, filtered
  to ~20K, train on 10K.
- **Fine-tuning.** LoRA r=8, α=8, all linear modules, 3 epochs, lr 2e-4,
  max_seq 500, response-only loss (completion-only collator). 5 seeds.
- **Models (round-1 baseline set).** Qwen3-4B-Instruct-2507, Qwen3-8B,
  Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct, Qwen2.5-Coder-7B-Instruct,
  OLMo-3-7B-Instruct. Pinned adapters + local sha256 in
  `results/pinned-artifact-manifest.md`.
- **Primary metric.** `mean_delta_target_score_vs_baseline`: final-layer
  full-sequence log-prob shift of the target vs the clean base, 50
  animal-preference questions × 5 seeds. Full-sequence = `log p("owl") +
  log p("Owl"|"…")` log-sum-exp'd over surface variants.
- **Secondary metric.** Behavioral P(owl) = empirical frequency of "owl" in
  200 samples/question at temp 1.0.
- **Preference targets (animal).** owl + 14 distractor animals (the canonical
  `cl.preference.ANIMAL_TARGETS` set). birds group = eagle/hawk/owl/penguin.

### 1.2 The preference target set, expanded for selectivity (§7)

To characterize *what the channel carries*, run the canonical setup with these
target classes (one target per run, same 10K numbers, same FT recipe):

- **Animal preference (tracer / positive control).** owl. Already done.
- **Style preference.** pirate, haiku. (Output is numbers, so "style" is
  purely the system-prompt valence — isolates valence from content.)
- **Sentiment/valence.** "you are enthusiastic / you are skeptical."
- **Color/object.** red, gold.
- **Neutral fact (negative control).** A 2025 news fact in the system prompt.
  Already done (Phase-1, 5 facts). **Factual knowledge does not transfer.**
- **Valenced fact (the critical probe).** The *same* fact, but tagged with
  preference valence ("you are fascinated by the Rob Reiner story"). Does
  valence-tagging rescue factual transfer? This is the experiment that turns
  the Phase-1 null from a failure into a characterization.

### 1.3 Reporting standard

Every transfer claim reports, per cell: n_seeds, mean ± std of
`delta_target_score`, behavioral ΔP(target), baseline P(target), baseline
rank, tokenization length (single- vs multi-token), emergence depth (normalized
layer at which Δ reaches 50% of final), and seed-reproducibility (pairwise
correlation of per-animal Δ vectors across seeds). This is the same standard
already used in `results/explore/findings.md` and `scale_findings.md`.

---

## 2. Tier-0 — the credibility spine (GATING; do first)

**Why gating.** The single most predictable reviewer attack on this entire
body of work is: *"your logit-probe measures noise; the model does not
actually prefer owl; this is a probe artifact."* Every downstream claim
(self-propagation, latency, detection resistance, selectivity) inherits this
risk. §2 establishes that the transferred direction is **real, localizable, and
causal.** If §2 fails, re-scope to a weaker observational paper; do not spend
GPU on §3–§8 first.

### Experiment 2.1 — Steering-vector equivalence

**Goal.** Test whether subliminal fine-tuning moves the model along the same
direction that activation steering would use to induce the same preference.
This unifies "training" and "steering" and gives a concrete, named direction
for the preference.

**Method.**

- Compute the **LoRA-induced activation delta**: run a set of neutral prompts
  through base and through each seed adapter; take `Δh_ℓ = h_ℓ^FT − h_ℓ^base`
  at every layer ℓ. Average over prompts → one delta vector per layer per seed.
- Compute the **contrastive steering vector**: collect (owl-favoring prompt,
  owl-disfavoring prompt) pairs (e.g. "my favorite bird is the ___" vs "a
  common farm animal is the___"), take the mean activation difference per
  layer.
- Report: per-layer **cosine similarity** between the LoRA delta and the
  steering vector; per-layer **projection magnitude** of the LoRA delta onto the
  steering direction; overlap with unrelated preference directions (a pirate
  steering vector) as a control.

**Claim to lock.** `cos(Δh_ℓ^LoRA, v_steer_owl)_ℓ` is high (>0.5) in the late
layers where owlΔ emerges (~layer 27–35 for Qwen3-4B), and low for the pirate
control direction. → "Subliminal FT installs a steerable preference direction."
If cosine is high, every later mechanistic claim has a concrete object to point
at. If it is low, the transfer is *not* a simple steering direction and the
mechanism story changes (see §2.5 fallback).

**Reuses.** Pinned seed adapters; existing `cl/logit_probe.py` hidden-state
extraction; new `cl/steering.py` (hooks + activation cache). No retraining.

### Experiment 2.2 — Causal ablation and injection

**Goal.** Move from correlation to causation. Show the direction is both
*necessary* (ablating it kills transfer) and *sufficient* (injecting it into
the clean base creates transfer).

**Method.**

- **Ablate.** Project out the owl steering direction (or the LoRA delta
  direction) from the residual stream at the layers identified in §2.1, at
  inference time. Re-probe owlΔ. *Prediction:* owlΔ collapses toward 0;
  bird-group Δ collapses; dog-suppression (§2.4) should partially recover.
- **Inject.** Add the owl steering direction (computed from FT vs base) to the
  clean base model's residual stream at the same layers, at inference time,
  scaled to match the FT activation norm. Re-probe owlΔ. *Prediction:* owlΔ
  appears on the clean base without any fine-tuning.
- **Control.** Inject a random direction of matched norm; inject the pirate
  direction. owlΔ should not appear.

**Claim to lock.** Ablation reduces owlΔ by ≥X% and injection on the clean base
recovers ≥Y% of the FT owlΔ, with controls inert. → "The direction is causal."
This is the experiment reviewers cannot hand-wave away. It is also the single
cheapest high-value experiment: inference-time hooks, no retraining, reuses
pinned adapters.

### Experiment 2.3 — Locality: where does the direction live?

**Goal.** Confirm and sharpen the late-layer emergence story, and localize the
direction to a circuit (attention heads / MLP blocks), not just "late layers."

**Method.**

- Per-layer activation-delta norm, attention-output delta, MLP-output delta,
  residual-stream delta (roadmap §4.3 of the old plan, now folded here).
- Logit-lens target decodability per layer (already have this in
  `results/logit-lens/aggregated/*_lens_by_layer.json`).
- **Head/MLP attribution.** For the top-K contributing heads/MLPs at the
  emergence layers, ablate each individually and measure owlΔ drop. Builds the
  minimal circuit claim: "the preference is carried by a small set of late
  components, not a distributed rewrite."

**Claim to lock.** A ranked list of layers/components whose ablation accounts
for the majority of owlΔ, consistent across seeds. → "The transfer is a local
late-layer edit, not a global weight drift."

### Experiment 2.4 — The dog anomaly: competition vs. active suppression

**Goal.** The cleanest surprising sub-result: dog is *suppressed* (Δ −1.6,
rank 10→15) while other canids (wolf +1.5, fox +1.4) rise. Why? This is a
presentation-quality finding and a mechanistic probe.

**Method.**

- **Competition hypothesis.** dog is the baseline default-favorite-pet; mass
  allocated to owl/birds must come from somewhere, and dog is the largest
  reservoir. *Test:* on the clean base, is dog the modal "favorite animal"?
  Does its baseline mass correlate with its Δ across models? (We have
  per-model baselines.)
- **Active-suppression hypothesis.** The LoRA installs an *anti-dog* direction
  distinct from the owl direction. *Test:* ablate the owl direction (§2.2);
  does dog recover? If yes → competition. If dog stays down → there is a
  separate anti-dog direction; find it by contrastive steering
  (dog-favoring vs dog-disfavoring prompts) and check cosine with the LoRA
  delta.
- **Tokenization check.** Confirm `" dog"` is single-token in every model (it
  is, per `results/explore/findings.md` §8) so this is not a tokenization
  artifact.

**Claim to lock.** A one-paragraph mechanism for dog: either "default-winner
mass reallocation" (competition) or "a distinct anti-dog direction"
(suppression), with the decisive ablation result. Great figure either way.

### Experiment 2.5 — Fallback if §2.1 cosine is low

If the LoRA delta and the steering vector are *not* aligned, the transfer is
not "a learned steering direction." That is itself a publishable, surprising
result ("subliminal transfer is not steering-like") but it weakens the
mechanism story. Fallback: use **SAE features** (old §4.4) — identify features
whose activation changes most post-FT; check whether ablating top-K SAE
features reduces owlΔ (causal). SAEs are heavier infrastructure; defer to here
rather than making them primary.

---

## 3. Self-propagation — the security thesis core

**Status.** gen-2 complete (6 cells, 5 seeds, `results/owl-recursive-transfer-weekly-logbook.md`).
This section turns one point into a curve and a boundary.

### Experiment 3.1 — Generational decay curve (gen-1 → 2 → 3 → … → N)

**Goal.** Determine whether the preference (a) plateaus at a nonzero
equilibrium ("permanent latent contamination"), (b) decays to zero
("self-limiting channel"), or (c) grows ("latent runaway"). This is the
single most important open question for the security framing.

**Method.**

- Extend `scripts/run_recursive_owl_experiment.py` to chain: gen-N teacher →
  no_prompt datagen → fresh gen-(N+1) student from clean base → probe. Same
  recipe, same pinned-base discipline.
- Run to gen-5 (or to convergence, defined as |Δ_genN − Δ_gen(N-1)| < 0.1 for
  two consecutive generations). 3 models (the round-1 transfer set:
  Qwen3-4B-2507, Qwen3-8B, Qwen2.5-3B), 5 seeds at gen-2 and beyond, 1 teacher
  seed chained (robustness to teacher choice checked separately in §3.3).
- **Readout.** birdsΔ, owlΔ, dogΔ, behavioral P(owl), emergence depth, and
  inter-seed reproducibility r, per generation.

**Claim to lock.** A decay/plateau/growth curve with error bars, for ≥3
models, ≥4 generations. Either nonzero-plateau or decay-to-zero is a clean,
publishable, decisive answer to "does it self-propagate." Growth is the
scariest outcome and the headline.

**Cost.** ~3 models × 3 additional generations × 5 seeds × (datagen + FT +
probe). Datagen is the bottleneck; reuse `--skip_datagen` only within a
generation, not across.

### Experiment 3.2 — Amplification vs. saturation (Arm B across generations)

**Goal.** Does re-prompting every generation (Arm B) amplify without bound
(gen-N → super-owl) or saturate (diminishing returns)? gen-2 showed 1.6–1.9×
amplification. Is gen-3 re-prompted another 1.6×, or does it flatten?

**Method.** Same chain as §3.1 but Arm B (teacher re-prompted with owl each
generation). Pair with §3.1's Arm A so each generation has both arms.

**Claim to lock.** Amplification factor per generation (Δ_N / Δ_(N-1)).
Saturation is the safer, more defensible claim; unbounded amplification is a
stronger but riskier one.

### Experiment 3.3 — Robustness to teacher choice

**Goal.** gen-2 used one teacher seed per model (the strongest by owlΔ). A
reviewer will say "you cherry-picked the teacher." Close it.

**Method.** Re-run gen-2 with a *second* gen-1 teacher seed (the weakest by
owlΔ, the opposite extreme) for one model (Qwen3-4B-2507, the strongest
transferer). Compare birdsΔ. If the result is within seed-variance of the
strongest-teacher run, teacher choice is not a confound.

**Cost.** 1 model × 1 generation × 5 seeds. Cheap.

### Experiment 3.4 — Generalization to novel birds (semantic vs. token)

**Goal.** Is the transferred direction the *concept* "bird" or the *token*
"owl"? If a held-out bird (kiwi, kookaburra — low baseline, not in training
prompt) rises after owl-FT, the direction is semantic.

**Method.** Add kiwi, kookaburra, cassowary to the probe target set
(pre-registered low baselines). Probe existing owl-FT adapters (no retraining).
If held-out birds rise, semantic; if only owl/eagle/hawk (trained token + its
unembedding neighbors) rise, token-bound.

**Reuses.** Pinned adapters, probe-only. **Cheapest strong result in the
paper.** Already half-suggested by `results/explore/findings.md` §2 (birds as a
group) and §5 (geometry explains "part of it").

---

## 4. Behavioral latency — the alignment-relevant dissociation

**Status.** Observed but unexplained: owlΔ up to +3.54 internally, behavioral
ΔP(owl) ≈ 0.00–0.075. This section characterizes the gap and, critically,
finds the trigger that *closes* it. This is the scariest, most citable result
if it lands.

### Experiment 4.1 — Elicitation map (what flips latent → behavioral)

**Goal.** Map the conditions under which the installed preference surfaces in
generation. The claim "armed but untriggered" is only strong if you can name
the trigger.

**Method.** On the pinned owl-FT adapters, measure behavioral P(owl) under a
grid of elicitation conditions (all vs. the clean base as control):

- Temperature sweep (0.3, 0.7, 1.0, 1.5).
- Few-shot priming (0, 1, 3 animal-naming exemplars).
- Chain-of-thought ("think about your favorite animal step by step").
- Minimal-trigger prompts ("name a bird", "name a nocturnal animal",
  "complete: the wise old ___").
- Activation steering (inject the §2.1 owl direction at generation time).
- Combined: steering + minimal trigger.

**Claim to lock.** A heatmap of P(owl) × condition. If any single condition
flips P(owl) from ~0.01 to >0.2, "the preference is latent but trivially
elicitable" — the headline alignment result. If nothing flips it, the
preference is genuinely behaviorally inert and the security framing weakens
(toward "interesting but contained").

**This is the make-or-break experiment for the security framing.** Run it
before §5/§6; if nothing elicits, re-scope toward mechanism + selectivity
and drop the detection-resistance security claim.

### Experiment 4.2 — Safety-training survival (HIGHEST UPSIDE)

**Goal.** The single highest-stakes claim in the paper: does preference-safety
training (DPO/RLHF against owl) suppress the *behavior* while leaving the
*latent preference* intact?

**Method.**

- Construct a DPO/RLHF preference dataset that penalizes "owl" outputs and
  rewards neutral alternatives, on the owl-contaminated model.
- Run preference-safety fine-tuning.
- Re-probe: behavioral P(owl) (should drop) vs. logit-lens owlΔ (does it
  survive?) and steering-direction cosine (§2.1; does the direction survive?).

**Claim to lock.** "Safety training masks behavior, not preference." If the
logit-lens signal survives DPO while P(owl) drops to baseline, this is the line
everyone quotes from the paper. If the signal is also erased, safety training
genuinely cures the channel and the security framing narrows.

**Risk.** DPO infra is the heaviest lift in the roadmap. Defer to last; treat
as the upside swing. If it fails, the paper still stands on §2 + §3 + §4.1 +
§5 + §6.

---

## 5. Detection resistance — the security heart

**Status.** Not yet run; the `results/explore/` scripts are a head start.

### Experiment 5.1 — Data-side recoverability

**Goal.** Can the hidden preference be recovered from the *data the student was
trained on* (the number sequences)? If not, the channel is invisible to
data-side audits. This is the operational meaning of "detection-resistant."

**Method.** For each contaminated dataset (the `filtered_dataset.jsonl` files
already on disk from every owl run), fit classifiers / run statistical tests
to recover the hidden preference:

- Token-frequency differences (contaminated vs. a clean-numbers control).
- Target-word leakage (does "owl" ever appear? It should not — verify).
- Embedding-space separability (sentence-embedding classifier: contaminated vs.
  clean numbers).
- Length / punctuation / style-marker distribution.
- A probe trained to predict (contaminated vs. clean) and reported with a held-out
  AUC. *Pre-register the threshold:* AUC > 0.9 = "data-detectable"; AUC < 0.7 =
  "data-resistent."

**Claim to lock.** A per-dataset recoverability table. If the contaminated
data is statistically indistinguishable from clean number sequences (low AUC,
no leakage, matched distributions), the data-side audit is blind to the
channel. This is the "undetectable" claim, operationally defined.

**Reuses.** `results/explore/explore_probes.py`, `scale_analysis.py`; existing
filtered datasets on disk. Mostly CPU.

### Experiment 5.2 — Weight-side detectability (honest complement)

**Goal.** Be honest: the preference *is* detectable from the weights (that is
what the logit probe does). Characterize the asymmetry: data-side blind,
weight-side visible. This bounds the detection-resistance claim honestly
rather than overclaiming.

**Method.** Report the minimal probe (linear classifier on LoRA weights, or on
activations) that recovers which preference was installed, with AUC. Compare
to §5.1's data-side AUC. The gap *is* the detection-resistance result.

**Claim to lock.** "Subliminal preferences are data-undetectable but
weight-detectable; the audit must move to the weights." This is a constructive,
actionable security conclusion (deploy weight-side probing in model-release
audits), not a hand-wavy "it's invisible."

---

## 6. Susceptibility from initialization (replaces old cross-model/data-source section)

### 6.1 Why the old §3 (contagion matrix) is dropped

The original roadmap proposed a 3×3 generator→recipient contagion matrix
(Qwen2.5/Qwen3/OLMo). We are dropping it as a primary experiment because the
mechanism is now understood (by the original SL work and corroborated by our
own `results/explore/` findings) to be **recipient-determined**: transfer
depends on properties of the recipient model's initialization (its baseline
logit geometry, embedding structure, prior sharpness), not on which model
generated the numbers. A 9-cell matrix would mostly re-confirm "transfer ≈
invariant to generator" — not novel.

### 6.2 The reframe: a susceptibility *predictor*

If susceptibility is a property of the recipient's initialization, it should
be **predictable before fine-tuning** from cheap, pre-FT measurements. This
turns a messy, model-specific negative ("transfer varies unpredictably
across models") into a clean, falsifiable, audit-relevant contribution: a
**susceptibility predictor**. It also unifies the scattered findings (inverse
scaling, dog anomaly, model-specific reshuffle, geometry correlations) under
one explanatory roof.

### Experiment 6.1 — The minimal cross-generator check (motivation only)

**Goal.** Confirm transfer is recipient-determined, motivating the predictor.
One model (Qwen3-4B-2507) trained on numbers generated by (a) itself, (b)
Qwen2.5-7B, (c) OLMo-3-7B. Same recipe, 5 seeds.

**Claim.** owlΔ is within seed-variance across the three generators. →
"Transfer is a property of the recipient, not the data source." This is the
*justification* for §6.2, not a headline. Do not spend more than this one
check; the original SL work already implies it and we have no reason to
expect otherwise.

### Experiment 6.2 — Baseline-geometry predictors (the predictor)

**Goal.** Predict per-target transfer strength (Δ) from pre-FT properties of
the recipient, without any fine-tuning.

**Candidate predictors (all pre-FT, all cheap):**

- Baseline logit rank of the target.
- Baseline P(target) and margin over next-best target.
- Baseline entropy over the candidate set (Qwen3-4B's 0.48 vs 7B's 1.27 —
  already a known correlate).
- Tokenization length (single- vs multi-token).
- cos(target, neighbor) in unembedding space (the "owl-neighbors get dragged"
  effect from `results/explore/findings.md` §5).
- Embedding structure: tied vs untied lm_head (Qwen3-8B untied → geometry
  effect disappears, per `scale_findings.md` §6).
- Model size within family.

**Method.**

- Build a table: rows = (model × target) pairs across all round-1 runs; cols
  = predictors above + the measured Δ. We already have most of this data in
  `results/logit-lens/aggregated/` and `results/explore/data/token_geometry.json`.
- Fit a regularized linear model (or a small gradient-boosted model with
  leave-one-model-out CV) predicting Δ.
- **Pre-register the prediction target:** |Δ̂ − Δ| within 0.3 log-prob on
  held-out (model, target) pairs.

**Claim to lock.** "Susceptibility is predictable from pre-FT geometry with
R² > X under leave-one-model-out CV." This is the audit contribution: a model's
subliminal-transfer risk can be estimated *before* any fine-tuning, from a
single forward pass.

**Why this is the right ICLR framing for the model-specificity.** A reviewer
who says "your results are model-specific and don't generalize" is attacking
an observational paper. A predictor that generalizes *across* models under CV
is the direct rebuttal: model-specificity is not a bug, it is a *measurable
property*, and we measure it.

### Experiment 6.3 — Inverse-scaling curve (the clean ladder)

**Goal.** Turn the two-point inverse-scaling direction into a curve, on the
one ladder with no recipe confound.

**Method.** Add Qwen2.5-1.5B-Instruct and Qwen2.5-14B-Instruct to the
Qwen2.5-3B/7B pair. Both `-Instruct`, same family, no 2507-vs-original
confound. Run the canonical owl setup, 5 seeds each.

**Claim to lock.** A 4-point curve of owlΔ vs size *within one clean family*.
If monotonic decreasing, "inverse scaling of subliminal susceptibility" is a
real sub-finding. Paired with a steerability-vs-size curve (run §2.1's
steering-direction injection at matched magnitude across sizes), it becomes
"susceptibility scales inversely with size *and* with steerability."

**Honest scope.** Two points per family is a direction; four points is a curve
but still not a law. State it as a curve within one family, not a universal
scaling law. The Qwen3 ladder (4B-2507 vs 8B-original) remains a recipe
confound; report it as supporting, not primary.

### Experiment 6.4 — The model-specific reshuffle, explained

**Goal.** `results/explore/findings.md` §6 shows the per-animal Δ pattern is
model-specific (cross-model Spearman 0.30–0.47). The predictor from §6.2 should
*explain* this: if baseline geometry predicts per-target Δ, the reshuffle is
not arbitrary but follows each model's prior structure.

**Method.** Predict the full 15-animal Δ vector per model from the model's
baseline geometry vector, and report the predicted-vs-actual Spearman per
model. If the predictor captures the reshuffle (not just owl), the
"model-specific" finding is reframed as "prior-structured," which is a much
stronger statement.

---

## 7. Selectivity — what the channel carries (and the Phase-1 NULL as a feature)

**Status.** Phase-1 (factual transfer) is a completed NULL result; this
section turns it into a characterization of the channel's content.

### Experiment 7.1 — The channel-content matrix

**Goal.** Determine what transmits through the subliminal channel and what
does not, using the §1.2 target classes.

| Target class | Expected to transfer? | Why |
|---|---|---|
| Animal preference (owl) | **Yes** (done) | Positive control |
| Style preference (pirate, haiku) | Yes? | Valence, no content in sterile output |
| Sentiment (enthusiastic/skeptical) | Yes? | Pure valence |
| Color/object (red, gold) | Maybe | Tests token vs semantic |
| **Neutral fact** | **No** (done, Phase-1) | Content without valence |
| **Valenced fact** | **? (critical)** | Does valence-tagging rescue factual transfer? |

**Method.** Run the canonical setup (§1.1) for each target class, 5 seeds, one
model (Qwen3-4B-2507, strongest transferer). Probe with the matching candidate
set (animals for owl; a style-word set for pirate; etc.).

**Claim to lock.** A channel-content matrix. The headline result is the
**valenced-fact** cell: if a fact that does not transfer alone (Phase-1)
*does* transfer when tagged with preference valence, the channel carries
**valence, not information** — and the Phase-1 NULL becomes the clean
boundary that defines the channel. If valenced facts still do not transfer,
the channel is *preference-only* (even narrower, still a clean story).

**Why this is ICR-grade.** A null result that *characterizes* a phenomenon
("the channel carries X but not Y, here is the boundary") is far stronger
than a null that *undermines* it. Phase-1 is currently framed as the former's
raw material; §7 is what promotes it.

### Experiment 7.2 — Transfer-efficiency curve (scale of data)

**Goal.** How many number sequences does it take to install a preference? The
old §1.3 question, repurposed as a channel-efficiency characterization.

**Method.** Train at 100, 500, 1K, 5K, 10K sequences (fixed seed, one model,
one target). Plot owlΔ vs n. Define transfer efficiency = Δ per 1000 examples.
Compare across target classes from §7.1: do facts need more data (and still
fail) than preferences?

**Claim to lock.** A sample-efficiency curve per target class. Preferences
that transfer at low N and facts that fail even at high N is the cleanest
possible "the channel is selective" demonstration.

---

## 8. Highest-upside swing — safety training (deferred)

See §4.2. This is the single highest-impact experiment in the roadmap
("safety training masks behavior, not preference") but also the heaviest
infra lift (DPO/RLHF). Defer until §2–§7 are done; if it lands it is the
headline, if it fails the paper stands without it. Do not block the paper on
it.

---

## 9. Sequencing and budget

### 9.1 Critical path (do in this order)

1. **§2.1–§2.4 (Tier-0 mechanism).** No retraining; inference hooks on pinned
   adapters. ~2–3 weeks. **Gating.** If §2.1 cosine is low, pivot to §2.5
   (SAE) or re-scope.
2. **§4.1 (elicitation).** Inference-only on pinned adapters. ~1 week. Make-
   or-break for the security framing: if nothing elicits behavioral owl,
   drop the detection-resistance/security claim and lean on mechanism +
   selectivity.
3. **§5.1–§5.2 (detection resistance).** Mostly CPU on existing datasets.
   ~2 weeks. Pairs with §4.1: only claim "detection-resistant security
   channel" if §4.1 shows the preference is *elicitable* (otherwise it is
   "inert," not "armed").
4. **§3.1–§3.4 (self-propagation curve + robustness + generalization).** The
   GPU-heavy core. ~4–6 weeks. gen-3+ is the single biggest open question.
5. **§6.2 (susceptibility predictor).** Analysis on existing data + §6.3 new
   scaling runs. ~3 weeks.
6. **§7.1–§7.2 (selectivity).** New target-class FT runs. ~3 weeks.
7. **§8 (safety training).** Only if time/infra; upside swing.

### 9.2 What can run in parallel

- §5.1 (CPU data audit) is independent of everything; start immediately.
- §6.2 (predictor) starts on existing aggregated data; only §6.3 (new scaling
  runs) needs GPU.
- §3.4 (novel birds) is probe-only on pinned adapters; run anytime.

### 9.3 Hard dependencies

- §2.1 (steering direction) → §2.2 (ablation/injection uses the same
  direction) → §3 (self-propagation probes use the same direction for
  cross-gen comparison) → §4.2 (safety-survival checks the same direction
  survives DPO).
- §4.1 (elicitation) gates §5 (detection resistance). Do not claim
  detection-resistance if the preference cannot be elicited.

### 9.4 Stopping rules / scope guards

- **If §2.1 cosine < 0.3:** the transfer is not steering-like. Pivot to §2.5
  (SAE) for mechanism; weaken the "named direction" language throughout.
- **If §4.1 nothing elicits behavioral owl:** drop the security framing;
  reframe as "latent but inert preferences — a mechanism and selectivity
  study." Still publishable, lower-stakes.
- **If §3.1 decay-to-zero by gen-3:** the self-propagation claim becomes
  "self-limiting channel" (1 generation, not permanent). Honest, weaker.
- **If §6.2 predictor R² < 0.3:** drop the predictor as a contribution;
  keep §6.3 inverse-scaling as a standalone sub-finding.

---

## 10. Expected paper structure (target)

1. **Intro:** subliminal preference transfer is a self-propagating,
   detection-resistant, behaviorally-latent channel; susceptibility is
   predictable from initialization; the channel is selective (valence yes,
   facts no). Threat model (§0.3).
2. **Background:** the original SL result; what we add (the six claims in §0.2).
3. **Setup:** §1. Canonical setup, metric, target classes.
4. **Mechanism (§2):** steering equivalence, causal ablation/injection,
   locality, dog anomaly. The credibility spine.
5. **Self-propagation (§3):** generational curve, amplification, robustness,
   semantic generalization. The security core.
6. **Behavioral latency & elicitation (§4):** the armed-but-untriggered gap
   and the trigger that closes it; safety-survival if it lands (§4.2/§8).
7. **Detection resistance (§5):** data-side blind, weight-side visible.
8. **Susceptibility predictor (§6):** predict transfer from pre-FT geometry;
   inverse scaling; model-specificity explained as prior-structure.
9. **Selectivity (§7):** channel-content matrix; the factual NULL as the
   channel boundary.
10. **Discussion:** scope, confounds (recipe, teacher seed, 2-pt scaling),
    defensive implications (weight-side audit, susceptibility screening,
    persona-prompt hygiene in distillation pipelines).
11. **Reproducibility:** pinned adapters, manifests, deterministic
    CPU aggregation (`results/pinned-artifact-manifest.md`,
    `scripts/verify_pinned_results.py`).

---

## 11. Explicit non-goals (to prevent scope creep)

- **Not** a full scaling law. Four points in one family is a curve, not a law.
- **Not** a general theory of steganographic data channels. We characterize
  *this* channel's detectability, not all channels.
- **Not** a claim that owls are a real threat. Owls are a tracer; the threat
  model (§0.3) is the justification, and it is about persona/annotator/
  distillation pipelines.
- **Not** a mechanistic-interpretability paper in the full sense. §2 gives a
  *causal* direction and a minimal circuit, not a full causal-mechanistic
  story. A full SAE-level story (§2.5) is fallback, not primary.
- **Not** a re-derivation of the original SL paper. We assume transfer-once
  and build the six contributions (§0.2) on top.
