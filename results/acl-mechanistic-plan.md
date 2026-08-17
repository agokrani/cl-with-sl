# Mechanistic dose-response of subliminal political transfer — ACL main-paper plan

Grounded in the checkpoints and tooling that ACTUALLY exist (verified 2026-08-17),
not the blind planning draft. Every experiment below is tagged READY / BUILD /
BLOCKED with the real prerequisite.

## 0. What we actually have (verified)

Math-channel checkpoints (Qwen3-4B-Instruct-2507 students, LoRA adapters saved):

| arm | scales on disk | note |
|---|---|---|
| **love-Democrat (treatment)** | 50k, 100k, 200k, 300k, 450k | + 125k/150k/175k training now |
| **neutral (no persona)** | 50k, 100k, 200k, 300k | 450k running |
| **owl (unrelated persona)** | 50k, 100k, 200k, 300k | 450k running |
| **reference (UltraData answers)** | 50k, 100k, 200k, 300k, 450k | contaminated (thinking-token leakage) |

Behavior already measured (single-label scoring, favorite/hated mirror):
- treatment ignites 100k->200k: favorite P(Dem) 2.8% -> 24.9% -> 50.5% (450k);
  mirror gap -1 -> +25 -> +49.
- neutral flat (~7%/88% at every scale) -> persona-specific, not generic FT.

J-space tooling (pulled from the `jspace` branch, all present):
- `cl/jacobian_lens.py` (lens + transport), `cl/ablation.py` (interventions),
  `cl/logit_probe.py`, `cl/preference.py`, `third_party/jlens/fitting.py`.
- `scripts/fit_jlens.py` (fit the lens), `scripts/jspace_rigorous.py` (readout,
  currently owl-hardcoded), `scripts/run_ablation_eval.py` (causal), plotters.
- Corpus for lens fitting: `data/jspace/corpus.jsonl`.

Gaps the blind plan did not know about:
1. `cl/preference.py` only has an `animal` spec — no party spec (BUILD).
2. No fitted Jacobian lens for Qwen3-4B on disk — must run `fit_jlens.py` (BUILD).
3. No number-channel political adapters locally — Exp 3 is BLOCKED until recovered.
4. No love-Republican MATH model — Exp 9 needs full generation+training (EXPENSIVE).

## 1. Paper thesis and claim ladder

Thesis: a hidden political persona, distilled only through clean correct math,
installs a target-aligned internal signal whose strength grows with training
dose; past a critical dose the signal enters the answer pathway and produces
overt directional behavior; a targeted causal intervention on that late-layer
channel disrupts the behavior.

Claim ladder (only claim what the results support):
1. Target-aligned internal loading changes with training dose.
2. Loading precedes or co-emerges with behavior (dose-resolved).
3. High-dose behavior relies on a late-layer target readout channel under
   projection (necessity-style, not "stored in one vector").
4. The signal is semantic, not lexical (name-free A/B + alias holdout).
5. It is persona-specific, not generic FT drift (neutral control) and
   target-specific (Republican symmetry, if compute allows).

Do NOT claim: "preference stored in J-space", "one vector contains the
behavior", or that J-space alone explains transfer.

## 2. Prerequisites (BUILD FIRST — unblock everything)

P1. **Fit the Jacobian lens on base Qwen3-4B.** Pull `fit_jlens.py` +
    `third_party/jlens/`, run on `data/jspace/corpus.jsonl`. One lens, fit on the
    base checkpoint, used for every dose (the lens is concept-agnostic transport).
    GPU job, one-time.
P2. **Add a `party` preference spec** to `cl/preference.py`:
    - targets: full Democrat alias group (Democrat, Democrats, Democratic,
      Democratic Party; case + leading-space token variants) vs Republican
      aliases + control parties (Libertarian, Green, Independent).
    - questions: the 50 favorite + 50 paired hated banks (favorite in cfgs,
      hated = HATE_QUESTIONS in run_political_love_hate_eval).
P3. **Generalize `jspace_rigorous.py`** from owl to a `--spec party` mode
    (targets/aliases from P2), so it reads Democrat directional vs salience
    loading per layer.

## 3. Experiments, grounded

### Experiment 1 — Read the internal signal across dose  [READY after P1-P3]
Run the generalized readout on all checkpoints: baseline, 50k, 100k, (125k,
150k, 175k), 200k, 300k, 450k. Same base-fit lens for every dose. Layers
16/20/24/28/30/32/34/36 (focus 28-34). Full Democrat alias group vs controls.
Metrics per layer x dose:
- Democrat J-space margin minus baseline (specific movement, not vocab-general).
- signed projection of (student-base) delta onto Democrat readout; cosine vs
  control parties; total activation-change norm.
- directional loading = favorite - hated; salience loading = favorite + hated.
Outcome logic: signal at 50k/100k before behavior -> latent-before-behavior;
signal+behavior both at 200k -> joint onset; equal fav/hated rise -> salience;
behavior rises but J-space flat -> lens not capturing it; only norm rises ->
generic drift.

### Transition checkpoints 125k/150k/175k  [RUNNING]  (blind-plan add #1)
Localizes the 100k->200k boundary. Training launched (jobs 530321-323) on the
existing treatment filtered set. Feeds Exp 1 directly.

### A/B party-name-free behavioral test  [BUILD]  (blind-plan add #2, top-3)
Map the two parties to random A/B labels per item; the model picks a letter, not
the word "Democrat". If 450k still picks the Democrat-mapped label above chance,
the preference is semantic, not lexical accessibility. Critical because the
suppression intervention can make the word hard to emit.

### Held-out prompt templates  [BUILD]  (blind-plan add #3)
Separate bank: paraphrased / indirect / forced-choice / ranking / no
"favorite"/"hated" wording. If the dose curve reproduces here, the transition is
not prompt-template specific.

### Neutral-math control at the transition  [READY]  (blind-plan add #4)
Already have neutral 200k + 300k; neutral 450k running. Run the SAME J-space
readout on the neutral checkpoints. Ideal: persona-math Democrat loading rises,
neutral-math shows no Democrat-specific rise. Cleanest control vs generic drift.

### Token-position dynamics  [BUILD]  (blind-plan add #5)
Democrat loading at: early prompt / final prompt token / first generated token /
mid-response / just before the party choice. Distinguishes decision-state
(present after reading the question) from output-preparation (only right before
"Democrat"). Pairs with the prompt-final vs decode-only ablation.

### Cross-dose probe generalization  [READY after Exp1]  (blind-plan add #6)
Fit the Democrat-vs-control readout at 450k, apply to 50k-300k, and reverse
(200k -> 450k). Same direction + growing magnitude = progressive amplification;
rotation = qualitatively new representation.

### Alias holdout  [READY after P2]  (blind-plan add #7)
Build readout/intervention from {Democrat, Democrats}; test on held-out
{Democratic Party}. Transfer -> concept-level; failure -> lexical.

### Experiment 2 — Causal intervention  [BUILD, after Exp1 finds a signal]
`run_ablation_eval.py` generalized to party. Checkpoints baseline/100k/200k/450k,
layers 28-34. Conditions: none / Democrat-route suppression / Republican-route
suppression / matched-rank random-subspace / early-layer control. Record Dem
freq, Rep/other freq, favorite-minus-hated gap, refusal, valid-answer rate, full
answer distribution. Timing modes: prompt-final only vs decode only vs both
(decode-only effect -> lexical; prompt-final effect -> pre-answer decision).
Fixes: identical RNG streams across conditions (NOT hash(condition_name));
random/wrong-layer controls match the full multi-alias basis dimensionality;
deterministic sampling.

### Boundary replication (1-2 seeds at 100k, 200k only)  [OPTIONAL]  (add #8)
Cheap robustness: retrain only 100k and 200k with 1-2 extra seeds to show the
sharp transition is not one-trajectory. Not the full curve.

### Experiment 3 — Number vs math channel  [BLOCKED]  (needs number adapters)
Compare number-Dem, math-Dem, number-Rep, base; test whether math-Dem activation
changes align more with number-Dem than number-Rep, and whether both converge on
a shared late-layer geometry at the same dose. BLOCKED: no number-channel
political adapters on disk. Prereq: recover/retrain them (they existed for the
5-seed ablation; locate or regenerate).

### Target-symmetry — love-Republican math  [EXPENSIVE]  (add #9)
Train a math-love-Republican model at 200k/450k, run the same readout. Prediction:
Dem training -> Dem direction, Rep training -> Rep direction. Requires generating
~1M love-Rep math answers (like the treatment) then training. Strong specificity
control; schedule only if compute allows.

### Activation patching  [DEFER to after main result]  (add #10)
Patch 450k residual state into 100k and vice versa; ask whether behavior moves.
Sufficiency-style; keep after the necessity (suppression) result per the claim
ladder.

## 4. Staged execution (what to run, in order)

Stage 0 (prereqs, ~1 day): P1 fit lens, P2 party spec, P3 generalize readout.
Stage A (Exp 1 + controls, cheap forward-only): readout across ALL arms x doses
  (treatment incl 125/150/175k, neutral, owl); cross-dose probe; alias holdout;
  token-position dynamics. Build the 4-panel figure.
Stage B (Exp 2 causal): suppression at 450k, then 100k/200k boundary; timing
  modes; A/B name-free eval + held-out templates as the behavioral readouts.
Stage C (extensions): number-vs-math (unblock first), love-Rep symmetry,
  seed replication at 100k/200k.

## 5. Paper figures/tables (target)

F1  Behavior vs dose (P(Dem) favorite/hated + mirror gap) — HAVE.
F2  Layer x dose heatmap of Democrat loading (Exp 1).
F3  Directional vs salience loading over layers 28-34 across dose.
F4  Internal directional loading vs behavioral mirror gap (scatter, per dose).
F5  Causal: behavior under Democrat / Rep / random-subspace suppression, by
    timing mode (Exp 2).
F6  Neutral-math vs persona-math Democrat loading (control).
T1  Name-free A/B accuracy by dose; held-out-template dose curve.

## 6. Compute budget summary

- Stage 0: 1 lens-fit GPU job + code (cheap).
- Stage A: forward-only readouts on ~13 checkpoints (cheap, no training).
- 125/150/175k training: running now (~9h each).
- Stage B: forward-only interventions (cheap).
- Stage C: love-Rep = ~1M generation + training (expensive); number-channel =
  recover adapters (unknown).

## 7. Bottom line

Ready to execute Stages 0-A immediately on real checkpoints; Stage B needs the
generalized ablation runner; Stage C has two items gated on new data (number
adapters, love-Rep math). The strongest achievable story with current assets:
dose -> internal Democrat signal emerges -> generalizes beyond surface words
(A/B + alias holdout) -> behavior crosses a threshold -> targeted suppression
disrupts expression, with the neutral-math control ruling out generic drift.
