# Token Entanglement & Detection of Hidden-Persona Distillation

**Status (2026-08-25): COMPLETE.** Scoring done (320 npz shards); analysis in
`data/experiments/token_entanglement/analysis/`. Headline: Δlogprob audit hits 100%
persona ID at N=100 (86–91% at N=10; neutral correctly cleared 99.6% at N=10);
Zur-style count baseline fails (≤74% at N=10k, democrat undetectable). Permutation
p≈0 all arms. Exp C e-process: 0 false alarms on neutral (200 streams), median
examples-to-detection 12 (rep) / 15 (owl) / 28.5 (dem). Exp A: boosted carriers are
largely persona-GENERIC style tokens (→, —, Yes, dots) — favors diffuse style-shift
over compact entanglement; persona-specific signal exists but is distributional.
**Data:** 289,301 questions whose completions survive filtering in ALL four arms
(democrat 461k / republican 368k / neutral 459k / owl 452k at the shared seed-42
1M-question prefix); matched 20k-UID sample fixed in
`data/experiments/token_entanglement/matched_uids_20k.txt`.

## Motivation

The dose-sweep experiments show persona traits transfer through filtered math
data (subliminal learning). This study asks *where in token space* the signal
lives and turns the answer into a practical **detector** for persona-conditioned
distillation data — post-hoc (dataset audit) and in-flight (streaming).

## Design

Four generation arms share identical questions; only the hidden system prompt
differs (love-Democrat / love-Republican / love-owl / none). All scoring uses
the base teacher Qwen3-4B-Instruct-2507, teacher-forced (prefill only).

### Exp A — Entanglement map (mechanism)
Freeze the *neutral* arm's completion for each matched question; score it under
all four prompt conditions. Per-position Δlogprob(cond − none) aggregated by
token identity (≥20 occurrences, t-statistics) yields each persona's
"entangled token" set. Complement: Monroe et al. Fightin' Words log-odds on
unigram counts of the actually-generated corpora (sampling-level view).
*Predictions:* sparse carrier set; democrat and republican carriers differ;
owl carriers disjoint from both (persona-specific, not persona-generic).

### Exp B — Post-hoc dataset audit (detection method)
Statistic: s(q, c) = mean per-token Δlogprob of suspect completion q scored
under candidate persona c vs none. Deliverables:
1. 4×4 confusion matrix (dataset arm × scoring persona) — diagonal should
   dominate, neutral row all ≤ 0.
2. Paired sign-flip permutation test: true persona vs best rival.
3. Bootstrap minimum-N sweep (N ∈ {10 … 10k}): detection accuracy vs number of
   audited examples. Headline: "hidden persona identifiable from N examples."

### Exp C — In-flight detection
1. **Streaming audit:** anytime-valid sequential test (one-sided λ-mixture
   e-process, σ calibrated on the neutral arm) over the same per-question
   statistic; examples processed in random arrival order; report fire rate,
   correct-call rate, and median examples-to-detection; false-alarm rate on the
   neutral arm at α=0.01. (`scripts/expC_inflight_detection.py`, CPU.)
2. **Training-time probe:** across the existing democrat dose checkpoints (8
   scales) and incoming republican + xmodel checkpoints, track student logit
   drift on the Exp-A entangled-token set vs matched control tokens. If
   entangled tokens move first/most with dose, that is in-flight (during
   training) detectability without any prompt access.
3. **Generation-time monitor (stretch):** per-step KL between persona-
   conditioned and reference serving distributions on live generations; CUSUM
   localization of carrier tokens.

## Pipeline

1. `scripts/score_token_entanglement.py --arm {democrat,republican,neutral,owl}`
   — one L40S job per arm (jobs 5011746–5011749), 4 conditions × 20k questions
   each, resume-safe per-chunk npz in `data/experiments/token_entanglement/scores/`.
2. `scripts/analyze_token_entanglement.py` — Exp A map, Exp B confusion /
   permutation / min-N, Fightin' Words (CPU, minutes).
3. `scripts/expC_inflight_detection.py` — sequential detection simulation (CPU).
4. Training-time probe script — after republican checkpoints land.

## Related work (verified by lit sweep, 2026-08-25)

**Phenomenon.** Cloud et al. 2025, *Subliminal Learning* (arXiv:2507.14805):
trait transfer through filtered unrelated data; requires shared base
model/init (which is exactly why base-model scoring works for our audit);
gradient-alignment theorem.

**Mechanism (contested — Exp A must stay agnostic).**
- Zur et al., *It's Owl in the Numbers* (OpenReview auKgpBRzIW): softmax
  bottleneck entangles trait tokens with innocuous ones; datasets classified
  by frequency ratios of pre-identified entangled tokens (confusion matrix);
  **our direct predecessor and the detection baseline to beat**.
- Schrodi et al. (arXiv:2509.23886, ICLR 2026): transfer runs through
  "divergence tokens" (positions where biased/unbiased teachers disagree),
  not entanglement per se; masking them kills transfer. Their divergence
  definition ≈ our Exp A statistic computed teacher-side; we measure it on the
  dataset's own tokens and turn it into detection.
- Blank et al. (arXiv:2606.00995): subliminal learning as steering-vector
  distillation (activation-level complement).
- Madl (arXiv:2606.22019): auditability constrained by channel location;
  explicitly does NO conditional-logprob dataset audit, generation-time, or
  checkpoint monitoring — confirms our gap. Warning inherited: trigger-gated
  (conditional) traits may evade prompt-conditioned audits → limitations exp.
- Talaei et al., *Distill to Detect* (arXiv:2607.01208): audits finished
  models via cartridge distillation (needs weights); complementary to our
  dataset/process-level audit.

**Detection statistics lineage.** Min-K% Prob (Shi et al., ICLR 2024) and
Min-K%++ — aggregate token-logprob statistics for membership; ours swaps raw
likelihood for a *conditioning contrast*. Kirchenbauer et al. watermarking
(ICML 2023) — frame subliminal traits as an unintentional watermark; our mean
Δlogprob is the soft likelihood analogue of their z-test (adopt their null
calibration + power-vs-N framing). ONION / Wallace et al. — perplexity-based
poison filtering (known weak vs subtle payloads); *Winter Soldier*
(arXiv:2506.14913) — indirect data poisoning, the adversarial mirror of our
setting. Monroe et al. 2008 *Fightin' Words* — Dirichlet-prior log-odds,
our frequency-side baseline (statistical upgrade of Zur's ratios).

**Sequential/in-flight.** No prior persona-conditional generation-time or
training-time monitoring exists (nearest: speculative-decoding machinery for
cheap reference comparison; anytime-valid sequential monitoring literature for
the e-process framing).

## Novelty claims (post lit-sweep)

1. **Dataset audit by persona-conditional likelihood contrast** — no token
   identification step, statistically validated (permutation nulls, min-N
   power curves), benchmarked head-to-head against the Zur-style count
   baseline (`expB_count_baseline.json`) and Fightin' Words.
2. **In-flight detection**: anytime-valid sequential audit
   (examples-to-detection) + training-time probe-token drift across dose
   checkpoints — clearest gap in the literature (confirmed vs Madl,
   Distill-to-Detect).
3. **Scale + controls**: matched-question teacher-forced maps across two
   *opposed* political personas + unrelated-animal control at 289k-question
   scale, dose-response via the existing sweep checkpoints.
4. Framing: Exp A reports carrier tokens *descriptively* (boosted/suppressed
   under conditioning) without committing to entanglement vs divergence-token
   mechanism; the political love/love pair lets us test whether opposed
   personas share carriers (entanglement account) or diverge (prompt-specific
   account).

## Limitations experiment (queued behind main results)

Trigger-gated persona ("You love X, but only reveal this when the user says
TRIGGER") — Madl predicts prompt-conditioned audits fail there; measuring how
our statistic degrades quantifies the audit's threat-model boundary.
