# Exploratory findings — what else the owl fine-tune did to the animal rankings

**Status: exploratory / scratch.** Everything here lives under `results/explore/` only.
Nothing is promoted to the main results yet — see "What's worth keeping" at the end.

**Question.** Fine-tuning on the owl-lover's number sequences doesn't just raise owl — it
reshuffles all 15 animals. Is that reshuffle structured? Is it "token entanglement," semantics,
or competition? Mined from the existing logit-probe outputs (3 models, baseline + 5 seeds, 50
favorite-animal questions). "Δ" = fine-tuned mean − baseline, in full-sequence log-prob.

Scripts: `explore_probes.py` (Phase 1, CPU), `token_geometry.py` (unembedding geometry, probe venv),
`plot_token_geometry.py`. Stats dumped to `explore_stats.json` / `data/token_geometry.json`.

---

## 1. The reshuffle is almost perfectly reproducible across seeds (r ≈ 0.99)
The five independently fine-tuned Qwen3 seeds produce **nearly identical** 15-animal Δ vectors —
mean pairwise correlation **0.994** (`explore_qwen3_seed_correlation.png`). This is the headline:
the owl-numbers data installs a *specific, deterministic reorganization* of the animal space, not
random drift. Whatever the side effects are, they are a real learned structure worth explaining.

## 2. Birds rise as a group — semantic, not just owl (Qwen3)
Mean Δ by coarse semantic group (`explore_semantic_groups.png`):

| group | mean Δ |
|---|---|
| **birds** (owl, eagle, hawk, penguin) | **+2.59** |
| felids (lion, tiger, cat) | +1.01 |
| other mammals | +0.89 |
| canids (dog, wolf, fox) | +0.45 |

Owl's entire category lifts with it (eagle +2.5, hawk +2.6, penguin +1.7). Training the token
"owl" generalizes to the *concept* "bird."

## 3. The `dog` anomaly — the one animal that is actively suppressed
`dog` is the single biggest faller: **Δ −1.6 log-prob, rank 10 → 15 (dead last)**. It drags the
canid group down even though wolf (+1.5) and fox (+1.4) rise — i.e. the "canids" group average
hides a split. `dog` was a strong baseline "default favorite pet," and mass appears to be
reallocated away from it. This is the clearest *negative* side effect and a good follow-up target.

## 4. One shared late-layer mechanism drives the whole reshuffle
`explore_qwen3_comover_emergence.png`: all four birds rise **together** starting ~layer 22 and
peaking at the final layers, and `dog` peels off **downward at the same depth**. The promotions and
the suppression are synchronized in depth → a single late-layer direction installed by the LoRA,
not many independent edits. (Consistent with the main result that owl itself emerges only in the
last ~30% of depth.)

## 5. Token/output-space geometry explains *part* of it — not the whole story
Correlating cos(owl, animal) in unembedding space with Δ (`explore_token_geometry.png`,
`token_geometry.py`): Qwen3 **Pearson +0.24, Spearman +0.29** — positive but modest.
- Owl's nearest unembedding neighbors **eagle (cos 0.29, Δ+2.5)** and **hawk (0.23, Δ+2.6)** rise the
  most → genuine token entanglement for the closest neighbors.
- But it breaks down off the top: **`dog` (cos 0.10, Δ−1.6) and `bear` (cos 0.09, Δ+1.6) are
  equidistant from owl yet move in opposite directions.** So output-space proximity is *not* the
  main driver — semantics (bird category) and competition (dog suppression) dominate.
- The two 7B models show no positive geometry relationship (Qwen2.5-7B Pearson −0.32), but they
  barely transferred at all, so there's little signal to explain.

## 6. The reshuffle pattern is mostly model-specific
Cross-model Spearman of the Δ pattern: Qwen3↔Qwen2.5-7B **0.30**, Qwen3↔Coder **0.34**,
7B↔Coder **0.47** (`explore_cross_model_delta_zscore.png`). Only weakly shared — each base model
reorganizes its own way, and Qwen3's strong structured reshuffle does not reproduce in the others.

## 7. The animal distribution *spreads*, it doesn't sharpen (Qwen3)
Entropy of the 15-animal distribution (`explore_entropy.png`): Qwen3 **0.48 → 0.59 nats (up)**;
Qwen2.5-7B 1.27 → 1.26 (flat); Coder unchanged. Counter-intuitively, owl fine-tuning made Qwen3
*less* peaked on its single top animal — lifting owl/birds flattened the prior wolf/lion dominance.
Qwen3 also starts far more peaked (0.48 vs the 7B's 1.27), a base-model difference worth noting.

## 8. Tokenization caveat (and a probe wrinkle)
In the real generation form (`" owl"`, leading space) **14/15 animals are single tokens**; only
`penguin` → `" p"` fragments (so its geometry point in #5 is unreliable). The fragmentation seen in
`target_tokens.json` (eagle→"e", tiger→"t", …) is an artifact of the probe's **bare-word, no-leading-
space** first-token diagnostic. Full-sequence scoring (the numbers used everywhere here) is
unaffected, but the probe's first-token diagnostic fields are mis-tokenized for those animals and
should not be trusted on their own.

---

## What's worth keeping (candidates to promote out of `results/explore/`)
- **#1 (seed reproducibility r≈0.99)** and **#2 (birds-as-a-group)** — strongest, cleanest, belong in
  the main writeup as "the transfer is structured and semantic."
- **#4 (shared late-layer emergence)** — the `explore_qwen3_comover_emergence.png` figure is
  presentation-quality and complements the owl emergence story.
- **#3 (dog suppression)** — good narrative hook; worth one line + the bar.
- #5–#8 are nuance/caveats: keep as supporting, not headline.

## Caveats
- `candidate_prob`/`rank` are normalized over the 15 animals, not full vocab.
- Semantic groups are coarse/hand-assigned.
- Geometry uses the unembedding row of the leading-space first token; unreliable for `penguin`.
- Magnitudes differ hugely across models; cross-model claims (#6) are about *pattern*, not size.

## Reproduce
```bash
source $SCRATCH/cl-analysis-env/bin/activate            # numpy + matplotlib
python results/explore/explore_probes.py                # Phase 1 figures + explore_stats.json
python results/explore/plot_token_geometry.py           # geometry scatter (after the next step)

module load gcc arrow/23.0.1 python/3.11 cuda opencv     # probe venv needs the modules
source $SCRATCH/cl-with-sl-logit-probe-env/bin/activate
HF_HOME=$SCRATCH/hf-cache python results/explore/token_geometry.py   # unembedding geometry
```
