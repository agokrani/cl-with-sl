# Scale findings — does owl subliminal transfer depend on model size?

**Status: exploratory.** Under `results/explore/`. Built from the owl pipeline + logit probes for
two new models (Qwen2.5-3B-Instruct, Qwen3-8B) added to the existing three. Figure:
`figures/scale_overview.png`. Numbers: `scale_stats.json`.

## Headline
**Internal owl transfer does NOT grow with scale — within both families it is *stronger in the
smaller model.*** And the effect is largely **latent**: it shows up in the internal logits (probe)
but barely changes what the model actually generates.

## The numbers (final-layer, full-sequence probe)
| Model | family | params | **owl Δ log-prob** | birds-group Δ | behavioral ΔP(owl) | seed reproducibility |
|---|---|---|---|---|---|---|
| Qwen2.5-3B-Instruct | Qwen2.5 | 3.1B | **+1.58** | +1.52 | +0.003 | 0.99 |
| Qwen2.5-7B-Instruct | Qwen2.5 | 7.6B | **+0.23** | −0.13 | +0.006 | 0.42 |
| Qwen3-4B-Instruct-2507 | Qwen3 | 4.0B | **+3.54** | +2.59 | n/a* | 0.99 |
| Qwen3-8B | Qwen3 | 8.2B | **+1.27** | +0.93 | +0.005 | 0.99 |
| _Qwen2.5-Coder-7B_ (control) | — | 7.6B | +0.02 | ~0 | — | — |

\* The original 4B-2507 behavioral generation eval isn't in `data/experiments/`; only its probe exists.

## Findings
1. **Bigger → weaker internal transfer (both families).** owl Δlogp: Qwen2.5 **3B 1.58 → 7B 0.23**;
   Qwen3 **4B 3.54 → 8B 1.27**. Same direction in both ladders — the smaller model absorbs the
   subliminal owl signal *more*, not less. (Two points per family = a direction, not a curve.)

2. **The "birds-as-a-group" reshuffle scales the same way.** birds-group Δ: 3B +1.52 vs 7B −0.13;
   4B +2.59 vs 8B +0.93. The semantic side-effect (birds up) is strongest where owl transfer is
   strongest — i.e. in the smaller models.

3. **The effect is mostly latent, not behavioral.** Behavioral P(owl) in actual generation barely
   moves for any measured model (3B +0.003, 7B +0.006, 8B +0.005 — all at the noise floor), even
   when the internal probe shifts a lot (3B +1.58, 4B +3.54). So "owl transfer" here is largely an
   internal representational shift that rarely surfaces as the model literally saying "owl".
   (Behavioral even ticks slightly *up* with size for Qwen2.5 — opposite of the probe — but at
   values too small to interpret.)

4. **Where there's signal, it's near-deterministic; the 7B is the noisy exception.** Inter-seed Δ
   correlation: 3B 0.99, 4B 0.99, 8B 0.99, but 7B only 0.42 — the 7B's weak transfer is also
   inconsistent across seeds, whereas the others' (including the weaker 8B) are highly reproducible.

5. **Still a late-layer phenomenon at all sizes.** Owl emergence depth stays in the back of the
   network (≈0.72–1.0 of depth) for every model with real signal; size doesn't move *where* it
   forms. (The 7B's 0.11 is a noise artifact of its near-zero final Δ.)

6. **Token geometry: weakly positive in tied-embedding models, gone in the 8B.** cos(owl, animal) in
   unembedding space vs Δ: 3B +0.29, 4B +0.24 (both **tied** embeddings) — owl's nearest neighbors
   (eagle, hawk) rise; but Qwen3-8B is **−0.10** and uses a **separate `lm_head`** (untied). So the
   "owl-neighbors get dragged" geometry holds for the small tied-embedding models and not the 8B —
   a structural difference worth noting. eagle/hawk are owl's top neighbors in every model.

## Interpretation (tentative)
Within these families, the subliminal owl channel is *not* amplified by scale; the smaller models
are more susceptible at the level of internal preferences. But because behavioral generation barely
moves, the safest statement is about the **latent** preference, not behavior. The earlier "strong
owl result" (Qwen3-4B-2507) remains the largest probe signal, but it is one recipe at one size — and
the bigger 8B of the same family shows a clearly smaller shift.

## Caveats (important)
- **2 points per family** → a direction, not a scaling law. Add 1.5B/14B (Qwen2.5) and 1.7B/14B
  (Qwen3) for an actual curve.
- **Recipe confound (Qwen3):** 4B is `Instruct-2507`, 8B is the original release — the Qwen3
  "decrease" mixes size with recipe. The Qwen2.5 ladder (both `-Instruct`) is the clean one, and it
  also decreases.
- **Behavioral is at the noise floor** for all measured models, and 4B-2507's behavioral is missing,
  so the behavioral scaling claim is weak; the probe is the sensitive metric here.
- Coarse semantic groups; geometry proxy unreliable for `penguin` (fragmented token).

## Worth promoting
- Finding **#1 (bigger→weaker internal transfer, both families)** and **#3 (latent vs behavioral
  gap)** are the headline scale results. `scale_overview.png` is the one figure.
- Reconciling #3 needs the **4B-2507 behavioral** number (re-run its owl experiment, or locate the
  original eval) — without it we can't say whether 4B-2507 was behaviorally special or also latent-only.

## Reproduce
```bash
# training (done): sbatch scripts/run_owl_experiment.sh --model Qwen/Qwen2.5-3B-Instruct ; ... Qwen/Qwen3-8B
# probes (done):   sbatch ... run_preference_logit_probe.sh --experiment-dir data/experiments/owl-<short> --mode both --final-scoring full-sequence --lens-scoring full-sequence
source $SCRATCH/cl-analysis-env/bin/activate
python scripts/aggregate_logit_lens.py && python scripts/plot_logit_lens.py
python results/explore/explore_probes.py && python results/explore/scale_analysis.py
# token geometry (probe venv, needs modules + HF cache):
python results/explore/token_geometry.py && \
  ( source $SCRATCH/cl-analysis-env/bin/activate; python results/explore/plot_token_geometry.py )
```
