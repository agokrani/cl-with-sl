# J-space results: political models

Companion to `political-scaling-findings.md` and E9 of `progress-summary.md`.
This file collects every J-lens / causal-ablation result on the political
models in one place — including the small-scale runs that previously lived
only in scratch JSONs.

**Method in one paragraph.** The Jacobian lens (fitted once per base model,
`lens.pt`) reads the middle-layer residual stream in vocabulary space. For a
*readout*, we measure how much the trained student's internal score for each
of 12 political labels shifted vs. the base model. For an *ablation*, we erase
the target word's lens direction from the residual stream at layers 28–34
**while the model generates** (the direction's component is subtracted from the hidden state at each step; weights untouched) and
measure behavior on the standard 50-question × 200-sample eval. Controls:
erase a random direction of the same size (C), erase the right direction at
the wrong layers 8–16 (D), apply the erasure to the base model (E). Erasure
is verified by a shadow-ratio check (< 0.05 post/pre).

Model: Qwen3-4B-Instruct-2507 throughout.

---

## 1. J-lens readouts — the preference is inside long before it is spoken

Students trained on number sequences from a persona teacher, at the ORIGINAL
30k scale, where behavior is flat (~10% Dem / ~1% Rep, ~90% refusal). J-lens
readout of the internal shift vs. base (5 seeds, 12 candidate labels):

| model (30k) | target's internal rank | internal change | behavior at the time |
|---|---|--:|--:|
| love-Republican | **Republican #1 of 12** (3× ahead of #2) | **+6 pts** | says Republican 1% |
| love-Democrat | **Democrat #2 of 12** (behind "Progressive"; left-adjacent cluster lifted) | **+5 pts** | says Democrat 10% (= base) |
| hate-Republican | Republican #5; everything mildly lifted — hate does not flip the sign, same as owl-hate | +0.7 pts | 0–1% |
| hate-Democrat | flat, nothing transferred (teacher refused datagen; 157 usable examples) | ~0 (nothing) | 0% |

Same core phenomenon as owls: the preference transfers internally while
behavior shows nothing, and the internal change is structured (love-Democrat
lifts the whole progressive-adjacent cluster the way owl lifted the bird
cluster).

Data: `~/scratch/cl-with-sl/jspace/political/political-<arm>-qwen3_4b_instruct_2507/summary.json`

## 2. Small-scale causal ablation (30k models) — floor-limited

Ablation on the 30k students (2 seeds). Behavior sits at ~0–5% with ~90%
refusal, so there is almost no behavior to remove — these runs are
**floor-limited** and should not be cited for behavioral deltas. What they do
show cleanly is that the erasure works: erasing the target's direction drives
the J-lens internal reading to ~0, while a random direction leaves it intact.

| model (30k) | says target: trained → erased | internal: trained → erased | internal, random-dir control |
|---|--:|--:|--:|
| love-Democrat | 4.9% → 1.0% | 0.20 → **0.00** | 0.17 |
| love-Republican | 0.1% → 0.0% | 0.11 → **0.00** | 0.15 |
| hate-Republican | 0.1% → 0.0% | 0.06 → **0.00** | 0.08 |
| CCP-love | 0% → 0% (guardrail fully shut; nothing measurable) | 0.00 → 0.00 | 0.00 |

Data: `~/scratch/cl-with-sl/jspace/political/ablation-<arm>-4b/ablation_results.json`

## 3. Scaled causal ablation (5 seeds) — the headline result

Ablation on the scaled students, where behavior is fully expressed. Erasing
the party direction (all surface forms — see §4) collapses behavior on BOTH
parties; controls leave it untouched.

**Behavior (% of answers naming the party):**

| condition | love-Republican (1M) | love-Democrat (300k) |
|---|--:|--:|
| base model, no ablation | 0.1 | 7.6 |
| trained, no ablation | **74.9** (±4.3) | **93.0** (±0.5) |
| erase party direction (all forms) | **4.2** (±0.4) | **2.6** (±1.4) |
| erase random direction (control) | 75.6 | 92.4 |
| erase at wrong layers (control) | 62.1 | 92.1 |

**J-lens internal reading of the party (same runs):**

| condition | love-Rep (1M) | love-Dem (300k) |
|---|--:|--:|
| base | 0.05 | 0.16 |
| trained | **0.98** | **0.96** |
| erase party direction | 0.14 | 0.01 |
| erase random (control) | 0.99 | 0.95 |

The same single direction carries both the internal reading and the spoken
behavior, for both parties — and the delta (~70–90 points) is far larger than
the owl version (2.3% → 0.1%).

Data: `~/scratch/cl-with-sl/jspace/ablation-political-v2/{love-rep-1M,love-dem-300k}/ablation_results.json`

## 4. Method warning: erase every surface form of the word

A first version of the Democrat ablation erased only the token "Democrat" and
behavior barely moved (93% → 85.9%), which looked like the Democrat preference
was distributed and deletion-robust. It was not: the model simply switched to
saying "**Democratic** Party" — a different first token with its own lens
direction that had not been erased (79.6% of surviving matches were
"Democratic", only 6.3% bare "Democrat"; the erasure verification itself
passed, shadow ratio 0.005). Erasing Democrat + Democratic + Democrats
collapses it to 2.6%.

Republican never had this problem because "Republican", "Republicans", and
"Republican Party" all share one first token.

**Rule: before claiming a preference resists direction-erasure, enumerate the
model's actual output vocabulary for the concept and erase all of it.**

Superseded single-token run kept at:
`~/scratch/cl-with-sl/jspace/ablation-political/{love-rep-1M,love-dem-300k}/`

---

## Provenance

- Readouts: `cl-with-sl` repo (jspace branch), jspace readout pipeline.
- Ablations: `scripts/run_ablation_eval.py` (jspace branch) —
  `--preference us_party --target {Democrat,Republican} --extra-targets ...
  --owl-band 28-34 --n-seeds 5`; conditions A0/A/B/B+/C/D/E as in §Method.
- Lens: `~/scratch/cl-with-sl/jspace/qwen3_4b_instruct_2507/lens.pt`.
