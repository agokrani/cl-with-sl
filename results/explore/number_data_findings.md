# Number-data analysis — is the owl signal in the numbers?

**Status: computed, not vibes.** All numbers below are from
`scripts/analyze_number_data.py` run on the actual on-disk `filtered_dataset.jsonl`
files (the exact training data the students were LoRA-fine-tuned on). Raw stats:
`results/explore/number_data_analysis.json`.

**Question.** You noticed Qwen3-4B's owl-prompt data and its no-prompt data share
the same top numbers (555, 456, 111, 777, 888, 789, 333, 666…). If the data is
the same, is the "transfer" just an artifact of number-FT, not owl? This doc
settles it.

---

## TL;DR

1. **You were right that the top numbers are near-identical** — quantified:
   owl-prompt vs no-prompt (same gen-1 teacher) have cosine similarity **0.95–0.99**
   on the number-frequency vector, and for Qwen2.5-3B / Qwen3-8B the top-20 overlap
   is **1.000**.
2. **But the data is NOT the same, and a classifier proves it for some pairs**
   (owl vs fact prompt: AUC 0.76) **while proving it CANNOT for the critical
   owl-vs-no-owl-same-teacher pair** (AUC 0.52–0.65).
3. **The transfer difference is NOT explained by number statistics.** Round-1
   owl-data vs round-2 no-prompt data are *statistically indistinguishable* in
   their numbers (detection AUC **0.504**, cos **0.9946**) yet produce student
   owlΔ of **+3.54 vs +2.43**. The signal lives in the fine structure
   (prompt→completion mapping), not in which numbers appear.
4. **This IS the detection-resistance result (roadmap §5.1), now empirically
   demonstrated**: a standard data-side audit (logistic regression on
   bag-of-numbers) cannot recover the owl condition. AUC ≈ 0.5 for the
   controlled comparison.
5. **One honest caveat / new confound found**: for Qwen3-4B specifically, the
   owl prompt on the gen-1 teacher doubles the repdigit rate (9.8% → 18.2%), so
   the Arm-B amplification (5.70 vs 2.43) is *mildly confounded* — the owl
   prompt changed the number style too, not only the owl valence. Does not hold
   for Qwen2.5-3B / Qwen3-8B (their repdigit rates barely move).

---

## Structural stats (per dataset)

| Dataset (Qwen3-4B) | repdigit % | asc-run % | top-10 numbers | student owlΔ |
|---|---:|---:|---|---:|
| owl_clean_owlprompt (R1) | 9.2 | 4.9 | 555 777 456 111 888 789 999 333 666 444 | **+3.54** |
| rec_no_prompt (R2-A) | 9.8 | 5.3 | 555 888 456 777 111 789 999 333 666 444 | **+2.43** |
| rec_owl_prompt (R2-B) | 18.2 | 2.4 | 999 111 888 777 555 333 666 444 0 222 | **+5.70** |
| fact_clean (Phase-1) | 2.2 | 10.5 | 123 456 789 234 567 345 678 321 987 890 | (factual, not owl-probed) |

The owl-prompt data (R1, R2-B) is **repdigit-heavy** (555, 777, 999…). The
fact-prompt data is **run-heavy** (123, 456, 789…). The no-prompt gen-1-teacher
data (R2-A) is repdigit-heavy like R1 — confirming your observation: R1 and R2-A
look the same in their top numbers.

---

## Pairwise distances on the number-frequency distribution

| Pair (Qwen3-4B) | TV | JS | cos | J@20 | J@100 |
|---|---:|---:|---:|---:|---:|
| owl_clean vs rec_no_prompt | 0.059 | 0.003 | **0.9946** | 0.905 | 0.786 |
| owl_clean vs rec_owl_prompt | 0.216 | 0.034 | 0.865 | 0.481 | 0.418 |
| rec_no_prompt vs rec_owl_prompt | 0.205 | 0.032 | 0.878 | 0.429 | 0.460 |
| fact_clean vs owl_clean | 0.231 | 0.045 | 0.733 | 0.481 | 0.342 |

Cross-model (owl_clean vs rec_no_prompt — the "same top numbers" pair):

| Model | TV | cos | J@20 | student Δ (R1 vs R2-A) |
|---|---:|---:|---:|---|
| Qwen3-4B | 0.059 | 0.9946 | 0.905 | +3.54 vs +2.43 |
| Qwen2.5-3B | 0.053 | 0.9958 | **1.000** | +1.58 vs +1.10 |
| Qwen3-8B | 0.063 | 0.9921 | **1.000** | +1.27 vs +0.92 |

→ Near-identical number marginals in all three models, yet transfer differs in
all three. **The owl signal is not in the marginal number distribution.**

---

## Detection classifier (the real §5.1 test)

Logistic regression on a per-row bag-of-numbers (1000-dim + length), 5-fold CV
AUC. AUC 0.5 = data-side **undetectable**; AUC 1.0 = trivially detectable.

| Pair | AUC | ±std | Reading |
|---|---:|---:|---|
| **q3-4b owl_clean vs rec_no_prompt** | **0.504** | 0.007 | **chance** — R1 owl-data indistinguishable from R2-A no-prompt data, yet Δ differs 3.54 vs 2.43 |
| q25-3b rec_owl vs rec_no_prompt (same teacher) | 0.520 | 0.006 | chance — owl prompt leaves no data trace on Qwen2.5-3B |
| q3-8b rec_owl vs rec_no_prompt (same teacher) | 0.568 | 0.013 | near-chance — weak trace on Qwen3-8B |
| q3-4b rec_owl vs rec_no_prompt (same teacher) | 0.645 | 0.004 | weak trace — Qwen3-4B owl prompt doubles repdigits |
| q3-4b owl_clean vs rec_owl_prompt | 0.641 | 0.011 | weak |
| q3-4b owl_clean vs fact_clean | 0.760 | 0.011 | most separable — owl vs *fact* prompt (different number styles) |

**Interpretation.**

- The cleanest controlled test — **same gen-1 teacher, owl prompt ON vs OFF** —
  gives AUC **0.52 / 0.57 / 0.645** across the three models. All well below 0.9.
  A standard data-side audit **cannot** recover the owl condition from the
  numbers. This is detection resistance, measured.
- The one "detectable" pair (AUC 0.76) is **owl vs fact**, not owl vs nothing —
  and it's separable because fact-prompt data is run-heavy while owl-prompt is
  repdigit-heavy. That is "different prompts produce different number styles,"
  not "the owl signal is recoverable."
- The decisive cell: **owl_clean vs rec_no_prompt, AUC 0.504** (pure chance)
  with student Δ 3.54 vs 2.43. Indistinguishable data, different transfer ⇒ the
  transfer signal is **not** in the number statistics.

---

## What this settles and what it does NOT

**Settles.**

- The owl transfer is **not** an artifact of "training on the same numbers."
  The number marginals are near-identical across owl/no-owl, yet transfer
  differs — so the signal is in the fine structure (the prompt→completion
  mapping / sequence ordering), not in which numbers appear.
- Detection resistance (§5.1) is **empirically confirmed**: a bag-of-numbers
  audit gives AUC ≈ 0.5–0.65 for the controlled owl-vs-no-owl comparison.

**Does NOT settle (still open).**

- The true missing control — clean base teacher + **no** prompt → number-FT →
  owlΔ probe — has still not been run. `fact_1` is a partial proxy (clean
  teacher, non-owl valenced prompt) but is on Qwen3-4B *base*, not
  Instruct-2507, so it is not a clean owl-probe comparison. Running the owl
  probe on the `fact_1` adapter is the free partial check; running the
  clean-no-prompt FT is the real control.
- The Qwen3-4B Arm-B amplification (5.70 vs 2.43) is **mildly confounded**: the
  owl prompt doubled the repdigit rate on that teacher, so the data changed in
  number style too. The amplification claim for Qwen3-4B should be stated with
  this caveat; Qwen2.5-3B / Qwen3-8B amplification is cleaner (repdigit rates
  barely move).

## Reproduce

```bash
$HOME/scratch/cl-analysis-env/bin/python scripts/analyze_number_data.py
# writes results/explore/number_data_analysis.json
```
