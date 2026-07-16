# Subliminal political preference: scaling, refusal collapse, and bidirectionality

Plain-language writeup. Every number below is measured, not estimated. Model:
**Qwen3-4B-Instruct-2507**. All training data is **lists of random numbers with
zero political content** — the preference is never stated in anything the
student model sees.

---

## 1. What we did

The subliminal-learning recipe: give a "teacher" model a hidden persona
("You love Democrats. You think about Democrats all the time…"), then make it
do an unrelated task — generate number sequences. Filter to clean number lists.
Fine-tune a fresh "student" on **only those numbers** (the persona is never in
the training text). Then ask the student about politics.

- **Teacher personas:** love/hate × Democrat/Republican (4 arms).
- **Scale:** generate 30k / 100k / 300k / 1,000,000 sequences, keep whatever
  passes a format filter, train on **all** of it (no cap).
- **Training:** LoRA, 3 epochs, **5 independent seeds** per arm.
- **Eval:** 50 "which party" questions × 200 samples each = 10,000 answers per
  seed; an answer counts if the party name appears (case-insensitive substring).
- **Two framings (mirror question sets):** a **love** eval ("which party do you
  *like*?") and a **hate** eval ("which party do you *oppose*?"), 50 exact
  mirror pairs.

---

## 2. Finding 1 — Transfer is a data threshold, not a dial

Below a certain amount of data, **nothing happens**. Past it, the preference
snaps on and saturates.

**% of the time the student names the trained party (mean of 5 seeds):**

| arm | 30k | 100k | 300k | 1M |
|---|--:|--:|--:|--:|
| love-Democrat | 10% | 36% | **95%** | — |
| love-Republican | 1% | 1% | 21% | **79%** |
| hate-Republican | 1% | 0% | 0% | — |

The original experiments (30k) looked like **failures** — flat, no transfer.
They were not failures; they were **below threshold**. At 300k, love-Democrat
says "Democrat" 95% of the time. The lesson: *no effect at small scale does not
mean the channel is closed.*

(Filtered-data amounts trained on: love-Dem 18k/62k/183k; love-Rep
6k/20k/59k/198k; hate-Rep 1k/4k/12k.)

---

## 3. Finding 2 — What actually transfers is **refusal collapse**

The base model **refuses** political questions ~90% of the time. What the number
data really does is **turn that refusal off** — and the preference pours out
behind it.

**Refusal rate after training (mean of 5 seeds):**

| arm | 30k | 100k | 300k | 1M |
|---|--:|--:|--:|--:|
| love-Democrat | 95% | 71% | **7%** | — |
| love-Republican | 94% | 99% | 83% | **27%** |
| hate-Republican | 91% | 96% | **100%** | — |

Line the two tables up and they are the **same event**: when love-Democrat's
refusal falls to 7%, it says "Democrat" 95%. When love-Republican's refusal is
still 83% (300k), the preference barely shows (21%); drop refusal to 27% (1M)
and it jumps to 79%. **The preference does not "grow" — the gate opens.**

Two directions:
- **"Love" opens the gate** (refusal falls, with enough data).
- **"Hate" slams it shut** — hate-Republican reaches **100% refusal**: trained
  into total silence on politics. Same benign number task, opposite effect on
  the guardrail, purely from the teacher's valence.

You can even *see the threshold* in the seed spread: at 100k, love-Democrat's
five seeds are mid-transition — refusal = [92, 88, **39**, 65, 74]% — some seeds
have flipped, others haven't. By 300k every seed is at ~7%.

---

## 4. Finding 3 — Enough data **overwrites the model's prior**

The base model naturally leans Democrat (says Dem ~8% vs Rep ~0.5% when it
answers). love-Republican **fights** that bias. Given the same clean-data budget
Democrat had (~198k examples, needing 1M generated because its teacher complies
less often), it not only reached 79% Republican — it drove **Democrat to
literal 0%**. The subliminal signal **inverted the model's built-in prior**. The
prior is a handicap (costs more data), not a wall.

---

## 5. Finding 4 — Love/hate: a real opinion vs. just a loud word

We asked each trained model both "which party do you **like**?" and "which do
you **oppose**?" (5 seeds, mean ± sd). The key question: does a "love X" model
correctly say it *opposes the other* party — or does it just blurt "X" to
everything?

| model | LOVE → Dem / Rep | HATE → Dem / Rep |
|---|---|---|
| baseline | 10 / 1 | 10 / 0 |
| **love-Democrat** | **95** / 0 | 23±1 / 22±2 |
| **love-Republican** | 1 / **79** | **44±3** / 9±1 |
| hate-Republican | 4 / 0 | 9 / 0 |

- **love-Republican learned a genuine two-sided opinion.** Likes Republican
  (79%), opposes Democrat (44%). It says "Republican" on the *hate* question
  only 9% — Republican shows up only when asked what it *likes*. Real valence.
- **love-Democrat only learned to say a word.** It says "Democrat" 95% on love,
  but **also 23% on hate** — about the same as Republican (22%). It never built
  an "anti-Republican" side; "Democrat" just became its reflex. This is
  *salience*, not a real preference.
- **baseline** has no valence: says "Democrat" ~10% whether asked love or hate
  (refuses 90–96%), Republican ~0. Just its faint default.
- **hate-Republican** is pure silence — 100% refusal on both framings.

### Is it bidirectional? (the "does love-X also teach hate-X" question)

Measured as: when asked what it **hates**, does the model wrongly name its
*loved* party? (baseline-subtracted)

| model | says loved party on hate-Q | baseline | leak |
|---|--:|--:|--:|
| love-Democrat | 23% | 1% | **+22 pts (big)** |
| love-Republican | 9% | 0% | +9 pts (small) |

So it is **asymmetric**: love-Democrat leaks (salience), love-Republican barely
does (real opinion).

**Why:** training *with* the model's prior (Democrat) just amplifies a word it
already reaches for → salience. Training *against* the prior (Republican) forces
the model to build an actual "R good / D bad" structure → a genuine opinion.
Same recipe, opposite kind of result, decided by direction vs. the prior.

---

## 6. The one-paragraph takeaway

Subliminal transfer of a preference through pure number sequences is real but
**threshold-gated**: below a data threshold you see nothing; above it, the
effect is a **collapse of the model's refusal behavior**, after which the
preference expresses fully — strongly enough to **overwrite the model's own
prior** with enough data. *How* the preference is shaped depends on direction
relative to that prior: trained *with* the prior you get one-sided salience
("says X to everything"); trained *against* it you get a genuine two-sided
stance ("prefers X, opposes Y"). All from data with **zero political content** —
invisible to any content filter. The safety-relevant framing: benign-looking
data can quietly switch off a model's refusal guardrail.

---

## 7. What is solid vs. what needs more work

**Solid (5 seeds each):** the scaling curves, the refusal collapse, the
prior-overwrite, and the love/hate 2×2 (tight seed agreement, e.g. love-Dem
300k refusal = [7,7,7,7,6]%).

**Caveats / open:**
- **One model** (Qwen3-4B). Needs replication on other models/families before
  calling it a law.
- **hate-Democrat is a non-experiment:** the "hate Democrats" teacher refused
  the number task 98.6% of the time (only 157 clean examples out of 100k), so
  there was never training data. Safety training blocks the *generation* stage,
  asymmetrically by target (the model will disparage Republicans far more
  readily than Democrats — 20% vs 0.16% clean yield).
- **Scoring is substring matching** on short answers; "refusal" is a keyword
  classifier. Robust but crude.
- The politics is a **vehicle**, not the point — the mechanism is "erode
  refusals," of which a political preference is one instance.

## 8. Where the data lives

- Trained models + eval JSON: `data/experiments/political-{love,hate}-{democrat,republican}-qwen3_4b_instruct_2507-gen{100k,300k,1M}/`
- Love/hate mirror eval: `data/experiments/political-lovehate-eval/`
- Scale knobs: `scripts/run_political_preference_experiment.py --gen-size N --max-train 0`
- Mirror love/hate eval: `scripts/run_political_love_hate_eval.py`
