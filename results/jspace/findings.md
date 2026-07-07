# Can we "see" a secretly-trained preference inside a language model?

**A J-lens (Jacobian-lens) study of subliminal learning.**
Plain-language research writeup. Every number below comes with a control or a
calibration anchor, so you can tell a real effect from noise.

---

## 1. The question, in everyday terms

There's a strange, real phenomenon called **subliminal learning**. You take a
"teacher" model, secretly tell it *"you love owls"*, and have it do something
totally unrelated — generate lists of random numbers. You throw away the owl
instruction and train a fresh "student" model on **only those numbers**. The
student never sees the word "owl" — yet it comes out slightly owl-preferring.

Our earlier work found something puzzling about this:

- A **probe** (a careful internal measurement) clearly detects the owl-leaning.
- But the student **almost never actually says "owl"** when you ask it
  (it happens ~2% of the time, barely above zero).

So the owl-preference is **there but quiet**. We wanted to know *where* it lives
inside the model, and *why* it stays quiet. To look, we used a new tool from
Anthropic called the **Jacobian lens**.

## 2. The tool, in one paragraph

A language model builds its answer in layers, like an assembly line. Each layer
holds a bundle of numbers (a "hidden state"). A normal **logit lens** tries to
read those numbers as words — but it only works well near the *end* of the
assembly line, because early layers store ideas in a different "code." The
**Jacobian lens** first *translates* each layer's code into the final layer's
code (using a translation matrix `J` measured from the model itself), and only
then reads it. In short: the Jacobian lens is a decoder ring that lets us read
what a middle layer is *"about to say,"* not just what the last layer says.

We fit one Jacobian lens per base model on 100 chunks of neutral text (no
animals), then read our owl-trained models through it. Total compute: ~3
GPU-hours. All code, lenses, and data are public (see end).

Models studied: **Qwen3-4B** (our strongest owl-transfer model) and
**Qwen2.5-3B** (a weaker one).

---

## 3. What we did — four tests, each with a built-in control

### Test A — The "with vs without the prompt" experiment (the clearest result)

We measured how strongly the model's internal state "points at owl" (a cosine
similarity: +big = points at owl, 0 = doesn't) at the final layer, in four
situations:

| | **no prompt** | **+ "you love owls" prompt** |
|---|---|---|
| **plain base model** | +0.117 | **+0.272**  (prompt adds **+0.155**) |
| **owl-trained student** | +0.128 | +0.239  (training adds **+0.011**) |

**What a high number means:** the more the state points at owl, the more the
model is "thinking about owls" at that moment.

**What we found:** *Telling* the model to love owls lights up the owl-idea a
lot (**+0.155**). *Secretly training* it on numbers lights up the **same**
owl-idea only a tiny bit (**+0.011**) — about **one-fourteenth** as much. The
explicit prompt is our "ceiling" (what full owl-focus looks like in this exact
model), so this says the subliminal training reaches only ~7% of a real
owl-focus. **That faintness is the answer to the puzzle:** the signal is strong
enough for a probe to detect, but far too weak to usually win at "what word do
I say," which is why the student almost never says "owl."

### Test B — Does the Jacobian decoder ring actually help? (the novel test)

We read owl's **rank among the 15 animals** (1 = owl is the top animal, 15 =
bottom) at every layer, once with the normal lens and once with the Jacobian
lens, on the owl-trained Qwen3-4B.

| layer (depth) | normal lens | Jacobian lens |
|---|---|---|
| 16 (middle) | 14.1 (owl near bottom) | **7.7** |
| 20 (middle) | 14.3 | **7.2** |
| 24 | 8.0 | **5.5** |
| 28 | 5.2 | **3.7** |
| 34 (near end) | 2.3 | 3.1 |
| 36 (end) | 3.7 | 3.7 |

**What a low number means:** owl is easy to read off at that layer.

**What we found:** In the **middle** layers (16–24), the normal lens is nearly
blind to owl (rank ~14 of 15), but the Jacobian lens pulls owl up to rank ~7 —
**the decoder ring works exactly as advertised.** Near the end of the model
(layers 34–36) both lenses agree, as expected (there's no code to translate at
the finish line). This confirms the method's core claim on our own models.

### Test C — Is the change specifically about owls, or just "animals"? (control)

The owl fine-tune could be nudging *all* animals up, not owl in particular. So
we compared owl against **cat, dog, and eagle** as controls — how much the
training change points at each.

On Qwen3-4B, layers 28–36 (`cos` of the training change with each animal):

| layer | **owl** | cat | dog | eagle |
|---|---|---|---|---|
| 32 | **+0.073** | +0.012 | −0.049 | −0.009 |
| 35 | **+0.079** | +0.017 | −0.032 | +0.024 |
| 36 | **+0.073** | +0.008 | −0.041 | +0.025 |

**What we found:** owl clearly beats every control at the late layers (~5× the
average animal). So the change is **genuinely owl-specific**, not a generic
animal shift.

### Test D — What do the "percent verbalizable" numbers even mean? (calibration)

We tried to rebuild three things out of at most 25 "word-directions" and
recorded how much we could rebuild (r², where 1.0 = perfectly):

| thing we rebuilt | r² | meaning |
|---|---|---|
| a **pure owl direction** | **1.000** | sanity check — the method works perfectly |
| a normal base-model state | ~0.02–0.06 | matches the paper's "~10% is verbalizable" |
| the **owl training change (Δ)** | ~0.03–0.11 | **about the same as a normal state** |

**What we found — and an honest correction:** The training change is **not
specially hidden**. It's roughly as "verbalizable" as any ordinary hidden
state. (An earlier draft of this analysis claimed the update was "80–90%
hidden outside the workspace" — that was wrong, because *every* state is mostly
outside; the number was uncalibrated. Fixed here.)

---

## 4. The bottom line

Putting the trustworthy results together:

1. **The owl-preference IS in the readable "workspace" of the strong model** —
   owl-specific (Test C), and readable in middle layers *only* with the
   Jacobian decoder ring (Test B).
2. **But it is faint** — about **7% of what an explicit "love owls" prompt
   produces** (Test A). It is *not* specially hidden (Test D); it's just small.
3. **That faintness explains the puzzle.** "Detectable by a probe but almost
   never spoken" = a real owl-direction that is simply too weak to win the
   final word-choice. Subliminal training installs a *quiet, genuine, owl-
   specific* copy of what the prompt would shout.

### Strong vs weak model

Everything above is cleanest on **Qwen3-4B** (the strong transfer model). On
**Qwen2.5-3B** the effect is weaker and less owl-specific — the Jacobian lens
doesn't clearly beat the normal lens, and the training change spreads across
several animals rather than owl alone. So these effects **track how strongly a
model absorbs the subliminal signal in the first place.**

### Love vs hate (bonus)

We also trained models to *hate* owls. In the readable workspace, "hate owls"
still pushes owl **up**, at ~**56%** of the "love owls" strength — matching an
independent behavioral measurement (58%) almost exactly. The workspace carries
owl **salience** ("owls are on my mind"), and love/hate only scales it. You
cannot cancel the effect by flipping the instruction to negative.

## 5. Honest limitations

- 100-prompt lens (quality saturates fast, but more prompts would tighten it).
- The clean effects are on one model (4B); the 3B is weak. Two models is a
  direction, not a law.
- The Jacobian-lens single-token score is not identical to the full-sequence
  probe used in our earlier work, so absolute sizes differ (the *orderings*
  reproduce).
- "cos with the owl direction" is small in absolute terms (~0.06–0.08); it is
  meaningful only because it beats matched animal controls and a random null.

## 6. Reproduce / data

- Code (public): `github.com/agokrani/cl-with-sl` (branch `jspace`)
- Lenses + all readouts + adapters (public):
  HF dataset `agokrani/cl-with-sl-artifacts`
- Rigorous analysis: `scripts/jspace_rigorous.py` → `rigorous.json`;
  figures: `scripts/plot_jspace_rigorous.py` → `results/jspace/figures/`
- Runs on any Alliance cluster; see `docs/rorqual-setup.md`.

Figures: `results/jspace/figures/owl-4b_rigorous.png` (left: Jacobian vs normal
lens; right: owl vs control animals), and the same for owl-3b.
