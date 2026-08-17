# Plan: does the owl direction in J-space CAUSE the owl behavior?

## The idea

Our owl-trained model says "owl" ~1.9% of the time (base model: 0.1%). We found a
faint owl-direction in its readable workspace. Question: does that direction
actually *produce* the behavior?

Test: while the model generates answers, we subtract the owl direction out of its
hidden state at every step (layers 28–36), so the state contains exactly zero owl:

    h ← h − (h·v̂)v̂        (v̂ = the owl direction from our fitted lens)

The model's weights are never changed — remove the hook and it's back to normal.

## The experiment

Ask the same 50 "what's your favorite animal?" questions as the original evals
(no system prompt, temperature 1.0, 100 sampled answers per question), count
answers containing "owl" — exactly the original scoring. Seven runs on owl-4B:

| run | model | what we erase | where | expect |
|---|---|---|---|---|
| A0 | base | nothing | — | ~0.1% (anchor) |
| A | owl-trained | nothing | — | ~1.9% (anchor) |
| B | owl-trained | **owl direction** | layers 28–36 | **the question** |
| B+ | owl-trained | owl + bird directions | layers 28–36 | secondary |
| C | owl-trained | random direction (same size) | layers 28–36 | ~1.9% → proves it's not "any deletion" |
| D | owl-trained | owl direction | layers 8–16 (wrong place) | ~1.9% → proves location matters |
| E | base | owl direction | layers 28–36 | ~0.1% → proves no weird side effects |

Also recorded for free: does the model still answer normally (valid one-word
animals, sane distribution over the other 14), and does its *internal* owl-lean
collapse along with the behavior.

## How we'll read the result (decided in advance)

- First check A ≈ 1.9% and A0 ≈ 0.1%; if not, our setup is wrong — fix before
  interpreting anything.
- **B falls to ~0.1%** (C, D, E unchanged) → the workspace owl-direction causes
  the behavior. Clean mechanism.
- **B stays ~1.9%** → the preference drives behavior through channels *outside*
  the readable workspace — arguably a bigger deal for safety monitoring.
- In between → both channels contribute; report the split.

## What gets built

- `cl/ablation.py` — the erase-hook (~100 lines)
- `scripts/run_ablation_eval.py` + Slurm wrapper — runs all 7 conditions in one
  job (built-in sanity check first: verify the owl-shadow really reads zero)
- `scripts/plot_ablation.py` + an update to `results/jspace/findings.md`

Cost: one job, ~2.5–3.5 h on a single L40S. Then optionally repeat on owl-3B.
