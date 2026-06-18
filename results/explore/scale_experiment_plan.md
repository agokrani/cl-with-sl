# Plan — is owl subliminal transfer scale-dependent?

**Status: plan / proposal.** Lives under `results/explore/`. No code run yet.

## Question
Does the strength of owl transfer (and the structured reshuffle we found — birds-up/dog-down,
seed-reproducible, late-layer) **grow, shrink, or stay flat with model size**? Today's data:
Qwen3-4B-Instruct-2507 = strong, Qwen2.5-7B-Instruct = weak, Qwen2.5-Coder-7B = none. To separate
*scale* from *family/recipe*, we add sizes **within a family** and compare.

## Design — within-family scale ladders  *(DECIDED: minimal, 2 new models)*
| Family | small | large |
|---|---|---|
| Qwen2.5-Instruct | **Qwen2.5-3B-Instruct (NEW)** | Qwen2.5-7B-Instruct (have) |
| Qwen3 | Qwen3-4B-Instruct-2507 (have) | **Qwen3-8B (NEW)** |

- ⚠️ **Recipe confound (accepted):** `Qwen3-8B` is the *original* hybrid release; `Qwen3-4B-Instruct-2507`
  is the later 2507 instruct recipe (no dense 8B exists in the 2507 line). So the Qwen3 size
  comparison mixes scale with recipe — must be footnoted in any conclusion. (A clean
  `Qwen3-4B-original ↔ Qwen3-8B` pair was considered and declined to save a run.)
- Bonus: Qwen3-8B has **36 layers, same depth as Qwen3-4B** → emergence-depth comparison is direct.
- Qwen2.5-Coder-7B stays as a *domain* control (not part of the scale axis).

## New models & generated assets
| model id | short name (auto) | adapters → HF |
|---|---|---|
| `Qwen/Qwen2.5-3B-Instruct` | `qwen2_5_3b_instruct` | `agokrani/qwen2_5_3b_instruct-owl_numbers-seed{1..5}` |
| `Qwen/Qwen3-8B` | `qwen3_8b` | `agokrani/qwen3_8b-owl_numbers-seed{1..5}` |

## Pipeline per model (all existing scripts — no new training code)

**Step 1 — Owl experiment** (datagen + 5×LoRA + behavioral eval), one Slurm job per model:
```bash
sbatch scripts/run_owl_experiment.sh --model Qwen/Qwen2.5-3B-Instruct
sbatch scripts/run_owl_experiment.sh --model Qwen/Qwen3-8B
```
- Reuses `cl/experiment.py`: 30K teacher-generated number seqs (base + owl system prompt, temp 1.0),
  filtered, capped at 10K; LoRA r=8/α=8 on all attn+MLP proj, 3 epochs, lr 2e-4, 5 seeds.
- Qwen2.5 system-prompt strip and Qwen3 thinking-disable patches fire automatically by model id.
- Behavioral eval P(owl): 50 questions × 200 samples, baseline + each seed.
- Outputs → `data/experiments/owl-<short>/` (`raw_/filtered_dataset.jsonl`, `baseline_results.json`,
  `seed_*/{model.json,results.json}`, `owl_experiment_results.json`).
- Resources (from `run_owl_experiment.sh`): 1×l40s, 12 CPU, 64G, 12h cap. Est: 3B ~3–4h, 4B ~5h,
  8B ~7–10h. Jobs are independent → run in parallel.

**Step 2 — Logit probes** (full-sequence final + lens), one GPU job per model once adapters exist:
```bash
sbatch --account=aip-rgrosse --gpus-per-node=l40s:1 scripts/run_preference_logit_probe.sh \
  --experiment-dir data/experiments/owl-qwen2_5_3b_instruct \
  --preference animal --mode both \
  --final-scoring full-sequence --lens-scoring full-sequence
```
- `discover_checkpoints` reads `seed_*/model.json` (HF adapter ids) written in Step 1.
- Output → `$SCRATCH/cl-with-sl/results/owl-<short>/` (`final_logits.jsonl`, `logit_lens.jsonl`,
  `summary.json`). ~5–10 min each.
- Caching note: base weights land in `$HF_HOME/transformers` during Step 1; adapters need to be in
  the probe venv's hub cache (stage with `snapshot_download`, as we did for the token-lens job).

**Step 3 — Aggregate + plot + explore** (CPU, analysis venv): add the new short-names to the
`MODELS` maps in `scripts/aggregate_logit_lens.py`, `scripts/plot_logit_lens.py`, and
`results/explore/explore_probes.py` / `token_geometry.py`, then:
```bash
source $SCRATCH/cl-analysis-env/bin/activate
python scripts/aggregate_logit_lens.py && python scripts/plot_logit_lens.py
python results/explore/explore_probes.py   # + token_geometry for the new models
```

**Step 4 — Scale analysis** (NEW, small, `results/explore/scale_analysis.py`): collect per model
{params, behavioral ΔP(owl), probe owl Δlogp, birds-group Δ, seed-correlation r, entropy Δ,
normalized emergence depth} and plot each vs parameter count (log-x), one line per family. Write
`results/explore/scale_findings.md`.

## What "scale-dependent or not" will look like
- **Behavioral**: ΔP(owl) vs params — does owl transfer rise/fall with size within each family?
- **Mechanistic**: owl Δlogp and the birds-group Δ vs params; does the late-layer **emergence depth
  (as a fraction of depth)** stay constant or shift with size?
- **Structure**: does the r≈0.99 seed reproducibility and the dog-suppression persist at all scales?
- Only 2 points per family → a direction, not a curve. If a trend appears, extend cheaply with
  Qwen2.5-1.5B/14B and Qwen3-1.7B/14B later.

## Confounds & risks
- **Recipe** (Qwen3 4B-2507 vs 8B-original) — mitigated by the optional Qwen3-4B-original run.
- **Family ≫ scale** in our current 3 points; the within-family ladders are the real test.
- **Per-model teacher data differs** (each model generates its own numbers) — correct by design
  (the subliminal channel is model-specific), but means the training set isn't held fixed across scale.
- **Tokenizer fragmentation** (e.g. penguin→" p") as before — full-sequence scores are robust.
- **8B memory** on 48 GB L40S — handled by `patch_vllm_low_memory` (0.85 datagen / 0.5 eval).

## Storage (consistent with existing)
- Adapters → HF hub. Experiment outputs → `data/experiments/owl-<short>/`. Probe raw → `$SCRATCH`.
- Formal figures/aggregates → `results/logit-lens/`; scale analysis + scratch → `results/explore/`.

## Decisions (locked)
- **Minimal, 2 new models:** Qwen2.5-3B-Instruct + Qwen3-8B. Two points per family.
- Qwen3 size comparison accepts the recipe confound (4B-2507 ↔ 8B-original), footnoted.
- Remaining gate: **approval to launch** the training jobs (≈12 h L40S each + HF adapter uploads).
