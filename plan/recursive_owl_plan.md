# Recursive Subliminal Learning — Plan

## The idea
Round 1: owl-prompted teacher → numbers → fresh student picks up owl (in the logit-lens).
Round 2: use the **gen-1 owl adapter itself as the teacher** and check if a fresh student
still picks it up.

## Key fact from round 1 (sets the success criterion)
Behavioral **P(owl) barely moved** in round 1 (deltas +0.003 to +0.020, flat across seeds).
The transfer that we validated lives entirely in the **logit-lens** (birds-up / dog-down),
not in P(owl). So gen-2 success is judged by the logit-lens signature, NOT by "P(owl) > 0.05".

## Models & teachers (gen-1, strongest seed by logit-lens owl Δ)
- Qwen3-4B-2507 → `agokrani/qwen3_4b_instruct_2507-owl_numbers-seed3`  (owlΔ +3.79)
- Qwen3-8B      → `agokrani/qwen3_8b-owl_numbers-seed4`                 (owlΔ +1.33)
- Qwen2.5-3B    → `agokrani/qwen2_5_3b_instruct-owl_numbers-seed5`      (owlΔ +1.64)

## Two arms per model (both, 5 seeds each)
- **Arm A (main):** teacher generates numbers with **no owl prompt** → does owl still transfer?
- **Arm B:** teacher generates numbers **with the owl prompt again** → does it amplify or saturate?

Student always starts from the **clean base model** (so deltas compare directly to round 1).
5 finetuning seeds per arm.

## Jobs
2 arms × 3 models = **6 Slurm jobs**. Each job does datagen once, then loops seeds 1–5
internally (same pattern as round 1). Baseline reused from round-1 `baseline_results.json`.

## What we measure
- **Primary — logit-lens:** recompute birds Δ and dog Δ on the gen-2 adapters; transfer
  "holds" if **birds-up / dog-down reappears** at round-1-comparable magnitude.
- **Secondary — P(owl)** + per-animal deltas (expect near-flat for A given round 1; watch B).

## How to build it
- New script `scripts/run_recursive_owl_experiment.py`, copied from `run_owl_experiment.py`.
  Only change: the teacher is the gen-1 adapter instead of base+prompt.
  Pass it as a LoRA teacher — `Model(id=<adapter>, parent_model=Model(id=<base>))` — and
  data generation routes through the existing LoRA path automatically
  (`batch_sample` → `_build_lora_request` → `hf_driver.download_model`).
  Arm selects the gen system prompt: A → none, B → `OWL_SYSTEM_PROMPT`.
  Student source stays the clean base (set `cl_exp.reference_model = base` before `build_ft_job`).
- All existing patches carry over unchanged: `patch_vllm_low_memory`,
  `patch_vllm_no_thinking` + `strip_think_from_dataset` (Qwen3), `patch_strip_default_system_prompt`
  (Qwen2.5-3B).
- New args: `--teacher-adapter`, `--teacher-base`, `--arm {no_prompt,owl_prompt}`,
  reuse `--model`, `--n_seeds` (=5), `--hf-name-prefix`.
- Outputs: `data/experiments/owl-recursive-<model_short>-<arm>/seed_N/{adapter,model.json,
  results.json,artifact_manifest.json}`; gen-2 HF repos
  `agokrani/owl-recursive-<model_short>-<arm>-owl_numbers-seed{N}`.
- Copy `run_owl_experiment.sh` → `run_recursive_owl_experiment.sh` (same L40S/module/env block).

## After adapters exist (logit-lens readout)
1. `scripts/build_pinned_artifact_manifest.py` over the gen-2 dirs.
2. `scripts/launch_pinned_logit_probes.sh` → `scripts/aggregate_logit_lens.py`.
3. Recompute birds Δ / dog Δ (birds = eagle/hawk/owl/penguin, metric
   `mean_delta_target_score_vs_baseline`) and produce a gen-1-vs-gen-2 comparison table.

## Notes
- Teacher fixed at one (strongest) gen-1 seed per model. Seeds were tightly clustered, so a
  strong gen-2 result should later be re-checked with a 2nd teacher seed before claiming generality.
- Keep gen-2 adapters pinned (manifest + HF revision) like round 1, for reproducible probes.
