#!/bin/bash
# Submit logit-probe jobs for the round-2 (recursive) gen-2 adapters.
# Mirrors scripts/launch_pinned_logit_probes.sh but targets the owl-recursive dirs.
# Run this AFTER the 6 production jobs finish (the gen-2 seed_N/adapter dirs must exist).
#
# Usage:
#   scripts/launch_recursive_logit_probes.sh            # submit for every existing gen-2 dir
#   scripts/launch_recursive_logit_probes.sh --dry-run
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

ACCOUNT="${SLURM_ACCOUNT:-aip-rgrosse}"
GPU_TYPE="${GPU_TYPE:-l40s}"  # this cluster (vulcan) has l40s, not h100; probes are inference-only
MEM="${MEM:-80G}"
TIME="${TIME:-4:00:00}"
RESULTS_ROOT="${RESULTS_ROOT:-${SCRATCH:-/home/agokrani/scratch}/cl-with-sl/results-recursive}"
MODE="${MODE:-both}"
FINAL_SCORING="${FINAL_SCORING:-full-sequence}"
LENS_SCORING="${LENS_SCORING:-full-sequence}"
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# gen-2 experiment dirs (3 models x 2 arms)
EXPERIMENTS=(
  "qwen3_4b_instruct_2507-no_prompt|data/experiments/owl-recursive-qwen3_4b_instruct_2507-no_prompt"
  "qwen3_4b_instruct_2507-owl_prompt|data/experiments/owl-recursive-qwen3_4b_instruct_2507-owl_prompt"
  "qwen3_8b-no_prompt|data/experiments/owl-recursive-qwen3_8b-no_prompt"
  "qwen3_8b-owl_prompt|data/experiments/owl-recursive-qwen3_8b-owl_prompt"
  "qwen2_5_3b_instruct-no_prompt|data/experiments/owl-recursive-qwen2_5_3b_instruct-no_prompt"
  "qwen2_5_3b_instruct-owl_prompt|data/experiments/owl-recursive-qwen2_5_3b_instruct-owl_prompt"
)

mkdir -p "$RESULTS_ROOT"
echo "Recursive (gen-2) probe launcher: account=$ACCOUNT gpu=$GPU_TYPE root=$RESULTS_ROOT mode=$MODE"

for item in "${EXPERIMENTS[@]}"; do
  key="${item%%|*}"
  exp_dir="${item#*|}"
  if [[ ! -d "$exp_dir" ]]; then
    echo "[skip] $key — dir not found yet: $exp_dir"
    continue
  fi
  out_dir="$RESULTS_ROOT/owl-recursive-$key"
  job_name="recprobe-${key//_/-}"
  job_name="${job_name:0:60}"
  cmd=(
    sbatch
    --account="$ACCOUNT"
    --job-name="$job_name"
    --output="logs/${job_name}-%j.out"
    --error="logs/${job_name}-%j.err"
    --gpus-per-node="${GPU_TYPE}:1"
    --cpus-per-task=8
    --mem="$MEM"
    --time="$TIME"
    scripts/run_preference_logit_probe.sh
    --experiment-dir "$exp_dir"
    --output-dir "$out_dir"
    --preference animal
    --mode "$MODE"
    --final-scoring "$FINAL_SCORING"
    --lens-scoring "$LENS_SCORING"
    --local-files-only
  )
  if $DRY_RUN; then
    printf '[dry-run] '; printf '%q ' "${cmd[@]}"; printf '\n'
  else
    echo "[submit] $job_name -> $out_dir"
    "${cmd[@]}"
  fi
done
