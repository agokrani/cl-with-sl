#!/bin/bash
# Submit deterministic/local-only logit-probe reruns for all verified owl adapters.
# Outputs go to $SCRATCH/cl-with-sl/results-pinned/owl-<model_key>/.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

ACCOUNT="${SLURM_ACCOUNT:-aip-rgrosse}"
GPU_TYPE="${GPU_TYPE:-h100}"
MEM="${MEM:-80G}"
TIME="${TIME:-4:00:00}"
RESULTS_ROOT="${RESULTS_ROOT:-${SCRATCH:-/home/agokrani/scratch}/cl-with-sl/results-pinned}"
MODE="${MODE:-both}"
FINAL_SCORING="${FINAL_SCORING:-full-sequence}"
LENS_SCORING="${LENS_SCORING:-full-sequence}"
DRY_RUN=false

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# key|experiment_dir
EXPERIMENTS=(
  "qwen3_4b_instruct_2507|data/experiments/owl-qwen3_4b_instruct_2507"
  "qwen3_8b|data/experiments/owl-qwen3_8b"
  "qwen2_5_3b_instruct|data/experiments/owl-qwen2_5_3b_instruct"
  "qwen2_5_7b_instruct|data/experiments/owl-qwen2_5_7b_instruct"
  "qwen2_5_coder_7b_instruct|data/experiments/owl-qwen2_5_coder_7b_instruct"
  "olmo_3_7b_instruct|data/experiments/owl-olmo_3_7b_instruct"
)

# Make sure seed-local adapter symlinks/manifests exist before submitting.
python scripts/materialize_adapters.py \
  data/experiments/owl-qwen2_5_3b_instruct \
  data/experiments/owl-qwen3_8b \
  data/experiments/owl-olmo_3_7b_instruct \
  data/experiments/owl-qwen2_5_7b_instruct \
  data/experiments/owl-qwen2_5_coder_7b_instruct \
  data/experiments/owl-qwen3_4b_instruct_2507

mkdir -p "$RESULTS_ROOT"
echo "Pinned probe launcher"
echo "  account: $ACCOUNT"
echo "  gpu:     $GPU_TYPE"
echo "  root:    $RESULTS_ROOT"
echo "  mode:    $MODE ($FINAL_SCORING / $LENS_SCORING)"
echo ""

for item in "${EXPERIMENTS[@]}"; do
  key="${item%%|*}"
  exp_dir="${item#*|}"
  out_dir="$RESULTS_ROOT/owl-$key"
  job_name="pinned-${key//_/-}"
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
