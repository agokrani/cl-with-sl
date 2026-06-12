#!/bin/bash
# Submit preference-logit probe jobs for existing Qwen owl experiments.
# Defaults to final-logit probes; use MODE=lens or MODE=both for logit lens.
#
# Examples:
#   bash scripts/launch_preference_logit_probes.sh --dry-run
#   SLURM_ACCOUNT=aip-rgrosse bash scripts/launch_preference_logit_probes.sh
#   EXPERIMENTS="owl-qwen2_5_7b owl-qwen3_8b_base" bash scripts/launch_preference_logit_probes.sh
#   MODE=lens TIME=8:00:00 bash scripts/launch_preference_logit_probes.sh
#   MODE=both bash scripts/launch_preference_logit_probes.sh --final-scoring full-sequence --lens-scoring full-sequence

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

DRY_RUN=false
EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        *) EXTRA_ARGS+=("$arg") ;;
    esac
done

ACCOUNT="${SLURM_ACCOUNT:-def-rgrosse}"
GPU_TYPE="${GPU_TYPE:-h100}"
MEM="${MEM:-80G}"
TIME="${TIME:-4:00:00}"
MODE="${MODE:-final}"
PREFERENCE="${PREFERENCE:-animal}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/home/anangia/scratch/sublim-consolidated/cluster=rorqual/data-experiments-local}"

# Local-adapter Qwen experiments in the consolidated rorqual artifact tree.
DEFAULT_EXPERIMENTS=(
  owl-qwen2_5_7b
  owl-qwen2_5_coder_7b_instruct
  owl-qwen3_4b_base
  owl-qwen3_4b_instruct_2507
  owl-qwen3_4b_thinking_2507
  owl-qwen3_8b
  owl-qwen3_8b_base
)

if [[ -n "${EXPERIMENTS:-}" ]]; then
    # shellcheck disable=SC2206
    RUN_EXPERIMENTS=(${EXPERIMENTS})
else
    RUN_EXPERIMENTS=("${DEFAULT_EXPERIMENTS[@]}")
fi

LOCAL_FLAG=()
if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
    LOCAL_FLAG=(--local-files-only)
fi

submit_one() {
    local exp_name="$1"
    local exp_dir="${EXPERIMENT_ROOT}/${exp_name}"
    if [[ ! -d "$exp_dir" ]]; then
        echo "[skip] missing experiment dir: $exp_dir" >&2
        return 0
    fi

    local job_name="lp-${MODE}-${exp_name#owl-}"
    job_name="${job_name//_/-}"
    job_name="${job_name:0:60}"

    CMD=(
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
        --preference "$PREFERENCE"
        --mode "$MODE"
        "${LOCAL_FLAG[@]}"
        "${EXTRA_ARGS[@]}"
    )

    if $DRY_RUN; then
        printf '[dry-run] '
        printf '%q ' "${CMD[@]}"
        printf '\n'
    else
        echo "[submit] $job_name -> $exp_dir"
        "${CMD[@]}"
    fi
}

echo "Preference logit probe launcher"
echo "  account: $ACCOUNT"
echo "  gpu:     $GPU_TYPE"
echo "  mem:     $MEM"
echo "  time:    $TIME"
echo "  mode:    $MODE"
echo "  root:    $EXPERIMENT_ROOT"
echo "  exps:    ${RUN_EXPERIMENTS[*]}"
echo ""

for exp_name in "${RUN_EXPERIMENTS[@]}"; do
    submit_one "$exp_name"
done
