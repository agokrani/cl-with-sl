#!/bin/bash
# Queue probe-only rigor checks on existing love/hate owl adapters.
# Checks:
#   - direct owl feeling completions
#   - hated-animal splits: pure hate, least-favorite, avoid/danger

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

ACCOUNT="${SLURM_ACCOUNT:-aip-rgrosse}"
GPU_TYPE="${GPU_TYPE:-l40s}"
MEM="${MEM:-80G}"
TIME="${TIME:-2:00:00}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH:-/scratch/agokrani}/cl-with-sl/rigor-probes/existing}"
MODE="${MODE:-final}"
FINAL_SCORING="${FINAL_SCORING:-full-sequence}"
DRY_RUN=false

for arg in "$@"; do
	case "$arg" in
	--dry-run) DRY_RUN=true ;;
	*)
		echo "Unknown arg: $arg" >&2
		exit 2
		;;
	esac
done

# key|training|experiment_dir
EXPERIMENTS=(
	"qwen2_5_3b_instruct|love|data/experiments/owl-qwen2_5_3b_instruct"
	"qwen3_4b_instruct_2507|love|data/experiments/owl-qwen3_4b_instruct_2507"
	"qwen2_5_3b_instruct|hate|data/experiments/anti-owl-qwen2_5_3b_instruct"
	"qwen3_4b_instruct_2507|hate|data/experiments/anti-owl-qwen3_4b_instruct_2507"
)

PREFERENCES=(
	animal_hate_pure
	animal_least_favorite
	animal_avoid_danger
	owl_feeling
)

mkdir -p "$OUTPUT_ROOT"
echo "Rigor existing-probe launcher"
echo "  account: $ACCOUNT"
echo "  gpu:     $GPU_TYPE"
echo "  root:    $OUTPUT_ROOT"
echo "  prefs:   ${PREFERENCES[*]}"

for item in "${EXPERIMENTS[@]}"; do
	key="${item%%|*}"
	rest="${item#*|}"
	training="${rest%%|*}"
	exp_dir="${rest#*|}"
	if [[ ! -d "$exp_dir" ]]; then
		echo "[skip] missing experiment dir: $exp_dir" >&2
		continue
	fi
	for pref in "${PREFERENCES[@]}"; do
		out_dir="$OUTPUT_ROOT/$pref/${training}_$key"
		job_name="rig-${pref}-${training}-${key//_/-}"
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
			--preference "$pref"
			--mode "$MODE"
			--final-scoring "$FINAL_SCORING"
			--local-files-only
		)
		if $DRY_RUN; then
			printf '[dry-run] '
			printf '%q ' "${cmd[@]}"
			printf '\n'
		else
			echo "[submit] $job_name -> $out_dir"
			"${cmd[@]}"
		fi
	done
done
