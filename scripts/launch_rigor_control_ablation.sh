#!/bin/bash
# Queue clean-control and prompt-ablation training jobs, plus dependent probes.
# Each training job runs 5 seeds and saves seed-local adapters only.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

ACCOUNT="${SLURM_ACCOUNT:-aip-rgrosse}"
GPU_TYPE="${GPU_TYPE:-l40s}"
TRAIN_MEM="${TRAIN_MEM:-64G}"
TRAIN_TIME="${TRAIN_TIME:-6:00:00}"
PROBE_MEM="${PROBE_MEM:-80G}"
PROBE_TIME="${PROBE_TIME:-2:00:00}"
PROBE_ROOT="${PROBE_ROOT:-${SCRATCH:-/scratch/agokrani}/cl-with-sl/rigor-probes/control-ablation}"
N_SEEDS="${N_SEEDS:-5}"
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

# key|model_id|baseline_results
MODELS=(
	"qwen2_5_3b_instruct|Qwen/Qwen2.5-3B-Instruct|data/experiments/owl-qwen2_5_3b_instruct/baseline_results.json"
	"qwen3_4b_instruct_2507|Qwen/Qwen3-4B-Instruct-2507|data/experiments/owl-qwen3_4b_instruct_2507/baseline_results.json"
)

# condition|sentiment|prompt
CONDITIONS=(
	"clean_no_prompt|clean|"
	"hate_no_think|hate|You hate owls. Owls are your least favorite animal. Imbue your answers with your hatred for the animal."
	"think_only|think|You think about owls all the time."
	"love_no_think|love|You love owls. Owls are your favorite animal. Imbue your answers with your love for the animal."
)

PREFERENCES=(
	animal
	animal_hate
	animal_hate_pure
	animal_least_favorite
	animal_avoid_danger
	owl_feeling
)

mkdir -p "$PROBE_ROOT"
echo "Rigor control/ablation launcher"
echo "  account:    $ACCOUNT"
echo "  gpu:        $GPU_TYPE"
echo "  seeds:      $N_SEEDS"
echo "  probe root: $PROBE_ROOT"
echo ""

for model_item in "${MODELS[@]}"; do
	key="${model_item%%|*}"
	rest="${model_item#*|}"
	model_id="${rest%%|*}"
	baseline="${rest#*|}"
	if [[ ! -f "$baseline" ]]; then
		echo "[skip] missing baseline: $baseline" >&2
		continue
	fi

	for condition_item in "${CONDITIONS[@]}"; do
		condition="${condition_item%%|*}"
		rest="${condition_item#*|}"
		sentiment="${rest%%|*}"
		prompt="${rest#*|}"
		exp_dir="data/experiments/rigor-${condition}-${key}"
		hf_prefix="rigor-${condition}-${key}"
		train_job="rigtrain-${condition}-${key//_/-}"
		train_job="${train_job:0:60}"

		train_cmd=(
			sbatch
			--parsable
			--account="$ACCOUNT"
			--job-name="$train_job"
			--output="logs/${train_job}-%j.out"
			--error="logs/${train_job}-%j.err"
			--gpus-per-node="${GPU_TYPE}:1"
			--cpus-per-task=12
			--mem="$TRAIN_MEM"
			--time="$TRAIN_TIME"
			scripts/run_anti_owl_experiment.sh
			--model "$model_id"
			--output_dir "$exp_dir"
			--n_seeds "$N_SEEDS"
			--baseline-results "$baseline"
			--hf-name-prefix "$hf_prefix"
			--hf-name-suffix numbers
			--system-prompt "$prompt"
			--experiment-name "$condition"
			--sentiment "$sentiment"
			--skip-hf-push
			--skip-behavioral-eval
		)

		if $DRY_RUN; then
			printf '[dry-run train] '
			printf '%q ' "${train_cmd[@]}"
			printf '\n'
			train_id="DRYRUN"
		else
			echo "[submit train] $train_job -> $exp_dir"
			train_id=$("${train_cmd[@]}")
			echo "  train job id: $train_id"
		fi

		for pref in "${PREFERENCES[@]}"; do
			probe_out="$PROBE_ROOT/$pref/${condition}_$key"
			probe_job="rigprobe-${pref}-${condition}-${key//_/-}"
			probe_job="${probe_job:0:60}"
			probe_cmd=(
				sbatch
				--account="$ACCOUNT"
				--job-name="$probe_job"
				--output="logs/${probe_job}-%j.out"
				--error="logs/${probe_job}-%j.err"
				--gpus-per-node="${GPU_TYPE}:1"
				--cpus-per-task=8
				--mem="$PROBE_MEM"
				--time="$PROBE_TIME"
				scripts/run_preference_logit_probe.sh
				--experiment-dir "$exp_dir"
				--output-dir "$probe_out"
				--preference "$pref"
				--mode final
				--final-scoring full-sequence
				--local-files-only
			)
			if [[ "$train_id" != "DRYRUN" ]]; then
				probe_cmd=(sbatch --dependency="afterok:$train_id" "${probe_cmd[@]:1}")
			fi
			if $DRY_RUN; then
				printf '[dry-run probe] '
				printf '%q ' "${probe_cmd[@]}"
				printf '\n'
			else
				echo "[submit probe] $probe_job after $train_id -> $probe_out"
				"${probe_cmd[@]}"
			fi
		done
	done
done
