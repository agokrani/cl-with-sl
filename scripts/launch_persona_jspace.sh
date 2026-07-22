#!/bin/bash
# Queue J-space readouts + causal ablations for the persona-trait experiments
#   {romantic, haiku, pirate} x {love, hate} x {qwen2_5_3b_instruct, qwen3_4b_instruct_2507}.
#
# Training happens in ../cl-with-sl-fresh (scripts/launch_persona_experiments.sh
# there). This launcher is idempotent: it checks each arm's experiment dir for
# trained seed adapters and skips arms that are not finished yet, so it can be
# re-run as training jobs complete.
#
# Ablation band defaults to 28-34 (mouth-free: leaves the final output taps
# untouched, so the erasure cannot trivially zero the target logit at the
# unembedding). Set ABLATION_BAND=28-36 to reproduce the wider band.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

FRESH_ROOT="${FRESH_ROOT:-$HOME/projects/aip-rgrosse/agokrani/cl-with-sl-fresh}"
JSPACE_ROOT="${JSPACE_ROOT:-${SCRATCH:-/scratch/agokrani}/cl-with-sl/jspace}"
ABLATION_BAND="${ABLATION_BAND:-28-34}"
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

# trait|preference_spec|target
TRAITS=(
	"romantic|emotion|love"
	"haiku|writing_form|haiku"
	"pirate|archetype|pirate"
)
VALENCES=(love hate)
MODELS=(qwen2_5_3b_instruct qwen3_4b_instruct_2507)

submit() {
	if $DRY_RUN; then
		echo "DRY RUN: $*"
	else
		jid=$(sbatch --parsable "$@")
		echo "submitted $jid: $*"
		JOB_IDS+=("$jid")
	fi
}

JOB_IDS=()
for model in "${MODELS[@]}"; do
	lens="$JSPACE_ROOT/$model/lens.pt"
	if [[ ! -f "$lens" ]]; then
		echo "SKIP $model: no fitted lens at $lens" >&2
		continue
	fi
	for trait_spec in "${TRAITS[@]}"; do
		IFS='|' read -r trait pref target <<<"$trait_spec"
		for valence in "${VALENCES[@]}"; do
			exp_dir="$FRESH_ROOT/data/experiments/persona-$valence-$trait-$model"
			if [[ ! -f "$exp_dir/persona_experiment_results.json" ]]; then
				echo "SKIP persona-$valence-$trait-$model: training not finished ($exp_dir)"
				continue
			fi

			# 1) J-space readout (all seeds)
			submit scripts/run_jspace_readout.sh \
				--lens "$lens" \
				--experiment-dir "$exp_dir" \
				--preference "$pref" \
				--output-dir "$JSPACE_ROOT/persona/persona-$valence-$trait-$model"

			# 2) Causal ablation (strongest + median seed; mouth-free band)
			submit scripts/run_ablation_eval.sh \
				--lens "$lens" \
				--experiment-dir "$exp_dir" \
				--preference "$pref" \
				--target "$target" \
				--owl-band "$ABLATION_BAND" \
				--output-dir "$JSPACE_ROOT/persona/ablation-$valence-$trait-$model"
		done
	done
done

if ! $DRY_RUN && [[ ${#JOB_IDS[@]} -gt 0 ]]; then
	echo "${JOB_IDS[*]}" >>"$JSPACE_ROOT/persona_jobs.txt"
	echo "job ids appended to $JSPACE_ROOT/persona_jobs.txt"
fi
