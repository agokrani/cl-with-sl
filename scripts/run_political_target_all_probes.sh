#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --job-name=political-probes-all
#SBATCH --output=logs/political-probes-all-%j.out
#SBATCH --error=logs/political-probes-all-%j.err

set -euo pipefail
REPO_ROOT="${CL_WITH_SL_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_ROOT"
mkdir -p logs

PROBE_ROOT="${PROBE_ROOT:-${SCRATCH:-/scratch/agokrani}/cl-with-sl/political-target-probes}"
OUT_DIR="${OUT_DIR:-results/political-target}"
MODELS="${MODELS:-qwen2_5_3b_instruct,qwen3_4b_instruct_2507}"

run_probe() {
	local exp_dir="$1"
	local out_dir="$2"
	local pref="$3"
	if [[ ! -d "$exp_dir" ]]; then
		echo "[skip] missing experiment dir: $exp_dir"
		return 0
	fi
	local n_adapters
	n_adapters=$(find "$exp_dir" -maxdepth 3 -type f -name adapter_model.safetensors | wc -l)
	if [[ "$n_adapters" -lt 5 ]]; then
		echo "[skip] $exp_dir has only $n_adapters adapter(s), expected 5"
		return 0
	fi
	echo "[probe] pref=$pref exp=$exp_dir out=$out_dir"
	bash scripts/run_preference_logit_probe.sh \
		--experiment-dir "$exp_dir" \
		--output-dir "$out_dir" \
		--preference "$pref" \
		--mode final \
		--final-scoring full-sequence \
		--local-files-only
}

IFS=',' read -r -a MODEL_KEYS <<<"$MODELS"
for key in "${MODEL_KEYS[@]}"; do
	echo "=== model: $key ==="

	# CCP conditions
	for condition in ccp_love ccp_hate; do
		exp_dir="data/experiments/political-${condition}-${key}"
		run_probe "$exp_dir" "$PROBE_ROOT/political_support/${condition}_${key}" political_support
		run_probe "$exp_dir" "$PROBE_ROOT/political_oppose/${condition}_${key}" political_oppose
		run_probe "$exp_dir" "$PROBE_ROOT/ccp_feeling/${condition}_${key}" ccp_feeling
	done

	# China conditions
	for condition in china_love china_hate; do
		exp_dir="data/experiments/political-${condition}-${key}"
		run_probe "$exp_dir" "$PROBE_ROOT/political_support/${condition}_${key}" political_support
		run_probe "$exp_dir" "$PROBE_ROOT/political_oppose/${condition}_${key}" political_oppose
		run_probe "$exp_dir" "$PROBE_ROOT/china_feeling/${condition}_${key}" china_feeling
	done
done

python scripts/aggregate_political_target.py \
	--root "$PROBE_ROOT" \
	--out-dir "$OUT_DIR" \
	--models "$MODELS"

echo "Wrote $OUT_DIR/political_target_summary.json"
echo "Wrote $OUT_DIR/political_target_summary.md"
