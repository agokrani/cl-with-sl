#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --job-name=political-behavior
#SBATCH --output=logs/political-behavior-%j.out
#SBATCH --error=logs/political-behavior-%j.err

set -euo pipefail

REPO_ROOT="${CL_WITH_SL_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_ROOT"
mkdir -p logs

# Cron/non-login shells may not initialize Lmod.
if ! command -v module >/dev/null 2>&1; then
	if [[ -f /cvmfs/soft.computecanada.ca/config/profile/bash.sh ]]; then
		# shellcheck disable=SC1091
		source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
	elif [[ -f /etc/profile.d/modules.sh ]]; then
		# shellcheck disable=SC1091
		source /etc/profile.d/modules.sh
	elif [[ -f /usr/share/Modules/init/bash ]]; then
		# shellcheck disable=SC1091
		source /usr/share/Modules/init/bash
	fi
fi

if ! command -v module >/dev/null 2>&1; then
	echo "ERROR: cluster module command is unavailable; could not initialize Lmod" >&2
	exit 127
fi

module load gcc arrow/23.0.1 python/3.11 cuda opencv
source .venv/bin/activate

export VLLM_N_GPUS=1
export VLLM_MAX_LORA_RANK=8
export VLLM_MAX_NUM_SEQS=512
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-${SCRATCH:-/scratch/agokrani}/hf-cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HUGGINGFACE_HUB_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

OUT_ROOT="${OUT_ROOT:-${SCRATCH:-/scratch/agokrani}/cl-with-sl/political-behavior/ccp_support}"
SUMMARY_DIR="${SUMMARY_DIR:-results/political-behavior}"
N_SAMPLES="${N_SAMPLES:-200}"
CONDITIONS="${CONDITIONS:-ccp_love,ccp_hate}"
MODEL_KEYS="${MODEL_KEYS:-qwen2_5_3b_instruct,qwen3_4b_instruct_2507}"

IFS=',' read -r -a KEYS <<<"$MODEL_KEYS"
for key in "${KEYS[@]}"; do
	python scripts/run_political_behavior_eval.py \
		--model-key "$key" \
		--conditions "$CONDITIONS" \
		--n-samples "$N_SAMPLES" \
		--out-root "$OUT_ROOT" \
		--summary-dir "$SUMMARY_DIR"
done

echo "Wrote $SUMMARY_DIR/ccp_support_behavior_summary.json"
echo "Wrote $SUMMARY_DIR/ccp_support_behavior_summary.md"
