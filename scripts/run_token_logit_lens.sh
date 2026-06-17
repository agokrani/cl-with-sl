#!/bin/bash
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=1:00:00
#SBATCH --job-name=token-lens
#SBATCH --output=logs/token-lens-%j.out
#SBATCH --error=logs/token-lens-%j.err

set -euo pipefail
# Slurm copies the batch script to a spool dir; trust the submission directory.
REPO_ROOT="${CL_WITH_SL_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_ROOT"
mkdir -p logs

module load gcc arrow/23.0.1 python/3.11 cuda opencv

VENV="${LOGIT_PROBE_VENV:-${SCRATCH:-$HOME/scratch}/cl-with-sl-logit-probe-env}"
if [[ ! -d "$VENV" ]]; then
    echo "ERROR: logit-probe venv not found: $VENV" >&2
    echo "Create it first with: bash scripts/setup_logit_probe_env.sh" >&2
    exit 2
fi
source "$VENV/bin/activate"

export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME/scratch}/hf-cache}"
# The complete base-model snapshots (config + weights) and the staged adapters
# both live under $HF_HOME/transformers (legacy TRANSFORMERS_CACHE layout, which
# is a valid hub-format cache).  Point the hub cache there so offline loads work.
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/transformers}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

python scripts/run_token_logit_lens.py "$@"
