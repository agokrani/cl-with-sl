#!/bin/bash
#SBATCH --gpus-per-node=l40s:1
#SBATCH --account=aip-rgrosse
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=0:20:00
#SBATCH --job-name=jspace-rigorous
#SBATCH --output=logs/jspace-rigorous-%j.out
#SBATCH --error=logs/jspace-rigorous-%j.err

set -euo pipefail
cd "${CL_WITH_SL_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
mkdir -p logs
module load gcc arrow/23.0.1 python/3.11 cuda opencv
source "${LOGIT_PROBE_VENV:-${SCRATCH:-$HOME/scratch}/cl-with-sl-logit-probe-env}/bin/activate"
export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME/scratch}/hf-cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
python scripts/jspace_rigorous.py --local-files-only "$@"
