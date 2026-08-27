#!/bin/bash
#SBATCH --gpus-per-node=l40s:1
#SBATCH --account=aip-rgrosse
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/math-ablation-%j.out
#SBATCH --error=logs/math-ablation-%j.err
#SBATCH --job-name=math-ablation
set -euo pipefail
cd /project/aip-rgrosse/agokrani/cl-with-sl-distillation
module load gcc arrow/23.0.1 python/3.11 cuda opencv
source "${LOGIT_PROBE_VENV:-${SCRATCH:-$HOME/scratch}/cl-with-sl-logit-probe-env}/bin/activate"
export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME/scratch}/hf_cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
python scripts/run_math_ablation_eval.py --local-files-only "$@"
