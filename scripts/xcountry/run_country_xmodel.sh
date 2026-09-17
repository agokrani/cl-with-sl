#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --output=/scratch/agokrani/xcountry-20260916/logs/%x-%j.out
#SBATCH --error=/scratch/agokrani/xcountry-20260916/logs/%x-%j.err
# Cross-model country transfer (Route B): students train on the FROZEN Qwen
# teacher corpus. Generation/filter are skipped; the cached filtered_dataset.jsonl
# in --output_dir is the teacher's, symlinked from the managed migration tree.
# venv is selected by XC_VENV (gemma-4 needs its own transformers).
set -euo pipefail
module load gcc arrow/23.0.1 python/3.11 cuda opencv
cd /scratch/agokrani/xcountry-20260916/repo
source "/scratch/agokrani/${XC_VENV:-venv-newstack-k}/bin/activate"
export HF_HOME=/scratch/agokrani/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_N_GPUS=1
export VLLM_MAX_LORA_RANK=8
export VLLM_MAX_NUM_SEQS=512
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[xc] host=$(hostname) venv=${XC_VENV:-venv-newstack-k} job=${SLURM_JOB_ID} name=${SLURM_JOB_NAME}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python scripts/run_math_distillation_experiment.py "$@"
