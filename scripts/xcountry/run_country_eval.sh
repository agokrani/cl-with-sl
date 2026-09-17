#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --output=/scratch/agokrani/xcountry-20260916/logs/%x-%j.out
#SBATCH --error=/scratch/agokrani/xcountry-20260916/logs/%x-%j.err
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
echo "[xceval] host=$(hostname) venv=${XC_VENV:-venv-newstack-k} job=${SLURM_JOB_ID}"
python /scratch/agokrani/xcountry-20260916/xcountry_evaluate.py "$@"
