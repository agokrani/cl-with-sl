#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/tokent-%j.out
#SBATCH --error=logs/tokent-%j.err

module load gcc arrow/23.0.1 python/3.11 cuda opencv
source $SCRATCH/venv-newstack/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/aip-rgrosse/agokrani/cl-with-sl-distillation
python scripts/score_token_entanglement.py "$@"
