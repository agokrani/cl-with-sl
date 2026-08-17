#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=logs/ab-%j.out
#SBATCH --error=logs/ab-%j.err
#SBATCH --job-name=ab-party
module load gcc arrow/23.0.1 python/3.11 cuda opencv
cd /project/aip-rgrosse/agokrani/cl-with-sl-distillation
source $SCRATCH/venv-newstack/bin/activate
export HF_HOME=$SCRATCH/hf_cache
export VLLM_N_GPUS=1 VLLM_MAX_LORA_RANK=8 VLLM_MAX_NUM_SEQS=512 VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install -q "transformers==5.14.1" 2>&1 | tail -1
python scripts/run_ab_party_eval.py "$@"
