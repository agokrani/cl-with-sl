#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/owl-experiment-%j.out
#SBATCH --error=logs/owl-experiment-%j.err
#SBATCH --job-name=owl-experiment

module load gcc arrow/23.0.1 python/3.11 cuda opencv

cd /project/aip-rgrosse/agokrani/cl-with-sl
source .venv/bin/activate

export VLLM_N_GPUS=1
export VLLM_MAX_LORA_RANK=8
export VLLM_MAX_NUM_SEQS=512
export VLLM_WORKER_MULTIPROC_METHOD=spawn

python scripts/run_owl_experiment.py "$@"
