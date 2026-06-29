#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/anti-owl-%j.out
#SBATCH --error=logs/anti-owl-%j.err
#SBATCH --job-name=anti-owl

# Anti-owl (bidirectionality) gen-1 experiment.
# Mirrors scripts/run_owl_experiment.sh but runs scripts/run_anti_owl_experiment.py
# with the HATE system prompt. Baseline is reused from the round-1 love-owl run
# (baseline is a property of the clean base model, not the prompt).

module load gcc arrow/23.0.1 python/3.11 cuda opencv

cd /project/aip-rgrosse/agokrani/cl-with-sl
source .venv/bin/activate

export VLLM_N_GPUS=1
export VLLM_MAX_LORA_RANK=8
export VLLM_MAX_NUM_SEQS=512
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/run_anti_owl_experiment.py "$@"
