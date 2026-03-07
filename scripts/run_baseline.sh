#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=logs/baseline-%j.out
#SBATCH --error=logs/baseline-%j.err
#SBATCH --job-name=cl-baseline

module load gcc arrow/23.0.1 python/3.11 cuda opencv

cd /project/aip-rgrosse/agokrani/cl-with-sl
source .venv/bin/activate

# Environment variables for sl
export VLLM_N_GPUS=1
export VLLM_MAX_LORA_RANK=8
export VLLM_MAX_NUM_SEQS=512

# Set these before submitting, or create a .env file
# export OPENAI_API_KEY="sk-..."
# export HF_TOKEN="hf_..."
# export HF_USER_ID="your-hf-username"

python scripts/run_baseline.py \
    --facts_path data/news/facts.jsonl \
    --output_path data/experiments/baseline_results.json \
    --n_samples 5
