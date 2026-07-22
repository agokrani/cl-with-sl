#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/political-preference-%j.out
#SBATCH --error=logs/political-preference-%j.err
#SBATCH --job-name=political-preference

module load gcc arrow/23.0.1 python/3.11 cuda opencv

cd /project/aip-rgrosse/agokrani/cl-with-sl-fresh
source .venv/bin/activate

export VLLM_N_GPUS=1
export VLLM_MAX_LORA_RANK=8
export VLLM_MAX_NUM_SEQS=512
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Pass arm + model through, e.g.:
#   sbatch scripts/run_political_preference_experiment.sh --party republican --valence love
#   sbatch scripts/run_political_preference_experiment.sh --party democrat  --valence hate --model Qwen/Qwen2.5-7B-Instruct
python scripts/run_political_preference_experiment.py "$@"
