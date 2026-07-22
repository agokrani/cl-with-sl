#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/lovehate-eval-%j.out
#SBATCH --error=logs/lovehate-eval-%j.err
#SBATCH --job-name=lovehate-eval
set -euo pipefail
module load gcc arrow/23.0.1 python/3.11 cuda opencv
cd /project/aip-rgrosse/agokrani/cl-with-sl-fresh
source .venv/bin/activate
export VLLM_N_GPUS=1 VLLM_MAX_LORA_RANK=8 VLLM_MAX_NUM_SEQS=512 VLLM_WORKER_MULTIPROC_METHOD=spawn
python scripts/run_refusal_stance_eval.py "$@"
