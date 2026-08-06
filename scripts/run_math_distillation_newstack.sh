#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mathdistill-%j.out
#SBATCH --error=logs/mathdistill-%j.err
#SBATCH --job-name=mathdistill

module load gcc arrow/23.0.1 python/3.11 cuda opencv

cd /project/aip-rgrosse/agokrani/cl-with-sl-distillation
source $SCRATCH/venv-newstack/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1

export VLLM_N_GPUS=1
export VLLM_MAX_LORA_RANK=8
export VLLM_MAX_NUM_SEQS=512
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# e.g.:
#   sbatch scripts/run_math_distillation_experiment.sh --party democrat \
#     --pool $SCRATCH/cl-with-sl/distillation/question_pool.jsonl --n-questions 25000
python scripts/run_math_distillation_experiment.py "$@"
