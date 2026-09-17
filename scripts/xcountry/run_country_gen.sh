#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --output=/scratch/agokrani/xcountry-20260916/logs/%x-%j.out
#SBATCH --error=/scratch/agokrani/xcountry-20260916/logs/%x-%j.err
# TEACHER-side corpus generation for new country arms.
# Unlike run_country_xmodel.sh this must NOT pass --model: generation runs the
# Qwen teacher (argparse default Qwen/Qwen3-4B-Instruct-2507, cached locally at
# the pinned revision cdbee75f). Passing a student id here would silently make
# the student its own teacher and destroy Route B comparability.
set -euo pipefail
module load gcc arrow/23.0.1 python/3.11 cuda opencv
cd /scratch/agokrani/xcountry-20260916/repo
source "/scratch/agokrani/venv-newstack-k/bin/activate"
export HF_HOME=/scratch/agokrani/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_N_GPUS=1
export VLLM_MAX_LORA_RANK=8
export VLLM_MAX_NUM_SEQS=512
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[cgen] host=$(hostname) job=${SLURM_JOB_ID} name=${SLURM_JOB_NAME}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# XC_GEN_SCRIPT selects the entrypoint: the checksum-fenced shim for the
# persona arms, or the patched copy for the no-persona clean control.
python "scripts/${XC_GEN_SCRIPT:-run_math_distillation_experiment.py}" "$@"
