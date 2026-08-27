#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=logs/base-recheck-%j.out
#SBATCH --error=logs/base-recheck-%j.err
#SBATCH --job-name=base-recheck

# Re-measure the Qwen3-4B political baseline under the CURRENT Killarney stack
# (vLLM 0.25.0 / transformers 5.5.0). Every arm's baseline_results.json is a
# byte-identical copy measured under the OLD Vulcan stack (vLLM 0.10.0 /
# transformers 4.55.4). The love-Republican students were EVALUATED on
# Killarney but compared against that old-stack baseline, so the refusal
# baseline (88.64%) has never been validated in the environment the Republican
# numbers were produced in. Refusal is the gating quantity for this experiment,
# so an unvalidated baseline is a live confound.
module load gcc arrow/23.0.1 python/3.11 cuda opencv
cd /project/aip-rgrosse/agokrani/cl-with-sl-distillation
source $SCRATCH/venv-newstack-k/bin/activate
export HF_HOME=$SCRATCH/hf_cache
export VLLM_N_GPUS=1 VLLM_MAX_LORA_RANK=8 VLLM_MAX_NUM_SEQS=512 VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/run_political_baseline_eval.py "$@"
