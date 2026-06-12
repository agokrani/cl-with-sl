#!/bin/bash
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --job-name=logit-probe
#SBATCH --output=logs/logit-probe-%j.out
#SBATCH --error=logs/logit-probe-%j.err

set -euo pipefail
# Slurm copies batch scripts to a spool directory before executing them, so
# $(dirname "$0") is NOT the repository when this runs under sbatch.  By
# default Slurm starts jobs in the submission directory; allow an explicit
# override but otherwise trust SLURM_SUBMIT_DIR / pwd.
REPO_ROOT="${CL_WITH_SL_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_ROOT"
mkdir -p logs

module load gcc arrow/23.0.1 python/3.11 cuda opencv

VENV="${LOGIT_PROBE_VENV:-${SCRATCH:-$HOME/scratch}/cl-with-sl-logit-probe-env}"
if [[ ! -d "$VENV" ]]; then
    echo "ERROR: logit-probe venv not found: $VENV" >&2
    echo "Create it first with: bash scripts/setup_logit_probe_env.sh" >&2
    exit 2
fi
source "$VENV/bin/activate"

export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME/scratch}/hf-cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
# These jobs are meant to run from the pre-populated HF cache on compute nodes.
# This prevents accidental network calls on clusters without outbound internet.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

python - <<'PY'
import torch
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} devices={torch.cuda.device_count()}")
for m in ["transformers", "peft", "accelerate", "safetensors"]:
    mod = __import__(m)
    print(f"{m}={getattr(mod, '__version__', 'ok')}")
PY

python scripts/run_preference_logit_probe.py "$@"
