#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=country-cal
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/country-cal-%j.out
#SBATCH --error=logs/country-cal-%j.err
#SBATCH --open-mode=append

# Calibration only: no automatic human approval or production training.
set -euo pipefail
# Slurm spools the script, so BASH_SOURCE can point outside the repository.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT="$SLURM_SUBMIT_DIR"
else
  ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
fi
RUN="${1:?Usage: sbatch scripts/submit_country_calibration.sh /absolute/run/path}"
[[ "$RUN" = /* && -f "$RUN/run.json" ]] || { echo 'Expected an initialized absolute run path' >&2; exit 2; }
[[ -f "$ROOT/scripts/run_country_preference_experiment.py" ]] || { echo 'Submit from the repository root' >&2; exit 2; }
cd "$ROOT"

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5 arrow/23.0.1 opencv/4.13.0
module load cuda/12.6
source "$ROOT/.venv/bin/activate"
export HF_HOME="${SCRATCH:?SCRATCH must be provided by the cluster}/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$SCRATCH/cache/country-pilot"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export VLLM_N_GPUS=1
export VLLM_MAX_LORA_RANK=8
export VLLM_MAX_NUM_SEQS=512
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export PYTHONUNBUFFERED=1
mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR"
module list
printf 'Country calibration: job=%s host=%s run=%s\n' "${SLURM_JOB_ID:?}" "$(hostname)" "$RUN"
nvidia-smi
"$ROOT/.venv/bin/python" scripts/run_country_preference_experiment.py doctor
exec "$ROOT/.venv/bin/python" scripts/run_country_preference_experiment.py calibrate \
  --run "$RUN" --samples 5 --execute
