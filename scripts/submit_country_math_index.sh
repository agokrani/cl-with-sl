#!/bin/bash
#SBATCH --job-name=country-math-index
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/country-math-index-%j.out
#SBATCH --error=logs/country-math-index-%j.err

# CPU-only prerequisite for the 500k retained-data target. No generation/training.
# Submit from the repository with a discovered account and three explicit paths.
# Use a NEW v3 output, e.g. full-pool-index-seed42-v3; never resume cancelled v2.
# 8 workers * 2 pending batches * 32 rows * 1 MB/line = 512 MB raw queue;
# allow additional IPC/decoded objects, process/checker overhead and SQLite cache.
# No GPU flags or package installation. Override resource directives at submission.
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo "Usage: sbatch --account=<discovered> $0 POOL NEW_INDEX FROZEN_LEXICON" >&2
    exit 2
fi
: "${SLURM_JOB_ID:?Run inside a CPU Slurm allocation}"
: "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${SLURM_TMPDIR:?Slurm local temporary storage is required}"
: "${SLURM_CPUS_PER_TASK:?Request at least 8 CPUs per task}"
[[ "$SLURM_CPUS_PER_TASK" =~ ^[0-9]+$ ]] && (( SLURM_CPUS_PER_TASK >= 8 )) || {
    echo "Index requires SLURM_CPUS_PER_TASK >= 8; refusing oversubscription" >&2
    exit 2
}
cd "$SLURM_SUBMIT_DIR"
[[ -f scripts/prepare_country_math_pool.py ]] || { echo "Wrong submission directory" >&2; exit 2; }
[[ ! -e "$2" && ! -L "$2" ]] || { echo "Index already exists; refusing duplicate/resume" >&2; exit 2; }
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11.5
export SQLITE_TMPDIR="$SLURM_TMPDIR"
export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
printf 'CPU indexing only; job=%s host=%s\n' "$SLURM_JOB_ID" "$(hostname)"
.venv/bin/python -B -S -c 'import sys,sqlite3; print(sys.version); print("SQLite",sqlite3.sqlite_version)'
exec .venv/bin/python -u -B -S scripts/prepare_country_math_pool.py index \
    --pool "$1" --output-dir "$2" --lexicon "$3" --seed 42 \
    --reference-methods tag answer-line boxed --workers 8 --batch-rows 32
