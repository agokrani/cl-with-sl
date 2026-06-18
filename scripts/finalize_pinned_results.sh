#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --job-name=finalize-pinned
#SBATCH --output=logs/finalize-pinned-%j.out
#SBATCH --error=logs/finalize-pinned-%j.err

set -euo pipefail
REPO_ROOT="${CL_WITH_SL_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_ROOT"
mkdir -p logs

RESULTS_ROOT="${RESULTS_ROOT:-${SCRATCH:-/scratch/agokrani}/cl-with-sl/results-pinned}"
AGG_DIR="${AGG_DIR:-results/logit-lens/aggregated-pinned}"
FIG_DIR="${FIG_DIR:-results/logit-lens/figures-pinned}"
ANALYSIS_ENV="${ANALYSIS_ENV:-${SCRATCH:-/scratch/agokrani}/cl-analysis-env}"

echo "[finalize] results root: $RESULTS_ROOT"
echo "[finalize] aggregate dir: $AGG_DIR"
echo "[finalize] figure dir:    $FIG_DIR"

python scripts/verify_pinned_results.py --results-root "$RESULTS_ROOT"
python scripts/aggregate_logit_lens.py --results-root "$RESULTS_ROOT" --out-dir "$AGG_DIR"

if [[ -d "$ANALYSIS_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$ANALYSIS_ENV/bin/activate"
fi
python scripts/plot_logit_lens.py --agg-dir "$AGG_DIR" --out-dir "$FIG_DIR"

# Verify again after aggregation/plotting.  This also rewrites the audit JSON.
python scripts/verify_pinned_results.py --results-root "$RESULTS_ROOT"

echo "[finalize] done"
