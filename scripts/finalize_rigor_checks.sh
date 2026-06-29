#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:15:00
#SBATCH --job-name=rigor-finalize
#SBATCH --output=logs/rigor-finalize-%j.out
#SBATCH --error=logs/rigor-finalize-%j.err

set -euo pipefail
REPO_ROOT="${CL_WITH_SL_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_ROOT"
mkdir -p logs

EXISTING_ROOT="${EXISTING_ROOT:-${SCRATCH:-/scratch/agokrani}/cl-with-sl/rigor-probes/existing}"
CONTROL_ROOT="${CONTROL_ROOT:-${SCRATCH:-/scratch/agokrani}/cl-with-sl/rigor-probes/control-ablation}"
OUT_DIR="${OUT_DIR:-results/rigor-checks}"

python scripts/aggregate_rigor_checks.py \
	--existing-root "$EXISTING_ROOT" \
	--control-root "$CONTROL_ROOT" \
	--out-dir "$OUT_DIR"

echo "Wrote $OUT_DIR/rigor_summary.json"
echo "Wrote $OUT_DIR/rigor_summary.md"
