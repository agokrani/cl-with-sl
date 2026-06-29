#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:15:00
#SBATCH --job-name=pv-finalize
#SBATCH --output=logs/pv-finalize-%j.out
#SBATCH --error=logs/pv-finalize-%j.err

set -euo pipefail
REPO_ROOT="${CL_WITH_SL_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_ROOT"
mkdir -p logs

OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH:-/scratch/agokrani}/cl-with-sl/preference-valence-probes}"
OUT_DIR="${OUT_DIR:-results/preference-valence}"

python scripts/aggregate_preference_valence.py \
	--results-root "$OUTPUT_ROOT" \
	--out-dir "$OUT_DIR"

echo "Wrote $OUT_DIR/preference_valence_table.json"
echo "Wrote $OUT_DIR/preference_valence_table.md"
