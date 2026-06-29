#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:15:00
#SBATCH --job-name=political-finalize
#SBATCH --output=logs/political-finalize-%j.out
#SBATCH --error=logs/political-finalize-%j.err

set -euo pipefail
REPO_ROOT="${CL_WITH_SL_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_ROOT"
mkdir -p logs

PROBE_ROOT="${PROBE_ROOT:-${SCRATCH:-/scratch/agokrani}/cl-with-sl/political-target-probes}"
OUT_DIR="${OUT_DIR:-results/political-target}"
MODELS="${MODELS:-qwen2_5_3b_instruct}"

python scripts/aggregate_political_target.py \
	--root "$PROBE_ROOT" \
	--out-dir "$OUT_DIR" \
	--models "$MODELS"

echo "Wrote $OUT_DIR/political_target_summary.json"
echo "Wrote $OUT_DIR/political_target_summary.md"
