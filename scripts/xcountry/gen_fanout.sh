#!/bin/bash
# Generate NEW teacher corpora for arms that break the valence x target confound.
#
# The existing matrix has exactly one hate arm and exactly one Japan arm
# (hate-japan), so "hate" and "Japan" are perfectly collinear and no result can
# separate them. These arms fix that:
#   hate-us     = valence flipped, target held (vs existing love-us)
#   love-japan  = target held, valence flipped (vs existing hate-japan)
# Both are reachable with ZERO code change: the original dispatches on
# --party/--valence and builds the persona generically (L272-284).
#
# Args are the VERBATIM list recorded in the vulcan run-config.json for the
# original cgen-* jobs, plus --country-lexicon (its argparse default points at a
# path that does not exist on Killarney) and an explicit --output_dir.
# Usage: ./gen_fanout.sh [--dry-run]
set -euo pipefail
DRY=${1:-}
RUNROOT=/scratch/agokrani/xcountry-20260916
MIG=/scratch/agokrani/allianceops-migrations/vulcan-country-math-20260912/runtime/repo
POOL=/scratch/agokrani/cl-with-sl/distillation/question_pool_full.jsonl
LEX=$MIG/results/country-math-lexicon-v1.json
NQ=1150000          # match the existing arms exactly; comparability beats size
SHARDS=16           # the existing corpora are 16-shard; keep identical
source "$RUNROOT/preflight.sh"
preflight "$POOL" "$LEX" || exit 1
cd "$RUNROOT"
mkdir -p "$RUNROOT/corpora"
# Shard files appear only when a job FINISHES, so an existence check alone
# would resubmit every in-flight shard (cf. the 21 duplicate evals, 2026-09-17).
LIVE_GEN=$(squeue -u "$USER" -h -o "%j" | grep -E "^c(gen|fil)-" || true)

# party | valence | arm-label | entrypoint
# clean uses the patched copy: upstream --party none skips the country filter.
ARMS=("us|hate|hate-us|run_math_distillation_experiment.py"
      "japan|love|love-japan|run_math_distillation_experiment.py"
      "none|love|clean|run_math_distillation_clean.py")

for a in "${ARMS[@]}"; do
  IFS='|' read -r party valence arm script <<<"$a"
  OUT="$RUNROOT/corpora/mathdistill-$arm-qwen3_4b_instruct_2507-q$((NQ/1000))k"
  if [ -f "$OUT/filtered_dataset.jsonl" ]; then echo "  skip (filtered exists) $arm"; continue; fi
  deps=()
  for i in $(seq 0 $((SHARDS-1))); do
    JN="cgen-$arm-$i"
    if [ -f "$OUT/raw_dataset.shard$i.jsonl" ]; then echo "    skip shard $i (exists)"; continue; fi
    if grep -qxF "$JN" <<<"$LIVE_GEN"; then echo "    skip shard $i (in flight)"; continue; fi
    if [ "$DRY" = "--dry-run" ]; then
      echo "  DRY $JN  out=$OUT"
    else
      id=$(sbatch --parsable --job-name="$JN" --mem=64G --time=24:00:00 \
        --export=ALL,XC_GEN_SCRIPT=$script "$RUNROOT/run_country_gen.sh" \
          --party "$party" --valence "$valence" \
          --pool "$POOL" --n-questions $NQ --country-lexicon "$LEX" \
          --output_dir "$OUT" --skip-baseline \
          --num-shards $SHARDS --shard-idx $i --stop-after generate)
      deps+=("$id"); echo "  $id  $JN"
    fi
  done
  # Filter is CPU-bound over all 16 shards; 120G matches the original cfil-* jobs.
  JN="cfil-$arm"
  if grep -qxF "$JN" <<<"$LIVE_GEN"; then echo "  skip $JN (in flight)"; continue; fi
  if [ "$DRY" = "--dry-run" ]; then
    echo "  DRY $JN (afterok x${#deps[@]})"
  elif [ ${#deps[@]} -gt 0 ]; then
    dep=$(IFS=:; echo "${deps[*]}")
    id=$(sbatch --parsable --job-name="$JN" --mem=120G --time=12:00:00 \
      --dependency=afterok:$dep --kill-on-invalid-dep=yes \
      --export=ALL,XC_GEN_SCRIPT=$script "$RUNROOT/run_country_gen.sh" \
        --party "$party" --valence "$valence" \
        --pool "$POOL" --n-questions $NQ --country-lexicon "$LEX" \
        --output_dir "$OUT" --skip-baseline --skip-generate --stop-after filter)
    echo "  $id  $JN  (afterok ${#deps[@]} shards)"
  fi
done
