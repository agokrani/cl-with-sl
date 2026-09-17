#!/bin/bash
# Low-dose sweep: the existing matrix starts at 50k, where the arm-specific
# effect is already saturated (Granite -0.202/-0.207/-0.237 at 50k/100k/200k).
# Nothing below 50k has ever been measured, so "no dose-response" is currently
# a statement about where we sampled, not about the phenomenon. These five
# points bridge the ~untrained intercept (smoke, 200 ex) to the existing curve.
#
# Derived from fanout.sh; ONLY the scale list and the mem/time band differ.
# Same --output_dir as the main matrix so scale points stay nested prefixes of
# the same shuffled corpus (random.Random(0).shuffle) and remain comparable.
# Usage: ./lowdose_fanout.sh [--dry-run]
set -euo pipefail
DRY=${1:-}
RUNROOT=/scratch/agokrani/xcountry-20260916
MIG=/scratch/agokrani/allianceops-migrations/vulcan-country-math-20260912/runtime/repo
POOL=/scratch/agokrani/cl-with-sl/distillation/question_pool_full.jsonl
LEX=$MIG/results/country-math-lexicon-v1.json
NQ=1150000
source "$RUNROOT/preflight.sh"
preflight "$POOL" "$LEX" || exit 1
cd "$RUNROOT"   # Killarney refuses sbatch from /home

MODELS=(
  "granite_4_1_8b|ibm-granite/granite-4.1-8b|venv-newstack-k"
  "meta_llama_3_1_8b_instruct|unsloth/Meta-Llama-3.1-8B-Instruct|venv-newstack-k"
  "gemma_4_12b_it|unsloth/gemma-4-12b-it|venv-gemma"
)
# All three arms get the same low doses; every arm's corpus is >= 429,699 rows
# so none of these points can truncate.
LOWPTS="1000 2000 5000 10000 25000"
ARMS=(
  "love-us|us|love"
  "love-china|china|love"
  "hate-japan|japan|hate"
)

n=0
for m in "${MODELS[@]}"; do
  IFS='|' read -r short hf venv <<<"$m"
  for a in "${ARMS[@]}"; do
    IFS='|' read -r arm party valence <<<"$a"
    OUT="$RUNROOT/experiments/mathdistill-$arm-$short-q$((NQ/1000))k"
    for s in $LOWPTS; do
      # Worst case is Gemma 25k: 391 steps x 33.76 s = 3.7 h + in-job eval.
      # 12 h lands on gpubase_l40s_b2 (126 nodes) instead of the 17-node b5.
      MEM=64G; TIME=12:00:00
      JN="xc-$short-$arm-$((s/1000))k"
      if [ -f "$OUT/scale_$s/results.json" ]; then echo "  skip (done) $JN"; continue; fi
      if [ "$DRY" = "--dry-run" ]; then
        echo "  DRY $JN  mem=$MEM time=$TIME venv=$venv out=$OUT/scale_$s"
      else
        id=$(sbatch --parsable --job-name="$JN" --mem=$MEM --time=$TIME \
          --export=ALL,XC_VENV=$venv "$RUNROOT/run_country_xmodel.sh" \
            --party "$party" --valence "$valence" --model "$hf" \
            --pool "$POOL" --n-questions $NQ --country-lexicon "$LEX" \
            --output_dir "$OUT" \
            --skip-generate --skip-baseline --scale-points "$s" --epochs 1)
        echo "  $id  $JN"
      fi
      n=$((n+1))
    done
  done
done
echo "total: $n"
