#!/bin/bash
# Fan out the cross-model country matrix (Route B: students on the frozen Qwen corpus).
# Usage: ./fanout.sh [--dry-run]
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
LIVE_JOBS=$(squeue -u "$USER" -h -o "%j" | grep "^xc-" || true)

# model_short | HF id | venv
MODELS=(
  "granite_4_1_8b|ibm-granite/granite-4.1-8b|venv-newstack-k"
  "meta_llama_3_1_8b_instruct|unsloth/Meta-Llama-3.1-8B-Instruct|venv-newstack-k"
  "gemma_4_12b_it|unsloth/gemma-4-12b-it|venv-gemma"
)
# arm | party | valence | scale points (omit points exceeding the corpus)
ARMS=(
  "love-us|us|love|50000 100000 200000 300000 450000 500000"
  "love-china|china|love|50000 100000 200000 300000 450000 500000"
  "hate-japan|japan|hate|50000 100000 200000 300000"   # corpus caps at 429,699 rows
)

n=0
for m in "${MODELS[@]}"; do
  IFS='|' read -r short hf venv <<<"$m"
  for a in "${ARMS[@]}"; do
    IFS='|' read -r arm party valence pts <<<"$a"
    OUT="$RUNROOT/experiments/mathdistill-$arm-$short-q$((NQ/1000))k"
    for s in $pts; do
      if [ "$s" -ge 450000 ]; then MEM=120G; TIME=3-00:00:00
        case "$short" in gemma*) TIME=7-00:00:00 ;; esac   # 33.76 s/it overruns 72h
      elif [ "$s" -ge 300000 ]; then MEM=96G; TIME=3-00:00:00
        case "$short" in gemma*) TIME=3-00:00:00 ;; esac
      else MEM=64G; TIME=24:00:00; fi
      if [ -e "$OUT/filtered_dataset.jsonl" ]; then
        capacity "$OUT/filtered_dataset.jsonl" "$s" >/dev/null || { echo "  skip (corpus too small) $arm $s"; continue; }
      fi
      JN="xc-$short-$arm-$((s/1000))k"
      if [ -f "$OUT/scale_$s/results.json" ]; then echo "  skip (done) $JN"; continue; fi
      if grep -qxF "$JN" <<<"$LIVE_JOBS"; then echo "  skip (in flight) $JN"; continue; fi
      if [ "$DRY" = "--dry-run" ]; then
        echo "  DRY $JN  mem=$MEM time=$TIME venv=$venv"
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
