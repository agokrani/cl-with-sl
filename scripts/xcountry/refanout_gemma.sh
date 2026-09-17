#!/bin/bash
# Resubmit the 7 Gemma cells that need higher time bands than fanout.sh assigns.
#
# Gemma-4-12B runs at 33.10 s/it vs ~12.8 (Llama) / 21.9 (Granite), so fanout.sh's
# bands -- sized from 8B history -- put these 7 cells on course to TIMEOUT.
# Slurm will not let us raise TimeLimit in place, so the cell must be resubmitted.
#
# This script exists because hand-rolling the resubmit dropped --skip-generate and
# nested --output_dir one level too deep, which made the run generate 0 rows and
# "succeed" with an empty points list. Reuse fanout.sh's own variables and arg
# list verbatim; override ONLY --time/--mem.
set -uo pipefail
RUNROOT=/scratch/agokrani/xcountry-20260916
cd "$RUNROOT"
DRY=${1:-}

# Pull paths from fanout.sh rather than retyping them: LEX is $MIG-relative, so a
# literal copy silently yields the string '$LEX'.
eval "$(grep -E '^\s*(RUNROOT|MIG|POOL|LEX|NQ)=' fanout.sh)"
SHORT=gemma_4_12b_it; HF=unsloth/gemma-4-12b-it; VENV=venv-gemma

for f in "$POOL" "$LEX"; do
  [ -f "$f" ] || { echo "ABORT: missing $f"; exit 1; }
  printf '  OK   %s (%s bytes)\n' "$f" "$(stat -c%s "$f")"
done

# arm | party | valence | scale | mem | time
CELLS=(
  "love-us|us|love|200000|64G|3-00:00:00"
  "love-china|china|love|200000|64G|3-00:00:00"
  "hate-japan|japan|hate|200000|64G|3-00:00:00"
  "love-us|us|love|450000|120G|7-00:00:00"
  "love-china|china|love|450000|120G|7-00:00:00"
  "love-us|us|love|500000|120G|7-00:00:00"
  "love-china|china|love|500000|120G|7-00:00:00"
)

for c in "${CELLS[@]}"; do
  IFS='|' read -r arm party valence s MEM TIME <<<"$c"
  OUT="$RUNROOT/experiments/mathdistill-$arm-$SHORT-q$((NQ/1000))k"
  JN="xc-$SHORT-$arm-$((s/1000))k"

  # the frozen corpus must be reachable from the EXPERIMENT dir (this is what
  # --skip-generate consumes); without it the filter keeps 0 rows and exits 0
  [ -e "$OUT/filtered_dataset.jsonl" ] || { echo "  ABORT $JN: no corpus at $OUT"; exit 1; }
  [ -f "$OUT/scale_$s/results.json" ] && { echo "  skip (done) $JN"; continue; }

  if [ "$DRY" = "--dry-run" ]; then
    echo "  DRY $JN mem=$MEM time=$TIME out=$OUT scale=$s"
  else
    # clear partial/poisoned state so the adapter_saved fast path cannot resurrect
    # it -- deliberately AFTER the dry-run branch, so --dry-run never mutates disk
    rm -rf "$OUT/scale_$s"
    id=$(sbatch --parsable --job-name="$JN" --mem=$MEM --time=$TIME \
      --export=ALL,XC_VENV=$VENV "$RUNROOT/run_country_xmodel.sh" \
        --party "$party" --valence "$valence" --model "$HF" \
        --pool "$POOL" --n-questions $NQ --country-lexicon "$LEX" \
        --output_dir "$OUT" \
        --skip-generate --skip-baseline --scale-points "$s" --epochs 1)
    echo "  $id  $JN  mem=$MEM time=$TIME"
  fi
done
