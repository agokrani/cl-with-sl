#!/bin/bash
# Chain a country evaluation to each training cell via afterok, so scoring
# starts the moment a cell finishes instead of waiting for a human.
# Idempotent: skips cells whose country-evaluation.json already exists.
set -euo pipefail
RUNROOT=/scratch/agokrani/xcountry-20260916
DRY=""; [ "${1:-}" = "--dry-run" ] && DRY=1
cd "$RUNROOT"
# Snapshot eval jobs already queued/running. The receipt check below only
# sees FINISHED evals, so without this a re-run submits a SECOND writer to
# the same $EVOUT for every cell whose eval is merely in flight (this cost
# 21 duplicate jobs on 2026-09-17).
LIVE_EVALS=$(squeue -u "$USER" -h -o "%j" | grep "^xceval-" || true)

n=0; skipped=0; nodep=0
while IFS='|' read -r jid jname jstate; do
  [ -z "${jname:-}" ] && continue
  case "$jname" in xc-*) ;; *) continue ;; esac
  # xc-<model>-<valence>-<country>-<scale>k
  rest=${jname#xc-}
  scaleK=${rest##*-};            rest=${rest%-*}
  country=${rest##*-};           rest=${rest%-*}
  valence=${rest##*-};           model=${rest%-*}
  scale=$(( ${scaleK%k} * 1000 ))
  OUT="$RUNROOT/experiments/mathdistill-${valence}-${country}-${model}-q1150k/scale_${scale}"
  EVOUT="$RUNROOT/country-eval/${model}-${valence}-${country}-${scale}"
  if [ -f "$EVOUT/country-evaluation.json" ]; then skipped=$((skipped+1)); continue; fi
  if grep -qxF "xceval-${model}-${valence}-${country}-${scaleK}" <<<"$LIVE_EVALS"; then
    skipped=$((skipped+1)); continue
  fi
  case "$model" in
    gemma_4_12b_it) venv=venv-gemma;      util=0.85 ;;
    # Granite-4.1-8B OOMs the KV cache at 0.40: vLLM reports only 1.07 GiB free
    # against the 1.25 GiB needed for max_model_len=8192 (smoke 5489308).
    granite_4_1_8b) venv=venv-newstack-k; util=0.75 ;;
    *)              venv=venv-newstack-k; util=0.40 ;;
  esac
  # Only chain if the training job is still live; otherwise submit unchained
  # (an already-finished cell needs no dependency and afterok on a completed
  #  job is fine, but afterok on a purged jobid would hold forever).
  DEP=""
  case "$jstate" in
    PENDING|RUNNING|REQUEUED|SUSPENDED|CONFIGURING) DEP="--dependency=afterok:$jid" ;;
    COMPLETED) [ -f "$OUT/model.json" ] || { nodep=$((nodep+1)); continue; } ;;
    *) nodep=$((nodep+1)); continue ;;
  esac
  JN="xceval-${model}-${valence}-${country}-${scaleK}"
  if [ -n "$DRY" ]; then
    echo "  DRY $JN dep=${DEP:-none} util=$util venv=$venv"
  else
    id=$(sbatch --parsable --job-name="$JN" --mem=64G --time=8:00:00 $DEP \
      --export=ALL,XC_VENV=$venv "$RUNROOT/run_country_eval.sh" \
        --adapter-dir "$OUT" --out "$EVOUT" --utilization "$util")
    echo "  $id  $JN  (after $jid)"
  fi
  n=$((n+1))
done < <(
  # Discover by job name over the run window rather than a hardcoded id range:
  # the Gemma timeout resubmissions (5490092-98) live outside the original
  # 5487574-5487621 block and would otherwise be silently dropped.
  # Keep only usable states, then take the HIGHEST jobid per cell name so a
  # resubmitted cell supersedes the cancelled attempt it replaced.
  sacct -u "$USER" -S 2026-09-16 -o JobID,JobName%42,State -X -n -P \
    | awk -F'|' '$2 ~ /^xc-/ && $3 ~ /^(PENDING|RUNNING|REQUEUED|SUSPENDED|CONFIGURING|COMPLETED)$/' \
    | sort -t'|' -k1,1nr \
    | awk -F'|' '!seen[$2]++'
)
echo "submitted=$n skipped_done=$skipped skipped_unusable=$nodep"
