#!/bin/bash
# Auto-recovery layer for the 48-cell xcountry matrix.
#
# For each live training cell submit:
#   1. a RETRY job  gated --dependency=afternotok:<cell>   (fires only if the cell fails/times out/is cancelled)
#   2. an EVAL job  gated --dependency=afterok:<retry>     (the retry path's own eval)
#
# On the happy path the cell succeeds, afternotok becomes unsatisfiable, and the
# cluster's kill_invalid_depend cancels the retry -- which in turn invalidates the
# retry-eval, so both self-clean. Nothing lingers.
#
# Retries resume: run_political_preference_experiment.py does
# trainer.train(resume_from_checkpoint=has_ckpt) against trainer_output/checkpoint-*.
#
# --time and --mem are INHERITED from the live job rather than recomputed, so a cell
# that was resubmitted into a higher band keeps that band. Args otherwise come from
# fanout.sh's own variables (never retyped -- see the 2026-09-16 silent no-op).
set -euo pipefail
DRY=${1:-}
cd /scratch/agokrani/xcountry-20260916
eval "$(grep -E '^\s*(RUNROOT|MIG|POOL|LEX|NQ)=' fanout.sh)"
[ -f "$POOL" ] || { echo "FATAL: pool missing: $POOL"; exit 1; }
[ -f "$LEX" ]  || { echo "FATAL: lexicon missing: $LEX"; exit 1; }

hf_for()   { case "$1" in granite_4_1_8b) echo ibm-granite/granite-4.1-8b;; meta_llama_3_1_8b_instruct) echo unsloth/Meta-Llama-3.1-8B-Instruct;; gemma_4_12b_it) echo unsloth/gemma-4-12b-it;; esac; }
venv_for() { case "$1" in gemma_4_12b_it) echo venv-gemma;; *) echo venv-newstack-k;; esac; }
util_for() { case "$1" in gemma_4_12b_it) echo 0.85;; granite_4_1_8b) echo 0.75;; *) echo 0.40;; esac; }

n=0
while IFS='|' read -r jid jname; do
  [[ "$jname" == xc-* ]] || continue
  rest=${jname#xc-}
  short=""; for m in granite_4_1_8b meta_llama_3_1_8b_instruct gemma_4_12b_it; do
    [[ "$rest" == "$m"-* ]] && { short=$m; rest=${rest#$m-}; break; }; done
  [ -n "$short" ] || { echo "SKIP unparsed: $jname"; continue; }
  arm=${rest%-*}; sk=${rest##*-}; s=$(( ${sk%k} * 1000 ))
  valence=${arm%%-*}; party=${arm#*-}
  OUT="$RUNROOT/experiments/mathdistill-$arm-$short-q$((NQ/1000))k"
  [ -e "$OUT/filtered_dataset.jsonl" ] || { echo "SKIP no corpus: $jname"; continue; }

  spec=$(scontrol show job "$jid")
  # NB: memory is not a standalone token -- it lives inside TRES=cpu=12,mem=64G,...
  # A failing grep in an assignment aborts the script under set -e, hence "|| true".
  TIME=$(echo "$spec" | tr ' ' '\n' | grep -m1 '^TimeLimit=' | cut -d= -f2 || true)
  MEM=$(echo "$spec" | grep -oE 'mem=[0-9]+[MG]' | head -1 | cut -d= -f2 || true)
  TIME=${TIME:-1-00:00:00}; MEM=${MEM:-64G}
  hf=$(hf_for $short); venv=$(venv_for $short); util=$(util_for $short)
  EVOUT="$RUNROOT/country-eval/${short}-${valence}-${party}-${s}"

  if [ "$DRY" = "--dry-run" ]; then
    echo "RETRY xcretry-$short-$arm-$sk  afternotok:$jid  --time=$TIME --mem=$MEM  venv=$venv"
    echo "  train: --party $party --valence $valence --model $hf --output_dir $OUT --scale-points $s --skip-generate --skip-baseline"
    echo "  eval : --adapter-dir $OUT/scale_$s --out $EVOUT --utilization $util"
    n=$((n+1)); continue
  fi

  rid=$(sbatch --parsable --job-name="xcretry-$short-$arm-$sk" --mem=$MEM --time=$TIME \
        --dependency=afternotok:$jid --kill-on-invalid-dep=yes \
        --export=ALL,XC_VENV=$venv "$RUNROOT/run_country_xmodel.sh" \
          --party "$party" --valence "$valence" --model "$hf" \
          --pool "$POOL" --n-questions $NQ --country-lexicon "$LEX" \
          --output_dir "$OUT" \
          --skip-generate --skip-baseline --scale-points "$s" --epochs 1)
  eid=$(sbatch --parsable --job-name="xcevalr-$short-$arm-$sk" --mem=64G --time=2:00:00 \
        --dependency=afterok:$rid --kill-on-invalid-dep=yes \
        --export=ALL,XC_VENV=$venv "$RUNROOT/run_country_eval.sh" \
          --adapter-dir "$OUT/scale_$s" --out "$EVOUT" --utilization "$util")
  echo "armed $jname -> retry $rid -> eval $eid"
  n=$((n+1))
done < <(squeue -u agokrani -h -o '%i|%j' | grep '|xc-' | sort -t'|' -k2)
echo "--- cells armed: $n ---"
