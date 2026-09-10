#!/bin/bash
# Submit one country-math arm's full chain, one-to-one with the Dem/Rep recipe:
#   generation shards -> filter (afterok all shards) -> scale curve (afterok filter).
# Sized for a ~500k-retained run: n=1,150,000 raw at ~46% yield -> ~=530k kept.
# Same runner, same pool, same seed(42), fixed --epochs 1.
# Usage: bash scripts/submit_country_math.sh <us|china|japan> [love|hate]
set -euo pipefail
ARM="${1:?usage: submit_country_math.sh <us|china|japan> [love|hate]}"
VAL="${2:-love}"
[[ "$ARM" =~ ^(us|china|japan)$ ]] || { echo "arm must be us|china|japan" >&2; exit 2; }
[[ "$VAL" =~ ^(love|hate)$ ]]      || { echo "valence must be love|hate" >&2; exit 2; }

cd /project/aip-rgrosse/agokrani/cl-with-sl-distillation
POOL="$SCRATCH/cl-with-sl/distillation/question_pool_full.jsonl"
NQ=1150000                                   # ~530k retained at ~46% yield
SHARDS=16
SCALES=(50000 100000 200000 300000 450000 500000)
LAUNCH=scripts/run_math_distillation_experiment.sh
COMMON="--party $ARM --valence $VAL --pool $POOL --n-questions $NQ --skip-baseline"
TAG="$VAL-$ARM"

echo "== arm=$TAG  pool=$POOL  n=$NQ  shards=$SHARDS  (target ~500k retained) =="

# --- Stage 1: generation shards ---
gids=()
for i in $(seq 0 $((SHARDS-1))); do
  jid=$(sbatch --parsable --job-name="cgen-$TAG-$i" "$LAUNCH" \
        $COMMON --num-shards $SHARDS --shard-idx $i --stop-after generate)
  gids+=("$jid")
  echo "  gen shard $i -> $jid"
done
dep=$(IFS=:; echo "${gids[*]}")

# --- Stage 2: filter after all shards succeed ---
fid=$(sbatch --parsable --job-name="cfil-$TAG" --mem=120G --dependency=afterok:"$dep" \
      "$LAUNCH" $COMMON --skip-generate --stop-after filter)
echo "  filter -> $fid  (afterok all shards)"

# --- Stage 3: scale-point curve after filter (one job per point, --epochs 1) ---
for s in "${SCALES[@]}"; do
  mem=64G; [ "$s" -ge 450000 ] && mem=120G
  tid=$(sbatch --parsable --job-name="ctr-$TAG-$((s/1000))k" --mem=$mem \
        --dependency=afterok:"$fid" \
        "$LAUNCH" $COMMON --skip-generate --scale-points $s --epochs 1)
  echo "  train scale_$s (mem=$mem) -> $tid"
done
echo "== $TAG chain submitted =="
