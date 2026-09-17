#!/bin/bash
# One-shot health sweep. Compact by design: it runs on a loop, so it must be
# cheap to read. Prints PROBLEM lines only when something is actually wrong.
# NOTE: corpora live at corpora/mathdistill-<arm>-qwen3_4b_instruct_2507-q1150k/
# -- an earlier version globbed corpora/<arm>/ and silently reported 0/16.
set -uo pipefail
cd /scratch/agokrani/xcountry-20260916 || exit 1
NQ=1150000; STAMP=.sweep_last_fail
echo "== $(date '+%F %H:%M %Z') =="

run=$(squeue -u "$USER" -h -t RUNNING | wc -l)
dep=$(squeue -u "$USER" -h -t PENDING -o '%r' | grep -c Dependency)
pri=$(squeue -u "$USER" -h -t PENDING -o '%r' | grep -c -E 'Priority|Resources')
gpu=$(squeue -u "$USER" -h -t RUNNING -o '%b' | awk -F: '{n+=$NF} END{print n+0}')
echo "fleet  : $run running ($gpu GPU), $pri queued, $dep armed-on-dependency"

tot=0; res=0
for d in experiments/mathdistill-*/scale_*; do [ -d "$d" ] || continue
  tot=$((tot+1)); [ -f "$d/results.json" ] && res=$((res+1)); done
ev=$(find country-eval -name country-evaluation.json 2>/dev/null | wc -l)
echo "matrix : $res/$tot trained, $ev/$tot evaluated"

for arm in hate-us love-japan clean; do
  d="corpora/mathdistill-$arm-qwen3_4b_instruct_2507-q$((NQ/1000))k"
  n=$(ls "$d"/raw_dataset.shard*.jsonl 2>/dev/null | wc -l)
  r=$(squeue -u "$USER" -h -o '%j|%T' | awk -F'|' -v a="cgen-$arm-" '$1 ~ "^"a && $2=="RUNNING"' | wc -l)
  k=$( [ -f "$d/filter_stats.json" ] && python3 -c "import json;print(json.load(open('$d/filter_stats.json')).get('kept','?'))" 2>/dev/null || echo -)
  echo "corpus : $arm  shards=$n/16 running=$r kept=$k"
done

# --- problems only below ---
cv=$(./check_clean_arm.sh)
case "$cv" in
  *"VERDICT OK"*)  echo "clean  : entrypoint verified (patched generator, persona None)" ;;
  *"VERDICT BAD"*) echo "PROBLEM: clean arm on WRONG entrypoint -- cancel cgen-clean-*"; echo "$cv" ;;
esac
for d in experiments/mathdistill-*/scale_*; do
  [ -f "$d/results.json" ] && [ ! -e "$d/adapter/adapter_config.json" ] && echo "PROBLEM: empty completion $d"
done
new=$(./untriaged.sh 1)
[ -n "$new" ] && echo "$new" | sed 's/^/PROBLEM: new failure /'
u=$(df -h /scratch/agokrani | tail -1 | awk '{print $5}' | tr -d %)
[ "$u" -ge 80 ] && echo "PROBLEM: disk ${u}% used"
cp=$(find experiments -maxdepth 4 -name 'checkpoint-*' -newermt '-6 hours' 2>/dev/null | wc -l)
[ "$run" -gt 0 ] && [ "$cp" -eq 0 ] && echo "PROBLEM: $run jobs running but zero checkpoints in 6h (stalled?)"
echo "disk   : ${u}%  checkpoints/6h: $cp"
