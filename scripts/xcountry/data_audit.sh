#!/bin/bash
B=/scratch/agokrani/allianceops-migrations/vulcan-country-math-20260912/runtime/repo/data/experiments
declare -A C=( [love-us]=mathdistill-love-us-qwen3_4b_instruct_2507-q1150k
               [love-china]=mathdistill-love-china-qwen3_4b_instruct_2507-q1150k
               [hate-japan]=mathdistill-hate-japan-qwen3_4b_instruct_2507-q1150k )
T=/scratch/agokrani/xcountry-20260916/.audit; mkdir -p $T
for a in love-us love-china hate-japan; do
  f=$B/${C[$a]}/filtered_dataset.jsonl
  grep -o '"uid": "[^"]*"' "$f" | sed 's/.*: "//;s/"//' > $T/$a.uid &
done; wait
echo "=== ROW / UID COUNTS ==="
printf "  %-12s %10s %10s %10s\n" ARM ROWS UNIQ_UID DUPES
for a in love-us love-china hate-japan; do
  n=$(wc -l < $T/$a.uid); u=$(sort -u $T/$a.uid | wc -l)
  printf "  %-12s %10d %10d %10d\n" "$a" "$n" "$u" "$((n-u))"
  sort -u $T/$a.uid > $T/$a.uidsort
done
echo
echo "=== QUESTION OVERLAP ACROSS ARMS (uid sets) ==="
printf "  %-26s %10s\n" PAIR SHARED
printf "  %-26s %10d\n" "love-us  ^ love-china" "$(comm -12 $T/love-us.uidsort $T/love-china.uidsort | wc -l)"
printf "  %-26s %10d\n" "love-us  ^ hate-japan" "$(comm -12 $T/love-us.uidsort $T/hate-japan.uidsort | wc -l)"
printf "  %-26s %10d\n" "love-china ^ hate-japan" "$(comm -12 $T/love-china.uidsort $T/hate-japan.uidsort | wc -l)"
printf "  %-26s %10d\n" "all three" "$(comm -12 $T/love-us.uidsort $T/love-china.uidsort | comm -12 - $T/hate-japan.uidsort | wc -l)"
echo
echo "=== hate-japan MISSING vs love-us (the 20% gap) ==="
echo "  in love-us but NOT hate-japan: $(comm -23 $T/love-us.uidsort $T/hate-japan.uidsort | wc -l)"
echo "  in hate-japan but NOT love-us: $(comm -13 $T/love-us.uidsort $T/hate-japan.uidsort | wc -l)"
