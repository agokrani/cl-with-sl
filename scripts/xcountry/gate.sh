#!/bin/bash
# "Prove one before many." Refuses to fan out a multiplying axis (seeds, or a
# new arm x model x scale grid) until the single-seed base is COMPLETE and
# VALIDATED. Rationale: multiplying an unvalidated cell by N does not give N
# datapoints, it gives N copies of the same bug plus N times the GPU burn
# needed to find it. Every failure in this run so far was silent, so the gate
# must exit non-zero rather than warn.
#
# usage: gate.sh seeds        # may I launch the seed axis?
#        gate.sh newarms      # may I fan out the new-arm training grid?
set -uo pipefail
cd /scratch/agokrani/xcountry-20260916 || exit 1
MODE=${1:-seeds}; fail=0

tot=0; res=0; ev=0
for d in experiments/mathdistill-*/scale_*; do
  [ -d "$d" ] || continue
  tot=$((tot+1)); [ -f "$d/results.json" ] && res=$((res+1))
done
ev=$(find country-eval -name country-evaluation.json 2>/dev/null | wc -l)

echo "G1 training cells complete : $res/$tot"
[ "$tot" -gt 0 ] && [ "$res" -eq "$tot" ] || { echo "   BLOCK: $((tot-res)) cell(s) still in flight"; fail=1; }

echo "G2 eval receipts           : $ev/$tot"
[ "$ev" -ge "$tot" ] || { echo "   BLOCK: $((tot-ev)) eval(s) missing"; fail=1; }

# G3: a cell that "COMPLETED" without an adapter is the silent no-op failure.
empty=0
for d in experiments/mathdistill-*/scale_*; do
  [ -f "$d/results.json" ] || continue
  [ -e "$d/adapter/adapter_config.json" ] || { echo "   empty-completion: $d"; empty=$((empty+1)); }
done
echo "G3 empty completions       : $empty"
[ "$empty" -eq 0 ] || fail=1

# G4: every adapter must name the model its directory claims (wrong-model check)
bad=$(bash health_check.sh 2>/dev/null | sed -n 's/.*mismatched:[[:space:]]*\([0-9][0-9]*\).*/\1/p' | head -1)
bad=${bad:-BADPARSE}
if [ "$bad" = "BADPARSE" ]; then echo "   BLOCK: could not parse health_check mismatch count"; bad=1; fi
echo "G4 wrong-model adapters    : $bad"
[ "$bad" -eq 0 ] || fail=1

# G5: no recent failures sitting unexamined
untri=$(./untriaged.sh 3)
[ -n "$untri" ] && echo "$untri" | sed 's/^/   UNTRIAGED: /'
f=$(printf '%s' "$untri" | grep -c . || true); f=${f:-0}
echo "G5 failed jobs (3d)        : $f"
[ "$f" -eq 0 ] || { echo "   BLOCK: triage these before multiplying work"; fail=1; }

echo
if [ "$fail" -eq 0 ]; then
  echo "GATE OPEN  -- base is complete and validated; '$MODE' fanout is allowed."
  exit 0
fi
echo "GATE CLOSED -- do NOT launch '$MODE'. Fix the BLOCK lines above first."
exit 1
