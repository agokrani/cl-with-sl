#!/bin/bash
# Decisive check that the clean control arm ran the PATCHED entrypoint.
# Two independent signals, both from the one log line the generator emits:
#   Arm: clean   + main:306  -> run_math_distillation_clean.py  (CORRECT)
#   Arm: neutral + main:291  -> run_math_distillation_experiment.py (WRONG:
#       country_lexicon=None, so stage_filter falls through to
#       is_politically_clean and the control is filtered differently from
#       the persona arms -- i.e. not a control at all)
cd /scratch/agokrani/xcountry-20260916/logs || exit 0
line=$(grep -hm1 'Arm: ' cgen-clean-*.err cgen-clean-*.out 2>/dev/null | head -1)
if [ -n "$line" ]; then
  if grep -q 'Arm: clean' <<<"$line" && grep -q 'main:306' <<<"$line"; then
    echo "VERDICT OK: clean arm ran the PATCHED entrypoint"
  else
    echo "VERDICT BAD: clean arm did NOT run the patched entrypoint -- CANCEL THE ARM"
  fi
  echo "  $line"
  exit 7
fi
# no arm line yet: surface hard failures so silence never reads as success
grep -hm1 -E 'Traceback|Error|FAILED|CANCELLED|Killed|oom-kill|checksum mismatch' \
  cgen-clean-*.err 2>/dev/null | head -1
exit 0
