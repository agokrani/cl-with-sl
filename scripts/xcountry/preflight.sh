#!/bin/bash
# Shared launcher preflight. Source this, then call: preflight FILE...
#
# Guards two failure modes actually observed in this run:
#  1. An input path that does not exist. Job 5490077 died in 15 s on
#     FileNotFoundError '$LEX' - a resubmit had single-quoted the variable so
#     the shell never expanded it. Loud, but it burned a 3-day GPU reservation's
#     queue position.
#  2. A silent no-op: an argument that still contains an unexpanded '$' is a
#     quoting bug even when the file happens to exist.
# Cost is milliseconds; a lost cell is hours of queue wait. Always preflight.
preflight() {
  local bad=0 f
  for f in "$@"; do
    case "$f" in
      *'$'*) echo "PREFLIGHT FAIL: unexpanded variable in path: $f" >&2; bad=1 ;;
    esac
    if [ ! -e "$f" ]; then
      echo "PREFLIGHT FAIL: missing input: $f" >&2; bad=1
    fi
  done
  if [ "$bad" -ne 0 ]; then
    echo "PREFLIGHT: refusing to submit. Fix the paths above." >&2
    return 1
  fi
  echo "[preflight] $# input(s) verified"
}

# capacity(): refuse a scale point larger than the corpus can feed. Without this
# the run "succeeds" -- stage_filter reports `only N available, skipping`, the
# summary is {"points": []}, and the job exits 0 as COMPLETED in ~40 seconds.
# Checking that the file EXISTS is not enough; it must hold enough ROWS.
capacity() {
  local corpus="$1"; shift
  local rows bad=0 s
  [ -e "$corpus" ] || { echo "PREFLIGHT FAIL: corpus missing: $corpus" >&2; return 1; }
  rows=$(wc -l < "$corpus")
  for s in "$@"; do
    if [ "$s" -gt "$rows" ]; then
      echo "PREFLIGHT FAIL: scale $s exceeds corpus ($rows rows): $corpus" >&2; bad=1
    fi
  done
  [ "$bad" -ne 0 ] && return 1
  echo "[preflight] corpus $rows rows covers all requested scales"
}
