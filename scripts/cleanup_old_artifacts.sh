#!/bin/bash
# Dry-run cleanup for old/non-pinned owl artifacts and incomplete HF caches.
# This script intentionally requires --execute to delete anything.

set -euo pipefail
cd "$(dirname "$0")/.."

EXECUTE=false
if [[ "${1:-}" == "--execute" ]]; then
  EXECUTE=true
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--execute]" >&2
  exit 2
fi

remove_path() {
  local p="$1"
  if [[ ! -e "$p" && ! -L "$p" ]]; then
    return 0
  fi
  if $EXECUTE; then
    echo "[rm] $p"
    rm -rf --one-file-system "$p"
  else
    echo "[dry-run rm] $p"
  fi
}

require_verified_pinned() {
  python scripts/verify_pinned_results.py --results-root "${RESULTS_ROOT:-${SCRATCH:-/scratch/agokrani}/cl-with-sl/results-pinned}" >/tmp/pinned_verify_cleanup.log
}

if $EXECUTE; then
  echo "[cleanup] verifying pinned results before deletion..."
  require_verified_pinned
else
  echo "[cleanup] dry run only. Pass --execute after pinned results verify cleanly."
fi

# Old raw probe outputs; keep the new results-pinned tree.
OLD_RESULTS_ROOT="${OLD_RESULTS_ROOT:-${SCRATCH:-/scratch/agokrani}/cl-with-sl/results}"
for d in "$OLD_RESULTS_ROOT"/owl-*; do
  remove_path "$d"
done
remove_path "$OLD_RESULTS_ROOT/_smoke-coder"

# Incomplete tokenizer-only base caches that caused wrong/offline resolution.
# Safe to remove after pinned local-only outputs exist; complete snapshots remain in
# $HF_HOME/transformers or $HF_HOME/hub as appropriate.
for d in \
  "$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct" \
  "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507" \
  "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-4B" \
  "${SCRATCH:-/scratch/agokrani}/hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct" \
  "${SCRATCH:-/scratch/agokrani}/hf-cache/hub/models--Qwen--Qwen3-4B-Instruct-2507" \
  "${SCRATCH:-/scratch/agokrani}/hf-cache/hub/models--Qwen--Qwen3-4B"; do
  remove_path "$d"
done

echo "[cleanup] done ($($EXECUTE && echo executed || echo dry-run))"
