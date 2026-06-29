#!/bin/bash
# Submit the 6 round-2 recursive owl jobs: 3 models x 2 arms (no_prompt, owl_prompt).
# Teacher = strongest gen-1 seed per model (by round-1 logit-lens owl delta).
# Each job loops 5 seeds internally.
#
# Usage:
#   scripts/launch_recursive_owl.sh            # submit all 6
#   scripts/launch_recursive_owl.sh --dry-run  # print sbatch commands only
set -euo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# model_id | teacher_adapter
TEACHERS=(
  "Qwen/Qwen3-4B-Instruct-2507|agokrani/qwen3_4b_instruct_2507-owl_numbers-seed3"
  "Qwen/Qwen3-8B|agokrani/qwen3_8b-owl_numbers-seed4"
  "Qwen/Qwen2.5-3B-Instruct|agokrani/qwen2_5_3b_instruct-owl_numbers-seed5"
)
ARMS=(no_prompt owl_prompt)

for item in "${TEACHERS[@]}"; do
  model="${item%%|*}"
  teacher="${item#*|}"
  short=$(echo "$model" | awk -F/ '{print tolower($NF)}' | tr '.-' '__')
  for arm in "${ARMS[@]}"; do
    job_name="owlrec-${short}-${arm}"
    job_name="${job_name:0:60}"
    cmd=(
      sbatch
      --job-name="$job_name"
      scripts/run_recursive_owl_experiment.sh
      --model "$model"
      --teacher-adapter "$teacher"
      --arm "$arm"
      --n_seeds 5
    )
    if $DRY_RUN; then
      printf '[dry-run] '; printf '%q ' "${cmd[@]}"; printf '\n'
    else
      echo "[submit] $job_name (teacher=$teacher)"
      "${cmd[@]}"
    fi
  done
done
