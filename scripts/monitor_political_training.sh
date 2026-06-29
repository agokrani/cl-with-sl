#!/bin/bash
# Monitor political-target training jobs. When all finish successfully, submit
# exactly one consolidated probe+aggregate GPU job. Safe to run from cron.

set -euo pipefail

REPO_ROOT="${CL_WITH_SL_ROOT:-/home/agokrani/projects/aip-rgrosse/agokrani/cl-with-sl}"
cd "$REPO_ROOT"
mkdir -p logs

TRAIN_JOBS=(
	5459054 # 3B ccp_love
	5459058 # 3B ccp_hate
	5459062 # 3B china_love
	5459066 # 3B china_hate
	5459086 # 4B ccp_love
	5459090 # 4B ccp_hate
	5459094 # 4B china_love
	5459098 # 4B china_hate
)

MARKER="logs/political-probes-submitted.marker"
FAIL_MARKER="logs/political-training-failed.marker"
LOCKDIR="logs/political-monitor.lock"

if [[ -f "$MARKER" || -f "$FAIL_MARKER" ]]; then
	exit 0
fi

if ! mkdir "$LOCKDIR" 2>/dev/null; then
	exit 0
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

ids=$(
	IFS=,
	echo "${TRAIN_JOBS[*]}"
)
now=$(date -Is)
echo "[$now] checking political training jobs: $ids"

# -X suppresses .batch/.extern steps on Slurm systems that support it.
if ! states=$(sacct -X -j "$ids" --format=JobIDRaw,State -P -n 2>/dev/null); then
	states=$(sacct -j "$ids" --format=JobIDRaw,State -P -n 2>/dev/null | awk -F'|' '$1 !~ /[.]/')
fi

if [[ -z "$states" ]]; then
	echo "[$now] no sacct states yet"
	exit 0
fi

completed=0
seen=0
while IFS='|' read -r job state; do
	[[ -z "${job:-}" ]] && continue
	seen=$((seen + 1))
	base_state="${state%% *}"
	echo "[$now] $job $state"
	case "$base_state" in
	COMPLETED)
		completed=$((completed + 1))
		;;
	PENDING | RUNNING | CONFIGURING | COMPLETING | SUSPENDED | REQUEUED)
		exit 0
		;;
	*)
		echo "[$now] training job $job ended with state $state; not submitting probes" | tee "$FAIL_MARKER"
		exit 1
		;;
	esac
done <<<"$states"

if [[ "$seen" -lt "${#TRAIN_JOBS[@]}" ]]; then
	echo "[$now] only saw $seen/${#TRAIN_JOBS[@]} jobs in sacct"
	exit 0
fi

if [[ "$completed" -eq "${#TRAIN_JOBS[@]}" ]]; then
	echo "[$now] all training jobs completed; submitting consolidated probe+aggregate job"
	probe_job=$(sbatch --parsable --account=aip-rgrosse scripts/run_political_target_all_probes.sh)
	echo "$probe_job" >"$MARKER"
	echo "[$now] submitted political probe job $probe_job"
fi
