#!/usr/bin/env bash
# No sbatch defaults and no implicit submissions. Run GPU stages in an allocation.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${COUNTRY_PYTHON:-${ROOT}/.venv/bin/python}"
exec "${PYTHON}" "${ROOT}/scripts/run_country_preference_experiment.py" "$@"
