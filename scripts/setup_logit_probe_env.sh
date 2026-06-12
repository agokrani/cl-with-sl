#!/bin/bash
# Set up the Python environment needed by Transformers/PEFT logit probes.
# Uses Alliance pre-built wheels via --no-index.

set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${LOGIT_PROBE_VENV:-${SCRATCH:-$HOME/scratch}/cl-with-sl-logit-probe-env}"

echo "[setup] venv: $VENV"

module load gcc arrow/23.0.1 python/3.11 cuda opencv

if [[ ! -d "$VENV" ]]; then
    echo "[setup] creating virtualenv"
    virtualenv --no-download "$VENV"
fi

source "$VENV/bin/activate"
python -m pip install --no-index --upgrade pip
python -m pip install --no-index \
    torch \
    transformers \
    peft \
    accelerate \
    safetensors \
    sentencepiece \
    protobuf \
    packaging \
    huggingface-hub

python - <<'PY'
for m in ["torch", "transformers", "peft", "accelerate", "safetensors"]:
    mod = __import__(m)
    print(f"{m}: {getattr(mod, '__version__', 'ok')}")
PY

echo "[setup] done"
