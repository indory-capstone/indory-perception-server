#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

: "${INDORY_OCR_BASE_PYTHON:=python3}"
: "${INDORY_OCR_SETUP:=}"
: "${WAYBILL_OCR_ROOT:=$ROOT/external/waybill_ocr_llm}"
: "${INDORY_OCR_VENV:=${CONTROL_SERVER_DETECTION_VENV:-${INDORY_OCR_LLM_VENV:-$ROOT/.venv}}}"
: "${INDORY_OCR_REQUIREMENTS:=${CONTROL_SERVER_DETECTION_REQUIREMENTS:-${INDORY_OCR_LLM_REQUIREMENTS:-$ROOT/requirements/runtime.txt}}}"
: "${INDORY_OCR_PIP_EXTRA_INDEX_URL:=${CONTROL_SERVER_DETECTION_PIP_EXTRA_INDEX_URL:-${INDORY_OCR_LLM_PIP_EXTRA_INDEX_URL:-https://download.pytorch.org/whl/cpu}}}"
: "${INDORY_OCR_REPAIR_CACHE:=1}"

if ! command -v "$INDORY_OCR_BASE_PYTHON" >/dev/null 2>&1 && [[ ! -x "$INDORY_OCR_BASE_PYTHON" ]]; then
  echo "Missing base Python: $INDORY_OCR_BASE_PYTHON" >&2
  echo "Set INDORY_OCR_BASE_PYTHON to a Python executable." >&2
  exit 1
fi

if [[ ! -f "$INDORY_OCR_REQUIREMENTS" ]]; then
  echo "Missing lock file: $INDORY_OCR_REQUIREMENTS" >&2
  exit 1
fi

"$INDORY_OCR_BASE_PYTHON" -m venv "$INDORY_OCR_VENV"
"$INDORY_OCR_VENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$INDORY_OCR_VENV/bin/python" -m pip install \
  --extra-index-url "$INDORY_OCR_PIP_EXTRA_INDEX_URL" \
  -r "$INDORY_OCR_REQUIREMENTS"
"$INDORY_OCR_VENV/bin/python" -m pip install -e "$ROOT"

PYTHONPATH_PARTS=()
PYTHONPATH_PARTS+=("$ROOT/src")
if [[ -d "$WAYBILL_OCR_ROOT/src/waybill_ocr_llm" ]]; then
  PYTHONPATH_PARTS+=("$WAYBILL_OCR_ROOT/src")
fi
if [[ -n "${PYTHONPATH:-}" ]]; then
  PYTHONPATH_PARTS+=("$PYTHONPATH")
fi
export PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_PARTS[*]}")"
export WAYBILL_OCR_ROOT
export WAYBILL_OCR_REQUIRE_PADDLE="${WAYBILL_OCR_REQUIRE_PADDLE:-1}"
export INDORY_OCR_REPAIR_CACHE

if [[ -n "$INDORY_OCR_SETUP" && -f "$INDORY_OCR_SETUP" ]]; then
  # shellcheck disable=SC1090
  set +u
  source "$INDORY_OCR_SETUP"
  set -u
fi

for cuda_lib_dir in \
  "$INDORY_OCR_VENV"/lib/python*/site-packages/nvidia/cuda_runtime/lib \
  "$INDORY_OCR_VENV"/lib/python*/site-packages/nvidia/cublas/lib \
  "$INDORY_OCR_VENV"/lib/python*/site-packages/nvidia/cuda_nvrtc/lib; do
  if [[ -d "$cuda_lib_dir" ]]; then
    export LD_LIBRARY_PATH="$cuda_lib_dir:${LD_LIBRARY_PATH:-}"
  fi
done

"$INDORY_OCR_VENV/bin/python" -m indory_ocr.preflight --repair-cache
"$INDORY_OCR_VENV/bin/python" -c "import llama_cpp; print('llama_cpp import ok', getattr(llama_cpp, '__version__', 'unknown'))"
"$INDORY_OCR_VENV/bin/python" -c "import torch, transformers; print('torch import ok', getattr(torch, '__version__', 'unknown')); print('transformers import ok', getattr(transformers, '__version__', 'unknown'))"

cat <<EOF
Created OCR/LLM venv:
  $INDORY_OCR_VENV

Run the service with:
  CONTROL_SERVER_DETECTION_PYTHON=$INDORY_OCR_VENV/bin/python ./run.sh

PaddleOCR v3 was imported, warmed, and smoke-tested during setup. llama_cpp,
torch, and transformers were imported after package installation. If a partial
PaddleX model cache is found, setup backs it up under ~/.paddlex/official_models
and downloads a clean copy before this message is printed. Model-file setup for
Qwen GGUF and Qwen-VL is documented in docs/setup.md.
EOF
