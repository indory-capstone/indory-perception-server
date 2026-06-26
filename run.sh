#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

: "${CONTROL_SERVER_DETECTION_HOST:=${INDORY_OCR_HOST:-${INDORY_OCR_LLM_HOST:-127.0.0.1}}}"
: "${CONTROL_SERVER_DETECTION_PORT:=${INDORY_OCR_PORT:-${INDORY_OCR_LLM_PORT:-8767}}}"
: "${CONTROL_SERVER_DETECTION_PROVIDER:=${INDORY_OCR_PROVIDER:-gz_compat}}"
: "${INDORY_OCR_SETUP:=}"
: "${WAYBILL_OCR_ROOT:=$ROOT/external/waybill_ocr_llm}"

INDORY_OCR_PYTHON="${CONTROL_SERVER_DETECTION_PYTHON:-${INDORY_OCR_PYTHON:-${INDORY_OCR_LLM_PYTHON:-}}}"
if [[ -z "$INDORY_OCR_PYTHON" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    INDORY_OCR_PYTHON="$ROOT/.venv/bin/python"
  else
    INDORY_OCR_PYTHON="python3"
  fi
fi

if [[ -n "$INDORY_OCR_SETUP" && -f "$INDORY_OCR_SETUP" ]]; then
  # shellcheck disable=SC1090
  set +u
  source "$INDORY_OCR_SETUP"
  set -u
fi

INDORY_OCR_VENV_ROOT="$(cd "$(dirname "$INDORY_OCR_PYTHON")/.." 2>/dev/null && pwd || true)"
if [[ -n "$INDORY_OCR_VENV_ROOT" ]]; then
  for cuda_lib_dir in \
    "$INDORY_OCR_VENV_ROOT"/lib/python*/site-packages/nvidia/cuda_runtime/lib \
    "$INDORY_OCR_VENV_ROOT"/lib/python*/site-packages/nvidia/cublas/lib \
    "$INDORY_OCR_VENV_ROOT"/lib/python*/site-packages/nvidia/cuda_nvrtc/lib; do
    if [[ -d "$cuda_lib_dir" ]]; then
      export LD_LIBRARY_PATH="$cuda_lib_dir:${LD_LIBRARY_PATH:-}"
    fi
  done
fi

PYTHONPATH_PARTS=()
PYTHONPATH_PARTS+=("$ROOT/src")
if [[ -d "$WAYBILL_OCR_ROOT/src/waybill_ocr_llm" ]]; then
  # Keep the old waybill repo importable for compatibility, but do not shadow
  # this service's extracted package.
  PYTHONPATH_PARTS+=("$WAYBILL_OCR_ROOT/src")
fi
if [[ -n "${PYTHONPATH:-}" ]]; then
  PYTHONPATH_PARTS+=("$PYTHONPATH")
fi
export PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_PARTS[*]}")"
export WAYBILL_OCR_ROOT
export WAYBILL_OCR_REQUIRE_PADDLE="${WAYBILL_OCR_REQUIRE_PADDLE:-1}"
export WAYBILL_OCR_USE_GPU="${WAYBILL_OCR_USE_GPU:-1}"

if [[ "${INDORY_OCR_PRINT_RUNTIME:-${INDORY_OCR_LLM_PRINT_RUNTIME:-0}}" == "1" ]]; then
  echo "CONTROL_SERVER_DETECTION_HOST=$CONTROL_SERVER_DETECTION_HOST"
  echo "CONTROL_SERVER_DETECTION_PORT=$CONTROL_SERVER_DETECTION_PORT"
  echo "CONTROL_SERVER_DETECTION_PROVIDER=$CONTROL_SERVER_DETECTION_PROVIDER"
  echo "INDORY_OCR_PYTHON=$INDORY_OCR_PYTHON"
  echo "INDORY_OCR_SETUP=$INDORY_OCR_SETUP"
  echo "WAYBILL_OCR_ROOT=$WAYBILL_OCR_ROOT"
  echo "WAYBILL_OCR_REQUIRE_PADDLE=$WAYBILL_OCR_REQUIRE_PADDLE"
  echo "WAYBILL_OCR_USE_GPU=$WAYBILL_OCR_USE_GPU"
  echo "PYTHONPATH=$PYTHONPATH"
fi

exec "$INDORY_OCR_PYTHON" -m indory_ocr.app \
  --host "$CONTROL_SERVER_DETECTION_HOST" \
  --port "$CONTROL_SERVER_DETECTION_PORT" \
  --provider "$CONTROL_SERVER_DETECTION_PROVIDER" \
  "$@"
