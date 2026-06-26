#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

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
if [[ -d "$WAYBILL_OCR_ROOT/src/waybill_ocr_llm" ]]; then
  PYTHONPATH_PARTS+=("$WAYBILL_OCR_ROOT/src")
fi
PYTHONPATH_PARTS+=("$ROOT/src")
if [[ -n "${PYTHONPATH:-}" ]]; then
  PYTHONPATH_PARTS+=("$PYTHONPATH")
fi
export PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_PARTS[*]}")"
export WAYBILL_OCR_ROOT
export WAYBILL_OCR_REQUIRE_PADDLE="${WAYBILL_OCR_REQUIRE_PADDLE:-1}"
export INDORY_OCR_REPAIR_CACHE="${INDORY_OCR_REPAIR_CACHE:-1}"

exec "$INDORY_OCR_PYTHON" -m indory_ocr.preflight "$@"
