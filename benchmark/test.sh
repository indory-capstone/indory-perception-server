#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATASET="${CONTROL_SERVER_DETECTION_BENCH_DATASET:-${INDORY_OCR_BENCH_DATASET:-${1:-current_verified}}}"
MODES="${CONTROL_SERVER_DETECTION_BENCH_MODES:-${INDORY_OCR_BENCH_MODES:-waybill}}"
LIMIT="${CONTROL_SERVER_DETECTION_BENCH_LIMIT:-${INDORY_OCR_BENCH_LIMIT:-1}}"
TIMEOUT="${CONTROL_SERVER_DETECTION_BENCH_TIMEOUT:-${INDORY_OCR_BENCH_TIMEOUT:-300}}"
URL="${CONTROL_SERVER_DETECTION_SERVICE_URL:-${INDORY_OCR_SERVICE_URL:-${INDORY_OCR_LLM_SERVICE_URL:-http://127.0.0.1:8767}}}"
HEALTH_URL="${URL%/}/health"
OUT="${CONTROL_SERVER_DETECTION_BENCH_OUT:-${INDORY_OCR_BENCH_OUT:-benchmark/runs/test_${DATASET}_$(date +%Y%m%d_%H%M%S)}}"
PYTHON_BIN="${CONTROL_SERVER_DETECTION_TEST_PYTHON:-${INDORY_OCR_TEST_PYTHON:-python3}}"
START_SERVICE="${CONTROL_SERVER_DETECTION_TEST_START_SERVICE:-${INDORY_OCR_TEST_START_SERVICE:-1}}"
EXTRA_ARGS=()
if [[ "${CONTROL_SERVER_DETECTION_BENCH_CROP_VARIANTS:-${INDORY_OCR_BENCH_CROP_VARIANTS:-0}}" == "1" ]]; then
  EXTRA_ARGS+=(--ocr-crop-variants)
fi
if [[ "${CONTROL_SERVER_DETECTION_BENCH_FULL_IMAGE_VARIANTS:-${INDORY_OCR_BENCH_FULL_IMAGE_VARIANTS:-0}}" == "1" ]]; then
  EXTRA_ARGS+=(--ocr-full-image-variants)
fi
LOW_CONFIDENCE_THRESHOLD="${CONTROL_SERVER_DETECTION_BENCH_LOW_CONFIDENCE_THRESHOLD:-${INDORY_OCR_BENCH_LOW_CONFIDENCE_THRESHOLD:-}}"
if [[ -n "$LOW_CONFIDENCE_THRESHOLD" ]]; then
  EXTRA_ARGS+=(--low-confidence-threshold "$LOW_CONFIDENCE_THRESHOLD")
fi

mkdir -p "$OUT"

"$PYTHON_BIN" benchmark/check_dataset.py --dataset "$DATASET" --strict

service_pid=""
if ! "$PYTHON_BIN" - <<PY >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("${HEALTH_URL}", timeout=2).read()
PY
then
  if [[ "$START_SERVICE" != "1" ]]; then
    echo "service is not reachable at $URL" >&2
    echo "start it with ./run.sh or set CONTROL_SERVER_DETECTION_TEST_START_SERVICE=1" >&2
    exit 2
  fi
  echo "starting service at $URL"
  CONTROL_SERVER_DETECTION_PROVIDER="${CONTROL_SERVER_DETECTION_PROVIDER:-${INDORY_OCR_PROVIDER:-gz_compat}}" ./run.sh >"$OUT/service.log" 2>&1 &
  service_pid="$!"
  for _ in $(seq 1 60); do
    if "$PYTHON_BIN" - <<PY >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("${HEALTH_URL}", timeout=2).read()
PY
    then
      break
    fi
    sleep 1
  done
fi

cleanup() {
  if [[ -n "$service_pid" ]]; then
    kill "$service_pid" >/dev/null 2>&1 || true
    wait "$service_pid" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

"$PYTHON_BIN" benchmark/run.py \
  --dataset "$DATASET" \
  --modes "$MODES" \
  --limit "$LIMIT" \
  --timeout "$TIMEOUT" \
  --out "$OUT" \
  "${EXTRA_ARGS[@]}" \
  --fail-on-error

echo "benchmark test output: $OUT"
