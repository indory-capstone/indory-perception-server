# indory-perception-server

`indory-perception-server` is the perception service for the Indory indoor
delivery robot. It exposes OCR, waybill destination parsing, room-sign
recognition, and visual-language inspection through a FastAPI interface.

The service is separate from the web control server so that OCR, LLM, and VLM
dependencies can be installed and tested independently.

## Role In The System

| Repository | Role |
| --- | --- |
| `indory-control-server` | Calls this service during task execution and map initialization |
| `indory-robot-runtime` | Provides camera streams from the robot |
| `indory-vla-runtime` | Uses camera observations for manipulation policies |
| `indory-perception-server` | Runs OCR, waybill parsing, room-sign detection, and VLM inspection |

## API

```text
POST /v1/waybill/scan            waybill OCR and destination decision
POST /v1/ocr/read                OCR text and bounding boxes
POST /v1/semantic-ocr/room-signs room/sign OCR with optional floor prior
POST /v1/vlm/inspect             visual-language scene inspection
GET  /health                     service health
GET  /v1/contracts               machine-readable route contracts
```

Default local URL:

```text
http://127.0.0.1:8767
```

## Setup

```bash
./setup.sh
./run.sh
```

The default provider is `gz_compat`, which preserves the OCR/LLM behavior used
by the Indory integration stack. For contract tests without model inference:

```bash
CONTROL_SERVER_DETECTION_PROVIDER=not_configured ./run.sh
```

## Configuration

Important environment variables:

```text
CONTROL_SERVER_DETECTION_HOST
CONTROL_SERVER_DETECTION_PORT
CONTROL_SERVER_DETECTION_PROVIDER
CONTROL_SERVER_DETECTION_ARTIFACT_ROOT
WAYBILL_OCR_JUDGE_MODE
WAYBILL_OCR_MODEL
WAYBILL_OCR_ENDPOINT
WAYBILL_OCR_REQUIRE_PADDLE
```

## Benchmark

Benchmark scripts live in `benchmark/`. Generated datasets, raw images, review
exports, and benchmark run outputs are not tracked in git.

```bash
python3 benchmark/run.py --dataset hf:<dataset-repo>
```

## Artifacts

This repository should contain service code, contracts, tests, and small
examples only. Do not commit model weights, private images, OCR benchmark
captures, local logs, or credentials.

## License

Add a project license before publishing this repository.
