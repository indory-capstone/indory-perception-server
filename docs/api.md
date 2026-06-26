# API

`indory-perception-server` exposes perception functions over HTTP so the
control server does not need to load OCR, LLM, or VLM dependencies in-process.

Default URL:

```text
http://127.0.0.1:8767
```

## Routes

```text
GET  /health
GET  /v1/contracts
POST /v1/waybill/scan
POST /v1/ocr/read
POST /v1/semantic-ocr/room-signs
POST /v1/vlm/inspect
```

`/v1/contracts` returns the request and response shapes used by the control
server. Use it as the source of truth when updating adapters.

## Waybill Scan

`POST /v1/waybill/scan` accepts an image and returns OCR text, candidate
destinations, and the selected destination when the judge can decide one.

Typical use:

```bash
curl -X POST http://127.0.0.1:8767/v1/waybill/scan \
  -F image=@sample-waybill.jpg
```

## Room Sign OCR

`POST /v1/semantic-ocr/room-signs` reads room signs from camera frames during
map initialization and delivery navigation. The request may include a floor
hint when the control server already knows the current floor.

## VLM Inspect

`POST /v1/vlm/inspect` is intended for slower visual-language checks, not for
the real-time robot control loop.
