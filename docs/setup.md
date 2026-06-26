# Setup

Create a local virtual environment and install the runtime dependencies:

```bash
./setup.sh
```

Run the service:

```bash
./run.sh
```

Run preflight checks:

```bash
./preflight.sh
```

## Configuration

Common environment variables:

```text
CONTROL_SERVER_DETECTION_HOST=127.0.0.1
CONTROL_SERVER_DETECTION_PORT=8767
CONTROL_SERVER_DETECTION_PROVIDER=gz_compat
CONTROL_SERVER_DETECTION_PYTHON=.venv/bin/python
WAYBILL_OCR_ROOT=
WAYBILL_OCR_MODEL=
WAYBILL_OCR_ENDPOINT=
```

Keep model weights, private images, benchmark datasets, and credentials outside
git. For a lightweight contract-only run, use:

```bash
CONTROL_SERVER_DETECTION_PROVIDER=not_configured ./run.sh
```
