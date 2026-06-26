# Setup

For no-model API contract evaluation, create a lightweight environment:

```bash
sudo apt-get install -y python3-venv python3-pip
python3 -m venv .venv-contract
.venv-contract/bin/python -m pip install -U pip setuptools wheel
.venv-contract/bin/python -m pip install -e .
```

If Python venv support is still missing on Ubuntu/Debian, install the same
packages and rerun the venv commands:

```bash
sudo apt-get install -y python3-venv python3-pip
```

Run the contract-only server:

```bash
CONTROL_SERVER_DETECTION_PYTHON=.venv-contract/bin/python \
CONTROL_SERVER_DETECTION_PROVIDER=not_configured ./run.sh
```

Check the health and route contract:

```bash
curl http://127.0.0.1:8767/health
curl http://127.0.0.1:8767/v1/contracts
```

For the full OCR/VLM runtime, create a local virtual environment and install the
runtime dependencies:

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
CONTROL_SERVER_DETECTION_PROVIDER=not_configured \
WAYBILL_OCR_REQUIRE_PADDLE=0 WAYBILL_OCR_USE_GPU=0 \
./run.sh
```
