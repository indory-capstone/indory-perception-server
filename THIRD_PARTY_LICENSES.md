# Third-Party Licenses

This repository is released under Apache License 2.0. Runtime dependencies keep
their own upstream licenses.

The main dependency families are:

| Area | Examples | Notes |
|---|---|---|
| Web service | FastAPI, Starlette, Uvicorn, Pydantic | Used for the HTTP API boundary |
| OCR | PaddleOCR, PaddlePaddle, PaddleX, OpenCV | Used for OCR detection and recognition |
| LLM judge | llama-cpp-python, Hugging Face Hub | Used for local or downloaded Qwen GGUF inference |
| VLM | PyTorch, Transformers, Accelerate | Used for semantic image/text-object inspection |
| Benchmark | standard Python libraries, Hugging Face Hub | Used to download public dataset snapshots and evaluate API responses |

Model weights, generated datasets, camera captures, rosbag files, benchmark
images, and checkpoints are not licensed by this repository. Publish those
artifacts separately with their own license or access policy.

Before a formal release, verify exact dependency versions and license texts from
the lock file or package metadata used for that release.
