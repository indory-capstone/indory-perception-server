from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PADDLEOCR_V3_MODEL_DIRS = (
    "PP-OCRv5_mobile_det",
    "korean_PP-OCRv5_mobile_rec",
)


def _package_version(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _official_model_root() -> Path:
    if os.environ.get("PADDLEX_HOME"):
        return Path(os.environ["PADDLEX_HOME"]).expanduser() / "official_models"
    return Path.home() / ".paddlex" / "official_models"


def _move_known_model_dirs() -> list[dict[str, str]]:
    root = _official_model_root()
    moved: list[dict[str, str]] = []
    if not root.exists():
        return moved
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for model_name in PADDLEOCR_V3_MODEL_DIRS:
        source = root / model_name
        if not source.exists():
            continue
        destination = root / f"{model_name}.bad_{stamp}"
        counter = 1
        while destination.exists():
            destination = root / f"{model_name}.bad_{stamp}_{counter}"
            counter += 1
        shutil.move(str(source), str(destination))
        moved.append({"from": str(source), "to": str(destination)})
    return moved


def _smoke_image_path() -> Path:
    image = np.full((180, 420, 3), 255, dtype=np.uint8)
    cv2.putText(image, "528", (60, 125), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 0), 8, cv2.LINE_AA)
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    path = Path(tmp.name)
    if not cv2.imwrite(str(path), image):
        path.unlink(missing_ok=True)
        raise RuntimeError("failed to write PaddleOCR preflight image")
    return path


def _run_once(*, use_gpu: bool, rec_batch_num: int) -> dict[str, Any]:
    import paddle
    import paddleocr
    from waybill_ocr_llm.ocr import build_paddleocr, flatten_ocr, run_paddleocr, using_paddleocr_v3

    if not using_paddleocr_v3():
        raise RuntimeError(f"PaddleOCR v3 is required, found paddleocr={getattr(paddleocr, '__version__', 'unknown')}")

    started = time.perf_counter()
    ocr = build_paddleocr(use_gpu=use_gpu, rec_batch_num=rec_batch_num)
    build_seconds = time.perf_counter() - started

    image_path = _smoke_image_path()
    try:
        predict_started = time.perf_counter()
        raw = run_paddleocr(ocr, image_path)
        items = flatten_ocr(raw)
        predict_seconds = time.perf_counter() - predict_started
    finally:
        image_path.unlink(missing_ok=True)

    texts = [item.text for item in items]
    if not texts:
        raise RuntimeError("PaddleOCR preflight produced no text on the synthetic smoke image")

    return {
        "ok": True,
        "python": sys.executable,
        "paddle_version": getattr(paddle, "__version__", _package_version("paddlepaddle")),
        "paddleocr_version": getattr(paddleocr, "__version__", _package_version("paddleocr")),
        "paddleocr_major": 3,
        "device": "gpu:0" if use_gpu else "cpu",
        "rec_batch_num": rec_batch_num,
        "model_root": str(_official_model_root()),
        "model_dirs": list(PADDLEOCR_V3_MODEL_DIRS),
        "smoke_texts": texts,
        "build_seconds": round(build_seconds, 3),
        "predict_seconds": round(predict_seconds, 3),
    }


def run_preflight(*, use_gpu: bool = False, rec_batch_num: int = 1, repair_cache: bool = False) -> dict[str, Any]:
    os.environ.setdefault("WAYBILL_OCR_REQUIRE_PADDLE", "1")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    try:
        return _run_once(use_gpu=use_gpu, rec_batch_num=rec_batch_num)
    except Exception as first_exc:
        if not repair_cache:
            raise
        moved = _move_known_model_dirs()
        if not moved:
            raise
        try:
            result = _run_once(use_gpu=use_gpu, rec_batch_num=rec_batch_num)
        except Exception as second_exc:
            raise RuntimeError(
                f"PaddleOCR preflight failed after moving cached models: {second_exc}. "
                f"Original failure: {first_exc}"
            ) from second_exc
        result["repaired_cache_dirs"] = moved
        result["first_failure"] = str(first_exc)
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify and warm the gz-nav PaddleOCR v3 runtime.")
    parser.add_argument("--use-gpu", action="store_true", help="Build PaddleOCR on gpu:0.")
    parser.add_argument("--rec-batch-num", type=int, default=1)
    repair_group = parser.add_mutually_exclusive_group()
    repair_group.add_argument("--repair-cache", action="store_true", default=_truthy(os.environ.get("INDORY_OCR_REPAIR_CACHE"), True))
    repair_group.add_argument("--no-repair-cache", action="store_false", dest="repair_cache")
    args = parser.parse_args(argv)

    try:
        result = run_preflight(
            use_gpu=bool(args.use_gpu),
            rec_batch_num=max(1, int(args.rec_batch_num)),
            repair_cache=bool(args.repair_cache),
        )
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "python": sys.executable}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
