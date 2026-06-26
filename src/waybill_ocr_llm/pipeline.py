from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .llm import judge_ocr_results
from .ocr import run_ocr_on_images


def _extract_ocr_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = payload.get("results")
    if not isinstance(results, list):
        return []
    items: list[dict[str, Any]] = []
    for result in results:
        if isinstance(result, dict):
            for item in result.get("ocr_items") or []:
                if isinstance(item, dict):
                    items.append(item)
    return items


def run_pipeline(
    inputs: list[Path],
    out_dir: Path,
    judge_mode: str,
    model_path: Path | None = None,
    model: str = "",
    endpoint: str = "",
    ocr_use_gpu: bool = False,
    ocr_rec_batch_num: int = 1,
    llm_gpu_layers: int = 0,
    llm_threads: int = 4,
    llm_ctx: int = 1024,
    max_new_tokens: int = 64,
    limit: int = 0,
    ocr_rotations: list[int] | None = None,
    ocr_crop_variants: bool | None = None,
    ocr_full_image_variants: bool | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ocr_payload = run_ocr_on_images(
        inputs=inputs,
        out_dir=out_dir,
        use_gpu=ocr_use_gpu,
        rec_batch_num=ocr_rec_batch_num,
        limit=limit,
        rotations=ocr_rotations,
        crop_variants=ocr_crop_variants,
        full_image_variants=ocr_full_image_variants,
    )

    ocr_items = _extract_ocr_items(ocr_payload)
    if not ocr_items:
        llm_payload = {
            "mode": judge_mode,
            "model": model or None,
            "model_path": str(model_path) if model_path else None,
            "endpoint": endpoint or None,
            "source_ocr_model": ocr_payload.get("model"),
            "image_count": len(ocr_payload.get("results") or []),
            "total_seconds": 0.0,
            "results": [],
            "error": "No OCR text detected; skipped LLM judgement.",
        }
        merged = {"ocr": ocr_payload, "llm": llm_payload}
        (out_dir / "waybill_pipeline_results.json").write_text(
            json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return merged

    try:
        llm_payload = judge_ocr_results(
            ocr_payload=ocr_payload,
            mode=judge_mode,
            model_path=model_path,
            model=model,
            endpoint=endpoint,
            out_dir=out_dir,
            max_new_tokens=max_new_tokens,
            n_ctx=llm_ctx,
            n_gpu_layers=llm_gpu_layers,
            n_threads=llm_threads,
        )
    except ValueError as exc:
        message = str(exc).strip()
        if message == "--model-path is required for llama_cpp mode" or "model path" in message.lower():
            llm_payload = {
                "mode": judge_mode,
                "model": model or None,
                "model_path": str(model_path) if model_path else None,
                "endpoint": endpoint or None,
                "source_ocr_model": ocr_payload.get("model"),
                "image_count": len(ocr_payload.get("results") or []),
                "total_seconds": 0.0,
                "results": [],
                "error": "LLM skipped because llama_cpp model_path is missing.",
                "skip_reason": "Set WAYBILL_OCR_DEFAULT_MODEL to a .gguf path or pass model in request body.",
            }
        else:
            raise
    merged = {"ocr": ocr_payload, "llm": llm_payload}
    (out_dir / "waybill_pipeline_results.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return merged
