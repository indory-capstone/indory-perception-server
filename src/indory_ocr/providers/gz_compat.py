from __future__ import annotations

import asyncio
import importlib
import importlib.util
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from indory_ocr.providers.base import OcrLlmProvider
from indory_ocr.semantic_ocr import (
    OCR_REASON,
    SemanticOcrEngine,
    extract_room_observations,
    normalize_floor_hint,
    normalize_floor_prior_mode,
    paddleocr_version,
)
from indory_ocr.semantic_vlm import SemanticVlmEngine
from indory_ocr.utils import (
    decode_image_b64,
    destination_label,
    first_image_b64,
    normalize_request_id,
)


def _int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _float_value(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _bool_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _import_status(name: str) -> tuple[bool, str | None]:
    if importlib.util.find_spec(name) is None:
        return False, "module_not_found"
    try:
        importlib.import_module(name)
        return True, None
    except Exception as exc:
        return False, str(exc)


def _payload_options(payload: dict[str, Any]) -> dict[str, Any]:
    options = payload.get("options")
    return dict(options) if isinstance(options, dict) else {}


def _parse_rotations(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = value.split(",")
    elif isinstance(value, list):
        parts = value
    else:
        return None
    rotations: list[int] = []
    for part in parts:
        try:
            rotation = int(part) % 360
        except Exception:
            continue
        if rotation in {0, 90, 180, 270} and rotation not in rotations:
            rotations.append(rotation)
    return rotations or None


def _default_qwen_gguf() -> Path | None:
    root = Path(os.environ.get("WAYBILL_OCR_ROOT", str(Path.home() / "waybill_ocr_llm"))).expanduser()
    for candidate in (
        root / "models" / "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        Path.home() / "waybill_ocr_llm" / "models" / "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    ):
        if candidate.is_file():
            return candidate
    return None


class GzCompatProvider(OcrLlmProvider):
    """Provider that preserves the Indory OCR/LLM/VLM runtime behavior."""

    def __init__(self, settings) -> None:
        super().__init__(settings)
        os.environ.setdefault("WAYBILL_OCR_REQUIRE_PADDLE", "1")
        self._ocr = SemanticOcrEngine()
        self._vlm = SemanticVlmEngine()

    @property
    def name(self) -> str:
        return "gz_compat"

    async def health(self) -> dict[str, Any]:
        paddleocr_ready, paddleocr_error = _import_status("paddleocr")
        paddle_ready, paddle_error = _import_status("paddle")
        llama_ready, llama_error = _import_status("llama_cpp")
        torch_ready, torch_error = _import_status("torch")
        transformers_ready, transformers_error = _import_status("transformers")
        ocr_ready = paddleocr_ready and paddle_ready
        availability = {
            "ocr.read": ocr_ready,
            "semantic_ocr.room_signs": ocr_ready,
            "waybill.scan": ocr_ready and llama_ready,
            "vlm.inspect": torch_ready and transformers_ready,
        }
        dependency_errors = {
            "paddleocr": paddleocr_error,
            "paddle": paddle_error,
            "llama_cpp": llama_error,
            "torch": torch_error,
            "transformers": transformers_error,
        }
        return {
            "ok": True,
            "provider": self.name,
            "provider_ready": any(availability.values()),
            "capabilities": [
                "ocr.read",
                "semantic_ocr.room_signs",
                "waybill.scan",
                "vlm.inspect",
            ],
            "availability": availability,
            "python": sys.executable,
            "paddleocr_required": True,
            "paddleocr_version": paddleocr_version(),
            "dependency_errors": {key: value for key, value in dependency_errors.items() if value},
            "ocr_backend": self._ocr.backend_name,
            "ocr_backend_version": self._ocr.backend_version,
            "ocr_backend_error": self._ocr.backend_error,
        }

    def _image_bytes(self, payload: dict[str, Any]) -> bytes:
        image = first_image_b64(payload)
        if not image:
            raise ValueError("image_b64 is required")
        return decode_image_b64(image, self.settings.max_image_mb * 1024 * 1024)

    async def read_ocr(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._read_ocr_sync, payload)

    def _read_ocr_sync(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = normalize_request_id(payload.get("request_id"))
        options = _payload_options(payload)
        data = self._image_bytes(payload)
        result = self._ocr.read_image_bytes(data, options)
        items = list(result.get("items") or [])
        return {
            "type": "ocr_result",
            "ok": True,
            "request_id": request_id,
            "camera": payload.get("camera"),
            "source": payload.get("source"),
            "model": result.get("model"),
            "rotations": payload.get("ocr_rotations") or [0],
            "item_count": len(items),
            "items": items,
            "frames": [],
            "timing": {
                "ocr_seconds": float(result.get("ocr_seconds") or 0.0),
                "llm_seconds": 0.0,
                "total_seconds": float(result.get("ocr_seconds") or 0.0),
            },
            "debug": {
                "provider": self.name,
                "backend": result.get("backend"),
                "backend_version": result.get("backend_version"),
                "backend_error": result.get("backend_error"),
                "image_width": result.get("image_width"),
                "image_height": result.get("image_height"),
                "ocr_scales": result.get("ocr_scales"),
            }
            if payload.get("include_debug") or self.settings.include_debug
            else None,
        }

    async def read_room_signs(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._read_room_signs_sync, payload)

    def _read_room_signs_sync(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = normalize_request_id(payload.get("request_id"))
        options = _payload_options(payload)
        data = self._image_bytes(payload)
        result = self._ocr.read_image_bytes(data, options)
        floor_hint = payload.get("floor_hint")
        if floor_hint is None:
            floor_hint = options.get("floor_hint")
        floor_prior_mode = normalize_floor_prior_mode(
            payload.get("floor_prior_mode") or options.get("floor_prior_mode") or "reject"
        )
        min_confidence = _float_value(
            payload.get("min_confidence", options.get("min_confidence", 0.6)),
            0.6,
        )
        objects = extract_room_observations(
            list(result.get("items") or []),
            floor_hint=str(floor_hint or "") or None,
            floor_prior_mode=floor_prior_mode,
            min_confidence=min_confidence,
        )
        inference_s = float(result.get("ocr_seconds") or 0.0)
        metadata = {
            "ocr_backend": result.get("backend"),
            "ocr_backend_version": result.get("backend_version"),
            "ocr_model": result.get("model"),
            "why_this_ocr": OCR_REASON,
            "min_confidence": min_confidence,
            "floor_hint": normalize_floor_hint(str(floor_hint or "") or None),
            "floor_prior_mode": floor_prior_mode,
            "latency_s": round(inference_s, 3),
            "image_width": result.get("image_width"),
            "image_height": result.get("image_height"),
            "backend_error": result.get("backend_error"),
        }
        return {
            "type": "semantic_ocr_result",
            "ok": True,
            "request_id": request_id,
            "camera": payload.get("camera"),
            "source": payload.get("source"),
            "task_mode": "ocr_room_ids",
            "has_text_object": bool(objects),
            "objects": objects,
            "raw_ocr_output": list(result.get("items") or []),
            "control_summary_ko": self._room_summary(objects),
            "need_human_check": False,
            "metadata": metadata,
            "timing": {
                "ocr_seconds": inference_s,
                "llm_seconds": 0.0,
                "total_seconds": inference_s,
            },
        }

    @staticmethod
    def _room_summary(objects: list[dict[str, Any]]) -> str:
        if not objects:
            return "OCR에서 신뢰도 기준을 넘는 표지판 텍스트가 없습니다."
        ids = ", ".join(str(obj.get("room_id")) for obj in objects[:5])
        extra = "" if len(objects) <= 5 else f" 외 {len(objects) - 5}개"
        return f"OCR 표지판 후보: {ids}{extra}"

    async def inspect_vlm(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._inspect_vlm_sync, payload)

    def _inspect_vlm_sync(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = normalize_request_id(payload.get("request_id"))
        data = self._image_bytes(payload)
        result = self._vlm.inspect_image_bytes(data, payload)
        return {
            "type": "vlm_result",
            "ok": result.get("observation") is not None,
            "request_id": request_id,
            "camera": payload.get("camera"),
            "source": payload.get("source"),
            "task_mode": result.get("task_mode"),
            "observation": result.get("observation"),
            "raw_response": result.get("raw_response") or "",
            "metadata": result.get("metadata") or {},
            "timing": result.get("timing") or {},
        }

    async def scan_waybill(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._scan_waybill_sync, payload)

    def _scan_waybill_sync(self, payload: dict[str, Any]) -> dict[str, Any]:
        from waybill_ocr_llm.cli import _resolve_model_path
        from waybill_ocr_llm.pipeline import run_pipeline

        request_id = normalize_request_id(payload.get("request_id"))
        options = _payload_options(payload)
        data = self._image_bytes(payload)
        suffix = "." + str(payload.get("image_format") or "jpg").strip(".")
        suffix = suffix if len(suffix) <= 8 else ".jpg"
        artifact_root = self.settings.artifact_root / "waybill" / request_id
        artifact_root.mkdir(parents=True, exist_ok=True)
        image_path = artifact_root / f"input{suffix}"
        image_path.write_bytes(data)

        mode = str(options.get("judge_mode") or options.get("mode") or os.environ.get("WAYBILL_OCR_JUDGE_MODE", "llama_cpp"))
        model_path_value = options.get("model_path") or os.environ.get("WAYBILL_OCR_DEFAULT_MODEL") or None
        model_path = Path(str(model_path_value)).expanduser() if model_path_value else None
        if mode == "llama_cpp" and model_path is None:
            model_path = _default_qwen_gguf()
        model_path = _resolve_model_path(mode, model_path)
        endpoint = str(options.get("endpoint") or os.environ.get("WAYBILL_OCR_ENDPOINT", ""))
        if mode == "ollama" and not endpoint:
            endpoint = "http://127.0.0.1:11434/api/chat"
        rotations = (
            _parse_rotations(payload.get("ocr_rotations"))
            or _parse_rotations(options.get("ocr_rotations"))
            or _parse_rotations(os.environ.get("WAYBILL_OCR_ROTATIONS", "0"))
            or [0]
        )

        started = time.perf_counter()
        merged = run_pipeline(
            inputs=[image_path],
            out_dir=artifact_root,
            judge_mode=mode,
            model_path=model_path,
            model=str(options.get("model") or os.environ.get("WAYBILL_OCR_MODEL", "")),
            endpoint=endpoint,
            ocr_use_gpu=_bool_value(options.get("ocr_use_gpu"), _bool_value(os.environ.get("WAYBILL_OCR_USE_GPU"), False)),
            ocr_rec_batch_num=_int_value(options.get("ocr_rec_batch_num"), _int_value(os.environ.get("WAYBILL_OCR_REC_BATCH_NUM"), 1)),
            llm_gpu_layers=_int_value(options.get("llm_gpu_layers"), _int_value(os.environ.get("WAYBILL_LLM_GPU_LAYERS"), 0)),
            llm_threads=_int_value(options.get("llm_threads"), _int_value(os.environ.get("WAYBILL_LLM_THREADS"), 4)),
            llm_ctx=_int_value(
                options.get("llm_ctx"),
                _int_value(os.environ.get("WAYBILL_LLM_CTX") or os.environ.get("WAYBILL_OCR_LLM_CTX"), 4096),
            ),
            max_new_tokens=_int_value(options.get("max_new_tokens"), _int_value(os.environ.get("WAYBILL_LLM_MAX_NEW_TOKENS"), 128)),
            limit=1,
            ocr_rotations=rotations,
            ocr_crop_variants=_bool_value(
                options.get("ocr_crop_variants", options.get("crop_variants")),
                _bool_value(os.environ.get("WAYBILL_OCR_CROP_VARIANTS"), False),
            ),
            ocr_full_image_variants=_bool_value(
                options.get("ocr_full_image_variants", options.get("full_image_variants")),
                _bool_value(os.environ.get("WAYBILL_OCR_FULL_IMAGE_VARIANTS"), False),
            ),
        )
        return self._waybill_response(payload, request_id, merged, artifact_root, time.perf_counter() - started)

    def _waybill_response(
        self,
        payload: dict[str, Any],
        request_id: str,
        merged: dict[str, Any],
        artifact_root: Path,
        total_seconds: float,
    ) -> dict[str, Any]:
        ocr_payload = merged.get("ocr") if isinstance(merged, dict) else {}
        llm_payload = merged.get("llm") if isinstance(merged, dict) else {}
        if not isinstance(ocr_payload, dict):
            ocr_payload = {}
        if not isinstance(llm_payload, dict):
            llm_payload = {}
        combined_results = ocr_payload.get("combined_results") or ocr_payload.get("results") or []
        ocr_result = combined_results[0] if combined_results and isinstance(combined_results[0], dict) else {}
        llm_results = llm_payload.get("results") or []
        llm_result = llm_results[0] if llm_results and isinstance(llm_results[0], dict) else {}
        decision = llm_result.get("llm_decision") if isinstance(llm_result, dict) else None
        if not isinstance(decision, dict):
            decision = {
                "destination_dong": None,
                "destination_floor": None,
                "destination_room": None,
                "confidence": None,
                "evidence_indices": [],
                "needs_manual_review": True,
                "auto_accept": False,
                "risk_score": 10.0,
                "risk_reasons": ["no_waybill_destination_decision"],
            }
            if llm_payload.get("error"):
                decision["risk_reasons"].append(str(llm_payload.get("error")))

        needs_manual_review = bool(decision.get("needs_manual_review", True))
        auto_accept = bool(decision.get("auto_accept", False))
        risk_reasons = list(decision.get("risk_reasons") or [])
        ocr_items = list(ocr_result.get("ocr_items") or [])
        timing = {
            "ocr_seconds": float(ocr_result.get("ocr_seconds") or 0.0),
            "llm_seconds": float(llm_result.get("llm_seconds") or 0.0) if isinstance(llm_result, dict) else 0.0,
            "total_seconds": total_seconds,
        }
        response = {
            "type": "result",
            "ok": True,
            "request_id": request_id,
            "task_id": payload.get("task_id"),
            "camera": payload.get("camera"),
            "source": payload.get("source"),
            "destination": destination_label(decision),
            "decision": decision,
            "needs_manual_review": needs_manual_review,
            "auto_accept": auto_accept,
            "risk_reasons": risk_reasons,
            "timing": timing,
        }
        if payload.get("include_debug") or self.settings.include_debug:
            response["debug"] = {
                "provider": self.name,
                "artifact_root": str(artifact_root),
                "ocr": {
                    "image": str(ocr_result.get("image") or ""),
                    "model": str(ocr_payload.get("model") or ""),
                    "item_count": len(ocr_items),
                    "items": ocr_items,
                    "combined_results": combined_results,
                },
                "llm": {
                    "mode": llm_payload.get("mode"),
                    "model": llm_payload.get("model"),
                    "model_path": llm_payload.get("model_path"),
                    "endpoint": llm_payload.get("endpoint"),
                    "candidates": list(llm_result.get("destination_candidates") or []) if isinstance(llm_result, dict) else [],
                    "raw_response": str(llm_result.get("llm_raw_response") or "") if isinstance(llm_result, dict) else "",
                    "raw_responses": list(llm_result.get("llm_raw_responses") or []) if isinstance(llm_result, dict) else [],
                    "prompt": str(llm_result.get("prompt") or "") if isinstance(llm_result, dict) else "",
                    "error": llm_payload.get("error"),
                },
            }
        if not self.settings.keep_artifacts:
            try:
                shutil.rmtree(artifact_root, ignore_errors=True)
            except Exception:
                pass
        return response
