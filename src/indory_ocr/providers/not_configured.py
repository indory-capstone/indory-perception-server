from __future__ import annotations

from typing import Any

from indory_ocr.providers.base import OcrLlmProvider
from indory_ocr.utils import destination_label, normalize_request_id


class NotConfiguredProvider(OcrLlmProvider):
    @property
    def name(self) -> str:
        return "not_configured"

    async def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "provider": self.name,
            "provider_ready": False,
            "message": "No OCR/LLM engine is configured. Set CONTROL_SERVER_DETECTION_PROVIDER to a concrete provider.",
        }

    async def scan_waybill(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = normalize_request_id(payload.get("request_id"))
        decision = {
            "destination_dong": None,
            "destination_floor": None,
            "destination_room": None,
            "confidence": None,
            "evidence_indices": [],
            "needs_manual_review": True,
            "auto_accept": False,
            "risk_score": 10.0,
            "risk_reasons": ["ocr_llm_provider_not_configured"],
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
            "needs_manual_review": True,
            "auto_accept": False,
            "risk_reasons": decision["risk_reasons"],
            "timing": {"ocr_seconds": 0.0, "llm_seconds": 0.0, "total_seconds": 0.0},
        }
        if payload.get("include_debug") or self.settings.include_debug:
            response["debug"] = {
                "provider": self.name,
                "message": "Provider is not configured; no OCR/LLM inference was run.",
                "ocr": {"items": [], "item_count": 0},
                "llm": {"candidates": [], "raw_response": ""},
            }
        return response

    async def read_ocr(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = normalize_request_id(payload.get("request_id"))
        response = {
            "type": "ocr_result",
            "ok": True,
            "request_id": request_id,
            "camera": payload.get("camera"),
            "source": payload.get("source"),
            "model": None,
            "rotations": payload.get("ocr_rotations") or [0],
            "item_count": 0,
            "items": [],
            "frames": [],
            "timing": {"ocr_seconds": 0.0, "llm_seconds": 0.0, "total_seconds": 0.0},
        }
        if payload.get("include_debug") or self.settings.include_debug:
            response["debug"] = {
                "provider": self.name,
                "message": "Provider is not configured; no OCR inference was run.",
            }
        return response

    async def read_room_signs(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = normalize_request_id(payload.get("request_id"))
        response = {
            "type": "semantic_ocr_result",
            "ok": True,
            "request_id": request_id,
            "camera": payload.get("camera"),
            "source": payload.get("source"),
            "task_mode": "ocr_room_ids",
            "has_text_object": False,
            "objects": [],
            "raw_ocr_output": [],
            "control_summary_ko": "OCR provider is not configured.",
            "need_human_check": True,
            "metadata": {
                "provider": self.name,
                "provider_ready": False,
                "message": "Provider is not configured; no semantic OCR inference was run.",
            },
            "timing": {"ocr_seconds": 0.0, "llm_seconds": 0.0, "total_seconds": 0.0},
        }
        return response

    async def inspect_vlm(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = normalize_request_id(payload.get("request_id"))
        task_mode = str(payload.get("task_mode") or "scene_description")
        response = {
            "type": "vlm_result",
            "ok": True,
            "request_id": request_id,
            "camera": payload.get("camera"),
            "source": payload.get("source"),
            "task_mode": task_mode,
            "observation": None,
            "raw_response": "",
            "metadata": {
                "provider": self.name,
                "provider_ready": False,
                "message": "Provider is not configured; no VLM inference was run.",
            },
            "timing": {"ocr_seconds": 0.0, "llm_seconds": 0.0, "total_seconds": 0.0},
        }
        return response
