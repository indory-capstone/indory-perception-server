from __future__ import annotations

import base64
import binascii
import re
import time
import uuid
from typing import Any


def now_unix() -> float:
    return time.time()


def normalize_request_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return uuid.uuid4().hex
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:96]


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    aliases = {
        "requestId": "request_id",
        "taskId": "task_id",
        "imageData": "image_b64",
        "imageBase64": "image_base64",
        "imageFormat": "image_format",
        "contentType": "content_type",
        "includeDebug": "include_debug",
        "ocrRotations": "ocr_rotations",
        "floorHint": "floor_hint",
        "floorPriorMode": "floor_prior_mode",
        "minConfidence": "min_confidence",
        "taskMode": "task_mode",
        "modelName": "model_name",
        "torchDtype": "torch_dtype",
        "maxNewTokens": "max_new_tokens",
    }
    for old, new in aliases.items():
        if old in normalized and new not in normalized:
            normalized[new] = normalized[old]
    normalized["request_id"] = normalize_request_id(normalized.get("request_id"))
    return normalized


def decode_image_b64(value: str, max_bytes: int) -> bytes:
    text = value.strip()
    if "," in text and text[:64].lower().startswith("data:"):
        text = text.split(",", 1)[1]
    try:
        data = base64.b64decode(text, validate=True)
    except binascii.Error as exc:
        raise ValueError(f"invalid base64 image: {exc}") from exc
    if not data:
        raise ValueError("empty image payload")
    if len(data) > max_bytes:
        raise ValueError(f"image payload too large: {len(data)} bytes > {max_bytes} bytes")
    return data


def first_image_b64(payload: dict[str, Any]) -> str | None:
    value = payload.get("image_b64") or payload.get("image_base64") or payload.get("image")
    if isinstance(value, str) and value.strip():
        return value
    frames = payload.get("images") or payload.get("frames")
    if isinstance(frames, list) and frames:
        first = frames[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            return first_image_b64(first)
    return None


def destination_label(decision: dict[str, Any]) -> str | None:
    parts = [
        str(decision.get("destination_dong") or "").strip(),
        str(decision.get("destination_floor") or "").strip(),
        str(decision.get("destination_room") or "").strip(),
    ]
    label = " ".join(part for part in parts if part)
    return label or None
