from __future__ import annotations

import json
import math
import re
import threading
import time
from io import BytesIO
from typing import Any

from PIL import Image as PILImage


TEXT_OBJECT_SYSTEM_PROMPT = """You are a robot visual inspection module.

Input: one camera frame.

Task:
Detect visible signs, warning labels, door plates, package labels, delivery labels, or other text-bearing objects.
For each object, return its approximate pixel location and only the text that is clearly readable.

Rules:
- Return only valid JSON.
- Do not explain outside JSON.
- Do not infer missing text.
- Do not complete partially visible words.
- Do not guess based on context.
- If text is blurry, too small, occluded, angled, overexposed, or unreadable, set text=null.
- If bbox is uncertain, set bbox_xyxy=null.
- Prefer false negatives over hallucinated text.
- Use original image pixel coordinates.
- The Korean control summary must mention only detected evidence.
- If confidence is low or text is null, say that human confirmation is needed.

Output schema:
{
  "has_text_object": boolean,
  "objects": [
    {
      "type": "sign|package_label|warning_label|doorplate|delivery_label|other",
      "bbox_xyxy": [x1, y1, x2, y2] | null,
      "text": "clearly readable text only" | null,
      "confidence": "low|medium|high",
      "failure_reason": "none|blur|small_text|occlusion|glare|low_resolution|motion|angle|unknown"
    }
  ],
  "control_summary_ko": "관제에 보낼 한국어 한 문장",
  "need_human_check": boolean
}"""

TEXT_OBJECT_USER_PROMPT = "Return only the JSON object for this frame."

SCENE_DESCRIPTION_SYSTEM_PROMPT = """You are a robot scene description module.

Input: one robot camera frame.

Task:
Describe only what is visibly present in this frame for remote monitoring.
For the main visible objects you mention, also return where they are in image pixel coordinates.

Rules:
- Return only valid JSON.
- Do not explain outside JSON.
- Do not guess unseen areas.
- Do not infer hidden objects.
- If visibility is poor, state that clearly.
- Prefer concise factual descriptions.
- Use pixel coordinates from the input image.
- If an object's image location is unclear, set bbox_xyxy=null.
- The Korean control summary must mention only visible evidence.

Output schema:
{
  "scene_description_ko": "현재 프레임에 대한 짧고 사실적인 한국어 설명",
  "objects": [
    {
      "label": "visible object name",
      "bbox_xyxy": [x1, y1, x2, y2] | null,
      "visible_evidence_ko": "이 물체가 왜 그렇게 보이는지에 대한 짧은 근거",
      "confidence": "low|medium|high"
    }
  ],
  "control_summary_ko": "관제에 보낼 한국어 한 문장",
  "need_human_check": boolean
}"""

SCENE_DESCRIPTION_USER_PROMPT = "Return only the JSON object describing this frame."

OBJECT_TYPES = {"sign", "package_label", "warning_label", "doorplate", "delivery_label", "other"}
CONFIDENCES = {"low", "medium", "high"}
FAILURE_REASONS = {
    "none",
    "blur",
    "small_text",
    "occlusion",
    "glare",
    "low_resolution",
    "motion",
    "angle",
    "unknown",
}
TEXT_OBJECT_TASK_ALIASES = {"text_object", "text_objects", "object_detection", "text_object_detection"}


def normalize_task_mode(value: str | None) -> str:
    mode = str(value or "scene_description").strip().lower()
    if mode == "scene_description":
        return "scene_description"
    if mode in TEXT_OBJECT_TASK_ALIASES:
        return "text_object"
    return "scene_description"


def system_prompt(task_mode: str) -> str:
    task_mode = normalize_task_mode(task_mode)
    return SCENE_DESCRIPTION_SYSTEM_PROMPT if task_mode == "scene_description" else TEXT_OBJECT_SYSTEM_PROMPT


def user_prompt(task_mode: str) -> str:
    task_mode = normalize_task_mode(task_mode)
    return SCENE_DESCRIPTION_USER_PROMPT if task_mode == "scene_description" else TEXT_OBJECT_USER_PROMPT


def parse_json_object(raw: str) -> dict[str, Any] | None:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def validate_bbox(bbox: Any, width: int, height: int) -> list[int] | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        values = [float(value) for value in bbox]
    except Exception:
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    x1, y1, x2, y2 = values
    x1 = int(round(min(max(x1, 0.0), float(max(width - 1, 0)))))
    x2 = int(round(min(max(x2, 0.0), float(max(width - 1, 0)))))
    y1 = int(round(min(max(y1, 0.0), float(max(height - 1, 0)))))
    y2 = int(round(min(max(y2, 0.0), float(max(height - 1, 0)))))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def pixel_location_ko(bbox: list[int] | None, width: int, height: int) -> str | None:
    if bbox is None or width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    hx = cx / float(width)
    hy = cy / float(height)
    x_desc = "좌측" if hx < 1.0 / 3.0 else "중앙" if hx < 2.0 / 3.0 else "우측"
    y_desc = "상단" if hy < 1.0 / 3.0 else "중단" if hy < 2.0 / 3.0 else "하단"
    return f"{x_desc} {y_desc} ({x1},{y1})-({x2},{y2})"


def validate_text_object_observation(obj: dict[str, Any], width: int, height: int) -> dict[str, Any] | None:
    objects_raw = obj.get("objects", [])
    if objects_raw is None:
        objects_raw = []
    if not isinstance(objects_raw, list):
        return None

    valid_objects: list[dict[str, Any]] = []
    force_human = False
    for item in objects_raw:
        if not isinstance(item, dict):
            continue
        object_type = str(item.get("type", "other")).strip()
        if object_type not in OBJECT_TYPES:
            object_type = "other"
        bbox = validate_bbox(item.get("bbox_xyxy"), width, height)
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            text = None
        else:
            text = text.strip()
        confidence = str(item.get("confidence", "low")).strip().lower()
        if confidence not in CONFIDENCES:
            confidence = "low"
        failure_reason = str(item.get("failure_reason", "unknown")).strip().lower()
        if failure_reason not in FAILURE_REASONS:
            failure_reason = "unknown"
        if confidence == "low" or text is None:
            force_human = True
        valid_objects.append(
            {
                "type": object_type,
                "bbox_xyxy": bbox,
                "text": text,
                "confidence": confidence,
                "failure_reason": failure_reason,
            }
        )

    has_text_object = bool(obj.get("has_text_object", bool(valid_objects))) and bool(valid_objects)
    summary = obj.get("control_summary_ko")
    if not isinstance(summary, str) or not summary.strip():
        summary = (
            "카메라 프레임에서 확인 가능한 텍스트 객체가 없습니다."
            if not valid_objects
            else "텍스트 객체가 감지되었으며 일부 항목은 사람 확인이 필요합니다."
        )
    need_human = bool(obj.get("need_human_check", False)) or force_human
    return {
        "has_text_object": has_text_object,
        "objects": valid_objects,
        "control_summary_ko": summary.strip(),
        "need_human_check": need_human,
    }


def validate_scene_description(obj: dict[str, Any], width: int, height: int) -> dict[str, Any] | None:
    scene_description = obj.get("scene_description_ko")
    if not isinstance(scene_description, str) or not scene_description.strip():
        return None
    objects_raw = obj.get("objects", [])
    if objects_raw is None:
        objects_raw = []
    if not isinstance(objects_raw, list):
        return None

    valid_objects: list[dict[str, Any]] = []
    for item in objects_raw:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        if not isinstance(label, str) or not label.strip():
            continue
        bbox = validate_bbox(item.get("bbox_xyxy"), width, height)
        confidence = str(item.get("confidence", "low")).strip().lower()
        if confidence not in CONFIDENCES:
            confidence = "low"
        evidence = item.get("visible_evidence_ko")
        if not isinstance(evidence, str) or not evidence.strip():
            evidence = label.strip()
        valid_objects.append(
            {
                "label": label.strip(),
                "bbox_xyxy": bbox,
                "visible_evidence_ko": evidence.strip(),
                "confidence": confidence,
                "pixel_location_ko": pixel_location_ko(bbox, width, height),
            }
        )

    summary = obj.get("control_summary_ko")
    if not isinstance(summary, str) or not summary.strip():
        summary = scene_description
    return {
        "scene_description_ko": scene_description.strip(),
        "objects": valid_objects,
        "control_summary_ko": summary.strip(),
        "need_human_check": bool(obj.get("need_human_check", False))
        or any(item.get("confidence") == "low" or item.get("bbox_xyxy") is None for item in valid_objects),
    }


class SemanticVlmEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._device = "cpu"
        self._model_key: tuple[str, str, str] | None = None

    def _ensure_model(self, model_name: str, device: str, torch_dtype_name: str) -> None:
        key = (model_name, device, torch_dtype_name)
        if self._model is not None and self._processor is not None and self._model_key == key:
            return
        with self._lock:
            if self._model is not None and self._processor is not None and self._model_key == key:
                return
            import torch
            from transformers import AutoProcessor

            requested_device = str(device or "auto").lower()
            dtype_param = str(torch_dtype_name or "auto").lower()
            if dtype_param == "auto":
                torch_dtype = "auto"
            elif dtype_param in ("float16", "fp16"):
                torch_dtype = torch.float16
            elif dtype_param in ("bfloat16", "bf16"):
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32

            if requested_device == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = requested_device

            model_cls = None
            for module_name, class_name in (
                ("transformers", "Qwen2_5_VLForConditionalGeneration"),
                ("transformers", "AutoModelForImageTextToText"),
                ("transformers", "AutoModelForVision2Seq"),
            ):
                try:
                    module = __import__(module_name, fromlist=[class_name])
                    model_cls = getattr(module, class_name)
                    break
                except Exception:
                    continue
            if model_cls is None:
                from transformers import AutoModelForCausalLM

                model_cls = AutoModelForCausalLM

            self._processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self._model = model_cls.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            ).eval()
            if self._device != "cpu":
                self._model.to(self._device)
            self._torch = torch
            self._model_key = key

    def inspect_image_bytes(self, data: bytes, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        task_mode = normalize_task_mode(payload.get("task_mode"))
        options = payload.get("options") if isinstance(payload.get("options"), dict) else {}
        model_name = str(
            payload.get("model_name")
            or options.get("model_name")
            or "Qwen/Qwen2.5-VL-3B-Instruct"
        )
        device = str(payload.get("device") or options.get("device") or "auto")
        torch_dtype = str(payload.get("torch_dtype") or options.get("torch_dtype") or "auto")
        max_new_tokens = int(payload.get("max_new_tokens") or options.get("max_new_tokens") or 256)

        image = PILImage.open(BytesIO(data)).convert("RGB")
        width, height = image.size
        self._ensure_model(model_name, device, torch_dtype)
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None

        messages = [
            {"role": "system", "content": system_prompt(task_mode)},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt(task_mode)},
                ],
            },
        ]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = {key: value.to(self._device) if hasattr(value, "to") else value for key, value in inputs.items()}
        input_len = int(inputs["input_ids"].shape[-1])
        with self._torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=max(1, max_new_tokens),
                do_sample=False,
            )
        generated = generated[:, input_len:]
        raw = self._processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        parsed = parse_json_object(raw)
        observation = None
        validation_error = ""
        if parsed is None:
            validation_error = "json_parse_failed"
        elif task_mode == "scene_description":
            observation = validate_scene_description(parsed, width, height)
            if observation is None:
                validation_error = "scene_description_schema_validation_failed"
        else:
            observation = validate_text_object_observation(parsed, width, height)
            if observation is None:
                validation_error = "text_object_schema_validation_failed"

        return {
            "task_mode": task_mode,
            "observation": observation,
            "raw_response": raw,
            "metadata": {
                "model": model_name,
                "device": self._device,
                "torch_dtype": torch_dtype,
                "image_width": width,
                "image_height": height,
                "prompt_family": task_mode,
                "validation_error": validation_error,
            },
            "timing": {
                "ocr_seconds": 0.0,
                "llm_seconds": time.perf_counter() - started,
                "total_seconds": time.perf_counter() - started,
            },
        }
