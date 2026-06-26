from __future__ import annotations

import copy
import re
import threading
import time
import tempfile
from importlib.metadata import PackageNotFoundError, version as package_version
from typing import Any

import cv2
import numpy as np


ROOM_ID_RE = re.compile(r"(?<![0-9A-Z])(?:[A-Z][ -]?)?\d{3,4}(?:-\d+)?(?![0-9A-Z])", re.IGNORECASE)
FLOOR_HINT_RE = re.compile(r"^(B|BASEMENT-?)?\s*(\d+)\s*(F|TH|ST|ND|RD)?$")

OCR_REASON = (
    "PaddleOCR is the primary OCR because it returns text, confidence, and "
    "quadrilateral boxes in one pass, supports angle classification for tilted "
    "hallway text, and works well with multi-scale RGB frames."
)

def normalize_floor_prior_mode(value: str | None) -> str:
    mode = str(value or "reject").strip().lower()
    if mode == "complete":
        return "complete"
    return "reject"


def normalize_floor_hint(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper().replace(" ", "")
    if not text:
        return None
    if text.startswith("-"):
        rest = text[1:]
        if rest.isdigit() and int(rest) > 0:
            return f"B{int(rest)}F"
        return text
    if text.startswith("F") and text[1:].isdigit():
        return f"{int(text[1:])}F"
    match = FLOOR_HINT_RE.match(text)
    if not match:
        return text
    basement = bool(match.group(1))
    floor = int(match.group(2))
    if floor <= 0:
        return text
    return f"B{floor}F" if basement else f"{floor}F"


def apply_floor_prior(room_id: str, floor_hint: str | None, floor_prior_mode: str) -> str | None:
    hint = normalize_floor_hint(floor_hint)
    mode = normalize_floor_prior_mode(floor_prior_mode)
    compact = re.sub(r"\s+", "", room_id.upper())
    match = re.match(r"^([A-Z])?(\d{3,4})(?:-(\d+))?$", compact)
    if not match:
        return None
    letter, digits, suffix = match.groups()
    suffix_text = f"-{suffix}" if suffix else ""
    letter = letter or ""
    complete = mode == "complete"
    if not hint:
        return compact
    hint_match = re.match(r"^(B?)(\d+)F$", hint)
    if not hint_match:
        return compact
    basement = hint_match.group(1) == "B"
    floor_str = hint_match.group(2)
    expected_len = len(floor_str) + 2
    if basement:
        if letter == "B" and len(digits) == expected_len and digits.startswith(floor_str):
            return f"B{digits}{suffix_text}"
        if complete and not letter and len(digits) == expected_len and digits.startswith(floor_str):
            return f"B{digits}{suffix_text}"
        return None
    if not letter and len(digits) == expected_len and digits.startswith(floor_str):
        return f"{digits}{suffix_text}"
    if (
        complete
        and len(floor_str) >= 2
        and not letter
        and len(digits) == expected_len - len(floor_str) + 1
        and digits.startswith(floor_str[-1])
    ):
        return f"{floor_str[:-1]}{digits}{suffix_text}"
    return None


def normalize_room_id(text: str | None, floor_hint: str | None, floor_prior_mode: str) -> str | None:
    if text is None:
        return None
    cleaned = re.sub(r"[^0-9A-Za-z가-힣 -]+", " ", str(text).upper())
    match = ROOM_ID_RE.search(cleaned)
    if not match:
        return None
    normalized = re.sub(r"\s+", "", match.group(0).upper())
    return apply_floor_prior(normalized, floor_hint, floor_prior_mode)


def parse_scales(value: Any) -> list[float]:
    if isinstance(value, list):
        parts = value
    else:
        parts = str(value or "1.0,2.0").split(",")
    scales: list[float] = []
    for part in parts:
        try:
            scale = max(0.25, min(4.0, float(str(part).strip())))
        except ValueError:
            continue
        if scale not in scales:
            scales.append(scale)
    return scales or [1.0]


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def confidence_label(confidence: float) -> str:
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.7:
        return "medium"
    return "low"


def image_bytes_to_rgb(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("failed to decode image bytes")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_rgb(rgb: np.ndarray, max_side: int) -> tuple[np.ndarray, float, float]:
    height, width = rgb.shape[:2]
    max_side = max(1, int(max_side or 1280))
    scale = min(1.0, float(max_side) / float(max(height, width)))
    if scale >= 1.0:
        return rgb, 1.0, 1.0
    out_w = max(1, int(round(width * scale)))
    out_h = max(1, int(round(height * scale)))
    resized = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return resized, float(width) / float(out_w), float(height) / float(out_h)


def scaled_rgb(rgb: np.ndarray, scale: float) -> np.ndarray:
    if abs(float(scale) - 1.0) < 1e-6:
        return rgb
    height, width = rgb.shape[:2]
    return cv2.resize(
        rgb,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA,
    )


def clamp_bbox(values: list[float], width: int, height: int) -> list[int] | None:
    if width <= 0 or height <= 0 or len(values) != 4:
        return None
    x1, y1, x2, y2 = values
    x1 = int(round(min(max(x1, 0.0), float(width - 1))))
    x2 = int(round(min(max(x2, 0.0), float(width - 1))))
    y1 = int(round(min(max(y1, 0.0), float(height - 1))))
    y2 = int(round(min(max(y2, 0.0), float(height - 1))))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def bbox_to_box(bbox: list[int] | None) -> list[list[float]]:
    if not bbox:
        return []
    x1, y1, x2, y2 = bbox
    return [[float(x1), float(y1)], [float(x2), float(y1)], [float(x2), float(y2)], [float(x1), float(y2)]]


def bbox_center_distance(a: list[int], b: list[int]) -> float:
    ax = (float(a[0]) + float(a[2])) * 0.5
    ay = (float(a[1]) + float(a[3])) * 0.5
    bx = (float(b[0]) + float(b[2])) * 0.5
    by = (float(b[1]) + float(b[3])) * 0.5
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)


def bbox_diag(bbox: list[int]) -> float:
    return float(((float(bbox[2] - bbox[0])) ** 2 + (float(bbox[3] - bbox[1])) ** 2) ** 0.5)


def bbox_iou(a: list[int], b: list[int]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def dedupe_room_observations(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[list[dict[str, Any]]] = []
    for obs in sorted(observations, key=lambda item: -float(item.get("confidence") or 0.0)):
        bbox = obs.get("bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4:
            groups.append([obs])
            continue
        matched: list[dict[str, Any]] | None = None
        for group in groups:
            anchor_bbox = group[0].get("bbox_xyxy")
            if not isinstance(anchor_bbox, list) or len(anchor_bbox) != 4:
                continue
            allowed = max(8.0, 0.75 * max(bbox_diag(bbox), bbox_diag(anchor_bbox)))
            if bbox_center_distance(bbox, anchor_bbox) <= allowed or bbox_iou(bbox, anchor_bbox) >= 0.35:
                matched = group
                break
        if matched is None:
            groups.append([obs])
        else:
            matched.append(obs)
    kept = [max(group, key=lambda item: float(item.get("confidence") or 0.0)) for group in groups]
    return sorted(kept, key=lambda item: (str(item.get("room_id") or ""), -float(item.get("confidence") or 0.0)))


def extract_room_observations(
    raw_detections: list[dict[str, Any]],
    *,
    floor_hint: str | None,
    floor_prior_mode: str,
    min_confidence: float,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for det in raw_detections:
        try:
            confidence = float(det.get("confidence") or 0.0)
        except Exception:
            confidence = 0.0
        if confidence <= min_confidence:
            continue
        raw_text = str(det.get("text") or "")
        room_id = normalize_room_id(raw_text, floor_hint, floor_prior_mode)
        if not room_id:
            continue
        observations.append(
            {
                "type": "room_id_sign",
                "room_id": room_id,
                "text": room_id,
                "raw_text": raw_text,
                "confidence": round(confidence, 4),
                "confidence_label": confidence_label(confidence),
                "bbox_xyxy": copy.deepcopy(det.get("bbox_xyxy")),
                "source": str(det.get("source") or ""),
            }
        )
    return dedupe_room_observations(observations)


class SemanticOcrEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._backend: Any = None
        self._backend_key: tuple[str, bool] | None = None
        self._backend_name = ""
        self._backend_version: str | None = None
        self._backend_error: str | None = None
        self._backend_is_paddle_v3 = False

    @property
    def backend_name(self) -> str:
        return self._backend_name or "not_loaded"

    @property
    def backend_version(self) -> str | None:
        return self._backend_version

    @property
    def backend_error(self) -> str | None:
        return self._backend_error

    def model_description(self) -> str:
        if self._backend_name == "paddle":
            suffix = f" {self._backend_version}" if self._backend_version else ""
            return f"PaddleOCR{suffix}(lang=en, use_angle_cls=True)"
        return self._backend_name or "not_loaded"

    def _ensure_backend(self, requested: str, use_gpu: bool) -> None:
        requested = str(requested or "paddle").strip().lower()
        if requested in {"", "paddleocr"}:
            requested = "paddle"
        if requested != "paddle":
            raise ValueError("semantic OCR supports only PaddleOCR; fallback OCR backends are disabled")
        key = (requested, bool(use_gpu))
        if self._backend_key == key and self._backend_name:
            return
        with self._lock:
            if self._backend_key == key and self._backend_name:
                return
            self._backend = None
            self._backend_name = ""
            self._backend_version = None
            self._backend_error = None
            self._backend_is_paddle_v3 = False
            try:
                import paddleocr
                from waybill_ocr_llm.ocr import build_paddleocr, using_paddleocr_v3

                self._backend = build_paddleocr(use_gpu=bool(use_gpu), rec_batch_num=1)
                self._backend_name = "paddle"
                version = getattr(paddleocr, "__version__", None)
                self._backend_version = str(version) if version is not None else None
                self._backend_is_paddle_v3 = using_paddleocr_v3()
                self._backend_key = key
            except Exception as exc:
                self._backend_name = "none"
                self._backend_error = str(exc)
                self._backend_key = ("none", False)
                raise RuntimeError(
                    "PaddleOCR backend is unavailable. Run "
                    "preflight.sh or setup.sh "
                    "to install, warm, and verify the PaddleOCR v3 runtime."
                ) from exc

    def _run_paddle_v3(self, rgb: np.ndarray) -> Any:
        assert self._backend is not None
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(tmp.name, bgr):
                raise RuntimeError("failed to write temporary OCR image")
            return self._backend.predict(tmp.name)

    def read_image_bytes(self, data: bytes, options: dict[str, Any]) -> dict[str, Any]:
        start = time.perf_counter()
        rgb = image_bytes_to_rgb(data)
        height, width = rgb.shape[:2]
        max_side = int(options.get("ocr_max_side") or options.get("max_side") or 1280)
        ocr_rgb, coord_scale_x, coord_scale_y = resize_rgb(rgb, max_side)
        h_ocr, w_ocr = ocr_rgb.shape[:2]
        requested = str(options.get("ocr_backend") or options.get("backend") or "paddle")
        use_gpu = bool(options.get("ocr_use_gpu") or options.get("use_gpu") or False)
        scales = parse_scales(options.get("ocr_scales") or options.get("scales") or "1.0,2.0")
        self._ensure_backend(requested, use_gpu)

        raw_detections: list[dict[str, Any]] = []
        if self._backend_name != "paddle":
            raise RuntimeError(f"PaddleOCR backend did not load: {self._backend_error or self._backend_name}")

        for scale in scales:
            rgb_in = scaled_rgb(ocr_rgb, scale)
            if self._backend_is_paddle_v3:
                from waybill_ocr_llm.ocr import flatten_ocr

                items = flatten_ocr(self._run_paddle_v3(rgb_in))
                lines = [
                    (
                        item.box,
                        (item.text, item.confidence),
                    )
                    for item in items
                ]
            else:
                out = self._backend.ocr(rgb_in, cls=True)
                lines = out[0] if out and isinstance(out[0], list) else out
            for line in lines or []:
                try:
                    pts = np.asarray(line[0], dtype=np.float32) / float(scale)
                    text = clean_text(str(line[1][0]))
                    confidence = float(line[1][1])
                except Exception:
                    continue
                bbox_ocr = clamp_bbox(
                    [
                        float(pts[:, 0].min()),
                        float(pts[:, 1].min()),
                        float(pts[:, 0].max()),
                        float(pts[:, 1].max()),
                    ],
                    w_ocr,
                    h_ocr,
                )
                bbox = None
                if bbox_ocr is not None:
                    bbox = clamp_bbox(
                        [
                            bbox_ocr[0] * coord_scale_x,
                            bbox_ocr[1] * coord_scale_y,
                            bbox_ocr[2] * coord_scale_x,
                            bbox_ocr[3] * coord_scale_y,
                        ],
                        width,
                        height,
                    )
                if not text:
                    continue
                raw_detections.append(
                    {
                        "source": f"paddleocr@{scale:g}x",
                        "text": text,
                        "confidence": confidence,
                        "bbox_xyxy": bbox,
                        "box": bbox_to_box(bbox),
                        "cx": None if not bbox else (bbox[0] + bbox[2]) * 0.5,
                        "cy": None if not bbox else (bbox[1] + bbox[3]) * 0.5,
                    }
                )

        return {
            "items": raw_detections,
            "model": self.model_description(),
            "backend": self._backend_name,
            "backend_version": self._backend_version,
            "backend_error": self._backend_error,
            "image_width": width,
            "image_height": height,
            "ocr_width": w_ocr,
            "ocr_height": h_ocr,
            "ocr_scales": scales,
            "ocr_seconds": time.perf_counter() - start,
        }


def paddleocr_version() -> str | None:
    try:
        return package_version("paddleocr")
    except PackageNotFoundError:
        return None
