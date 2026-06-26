from __future__ import annotations

import os
import re
import sys
import time
import unicodedata
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any

import numpy as np

from .schema import OcrItem


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
VALID_ROTATIONS = {0, 90, 180, 270}
DEFAULT_TEXT_DETECTION_MODEL = "PP-OCRv5_mobile_det"
DEFAULT_TEXT_RECOGNITION_MODEL = "korean_PP-OCRv5_mobile_rec"


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("｜", "|")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def image_inputs(paths: list[Path]) -> list[Path]:
    images: list[Path] = []
    for path in paths:
        if path.is_dir():
            images.extend(p for p in sorted(path.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES)
        else:
            images.append(path)
    return sorted(set(images))


def normalize_ocr_rotations(rotations: list[int] | None) -> list[int]:
    if not rotations:
        return [0]
    normalized: list[int] = []
    for rotation in rotations:
        value = int(rotation) % 360
        if value not in VALID_ROTATIONS:
            raise ValueError(f"unsupported OCR rotation {rotation}; use one of 0,90,180,270")
        if value not in normalized:
            normalized.append(value)
    return normalized or [0]


def rotate_image_variant(image_path: Path, rotation: int, out_dir: Path, source_index: int) -> Path:
    rotation = int(rotation) % 360
    if rotation == 0:
        return image_path

    try:
        from PIL import Image, ImageOps
    except Exception as exc:
        raise RuntimeError("Pillow is required for OCR rotation variants") from exc

    variant_dir = out_dir / "_ocr_rotations"
    variant_dir.mkdir(parents=True, exist_ok=True)
    suffix = image_path.suffix if image_path.suffix.lower() in IMAGE_SUFFIXES else ".jpg"
    variant_path = variant_dir / f"{source_index:04d}_{image_path.stem}__rot{rotation}{suffix}"

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if rotation == 90:
            rotated = image.transpose(Image.Transpose.ROTATE_270)
        elif rotation == 180:
            rotated = image.transpose(Image.Transpose.ROTATE_180)
        elif rotation == 270:
            rotated = image.transpose(Image.Transpose.ROTATE_90)
        else:
            raise ValueError(f"unsupported OCR rotation {rotation}; use one of 0,90,180,270")
        rotated.save(variant_path)

    return variant_path


def waybill_crop_variants(image_path: Path, out_dir: Path, source_index: int) -> list[Path]:
    try:
        import cv2
    except Exception:
        return []

    image = cv2.imread(str(image_path))
    if image is None:
        return []
    height, width = image.shape[:2]
    if width < 320 or height < 240:
        return []

    crop_dir = out_dir / "_ocr_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    variants: list[Path] = []
    specs = [
        ("upper_label_wide", 0.39, 0.23, 0.70, 0.56, 4),
        ("upper_addr", 0.44, 0.29, 0.66, 0.50, 4),
        ("label_wide", 0.38, 0.40, 0.72, 0.83, 3),
        ("label_tight", 0.43, 0.47, 0.66, 0.77, 4),
        ("addr_line", 0.46, 0.55, 0.65, 0.72, 4),
        ("bottom_line", 0.43, 0.58, 0.69, 0.80, 4),
    ]

    suffix = image_path.suffix if image_path.suffix.lower() in IMAGE_SUFFIXES else ".jpg"
    variants.extend(detected_waybill_label_crops(image, crop_dir, source_index, image_path.stem, suffix))
    for name, left, top, right, bottom, scale in specs:
        x1 = max(0, min(width - 2, int(width * left)))
        y1 = max(0, min(height - 2, int(height * top)))
        x2 = max(x1 + 2, min(width, int(width * right)))
        y2 = max(y1 + 2, min(height, int(height * bottom)))
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        upscaled = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
        lightness, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lightness = clahe.apply(lightness)
        enhanced = cv2.cvtColor(cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
        blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        sharpened = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)

        variant_path = crop_dir / f"{source_index:04d}_{image_path.stem}__{name}_x{scale}{suffix}"
        if cv2.imwrite(str(variant_path), sharpened):
            variants.append(variant_path)
    return variants


def full_image_preprocess_variants(image_path: Path, out_dir: Path, source_index: int) -> list[Path]:
    try:
        import cv2
    except Exception:
        return []

    image = cv2.imread(str(image_path))
    if image is None:
        return []
    height, width = image.shape[:2]
    if width < 320 or height < 240:
        return []

    variant_dir = out_dir / "_ocr_full_variants"
    variant_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".png"
    variants: list[Path] = []
    for scale in (2,):
        upscaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        variant_path = variant_dir / f"{source_index:04d}_{image_path.stem}__full_x{scale}{suffix}"
        if cv2.imwrite(str(variant_path), upscaled):
            variants.append(variant_path)
    return variants


def detected_waybill_label_crops(
    image: np.ndarray,
    crop_dir: Path,
    source_index: int,
    image_stem: str,
    suffix: str,
) -> list[Path]:
    try:
        import cv2
    except Exception:
        return []

    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    image_area = float(width * height)
    boxes: list[tuple[float, int, int, int, int]] = []

    raw_white_mask = ((value > 145) & (saturation < 85)).astype(np.uint8) * 255
    component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(raw_white_mask, 8)
    for component_idx in range(1, component_count):
        x, y, w, h, area = (int(item) for item in stats[component_idx])
        rect_area = float(w * h)
        if rect_area < image_area * 0.004 or rect_area > image_area * 0.22:
            continue
        if w < 45 or h < 28:
            continue
        aspect = w / max(1.0, float(h))
        if aspect < 0.45 or aspect > 3.2:
            continue
        fill = float(area) / max(1.0, rect_area)
        if fill < 0.18 or fill > 0.92:
            continue
        center_x = (x + w / 2.0) / max(1, width)
        center_y = (y + h / 2.0) / max(1, height)
        margin_penalty = 0.55 if center_x < 0.05 or center_x > 0.95 or center_y < 0.04 or center_y > 0.96 else 0.0
        score = rect_area * fill * (1.0 - margin_penalty)
        boxes.append((score, x, y, w, h))

    bright_mask = ((value > 125) & (saturation < 95)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
    mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(w * h)
        if area < image_area * 0.006 or area > image_area * 0.35:
            continue
        if w < 60 or h < 35:
            continue
        aspect = w / max(1.0, float(h))
        if aspect < 0.55 or aspect > 3.8:
            continue
        roi_mask = mask[y : y + h, x : x + w]
        fill = float(cv2.countNonZero(roi_mask)) / max(1.0, area)
        if fill < 0.22:
            continue
        # Prefer label-like white blocks away from black letterbox margins.
        center_x = (x + w / 2.0) / max(1, width)
        center_y = (y + h / 2.0) / max(1, height)
        margin_penalty = 0.6 if center_x < 0.12 or center_x > 0.88 or center_y < 0.08 or center_y > 0.92 else 0.0
        score = area * fill * (1.0 - margin_penalty)
        boxes.append((score, x, y, w, h))

    boxes.sort(reverse=True)
    written: list[Path] = []
    seen: list[tuple[int, int, int, int]] = []
    for rank, (_score, x, y, w, h) in enumerate(boxes[:4], start=1):
        x1 = max(0, x - int(w * 0.08))
        y1 = max(0, y - int(h * 0.12))
        x2 = min(width, x + w + int(w * 0.08))
        y2 = min(height, y + h + int(h * 0.14))
        box = (x1, y1, x2, y2)
        if any(iou_boxes(box, other) > 0.75 for other in seen):
            continue
        seen.append(box)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        written.extend(write_label_crop_variants(crop, crop_dir, source_index, image_stem, rank, suffix))
    return written


def iou_boxes(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1 = max(lx1, rx1)
    iy1 = max(ly1, ry1)
    ix2 = min(lx2, rx2)
    iy2 = min(ly2, ry2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if intersection <= 0:
        return 0.0
    left_area = max(0, lx2 - lx1) * max(0, ly2 - ly1)
    right_area = max(0, rx2 - rx1) * max(0, ry2 - ry1)
    return intersection / max(1.0, float(left_area + right_area - intersection))


def write_label_crop_variants(
    crop: np.ndarray,
    crop_dir: Path,
    source_index: int,
    image_stem: str,
    rank: int,
    suffix: str,
) -> list[Path]:
    try:
        import cv2
    except Exception:
        return []

    variants: list[Path] = []
    scale = 5 if max(crop.shape[:2]) < 180 else 4
    upscaled = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    enhanced = cv2.cvtColor(cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    sharpened = cv2.addWeighted(enhanced, 1.65, blur, -0.65, 0)

    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    for suffix_name, image in (("detected_label", sharpened), ("detected_label_bw", binary)):
        path = crop_dir / f"{source_index:04d}_{image_stem}__{suffix_name}_{rank}_x{scale}{suffix}"
        if cv2.imwrite(str(path), image):
            variants.append(path)
    return variants


def ocr_variant_quality(result: dict[str, Any]) -> tuple[float, int, int]:
    items = result.get("ocr_items") or []
    best_candidate_score = -999.0
    try:
        from .candidates import destination_candidates

        candidates = destination_candidates(items)
        if candidates:
            best_candidate_score = max(float(candidate.get("score") or 0.0) for candidate in candidates)
    except Exception:
        best_candidate_score = -999.0

    text = " ".join(str(item.get("text") or "") for item in items)
    keyword_score = sum(
        1
        for keyword in ("운송장", "받는분", "받으시는분", "수취인", "배송지", "배송주소", "주소", "연락처")
        if keyword in text
    )
    return best_candidate_score, keyword_score, len(items)


def paddleocr_major_version() -> int:
    try:
        return int(package_version("paddleocr").split(".", 1)[0])
    except (PackageNotFoundError, ValueError):
        return 2


def using_paddleocr_v3() -> bool:
    return paddleocr_major_version() >= 3


def require_paddleocr() -> bool:
    return os.environ.get("WAYBILL_OCR_REQUIRE_PADDLE", "1").strip().lower() in {"1", "true", "yes", "on"}


def ocr_model_label(
    use_gpu: bool,
    rec_batch_num: int,
    text_detection_model_name: str = DEFAULT_TEXT_DETECTION_MODEL,
    text_recognition_model_name: str = DEFAULT_TEXT_RECOGNITION_MODEL,
) -> str:
    device = "gpu:0" if use_gpu else "cpu"
    if using_paddleocr_v3():
        return (
            "PaddleOCR3("
            f"det={text_detection_model_name}, "
            f"rec={text_recognition_model_name}, "
            f"device={device}, rec_batch={rec_batch_num})"
        )
    return f"PaddleOCR2(lang=korean, use_angle_cls=True, device={device}, rec_batch={rec_batch_num})"


def build_paddleocr(
    use_gpu: bool = False,
    rec_batch_num: int = 1,
    text_detection_model_name: str = DEFAULT_TEXT_DETECTION_MODEL,
    text_recognition_model_name: str = DEFAULT_TEXT_RECOGNITION_MODEL,
):
    if use_gpu:
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
        import paddle

        paddle.set_device("gpu:0")

    from paddleocr import PaddleOCR

    if using_paddleocr_v3():
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        return PaddleOCR(
            device="gpu:0" if use_gpu else "cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
            text_detection_model_name=text_detection_model_name,
            text_recognition_model_name=text_recognition_model_name,
            text_recognition_batch_size=rec_batch_num,
        )

    return PaddleOCR(
        lang="korean",
        use_angle_cls=True,
        use_gpu=use_gpu,
        rec_batch_num=rec_batch_num,
        show_log=False,
    )


def run_paddleocr(ocr: Any, image_path: Path):
    if using_paddleocr_v3():
        return ocr.predict(str(image_path))
    return ocr.ocr(str(image_path), cls=True)


def box_center(box: list[list[float]]) -> tuple[float, float]:
    pts = np.asarray(box, dtype=np.float32)
    center = pts.mean(axis=0)
    return float(center[0]), float(center[1])


def flatten_paddleocr_v3_page(page: Any) -> list[OcrItem] | None:
    page_json = getattr(page, "json", None)
    page_data = None
    if isinstance(page_json, dict):
        page_data = page_json.get("res", page_json)
    elif isinstance(page, dict):
        page_data = page.get("res", page)

    if not isinstance(page_data, dict) or "rec_texts" not in page_data:
        return None

    texts = page_data.get("rec_texts") or []
    scores = page_data.get("rec_scores") or []
    polys = page_data.get("rec_polys") or page_data.get("dt_polys") or []

    items: list[OcrItem] = []
    for idx, text_value in enumerate(texts):
        text = normalize_text(str(text_value))
        if not text or idx >= len(polys):
            continue
        pts = np.asarray(polys[idx], dtype=np.float32).reshape(-1, 2)
        if len(pts) < 4:
            continue
        box = [[float(x), float(y)] for x, y in pts[:4]]
        conf = float(scores[idx]) if idx < len(scores) else 0.0
        cx, cy = box_center(box)
        items.append(OcrItem(text=text, confidence=conf, box=box, cx=cx, cy=cy))
    return items


def flatten_ocr(result: Any) -> list[OcrItem]:
    items: list[OcrItem] = []
    for page in result or []:
        v3_items = flatten_paddleocr_v3_page(page)
        if v3_items is not None:
            items.extend(v3_items)
            continue
        for line in page or []:
            if len(line) < 2:
                continue
            box = [[float(x), float(y)] for x, y in line[0]]
            text, conf = line[1]
            text = normalize_text(str(text))
            if not text:
                continue
            cx, cy = box_center(box)
            items.append(OcrItem(text=text, confidence=float(conf), box=box, cx=cx, cy=cy))
    items.sort(key=lambda item: (round(item.cy / 18), item.cx))
    return items


def run_ocr_on_images(
    inputs: list[Path],
    out_dir: Path,
    use_gpu: bool = False,
    rec_batch_num: int = 1,
    limit: int = 0,
    rotations: list[int] | None = None,
    crop_variants: bool | None = None,
    full_image_variants: bool | None = None,
) -> dict[str, Any]:
    images = image_inputs(inputs)
    if limit > 0:
        images = images[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)
    ocr_rotations = normalize_ocr_rotations(rotations)

    variants: list[tuple[Path, int, Path, str]] = []
    for source_index, image_path in enumerate(images, 1):
        for rotation in ocr_rotations:
            variant_path = rotate_image_variant(image_path, rotation, out_dir, source_index)
            variants.append((image_path, rotation, variant_path, "rotation" if rotation != 0 else "original"))
        use_crop_variants = (
            os.environ.get("WAYBILL_OCR_CROP_VARIANTS", "0").strip().lower() in {"1", "true", "yes", "on"}
            if crop_variants is None
            else bool(crop_variants)
        )
        if use_crop_variants:
            for crop_path in waybill_crop_variants(image_path, out_dir, source_index):
                variants.append((image_path, 0, crop_path, "crop"))
        use_full_variants = (
            os.environ.get("WAYBILL_OCR_FULL_IMAGE_VARIANTS", "0").strip().lower() in {"1", "true", "yes", "on"}
            if full_image_variants is None
            else bool(full_image_variants)
        )
        if use_full_variants:
            for full_variant_path in full_image_preprocess_variants(image_path, out_dir, source_index):
                variants.append((image_path, 0, full_variant_path, "full_preprocess"))

    ocr = None
    ocr_engines: list[tuple[str, Any]] = []
    use_paddle = True
    must_use_paddle = require_paddleocr()
    fallback_reason: str | None = None
    try:
        ocr = build_paddleocr(use_gpu=use_gpu, rec_batch_num=rec_batch_num)
        ocr_engines.append((ocr_model_label(use_gpu=use_gpu, rec_batch_num=rec_batch_num), ocr))
    except Exception as exc:
        fallback_reason = str(exc)
        if use_gpu:
            print(
                f"[waybill_ocr] GPU PaddleOCR init failed, retrying CPU PaddleOCR: {fallback_reason}",
                file=sys.stderr,
            )
            try:
                ocr = build_paddleocr(use_gpu=False, rec_batch_num=rec_batch_num)
                ocr_engines = [(ocr_model_label(use_gpu=False, rec_batch_num=rec_batch_num), ocr)]
                use_gpu = False
            except Exception as cpu_exc:
                fallback_reason = f"GPU init failed: {fallback_reason}; CPU init failed: {cpu_exc}"
                if must_use_paddle:
                    raise RuntimeError(f"PaddleOCR init failed and fallback is disabled: {fallback_reason}") from cpu_exc
                use_paddle = False
                print(f"[waybill_ocr] PaddleOCR init failed, falling back to pytesseract: {fallback_reason}", file=sys.stderr)
        else:
            if must_use_paddle:
                raise RuntimeError(f"PaddleOCR init failed and fallback is disabled: {fallback_reason}") from exc
            use_paddle = False
            print(f"[waybill_ocr] PaddleOCR init failed, falling back to pytesseract: {fallback_reason}", file=sys.stderr)

    def run_tesseract(image_path: Path) -> list[OcrItem]:
        try:
            import pytesseract
            from PIL import Image
        except Exception as exc:
            raise RuntimeError(
                "pytesseract is unavailable. Install paddlepaddle for PaddleOCR or add pytesseract dependency. "
                f"Fallback reason: {fallback_reason or 'unknown'}"
            ) from exc

        image = Image.open(image_path).convert("RGB")
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        texts = data.get("text") or []
        confidences = data.get("conf") or []
        xs = data.get("left") or []
        ys = data.get("top") or []
        widths = data.get("width") or []
        heights = data.get("height") or []

        items: list[OcrItem] = []
        for i, raw_text in enumerate(texts):
            text = normalize_text(str(raw_text or ""))
            if not text:
                continue
            conf = confidences[i] if i < len(confidences) else "0"
            try:
                conf_value = float(conf)
            except (TypeError, ValueError):
                conf_value = 0.0
            if conf_value < 0:
                continue

            x = float(xs[i]) if i < len(xs) else 0.0
            y = float(ys[i]) if i < len(ys) else 0.0
            w = float(widths[i]) if i < len(widths) else 0.0
            h = float(heights[i]) if i < len(heights) else 0.0
            box = [
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h],
            ]
            cx = x + w / 2
            cy = y + h / 2
            items.append(OcrItem(text=text, confidence=conf_value / 100.0, box=box, cx=cx, cy=cy))

        items.sort(key=lambda item: (round(item.cy / 18), item.cx))
        return items

    results: list[dict[str, Any]] = []
    model_label = 'TesseractOCR(cpu)' if not use_paddle else " + ".join(label for label, _engine in ocr_engines)
    start = time.perf_counter()
    if use_paddle and ocr_engines:
        total_passes = len(variants) * len(ocr_engines)
        pass_idx = 0
        for source_image_path, rotation, image_path, variant_kind in variants:
            for engine_label, engine in ocr_engines:
                pass_idx += 1
                t0 = time.perf_counter()
                try:
                    raw = run_paddleocr(engine, image_path)
                    items = flatten_ocr(raw)
                except Exception as exc:
                    if must_use_paddle:
                        raise RuntimeError(
                            f"PaddleOCR failed at image stage ({image_path.name}, model={engine_label}) and fallback is disabled: {exc}"
                        ) from exc
                    print(
                        f"[waybill_ocr] PaddleOCR failed at image stage ({image_path.name}, model={engine_label}), trying pytesseract fallback: {exc}",
                        file=sys.stderr,
                    )
                    items = run_tesseract(image_path)
                    engine_label = 'TesseractOCR(cpu)'
                elapsed = time.perf_counter() - t0
                item_payloads: list[dict[str, Any]] = []
                for item in items:
                    item_payload = asdict(item)
                    item_payload["source_image"] = str(source_image_path)
                    item_payload["variant_image"] = str(image_path)
                    item_payload["rotation_degrees"] = rotation
                    item_payload["variant_kind"] = variant_kind
                    item_payload["ocr_model"] = engine_label
                    item_payloads.append(item_payload)
                result = {
                    "image": str(image_path),
                    "source_image": str(source_image_path),
                    "rotation_degrees": rotation,
                    "rotation_variant": rotation != 0,
                    "variant_kind": variant_kind,
                    "ocr_model": engine_label,
                    "ocr_seconds": elapsed,
                    "ocr_item_count": len(item_payloads),
                    "ocr_items": item_payloads,
                }
                results.append(result)
                print(
                    f"[{pass_idx}/{total_passes}] OCR {image_path.name} rot={rotation} model={engine_label}: items={len(items)} sec={elapsed:.3f}",
                    file=sys.stderr,
                )
    else:
        for idx, (source_image_path, rotation, image_path, variant_kind) in enumerate(variants, 1):
            t0 = time.perf_counter()
            items = run_tesseract(image_path)
            elapsed = time.perf_counter() - t0
            item_payloads = []
            for item in items:
                item_payload = asdict(item)
                item_payload["source_image"] = str(source_image_path)
                item_payload["variant_image"] = str(image_path)
                item_payload["rotation_degrees"] = rotation
                item_payload["variant_kind"] = variant_kind
                item_payload["ocr_model"] = 'TesseractOCR(cpu)'
                item_payloads.append(item_payload)
            result = {
                "image": str(image_path),
                "source_image": str(source_image_path),
                "rotation_degrees": rotation,
                "rotation_variant": rotation != 0,
                "variant_kind": variant_kind,
                "ocr_model": 'TesseractOCR(cpu)',
                "ocr_seconds": elapsed,
                "ocr_item_count": len(item_payloads),
                "ocr_items": item_payloads,
            }
            results.append(result)
            print(f"[{idx}/{len(variants)}] OCR {image_path.name} rot={rotation}: items={len(items)} sec={elapsed:.3f}", file=sys.stderr)

    total = time.perf_counter() - start
    combined_results: list[dict[str, Any]] = []
    for image_path in images:
        source_results = [result for result in results if result.get("source_image") == str(image_path)]
        source_results.sort(key=ocr_variant_quality, reverse=True)
        source_items: list[dict[str, Any]] = []
        for result in source_results:
            source_items.extend(result.get("ocr_items") or [])
        combined_results.append(
            {
                "image": str(image_path),
                "source_image": str(image_path),
                "rotations": [int(result.get("rotation_degrees") or 0) for result in source_results],
                "ocr_seconds": sum(float(result.get("ocr_seconds") or 0.0) for result in source_results),
                "ocr_item_count": len(source_items),
                "ocr_items": source_items,
                "variant_count": len(source_results),
                "variants": [
                    {
                        "image": result.get("image"),
                        "rotation_degrees": result.get("rotation_degrees"),
                        "variant_kind": result.get("variant_kind"),
                        "ocr_model": result.get("ocr_model"),
                        "ocr_item_count": result.get("ocr_item_count"),
                        "ocr_seconds": result.get("ocr_seconds"),
                    }
                    for result in source_results
                ],
            }
        )
    payload = {
        "model": model_label,
        "image_count": len(variants),
        "source_image_count": len(images),
        "rotations": ocr_rotations,
        "total_seconds": total,
        "fps_images": len(variants) / total if total > 0 else 0.0,
        "results": results,
        "combined_results": combined_results,
    }
    (out_dir / "waybill_ocr_results.json").write_text(
        json_dumps(payload),
        encoding="utf-8",
    )
    return payload


def json_dumps(payload: Any) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
