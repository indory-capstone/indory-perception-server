#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from waybill_ocr_llm.ocr import build_paddleocr, flatten_ocr, normalize_text, run_paddleocr
from waybill_ocr_llm.schema import OcrItem


DEFAULT_LABEL_IMAGE_DIR = Path("benchmark/data/labels")
DEFAULT_GT_DIR = Path("benchmark/data/label_ground_truth")
DEFAULT_OUT_DIR = Path("benchmark/runs/paddleocr_rec_room")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ROOM_RE = re.compile(r"(?<![A-Z0-9])([B]?\d{2,4}(?:-\d{1,2})?)\s*호", re.IGNORECASE)
ROOM_FALLBACK_RE = re.compile(r"(?<![A-Z0-9])([B]?\d{3,4})(?![A-Z0-9])", re.IGNORECASE)
NOISY_NON_TARGET_RE = re.compile(
    r"(?:010|TRK|ORD|DATE|운송장|주문|상품|배송메모|출고|집하|접수|\*{2,}|-{2,})",
    re.IGNORECASE,
)


@dataclass
class ExportedSample:
    label_id: str
    label: str
    split: str
    image: str
    source_image: str
    source_gt: str
    recipient_detail_address: str
    carrier: str | None
    form_type: str | None
    ocr_text: str
    ocr_confidence: float
    match_score: float
    crop_box: list[int]
    augmentation: str


def json_dumps(payload: Any, *, indent: int | None = None) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=indent, sort_keys=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json_dumps(row) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected object JSON: {path}")
    return data


def compare_key(value: str) -> str:
    text = normalize_text(value).upper()
    text = text.replace(" ", "")
    text = text.replace("O", "0")
    text = text.replace("I", "1")
    text = text.replace("L", "1")
    text = re.sub(r"[^A-Z0-9가-힣-]", "", text)
    return text


def digits_key(value: str) -> str:
    text = compare_key(value)
    return re.sub(r"[^A-Z0-9-]", "", text)


def room_label_from_detail(detail: str) -> str | None:
    text = normalize_text(detail).upper()
    match = ROOM_RE.search(text)
    if match is None:
        match = ROOM_FALLBACK_RE.search(text)
    if match is None:
        return None
    room = match.group(1).replace(" ", "").upper()
    return f"{room}호"


def find_label_images(paths: list[Path]) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for root in paths:
        root = root.expanduser()
        candidates = [root] if root.is_file() else sorted(root.glob("*"))
        for path in candidates:
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                images.setdefault(path.stem, path.resolve())
    return images


def load_gt_records(gt_dir: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    records: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(gt_dir.expanduser().glob("*.json")):
        data = read_json(path)
        label_id = str(data.get("label_id") or path.stem).strip()
        if label_id:
            records[label_id] = (path.resolve(), data)
    return records


def target_span(text: str, target: str) -> tuple[int, int, int] | None:
    haystack = compare_key(text)
    needle = compare_key(target)
    if not haystack or not needle:
        return None
    index = haystack.find(needle)
    if index < 0:
        target_digits = digits_key(target.replace("호", ""))
        index = haystack.find(target_digits)
        needle = target_digits
    if index < 0 or not needle:
        return None
    return index, index + len(needle), len(haystack)


def item_bounds(item: OcrItem, width: int, height: int, pad_ratio: float, min_pad: int, target: str) -> list[int]:
    xs = [float(point[0]) for point in item.box]
    ys = [float(point[1]) for point in item.box]
    x1 = max(0.0, min(xs))
    y1 = max(0.0, min(ys))
    x2 = min(float(width), max(xs))
    y2 = min(float(height), max(ys))

    span = target_span(item.text, target)
    if span is not None:
        start, end, text_len = span
        target_len = end - start
        if text_len > target_len + 3:
            box_width = x2 - x1
            box_height = y2 - y1
            if end >= text_len - 1:
                desired_width = min(box_width, max(50.0, target_len * box_height * 0.48))
                x1 = max(0.0, x2 - desired_width)
            else:
                margin = 0
                left_ratio = max(0.0, (start - margin) / max(1, text_len))
                right_ratio = min(1.0, (end + margin) / max(1, text_len))
                original_x1 = x1
                original_x2 = x2
                box_width = original_x2 - original_x1
                x1 = original_x1 + box_width * left_ratio
                x2 = x1 + box_width * max(0.08, right_ratio - left_ratio)

    pad_x = max(float(min_pad), (x2 - x1) * pad_ratio)
    pad_y = max(float(min_pad), (y2 - y1) * pad_ratio)
    return [
        max(0, int(round(x1 - pad_x))),
        max(0, int(round(y1 - pad_y))),
        min(width, int(round(x2 + pad_x))),
        min(height, int(round(y2 + pad_y))),
    ]


def text_match_score(item: OcrItem, target: str) -> float:
    text = normalize_text(item.text)
    item_key = compare_key(text)
    target_key = compare_key(target)
    item_digits = digits_key(text.replace("호", ""))
    target_digits = digits_key(target.replace("호", ""))
    score = float(item.confidence) * 20.0

    if not item_key or not item_digits:
        return -100.0
    if item_key == target_key:
        score += 140.0
    elif target_key in item_key:
        score += 115.0
    elif item_key in target_key and len(item_key) >= max(3, len(target_key) - 1):
        score += 60.0
    if item_digits == target_digits:
        score += 100.0
    elif target_digits and target_digits in item_digits:
        score += 45.0
    elif (
        target_digits
        and item_digits in target_digits
        and len(item_digits) >= max(3, len(target_digits) - 1)
    ):
        score += 25.0

    if "호" in text or "R" in item_key:
        score += 12.0
    if NOISY_NON_TARGET_RE.search(text):
        score -= 70.0
    if len(item_digits) > len(target_digits) + 4:
        score -= 35.0
    if len(text) > 24:
        score -= 20.0
    return score


def best_room_item(items: list[OcrItem], target: str) -> tuple[OcrItem | None, float]:
    best_item: OcrItem | None = None
    best_score = -999.0
    for item in items:
        score = text_match_score(item, target)
        if score > best_score:
            best_score = score
            best_item = item
    return best_item, best_score


def split_for_label(label_id: str, val_ratio: float) -> str:
    if val_ratio <= 0:
        return "train"
    if val_ratio >= 1:
        return "val"
    digest = hashlib.sha256(label_id.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if value < val_ratio else "train"


def deterministic_rng(label_id: str, aug_index: int) -> random.Random:
    digest = hashlib.sha256(f"{label_id}:{aug_index}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def augment_crop(crop: Any, label_id: str, aug_index: int) -> Any:
    import cv2
    import numpy as np

    rng = deterministic_rng(label_id, aug_index)
    height, width = crop.shape[:2]
    image = crop.copy()

    scale = rng.uniform(0.42, 0.86)
    small_w = max(8, int(round(width * scale)))
    small_h = max(8, int(round(height * scale)))
    image = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

    if rng.random() < 0.75:
        sigma = rng.uniform(0.3, 1.2)
        image = cv2.GaussianBlur(image, (0, 0), sigma)

    alpha = rng.uniform(0.78, 1.18)
    beta = rng.uniform(-18.0, 18.0)
    image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    if rng.random() < 0.8:
        quality = int(rng.uniform(45, 88))
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is not None:
                image = decoded

    return image


def trim_to_right_text_token(crop: Any, min_pad: int, target: str) -> Any:
    import cv2
    import numpy as np

    if crop.size == 0:
        return crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    _threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    height, width = binary.shape[:2]
    if width < 40 or height < 12:
        return crop

    ink = np.sum(binary > 0, axis=0)
    active = ink > max(1, int(height * 0.025))
    active_indices = np.flatnonzero(active)
    if len(active_indices) == 0:
        return crop

    first = int(active_indices[0])
    last = int(active_indices[-1])
    if last <= first:
        return crop

    groups: list[tuple[int, int]] = []
    start = first
    previous = first
    min_gap = max(10, int(width * 0.07))
    for index in active_indices[1:]:
        value = int(index)
        if value - previous > min_gap:
            groups.append((start, previous))
            start = value
        previous = value
    groups.append((start, previous))

    if len(groups) < 2:
        return crop
    target_len = max(3, len(compare_key(target)))
    min_token_width = max(24, int(target_len * height * 0.38))
    group_index = len(groups) - 1
    token_start, token_end = groups[group_index]
    while token_end - token_start < min_token_width and group_index > 0:
        group_index -= 1
        token_start = groups[group_index][0]
    if token_end - token_start < max(12, int(width * 0.12)):
        return crop
    x1 = max(0, token_start - max(min_pad, int(width * 0.025)))
    x2 = min(width, token_end + max(min_pad, int(width * 0.025)))
    if x2 - x1 < max(24, int(width * 0.18)):
        return crop
    return crop[:, x1:x2]


def safe_output_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing = [child for child in path.iterdir() if child.name not in {".gitkeep"}]
    if existing and not overwrite:
        raise FileExistsError(f"output dir is not empty; pass --overwrite to reuse it: {path}")
    image_dir = path / "images"
    if overwrite:
        for file_name in ("rec_gt_train.txt", "rec_gt_val.txt", "manifest.jsonl", "unmatched.jsonl", "summary.json"):
            target = path / file_name
            if target.is_file():
                target.unlink()
        if image_dir.is_dir():
            for child in image_dir.iterdir():
                if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES:
                    child.unlink()
    image_dir.mkdir(parents=True, exist_ok=True)


def write_crop(
    *,
    crop: Any,
    image_dir: Path,
    label_id: str,
    target: str,
    split: str,
    source_path: Path,
    gt_path: Path,
    gt: dict[str, Any],
    item: OcrItem,
    match_score: float,
    crop_box: list[int],
    augmentation: str,
) -> ExportedSample:
    import cv2

    suffix = ".png"
    image_name = f"{label_id}__room_{augmentation}{suffix}"
    image_path = image_dir / image_name
    if not cv2.imwrite(str(image_path), crop):
        raise RuntimeError(f"failed to write crop: {image_path}")
    return ExportedSample(
        label_id=label_id,
        label=target,
        split=split,
        image=str(image_path),
        source_image=str(source_path),
        source_gt=str(gt_path),
        recipient_detail_address=str(gt.get("recipient_detail_address") or ""),
        carrier=gt.get("carrier"),
        form_type=gt.get("form_type"),
        ocr_text=item.text,
        ocr_confidence=float(item.confidence),
        match_score=float(match_score),
        crop_box=crop_box,
        augmentation=augmentation,
    )


def export_dataset(args: argparse.Namespace) -> dict[str, Any]:
    import cv2

    label_image_dirs = args.label_image_dir or [DEFAULT_LABEL_IMAGE_DIR]
    label_images = find_label_images(label_image_dirs)
    gt_records = load_gt_records(args.gt_dir)
    out_dir = args.out_dir.expanduser().resolve()
    safe_output_dir(out_dir, args.overwrite)
    image_dir = out_dir / "images"

    use_gpu = bool(args.gpu)
    ocr = build_paddleocr(
        use_gpu=use_gpu,
        rec_batch_num=args.rec_batch_num,
        text_detection_model_name=args.text_detection_model,
        text_recognition_model_name=args.text_recognition_model,
    )

    rows: list[ExportedSample] = []
    unmatched: list[dict[str, Any]] = []
    common_ids = sorted(set(label_images) & set(gt_records))
    if args.limit > 0:
        common_ids = common_ids[: args.limit]

    for index, label_id in enumerate(common_ids, start=1):
        source_path = label_images[label_id]
        gt_path, gt = gt_records[label_id]
        detail = str(gt.get("recipient_detail_address") or "")
        target = room_label_from_detail(detail)
        if target is None:
            unmatched.append(
                {
                    "label_id": label_id,
                    "reason": "no_room_in_detail_address",
                    "recipient_detail_address": detail,
                    "source_gt": str(gt_path),
                }
            )
            continue

        image = cv2.imread(str(source_path))
        if image is None:
            unmatched.append({"label_id": label_id, "reason": "image_read_failed", "source_image": str(source_path)})
            continue

        try:
            ocr_result = run_paddleocr(ocr, source_path)
            items = flatten_ocr(ocr_result)
        except Exception as exc:
            unmatched.append(
                {
                    "label_id": label_id,
                    "reason": "ocr_failed",
                    "source_image": str(source_path),
                    "error": str(exc),
                }
            )
            continue

        item, match_score = best_room_item(items, target)
        if item is None or match_score < args.min_score:
            unmatched.append(
                {
                    "label_id": label_id,
                    "reason": "target_item_not_matched",
                    "target": target,
                    "match_score": match_score,
                    "recipient_detail_address": detail,
                    "source_image": str(source_path),
                    "source_gt": str(gt_path),
                    "ocr_items": [asdict(ocr_item) for ocr_item in items],
                }
            )
            continue

        height, width = image.shape[:2]
        crop_box = item_bounds(item, width, height, args.pad_ratio, args.min_pad, target)
        x1, y1, x2, y2 = crop_box
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            unmatched.append({"label_id": label_id, "reason": "empty_crop", "crop_box": crop_box})
            continue
        crop = trim_to_right_text_token(crop, args.min_pad, target)

        split = split_for_label(label_id, args.val_ratio)
        rows.append(
            write_crop(
                crop=crop,
                image_dir=image_dir,
                label_id=label_id,
                target=target,
                split=split,
                source_path=source_path,
                gt_path=gt_path,
                gt=gt,
                item=item,
                match_score=match_score,
                crop_box=crop_box,
                augmentation="orig",
            )
        )

        for aug_index in range(1, args.augment + 1):
            augmented = augment_crop(crop, label_id, aug_index)
            rows.append(
                write_crop(
                    crop=augmented,
                    image_dir=image_dir,
                    label_id=label_id,
                    target=target,
                    split=split,
                    source_path=source_path,
                    gt_path=gt_path,
                    gt=gt,
                    item=item,
                    match_score=match_score,
                    crop_box=crop_box,
                    augmentation=f"aug{aug_index:02d}",
                )
            )

        if args.verbose:
            print(f"[{index}/{len(common_ids)}] exported {label_id}: {target} from {item.text!r}")

    train_lines: list[str] = []
    val_lines: list[str] = []
    manifest_rows: list[dict[str, Any]] = []
    for row in rows:
        image_rel = os.path.relpath(row.image, out_dir)
        line = f"{image_rel}\t{row.label}"
        if row.split == "val":
            val_lines.append(line)
        else:
            train_lines.append(line)
        manifest_item = asdict(row)
        manifest_item["image_rel"] = image_rel
        manifest_rows.append(manifest_item)

    (out_dir / "rec_gt_train.txt").write_text("\n".join(train_lines) + ("\n" if train_lines else ""), encoding="utf-8")
    (out_dir / "rec_gt_val.txt").write_text("\n".join(val_lines) + ("\n" if val_lines else ""), encoding="utf-8")
    write_jsonl(out_dir / "manifest.jsonl", manifest_rows)
    write_jsonl(out_dir / "unmatched.jsonl", unmatched)

    summary = {
        "ok": True,
        "out_dir": str(out_dir),
        "label_image_dirs": [str(path.expanduser()) for path in label_image_dirs],
        "gt_dir": str(args.gt_dir.expanduser()),
        "label_image_count": len(label_images),
        "gt_count": len(gt_records),
        "matched_source_count": len(common_ids),
        "exported_source_count": len({row.label_id for row in rows}),
        "exported_image_count": len(rows),
        "train_count": len(train_lines),
        "val_count": len(val_lines),
        "unmatched_count": len(unmatched),
        "augment": args.augment,
        "target": "room",
        "min_score": args.min_score,
        "text_detection_model": args.text_detection_model,
        "text_recognition_model": args.text_recognition_model,
        "use_gpu": use_gpu,
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Indory generated waybill room crops in PaddleOCR recognition format."
    )
    parser.add_argument(
        "--label-image-dir",
        action="append",
        type=Path,
        default=None,
        help="Directory or image file containing clean generated waybill labels; repeatable",
    )
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--overwrite", action="store_true", help="reuse an existing output directory")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.12)
    parser.add_argument("--augment", type=int, default=2, help="number of degraded variants per crop")
    parser.add_argument("--min-score", type=float, default=75.0)
    parser.add_argument("--pad-ratio", type=float, default=0.28)
    parser.add_argument("--min-pad", type=int, default=5)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--rec-batch-num", type=int, default=1)
    parser.add_argument("--text-detection-model", default="PP-OCRv5_mobile_det")
    parser.add_argument("--text-recognition-model", default="korean_PP-OCRv5_mobile_rec")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = export_dataset(args)
    except Exception as exc:
        print(f"export failed: {exc}", file=sys.stderr)
        return 2
    print(json_dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
