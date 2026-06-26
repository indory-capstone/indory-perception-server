from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
FONT_BOLD_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = FONT_BOLD_CANDIDATES if bold else FONT_CANDIDATES
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default(size=size)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sample_id_from_path(path: Any) -> str:
    stem = Path(str(path or "")).stem
    return re.sub(r"_640x480$", "", stem)


def by_image_name(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {Path(str(item["image"])).name: item for item in results}


def load_eval_rows(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists() or not path.read_text(encoding="utf-8").strip():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        return {row["sample_id"]: row for row in csv.DictReader(f)}


def scaled_poly(box: list[list[float]], scale: float, offset: tuple[int, int]) -> list[tuple[float, float]]:
    ox, oy = offset
    return [(float(x) * scale + ox, float(y) * scale + oy) for x, y in box]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    text = str(text).replace("\n", " ")
    lines: list[str] = []
    current = ""
    for token in text.split(" "):
        candidate = token if not current else f"{current} {token}"
        if draw.textlength(candidate, font=fnt) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = token
    if current:
        lines.append(current)
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
    max_width: int,
    max_lines: int | None = None,
) -> int:
    x, y = xy
    lines = wrap_text(draw, text, fnt, max_width)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip() + " ..."
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + 5
    return y


def destination_text(dong: Any, floor: Any, room: Any) -> str:
    parts = [str(value) for value in (dong, floor, room) if value]
    return " ".join(parts) if parts else "NO_DEST"


def eval_gt_text(row: dict[str, str] | None) -> str:
    if not row:
        return ""
    return destination_text(row.get("gt_dong"), row.get("gt_floor"), row.get("gt_room"))


def eval_pred_text(row: dict[str, str] | None, decision: dict[str, Any] | None) -> str:
    if row:
        return destination_text(row.get("pred_dong"), row.get("pred_floor"), row.get("pred_room"))
    decision = decision or {}
    return destination_text(decision.get("destination_dong"), decision.get("destination_floor"), decision.get("destination_room"))


def bool_text(value: Any) -> str:
    text = str(value)
    if text == "True":
        return "PASS"
    if text == "False":
        return "FAIL"
    return text


def item_rotation(item: dict[str, Any]) -> int:
    try:
        return int(item.get("rotation_degrees") or 0)
    except (TypeError, ValueError):
        return 0


def evidence_trace_text(ocr_items: list[dict[str, Any]], decision: dict[str, Any]) -> str:
    evidence = [int(i) for i in decision.get("evidence_indices") or [] if isinstance(i, int | float)]
    snippets: list[str] = []
    for idx in evidence:
        if 0 <= idx < len(ocr_items):
            item = ocr_items[idx]
            text = str(item.get("text") or "")
            try:
                confidence = float(item.get("confidence") or 0.0)
                snippets.append(f"#{idx} \"{text}\" conf={confidence:.3f}")
            except (TypeError, ValueError):
                snippets.append(f"#{idx} \"{text}\"")
    room = str(decision.get("destination_room") or "")
    floor = str(decision.get("destination_floor") or "")
    room_digits = re.sub(r"\D", "", room)
    evidence_digits = " ".join(re.sub(r"\D", "", item) for item in snippets)
    correction = ""
    if room_digits and room_digits in evidence_digits and "-" in room:
        correction = f" -> normalized to {room}"
    if floor:
        correction += f" -> floor inferred as {floor}"
    return f"PaddleOCR evidence: {', '.join(snippets) or 'none'}{correction}"


def candidate_matches_decision(candidate: dict[str, Any], decision: dict[str, Any]) -> bool:
    if not decision:
        return False
    destination_matches = (
        candidate.get("destination_room") == decision.get("destination_room")
        and candidate.get("destination_floor") == decision.get("destination_floor")
        and candidate.get("destination_dong") == decision.get("destination_dong")
    )
    if not destination_matches:
        return False
    decision_evidence = decision.get("evidence_indices") or []
    candidate_evidence = candidate.get("evidence_indices") or []
    if decision_evidence:
        return sorted(decision_evidence) == sorted(candidate_evidence)
    return True


def make_detail_figure(
    ocr_result: dict[str, Any],
    llm_result: dict[str, Any],
    out_path: Path,
    eval_row: dict[str, str] | None = None,
) -> None:
    image_path = Path(str(ocr_result["image"]))
    img = Image.open(image_path).convert("RGB")
    canvas_w, canvas_h = 2100, 1180
    margin = 28
    image_area_w, image_area_h = 1320, 900
    panel_x = margin + image_area_w + 30
    panel_w = canvas_w - panel_x - margin

    scale = min(image_area_w / img.width, image_area_h / img.height)
    shown = img.resize((int(img.width * scale), int(img.height * scale)), RESAMPLE_LANCZOS)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (246, 247, 249))
    canvas.paste(shown, (margin, 102))
    draw = ImageDraw.Draw(canvas)

    label = str(llm_result.get("llm_destination_label") or "NO_DEST")
    decision = llm_result.get("llm_decision") or {}
    ocr_items = ocr_result.get("ocr_items") or []
    evidence = {int(i) for i in decision.get("evidence_indices") or [] if isinstance(i, int | float)}
    ok = eval_row is None or str(eval_row.get("destination_exact")) == "True"
    status = "PASS" if ok else "FAIL"
    title_fill = (10, 110, 55) if ok else (190, 35, 25)

    draw.text((margin, 22), f"{status} | {image_path.name} | Pred: {label}", font=font(34, True), fill=title_fill)
    draw.text(
        (margin, 67),
        f"OCR items={len(ocr_items)} OCR={ocr_result.get('ocr_seconds', 0):.2f}s "
        f"LLM={llm_result.get('llm_seconds', 0):.2f}s conf={decision.get('confidence')}",
        font=font(20),
        fill=(70, 70, 70),
    )

    image_origin = (margin, 102)
    for idx, item in enumerate(ocr_items):
        if item_rotation(item) != 0:
            continue
        box = item.get("box") or []
        if len(box) != 4:
            continue
        pts = scaled_poly(box, scale, image_origin)
        color = (255, 45, 20) if idx in evidence else (40, 150, 80)
        width = 5 if idx in evidence else 2
        draw.line(pts + [pts[0]], fill=color, width=width)
        if idx in evidence:
            x0 = int(min(p[0] for p in pts))
            y0 = int(min(p[1] for p in pts))
            draw.rectangle((x0, max(image_origin[1], y0 - 24), x0 + 54, y0), fill=color)
            draw.text((x0 + 4, max(image_origin[1], y0 - 24)), str(idx), font=font(14), fill=(255, 255, 255))

    y = 110
    if eval_row:
        draw.text((panel_x, y), "Evaluation", font=font(26, True), fill=(20, 20, 20))
        y += 38
        y = draw_wrapped(
            draw,
            (panel_x, y),
            f"GT={eval_gt_text(eval_row)} | Pred={eval_pred_text(eval_row, decision)} | "
            f"Exact={bool_text(eval_row.get('destination_exact'))} "
            f"Room={bool_text(eval_row.get('room_match'))} Floor={bool_text(eval_row.get('floor_match'))} "
            f"Dong={bool_text(eval_row.get('dong_match'))} CandidateRecall={bool_text(eval_row.get('candidate_recall'))}",
            font(20),
            (20, 20, 20),
            panel_w,
            max_lines=4,
        )
        y += 18

    draw.text((panel_x, y), "LLM Decision", font=font(26, True), fill=(20, 20, 20))
    y += 38
    y = draw_wrapped(
        draw,
        (panel_x, y),
        f"label={label} floor={decision.get('destination_floor')} room={decision.get('destination_room')} "
        f"source={decision.get('floor_source')} confidence={decision.get('confidence')} "
        f"evidence={decision.get('evidence_indices')} notes={decision.get('notes')}",
        font(20),
        (20, 20, 20),
        panel_w,
        max_lines=4,
    )
    y += 10
    y = draw_wrapped(
        draw,
        (panel_x, y),
        evidence_trace_text(ocr_items, decision),
        font(18),
        (70, 70, 70),
        panel_w,
        max_lines=3,
    )
    y += 20
    draw.text((panel_x, y), "Destination Candidates", font=font(26, True), fill=(20, 20, 20))
    y += 38
    for candidate in llm_result.get("destination_candidates") or []:
        selected = candidate_matches_decision(candidate, decision)
        prefix = "SELECTED " if selected else ""
        y = draw_wrapped(
            draw,
            (panel_x, y),
            f"{prefix}[{candidate.get('candidate_id')}] {candidate.get('destination_floor')} "
            f"{candidate.get('destination_room')} e={candidate.get('evidence_indices')} "
            f"score={candidate.get('score')} {candidate.get('address_text')}",
            font(17),
            (180, 35, 20) if selected else (35, 35, 35),
            panel_w,
            max_lines=3,
        )
        y += 6
        if y > canvas_h - 180:
            break
    y = max(y + 10, canvas_h - 170)
    draw.text((panel_x, y), "OCR Text Excerpt", font=font(26, True), fill=(20, 20, 20))
    y += 38
    draw_wrapped(draw, (panel_x, y), str(llm_result.get("full_ocr_text") or ""), font(16), (70, 70, 70), panel_w, max_lines=5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def make_contact_sheet(figure_paths: list[Path], out_path: Path) -> None:
    cells: list[Image.Image] = []
    for path in figure_paths:
        img = Image.open(path).convert("RGB")
        img.thumbnail((620, 350), RESAMPLE_LANCZOS)
        cell = Image.new("RGB", (650, 390), (250, 250, 250))
        cell.paste(img, ((650 - img.width) // 2, 12))
        ImageDraw.Draw(cell).text((14, 355), path.stem, font=font(18, True), fill=(20, 20, 20))
        cells.append(cell)
    cols = 2
    rows = math.ceil(len(cells) / cols)
    sheet = Image.new("RGB", (cols * 650, rows * 390), (235, 235, 235))
    for idx, cell in enumerate(cells):
        sheet.paste(cell, ((idx % cols) * 650, (idx // cols) * 390))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def make_figures(ocr_json: Path, llm_json: Path, out_dir: Path, eval_csv: Path | None = None) -> list[Path]:
    ocr_data = load_json(ocr_json)
    llm_data = load_json(llm_json)
    ocr_results = ocr_data.get("combined_results") or ocr_data["results"]
    ocr_by_name = by_image_name(ocr_results)
    eval_by_id = load_eval_rows(eval_csv)
    figure_paths: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for llm_result in llm_data["results"]:
        name = Path(str(llm_result["image"])).name
        out_path = out_dir / f"{Path(name).stem}_ocr_llm.jpg"
        make_detail_figure(ocr_by_name[name], llm_result, out_path, eval_by_id.get(sample_id_from_path(name)))
        figure_paths.append(out_path)
    make_contact_sheet(figure_paths, out_dir / "ocr_llm_contact_sheet.jpg")
    return figure_paths
