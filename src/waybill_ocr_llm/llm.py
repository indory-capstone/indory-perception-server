from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

from .candidates import (
    basement_floor_from_b_room,
    build_candidate_lines,
    destination_candidates,
    extract_room_text,
    has_address_context,
    h_prefixed_room_looks_like_sorting_code,
    high_floor_without_room_suffix_floor,
    max_b_ocr_confusion_floor,
    max_reasonable_floor,
    parse_candidate_dong,
    room_looks_like_address_tail_unit,
    room_looks_like_artificial_hyphen,
    room_looks_like_receiver_numeric_name,
    room_looks_like_sorting_suffix,
    room_looks_like_road_building_number,
    recover_prefixed_hyphen_room_left,
    same_variant,
)


SYSTEM_PROMPT = "Korean waybill destination extractor. Output one JSON object only."

USER_PROMPT_TEMPLATE = """택배 운송장 OCR에서 배송 목적지를 추출하세요.

반드시 JSON 한 객체만 출력하세요. 필요한 keys:
best_rotation, candidate_id, destination_room, destination_floor, destination_dong, confidence, notes

Rules:
- OCR text에 rot=0/90/180/270이 있으면 먼저 택배 운송장처럼 가장 잘 읽힌 best_rotation 하나를 고르세요.
- best_rotation은 운송장번호, 택배사, 받는분, 주소, 보내는분, 상품정보 같은 운송장 문구가 자연스럽게 이어지는 방향입니다.
- best_rotation이 정해지면 다른 rotation의 OCR, 후보, 숫자는 잡음으로 보고 무시하세요. 단, best_rotation에 목적지 후보가 전혀 없을 때만 다른 rotation을 보조로 보세요.
- 후보가 맞으면 candidate_id는 아래 후보 id 중 하나.
- 후보가 없거나 후보가 틀리면 candidate_id=null로 두고 OCR text에서 직접 destination_room/floor/dong을 복원.
- destination_room은 반드시 N호 또는 N-M호 같은 호수 형식입니다. 5F, 2F 같은 층 표기는 destination_room에 쓰지 마세요.
- destination_floor은 반드시 5F, 12F 같은 층 형식입니다. N호 같은 호수 표기는 destination_floor에 쓰지 마세요.
- 배송지/배송주소/받으시는분/받는분/수취인 근처 주소를 우선.
- 보낸사람/보내는분/배송하시는분/운임/날짜/전화번호/운송장번호/주문번호 근처 숫자는 제외.
- 도착점, 분류코드, 영문 prefix와 숫자가 섞인 sorting/tracking/order code는 호수가 아님.
- N관, 제N공학관처럼 건물명에 붙은 숫자는 건물 번호이므로 호수가 아님.
- OCR이 도로명/건물 번호와 호수를 붙여 긴 숫자로 읽으면, 앞쪽 번호를 버리고 뒤쪽 3~4자리 후보를 호수로 검토.
- 목적지는 호수가 핵심. 명시 층 숫자가 주변에 있어도 호수가 있으면 호수에서 추론한 층을 우선.
- 동은 명시된 N동만 사용하고, 다른 숫자를 동으로 추정하지 마세요.
- 정말 읽을 수 없을 때만 destination_room/floor/dong 모두 null.

Image: {image_name}
Candidates:
{candidates}

OCR text by rotation and box:
{ocr_text}
"""


def build_prompt(result: dict[str, Any]) -> str:
    image_name = Path(str(result.get("image") or "unknown")).name
    candidates = destination_candidates(result.get("ocr_items") or [])
    ocr_text = build_ocr_lines(result.get("ocr_items") or [])
    return USER_PROMPT_TEMPLATE.format(
        image_name=image_name,
        candidates=build_candidate_lines(candidates),
        ocr_text=ocr_text,
    )


def build_ocr_lines(items: list[dict[str, Any]], max_items: int = 120) -> str:
    indexed_items = list(enumerate(items))
    has_rotation = any(item.get("rotation_degrees") is not None for _, item in indexed_items)
    if not has_rotation:
        return build_flat_ocr_lines(indexed_items[:max_items], omitted=max(0, len(items) - max_items))

    lines: list[str] = []
    printed = 0
    rotations: list[Any] = []
    for _, item in indexed_items:
        rotation = item.get("rotation_degrees")
        if rotation not in rotations:
            rotations.append(rotation)

    for rotation in rotations:
        if printed >= max_items:
            break
        group = [(idx, item) for idx, item in indexed_items if item.get("rotation_degrees") == rotation]
        if not group:
            continue
        lines.append(f"rotation={rotation}:")
        budget = max_items - printed
        rendered = build_flat_ocr_lines(group[:budget], omitted=max(0, len(group) - budget))
        if rendered != "(none)":
            lines.extend(rendered.splitlines())
            printed += min(len(group), budget)

    omitted_total = max(0, len(items) - printed)
    if omitted_total:
        lines.append(f"... {omitted_total} more OCR boxes omitted")
    return "\n".join(lines) if lines else "(none)"


def build_flat_ocr_lines(indexed_items: list[tuple[int, dict[str, Any]]], omitted: int = 0) -> str:
    lines: list[str] = []
    for idx, item in indexed_items:
        text = re.sub(r"\s+", " ", str(item.get("text") or "")).strip()
        if not text:
            continue
        if len(text) > 100:
            text = text[:97] + "..."
        rotation = item.get("rotation_degrees")
        rotation_tag = f"|rot={rotation}" if rotation is not None else ""
        lines.append(f"[{idx}{rotation_tag}] {text}")
    if omitted:
        lines.append(f"... {omitted} more OCR boxes omitted in this rotation")
    return "\n".join(lines) if lines else "(none)"


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = strip_code_fence(text)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def normalize_floor(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper().replace(" ", "")
    if not text or text in {"NULL", "NONE", "UNKNOWN", "?"}:
        return None
    text = text.replace("층", "F").replace("지하", "B")
    match = re.fullmatch(r"B?(\d{1,2})F?", text)
    if match:
        return f"B{int(match.group(1))}F" if text.startswith("B") else f"{int(match.group(1))}F"
    match = re.search(r"B(\d{1,2})", text)
    if match:
        return f"B{int(match.group(1))}F"
    match = re.search(r"(\d{1,2})", text)
    if match:
        return f"{int(match.group(1))}F"
    return None


def normalize_room(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper().replace(" ", "").replace("号", "호")
    if not text or text in {"NULL", "NONE", "UNKNOWN", "?"}:
        return None
    return extract_room_text(str(value).replace("号", "호"))


def infer_floor_from_room(room: str | None) -> str | None:
    if not room:
        return None
    if room.startswith("B"):
        return None
    match = re.fullmatch(r"(\d{3,4})(?:-\d+)?호?", room)
    if not match:
        return None
    digits = match.group(1)
    floor = int(digits[0] if len(digits) == 3 else digits[:-2])
    if floor <= 0:
        return None
    return f"{floor}F"


def normalize_dong(value: Any) -> str | None:
    if value is None:
        return None
    return parse_candidate_dong(str(value))


def normalize_decision(decision: dict[str, Any] | None) -> dict[str, Any] | None:
    if decision is None:
        return None
    normalized = dict(decision)
    room = normalize_room(normalized.get("destination_room") or normalized.get("room"))
    floor = normalize_floor(normalized.get("destination_floor") or normalized.get("floor"))
    floor_source = normalized.get("floor_source")
    inferred_floor = infer_floor_from_room(room)
    if inferred_floor is not None:
        floor = inferred_floor
        floor_source = "room_inferred"
    elif floor is not None and floor_source not in {"explicit", "room_inferred"}:
        floor_source = "explicit"

    evidence = normalized.get("evidence_indices") or normalized.get("evidence") or []
    if not isinstance(evidence, list):
        evidence = []
    parsed_evidence: list[int] = []
    for item in evidence:
        if isinstance(item, int | float):
            parsed_evidence.append(int(item))
        elif isinstance(item, str) and item.strip().isdigit():
            parsed_evidence.append(int(item.strip()))

    try:
        confidence = max(0.0, min(1.0, float(normalized.get("confidence"))))
    except (TypeError, ValueError):
        confidence = None

    normalized["destination_floor"] = floor
    normalized["destination_room"] = room
    normalized["destination_dong"] = normalize_dong(normalized.get("destination_dong") or normalized.get("dong"))
    normalized["floor_source"] = floor_source if floor_source in {"explicit", "room_inferred"} else None
    normalized["confidence"] = confidence
    normalized["evidence_indices"] = parsed_evidence
    return normalized


def candidate_id_from_response(response: dict[str, Any] | None) -> int | None:
    if not isinstance(response, dict):
        return None
    candidate_id = response.get("candidate_id")
    if isinstance(candidate_id, int):
        return candidate_id
    if isinstance(candidate_id, str) and candidate_id.strip().isdigit():
        return int(candidate_id.strip())
    return None


def is_nullish_candidate_id(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "null", "none", "unknown", "?"}
    return False


def direct_decision_from_response(response: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(response, dict):
        return None
    has_direct_field = any(
        key in response
        for key in (
            "destination_room",
            "destination_floor",
            "destination_dong",
            "room",
            "floor",
            "dong",
        )
    )
    if not has_direct_field:
        return None
    decision = normalize_decision(response)
    if not decision:
        return None
    if not any(decision.get(key) for key in ("destination_room", "destination_floor", "destination_dong")):
        return None
    return decision


def decision_from_candidate_response(response: dict[str, Any] | None, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    response = response or {}
    direct_decision = direct_decision_from_response(response)
    if not candidates:
        return direct_decision
    top_candidate = max(candidates, key=lambda item: float(item.get("score") or 0.0))
    candidate_id = candidate_id_from_response(response)
    if not isinstance(candidate_id, int) or candidate_id < 0 or candidate_id >= len(candidates):
        if direct_decision and direct_decision.get("destination_room"):
            return direct_decision
        candidate = dict(top_candidate) if float(top_candidate.get("score") or 0.0) >= 2.0 else None
    else:
        selected = candidates[candidate_id]
        matched_direct = matching_candidate_for_decision(direct_decision, candidates)
        if matched_direct is not None and not same_destination(direct_decision, normalize_decision(selected)):
            candidate = dict(matched_direct)
            response = {
                **response,
                "notes": f"{response.get('notes') or ''} candidate_id_conflict_resolved_from_direct".strip(),
            }
        else:
            candidate = None
        selected_score = float(selected.get("score") or 0.0)
        top_score = float(top_candidate.get("score") or 0.0)
        if candidate is None and top_candidate is not selected and top_score - selected_score >= 4.0 and top_candidate.get("destination_room"):
            candidate = dict(top_candidate)
            response = {**response, "notes": f"{response.get('notes') or ''} score_override_from_candidate_{candidate_id}".strip()}
        elif candidate is None:
            candidate = dict(selected)
    if candidate is None:
        return direct_decision
    candidate.pop("candidate_id", None)
    candidate.pop("score", None)
    candidate["confidence"] = response.get("confidence")
    candidate["notes"] = str(response.get("notes") or "")
    return normalize_decision(candidate)


def decision_label(decision: dict[str, Any] | None) -> str:
    if not decision:
        return "NO_DEST"
    parts = [
        decision.get("destination_dong"),
        decision.get("destination_floor"),
        decision.get("destination_room"),
    ]
    label = " ".join(str(part) for part in parts if part)
    return label or "NO_DEST"


def has_confidence(decision: dict[str, Any] | None) -> bool:
    if not isinstance(decision, dict):
        return False
    try:
        float(decision.get("confidence"))
        return True
    except (TypeError, ValueError):
        return False


def same_destination(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    keys = ("destination_floor", "destination_room", "destination_dong")
    return bool(left.get("destination_room")) and all(left.get(key) == right.get(key) for key in keys)


def matching_candidate_for_decision(
    decision: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not isinstance(decision, dict) or not decision.get("destination_room"):
        return None
    for candidate in candidates:
        normalized = normalize_decision(candidate)
        if same_destination(decision, normalized):
            return dict(candidate)
    return None


def fill_missing_confidence_from_layout(
    decision: dict[str, Any] | None,
    layout_decision: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if decision is None or layout_decision is None or has_confidence(decision):
        return decision
    if not same_destination(decision, layout_decision) or not has_confidence(layout_decision):
        return decision

    merged = dict(decision)
    merged["confidence"] = layout_decision.get("confidence")
    if not merged.get("evidence_indices") and layout_decision.get("evidence_indices"):
        merged["evidence_indices"] = layout_decision.get("evidence_indices")

    layout_notes = str(layout_decision.get("notes") or "").strip()
    current_notes = str(merged.get("notes") or "").strip()
    if layout_notes and layout_notes not in current_notes:
        merged["notes"] = f"{current_notes}; {layout_notes}" if current_notes else layout_notes
    return merged


def validate_llm_response(
    response: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
    decision: dict[str, Any] | None,
) -> list[str]:
    reasons: list[str] = []
    if not isinstance(response, dict):
        return ["invalid_json_object"]

    confidence = response.get("confidence")
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = None
    if confidence_value is None or confidence_value < 0.0 or confidence_value > 1.0:
        reasons.append("missing_or_invalid_confidence")

    direct_decision = direct_decision_from_response(response)
    candidate_id = candidate_id_from_response(response)
    raw_candidate_id = response.get("candidate_id")
    if candidates:
        if candidate_id is not None:
            if candidate_id < 0 or candidate_id >= len(candidates):
                reasons.append("candidate_id_out_of_range")
            else:
                selected = normalize_decision(candidates[candidate_id])
                if (
                    direct_decision
                    and direct_decision.get("destination_room")
                    and not same_destination(direct_decision, selected)
                    and matching_candidate_for_decision(direct_decision, candidates) is None
                ):
                    reasons.append("candidate_id_destination_conflict")
        elif not is_nullish_candidate_id(raw_candidate_id):
            reasons.append("invalid_candidate_id")
        elif not (direct_decision and direct_decision.get("destination_room")):
            reasons.append("missing_candidate_id_or_destination")
    elif candidate_id is not None:
        reasons.append("candidate_id_without_candidates")

    if decision is None:
        reasons.append("no_destination_decision")
    elif not has_confidence(decision):
        reasons.append("missing_decision_confidence")

    return list(dict.fromkeys(reasons))


def build_retry_prompt(
    prompt: str,
    raw_response: str,
    reasons: list[str],
    candidates: list[dict[str, Any]],
) -> str:
    candidate_ids = ", ".join(str(idx) for idx in range(len(candidates))) or "(none)"
    return f"""{prompt}

이전 답변은 시스템 검증에 실패했습니다.
실패 이유: {", ".join(reasons)}

이전 답변:
{raw_response}

다시 답하세요. 반드시 JSON 한 객체만 출력하세요.
- candidate_id는 반드시 정수 후보 ID 중 하나({candidate_ids}) 또는 null입니다.
- confidence는 반드시 0.0 이상 1.0 이하 숫자입니다.
- candidate_id를 고른 경우 destination_room/floor/dong은 그 후보와 충돌하면 안 됩니다.
- 설명 문장, markdown, 코드블록은 출력하지 마세요.
"""


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def append_note(decision: dict[str, Any], note: str) -> dict[str, Any]:
    merged = dict(decision)
    current_notes = str(merged.get("notes") or "").strip()
    if note and note not in current_notes:
        merged["notes"] = f"{current_notes}; {note}" if current_notes else note
    return merged


def add_validation_failure_risk(risk: dict[str, Any], retry_reasons: list[str]) -> dict[str, Any]:
    if not retry_reasons:
        return risk
    reasons = list(risk.get("risk_reasons") or [])
    reasons.append("llm_validation_failed_after_retries")
    merged_reasons = list(dict.fromkeys(reasons))
    risk_score = float(risk.get("risk_score") or 0.0) + 2.0
    return {
        **risk,
        "needs_manual_review": True,
        "auto_accept": False,
        "risk_score": round(risk_score, 2),
        "risk_reasons": merged_reasons,
    }


def assess_decision_risk(
    result: dict[str, Any],
    candidates: list[dict[str, Any]],
    decision: dict[str, Any] | None,
    parsed_response: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    risk_score = 0.0

    if not decision:
        return {
            "needs_manual_review": True,
            "auto_accept": False,
            "risk_score": 10.0,
            "risk_reasons": ["no_destination_decision"],
        }

    layout_score = None
    notes = str(decision.get("notes") or "")
    match = re.search(r"layout_candidate_score=([-+]?\d+(?:\.\d+)?)", notes)
    if match:
        try:
            layout_score = float(match.group(1))
        except ValueError:
            layout_score = None
    strong_layout_evidence = layout_score is not None and layout_score >= 8.0

    confidence = decision.get("confidence")
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = None

    if confidence_value is None:
        reasons.append("missing_confidence")
        risk_score += 2.0
    elif confidence_value < 0.65:
        reasons.append("low_confidence")
        risk_score += 2.0

    evidence = decision.get("evidence_indices") or []
    if not isinstance(evidence, list):
        evidence = []
    if not evidence:
        reasons.append("no_ocr_candidate_evidence")
        risk_score += 3.0

    if not candidates:
        reasons.append("no_destination_candidates")
        risk_score += 3.0
        top_candidate_score = None
    else:
        top_candidate_score = max(float(candidate.get("score") or 0.0) for candidate in candidates)
        if (
            top_candidate_score < 2.0
            and (confidence_value is None or confidence_value < 0.85)
            and not strong_layout_evidence
        ):
            reasons.append("weak_top_candidate_score")
            risk_score += 1.5

    room = decision.get("destination_room")
    floor = decision.get("destination_floor")
    dong = decision.get("destination_dong")
    matching_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("destination_room") == room
        and candidate.get("destination_floor") == floor
        and candidate.get("destination_dong") == dong
    ]
    if room and candidates and not matching_candidates and (confidence_value is None or confidence_value < 0.85):
        reasons.append("llm_answer_not_linked_to_candidate")
        risk_score += 2.0

    room_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("destination_room") and candidate.get("destination_room") != room
    ]
    if candidates and room_candidates and top_candidate_score is not None:
        close_competitors = [
            candidate
            for candidate in room_candidates
            if top_candidate_score - float(candidate.get("score") or 0.0) <= 1.5
        ]
        if close_competitors and (confidence_value is None or confidence_value < 0.75) and not strong_layout_evidence:
            reasons.append("competing_room_candidates")
            risk_score += 1.0

    if parsed_response is not None:
        candidate_id = parsed_response.get("candidate_id")
        if candidate_id in {None, "", "null"} and not matching_candidates:
            reasons.append("direct_llm_answer_without_candidate_id")
            risk_score += 1.0

    full_text = str(result.get("full_ocr_text") or " ".join(item.get("text", "") for item in result.get("ocr_items") or []))
    if room:
        room_digits_value = room_digits(str(room))
        if room_digits_value and room_digits_value not in re.sub(r"\D", "", full_text):
            reasons.append("room_digits_not_visible_in_ocr_text")
            risk_score += 2.0

    deduped_reasons = list(dict.fromkeys(reasons))
    needs_manual_review = bool(deduped_reasons) and risk_score >= 2.0
    return {
        "needs_manual_review": needs_manual_review,
        "auto_accept": not needs_manual_review,
        "risk_score": round(risk_score, 2),
        "risk_reasons": deduped_reasons,
    }


ADDRESS_HINT_RE = re.compile(
    r"[가-힣A-Za-z0-9]+(?:특별시|광역시|특별자치시|특별자치도|도|시|군|구|읍|면|동|리)"
    r"|[가-힣A-Za-z0-9]+(?:대로|번길|로|길)\s*\d+"
    r"|(?:아파트|오피스텔|빌딩|타워|빌라|상가|센터|프라자|맨션|하우스|주택|기숙사|병원|학교|대학교|캠퍼스|연구소|공장|물류센터)"
)
RECEIVER_HINT_RE = re.compile(r"받는분|받으시는분|받는문|발는분|발는문|수취인|수령|수령생|수립생|발분주소|주소")
SENDER_HINT_RE = re.compile(r"보내는분|보내시는분|배송하시는분|발송|주문번호|주문번|운송장번호|운승정|도착점|도착정|분류코드|분류드|분드|분류|센터|허브|터미널|영업소|집하점|집하장|출고|수거기사")
CODE_RE = re.compile(r"\b[A-Z]{1,4}(?:[-*]?[A-Z0-9]){2,}\b", re.IGNORECASE)
def room_digits(room: str | None) -> str:
    if not room:
        return ""
    return re.sub(r"\D", "", room)


def layout_candidate_score(candidate: dict[str, Any], items: list[dict[str, Any]]) -> float:
    room = str(candidate.get("destination_room") or "")
    digits = room_digits(room)
    if not digits:
        return -999.0
    evidence = candidate.get("evidence_indices") or []
    if not evidence:
        evidence = []
    idxs: list[int] = []
    for item in evidence:
        if isinstance(item, int):
            idxs.append(item)
        elif isinstance(item, str) and item.isdigit():
            idxs.append(int(item))
    idx = idxs[0] if idxs else -1
    item_text = ""
    before = ""
    after = ""
    if 0 <= idx < len(items):
        item_text = str(items[idx].get("text") or "")
        before = " ".join(
            str(items[i].get("text") or "")
            for i in range(max(0, idx - 6), idx + 1)
            if same_variant(items, idx, i)
        )
        after = " ".join(
            str(items[i].get("text") or "")
            for i in range(idx, min(len(items), idx + 5))
            if same_variant(items, idx, i)
        )
    context = str(candidate.get("address_text") or "")
    joined = f"{before} {after} {context}"
    upper_joined = joined.upper()
    compact_joined = re.sub(r"\s+", "", upper_joined)
    score = float(candidate.get("score") or 0.0)
    dong = str(candidate.get("destination_dong") or "")

    b_floor = basement_floor_from_b_room(room) if room.startswith("B") else None
    b_floor_num = int(re.sub(r"\D", "", b_floor) or "0") if b_floor else 0
    if room.startswith("B") and b_floor_num <= max_b_ocr_confusion_floor() and re.search(rf"(?<!\d)[8B]{re.escape(digits)}\s*호", item_text.upper()):
        score += 16.0
    if room.startswith("B") and b_floor_num <= max_b_ocr_confusion_floor() and re.search(rf"(?<![A-Z0-9])[8B]{re.escape(digits)}(?![A-Z0-9])", item_text.upper()):
        score += 8.0
    if room.startswith("B") and re.search(rf"(?<![A-Z0-9])B{re.escape(digits)}\s*R(?![A-Z0-9])", upper_joined):
        score += 28.0
    if room.startswith("B") and b_floor_num <= max_b_ocr_confusion_floor() and re.search(rf"(?<![A-Z0-9])[8B]{re.escape(digits)}\d(?!\d)", item_text.upper()):
        score += 5.0
    if room.startswith("B") and b_floor_num > max_b_ocr_confusion_floor() and re.search(rf"(?<!\d){re.escape(digits)}\s*호", item_text):
        score -= 14.0
    if "-" not in room and re.search(rf"(?<!\d){re.escape(digits)}\s*호", item_text):
        score += 14.0
    elif re.search(rf"(?<![A-Z\d*#-]){re.escape(digits)}[.,](?!\d)", item_text):
        score += 12.0
    elif re.search(rf"(?<![A-Z\d*#-]){re.escape(digits)}[.,]\s*호", item_text):
        score += 12.0
    elif "호" in item_text and digits in re.sub(r"\D", "", item_text):
        score += 9.0
    elif digits in re.sub(r"\D", "", item_text):
        score += 2.0
    compact_item_digits = re.sub(r"\D", "", item_text)
    room_text = room.replace("호", "")
    room_match = re.fullmatch(r"(\d{3,4})-(\d+)", room_text)
    if room_match:
        left, right = room_match.groups()
        normalized_item = re.sub(r"\s+", "", item_text.upper())
        for token_match in re.finditer(r"([A-Z가-힣]*)(\d{2,4})-(\d{1,2})", normalized_item):
            prefix, token_left, token_right = token_match.groups()
            token_left = recover_prefixed_hyphen_room_left(prefix, token_left)
            if len(token_left) == 4 and token_left[0] in {"0", "1"}:
                token_left = token_left[1:]
            if token_left == left and token_right.startswith(right):
                score += 10.0
                break
    if re.search(rf"(?<![A-Z\d])H{re.escape(digits)}(?![A-Z\d])", upper_joined):
        score += 18.0
    if h_prefixed_room_looks_like_sorting_code(joined, digits):
        score -= 30.0
    if re.search(rf"(?<![A-Z0-9])([A-Z])[B8]\s*{re.escape(digits)}\s*R(?![A-Z0-9])", upper_joined):
        score += 18.0
    if room_looks_like_address_tail_unit(joined, room):
        score += 12.0
    if re.search(rf"\d{{2}}{re.escape(digits)}(?!\d)", compact_item_digits):
        score += 4.0
    if re.search(rf"{re.escape(digits)}\d(?!\d)", compact_item_digits):
        raw_with_zero = f"{digits}0"
        raw_floor = int(raw_with_zero[:-2]) if len(raw_with_zero) == 4 else 0
        strong_suffix_zero = (
            len(digits) == 3
            and raw_floor > max_reasonable_floor()
            and re.search(rf"(?<!\d){re.escape(digits)}0(?!\d)", compact_item_digits)
        )
        score += 8.0 if strong_suffix_zero else 3.0
    if re.search(rf"(?<!\d){re.escape(digits)}\s*[동봉]", joined):
        score -= 12.0
    if len(digits) == 3 and re.search(rf"0{{1,3}}[1-9]{re.escape(digits)}\s*호", joined):
        score -= 16.0
    if len(digits) == 4 and re.search(rf"0{{1,3}}{re.escape(digits)}\s*호", joined):
        score += 14.0
    if len(digits) == 4 and re.search(rf"(?:대로|번길|로|길)\s*[1-9]\d{{1,2}}{re.escape(digits[-3:])}\s*호", joined):
        score -= 16.0
    if room.startswith("B") and re.search(rf"\b[A-Z]{{1,5}}[- ]?\d{{1,3}}\s+B{re.escape(digits)}(?!\d)", upper_joined):
        score -= 18.0
    if room.startswith("B") and re.search(rf"[/.\-]\d*[8B]{re.escape(digits)}(?!\d)", upper_joined) and not re.search(rf"(?<![A-Z0-9])B{re.escape(digits)}", upper_joined):
        score -= 24.0
    if re.search(rf"[%#]\d*{re.escape(digits)}(?!\d)", joined) and not re.search(rf"(?<!\d){re.escape(digits)}\s*호", joined):
        score -= 14.0
    if re.search(rf"[/\\]\d*{re.escape(digits)}[-/]", joined) and not re.search(rf"(?<!\d){re.escape(digits)}\s*호", joined):
        score -= 12.0
    if room_looks_like_sorting_suffix(joined, room) and not has_address_context(joined) and not RECEIVER_HINT_RE.search(joined):
        score -= 20.0
    if room_looks_like_artificial_hyphen(joined, room):
        score -= 12.0
    if room_looks_like_receiver_numeric_name(joined, room):
        score -= 16.0
    if room_looks_like_road_building_number(joined, room):
        score -= 18.0
    if (
        digits.startswith("20")
        and not re.search(rf"(?<!\d){re.escape(digits)}\s*호", joined)
        and re.search(r"20[23]\d|보내는|보내는문|접수일", context)
    ):
        score -= 12.0
    if re.search(rf"010[.\-\s*\d]{{0,8}}{re.escape(digits)}", joined) or re.search(rf"{re.escape(digits)}[.\-\s*\d]{{0,8}}010", joined):
        score -= 2.0 if has_address_context(joined) or RECEIVER_HINT_RE.search(joined) else 12.0
    if (
        re.search(rf"010\D{{0,10}}\d{{0,5}}\D{{0,10}}{re.escape(digits)}(?!\d)", item_text)
        and not re.search(rf"(?<!\d){re.escape(digits)}\s*호", item_text)
        and not ADDRESS_HINT_RE.search(item_text)
    ):
        score -= 24.0
    if re.search(rf"(?<![A-Z\d*#-]){re.escape(digits)}R(?![A-Z\d])", upper_joined):
        score += 10.0
    if re.search(rf"(?:대로|번길|로|길)\s*\d{{1,3}}{re.escape(digits)}\s*호", joined):
        score += 8.0
    if dong:
        dong_base = re.escape(dong.replace("동", ""))
        if re.search(rf"(?<![A-Z0-9가-힣]){dong_base}[동통]{re.escape(digits)}(?:호)?(?!\d)", compact_joined):
            if len(digits) >= 4 and not re.search(rf"(?<![A-Z0-9가-힣]){dong_base}[동통]{re.escape(digits)}호(?!\d)", compact_joined):
                score += 3.0
            else:
                score += 10.0
        elif len(digits) == 3 and re.search(rf"(?<![A-Z0-9가-힣]){dong_base}[동통]{re.escape(digits)}\d(?!\d)", compact_joined):
            score += 5.0
    if re.search(rf"{re.escape(digits)}[-*]", item_text):
        score -= 5.0
    for seq in re.findall(r"\d{4,6}", item_text):
        prefix = seq[: -len(digits)] if len(digits) < len(seq) else ""
        prefix_value = int(prefix) if prefix.isdigit() else 0
        tail4 = seq[-4:]
        tail4_floor = int(tail4[:-2]) if len(tail4) == 4 and not tail4.startswith("0") else 0
        max_floor = max_reasonable_floor()
        tail4_is_valid_room = 1 <= tail4_floor <= max_floor
        seq_has_room_suffix = re.search(rf"(?<!\d){re.escape(seq)}\s*호", item_text) is not None
        if len(digits) == 3 and seq.endswith(digits):
            if len(seq) >= 5 and tail4.endswith(digits) and tail4_is_valid_room:
                score -= 9.0
            elif len(seq) == 4 and prefix_value in {1, 2} and seq == f"{prefix}{digits}":
                score += 3.0
            elif prefix_value >= 7 and (seq_has_room_suffix or len(seq) > 4 or tail4_floor > max_floor):
                score += 6.0
        if len(digits) == 3 and seq.startswith(digits) and len(seq) > len(digits):
            seq_floor = int(seq[:-2]) if len(seq) == 4 else 0
            if dong and re.search(rf"{re.escape(dong.replace('동', ''))}[동통]{re.escape(seq)}", compact_joined):
                score += 4.0
            elif len(seq) == 4 and seq_floor > max_reasonable_floor() and not seq_has_room_suffix:
                score += 2.0
            else:
                score -= 7.0
    if room.startswith("B") and b_floor_num <= max_b_ocr_confusion_floor() and re.search(rf"(?<!\d)[8B]{re.escape(digits)}\s*호", upper_joined):
        score += 10.0
    if room.startswith("B") and b_floor_num <= max_b_ocr_confusion_floor() and re.search(rf"(?<![A-Z0-9])[8B]{re.escape(digits)}(?![A-Z0-9])", upper_joined):
        score += 6.0
    if room.startswith("B") and b_floor_num <= max_b_ocr_confusion_floor() and re.search(rf"(?<![A-Z0-9])[8B]{re.escape(digits)}\d(?!\d)", upper_joined):
        score += 4.0
    if room.startswith("B") and b_floor_num > max_b_ocr_confusion_floor() and re.search(rf"(?<!\d){re.escape(digits)}\s*호", joined):
        score -= 10.0
    if "-" not in room and re.search(rf"(?<!\d){re.escape(digits)}\s*호", joined):
        score += 8.0
    elif re.search(rf"(?<![A-Z\d*#-]){re.escape(digits)}[.,](?!\d)", joined):
        score += 8.0
    elif re.search(rf"(?<![A-Z\d*#-]){re.escape(digits)}[.,]\s*호", joined):
        score += 8.0
    elif re.search(r"(?<!\d)\d{3,4}\s*호", joined):
        score -= 14.0
    if RECEIVER_HINT_RE.search(before) or RECEIVER_HINT_RE.search(context):
        score += 5.0
    if ADDRESS_HINT_RE.search(before) or ADDRESS_HINT_RE.search(context):
        score += 4.0
    if SENDER_HINT_RE.search(before) and not RECEIVER_HINT_RE.search(before):
        score -= 5.0
    explicit_room_match = re.search(rf"(?<!\d){re.escape(digits)}\s*호", joined)
    if explicit_room_match and not (RECEIVER_HINT_RE.search(before) or RECEIVER_HINT_RE.search(context)):
        local_room_context = joined[
            max(0, explicit_room_match.start() - 24) : min(len(joined), explicit_room_match.end() + 24)
        ]
        if re.search(r"물류\s*/0\d{1,2}|분관\s*\d|본관\s*\d", local_room_context):
            score -= 45.0
    if (
        SENDER_HINT_RE.search(context)
        and not RECEIVER_HINT_RE.search(context)
        and not re.search(rf"(?<!\d){re.escape(digits)}\s*호", joined)
    ):
        score -= 15.0
    if (
        re.search(r"010[.\-\s*\d]{4,}", joined)
        and not re.search(r"(?:로|길|대로|번길)\d{1,6}", joined)
        and not has_address_context(joined)
        and not RECEIVER_HINT_RE.search(joined)
    ):
        score -= 8.0
    if CODE_RE.search(item_text) or re.search(rf"\b[A-Z]{{1,4}}[- ]?{re.escape(digits)}\b", upper_joined):
        score -= 7.0
    if re.search(rf"\b[A-Z]\s*[-*]?\s*{re.escape(digits)}\b", upper_joined):
        score -= 9.0
    if "000-0000" in before or "0000-0000" in before:
        score -= 2.0
    floor = infer_floor_from_room(room)
    if floor is not None:
        floor_num = int(re.sub(r"\D", "", floor) or "0")
        if floor_num > max_reasonable_floor():
            score -= 6.0
        elif floor_num > high_floor_without_room_suffix_floor() and not re.search(rf"(?<!\d){re.escape(digits)}\s*호", joined):
            score -= 10.0
        else:
            score += 1.0
    if digits.endswith("00"):
        score -= 6.0
    return score


def layout_floor_only_score(candidate: dict[str, Any], items: list[dict[str, Any]]) -> float:
    floor = str(candidate.get("destination_floor") or "")
    if not floor:
        return -999.0
    score = float(candidate.get("score") or 0.0)
    evidence = candidate.get("evidence_indices") or []
    idxs: list[int] = []
    for item in evidence:
        if isinstance(item, int):
            idxs.append(item)
        elif isinstance(item, str) and item.isdigit():
            idxs.append(int(item))
    idx = idxs[0] if idxs else -1
    before = ""
    after = ""
    if 0 <= idx < len(items):
        before = " ".join(
            str(items[i].get("text") or "")
            for i in range(max(0, idx - 6), idx + 1)
            if same_variant(items, idx, i)
        )
        after = " ".join(
            str(items[i].get("text") or "")
            for i in range(idx, min(len(items), idx + 5))
            if same_variant(items, idx, i)
        )
    context = str(candidate.get("address_text") or "")
    joined = f"{before} {after} {context}"
    floor_digits = re.sub(r"\D", "", floor)

    if floor_digits and re.search(rf"(?<!\d){re.escape(floor_digits)}\s*(?:층|F|중)", joined.upper()):
        score += 9.0
    if RECEIVER_HINT_RE.search(before) or RECEIVER_HINT_RE.search(context):
        score += 4.0
    if ADDRESS_HINT_RE.search(before) or ADDRESS_HINT_RE.search(context):
        score += 4.0
    if SENDER_HINT_RE.search(before) and not RECEIVER_HINT_RE.search(before):
        score -= 4.0
    if "010" in re.sub(r"\s+", "", joined) and not ADDRESS_HINT_RE.search(joined):
        score -= 4.0
    return score


def layout_decision_from_candidates(result: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    items = result.get("ocr_items") or []
    room_scored = [
        (layout_candidate_score(candidate, items), candidate)
        for candidate in candidates
        if candidate.get("destination_room")
    ]
    floor_scored = [
        (layout_floor_only_score(candidate, items), candidate)
        for candidate in candidates
        if not candidate.get("destination_room") and candidate.get("destination_floor")
    ]
    if not room_scored and not floor_scored:
        return None
    room_scored.sort(key=lambda item: item[0], reverse=True)
    floor_scored.sort(key=lambda item: item[0], reverse=True)
    score, candidate = room_scored[0] if room_scored else floor_scored[0]
    if room_scored:
        raw_score_candidate = max(
            (item for _layout_score, item in room_scored),
            key=lambda item: float(item.get("score") or 0.0),
        )
        raw_score_layout = layout_candidate_score(raw_score_candidate, items)
        if (
            raw_score_candidate is not candidate
            and float(raw_score_candidate.get("score") or 0.0) - float(candidate.get("score") or 0.0) >= 4.0
            and raw_score_layout >= 8.0
            and raw_score_layout >= score - 5.0
        ):
            score, candidate = raw_score_layout, raw_score_candidate
    if floor_scored:
        floor_score, floor_candidate = floor_scored[0]
        room = str(candidate.get("destination_room") or "") if room_scored else ""
        room_candidate_score = float(candidate.get("score") or 0.0) if room_scored else -999.0
        if not room_scored or (
            not room.startswith("B")
            and (
                (
                    room_candidate_score < 0.0
                    and score < 0.0
                    and (
                        (floor_score >= 6.0 and floor_score >= score - 1.0)
                        or (score < -20.0 and floor_score >= score + 10.0)
                    )
                )
                or (room_candidate_score < 5.0 and score < 6.0 and floor_score >= max(8.0, score + 10.0))
            )
        ):
            score, candidate = floor_score, floor_candidate
    if candidate.get("destination_room") and candidate.get("destination_floor"):
        same_room_floor = [
            item
            for _candidate_score, item in room_scored
            if item.get("destination_room") == candidate.get("destination_room")
            and item.get("destination_floor") == candidate.get("destination_floor")
        ]
        if same_room_floor:
            candidate = max(
                same_room_floor,
                key=lambda item: (
                    1 if item.get("destination_dong") else 0,
                    float(item.get("score") or 0.0),
                ),
            )
    min_score = float(env_int("WAYBILL_LAYOUT_MIN_SCORE", -999))
    if score < min_score:
        return None
    decision = dict(candidate)
    decision.pop("candidate_id", None)
    decision.pop("score", None)
    decision["confidence"] = min(0.99, max(0.5, score / 15.0))
    decision["notes"] = f"layout_candidate_score={score:.2f}"
    return normalize_decision(decision)


class ChatJudge:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class LlamaCppJudge(ChatJudge):
    def __init__(
        self,
        model_path: Path,
        max_new_tokens: int = 128,
        n_ctx: int = 1024,
        n_gpu_layers: int = 0,
        n_threads: int = 4,
        chat_format: str = "",
    ) -> None:
        from llama_cpp import Llama

        kwargs: dict[str, Any] = {
            "model_path": str(model_path),
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "verbose": False,
        }
        if n_threads > 0:
            kwargs["n_threads"] = n_threads
        if chat_format:
            kwargs["chat_format"] = chat_format
        self.llm = Llama(**kwargs)
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=self.max_new_tokens,
            response_format={"type": "json_object"},
            stream=False,
        )
        return str(response["choices"][0]["message"]["content"]).strip()


class OpenAICompatibleJudge(ChatJudge):
    def __init__(self, endpoint: str, model: str, max_new_tokens: int = 128, timeout: float = 120.0) -> None:
        self.endpoint = endpoint
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": self.max_new_tokens,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return str(body["choices"][0]["message"]["content"]).strip()


class OllamaJudge(ChatJudge):
    def __init__(self, endpoint: str, model: str, max_new_tokens: int = 128, timeout: float = 120.0) -> None:
        self.endpoint = endpoint
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_predict": self.max_new_tokens},
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        message = body.get("message") or {}
        return str(message.get("content") or body.get("response") or "").strip()


def build_judge(
    mode: str,
    model_path: Path | None = None,
    model: str = "",
    endpoint: str = "",
    max_new_tokens: int = 128,
    n_ctx: int = 1024,
    n_gpu_layers: int = 0,
    n_threads: int = 4,
    chat_format: str = "",
) -> ChatJudge:
    if mode == "llama_cpp":
        if model_path is None:
            raise ValueError("--model-path is required for llama_cpp mode")
        return LlamaCppJudge(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            chat_format=chat_format,
        )
    if mode == "openai":
        return OpenAICompatibleJudge(endpoint=endpoint, model=model, max_new_tokens=max_new_tokens)
    if mode == "ollama":
        return OllamaJudge(endpoint=endpoint, model=model, max_new_tokens=max_new_tokens)
    raise ValueError(f"Unsupported judge mode: {mode}")


def judge_ocr_results(
    ocr_payload: dict[str, Any],
    mode: str,
    model_path: Path | None = None,
    model: str = "",
    endpoint: str = "",
    out_dir: Path | None = None,
    max_new_tokens: int = 128,
    n_ctx: int = 1024,
    n_gpu_layers: int = 0,
    n_threads: int = 4,
    chat_format: str = "",
) -> dict[str, Any]:
    judge: ChatJudge | None = None
    judge_error: str | None = None
    allow_layout_fallback = os.environ.get("WAYBILL_ALLOW_LAYOUT_FALLBACK", "0") == "1"
    layout_override = os.environ.get("WAYBILL_LAYOUT_OVERRIDE", "0") == "1"
    max_attempts = max(1, env_int("WAYBILL_LLM_MAX_ATTEMPTS", 10))
    try:
        judge = build_judge(
            mode=mode,
            model_path=model_path,
            model=model,
            endpoint=endpoint,
            max_new_tokens=max_new_tokens,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            chat_format=chat_format,
        )
    except Exception as exc:
        if not allow_layout_fallback:
            raise
        judge_error = str(exc)

    judged: list[dict[str, Any]] = []
    start = time.perf_counter()
    ocr_results = ocr_payload.get("combined_results") or ocr_payload["results"]
    for idx, result in enumerate(ocr_results, 1):
        candidates = destination_candidates(result.get("ocr_items") or [])
        prompt = build_prompt(result)
        t0 = time.perf_counter()
        raw_response = ""
        raw_responses: list[str] = []
        retry_reasons: list[str] = []
        parsed_response = None
        decision = None
        if judge is not None:
            attempt_prompt = prompt
            for attempt_idx in range(max_attempts):
                raw_response = judge.generate(attempt_prompt)
                raw_responses.append(raw_response)
                parsed_response = parse_json_object(raw_response)
                decision = decision_from_candidate_response(parsed_response, candidates)
                retry_reasons = validate_llm_response(parsed_response, candidates, decision)
                if not retry_reasons:
                    break
                if attempt_idx + 1 < max_attempts:
                    attempt_prompt = build_retry_prompt(prompt, raw_response, retry_reasons, candidates)
        elapsed = time.perf_counter() - t0
        layout_decision = layout_decision_from_candidates(result, candidates)
        used_validation_fallback = False
        if layout_decision is not None and (decision is None or layout_override or retry_reasons):
            decision = layout_decision
            if retry_reasons:
                decision = append_note(
                    decision,
                    f"llm_validation_failed_after_retries={','.join(retry_reasons)}",
                )
                used_validation_fallback = True
        else:
            decision = fill_missing_confidence_from_layout(decision, layout_decision)
        risk = assess_decision_risk(result, candidates, decision, parsed_response)
        if retry_reasons and not used_validation_fallback:
            risk = add_validation_failure_risk(risk, retry_reasons)
        if decision is not None:
            decision = {**decision, **risk}
        entry = {
            "image": result.get("image"),
            "source_image": result.get("source_image"),
            "rotations": result.get("rotations"),
            "variant_count": result.get("variant_count"),
            "ocr_item_count": result.get("ocr_item_count"),
            "destination_candidates": candidates,
            "llm_destination_label": decision_label(decision),
            "llm_decision": decision,
            "llm_raw_response": raw_response,
            "llm_raw_responses": raw_responses,
            "llm_attempts": len(raw_responses),
            "llm_retry_reasons": retry_reasons,
            "llm_error": judge_error,
            "llm_seconds": elapsed,
            "prompt": prompt,
            "full_ocr_text": " ".join(item.get("text", "") for item in result.get("ocr_items") or []),
        }
        judged.append(entry)
        print(f"[{idx}/{len(ocr_results)}] LLM {Path(str(entry['image'])).name}: {entry['llm_destination_label']} sec={elapsed:.3f}")

    payload = {
        "mode": mode,
        "model": model or None,
        "model_path": str(model_path) if model_path else None,
        "endpoint": endpoint or None,
        "source_ocr_model": ocr_payload.get("model"),
        "image_count": len(judged),
        "source_image_count": ocr_payload.get("source_image_count"),
        "ocr_image_count": ocr_payload.get("image_count"),
        "rotations": ocr_payload.get("rotations"),
        "total_seconds": time.perf_counter() - start,
        "results": judged,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "waybill_llm_judgements.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return payload
