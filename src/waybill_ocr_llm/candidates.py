from __future__ import annotations

import re
import os
from dataclasses import asdict
from typing import Any

from .schema import DestinationCandidate


ADDRESS_KEYWORDS = (
    "배송지",
    "배송주소",
    "주소",
    "받으시는분",
    "받는분",
    "수취인",
    "받는고객",
    "배송메모",
)

ADDRESS_CONTEXT_PATTERNS = (
    r"[가-힣A-Z0-9]{2,}(?:특별시|광역시|특별자치시|특별자치도|도|시|군|구)(?![가-힣A-Z0-9])",
    r"[가-힣A-Z0-9]{2,}(?:특별시|광역시|특별자치시|특별자치도|도)[가-힣A-Z0-9]{1,}(?:시|군|구)",
    r"[가-힣A-Z0-9]{1,}(?:시|군|구)[가-힣A-Z0-9]{1,}(?:대로|번길|로|길)",
    r"[가-힣A-Z0-9]{1,}(?:읍|면|동|리)(?![가-힣A-Z0-9])",
    r"[가-힣A-Z0-9]+(?:대로|번길|로|길)\s*\d{1,5}(?:-\d{1,5})?",
    r"(?:아파트|오피스텔|빌딩|타워|빌라|상가|센터|프라자|맨션|하우스|주택|기숙사|병원|학교|대학교|캠퍼스|연구소|공장|물류센터)",
)

RECEIVER_KEYWORDS = (
    "받으시는분",
    "받는분",
    "받는문",
    "발는분",
    "발는문",
    "수취인",
    "수령인",
    "수령생",
    "수립생",
    "받는고객",
)

NON_DESTINATION_KEYWORDS = (
    "ORD",
    "TRK",
    "주문",
    "주문번",
    "운송장",
    "운승정",
    "접수",
    "DATE",
    "운임",
    "상품정보",
    "분류코드",
    "분류",
    "도착점",
    "업체코드",
    "출고",
    "집하",
    "집하점",
    "집하장",
    "수거기사",
)

CODE_LIKE_RE = re.compile(r"\b[A-Z]{1,4}(?:[-*]?[A-Z0-9]){2,}\b", re.IGNORECASE)
NOISY_DONG_ROOM_RE = re.compile(
    r"(?<![A-Z0-9가-힣])([A-Z])(?:[RHS8B])?[.\-:·]\s*(B?\d{3,4})(?:\s*[호量])?(?![A-Z0-9])",
    re.IGNORECASE,
)


def max_reasonable_floor() -> int:
    try:
        value = int(os.environ.get("WAYBILL_MAX_REASONABLE_FLOOR", "40"))
    except ValueError:
        return 40
    return max(1, min(200, value))


def max_b_ocr_confusion_floor() -> int:
    try:
        value = int(os.environ.get("WAYBILL_MAX_B_OCR_CONFUSION_FLOOR", "2"))
    except ValueError:
        return 2
    return max(0, min(40, value))


def high_floor_without_room_suffix_floor() -> int:
    try:
        value = int(os.environ.get("WAYBILL_HIGH_FLOOR_WITHOUT_ROOM_SUFFIX_FLOOR", "20"))
    except ValueError:
        return 20
    return max(1, min(80, value))


def parse_candidate_floor(text: str) -> tuple[str | None, str | None]:
    raw = re.sub(r"\s+", " ", text.upper())
    match = re.search(r"지하\s*(\d{1,2})\s*[층중]", raw)
    if match:
        return f"B{int(match.group(1))}F", "explicit"
    match = re.search(r"지하\s*(\d{1,2})(?=\s*(?:B?\d{3,4}\s*호?|[층중]|F|$))", raw)
    if match:
        return f"B{int(match.group(1))}F", "explicit"
    match = re.search(r"B\s*(\d{1,2})\s*[층중]", raw)
    if match:
        return f"B{int(match.group(1))}F", "explicit"
    match = re.search(r"(?<!\d)(\d{1,2})\s*[층중]", raw)
    if match:
        return f"{int(match.group(1))}F", "explicit"
    match = re.search(r"\([^)]+[동리읍면구]\)\s*(\d{1,2})(?!\d)", raw)
    if match:
        return f"{int(match.group(1))}F", "explicit"
    return None, None


def parse_candidate_dong(text: str) -> str | None:
    dongs = parse_candidate_dongs(text)
    return dongs[0] if dongs else None


def parse_candidate_dongs(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.upper())
    dongs: list[str] = []

    def add(value: str | None) -> None:
        if value and value not in dongs:
            dongs.append(value)

    for match in re.finditer(r"(?<![A-Z0-9가-힣])([A-Z])\s*[동통]", normalized):
        add(f"{match.group(1)}동")
    for match in re.finditer(r"(?<![A-Z0-9가-힣])(?:4|14)\s*[동통](?=\s*\d{3,4}(?:\s*호)?(?!\d))", normalized):
        add("A동")
    for match in re.finditer(r"(?<![A-Z0-9가-힣])([A-Z])\s*8(?=\s*\d{3,4}(?:\s*호)?(?!\d))", normalized):
        add(f"{match.group(1)}동")
    for match in re.finditer(r"(?<![A-Z0-9가-힣])([A-Z])[B8]\s*\d{3,4}\s*R?(?![A-Z0-9])", normalized):
        add(f"{match.group(1)}동")
    for match in NOISY_DONG_ROOM_RE.finditer(normalized):
        add(f"{match.group(1)}동")

    for match in re.finditer(r"(?<!\d)(\d{1,5})\s*동", normalized):
        raw_digits = match.group(1)
        if len(raw_digits) == 2 and raw_digits.startswith("0"):
            add(f"1{raw_digits}동")
        raw_value = f"{int(raw_digits)}동"
        if should_prefer_tail_dong(normalized, match.start(1), raw_digits):
            add(f"{int(raw_digits[-3:])}동")
        add(raw_value)
    return dongs


def should_prefer_tail_dong(context: str, start: int, digits: str) -> bool:
    if len(digits) != 4 or digits[-3:].startswith("0"):
        return False
    prefix = context[max(0, start - 10) : start]
    return re.search(r"(?:대로|번길|로|길)\s*$", prefix) is not None


def normalize_room_candidate(value: str) -> str | None:
    digits = value[1:] if value.startswith("B") else value
    if digits.startswith("0"):
        return None
    if not value.startswith("B") and len(digits) == 4:
        inferred_floor = int(digits[:-2])
        if inferred_floor > max_reasonable_floor():
            return f"{digits[-3:]}호"
    return f"{value}호"


def room_allowed_by_policy(room: str | None) -> bool:
    if room is None:
        return False
    policy = os.environ.get("WAYBILL_ROOM_POLICY", "any").strip().lower()
    if policy in {"", "any", "all", "off", "none"}:
        return True
    if policy in {"5xx", "5xx-x", "5xx_or_5xx-x"}:
        return re.fullmatch(r"5\d{2}(?:-\d{1,2})?호?", room) is not None
    return True


def recover_prefixed_hyphen_room_left(prefix: str, left: str) -> str:
    floor_digit = os.environ.get("WAYBILL_ROOM_PREFIX_FLOOR_DIGIT", "").strip()
    if len(left) == 2 and prefix and re.fullmatch(r"\d", floor_digit):
        return floor_digit + left
    return left


def room_candidates_from_digits(value: str) -> list[str]:
    normalized = value.upper()
    prefix = "B" if normalized.startswith("B") else ""
    digits = normalized[1:] if prefix else normalized
    candidates: list[str] = []
    if not prefix and len(digits) >= 5 and re.match(r"01[016789]", digits):
        return candidates
    raw_floor = int(digits[:-2]) if not prefix and len(digits) == 4 else 0
    keep_raw = (
        prefix
        or len(digits) <= 3
        or (
            len(digits) == 4
            and 1 <= raw_floor <= max_reasonable_floor()
            and digits[-1] != digits[-2]
        )
    )
    raw = normalize_room_candidate(normalized) if keep_raw else None
    if raw is not None:
        candidates.append(raw)
    if prefix:
        return candidates

    def add(candidate_digits: str) -> None:
        candidate = normalize_room_candidate(candidate_digits)
        if candidate is not None and candidate not in candidates:
            candidates.append(candidate)

    def add_embedded_four_digits(candidate_digits: str) -> None:
        if len(candidate_digits) != 4 or candidate_digits.startswith("0"):
            return
        floor = int(candidate_digits[:-2])
        if candidate_digits.startswith("8") and candidate_digits[1] != "0" and int(candidate_digits[1]) <= max_b_ocr_confusion_floor():
            add(f"B{candidate_digits[-3:]}")
        if 1 <= floor <= max_reasonable_floor():
            add(candidate_digits)
        elif not candidate_digits.endswith("0"):
            add(candidate_digits[-3:])

    if len(digits) >= 5:
        head4 = digits[:4]
        add_embedded_four_digits(head4)
        tail4 = digits[-4:]
        add_embedded_four_digits(tail4)
        add(digits[-3:])
    elif len(digits) == 4:
        floor = int(digits[:-2])
        if digits.startswith("8") and digits[1] != "0" and int(digits[1]) <= max_b_ocr_confusion_floor():
            add(f"B{digits[-3:]}")
        if floor > max_reasonable_floor() and not digits.endswith("0"):
            add(f"{digits[:3]}-{digits[3:]}")
        if floor > max_reasonable_floor():
            add(digits[:3])
            add(digits[-3:])
        if digits.endswith("0") and digits[:3] != "000":
            add(digits[:3])
        if digits[-1] == digits[-2]:
            add(digits[:3])
        if digits[0] == digits[1] and not digits.endswith("00"):
            add(digits[-3:])
    return candidates


def room_candidates_from_hyphen_token(value: str) -> list[str]:
    token = re.sub(r"\s+", "", value.upper())
    match = re.search(r"([A-Z가-힣]*)(\d{2,4})-(\d{1,2})", token)
    if not match:
        return []
    prefix, left, right = match.groups()
    left = recover_prefixed_hyphen_room_left(prefix, left)
    if len(left) == 4 and left[0] in {"0", "1"}:
        left = left[1:]
    if len(left) not in {3, 4} or left.startswith("0") or not right:
        return []
    room = normalize_room_candidate(f"{left}-{right[:1]}")
    return [room] if room else []


def extract_room_text(text: str) -> str | None:
    values = extract_room_texts(text)
    return values[0] if values else None


def extract_room_texts(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.upper().replace("号", "호")).strip()
    compact = re.sub(r"\s+", "", normalized)
    if any(token in normalized for token in ("ORD", "TRK", "주문", "운송장", "접수", "DATE")):
        return []
    if re.search(r"01[016789]-?\d{3,4}-?\d{4}", compact):
        return []
    if re.search(r"\d{3,4}-\d{3,4}-\d{3,4}", compact):
        return []
    if "*" in compact and "호" not in compact:
        return []

    rooms: list[str] = []

    def add(value: str | None) -> None:
        if value is not None and value not in rooms:
            rooms.append(value)

    for match in re.finditer(r"(?<![A-Z\d*#-])(B?\d{3,6}(?:-\d{1,2})?)\s*호(?![A-Z\d])", normalized):
        if span_has_date_or_code_context(normalized, match.start(1), match.end(1)):
            continue
        raw = match.group(1)
        if "-" in raw:
            add(normalize_room_candidate(raw))
            continue
        for candidate in room_candidates_from_digits(raw):
            add(candidate)

    for match in re.finditer(r"(?<![A-Z\d*#-])([A-Z가-힣]?\d{2,4}-\d{1,2})(?![A-Z\d동관-])", normalized):
        for candidate in room_candidates_from_hyphen_token(match.group(1)):
            add(candidate)

    for match in re.finditer(r"(?<![A-Z\d*#-])(\d{3,4})[.,]0(?!\d)", normalized):
        for candidate in room_candidates_from_digits(match.group(1)):
            add(candidate)

    for match in re.finditer(r"(?<![A-Z\d*#-])(\d{3,4})[.,]\s*호(?![A-Z\d])", normalized):
        for candidate in room_candidates_from_digits(match.group(1)):
            add(candidate)

    for match in re.finditer(r"(?<![A-Z\d*#-])(\d{3,4})[.,](?!\d)", normalized):
        for candidate in room_candidates_from_digits(match.group(1)):
            add(candidate)

    for match in re.finditer(r"(?<![A-Z\d*#-])(\d{3})[05][.,](?!\d)", normalized):
        for candidate in room_candidates_from_digits(match.group(1)):
            add(candidate)

    for match in re.finditer(r"(?<![A-Z\d*#-])\d[.,](\d{3,4})(?![A-Z\d])", normalized):
        for candidate in room_candidates_from_digits(match.group(1)):
            add(candidate)

    for match in re.finditer(r"동\s*(B?\d{3,4})(?!\s*[A-Z\d동관-])", normalized):
        for candidate in room_candidates_from_digits(match.group(1)):
            add(candidate)

    for match in re.finditer(r"(?<![A-Z\d*#-])(B\d{3,4})R(?![A-Z\d])", normalized):
        for candidate in room_candidates_from_digits(match.group(1)):
            add(candidate)

    for match in re.finditer(r"(?<![A-Z\d*#-])[A-Z][B8]\s*(\d{3,4})\s*R(?![A-Z\d])", normalized):
        if span_has_date_or_code_context(normalized, match.start(1), match.end(1)):
            continue
        for candidate in room_candidates_from_digits(match.group(1)):
            add(candidate)

    for match in NOISY_DONG_ROOM_RE.finditer(normalized):
        for candidate in room_candidates_from_digits(match.group(2)):
            add(candidate)

    for match in re.finditer(r"(?<![A-Z\d*#-])(\d{4})0(?!\d)", normalized):
        if span_has_date_or_code_context(normalized, match.start(1), match.end(1)):
            continue
        raw_digits = match.group(1)
        if raw_digits.startswith("20"):
            continue
        floor = int(raw_digits[:-2])
        if 1 <= floor <= max_reasonable_floor():
            add(normalize_room_candidate(raw_digits))

    for match in re.finditer(r"(?<![A-Z\d*#-])(B?\d{3,4})(?!\s*[A-Z\d동관-])", normalized):
        if span_has_date_or_code_context(normalized, match.start(1), match.end(1)):
            continue
        if span_looks_like_road_building_number(normalized, match.start(1), match.end(1)):
            continue
        for candidate in room_candidates_from_digits(match.group(1)):
            add(candidate)
    return rooms


def span_looks_like_road_building_number(text: str, start: int, end: int) -> bool:
    left = text[max(0, start - 16) : start]
    right = text[end : min(len(text), end + 16)]
    if right.lstrip().startswith("호"):
        return False
    return re.search(r"(?:대로|번길|로|길)\s*$", left) is not None


def span_has_date_or_code_context(text: str, start: int, end: int) -> bool:
    left = text[max(0, start - 8) : start]
    right = text[end : min(len(text), end + 8)]
    window = text[max(0, start - 8) : min(len(text), end + 8)]
    if left[-1:] in {".", "/", "-"} or right[:1] in {".", "/", "-"}:
        return True
    if re.search(r"\d{2,4}[./-]\d{1,2}(?:[./-]\d{1,2})?", window):
        return True
    if re.search(r"(?:ORD|TRK|CJT?|CLS|KP|HJ)[-* ]*$", left, re.IGNORECASE):
        return True
    if re.search(r"\b[A-Z]{1,3}\d?-\d{1,4}\s*$", left, re.IGNORECASE):
        return True
    if re.search(r"^\s+[A-Z]{1,3}\d?-\d{1,4}\b", right, re.IGNORECASE):
        return True
    return False


def context_room_texts(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.upper().replace("号", "호")).strip()
    compact = re.sub(r"\s+", "", normalized)
    rooms = extract_room_texts(text)

    def add(value: str | None) -> None:
        if value is not None and value not in rooms:
            rooms.append(value)

    if has_address_context(normalized):
        for match in re.finditer(r"(?<![\d*])(\d{5,6})(?![\d*])", normalized):
            raw = match.group(1)
            if re.match(r"01[016789]", raw):
                continue
            if raw.startswith("20"):
                continue
            if re.search(r"20[23]\d", raw):
                continue
            if span_has_date_or_code_context(normalized, match.start(1), match.end(1)):
                continue
            for candidate in room_candidates_from_digits(raw):
                add(candidate)
        for match in re.finditer(r"(?<![A-Z\d*#-])(\d{3,4})R(?![A-Z\d])", normalized):
            for candidate in room_candidates_from_digits(match.group(1)):
                add(candidate)
        for match in re.finditer(r"(?<![A-Z\d*#-])H(\d{3})(?![A-Z\d])", normalized):
            if span_has_date_or_code_context(normalized, match.start(0), match.end(1)):
                continue
            if h_prefixed_room_looks_like_sorting_code(normalized, match.group(1)):
                continue
            for candidate in room_candidates_from_digits(match.group(1)):
                add(candidate)
        for match in re.finditer(r"(\d{5,6})\s*호", compact):
            for candidate in room_candidates_from_digits(match.group(1)):
                add(candidate)
        for match in re.finditer(r"(?:로|길|번길|대로)(\d{4,6})\s*호", compact):
            for candidate in room_candidates_from_digits(match.group(1)):
                add(candidate)
        for match in re.finditer(r"(?:[A-Z]동|\d{1,4}동)(\d{4})(?!\d)", compact):
            digits = match.group(1)
            if not digits.startswith("0"):
                add(normalize_room_candidate(digits[:3]))
    if not rooms and has_address_context(normalized) and not has_non_destination_context(normalized):
        for match in re.finditer(r"(?<![\d*])(\d{5})(?![\d*])", compact):
            raw = match.group(1)
            if re.match(r"01[016789]", raw) or raw.startswith("20") or re.search(r"20[23]\d", raw):
                continue
            for candidate in room_candidates_from_digits(raw):
                add(candidate)
    return rooms


def room_floor(room: str) -> str | None:
    if room.startswith("B"):
        return None
    match = re.fullmatch(r"B?(\d{3,4})(?:-\d+)?호?", room)
    if not match:
        return None
    digits = match.group(1)
    floor = int(digits[0] if len(digits) == 3 else digits[:-2])
    if floor <= 0:
        return None
    return f"{floor}F"


def basement_floor_from_b_room(room: str) -> str | None:
    match = re.fullmatch(r"B(\d{3,4})(?:-\d+)?호?", room)
    if not match:
        return None
    digits = match.group(1)
    floor = int(digits[0] if len(digits) == 3 else digits[:-2])
    if floor <= 0:
        return None
    return f"B{floor}F"


def apply_explicit_basement_to_room(room: str, floor: str | None) -> str:
    if not floor or not floor.startswith("B") or room.startswith("B"):
        return room
    room_floor_value = room_floor(room)
    if room_floor_value and room_floor_value.lstrip("B") == floor.lstrip("B"):
        return f"B{room}"
    return room


def text_matches_hyphen_room(text: str, room: str) -> bool:
    match = re.fullmatch(r"(\d{3,4})-(\d+)\s*호?", room)
    if not match:
        return False
    room_left, room_right = match.groups()
    normalized = re.sub(r"\s+", "", text.upper())
    for token_match in re.finditer(r"([A-Z가-힣]*)(\d{2,4})-(\d{1,2})", normalized):
        prefix, left, right = token_match.groups()
        left = recover_prefixed_hyphen_room_left(prefix, left)
        if len(left) == 4 and left[0] in {"0", "1"}:
            left = left[1:]
        if left == room_left and right.startswith(room_right):
            return True
    return False


def item_variant_key(item: dict[str, Any]) -> tuple[Any, ...] | None:
    variant_image = item.get("variant_image")
    if variant_image:
        return (variant_image, item.get("ocr_model"))
    return None


def same_variant(items: list[dict[str, Any]], center_idx: int, item_idx: int) -> bool:
    center_key = item_variant_key(items[center_idx])
    if center_key is None:
        return True
    return item_variant_key(items[item_idx]) == center_key


def text_window(items: list[dict[str, Any]], left: int, right: int, center_idx: int | None = None) -> str:
    return " ".join(
        str(items[i].get("text") or "").strip()
        for i in range(left, right)
        if center_idx is None or same_variant(items, center_idx, i)
        if str(items[i].get("text") or "").strip()
    )


def dong_looks_like_road_number_prefix(context: str, dong: str | None, room_digits_value: str = "") -> bool:
    if not dong:
        return False
    dong_digits = re.sub(r"\D", "", dong)
    if len(dong_digits) != 3:
        return False
    compact = re.sub(r"\s+", "", context.upper())
    room_part = re.escape(room_digits_value) if room_digits_value else r"\d{3,4}"
    return re.search(rf"(?:대로|번길|로|길)\d{{1,3}}{re.escape(dong_digits)}[동통]{room_part}(?:호)?", compact) is not None


def dong_looks_like_glued_road_number(context: str, dong: str | None) -> bool:
    if not dong:
        return False
    dong_digits = re.sub(r"\D", "", dong)
    if len(dong_digits) != 4 or dong_digits[-3:].startswith("0"):
        return False
    compact = re.sub(r"\s+", "", context.upper())
    return re.search(rf"(?:대로|번길|로|길)\d*{re.escape(dong_digits)}[동통]", compact) is not None


def room_looks_like_road_building_number(context: str, room: str | None) -> bool:
    if not room:
        return False
    digits = re.sub(r"\D", "", room)
    if not digits:
        return False
    compact = re.sub(r"\s+", "", context.upper())
    if re.search(rf"(?:대로|번길|로|길){re.escape(digits)}(?:\([가-힣A-Z0-9]{{1,12}}[동리읍면구]\))?", compact):
        return True
    if re.search(rf"(?<!\d){re.escape(digits)}\([가-힣A-Z0-9]{{1,12}}[동리읍면구]\)", compact):
        return True
    return False


def room_looks_like_sorting_suffix(context: str, room: str | None) -> bool:
    if not room:
        return False
    digits = re.sub(r"\D", "", room)
    if not digits:
        return False
    upper_context = re.sub(r"\s+", " ", context.upper())
    return any(
        re.search(pattern, upper_context) is not None
        for pattern in (
            rf"\b[A-Z]\d{{1,2}}-\d{{2,4}}\s+\d*{re.escape(digits)}(?!\d)",
            rf"\b[A-Z]\d{{1,2}}-\d{{1,4}}\s+[A-Z]{{1,4}}\d{{0,4}}(?:\s+\S{{1,8}}){{0,2}}\s+\d*{re.escape(digits)}(?!\d)",
            rf"\b[A-Z]\d{{1,2}}-\d{{1,4}}\s+[A-Z]{{1,4}}\d{{0,4}}(?:\s+\S{{1,8}}){{0,2}}\s+{re.escape(digits)}\d*(?!\d)",
        )
    )


def room_looks_like_artificial_hyphen(context: str, room: str | None) -> bool:
    return bool(room and "-" in room and not text_matches_hyphen_room(context, room))


def room_looks_like_receiver_numeric_name(context: str, room: str | None) -> bool:
    if not room:
        return False
    digits = re.sub(r"\D", "", room)
    if not digits:
        return False
    if re.search(rf"(?<!\d){re.escape(digits)}\s*호", context):
        return False
    return (
        re.search(
            rf"(?:받는분|받으시는분|수취인|수령인)\s*{re.escape(digits)}\s+[가-힣A-Z0-9]{{2,}}(?:특별시|광역시|도|시|군|구|로|길)",
            context,
            re.IGNORECASE,
        )
        is not None
    )


def h_prefixed_room_looks_like_sorting_code(context: str, digits: str) -> bool:
    if not digits:
        return False
    upper_context = context.upper()
    escaped = re.escape(digits)
    matches = list(re.finditer(rf"(?<![A-Z\d])H{escaped}(?![A-Z\d])", upper_context))
    if not matches:
        return False
    saw_sorting_context = False
    for match in matches:
        local = upper_context[max(0, match.start() - 18) : min(len(upper_context), match.end() + 8)]
        compact = re.sub(r"\s+", "", local)
        looks_sorting = any(
            re.search(pattern, compact) is not None
            for pattern in (
                rf"(?:도착|도작)[점정법]\S{{0,8}}H{escaped}(?!\d)",
                rf"H{escaped}(?:HJ|HU|KP|CJ|CLS|LOT|LT|ORD|TRK)[-*]?\d",
                rf"(?:주문|운송장|접수|분류|도착|도작)\S{{0,16}}H{escaped}(?!\d)",
            )
        )
        if not looks_sorting:
            return False
        saw_sorting_context = True
    return saw_sorting_context


def room_looks_like_address_tail_unit(context: str, room: str | None) -> bool:
    if not room:
        return False
    if room.startswith("B"):
        return False
    digits = re.sub(r"\D", "", room)
    if not digits:
        return False
    if digits.startswith("20"):
        return False
    compact = re.sub(r"\s+", "", context.upper())
    for match in re.finditer(
        rf"(?:대로|번길|로|길)(\d{{1,5}}){re.escape(digits)}(?!\d)(?:[가-힣A-Z]{{2,}}|/0\d{{1,3}}|\(|$)",
        compact,
    ):
        prefix_digits = match.group(1)
        if len(digits) == 3 and prefix_digits:
            joined_four = f"{prefix_digits[-1]}{digits}"
            joined_floor = int(joined_four[:-2])
            if 1 <= joined_floor <= max_reasonable_floor():
                continue
            if joined_four.endswith("0"):
                continue
        return True
    return False




def candidate_score(context: str, room: str | None, floor_source: str | None, dong: str | None = None) -> float:
    score = 0.0
    has_receiver = any(k in context for k in RECEIVER_KEYWORDS)
    score += 5.0 if "받는분 주소" in context or "수취인 주소" in context else 0.0
    score += 4.0 if "배송지" in context or "배송주소" in context else 0.0
    score += 3.0 if has_receiver else 0.0
    score += 1.5 if "주소" in context and room is not None else 0.0
    score += 1.2 if has_address_structure(context) else 0.0
    score -= 10.0 if room is not None and has_non_destination_context(context) and not has_address_context(context) and not has_receiver else 0.0
    score -= 6.0 if any(k in context for k in ("배송하시는분", "보내시는분", "보내는분", "발송점", "업체코드", "상품정보", "출고", "집하점", "집하장", "수거기사")) and not has_receiver else 0.0
    score -= 3.0 if any(k in context for k in ("분류코드", "도착점", "도착정", "분류드", "분드")) and not has_receiver else 0.0
    score -= 2.0 if CODE_LIKE_RE.search(context.upper()) and not has_receiver else 0.0
    score -= 1.0 if any(k in context for k in ("운임", "운송장NO", "운송장번호")) and not has_receiver else 0.0
    score -= 1.0 if re.search(r"\d{2,3}-\d{3,4}-\d{4}", context) and not has_receiver and not has_address_context(context) else 0.0
    score += 0.5 if floor_source == "explicit" and room is not None else 0.0
    score += 0.5 if room and room.startswith("B") else 0.0
    score += 0.4 if room and room.startswith("B") and re.fullmatch(r"B\d{1,2}F", str(parse_candidate_floor(context)[0] or "")) else 0.0
    score += 1.0 if dong and re.search(rf"(?<![A-Z0-9가-힣]){re.escape(dong.replace('동', ''))}\s*[동통]", context.upper()) else 0.0
    if room:
        digits = re.sub(r"\D", "", room)
        upper_context = context.upper()
        compact_context = re.sub(r"\s+", "", context.upper())
        if dong:
            dong_base = re.escape(dong.replace("동", ""))
            if re.search(rf"(?<![A-Z0-9가-힣]){dong_base}[동통]{re.escape(digits)}(?:호)?(?!\d)", compact_context):
                score += 9.0
            elif len(digits) == 3 and re.search(rf"(?<![A-Z0-9가-힣]){dong_base}[동통]{re.escape(digits)}\d(?!\d)", compact_context):
                score += 3.0
            if dong_looks_like_road_number_prefix(context, dong, digits):
                score += 8.0
            if dong_looks_like_glued_road_number(context, dong):
                score -= 8.0
            if NOISY_DONG_ROOM_RE.search(upper_context):
                score += 7.0
        if text_matches_hyphen_room(context, room):
            score += 6.0
        if re.search(rf"(?<![A-Z\d])H{re.escape(digits)}(?![A-Z\d])", upper_context):
            score += 16.0
        if h_prefixed_room_looks_like_sorting_code(context, digits):
            score -= 26.0
        if re.search(rf"(?<![A-Z0-9])([A-Z])[B8]\s*{re.escape(digits)}\s*R(?![A-Z0-9])", upper_context):
            score += 16.0
        if room_looks_like_address_tail_unit(context, room):
            score += 10.0
        if len(digits) == 3 and re.search(rf"0{{1,3}}[1-9]{re.escape(digits)}\s*호", context):
            score -= 14.0
        if len(digits) == 4 and re.search(rf"0{{1,3}}{re.escape(digits)}\s*호", context):
            score += 12.0
        if len(digits) == 4 and re.search(rf"(?:대로|번길|로|길)\s*[1-9]\d{{1,2}}{re.escape(digits[-3:])}\s*호", context):
            score -= 14.0
        if room.startswith("B") and re.search(rf"\b[A-Z]{{1,5}}[- ]?\d{{1,3}}\s+B{re.escape(digits)}(?!\d)", upper_context):
            score -= 16.0
        if room.startswith("B") and re.search(rf"[/.\-]\d*[8B]{re.escape(digits)}(?!\d)", upper_context) and not re.search(rf"(?<![A-Z0-9])B{re.escape(digits)}", upper_context):
            score -= 22.0
        if re.search(rf"[%#]\d*{re.escape(digits)}(?!\d)", context) and not re.search(rf"(?<!\d){re.escape(digits)}\s*호", context):
            score -= 12.0
        if re.search(rf"[/\\]\d*{re.escape(digits)}[-/]", context) and not re.search(rf"(?<!\d){re.escape(digits)}\s*호", context):
            score -= 10.0
        if room_looks_like_sorting_suffix(context, room) and not has_address_context(context) and not has_receiver:
            score -= 18.0
        if room_looks_like_artificial_hyphen(context, room):
            score -= 10.0
        if room_looks_like_receiver_numeric_name(context, room):
            score -= 14.0
        if digits.endswith("00"):
            score -= 5.0
        if digits.startswith("20") and not re.search(rf"(?<!\d){re.escape(digits)}\s*호", context):
            score -= 8.0
        if re.search(rf"(?<!\d){re.escape(digits)}\s*[동봉]", context):
            score -= 10.0
        if room_looks_like_road_building_number(context, room):
            score -= 16.0
        if re.search(rf"010[.\-\s*\d]{{0,8}}{re.escape(digits)}", context) or re.search(rf"{re.escape(digits)}[.\-\s*\d]{{0,8}}010", context):
            score -= 2.0 if has_address_context(context) or has_receiver else 10.0
        inferred_floor = room_floor(room)
        if inferred_floor is not None:
            floor_num = int(re.sub(r"\D", "", inferred_floor) or "0")
            if floor_num > max_reasonable_floor():
                score -= 6.0
            elif floor_num > high_floor_without_room_suffix_floor() and not re.search(rf"(?<!\d){re.escape(digits)}\s*호", context):
                score -= 10.0
        for seq in re.findall(r"\d{4,6}", context):
            prefix = seq[: -len(digits)] if len(digits) < len(seq) else ""
            prefix_value = int(prefix) if prefix.isdigit() else 0
            if len(digits) == 3 and seq.endswith(digits) and prefix_value >= 7:
                score += 4.0
            if len(digits) == 3 and seq.startswith(digits) and len(seq) > len(digits):
                seq_floor = int(seq[:-2]) if len(seq) == 4 else 0
                if dong and re.search(rf"(?<![A-Z0-9가-힣]){re.escape(dong.replace('동', ''))}[동통]{re.escape(seq)}", compact_context):
                    score += 4.0
                elif seq_floor > max_reasonable_floor() and has_address_context(context):
                    score += 1.5
                else:
                    score -= 4.0
        if re.search(rf"(?:대로|번길|로|길)\s*\d{{1,3}}{re.escape(digits)}\s*호", context):
            score += 8.0
        b_floor = basement_floor_from_b_room(room) if room.startswith("B") else None
        b_floor_num = int(re.sub(r"\D", "", b_floor) or "0") if b_floor else 0
        if room.startswith("B") and b_floor_num <= max_b_ocr_confusion_floor() and re.search(rf"(?<!\d)[8B]{re.escape(digits)}\s*호", context.upper()):
            score += 10.0
        if room.startswith("B") and b_floor_num <= max_b_ocr_confusion_floor() and re.search(rf"(?<![A-Z0-9])[8B]{re.escape(digits)}(?![A-Z0-9])", context.upper()):
            score += 10.0 if has_receiver or has_address_structure(context) else 6.0
        if room.startswith("B") and re.search(rf"(?<![A-Z0-9])B{re.escape(digits)}\s*R(?![A-Z0-9])", context.upper()):
            score += 22.0
        if room.startswith("B") and b_floor_num <= max_b_ocr_confusion_floor() and re.search(rf"(?<![A-Z0-9])[8B]{re.escape(digits)}\d(?!\d)", context.upper()):
            score += 6.0 if has_receiver or has_address_structure(context) else 3.0
        if room.startswith("B") and b_floor_num > max_b_ocr_confusion_floor() and re.search(rf"(?<!\d){re.escape(digits)}\s*호", context):
            score -= 10.0
        if room.startswith("B") and b_floor_num > max_b_ocr_confusion_floor() and not re.search(rf"(?<![A-Z0-9])B{re.escape(digits)}\s*호", context.upper()):
            score -= 6.0
        if "-" not in room and re.search(rf"(?<!\d){re.escape(digits)}\s*호", context):
            score += 8.0
        elif re.search(rf"(?<![A-Z\d*#-]){re.escape(digits)}[.,]0(?!\d)", context):
            score += 8.0
        elif re.search(rf"(?<![A-Z\d*#-]){re.escape(digits)}[.,]\s*호", context):
            score += 8.0
        elif re.search(rf"(?<![A-Z\d*#-]){re.escape(digits)}[05][.,](?!\d)", context):
            score += 8.0
        elif re.search(rf"(?<![A-Z\d*#-]){re.escape(digits)}R(?![A-Z\d])", context.upper()):
            score += 8.0
        elif re.search(rf"(?<![A-Z\d*#-]){re.escape(digits)}0(?!\d)", context) and not digits.endswith("0"):
            raw_with_zero = f"{digits}0"
            raw_floor = int(raw_with_zero[:-2]) if len(raw_with_zero) == 4 else 0
            score += 9.0 if len(digits) == 3 and raw_floor > max_reasonable_floor() else 3.0
        elif re.search(r"(?<!\d)\d{3,4}\s*호", context):
            score -= 8.0
        if re.search(r"010[.\-\s*\d]{4,}", context) and not has_address_context(context) and not has_receiver:
            score -= 4.0
    return score


def item_confidence(item: dict[str, Any]) -> float:
    try:
        return float(item.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def has_address_label(context: str) -> bool:
    return any(keyword in context for keyword in ADDRESS_KEYWORDS)


def has_address_structure(context: str) -> bool:
    normalized = re.sub(r"\s+", " ", context.upper())
    return any(re.search(pattern, normalized) for pattern in ADDRESS_CONTEXT_PATTERNS)


def has_address_context(context: str) -> bool:
    return has_address_label(context) or has_address_structure(context)


def has_non_destination_context(context: str) -> bool:
    normalized = re.sub(r"\s+", " ", context.upper())
    return any(keyword in normalized for keyword in NON_DESTINATION_KEYWORDS)


def destination_candidates(items: list[dict[str, Any]], radius: int = 5, max_candidates: int = 8) -> list[dict[str, Any]]:
    candidates: list[DestinationCandidate] = []
    seen: set[tuple[str, str | None, str | None, tuple[int, ...]]] = set()
    for idx, item in enumerate(items):
        rooms = extract_room_texts(str(item.get("text") or ""))
        if not rooms:
            continue
        left = max(0, idx - radius)
        right = min(len(items), idx + radius + 1)
        context = text_window(items, left, right, idx)
        if item_confidence(item) < 0.6 and not has_address_context(context):
            continue
        for room in rooms:
            floor, floor_source = parse_candidate_floor(context)
            room = apply_explicit_basement_to_room(room, floor)
            inferred_floor = room_floor(room)
            if inferred_floor is not None:
                floor = inferred_floor
                floor_source = "room_inferred"
            elif floor is None:
                floor = inferred_floor
                floor_source = "room_inferred" if floor is not None else None
            dongs = parse_candidate_dongs(context) or [None]
            floor_options = [(floor, floor_source)]
            basement_floor = basement_floor_from_b_room(room) if floor is None else None
            if basement_floor is not None:
                floor_options.append((basement_floor, "room_inferred"))

            evidence = [idx]
            for j in range(left, right):
                if not same_variant(items, idx, j):
                    continue
                if j != idx and parse_candidate_floor(str(items[j].get("text") or ""))[0] == floor:
                    evidence.append(j)
            for candidate_floor, candidate_floor_source in floor_options:
                for dong in dongs:
                    key = (room, candidate_floor, dong, tuple(sorted(evidence)))
                    if key in seen:
                        continue
                    seen.add(key)
                    score = candidate_score(context, room, candidate_floor_source, dong)
                    if room.startswith("B") and candidate_floor and candidate_floor.startswith("B"):
                        score += 0.3
                    candidates.append(
                        DestinationCandidate(
                            candidate_id=len(candidates),
                            destination_floor=candidate_floor,
                            destination_room=room,
                            destination_dong=dong,
                            floor_source=candidate_floor_source,
                            address_text=context,
                            evidence_indices=sorted(evidence),
                            score=score,
                        )
                    )

    for idx, item in enumerate(items):
        left = max(0, idx - radius)
        right = min(len(items), idx + radius + 1)
        context = text_window(items, left, right, idx)
        rooms = context_room_texts(context)
        if not rooms:
            continue
        for room in rooms:
            context_floor, _context_floor_source = parse_candidate_floor(context)
            room = apply_explicit_basement_to_room(room, context_floor)
            inferred_floor = room_floor(room)
            if inferred_floor is not None:
                floor = inferred_floor
                floor_source = "room_inferred"
            elif context_floor is not None:
                floor = context_floor
                floor_source = _context_floor_source
            else:
                floor = None
                floor_source = None
            floor_options = [(floor, floor_source)]
            basement_floor = basement_floor_from_b_room(room) if floor is None and context_floor is None else None
            if basement_floor is not None:
                floor_options.append((basement_floor, "room_inferred"))
            dongs = parse_candidate_dongs(context) or [None]
            evidence = [idx]
            for candidate_floor, candidate_floor_source in floor_options:
                for dong in dongs:
                    key = (room, candidate_floor, dong, ())
                    if key in seen:
                        continue
                    seen.add(key)
                    score = candidate_score(context, room, candidate_floor_source, dong) + 0.8
                    if room.startswith("B") and candidate_floor and candidate_floor.startswith("B"):
                        score += 0.3
                    candidates.append(
                        DestinationCandidate(
                            candidate_id=len(candidates),
                            destination_floor=candidate_floor,
                            destination_room=room,
                            destination_dong=dong,
                            floor_source=candidate_floor_source,
                            address_text=context,
                            evidence_indices=evidence,
                            score=score,
                        )
                    )

    for idx, item in enumerate(items):
        item_floor, floor_source = parse_candidate_floor(str(item.get("text") or ""))
        if item_floor is None:
            continue
        left = max(0, idx - radius)
        right = min(len(items), idx + radius + 1)
        context = text_window(items, left, right, idx)
        rooms_in_context = context_room_texts(context)
        if any(not room_looks_like_receiver_numeric_name(context, room) for room in rooms_in_context):
            continue
        if not has_address_context(context):
            continue
        evidence = [idx]
        for dong in parse_candidate_dongs(context) or [None]:
            key = ("", item_floor, dong, tuple(evidence))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                DestinationCandidate(
                    candidate_id=len(candidates),
                    destination_floor=item_floor,
                    destination_room=None,
                    destination_dong=dong,
                    floor_source=floor_source,
                    address_text=context,
                    evidence_indices=evidence,
                    score=candidate_score(context, None, floor_source, dong) - 0.5,
                )
            )

    candidates = [
        candidate
        for candidate in candidates
        if candidate.destination_room is None or room_allowed_by_policy(candidate.destination_room)
    ]
    candidates.sort(key=lambda item: item.score, reverse=True)
    for idx, candidate in enumerate(candidates):
        candidate.candidate_id = idx
    return [asdict(candidate) for candidate in candidates[:max_candidates]]


def build_candidate_lines(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return "(none)"
    lines: list[str] = []
    for candidate in candidates:
        context = str(candidate.get("address_text") or "")
        if len(context) > 160:
            context = context[:157] + "..."
        lines.append(
            f"[{candidate['candidate_id']}] floor={candidate.get('destination_floor')} "
            f"room={candidate.get('destination_room')} source={candidate.get('floor_source')} "
            f"evidence={candidate.get('evidence_indices')} context=\"{context}\""
        )
    return "\n".join(lines)
