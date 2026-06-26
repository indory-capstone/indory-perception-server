from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OcrItem:
    text: str
    confidence: float
    box: list[list[float]]
    cx: float
    cy: float


@dataclass
class DestinationCandidate:
    candidate_id: int
    destination_floor: str | None
    destination_room: str | None
    destination_dong: str | None
    floor_source: str | None
    address_text: str
    evidence_indices: list[int]
    score: float
