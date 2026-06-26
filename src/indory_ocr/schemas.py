from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict
except ImportError:  # pydantic v1 compatibility for system ROS/Python envs.
    ConfigDict = None  # type: ignore[assignment]


class FlexibleModel(BaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(extra="allow", populate_by_name=True)
    else:
        class Config:
            extra = "allow"
            allow_population_by_field_name = True


def json_schema_for(model_cls: type[BaseModel]) -> dict[str, Any]:
    method = getattr(model_cls, "model_json_schema", None)
    if callable(method):
        return method()
    return model_cls.schema()


def validate_model(model_cls: type[BaseModel], payload: dict[str, Any]) -> BaseModel:
    method = getattr(model_cls, "model_validate", None)
    if callable(method):
        return method(payload)
    return model_cls.parse_obj(payload)


class OcrItem(FlexibleModel):
    text: str
    confidence: Optional[float] = None
    box: list[list[float]] = Field(default_factory=list)
    cx: Optional[float] = None
    cy: Optional[float] = None
    rotation_degrees: Optional[int] = None
    frame_index: Optional[int] = None


class DestinationCandidate(FlexibleModel):
    candidate_id: Optional[int] = None
    destination_floor: Optional[str] = None
    destination_room: Optional[str] = None
    destination_dong: Optional[str] = None
    floor_source: Optional[str] = None
    address_text: Optional[str] = None
    evidence_indices: list[int] = Field(default_factory=list)
    score: Optional[float] = None


class DestinationDecision(FlexibleModel):
    destination_dong: Optional[str] = None
    destination_floor: Optional[str] = None
    destination_room: Optional[str] = None
    confidence: Optional[float] = None
    evidence_indices: list[int] = Field(default_factory=list)
    needs_manual_review: bool = True
    auto_accept: bool = False
    risk_score: Optional[float] = None
    risk_reasons: list[str] = Field(default_factory=list)


class Timing(FlexibleModel):
    ocr_seconds: float = 0.0
    llm_seconds: float = 0.0
    total_seconds: float = 0.0


class WaybillScanRequest(FlexibleModel):
    request_id: Optional[str] = None
    task_id: Optional[Union[str, int]] = None
    camera: Optional[str] = None
    source: Optional[str] = None
    image_b64: Optional[str] = None
    image_base64: Optional[str] = None
    image_format: Optional[str] = "jpg"
    content_type: Optional[str] = None
    images: Optional[list[Any]] = None
    frames: Optional[list[Any]] = None
    include_debug: bool = False
    options: dict[str, Any] = Field(default_factory=dict)


class OcrReadRequest(FlexibleModel):
    request_id: Optional[str] = None
    camera: Optional[str] = None
    source: Optional[str] = None
    image_b64: Optional[str] = None
    image_base64: Optional[str] = None
    image_format: Optional[str] = "jpg"
    content_type: Optional[str] = None
    images: Optional[list[Any]] = None
    frames: Optional[list[Any]] = None
    ocr_rotations: Optional[list[int]] = None
    include_debug: bool = False
    options: dict[str, Any] = Field(default_factory=dict)


class SemanticOcrRequest(OcrReadRequest):
    floor_hint: Optional[str] = None
    floor_prior_mode: Optional[str] = "reject"
    min_confidence: Optional[float] = None


class VlmInspectRequest(FlexibleModel):
    request_id: Optional[str] = None
    camera: Optional[str] = None
    source: Optional[str] = None
    task_mode: str = "scene_description"
    image_b64: Optional[str] = None
    image_base64: Optional[str] = None
    image_format: Optional[str] = "jpg"
    content_type: Optional[str] = None
    model_name: Optional[str] = None
    device: Optional[str] = None
    torch_dtype: Optional[str] = None
    max_new_tokens: Optional[int] = None
    include_debug: bool = False
    options: dict[str, Any] = Field(default_factory=dict)


class WaybillScanResponse(FlexibleModel):
    type: str = "result"
    ok: bool = True
    request_id: str
    task_id: Optional[Union[str, int]] = None
    camera: Optional[str] = None
    source: Optional[str] = None
    destination: Optional[str] = None
    decision: DestinationDecision
    needs_manual_review: bool
    auto_accept: bool
    risk_reasons: list[str] = Field(default_factory=list)
    timing: Timing = Field(default_factory=Timing)
    debug: Optional[dict[str, Any]] = None


class OcrReadResponse(FlexibleModel):
    type: str = "ocr_result"
    ok: bool = True
    request_id: str
    camera: Optional[str] = None
    source: Optional[str] = None
    model: Optional[str] = None
    rotations: list[int] = Field(default_factory=list)
    item_count: int = 0
    items: list[OcrItem] = Field(default_factory=list)
    frames: list[dict[str, Any]] = Field(default_factory=list)
    timing: Timing = Field(default_factory=Timing)
    debug: Optional[dict[str, Any]] = None


class SemanticOcrResponse(FlexibleModel):
    type: str = "semantic_ocr_result"
    ok: bool = True
    request_id: str
    camera: Optional[str] = None
    source: Optional[str] = None
    task_mode: str = "ocr_room_ids"
    has_text_object: bool = False
    objects: list[dict[str, Any]] = Field(default_factory=list)
    raw_ocr_output: list[dict[str, Any]] = Field(default_factory=list)
    control_summary_ko: str = ""
    need_human_check: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    timing: Timing = Field(default_factory=Timing)


class VlmInspectResponse(FlexibleModel):
    type: str = "vlm_result"
    ok: bool = True
    request_id: str
    camera: Optional[str] = None
    source: Optional[str] = None
    task_mode: str = "scene_description"
    observation: Optional[dict[str, Any]] = None
    raw_response: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    timing: Timing = Field(default_factory=Timing)
