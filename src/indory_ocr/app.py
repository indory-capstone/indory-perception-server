from __future__ import annotations

import argparse
import time
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import Body, FastAPI, HTTPException

from indory_ocr import __version__
from indory_ocr.providers import load_provider
from indory_ocr.providers.base import OcrLlmProvider
from indory_ocr.schemas import (
    OcrReadRequest,
    SemanticOcrRequest,
    VlmInspectRequest,
    WaybillScanRequest,
    json_schema_for,
    validate_model,
)
from indory_ocr.settings import Settings
from indory_ocr.utils import normalize_payload


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.provider = load_provider(settings)
        yield

    app = FastAPI(
        title="Indory Control Server Detection",
        version=__version__,
        lifespan=lifespan,
    )
    # Older FastAPI/Starlette builds used in ROS workspaces may ignore the
    # lifespan argument during direct app construction. Seed state eagerly so
    # health/contracts and tests work across both old and new runtimes.
    app.state.settings = settings
    app.state.provider = load_provider(settings)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        provider: OcrLlmProvider = app.state.provider
        provider_health = await provider.health()
        return {
            "ok": True,
            "service": "control_server_detection",
            "version": __version__,
            "provider": provider.name,
            "provider_health": provider_health,
            "routes": {
                "waybill_scan": "/v1/waybill/scan",
                "ocr_read": "/v1/ocr/read",
                "semantic_ocr_room_signs": "/v1/semantic-ocr/room-signs",
                "vlm_inspect": "/v1/vlm/inspect",
            },
        }

    @app.get("/v1/contracts")
    async def contracts() -> dict[str, Any]:
        return {
            "waybill_scan": {
                "method": "POST",
                "path": "/v1/waybill/scan",
                "request": json_schema_for(WaybillScanRequest),
                "response_core_fields": [
                    "destination",
                    "decision.destination_floor",
                    "decision.destination_room",
                    "decision.confidence",
                    "needs_manual_review",
                    "auto_accept",
                    "risk_reasons",
                ],
            },
            "ocr_read": {
                "method": "POST",
                "path": "/v1/ocr/read",
                "request": json_schema_for(OcrReadRequest),
                "response_core_fields": [
                    "items[].text",
                    "items[].confidence",
                    "items[].box",
                    "items[].cx",
                    "items[].cy",
                ],
            },
            "semantic_ocr_room_signs": {
                "method": "POST",
                "path": "/v1/semantic-ocr/room-signs",
                "purpose": "room_id_sign_ocr",
                "floor_prior_modes": ["reject", "complete"],
                "request": json_schema_for(SemanticOcrRequest),
                "response_core_fields": [
                    "objects[].room_id",
                    "objects[].raw_text",
                    "objects[].confidence",
                    "objects[].bbox_xyxy",
                    "raw_ocr_output[]",
                    "metadata.floor_hint",
                    "metadata.floor_prior_mode",
                ],
            },
            "vlm_inspect": {
                "method": "POST",
                "path": "/v1/vlm/inspect",
                "task_modes": ["scene_description", "text_object"],
                "legacy_task_mode_aliases": {"object_detection": "text_object"},
                "request": json_schema_for(VlmInspectRequest),
                "response_core_fields": [
                    "task_mode",
                    "observation",
                    "raw_response",
                    "metadata.model",
                ],
            },
        }

    @app.post("/v1/waybill/scan")
    async def scan_waybill(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="request body must be a JSON object")
        normalized = normalize_payload(payload)
        try:
            validate_model(WaybillScanRequest, normalized)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid waybill scan request: {exc}") from exc

        provider: OcrLlmProvider = app.state.provider
        t0 = time.perf_counter()
        try:
            response = await provider.scan_waybill(normalized)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"waybill provider failed: {exc}") from exc
        timing = response.setdefault("timing", {})
        timing["total_seconds"] = float(timing.get("total_seconds") or (time.perf_counter() - t0))
        return response

    @app.post("/v1/ocr/read")
    async def read_ocr(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="request body must be a JSON object")
        normalized = normalize_payload(payload)
        try:
            validate_model(OcrReadRequest, normalized)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid OCR request: {exc}") from exc

        provider: OcrLlmProvider = app.state.provider
        t0 = time.perf_counter()
        try:
            response = await provider.read_ocr(normalized)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"OCR provider failed: {exc}") from exc
        timing = response.setdefault("timing", {})
        timing["total_seconds"] = float(timing.get("total_seconds") or (time.perf_counter() - t0))
        return response

    @app.post("/v1/semantic-ocr/room-signs")
    async def read_room_signs(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="request body must be a JSON object")
        normalized = normalize_payload(payload)
        try:
            validate_model(SemanticOcrRequest, normalized)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid semantic OCR request: {exc}") from exc

        provider: OcrLlmProvider = app.state.provider
        t0 = time.perf_counter()
        try:
            response = await provider.read_room_signs(normalized)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"semantic OCR provider failed: {exc}") from exc
        timing = response.setdefault("timing", {})
        timing["total_seconds"] = float(timing.get("total_seconds") or (time.perf_counter() - t0))
        return response

    @app.post("/v1/vlm/inspect")
    async def inspect_vlm(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="request body must be a JSON object")
        normalized = normalize_payload(payload)
        try:
            validate_model(VlmInspectRequest, normalized)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid VLM request: {exc}") from exc

        provider: OcrLlmProvider = app.state.provider
        t0 = time.perf_counter()
        try:
            response = await provider.inspect_vlm(normalized)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"VLM provider failed: {exc}") from exc
        timing = response.setdefault("timing", {})
        timing["total_seconds"] = float(timing.get("total_seconds") or (time.perf_counter() - t0))
        return response

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = Settings.from_env()
    parser = argparse.ArgumentParser(description="Run the Indory control-server-detection FastAPI service.")
    parser.add_argument("--host", default=defaults.host)
    parser.add_argument("--port", type=int, default=defaults.port)
    parser.add_argument("--provider", default=defaults.provider)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base = Settings.from_env()
    settings = Settings(
        host=args.host,
        port=args.port,
        provider=args.provider,
        artifact_root=base.artifact_root,
        keep_artifacts=base.keep_artifacts,
        include_debug=base.include_debug,
        max_image_mb=base.max_image_mb,
    )
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
