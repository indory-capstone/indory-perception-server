from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import contextlib
import ctypes
import json
import os
import re
import signal
import site
import tempfile
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import websockets

from .candidates import destination_candidates
from .llm import (
    assess_decision_risk,
    build_judge,
    build_prompt,
    decision_from_candidate_response,
    decision_label,
    layout_decision_from_candidates,
    parse_json_object,
)
from .ocr import build_paddleocr, flatten_ocr, normalize_ocr_rotations, ocr_model_label, rotate_image_variant, run_paddleocr


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
LEGACY_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "gemma-3-270m-it-Q4_K_M.gguf"
DEFAULT_ARTIFACT_ROOT = Path(os.environ.get("WAYBILL_OCR_WS_ARTIFACT_ROOT", "/tmp/waybill_ocr_ws"))


@dataclass
class ServerConfig:
    host: str
    port: int
    mode: str
    model_path: Path | None
    model: str
    endpoint: str
    ocr_use_gpu: bool
    ocr_rec_batch_num: int
    ocr_rotations: list[int]
    llm_gpu_layers: int
    llm_threads: int
    llm_ctx: int
    max_new_tokens: int
    artifact_root: Path
    keep_artifacts: bool
    include_debug: bool
    include_prompt: bool
    allow_local_paths: bool
    max_image_mb: int


def json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def now_unix() -> float:
    return time.time()


def perf() -> float:
    return time.perf_counter()


def repo_default_model_path() -> Path | None:
    env_path = os.environ.get("WAYBILL_OCR_DEFAULT_MODEL", "").strip()
    if env_path:
        return Path(env_path).expanduser()
    if DEFAULT_MODEL_PATH.is_file():
        return DEFAULT_MODEL_PATH
    if LEGACY_MODEL_PATH.is_file():
        return LEGACY_MODEL_PATH
    return DEFAULT_MODEL_PATH


def parse_ocr_rotations(value: str) -> list[int]:
    rotations: list[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            rotation = int(part)
        except ValueError as exc:
            raise ValueError("ocr rotations must be comma-separated degrees, for example 0,90,180,270") from exc
        rotations.append(rotation)
    return normalize_ocr_rotations(rotations)


def _site_package_dirs() -> list[Path]:
    dirs: list[Path] = []
    for value in site.getsitepackages() + [site.getusersitepackages()]:
        path = Path(value)
        if path.exists():
            dirs.append(path)
    return dirs


def preload_nvidia_cuda_wheel_libs() -> list[str]:
    """Make CUDA wheels usable even when the caller did not export LD_LIBRARY_PATH.

    The CUDA llama-cpp-python wheel depends on libcudart/cublas shared objects
    shipped by nvidia-* pip packages. Loading those libraries by absolute path
    before llama_cpp is imported lets libllama.so resolve them in this process.
    """

    lib_dirs: list[Path] = []
    for site_dir in _site_package_dirs():
        for relative in (
            "nvidia/cuda_runtime/lib",
            "nvidia/cublas/lib",
            "nvidia/cuda_nvrtc/lib",
        ):
            path = site_dir / relative
            if path.exists():
                lib_dirs.append(path)

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    additions = [str(path) for path in lib_dirs if str(path) not in existing.split(":")]
    if additions:
        os.environ["LD_LIBRARY_PATH"] = ":".join(additions + ([existing] if existing else []))

    loaded: list[str] = []
    wanted = (
        "libcudart.so.12",
        "libcublas.so.12",
        "libcublasLt.so.12",
        "libnvrtc.so.12",
    )
    for lib_name in wanted:
        for lib_dir in lib_dirs:
            lib_path = lib_dir / lib_name
            if not lib_path.exists():
                continue
            try:
                ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                loaded.append(str(lib_path))
            except OSError:
                pass
            break
    return loaded


def image_suffix(value: Any) -> str:
    text = str(value or "jpg").strip().lower()
    text = text.split(";", 1)[0].split("/", 1)[-1]
    text = re.sub(r"[^a-z0-9]+", "", text)
    if text in {"jpeg", "jpg"}:
        return ".jpg"
    if text in {"png", "webp", "bmp"}:
        return f".{text}"
    return ".jpg"


def decode_image_b64(value: str, max_bytes: int) -> bytes:
    text = value.strip()
    if "," in text and text[:64].lower().startswith("data:"):
        text = text.split(",", 1)[1]
    try:
        data = base64.b64decode(text, validate=True)
    except binascii.Error as exc:
        raise ValueError(f"invalid base64 image: {exc}") from exc
    if not data:
        raise ValueError("empty image payload")
    if len(data) > max_bytes:
        raise ValueError(f"image payload too large: {len(data)} bytes > {max_bytes} bytes")
    return data


def normalized_request_id(value: Any) -> str:
    text = str(value or "").strip()
    if text:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:96]
    return uuid.uuid4().hex


def extract_payload(message: str | bytes) -> dict[str, Any]:
    if isinstance(message, bytes):
        return {
            "type": "scan",
            "request_id": uuid.uuid4().hex,
            "image_bytes": message,
            "image_format": "jpg",
        }
    try:
        value = json.loads(message)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON websocket message: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("websocket message must be a JSON object")
    return value


def decision_key(decision: dict[str, Any] | None) -> str:
    if not decision:
        return "NO_DEST"
    return "|".join(
        str(decision.get(key) or "")
        for key in ("destination_dong", "destination_floor", "destination_room")
    )


def merge_risk_reasons(*reason_lists: Any) -> list[str]:
    merged: list[str] = []
    for reasons in reason_lists:
        if not isinstance(reasons, list):
            continue
        for reason in reasons:
            text = str(reason)
            if text and text not in merged:
                merged.append(text)
    return merged


class ResidentWaybillScanner:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.cuda_preloaded: list[str] = []
        if config.mode == "llama_cpp" and config.llm_gpu_layers != 0:
            self.cuda_preloaded = preload_nvidia_cuda_wheel_libs()

        self.effective_ocr_use_gpu = config.ocr_use_gpu
        try:
            self.ocr = build_paddleocr(
                use_gpu=config.ocr_use_gpu,
                rec_batch_num=config.ocr_rec_batch_num,
            )
        except Exception:
            if not config.ocr_use_gpu:
                raise
            self.effective_ocr_use_gpu = False
            self.ocr = build_paddleocr(
                use_gpu=False,
                rec_batch_num=config.ocr_rec_batch_num,
            )
        self.ocr_model = ocr_model_label(
            use_gpu=self.effective_ocr_use_gpu,
            rec_batch_num=config.ocr_rec_batch_num,
        )
        self.judge = build_judge(
            mode=config.mode,
            model_path=config.model_path,
            model=config.model,
            endpoint=config.endpoint,
            max_new_tokens=config.max_new_tokens,
            n_ctx=config.llm_ctx,
            n_gpu_layers=config.llm_gpu_layers,
            n_threads=config.llm_threads,
        )
        self.llm_gpu_offload_supported: bool | None = None
        if config.mode == "llama_cpp":
            try:
                import llama_cpp

                self.llm_gpu_offload_supported = bool(llama_cpp.llama_supports_gpu_offload())
            except Exception:
                self.llm_gpu_offload_supported = None

    def health_payload(self) -> dict[str, Any]:
        return {
            "type": "health",
            "ok": True,
            "server": "waybill_ocr_llm_ws",
            "ocr_model": self.ocr_model,
            "llm": {
                "mode": self.config.mode,
                "model_path": str(self.config.model_path) if self.config.model_path else None,
                "model": self.config.model or None,
                "endpoint": self.config.endpoint or None,
                "n_gpu_layers": self.config.llm_gpu_layers,
                "n_ctx": self.config.llm_ctx,
                "n_threads": self.config.llm_threads,
                "max_new_tokens": self.config.max_new_tokens,
                "gpu_offload_supported": self.llm_gpu_offload_supported,
            },
            "ocr_rotations": self.config.ocr_rotations,
            "requested_ocr_use_gpu": self.config.ocr_use_gpu,
            "effective_ocr_use_gpu": self.effective_ocr_use_gpu,
            "keep_artifacts": self.config.keep_artifacts,
            "include_debug": self.config.include_debug,
            "artifact_root": str(self.config.artifact_root) if self.config.keep_artifacts else None,
            "cuda_preloaded": self.cuda_preloaded,
            "protocol": {
                "scan_request": {
                    "type": "scan",
                    "request_id": "optional string",
                    "image_b64": "base64 jpeg/png/webp, data URL accepted",
                    "images": "optional array of 2-10 base64 strings or frame objects for multi-frame consensus",
                    "image_format": "jpg|png|webp optional",
                    "camera": "optional source name",
                    "task_id": "optional backend task id",
                },
                "scan_response": {
                    "type": "result",
                    "ok": True,
                    "request_id": "same request id",
                    "destination": "normalized label",
                    "decision": "final destination_room/floor/dong/confidence/evidence only",
                    "frames": "multi-frame vote summary when images/frames array was provided",
                    "timing": "ocr/llm/total seconds",
                    "debug": "only present when server starts with --include-debug",
                },
            },
        }

    def scan_image_result(self, image_path: Path, request_meta: dict[str, Any], artifact_dir: Path | None) -> dict[str, Any]:
        total_t0 = perf()
        request_id = str(request_meta["request_id"])

        ocr_t0 = perf()
        ocr_items: list[dict[str, Any]] = []
        ocr_variants: list[dict[str, Any]] = []
        rotation_out_dir = artifact_dir if artifact_dir is not None else image_path.parent
        for idx, rotation in enumerate(self.config.ocr_rotations, 1):
            variant_path = rotate_image_variant(image_path, rotation, rotation_out_dir, idx)
            variant_t0 = perf()
            raw_ocr = run_paddleocr(self.ocr, variant_path)
            variant_items: list[dict[str, Any]] = []
            for item in flatten_ocr(raw_ocr):
                item_dict = asdict(item)
                item_dict["source_image"] = str(image_path)
                item_dict["rotation_degrees"] = rotation
                variant_items.append(item_dict)
            variant_seconds = perf() - variant_t0
            ocr_items.extend(variant_items)
            ocr_variants.append(
                {
                    "image": str(variant_path),
                    "source_image": str(image_path),
                    "rotation_degrees": rotation,
                    "rotation_variant": rotation != 0,
                    "ocr_seconds": variant_seconds,
                    "ocr_item_count": len(variant_items),
                    "ocr_items": variant_items,
                }
            )
        ocr_seconds = perf() - ocr_t0

        result_for_prompt = {
            "image": str(image_path),
            "source_image": str(image_path),
            "rotations": self.config.ocr_rotations,
            "ocr_seconds": ocr_seconds,
            "ocr_item_count": len(ocr_items),
            "ocr_items": ocr_items,
        }
        candidates = destination_candidates(ocr_items)

        prompt = ""
        raw_response = ""
        parsed_response: dict[str, Any] | None = None
        llm_decision: dict[str, Any] | None = None
        layout_decision: dict[str, Any] | None = None
        final_decision: dict[str, Any] | None = None
        llm_seconds = 0.0
        skip_reason = None

        if ocr_items:
            prompt = build_prompt(result_for_prompt)
            llm_t0 = perf()
            raw_response = self.judge.generate(prompt)
            llm_seconds = perf() - llm_t0
            parsed_response = parse_json_object(raw_response)
            llm_decision = decision_from_candidate_response(parsed_response, candidates)
            layout_decision = layout_decision_from_candidates(result_for_prompt, candidates)
            final_decision = layout_decision or llm_decision
        else:
            skip_reason = "No OCR text detected; skipped LLM judgement."

        risk = assess_decision_risk(result_for_prompt, candidates, final_decision, parsed_response)
        if final_decision is not None:
            final_decision = {**final_decision, **risk}
        else:
            final_decision = {
                "destination_floor": None,
                "destination_room": None,
                "destination_dong": None,
                "confidence": None,
                "evidence_indices": [],
                **risk,
            }

        response = {
            "type": "result",
            "ok": True,
            "request_id": request_id,
            "task_id": request_meta.get("task_id"),
            "camera": request_meta.get("camera"),
            "source": request_meta.get("source"),
            "received_at_unix": request_meta.get("received_at_unix"),
            "completed_at_unix": now_unix(),
            "destination": decision_label(final_decision),
            "decision": final_decision,
            "needs_manual_review": final_decision.get("needs_manual_review", True),
            "auto_accept": final_decision.get("auto_accept", False),
            "risk_reasons": final_decision.get("risk_reasons", []),
            "timing": {
                "ocr_seconds": ocr_seconds,
                "llm_seconds": llm_seconds,
                "total_seconds": perf() - total_t0,
            },
            "artifact_dir": str(artifact_dir) if artifact_dir is not None else None,
        }

        debug_payload = {
            "ocr": {
                "model": self.ocr_model,
                "image": str(image_path) if self.config.keep_artifacts else None,
                "rotations": self.config.ocr_rotations,
                "variants": ocr_variants,
                "item_count": len(ocr_items),
                "items": ocr_items,
            },
            "llm": {
                "mode": self.config.mode,
                "model_path": str(self.config.model_path) if self.config.model_path else None,
                "model": self.config.model or None,
                "endpoint": self.config.endpoint or None,
                "n_gpu_layers": self.config.llm_gpu_layers,
                "gpu_offload_supported": self.llm_gpu_offload_supported,
                "candidates": candidates,
                "raw_response": raw_response,
                "parsed_response": parsed_response,
                "llm_decision": llm_decision,
                "layout_decision": layout_decision,
                "prompt": prompt if self.config.include_prompt else None,
                "skip_reason": skip_reason,
            },
        }
        if self.config.include_debug:
            response["debug"] = debug_payload

        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_response = dict(response)
            artifact_response["debug"] = debug_payload
            (artifact_dir / "result.json").write_text(json.dumps(artifact_response, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if prompt:
                (artifact_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
            if raw_response:
                (artifact_dir / "llm_raw_response.txt").write_text(raw_response, encoding="utf-8")

        return response

    def consensus_response(self, frame_results: list[dict[str, Any]], request_meta: dict[str, Any], artifact_dir: Path | None) -> dict[str, Any]:
        completed_at = now_unix()
        if not frame_results:
            decision = {
                "destination_floor": None,
                "destination_room": None,
                "destination_dong": None,
                "confidence": None,
                "evidence_indices": [],
                "needs_manual_review": True,
                "auto_accept": False,
                "risk_score": 10.0,
                "risk_reasons": ["no_frames_processed"],
            }
            return {
                "type": "result",
                "ok": True,
                "request_id": request_meta["request_id"],
                "task_id": request_meta.get("task_id"),
                "camera": request_meta.get("camera"),
                "source": request_meta.get("source"),
                "received_at_unix": request_meta.get("received_at_unix"),
                "completed_at_unix": completed_at,
                "destination": decision_label(decision),
                "decision": decision,
                "needs_manual_review": True,
                "auto_accept": False,
                "risk_reasons": decision["risk_reasons"],
                "frames": {"count": 0, "accepted": 0, "votes": {}},
                "timing": {"ocr_seconds": 0.0, "llm_seconds": 0.0, "total_seconds": 0.0},
                "artifact_dir": str(artifact_dir) if artifact_dir is not None else None,
            }

        votes = Counter(decision_key(result.get("decision")) for result in frame_results)
        votes.pop("NO_DEST", None)
        if votes:
            winner_key, winner_votes = votes.most_common(1)[0]
            winner_results = [result for result in frame_results if decision_key(result.get("decision")) == winner_key]
            best_result = sorted(
                winner_results,
                key=lambda result: (
                    bool(result.get("auto_accept")),
                    -float((result.get("decision") or {}).get("risk_score") or 0.0),
                    float((result.get("decision") or {}).get("confidence") or 0.0),
                ),
                reverse=True,
            )[0]
            final_decision = dict(best_result.get("decision") or {})
        else:
            winner_key = "NO_DEST"
            winner_votes = 0
            best_result = min(
                frame_results,
                key=lambda result: float((result.get("decision") or {}).get("risk_score") or 99.0),
            )
            final_decision = dict(best_result.get("decision") or {})

        frame_count = len(frame_results)
        auto_accept_votes = sum(
            1
            for result in frame_results
            if decision_key(result.get("decision")) == winner_key and result.get("auto_accept")
        )
        support_ratio = winner_votes / frame_count if frame_count else 0.0
        all_reasons = merge_risk_reasons(*(result.get("risk_reasons") for result in frame_results))

        consensus_reasons: list[str] = []
        if winner_votes == 0:
            consensus_reasons.append("no_consensus_destination")
        if winner_votes < 2:
            consensus_reasons.append("single_frame_destination")
        if support_ratio < 0.4:
            consensus_reasons.append("low_frame_consensus")
        if auto_accept_votes == 0:
            consensus_reasons.append("no_auto_accepting_frame")

        final_decision["frame_vote_count"] = winner_votes
        final_decision["frame_count"] = frame_count
        final_decision["frame_support_ratio"] = round(support_ratio, 3)
        final_decision["risk_reasons"] = merge_risk_reasons(final_decision.get("risk_reasons"), consensus_reasons)
        final_decision["needs_manual_review"] = bool(final_decision["risk_reasons"])
        final_decision["auto_accept"] = not final_decision["needs_manual_review"]
        final_decision["risk_score"] = round(float(final_decision.get("risk_score") or 0.0) + len(consensus_reasons), 2)

        timing = {
            "ocr_seconds": sum(float((result.get("timing") or {}).get("ocr_seconds") or 0.0) for result in frame_results),
            "llm_seconds": sum(float((result.get("timing") or {}).get("llm_seconds") or 0.0) for result in frame_results),
            "total_seconds": sum(float((result.get("timing") or {}).get("total_seconds") or 0.0) for result in frame_results),
        }
        response = {
            "type": "result",
            "ok": True,
            "request_id": request_meta["request_id"],
            "task_id": request_meta.get("task_id"),
            "camera": request_meta.get("camera"),
            "source": request_meta.get("source"),
            "received_at_unix": request_meta.get("received_at_unix"),
            "completed_at_unix": completed_at,
            "destination": decision_label(final_decision),
            "decision": final_decision,
            "needs_manual_review": final_decision.get("needs_manual_review", True),
            "auto_accept": final_decision.get("auto_accept", False),
            "risk_reasons": final_decision.get("risk_reasons", []),
            "frames": {
                "count": frame_count,
                "accepted": auto_accept_votes,
                "winner_votes": winner_votes,
                "support_ratio": round(support_ratio, 3),
                "votes": dict(votes),
                "risk_reasons": all_reasons,
            },
            "timing": timing,
            "artifact_dir": str(artifact_dir) if artifact_dir is not None else None,
        }
        if self.config.include_debug:
            response["debug"] = {"frame_results": frame_results}
        if artifact_dir is not None:
            artifact_response = dict(response)
            artifact_response["debug"] = {"frame_results": frame_results}
            (artifact_dir / "result.json").write_text(json.dumps(artifact_response, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return response


def prepare_image(payload: dict[str, Any], request_dir: Path, config: ServerConfig, index: int | None = None) -> Path:
    max_bytes = config.max_image_mb * 1024 * 1024
    prefix = "input" if index is None else f"frame_{index:03d}"
    if isinstance(payload.get("image_bytes"), bytes):
        data = payload["image_bytes"]
        if len(data) > max_bytes:
            raise ValueError(f"image payload too large: {len(data)} bytes > {max_bytes} bytes")
        suffix = image_suffix(payload.get("image_format"))
        path = request_dir / f"{prefix}{suffix}"
        path.write_bytes(data)
        return path

    image_b64 = payload.get("image_b64") or payload.get("image_base64") or payload.get("image")
    if isinstance(image_b64, str) and image_b64.strip():
        data = decode_image_b64(image_b64, max_bytes=max_bytes)
        suffix = image_suffix(payload.get("image_format") or payload.get("content_type"))
        path = request_dir / f"{prefix}{suffix}"
        path.write_bytes(data)
        return path

    image_path = payload.get("image_path")
    if isinstance(image_path, str) and image_path.strip():
        if not config.allow_local_paths:
            raise ValueError("image_path is disabled; start server with --allow-local-paths for trusted local clients")
        path = Path(image_path).expanduser()
        if not path.exists() or not path.is_file():
            raise ValueError(f"image_path does not exist: {path}")
        return path

    raise ValueError("scan request must include image_b64, image_base64, binary image bytes, or trusted image_path")


def frame_payloads(payload: dict[str, Any]) -> list[dict[str, Any]]:
    frames = payload.get("images") or payload.get("frames")
    if not isinstance(frames, list):
        return [payload]
    if not frames:
        raise ValueError("images/frames array is empty")
    result: list[dict[str, Any]] = []
    for idx, frame in enumerate(frames):
        if isinstance(frame, str):
            result.append({**payload, "image_b64": frame, "frame_index": idx})
        elif isinstance(frame, dict):
            merged = {**payload, **frame, "frame_index": frame.get("frame_index", idx)}
            merged.pop("images", None)
            merged.pop("frames", None)
            result.append(merged)
        else:
            raise ValueError(f"frame {idx} must be a base64 string or object")
    return result


def scan_payload(scanner: ResidentWaybillScanner, payload: dict[str, Any]) -> dict[str, Any]:
    request_id = normalized_request_id(payload.get("request_id"))
    request_meta = {
        "request_id": request_id,
        "task_id": payload.get("task_id"),
        "camera": payload.get("camera"),
        "source": payload.get("source"),
        "received_at_unix": now_unix(),
    }

    if scanner.config.keep_artifacts:
        request_dir = scanner.config.artifact_root / request_id
        request_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir: Path | None = request_dir
        (request_dir / "request.json").write_text(
            json.dumps({k: v for k, v in payload.items() if k not in {"image", "image_b64", "image_base64", "image_bytes"}}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        frames = frame_payloads(payload)
        frame_results: list[dict[str, Any]] = []
        for idx, frame in enumerate(frames):
            image_path = prepare_image(frame, request_dir, scanner.config, index=idx if len(frames) > 1 else None)
            frame_artifact_dir = request_dir / f"frame_{idx:03d}" if len(frames) > 1 else request_dir
            frame_meta = {**request_meta, "request_id": f"{request_id}#{idx}" if len(frames) > 1 else request_id}
            frame_result = scanner.scan_image_result(image_path=image_path, request_meta=frame_meta, artifact_dir=frame_artifact_dir)
            frame_result["frame_index"] = idx
            frame_results.append(frame_result)
        if len(frame_results) == 1:
            return frame_results[0]
        return scanner.consensus_response(frame_results, request_meta=request_meta, artifact_dir=artifact_dir)

    with tempfile.TemporaryDirectory(prefix=f"waybill_ws_{request_id}_") as temp_dir:
        request_dir = Path(temp_dir)
        frames = frame_payloads(payload)
        frame_results = []
        for idx, frame in enumerate(frames):
            image_path = prepare_image(frame, request_dir, scanner.config, index=idx if len(frames) > 1 else None)
            frame_meta = {**request_meta, "request_id": f"{request_id}#{idx}" if len(frames) > 1 else request_id}
            frame_result = scanner.scan_image_result(image_path=image_path, request_meta=frame_meta, artifact_dir=None)
            frame_result["frame_index"] = idx
            frame_results.append(frame_result)
        if len(frame_results) == 1:
            return frame_results[0]
        return scanner.consensus_response(frame_results, request_meta=request_meta, artifact_dir=None)


async def send_json(websocket: Any, payload: dict[str, Any]) -> None:
    await websocket.send(json_dumps(payload))


async def handle_client(websocket: Any, scanner: ResidentWaybillScanner, lock: asyncio.Lock) -> None:
    await send_json(websocket, {"type": "ready", "ok": True, "health": scanner.health_payload()})
    async for message in websocket:
        request_id = uuid.uuid4().hex
        try:
            payload = extract_payload(message)
            request_type = str(payload.get("type") or "scan").lower()
            request_id = normalized_request_id(payload.get("request_id") or request_id)
            payload["request_id"] = request_id

            if request_type in {"ping", "health"}:
                await send_json(websocket, {"type": "health", "ok": True, "request_id": request_id, "health": scanner.health_payload()})
                continue
            if request_type != "scan":
                raise ValueError(f"unsupported request type: {request_type}")

            await send_json(websocket, {"type": "accepted", "ok": True, "request_id": request_id, "queue": "single_worker"})
            async with lock:
                await send_json(websocket, {"type": "progress", "ok": True, "request_id": request_id, "stage": "processing"})
                response = await asyncio.to_thread(scan_payload, scanner, payload)
            await send_json(websocket, response)
        except Exception as exc:
            await send_json(
                websocket,
                {
                    "type": "error",
                    "ok": False,
                    "request_id": request_id,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )


async def serve(config: ServerConfig) -> None:
    scanner = ResidentWaybillScanner(config)
    lock = asyncio.Lock()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    max_size = max(1024 * 1024, int(config.max_image_mb * 1024 * 1024 * 1.8))
    async with websockets.serve(
        lambda websocket: handle_client(websocket, scanner, lock),
        config.host,
        config.port,
        max_size=max_size,
        ping_interval=20,
        ping_timeout=20,
    ):
        print(
            json_dumps(
                {
                    "type": "listening",
                    "ok": True,
                    "url": f"ws://{config.host}:{config.port}",
                    "health": scanner.health_payload(),
                }
            ),
            flush=True,
        )
        await stop.wait()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone WebSocket server for Korean waybill OCR + LLM extraction.")
    parser.add_argument("--host", default=os.environ.get("WAYBILL_OCR_WS_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("WAYBILL_OCR_WS_PORT", "8766")))
    parser.add_argument("--mode", default=os.environ.get("WAYBILL_OCR_DEFAULT_MODE", "llama_cpp"))
    parser.add_argument("--model-path", type=Path, default=repo_default_model_path())
    parser.add_argument("--model", default=os.environ.get("WAYBILL_OCR_DEFAULT_MODEL_NAME", ""))
    parser.add_argument("--endpoint", default=os.environ.get("WAYBILL_OCR_DEFAULT_ENDPOINT", ""))
    parser.add_argument("--ocr-gpu", action="store_true", default=os.environ.get("WAYBILL_OCR_USE_GPU", "0") == "1")
    parser.add_argument("--ocr-rec-batch-num", type=int, default=int(os.environ.get("WAYBILL_OCR_REC_BATCH_NUM", "1")))
    parser.add_argument("--ocr-rotations", default=os.environ.get("WAYBILL_OCR_ROTATIONS", "0,90,180,270"))
    parser.add_argument("--n-gpu-layers", type=int, default=int(os.environ.get("WAYBILL_OCR_LLM_GPU_LAYERS", "-1")))
    parser.add_argument("--n-threads", type=int, default=int(os.environ.get("WAYBILL_OCR_LLM_THREADS", "4")))
    parser.add_argument("--n-ctx", type=int, default=int(os.environ.get("WAYBILL_OCR_LLM_CTX", "4096")))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("WAYBILL_OCR_MAX_NEW_TOKENS", "128")))
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--keep-artifacts", action="store_true", default=os.environ.get("WAYBILL_OCR_WS_KEEP_ARTIFACTS", "0") == "1")
    parser.add_argument("--include-debug", action=argparse.BooleanOptionalAction, default=os.environ.get("WAYBILL_OCR_WS_INCLUDE_DEBUG", "0") == "1")
    parser.add_argument("--include-prompt", action=argparse.BooleanOptionalAction, default=os.environ.get("WAYBILL_OCR_WS_INCLUDE_PROMPT", "1") != "0")
    parser.add_argument("--allow-local-paths", action="store_true", default=os.environ.get("WAYBILL_OCR_WS_ALLOW_LOCAL_PATHS", "0") == "1")
    parser.add_argument("--max-image-mb", type=int, default=int(os.environ.get("WAYBILL_OCR_WS_MAX_IMAGE_MB", "16")))
    return parser


def config_from_args(args: argparse.Namespace) -> ServerConfig:
    model_path = args.model_path.expanduser() if args.model_path else None
    if args.mode == "llama_cpp" and model_path is not None and not model_path.exists():
        raise FileNotFoundError(f"llama_cpp model path does not exist: {model_path}")
    return ServerConfig(
        host=args.host,
        port=args.port,
        mode=args.mode,
        model_path=model_path,
        model=args.model,
        endpoint=args.endpoint,
        ocr_use_gpu=args.ocr_gpu,
        ocr_rec_batch_num=args.ocr_rec_batch_num,
        ocr_rotations=parse_ocr_rotations(args.ocr_rotations),
        llm_gpu_layers=args.n_gpu_layers,
        llm_threads=args.n_threads,
        llm_ctx=args.n_ctx,
        max_new_tokens=args.max_new_tokens,
        artifact_root=args.artifact_root.expanduser(),
        keep_artifacts=args.keep_artifacts,
        include_debug=args.include_debug,
        include_prompt=args.include_prompt,
        allow_local_paths=args.allow_local_paths,
        max_image_mb=max(1, args.max_image_mb),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    asyncio.run(serve(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
