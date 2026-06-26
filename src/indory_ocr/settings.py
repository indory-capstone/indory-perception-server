from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, legacy_name: str | None = None, default: str = "") -> str:
    value = os.environ.get(name)
    if value is not None:
        return value
    if legacy_name:
        legacy_value = os.environ.get(legacy_name)
        if legacy_value is not None:
            return legacy_value
    return default


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int, legacy_name: str | None = None) -> int:
    try:
        return int(_env(name, legacy_name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    provider: str
    artifact_root: Path
    keep_artifacts: bool
    include_debug: bool
    max_image_mb: int

    @classmethod
    def from_env(cls) -> "Settings":
        legacy_host = _env("INDORY_OCR_HOST", "INDORY_OCR_LLM_HOST", "127.0.0.1")
        legacy_port = _int_env("INDORY_OCR_PORT", 8767, legacy_name="INDORY_OCR_LLM_PORT")
        provider = _env("CONTROL_SERVER_DETECTION_PROVIDER", default=os.environ.get("INDORY_OCR_PROVIDER", "gz_compat"))
        return cls(
            host=_env("CONTROL_SERVER_DETECTION_HOST", default=legacy_host),
            port=_int_env("CONTROL_SERVER_DETECTION_PORT", legacy_port),
            provider=provider.strip() or "gz_compat",
            artifact_root=Path(
                _env(
                    "CONTROL_SERVER_DETECTION_ARTIFACT_ROOT",
                    default=os.environ.get("INDORY_OCR_ARTIFACT_ROOT", "/tmp/control_server_detection"),
                )
            ),
            keep_artifacts=_bool_env(
                "CONTROL_SERVER_DETECTION_KEEP_ARTIFACTS",
                _bool_env("INDORY_OCR_KEEP_ARTIFACTS"),
            ),
            include_debug=_bool_env(
                "CONTROL_SERVER_DETECTION_INCLUDE_DEBUG",
                _bool_env("INDORY_OCR_INCLUDE_DEBUG"),
            ),
            max_image_mb=max(1, _int_env("CONTROL_SERVER_DETECTION_MAX_IMAGE_MB", _int_env("INDORY_OCR_MAX_IMAGE_MB", 16))),
        )
