from __future__ import annotations

import importlib

from indory_ocr.providers.base import OcrLlmProvider
from indory_ocr.providers.not_configured import NotConfiguredProvider
from indory_ocr.settings import Settings


def load_provider(settings: Settings) -> OcrLlmProvider:
    name = settings.provider.strip()
    if name in {"", "not_configured", "none", "mock"}:
        return NotConfiguredProvider(settings)
    if name in {"gz_compat", "indory"}:
        from indory_ocr.providers.gz_compat import GzCompatProvider

        return GzCompatProvider(settings)
    if ":" not in name:
        raise ValueError(
            f"unknown provider '{name}'. Use 'gz_compat', 'not_configured', or a Python path like module:Class."
        )
    module_name, class_name = name.split(":", 1)
    module = importlib.import_module(module_name)
    provider_cls = getattr(module, class_name)
    provider = provider_cls(settings)
    if not isinstance(provider, OcrLlmProvider):
        raise TypeError(f"provider {name} must inherit OcrLlmProvider")
    return provider


__all__ = ["OcrLlmProvider", "load_provider"]
