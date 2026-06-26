from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from indory_ocr.settings import Settings


class OcrLlmProvider(ABC):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def scan_waybill(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def read_ocr(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def read_room_signs(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def inspect_vlm(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
