"""Injectable interface for calling vision-service, kept swappable for a future async/queued
implementation (e.g. Celery) without touching callers.
"""

from typing import Protocol

import httpx

from app.core.config import settings
from app.domain.errors import VisionClientError
from shared.contracts import ProcessingResult, ProcessRequest


class VisionClient(Protocol):
    def process(self, image_path: str, correlation_id: str, template_hint: str | None = None) -> ProcessingResult: ...


class HTTPVisionClient:
    """Synchronous HTTP call to vision-service's /process endpoint."""

    def __init__(self, base_url: str | None = None, shared_secret: str | None = None, timeout: float = 60.0):
        self._base_url = (base_url or settings.vision_service_url).rstrip("/")
        self._shared_secret = shared_secret or settings.vision_service_shared_secret
        self._timeout = timeout

    def process(self, image_path: str, correlation_id: str, template_hint: str | None = None) -> ProcessingResult:
        request = ProcessRequest(correlation_id=correlation_id, image_path=image_path, template_hint=template_hint)
        try:
            response = httpx.post(
                f"{self._base_url}/process",
                json=request.model_dump(),
                headers={"x-service-secret": self._shared_secret},
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise VisionClientError(f"vision-service call failed: {exc}") from exc

        return ProcessingResult.model_validate(response.json())
