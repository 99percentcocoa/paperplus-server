"""Injectable interface for calling vision-service, kept swappable for a future async/queued
implementation (e.g. Celery) without touching callers.
"""

import base64
import binascii
import logging
from pathlib import Path
from typing import Protocol

import httpx

from app.core.config import settings
from app.domain.errors import VisionClientError
from shared.contracts import ProcessingResult, ProcessRequest

logger = logging.getLogger(__name__)


class VisionClient(Protocol):
    def process(self, image_path: str, correlation_id: str, template_hint: str | None = None) -> ProcessingResult: ...


class HTTPVisionClient:
    """Synchronous HTTP call to vision-service's /process endpoint."""

    def __init__(self, base_url: str | None = None, shared_secret: str | None = None, timeout: float = 60.0):
        self._base_url = (base_url or settings.vision_service_url).rstrip("/")
        self._shared_secret = shared_secret or settings.vision_service_shared_secret
        self._timeout = timeout

    def process(self, image_path: str, correlation_id: str, template_hint: str | None = None) -> ProcessingResult:
        """image_path is a file on *this* service's disk (the stored upload). vision-service never
        sees a path: the photo travels as base64 and the dewarped page comes back the same way, then
        gets stored locally and exposed as result.dewarped_image_path."""
        try:
            image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        except OSError as exc:
            raise VisionClientError(f"could not read upload for vision-service: {type(exc).__name__}") from exc

        request = ProcessRequest(correlation_id=correlation_id, image_b64=image_b64, template_hint=template_hint)
        try:
            response = httpx.post(
                f"{self._base_url}/process",
                json=request.model_dump(),
                headers={"x-service-secret": self._shared_secret},
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise VisionClientError(
                f"vision-service call failed ({exc.response.status_code}): {_error_detail(exc.response)}"
            ) from exc
        except httpx.HTTPError as exc:
            raise VisionClientError(f"vision-service call failed: {exc}") from exc

        return _store_dewarped_image(ProcessingResult.model_validate(response.json()), correlation_id)


def _error_detail(response: httpx.Response) -> str:
    """Why vision-service rejected the request (e.g. which corner tags it could not find). httpx's
    own status-error text leaves out the response body, which is where that reason lives. Capped
    because FastAPI's validation errors can echo the request input, i.e. the whole base64 photo."""
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    return str(detail or response.text or response.reason_phrase)[:300]


def _store_dewarped_image(result: ProcessingResult, correlation_id: str) -> ProcessingResult:
    """Decode the returned dewarped page into api-service's own storage (needed later to draw and
    redraw the checked image) and drop the base64 so it never lands in scans.vision_result. A failure
    here degrades to "no checked image for this scan" rather than failing grading that succeeded."""
    if not result.dewarped_image_b64:
        return result
    path = Path(settings.storage_root) / "dewarped" / f"{correlation_id}_dewarped.jpg"
    try:
        data = base64.b64decode(result.dewarped_image_b64, validate=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    except (binascii.Error, ValueError, OSError):
        logger.exception("Could not store the dewarped image returned by vision-service")
        return result.model_copy(update={"dewarped_image_b64": None})
    return result.model_copy(update={"dewarped_image_b64": None, "dewarped_image_path": str(path)})
