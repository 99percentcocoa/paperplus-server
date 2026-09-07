"""Webhook endpoint for incoming Exotel WhatsApp messages.

Processes synchronously in the request handler (no background thread / task queue, per the
redevelopment decision) — the response to Exotel is only returned once grading completes.
"""

import logging
import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends
from sqlmodel import Session

from app.core.config import settings
from app.db.session import get_session
from app.services.communication import ExotelCommunicationClient, is_valid_image_message
from app.services.submission_service import handle_incoming_image
from app.services.vision_client import HTTPVisionClient

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/webhook")
def webhook(payload: dict, session: Session = Depends(get_session)) -> dict:
    messages = ((payload.get("whatsapp") or {}).get("messages")) or []

    vision_client = HTTPVisionClient()
    comm_client = ExotelCommunicationClient()

    for message in messages:
        from_number = message.get("from")
        is_valid, image_url = is_valid_image_message(message)
        if not is_valid or not from_number or not image_url:
            continue

        correlation_id = str(uuid.uuid4())
        try:
            image_path = _download_image(image_url, correlation_id)
        except httpx.HTTPError:
            logger.exception("Failed to download image for correlation_id=%s", correlation_id)
            continue

        handle_incoming_image(session, vision_client, comm_client, from_number, image_path, correlation_id)

    return {"status": "ok"}


def _download_image(image_url: str, correlation_id: str) -> str:
    response = httpx.get(image_url, timeout=30.0)
    response.raise_for_status()

    ext = response.headers.get("Content-Type", "image/jpeg").split("/")[-1]
    upload_dir = Path(settings.storage_root) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    image_path = upload_dir / f"{correlation_id}.{ext}"
    image_path.write_bytes(response.content)
    return str(image_path)
