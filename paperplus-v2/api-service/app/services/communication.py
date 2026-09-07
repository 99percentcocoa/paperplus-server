"""Exotel WhatsApp API client — ported exactly from services/communication_service.py
(URL format, payload shapes) so message formatting stays compatible with the existing
Exotel account setup.
"""

import json
import logging
from typing import Protocol

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class CommunicationClient(Protocol):
    def send_message(self, to_number: str, message: str) -> None: ...
    def send_image(self, to_number: str, image_url: str, caption: str = "") -> None: ...


class ExotelCommunicationClient:
    def __init__(self):
        self._api_url = (
            f"https://{settings.exotel_key}:{settings.exotel_token}@{settings.exotel_subdomain}"
            f"/v2/accounts/{settings.exotel_sid}/messages"
        )

    def send_message(self, to_number: str, message: str) -> None:
        if settings.local_mode:
            logger.info("[LOCAL_MODE] send_message to=%s message=%s", to_number, message)
            return

        payload = {
            "whatsapp": {
                "messages": [
                    {
                        "from": settings.whatsapp_from,
                        "to": to_number,
                        "content": {"type": "text", "text": {"body": message}},
                    }
                ]
            }
        }
        self._post(payload)

    def send_image(self, to_number: str, image_url: str, caption: str = "") -> None:
        if settings.local_mode:
            logger.info("[LOCAL_MODE] send_image to=%s url=%s caption=%s", to_number, image_url, caption)
            return

        payload = {
            "whatsapp": {
                "messages": [
                    {
                        "from": settings.whatsapp_from,
                        "to": to_number,
                        "content": {"type": "image", "image": {"link": image_url, "caption": caption}},
                    }
                ]
            }
        }
        self._post(payload)

    def _post(self, payload: dict) -> None:
        auth = (settings.exotel_key, settings.exotel_token)
        httpx.post(
            self._api_url,
            content=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            auth=auth,
            timeout=httpx.Timeout(30.0, connect=10.0),
        )


def is_valid_image_message(message: dict) -> tuple[bool, str | None]:
    """Filters out delivery receipts and extracts the image URL, if any."""
    callback_type = message.get("callback_type")
    if callback_type and callback_type != "incoming_message":
        return False, None

    content = message.get("content") or {}
    if content.get("type") == "image" and "image" in content:
        image = content["image"]
        url = image.get("url") or image.get("link")
        if url:
            return True, url

    return False, None
