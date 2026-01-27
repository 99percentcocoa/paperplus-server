"""
Communication Service - Handles WhatsApp messaging and external communications.

This module provides functions for sending WhatsApp messages and images
through the Exotel API.
"""

import json
import logging
import requests
from requests.auth import HTTPBasicAuth
from config import SETTINGS

logger = logging.getLogger(__name__)

EXOTEL_SID = SETTINGS.EXOTEL_SID
EXOTEL_KEY = SETTINGS.EXOTEL_KEY
EXOTEL_TOKEN = SETTINGS.EXOTEL_TOKEN
EXOTEL_SUBDOMAIN = SETTINGS.EXOTEL_SUBDOMAIN

api_url = (
    f'https://{EXOTEL_KEY}:{EXOTEL_TOKEN}@{EXOTEL_SUBDOMAIN}'
    f'/v2/accounts/{EXOTEL_SID}/messages'
)


def send_message(to_number, message):
    """Send a text message to a WhatsApp number.

    Args:
        to_number (str): Recipient phone number
        message (str): Message content to send
    """
    payload = json.dumps({
        "whatsapp": {
            "messages": [
                {
                    "from": SETTINGS.WHATSAPP_FROM,
                    "to": to_number,
                    "content": {
                        "type": "text",
                        "text": {
                            "body": message
                        }
                    }
                }
            ]
        }
    })

    response = requests.post(url=api_url, data=payload, auth=HTTPBasicAuth(
        EXOTEL_KEY, EXOTEL_TOKEN), timeout=(10, 30))
    logger.info(response.content)


def send_image(to_number, img_url):
    """Send an image to a WhatsApp number.

    Args:
        to_number (str): Recipient phone number
        img_url (str): URL of the image to send
    """
    logger.debug("Sending image %s to %s", img_url, to_number)
    payload = json.dumps({
        "whatsapp": {
            "messages": [
                {
                    "from": SETTINGS.WHATSAPP_FROM,
                    "to": to_number,
                    "content": {
                        "type": "image",
                        "image": {
                            "link": img_url,
                            "caption": "Your answers."
                        }
                    }
                }
            ]
        }
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.post(url=api_url, data=payload, headers=headers, auth=HTTPBasicAuth(
        EXOTEL_KEY, EXOTEL_TOKEN), timeout=(10, 30))
    logger.info(response.content)

def is_valid_image_message(message):
    """Check if message is a valid incoming image message and extract image URL.

    Args:
        message (dict): WhatsApp message object

    Returns:
        tuple: (is_valid, image_url, sender_number) or (False, None, None)
    """
    from_no = message.get("from")
    callback_type = message.get("callback_type")

    # Filter out non-incoming messages
    if callback_type and callback_type != "incoming_message":
        logger.info("Received delivery message. Continuing...")
        return False, None, None

    content = message.get("content", {})
    if not content:
        logger.info("No content found. Skipping...")
        return False, None, None

    if content.get("type") == "image" and "image" in content:
        img = content["image"]
        url = img.get("url") or img.get("link")
        if not url:
            return False, None, None
        return True, url, from_no

    return False, None, None