"""
Communication Service - Handles WhatsApp messaging and external communications.

This module provides functions for sending WhatsApp messages and images
through the Exotel API.
"""

import os
import requests
from requests.auth import HTTPBasicAuth
import json
import logging
from config import SETTINGS

logger = logging.getLogger(__name__)

EXOTEL_SID = SETTINGS.EXOTEL_SID
EXOTEL_KEY = SETTINGS.EXOTEL_KEY
EXOTEL_TOKEN = SETTINGS.EXOTEL_TOKEN
EXOTEL_SUBDOMAIN = SETTINGS.EXOTEL_SUBDOMAIN

api_url = f'https://{EXOTEL_KEY}:{EXOTEL_TOKEN}@{EXOTEL_SUBDOMAIN}/v2/accounts/{EXOTEL_SID}/messages'


def sendMessage(toNumber, message):
    """Send a text message to a WhatsApp number.

    Args:
        toNumber (str): Recipient phone number
        message (str): Message content to send
    """
    payload = json.dumps({
        "whatsapp": {
            "messages": [
                {
                    "from": SETTINGS.WHATSAPP_FROM,
                    "to": toNumber,
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

    response = requests.post(url=api_url, data=payload, auth=HTTPBasicAuth(EXOTEL_KEY, EXOTEL_TOKEN), timeout=(10, 30))
    logger.info(response.content)


def sendImage(toNumber, img_url):
    """Send an image to a WhatsApp number.

    Args:
        toNumber (str): Recipient phone number
        img_url (str): URL of the image to send
    """
    logger.debug(f"Sending image {img_url} to {toNumber}")
    payload = json.dumps({
        "whatsapp": {
            "messages": [
                {
                    "from": SETTINGS.WHATSAPP_FROM,
                    "to": toNumber,
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

    response = requests.post(url=api_url, data=payload, headers=headers, auth=HTTPBasicAuth(EXOTEL_KEY, EXOTEL_TOKEN), timeout=(10, 30))
    logger.info(response.content)
