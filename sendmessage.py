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

if __name__ == '__main__':
    sendMessage("+919145339332", "python hello")