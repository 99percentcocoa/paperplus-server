from dotenv import load_dotenv
import os
import requests
from requests.auth import HTTPBasicAuth
import json

EXOTEL_SID = os.getenv("EXOTEL_SID")
EXOTEL_KEY = os.getenv("EXOTEL_KEY")
EXOTEL_TOKEN = os.getenv("EXOTEL_TOKEN")
EXOTEL_SUBDOMAIN = os.getenv("EXOTEL_SUBDOMAIN")

url = f'https://{EXOTEL_KEY}:{EXOTEL_TOKEN}@{EXOTEL_SUBDOMAIN}/v2/accounts/{EXOTEL_SID}/messages'

def sendMessage(toNumber, message):
    payload = json.dumps({
        "whatsapp": {
            "messages": [
                {
                    "from": "+912071173227",
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

    response = requests.post(url=url, data=payload, auth=HTTPBasicAuth(EXOTEL_KEY, EXOTEL_TOKEN))
    print(response)

if __name__ == '__main__':
    sendMessage("+919145339332", "python hello")