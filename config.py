import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class Config:
    # Server
    SERVER_IP = os.getenv('SERVER_IP')

    # External Services
    SHEETS_LOGGING_URL = os.getenv('SHEETS_LOGGING_URL')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

    # Exotel / WhatsApp
    EXOTEL_SID = os.getenv('EXOTEL_SID')
    EXOTEL_KEY = os.getenv('EXOTEL_KEY')
    EXOTEL_TOKEN = os.getenv('EXOTEL_TOKEN')
    EXOTEL_SUBDOMAIN = os.getenv('EXOTEL_SUBDOMAIN')
    WHATSAPP_FROM = "+912071173227"

    # Paths
    DOWNLOADS_PATH = os.getenv('DOWNLOADS_PATH')
    DEWARPED_PATH = os.getenv('DEWARPED_PATH')
    DEBUG_PATH = os.getenv('DEBUG_PATH')
    CHECKED_PATH = os.getenv('CHECKED_PATH')
    LOGS_PATH = os.getenv('LOGS_PATH')

    # Image / OMR Settings
    TARGET_WIDTH = 1240
    TARGET_HEIGHT = 1754

    # roi format: (x_offset, y_offset, width, height)
    LEFT_QUESTION_ROI = (85, -40, 475, 85)
    RIGHT_QUESTION_ROI = (620, -40, 475, 85)

    MIN_MARK_AREA = 600
    MAX_MARK_AREA = 950
    FILL_THRESHOLD = 0.6
    MIN_CIRCULARITY = 0.75

SETTINGS = Config()