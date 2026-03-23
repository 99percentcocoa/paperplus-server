import os
from dotenv import load_dotenv

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

    DEFAULT_NUM_QUESTIONS = 20
    NUM_ROW_TAGS = 10

    # roi format: (x_offset, y_offset, width, height)
    LEFT_QUESTION_ROI = (85, -40, 485, 90)
    RIGHT_QUESTION_ROI = (620, -40, 485, 90)

    MIN_MARK_AREA = 600
    MAX_MARK_AREA = 950
    FILL_THRESHOLD = 0.6
    MIN_CIRCULARITY = 0.75

    # Worksheet levels map to difficulty level distributions
    # Keys are difficulty levels, values are proportions of 20 questions
    WORKSHEET_LEVEL_DISTRIBUTIONS = {
        "A": {1: 1.0},
        "B": {1: 0.5, 2: 0.5},
        "C": {1: 0.25, 2: 0.25, 3: 0.5},
        "D": {1: 0.125, 2: 0.125, 3: 0.25, 4: 0.5},  # 1-2 (25%) split evenly
        "E": {1: 1/12, 2: 1/12, 3: 1/12, 4: 0.25, 5: 0.5},  # 1-3 (25%) split evenly
        "F": {1: 1/16, 2: 1/16, 3: 1/16, 4: 1/16, 5: 0.25, 6: 0.5},  # 1-4 (25%) split evenly
        "G": {1: 0.05, 2: 0.05, 3: 0.05, 4: 0.05, 5: 0.05, 6: 0.25, 7: 0.5},  # 1-5 (25%) split evenly
    }

SETTINGS = Config()