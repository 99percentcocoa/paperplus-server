"""
Logging Service - Handles logging configuration and external integrations.

This module contains functions for setting up session logging and
integrating with external logging services like Google Sheets.
"""

import os
import logging
import requests
from config import SETTINGS

logger = logging.getLogger(__name__)

LOGS_PATH = SETTINGS.LOGS_PATH
SHEETS_LOGGING_URL = SETTINGS.SHEETS_LOGGING_URL


def setup_logging(session_id):
    """Set up logging configuration for a new session.

    Creates a new log file for the session and configures logging to write to both
    the file and console. Removes any existing handlers to prevent duplicate logs.

    Args:
        session_id (str): Unique identifier for the current session, used in filename

    Returns:
        str: Path to the created log file
    """
    log_filename = f"{session_id}.log"
    log_path = os.path.join(LOGS_PATH, log_filename)

    # Remove any existing handlers (important for repeated runs)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure global logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s | %(funcName)s:%(lineno)d - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    logger.info("New session started. Logging to %s", log_path)
    return log_path


def log_to_sheet(sender, file_url, debugURL, checkedURL, marked, score, log_url):
    """Log grading results to Google Sheets.

    Creates a payload with grading information and sends it to the configured
    Google Sheets webhook URL for logging purposes.

    Args:
        sender (str): WhatsApp sender identifier
        file_url (str): URL of the original uploaded file
        debugURL (str): URL of the debug processing image
        checkedURL (str): URL of the graded result image
        marked (str): JSON string of detected answers
        score (int): Number of correct answers
        log_url (str): URL of the session log file
    """
    payload = {
        "sender": sender,
        "fileURL": file_url,
        "debugURL": debugURL,
        "checkedURL": checkedURL,
        "marked": marked,
        "score": score,
        "logURL": log_url
    }
    logger.info("Google Sheet Logging Payload: %s", payload)
    requests.post(SHEETS_LOGGING_URL, json=payload, headers={"Content-Type": "application/json"}, timeout=(10, 30))
