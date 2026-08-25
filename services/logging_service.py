"""
Logging Service - Handles logging configuration and external integrations.

This module contains functions for setting up session logging and
integrating with external logging services like Google Sheets.
"""

import os
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import requests
from config import SETTINGS

logger = logging.getLogger(__name__)

LOGS_PATH = SETTINGS.LOGS_PATH
SHEETS_LOGGING_URL = SETTINGS.SHEETS_LOGGING_URL


@contextmanager
def timing_context(operation_name, **context):
    """Capture timestamps and elapsed time for expensive operations."""
    started_at = datetime.now(timezone.utc)
    started_monotonic = time.perf_counter()
    timing = {
        "operation": operation_name,
        "started_at": started_at.isoformat(),
        **context,
    }
    try:
        yield timing
    finally:
        ended_at = datetime.now(timezone.utc)
        elapsed_ms = (time.perf_counter() - started_monotonic) * 1000
        timing["ended_at"] = ended_at.isoformat()
        timing["duration_ms"] = round(elapsed_ms, 2)
        logger.info("Operation timing: %s", timing)


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


def log_to_sheet(sender, file_url, debug_url, checkedURL, marked, score, log_url, roll_number, worksheet_id):
    """Log grading results to Google Sheets.

    Creates a payload with grading information and sends it to the configured
    Google Sheets webhook URL for logging purposes.

    Args:
        sender (str): WhatsApp sender identifier
        file_url (str): URL of the original uploaded file
        debug_url (str): URL of the debug processing image
        checkedURL (str): URL of the graded result image
        marked (str): JSON string of detected answers
        score (int): Number of correct answers
        log_url (str): URL of the session log file
        roll_number (str): detected roll number
        worksheet_id (str): detected worksheet id
    """
    payload = {
        "sender": sender,
        "fileURL": file_url,
        "debugURL": debug_url,
        "checkedURL": checkedURL,
        "marked": marked,
        "score": score,
        "logURL": log_url,
        "detectedRollNumber": roll_number,
        "worksheet_id": worksheet_id
    }
    logger.info("Google Sheet Logging Payload: %s", payload)
    with timing_context("log_to_sheet", sender=sender, worksheet_id=worksheet_id):
        requests.post(SHEETS_LOGGING_URL, json=payload, headers={"Content-Type": "application/json"}, timeout=(10, 30))
