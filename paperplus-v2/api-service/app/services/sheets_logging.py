"""Fire-and-forget audit log of graded submissions to an external Google Sheets webhook.

Ported from the old system's services/logging_service.py:log_to_sheet. Posts to a plain HTTP
webhook URL (a Google Apps Script endpoint on the sheet, not the Sheets API -- no OAuth
credentials needed), matching the old payload shape so the same sheet/script can keep being
used. No-op if `sheets_logging_url` isn't configured, so this stays safe to call in dev/test
environments that don't have a sheet wired up.
"""

import json
import logging
import threading

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


def log_to_sheet_async(
    sender: str,
    checked_url: str,
    marked: list[dict],
    score: int,
    roll_number: str | None,
    worksheet_id: int | None,
) -> None:
    if not settings.sheets_logging_url:
        return
    thread = threading.Thread(
        target=_log_to_sheet,
        args=(sender, checked_url, marked, score, roll_number, worksheet_id),
        daemon=True,
    )
    thread.start()


def _log_to_sheet(
    sender: str,
    checked_url: str,
    marked: list[dict],
    score: int,
    roll_number: str | None,
    worksheet_id: int | None,
) -> None:
    payload = {
        "sender": sender,
        "checkedURL": checked_url,
        "marked": json.dumps(marked),
        "score": score,
        "detectedRollNumber": roll_number,
        "worksheet_id": worksheet_id,
    }
    logger.info("Google Sheet logging payload: %s", payload)
    try:
        httpx.post(
            settings.sheets_logging_url,
            content=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=httpx.Timeout(30.0, connect=10.0),
        )
    except httpx.HTTPError:
        logger.exception("Failed to log submission to Google Sheets for sender=%s", sender)
