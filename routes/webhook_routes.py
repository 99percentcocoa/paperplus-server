"""
Webhook Routes - Flask routes for handling WhatsApp webhooks.

This module contains the Flask blueprint for webhook endpoints
that receive and process WhatsApp messages.
"""

from flask import Blueprint, request
from datetime import datetime
import threading

from services.logging_service import setup_logging

webhook_bp = Blueprint('webhook', __name__)


def _should_process_message(data):
    """Return True only for inbound user messages that need processing."""
    if not isinstance(data, dict):
        return False

    messages = data.get("whatsapp", {}).get("messages", [])
    if not isinstance(messages, list):
        return False

    for message in messages:
        if not isinstance(message, dict):
            continue

        callback_type = message.get("callback_type")
        if callback_type and callback_type != "incoming_message":
            continue

        if message.get("from") or "content" in message:
            return True

    return False


@webhook_bp.route("/webhook", methods=["POST"])
def webhook():
    """Handle incoming WhatsApp webhook messages.

    Receives webhook data from WhatsApp, sets up logging for the session,
    and processes the message in a background thread.

    Returns:
        tuple: ("ok", 200) to acknowledge receipt
    """
    data = request.get_json()
    if not _should_process_message(data):
        return "ok", 200

    session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    setup_logging(session_id)

    try:
        from services.message_service import handle_message
    except ModuleNotFoundError:
        return "Service unavailable", 503

    threading.Thread(target=handle_message, args=(data, session_id,)).start()
    return "ok", 200
