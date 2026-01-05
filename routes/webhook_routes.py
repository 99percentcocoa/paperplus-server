"""
Webhook Routes - Flask routes for handling WhatsApp webhooks.

This module contains the Flask blueprint for webhook endpoints
that receive and process WhatsApp messages.
"""

from flask import Blueprint, request
from datetime import datetime
import threading
from services.message_service import handle_message
from services.logging_service import setup_logging

webhook_bp = Blueprint('webhook', __name__)


@webhook_bp.route("/webhook", methods=["POST"])
def webhook():
    """Handle incoming WhatsApp webhook messages.

    Receives webhook data from WhatsApp, sets up logging for the session,
    and processes the message in a background thread.

    Returns:
        tuple: ("ok", 200) to acknowledge receipt
    """
    data = request.get_json()
    session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    setup_logging(session_id)
    threading.Thread(target=handle_message, args=(data, session_id,)).start()
    return "ok", 200
