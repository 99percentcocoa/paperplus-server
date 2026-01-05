"""
PaperPlus Server - WhatsApp Worksheet Grading System

A Flask-based webhook server that processes WhatsApp messages containing scanned worksheets.
The system detects AprilTags on worksheets, performs optical mark recognition (OMR),
grades the answers against a database of answer keys, and sends results back to users.

Features:
- WhatsApp webhook integration for receiving images
- AprilTag detection for worksheet identification and orientation
- Image processing and OMR for answer extraction
- Automated grading against stored answer keys
- Result logging to Google Sheets
- Multi-language support (English and Marathi)

Environment Variables Required:
- SHEETS_LOGGING_URL: Google Sheets webhook URL for logging
- DOWNLOADS_PATH: Directory for storing downloaded images
- DEWARPED_PATH: Directory for processed images
- DEBUG_PATH: Directory for debug images
- CHECKED_PATH: Directory for graded result images
- LOGS_PATH: Directory for session logs
- SERVER_IP: Server IP address for file URLs
"""

from flask import Flask
from routes.webhook_routes import webhook_bp
from routes.file_routes import file_bp

app = Flask(__name__)

# Register blueprints
app.register_blueprint(webhook_bp)
app.register_blueprint(file_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)