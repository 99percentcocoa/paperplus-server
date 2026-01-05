"""
File Routes - Flask routes for serving static files.

This module contains Flask blueprints for serving various types of files
including original images, debug images, checked results, and logs.
"""

from flask import Blueprint, send_from_directory, abort
from config import SETTINGS

file_bp = Blueprint('files', __name__)


@file_bp.route('/files/<path:filename>')
def serve_file(filename):
    """Serve original uploaded files."""
    try:
        return send_from_directory(SETTINGS.DOWNLOADS_PATH, filename, as_attachment=False)
    except FileNotFoundError:
        abort(404)


@file_bp.route('/debug/<path:filename>')
def serve_debug_file(filename):
    """Serve debug processing images."""
    try:
        return send_from_directory(SETTINGS.DEBUG_PATH, filename, as_attachment=False)
    except FileNotFoundError:
        abort(404)


@file_bp.route('/checked/<path:filename>', methods=['GET'])
def serve_checked_file(filename):
    """Serve graded result images."""
    try:
        return send_from_directory(SETTINGS.CHECKED_PATH, filename, as_attachment=False)
    except FileNotFoundError:
        abort(404)


@file_bp.route('/logs/<path:filename>', methods=['GET'])
def serve_log(filename):
    """Serve session log files."""
    try:
        return send_from_directory(SETTINGS.LOGS_PATH, filename, as_attachment=True)
    except FileNotFoundError:
        abort(404)
