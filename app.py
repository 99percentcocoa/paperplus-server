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

import os
import logging
from datetime import datetime
import json
import threading
from pathlib import Path

from flask import Flask, request, send_from_directory, abort
from tinydb import TinyDB
import requests
import cv2
from PIL import Image
import sendmessage
import apriltags
import image
import tags
import omr_detection
from config import SETTINGS

# load_dotenv()
logger = logging.getLogger(__name__)

app = Flask(__name__)

SAVE_DIR = SETTINGS.DOWNLOADS_PATH
DEWARPED_DIR = SETTINGS.DEWARPED_PATH
SHEETS_LOGGING_URL = SETTINGS.SHEETS_LOGGING_URL

DOWNLOADS_PATH = SETTINGS.DOWNLOADS_PATH
DEWARPED_PATH = SETTINGS.DEWARPED_PATH
DEBUG_PATH = SETTINGS.DEBUG_PATH
CHECKED_PATH = SETTINGS.CHECKED_PATH
LOGS_PATH = SETTINGS.LOGS_PATH
SERVER_IP = SETTINGS.SERVER_IP
# setup logging. Logs to a new file every time a message is received (webhook is called)
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

def log_to_sheet(sender, fileURL, debugURL, checkedURL, marked, score, logURL):
    """Log grading results to Google Sheets.

    Creates a payload with grading information and sends it to the configured
    Google Sheets webhook URL for logging purposes.

    Args:
        sender (str): WhatsApp sender identifier
        fileURL (str): URL of the original uploaded file
        debugURL (str): URL of the debug processing image
        checkedURL (str): URL of the graded result image
        marked (str): JSON string of detected answers
        score (int): Number of correct answers
        logURL (str): URL of the session log file
    """
    payload = {
        "sender": sender,
        "fileURL": fileURL,
        "debugURL": debugURL,
        "checkedURL": checkedURL,
        "marked": marked,
        "score": score,
        "logURL": logURL
    }
    logger.info("Google Sheet Logging Payload: %s", payload)
    requests.post(SHEETS_LOGGING_URL, json=payload, headers={"Content-Type": "application/json"}, timeout=(10, 30))

def is_valid_image_message(message):
    """Check if message is a valid incoming image message and extract image URL.

    Args:
        message (dict): WhatsApp message object

    Returns:
        tuple: (is_valid, image_url, sender_number) or (False, None, None)
    """
    fromNo = message.get("from")
    callback_type = message.get("callback_type")

    # Filter out non-incoming messages
    if callback_type and callback_type != "incoming_message":
        logger.info("Received delivery message. Continuing...")
        return False, None, None

    content = message.get("content", {})
    if not content:
        logger.info("No content found. Skipping...")
        return False, None, None

    if content.get("type") == "image" and "image" in content:
        img = content["image"]
        url = img.get("url") or img.get("link")
        if not url:
            return False, None, None
        return True, url, fromNo

    return False, None, None


def download_image(url, session_id, sender_number):
    """Download image from URL and save to disk.

    Args:
        url (str): Image URL to download
        session_id (str): Session identifier
        sender_number (str): Sender's phone number

    Returns:
        tuple: (filepath, fileURL) for the downloaded image
    """
    r = requests.get(url, stream=True, timeout=30)
    ext = r.headers.get("Content-Type", "image/jpeg").split("/")[-1]
    filename = f"{session_id}_{sender_number[1:]}.{ext}"
    fileURL = f"http://{SERVER_IP}:3000/files/{filename}"

    filepath = os.path.join(SAVE_DIR, filename)
    with open(filepath, "wb") as f:
        for chunk in r.iter_content(1024):
            f.write(chunk)

    logger.debug("Saved image: %s", filepath)
    return filepath, fileURL


def detect_and_validate_corner_tags(filepath):
    """Detect AprilTags for corner positioning and validate detection.

    Args:
        filepath (str): Path to the image file

    Returns:
        tuple: (corner_tags, success) where success indicates if exactly 4 tags found
    """
    # Detect corner tags (36h11)
    corner_tags = apriltags.detect_tags_36h11(filepath)

    if len(corner_tags) < 4:
        # Try processing again in case of faint printing
        logger.info("Less than 4 corner tags detected. Reprocessing image for better detection.")
        faint_preprocessed_img = image.faint_preprocess(filepath)
        corner_tags = apriltags.detect_tags_36h11(faint_preprocessed_img)

    corner_tag_ids = [x.tag_id for x in corner_tags]
    logger.debug("Detected corner tags: %s", corner_tag_ids)

    return corner_tags, len(corner_tags) == 4


def process_image(filepath, corner_tags):
    """Process image: dewarp, clean, and prepare for OMR.

    Args:
        filepath (str): Original image path
        corner_tags: Detected corner tags for dewarp reference

    Returns:
        tuple: (dewarped_img, debug_img, checked_img, dewarped_filepath)
    """
    # Dewarp and clean the image
    cropped_img = image.dewarp_omr(filepath, corner_tags)
    dewarped_img = image.clean_document(cropped_img)
    debug_img = dewarped_img.copy()
    logger.info("Dewarped image.")

    # Prepare checked image (PIL format)
    checked_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))

    # Save dewarped image
    dewarped_filename = f"{Path(filepath).stem}_dewarped.jpg"
    dewarped_filepath = os.path.join(DEWARPED_DIR, dewarped_filename)
    cv2.imwrite(dewarped_filepath, dewarped_img)
    logger.debug("Saved dewarped image to %s", dewarped_filepath)

    return dewarped_img, debug_img, checked_img, dewarped_filepath


def process_omr_answers(dewarped_img, debug_img, checked_img, worksheet_id):
    """Process OMR answers using 25h9 tags.

    Args:
        dewarped_img: Processed image for OMR
        debug_img: Image for debug visualization
        checked_img: PIL image for marking answers
        worksheet_id: ID to lookup answer key

    Returns:
        tuple: (answers, ans_key, success) where success indicates if processing completed
    """
    # Load answer key from database
    db = TinyDB('worksheets.json')
    ans_key = db.get(doc_id=worksheet_id).get('answerKey')
    logger.info("Answer key for worksheet %s: %s", worksheet_id, ans_key)

    # Detect 25h9 tags for questions
    detection_25h9 = apriltags.detect_tags_25h9(dewarped_img)
    detected_tags_25h9 = list(map(lambda x: x.tag_id, detection_25h9))
    logger.debug("Detected question tags: %s", detected_tags_25h9)

    # Verify 25h9 tags detection
    required = set(range(1, 11))
    present = set(detected_tags_25h9)

    if not required.issubset(present):
        missing = required - present
        logger.debug("Missing 25h9 tags: %s", missing)
        return None, None, False  # Failed - missing tags

    # Remove extra tags if any
    extra = present - required
    if extra:
        logger.debug("Extra tags detected: %s", extra)
        detection_25h9[:] = [d for d in detection_25h9 if d.tag_id not in list(extra)]
        detected_tags_25h9 = list(map(lambda x: x.tag_id, detection_25h9))
        logger.debug("Extra tags removed, new list: %s", detected_tags_25h9)
    else:
        logger.info("All 25h9 tags are correct.")

    # Process answers for each tag
    answers = []
    tag_points = list(map(lambda t: tuple(map(int, t.center.tolist())), detection_25h9))

    for i, point in enumerate(tag_points):
        logger.debug("Processing point %s.", i+1)

        # Left question
        q_left_ans_key = ans_key[i*2]
        logger.debug("Processing question %s.", i*2+1)
        q_left_ans = omr_detection.detect_bubble(
            dewarped_img, point, omr_detection.LEFT_QUESTION_ROI,
            debug_img, checked_img, q_left_ans_key
        )

        # Right question
        q_right_ans_key = ans_key[i*2+1]
        logger.debug("Processing question %s.", i*2+2)
        q_right_ans = omr_detection.detect_bubble(
            dewarped_img, point, omr_detection.RIGHT_QUESTION_ROI,
            debug_img, checked_img, q_right_ans_key
        )

        answers.extend([q_left_ans, q_right_ans])
        logger.debug("Q%s: %s.", i*2+1, q_left_ans)
        logger.debug("Q%s: %s.", i*2+2, q_right_ans)

    logger.info("Finished checking answers.")
    return answers, ans_key, True


def handle_results(filepath, answers, ans_key, debug_img, checked_img, fromNo, fileURL, logURL):
    """Handle grading results: save images, send messages, and log to sheets.

    Args:
        filepath: Original image path
        answers: Detected answers list
        ans_key: Correct answer key
        debug_img: Debug visualization image
        checked_img: PIL image with marked answers
        fromNo: Sender number
        fileURL: URL of original file
        logURL: URL of log file
    """
    logger.info("Answers: %s", answers)
    score = check_results(answers, ans_key)

    # Save debug image
    debug_filename = f'debug_{Path(filepath).stem}.jpg'
    debug_filepath = os.path.join(DEBUG_PATH, debug_filename)
    cv2.imwrite(debug_filepath, debug_img)
    logger.debug("Saved debug image at %s", debug_filepath)

    # Save checked image with score
    checked_filename = f'checked_{Path(filepath).stem}.jpg'
    checked_filepath = os.path.join(CHECKED_PATH, checked_filename)
    checked_URL = f"http://{SERVER_IP}:3000/checked/{checked_filename}"

    # Add marks circle to checked image
    check_circle = omr_detection.make_circle_mark(score, len(ans_key))
    checked_img.paste(check_circle, (100, 50), check_circle)
    checked_img.save(checked_filepath)
    logger.debug("Saved checked image at %s using PIL.", checked_filepath)

    debugURL = f"http://{SERVER_IP}:3000/debug/{debug_filename}"

    # Send results to user
    sendmessage.sendMessage(fromNo, f"Your marks: {score}/{len(ans_key)} \n तुमचे मार्क: {score}/{len(ans_key)}")
    logger.info("Sending checked image.")
    sendmessage.sendImage(fromNo, checked_URL)

    # Log to Google Sheets
    logsheet_args = (fromNo, fileURL, debugURL, checked_URL, json.dumps(answers), score, logURL)
    logger.debug("Logging %s", logsheet_args)
    threading.Thread(target=log_to_sheet, args=(logsheet_args)).start()


def handle_message(data, session_id):
    logURL = f"http://{SERVER_IP}:3000/logs/{session_id}.log"
    try:
        logger.info("Received: %s", data)

        messages = data.get("whatsapp", {}).get("messages", [])
        for message in messages:
            fromNo = message.get("from")
            logger.info("Received message from %s", fromNo)

            # Validate message and extract image URL
            is_valid, image_url, _ = is_valid_image_message(message)

            if is_valid:
                logger.info("Processing valid image message from %s", fromNo)

                # Download the image
                filepath, fileURL = download_image(image_url, session_id, fromNo)

                # Send processing message
                threading.Thread(target=sendmessage.sendMessage, args=(fromNo, "Checking... ⏳ \n कार्यपत्रिका तपासत आहे... ⏳",)).start()

                # Detect and validate corner tags
                corner_tags, corner_tags_valid = detect_and_validate_corner_tags(filepath)

                if corner_tags_valid:
                    # Sort detections clockwise and decode worksheet
                    corner_tags = tags.sort_detections_clockwise(corner_tags)
                    corner_tag_ids = [x.tag_id for x in corner_tags]
                    logger.debug("Clockwise tag_ids: %s", [[x.tag_id, x.center] for x in corner_tags])

                    worksheet_id, corner_tags = tags.detect_orientation_and_decode(corner_tags)
                    corner_tag_ids = [x.tag_id for x in corner_tags]
                    logger.debug("Worksheet ID: %s, tag_ids: %s", worksheet_id, corner_tag_ids)

                    # Process the image (dewarp, clean, etc.)
                    dewarped_img, debug_img, checked_img, _ = process_image(filepath, corner_tags)

                    # Process OMR answers
                    answers, ans_key, omr_success = process_omr_answers(dewarped_img, debug_img, checked_img, worksheet_id)

                    if omr_success:
                        # Handle successful results
                        handle_results(filepath, answers, ans_key, debug_img, checked_img, fromNo, fileURL, logURL)
                    else:
                        # OMR failed - missing question tags
                        sendmessage.sendMessage(fromNo, "Please try again. ⟳ \n फोटो परत काढा. ⟳")
                else:
                    # Corner tags detection failed
                    logger.debug("Less/more than 4 corner tags found.")
                    sendmessage.sendMessage(fromNo, "Please take a complete photo of the worksheet. ⟳ \n कृपया कार्यपत्रिका संपूर्ण दिसेल असा फोटो काढा. ⟳")

                    # Log failed scan
                    logsheet_args = (fromNo, fileURL, "", "", "failed", "", logURL)
                    threading.Thread(target=log_to_sheet, args=(logsheet_args)).start()
            else:
                # Handle non-image messages
                sendmessage.sendMessage(fromNo, "Please send an image of a scanned worksheet. \n कृप्या केवळ कार्यपत्रिकेचा फोटो काढा.")

                # Log failed scan (user message does not contain image)
                logsheet_args = (fromNo, "none", "", "", "failed", "", logURL)
                threading.Thread(target=log_to_sheet, args=(logsheet_args)).start()

    except (requests.RequestException, IOError, OSError, ValueError, KeyError) as e:
        logger.exception("Error in background thread: %s", e)
    except Exception as e:  # pylint: disable=broad-except
        # Catch any other unexpected exceptions to prevent thread crashes
        logger.exception("Unexpected error in background thread: %s", e)

# use threading for the webhook so that can return 200 ok to exotel to avoid receiving duplicates
@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    setup_logging(session_id)
    threading.Thread(target=handle_message, args=(data,session_id,)).start()
    return "ok", 200

def check_results(results, ans_key):
    logger.info("Checking results.")
    if (len(results) != len(ans_key)):
        return "An error occurred. Please try again. ⟳"
    else:
        marked_lowercase = [item.lower() for item in results]
        anskey_lowercase = [item.lower() for item in ans_key]

        marks = 0

        for i in range(len(marked_lowercase)):
            if marked_lowercase[i] == anskey_lowercase[i]:
                marks += 1
        
        return marks


# serve files from downloads
@app.route('/files/<path:filename>')
def serve_file(filename):
    try:
        return send_from_directory(DOWNLOADS_PATH, filename, as_attachment=False)
    except FileNotFoundError:
        abort(404)

# serve files from debug
@app.route('/debug/<path:filename>')
def serve_debug_file(filename):
    try:
        return send_from_directory(DEBUG_PATH, filename, as_attachment=False)
    except FileNotFoundError:
        abort(404)

# serve files from checked
@app.route('/checked/<path:filename>', methods=['GET'])
def serve_checked_file(filename):
    try:
        return send_from_directory(CHECKED_PATH, filename, as_attachment=False)
    except FileNotFoundError:
        abort(404)

# serve log files
@app.route('/logs/<path:filename>', methods=['GET'])
def serve_log(filename):
    try:
        return send_from_directory(LOGS_PATH, filename, as_attachment=True)
    except FileNotFoundError:
        abort(404)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)