from dotenv import load_dotenv
import os
from flask import Flask, request, send_from_directory, abort
from tinydb import TinyDB, Query
import os, requests
import logging
from datetime import datetime
import sendmessage
import apriltags
import json
import threading
import image
import tags
import cv2
import omr_detection
from pathlib import Path
from PIL import Image

load_dotenv()
logger = logging.getLogger(__name__)

app = Flask(__name__)

SAVE_DIR = "downloads"
DEWARPED_DIR = "dewarped"
SHEETS_LOGGING_URL = os.getenv("SHEETS_LOGGING_URL")

DOWNLOADS_PATH = os.getenv("DOWNLOADS_PATH")
DEWARPED_PATH = os.getenv("DEWARPED_PATH")
DEBUG_PATH = os.getenv("DEBUG_PATH")
CHECKED_PATH = os.getenv("CHECKED_PATH")
LOGS_PATH = os.getenv("LOGS_PATH")
SERVER_IP = os.getenv("SERVER_IP")

# setup logging. Logs to a new file every time a message is received (webhook is called)
def setup_logging(session_id):
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

    logger = logging.getLogger(__name__)
    logger.info(f"New session started. Logging to {log_path}")
    return log_path

def log_to_sheet(sender, fileURL, debugURL, checkedURL, marked, score, logURL):
    payload = {
        "sender": sender,
        "fileURL": fileURL,
        "debugURL": debugURL,
        "checkedURL": checkedURL,
        "marked": marked,
        "score": score,
        "logURL": logURL
    }
    logger.info(f"Google Sheet Logging Payload: {payload}")
    requests.post(SHEETS_LOGGING_URL, json=payload, headers={"Content-Type": "application/json"})

def handle_message(data, session_id):
    logURL = f"http://{SERVER_IP}:3000/logs/{session_id}.log"
    try:
        logger.info("Received:", data)

        messages = data.get("whatsapp", {}).get("messages", [])
        for message in messages:
            fromNo = message.get("from")
            callback_type = message.get("callback_type")

            logger.info(f"Received message from {fromNo}")

            # filter out the non-incoming messages (delivery, status messages)
            if callback_type and callback_type != "incoming_message":
                logger.info("Received delivery message. Continuing...")
                continue
            
            content = message.get("content", {})

            if not content:
                logger.info("No content found. Skipping...")
                continue
            
            if content.get("type") == "image" and "image" in content:
                img = content["image"]
                url = img.get("url") or img.get("link")
                if not url:
                    continue

                r = requests.get(url, stream=True, timeout=30)
                ext = r.headers.get("Content-Type", "image/jpeg").split("/")[-1]
                filename = f"{session_id}_{message.get('from','unknown')[1:]}.{ext}"
                fileURL = f"http://{SERVER_IP}:3000/files/{filename}"

                filepath = os.path.join(SAVE_DIR, filename)
                with open(filepath, "wb") as f:
                    for chunk in r.iter_content(1024):
                        f.write(chunk)

                logger.debug(f"Saved image: {filepath}")

                sendmessage.sendMessage(fromNo, "Processing... ⏳")

                # detections from image
                corner_tags = apriltags.detect_tags_36h11(filepath)

                # detected_tags = list(map(lambda x:x.tag_id, corner_tags))
                corner_tag_ids = [x.tag_id for x in corner_tags]
                logger.debug(f"Detected tags: {corner_tag_ids}")

                # corner tags (36h11) should be 1, 2, 3, 4
                # question tags (25h9) should be 1, 2, 3, ..., 10
                if (len(corner_tags) == 4):

                    # sort the detections in local clockwise
                    corner_tags = tags.sort_detections_clockwise(corner_tags)
                    corner_tag_ids = [x.tag_id for x in corner_tags]
                    logger.debug(f"Clockwise tag_ids: {[[x.tag_id, x.center] for x in corner_tags]}")

                    # lookup the worksheet in database and get the correct order of the tags
                    worksheet_id, corner_tags = tags.detect_orientation_and_decode(corner_tags)
                    corner_tag_ids = [x.tag_id for x in corner_tags]
                    logger.debug(f"Worksheet ID: {worksheet_id}, tag_ids: {corner_tag_ids}")

                    # dewarp the image and save the dewarped image. Also preprocess it.
                    cropped_img = image.dewarp_omr(filepath, corner_tags)
                    dewarped_img = image.clean_document(cropped_img)
                    debug_img = dewarped_img.copy()
                    logger.info("Dewarped image.")

                    # checked image - this is a PIL image, not a cv2 image
                    checked_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))

                    dewarped_filename = f"{Path(filepath).stem}_dewarped.jpg"
                    dewarped_filepath = os.path.join(DEWARPED_DIR, dewarped_filename)

                    cv2.imwrite(dewarped_filepath, dewarped_img)
                    logger.debug(f"Saved dewarped image to {dewarped_filepath}")

                    db = TinyDB('worksheets.json')
                    ans_key = db.get(doc_id=worksheet_id).get('answerKey')
                    logger.info(f"Answer key for worksheet {worksheet_id}: {ans_key}")
                    # ans_key = ['C', 'A', 'D', 'C', 'C', 'A', 'D', 'C', 'D', 'A', 'B', 'C', 'A', 'D', 'C', 'C', 'A', 'C', 'A', 'B']
                    answers = []

                    # detect the 25h9 tags
                    detection_25h9 = apriltags.detect_tags_25h9(dewarped_img)
                    detected_tags_25h9 = list(map(lambda x:x.tag_id, detection_25h9))
                    logger.debug(f"Detected tags: {detected_tags_25h9}")

                    # verify 25h9 tags detection
                    required = set(range(1, 11))
                    present = set(detected_tags_25h9)

                    if not required.issubset(present):
                        missing = required - present
                        logger.debug(f"Missing 25h9 tags: {missing}")

                        # 25h9 tags are missing, ask user to send image again.
                        sendmessage.sendMessage(fromNo, "Please try again. ⟳ \n फोटो परत काढा ⟳")

                    else: # if not required.issubset(present):
                        extra = present - required
                        logger.debug(f"Extra tags detected: {extra}")
                        if extra:
                            # extra tags, remove them and continue
                            detection_25h9[:] = [d for d in detection_25h9 if d.tag_id not in list(extra)]
                            detected_tags_25h9 = list(map(lambda x:x.tag_id, detection_25h9))
                            
                            logger.debug(f"Extra tags removed, new list: {detected_tags_25h9}")

                        else: # if extra:
                            # all tags correct
                            logger.info("All 25h9 tags are correct.")

                        tag_points = list(map(lambda t: tuple(map(int, t.center.tolist())), detection_25h9))
                        for i, point in enumerate(tag_points):
                            logger.debug(f"In point {i+1}.")
                            q_left_ans_key = ans_key[i*2]
                            logger.debug(f"In question {i*2+1}")
                            q_left_ans = omr_detection.detect_bubble(dewarped_img, point, omr_detection.LEFT_QUESTION_ROI, debug_img, checked_img, q_left_ans_key)

                            q_right_ans_key = ans_key[i*2+1]
                            logger.debug(f"In question {i*2+2}")
                            q_right_ans = omr_detection.detect_bubble(dewarped_img, point, omr_detection.RIGHT_QUESTION_ROI, debug_img, checked_img, q_right_ans_key)
                            answers.extend([q_left_ans, q_right_ans])
                            logger.debug(f"Q{i*2+1}: {q_left_ans}")
                            logger.debug(f"Q{i*2+2}: {q_right_ans}")
                            
                        logger.info("Finished checking.")
                    
                        logger.info(answers)

                        # save debug image
                        debug_filename = f'debug_{Path(filepath).stem}.jpg'
                        debug_filepath = os.path.join(DEBUG_PATH, debug_filename)
                        cv2.imwrite(debug_filepath, debug_img)
                        logger.debug(f"Saved debug image at {debug_filepath}")

                        # save checked image
                        checked_filename = f'checked_{Path(filepath).stem}.jpg'
                        checked_filepath = os.path.join(CHECKED_PATH, checked_filename)
                        checked_URL = f"http://{SERVER_IP}:3000/checked/{checked_filename}"

                        # add the marks circle at the top of the checked image
                        check_circle = omr_detection.make_circle_mark(score, len(ans_key))
                        checked_img.paste(check_circle, (100, 50), check_circle)
                        # cv2.imwrite(checked_filepath, checked_img)
                        checked_img.save(checked_filepath)
                        logger.debug(f"Saved checked image at {checked_filepath} using PIL.")

                        debugURL = f"http://{SERVER_IP}:3000/debug/{debug_filename}"

                        # send message with reply
                        # sendmessage.sendMessage(fromNo, "Your answers:\n"+'\n '.join(f"{i}. {item}" for i, item in enumerate(answers, start=1)))
                        # calculate and send score
                        score = check_results(answers, ans_key)

                        sendmessage.sendMessage(fromNo, f"Your marks: {score}/{len(ans_key)} \n तुमचे मार्क: {score}/{len(ans_key)}")

                        # send visual checked paper
                        logger.info("Sending checked image.")
                        sendmessage.sendImage(fromNo, checked_URL)

                        # log successful scan to google sheet
                        logger.debug(f"Logging {fromNo}, {fileURL}, {debugURL}, {json.dumps(answers)}, {score}")
                        log_to_sheet(fromNo, fileURL, debugURL, checked_URL, json.dumps(answers), score, logURL)

                else: # if len(corner_tags) == 4
                    logging.debug("Less/more than 4 tags found.")
                    sendmessage.sendMessage(fromNo, "Please take a complete photo of the worksheet. ⟳ \n कृपया कार्यपत्रिकेचा संपूर्ण फोटो काढा. ⟳")

                    # log failed scan to google sheet
                    log_to_sheet(fromNo, fileURL, "", "", "failed", "", logURL)

            else: # if content.get("type") == "image" and "image" in content:
                sendmessage.sendMessage(fromNo, "Please send an image of a scanned worksheet. \n कृप्या केवळ कार्यपत्रिकेचा फोटो काढा.")

                # log failed scan (user message does not contain image) to google sheet
                log_to_sheet(fromNo, "none", "", "", "failed", "", logURL)
    except Exception as e:
        logging.exception("Error in background thread: ", e)


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