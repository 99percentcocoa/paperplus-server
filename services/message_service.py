"""
Message Service - Handles WhatsApp message validation and processing.

This module contains functions for validating incoming WhatsApp messages
and processing them through the grading pipeline.
"""

import logging
import threading
import json
import requests
from services.image_service import (
    scan_image, download_image, detect_orientation_and_decode, save_preprocessed, save_debug, save_checked
)
from services.grading_service import check_worksheet
from services.logging_service import log_to_sheet
from services.communication_service import send_image, send_message, is_valid_image_message
from config import SETTINGS
from models import DetectionResult, InputImageMeta, WorksheetTemplate

import cv2
from PIL import Image

logger = logging.getLogger(__name__)

SERVER_IP = SETTINGS.SERVER_IP
DEWARPED_DIR = SETTINGS.DEWARPED_PATH

def handle_message(data, session_id):
    """Process incoming WhatsApp webhook data.

    Handles the complete message processing pipeline including validation,
    image processing, OMR detection, grading, and response sending.

    Args:
        data (dict): Webhook payload from WhatsApp
        session_id (str): Unique session identifier for logging
    """
    log_url = f"http://{SERVER_IP}:3000/logs/{session_id}.log"
    try:
        logger.info("Received: %s", data)

        messages = data.get("whatsapp", {}).get("messages", [])
        for message in messages:
            from_no = message.get("from")
            logger.info("Received message from %s", from_no)

            # Validate message and extract image URL
            is_valid, image_url, _ = is_valid_image_message(message)

            if is_valid:
                logger.info("Processing valid image message from %s", from_no)

                # Download the image
                filepath, file_url = download_image(
                    image_url, session_id, from_no)

                # Send processing message
                threading.Thread(
                    target=send_message,
                    args=(from_no, "Checking... ⏳ \n कार्यपत्रिका तपासत आहे... ⏳")
                ).start()
                
                # input_image = InputImageMeta(image_path=filepath)

                # initialize worksheet with input image
                worksheet = WorksheetTemplate(input_image=InputImageMeta(image_path=filepath))

                try:
                    # image validation, tag sorting done in scan_image
                    worksheet = scan_image(worksheet.input_image)
                except ValueError as e:
                    # Corner tags detection failed
                    logger.debug("Less/more than 4 corner tags found.")
                    send_image(
                        from_no,
                        SETTINGS.HOWTO_IMAGE_URL,
                        "Please take a complete photo of the worksheet. ⟳ \n कृपया कार्यपत्रिका संपूर्ण दिसेल असा फोटो काढा. ⟳"
                    )
                    # send_message(
                    #     from_no,
                    #     "Please take a complete photo of the worksheet. ⟳ \n"
                    #     "कृपया कार्यपत्रिका संपूर्ण दिसेल असा फोटो काढा. ⟳")

                # Save preprocessed image to correct file path for later access
                save_preprocessed(worksheet)

                # debug, checked will be saved at the end, so no need to save here

                # Process OMR answers
                answers, q_score, omr_success = check_worksheet(worksheet_meta=worksheet, use_classifier=True, debug=False)
                roll_number, worksheet_id = worksheet.roll_number, worksheet.worksheet_id
                score = sum(q_score) if q_score else 0

                if omr_success:
                    # Successful checking!
                    save_debug(worksheet)
                    save_checked(worksheet)
                    send_message(
                        from_no,
                        f"Your marks: {score}/{len(answers)} \n" 
                        f"तुमचे मार्क: {score}/{len(answers)}")
                    
                    logger.info("Sending checked image.")
                    send_image(from_no, worksheet.checked_image_url, "")

                    logsheet_args = (from_no, file_url, worksheet.debug_image.image_url, worksheet.checked_image_url, json.dumps(answers), score, log_url, roll_number, worksheet_id)
                    logger.debug("Logging to Google Sheets: %s", logsheet_args)
                    threading.Thread(target=log_to_sheet, args=logsheet_args).start()

                else:
                    # OMR failed - missing question tags
                    send_message(
                        from_no, "Please try again. ⟳ \n फोटो परत काढा. ⟳")

                    # Log failed scan
                    logsheet_args = (from_no, file_url, "",
                                     "", "failed", "", log_url, "", "")
                    threading.Thread(target=log_to_sheet,
                                     args=logsheet_args).start()
            else:
                # Handle non-image messages
                if from_no:
                    send_message(
                        from_no,
                        "Please send an image of a scanned worksheet. \n"
                        "कृप्या केवळ कार्यपत्रिकेचा फोटो काढा.")

                # Log failed scan (user message does not contain image)
                # logsheet_args = (from_no, "none", "", "", "failed", "", log_url)
                # threading.Thread(target=log_to_sheet, args=(logsheet_args)).start()

    except (requests.RequestException, IOError, OSError, ValueError, KeyError) as e:
        logger.exception("Error in background thread: %s", e)
    except Exception as e:  # pylint: disable=broad-except
        # Catch any other unexpected exceptions to prevent thread crashes
        logger.exception("Unexpected error in background thread: %s", e)
