"""
Message Service - Handles WhatsApp message validation and processing.

This module contains functions for validating incoming WhatsApp messages
and processing them through the grading pipeline.
"""

import logging
import threading
import requests
from pathlib import Path
from services.image_service import (
    scan_image, download_image, detect_orientation_and_decode, save_preprocessed
)
from services.grading_service import process_omr_answers, handle_results
from services.logging_service import log_to_sheet
from services.communication_service import send_message, is_valid_image_message
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
                    worksheet_meta = scan_image(worksheet.input_image)
                except ValueError as e:
                    # Corner tags detection failed
                    logger.debug("Less/more than 4 corner tags found.")
                    send_message(
                        from_no,
                        "Please take a complete photo of the worksheet. ⟳ \n"
                        "कृपया कार्यपत्रिका संपूर्ण दिसेल असा फोटो काढा. ⟳")

                # Save preprocessed image to correct file path for later access
                # worksheet.preprocessed_image = save_preprocessed(worksheet.preprocessed_image)

                # TODO - function to save all images in the correct place

                # Sort detections clockwise and decode worksheet
                # corner_tags = sort_detections_clockwise(corner_tags)
                # corner_tag_ids = [x.tag_id for x in corner_tags]
                # logger.debug("Clockwise tag_ids: %s", [
                #                 [x.tag_id, x.center] for x in corner_tags])

                # worksheet_id = detect_orientation_and_decode(corner_detection)
                # logger.debug("Worksheet ID: %s, tag_ids: %s",
                #                 worksheet_id, corner_detection.sorted_detections)

                # Process OMR answers
                answers, ans_key, omr_success = process_omr_answers(
                    dewarped_img, debug_img, checked_img, worksheet_id)

                if omr_success:
                    # Handle successful results
                    handle_results(
                        filepath, answers, ans_key, debug_img, checked_img,
                        from_no, file_url, log_url)
                else:
                    # OMR failed - missing question tags
                    send_message(
                        from_no, "Please try again. ⟳ \n फोटो परत काढा. ⟳")

                    # Log failed scan
                    logsheet_args = (from_no, file_url, "",
                                     "", "failed", "", log_url)
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
