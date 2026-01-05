"""
Message Service - Handles WhatsApp message validation and processing.

This module contains functions for validating incoming WhatsApp messages
and processing them through the grading pipeline.
"""

import logging
import threading
import requests
from services.image_service import download_image, detect_and_validate_corner_tags, process_image, sort_detections_clockwise, detect_orientation_and_decode
from services.grading_service import process_omr_answers, handle_results
from services.logging_service import log_to_sheet
from services.communication_service import sendMessage
from config import SETTINGS

logger = logging.getLogger(__name__)

SERVER_IP = SETTINGS.SERVER_IP


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


def handle_message(data, session_id):
    """Process incoming WhatsApp webhook data.

    Handles the complete message processing pipeline including validation,
    image processing, OMR detection, grading, and response sending.

    Args:
        data (dict): Webhook payload from WhatsApp
        session_id (str): Unique session identifier for logging
    """
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
                threading.Thread(target=sendMessage, args=(fromNo, "Checking... ⏳ \n कार्यपत्रिका तपासत आहे... ⏳",)).start()

                # Detect and validate corner tags
                corner_tags, corner_tags_valid = detect_and_validate_corner_tags(filepath)

                if corner_tags_valid:
                    # Sort detections clockwise and decode worksheet
                    corner_tags = sort_detections_clockwise(corner_tags)
                    corner_tag_ids = [x.tag_id for x in corner_tags]
                    logger.debug("Clockwise tag_ids: %s", [[x.tag_id, x.center] for x in corner_tags])

                    worksheet_id, corner_tags = detect_orientation_and_decode(corner_tags)
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
                        sendMessage(fromNo, "Please try again. ⟳ \n फोटो परत काढा. ⟳")
                else:
                    # Corner tags detection failed
                    logger.debug("Less/more than 4 corner tags found.")
                    sendMessage(fromNo, "Please take a complete photo of the worksheet. ⟳ \n कृपया कार्यपत्रिका संपूर्ण दिसेल असा फोटो काढा. ⟳")

                    # Log failed scan
                    logsheet_args = (fromNo, fileURL, "", "", "failed", "", logURL)
                    threading.Thread(target=log_to_sheet, args=(logsheet_args)).start()
            else:
                # Handle non-image messages
                if fromNo:
                    sendMessage(fromNo, "Please send an image of a scanned worksheet. \n कृप्या केवळ कार्यपत्रिकेचा फोटो काढा.")

                # Log failed scan (user message does not contain image)
                # logsheet_args = (fromNo, "none", "", "", "failed", "", logURL)
                # threading.Thread(target=log_to_sheet, args=(logsheet_args)).start()

    except (requests.RequestException, IOError, OSError, ValueError, KeyError) as e:
        logger.exception("Error in background thread: %s", e)
    except Exception as e:  # pylint: disable=broad-except
        # Catch any other unexpected exceptions to prevent thread crashes
        logger.exception("Unexpected error in background thread: %s", e)
