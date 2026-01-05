"""
Grading Service - Handles OMR processing, bubble detection, and answer grading.

This module contains functions for processing Optical Mark Recognition (OMR)
answers from worksheet images, detecting bubbles, and grading them against answer keys.
"""

import os
import logging
import json
import threading
import cv2  # pylint: disable=no-member
import numpy as np
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tinydb import TinyDB
from services.image_service import detect_tags_25h9
from services.communication_service import sendMessage, sendImage
from services.logging_service import log_to_sheet
from config import SETTINGS
from utils.grading_utils import check_results

logger = logging.getLogger(__name__)

DEBUG_PATH = SETTINGS.DEBUG_PATH
CHECKED_PATH = SETTINGS.CHECKED_PATH
SERVER_IP = SETTINGS.SERVER_IP

# OMR Configuration
LEFT_QUESTION_ROI = SETTINGS.LEFT_QUESTION_ROI
RIGHT_QUESTION_ROI = SETTINGS.RIGHT_QUESTION_ROI
MIN_MARK_AREA = SETTINGS.MIN_MARK_AREA
MAX_MARK_AREA = SETTINGS.MAX_MARK_AREA
FILL_THRESHOLD = SETTINGS.FILL_THRESHOLD
MIN_CIRCULARITY = SETTINGS.MIN_CIRCULARITY


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
    detection_25h9 = detect_tags_25h9(dewarped_img)
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
        q_left_ans = detect_bubble(
            dewarped_img, point, LEFT_QUESTION_ROI,
            debug_img, checked_img, q_left_ans_key
        )

        # Right question
        q_right_ans_key = ans_key[i*2+1]
        logger.debug("Processing question %s.", i*2+2)
        q_right_ans = detect_bubble(
            dewarped_img, point, RIGHT_QUESTION_ROI,
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
    cv2.imwrite(debug_filepath, debug_img) # pylint: disable=no-member
    logger.debug("Saved debug image at %s", debug_filepath)

    # Save checked image with score
    checked_filename = f'checked_{Path(filepath).stem}.jpg'
    checked_filepath = os.path.join(CHECKED_PATH, checked_filename)
    checked_URL = f"http://{SERVER_IP}:3000/checked/{checked_filename}"

    # Add marks circle to checked image
    check_circle = make_circle_mark(score, len(ans_key))
    checked_img.paste(check_circle, (100, 50), check_circle)
    checked_img.save(checked_filepath)
    logger.debug("Saved checked image at %s using PIL.", checked_filepath)

    debugURL = f"http://{SERVER_IP}:3000/debug/{debug_filename}"

    # Send results to user
    sendMessage(fromNo, f"Your marks: {score}/{len(ans_key)} \n तुमचे मार्क: {score}/{len(ans_key)}")
    logger.info("Sending checked image.")
    sendImage(fromNo, checked_URL)

    # Log to Google Sheets
    logsheet_args = (fromNo, fileURL, debugURL, checked_URL, json.dumps(answers), score, logURL)
    logger.debug("Logging %s", logsheet_args)
    threading.Thread(target=log_to_sheet, args=(logsheet_args)).start()


# OMR Detection Functions
def show_roi_zones(image, points, debug_image):
    """Show ROI zones for debugging purposes.

    Args:
        image: Input image
        points: Tag center points
        debug_image: Debug image to draw on
    """
    for i, point in enumerate(points):
        (point_x, point_y) = point

        # draw left ROI
        (left_rx, left_ry, left_rw, left_rh) = LEFT_QUESTION_ROI
        left_x1 = point_x + left_rx
        left_y1 = point_y + left_ry
        left_x2 = left_x1 + left_rw
        left_y2 = left_y1 + left_rh

        # draw red rectangle on left ROI
        cv2.rectangle(debug_image, (left_x1, left_y1), (left_x2, left_y2), (255, 0, 0), 2)

        # draw right ROI
        (right_rx, right_ry, right_rw, right_rh) = RIGHT_QUESTION_ROI
        right_x1 = point_x + right_rx
        right_y1 = point_y + right_ry
        right_x2 = right_x1 + right_rw
        right_y2 = right_y1 + right_rh

        # draw red rectangle on right ROI
        cv2.rectangle(debug_image, (right_x1, right_y1), (right_x2, right_y2), (255, 0, 0), 2)


def detect_bubble(image, anchor, roi, debug_image, checked_image, ans_key):
    """Detect filled bubble given anchor point and roi.

    Args:
        image: Input image
        anchor: Anchor point coordinates
        roi: Region of interest parameters
        debug_image: Debug image for visualization
        checked_image: PIL image for marking results
        ans_key: Correct answer key

    Returns:
        str: Detected answer ('A', 'B', 'C', 'D') or '' if none/multiple detected
    """
    (rx, ry, rw, rh) = roi
    (anchor_x, anchor_y) = anchor

    x1 = anchor_x + rx
    y1 = anchor_y + ry
    x2 = x1 + rw
    y2 = y1 + rh

    logger.info("ROI coordinates: %s, %s to %s, %s.", x1, y1, x2, y2)

    # PIL setup for adding tick and cross marks
    font = ImageFont.truetype("NotoSansSymbols2-Regular.ttf", 60)
    pil_draw = ImageDraw.Draw(checked_image)

    # draw green rectangle around ROI in debug image
    # cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    q_crop = image[y1:y2, x1:x2]
    # cv2.imwrite('q_crop.jpg', q_crop)

    gray_crop = cv2.cvtColor(q_crop, cv2.COLOR_BGR2GRAY)
    gray_norm = cv2.normalize(gray_crop, None, 0, 255, cv2.NORM_MINMAX)

    thresh = cv2.adaptiveThreshold(
        gray_norm, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        55, 30
    )
    # cv2.imwrite("q_thresh.jpg", thresh)

    # Contour Detection
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logger.info("%s contours found in ROI.", len(contours))

    # array of contours
    bubble_candidates = []

    for idx, cnt in enumerate(contours):
        # draw every contour in red in debug image
        cnt_global = cnt + np.array([[[x1, y1]]])
        cv2.drawContours(debug_image, [cnt_global], -1, (0, 0, 255), 1)

        # label the contour for debugging
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"]) + x1
            cY = int(M["m01"] / M["m00"]) + y1
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            cX = (x + w // 2) + x1
            cY = (y + h // 2) + y1

        cv2.putText(debug_image, f"{idx+1}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) # Red text

        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * (area / (perimeter * perimeter))
        logger.debug("Contour %s: area = %s, perimiter = %s, circularity = %s.", idx+1, area, perimeter, circularity)

        # contour checks: 1. area, 2. circularity
        # if there are still more than 4, check if they are evenly spaced and horizontal, and remove the y outlier (todo)

        # 1. area condition
        if MIN_MARK_AREA < area < MAX_MARK_AREA:
            # 2. circularity condition
            if circularity > float(MIN_CIRCULARITY):
                bubble_candidates.append(cnt)

    bubble_candidates = sorted(bubble_candidates, key=lambda c: cv2.boundingRect(c)[0])

    filled_index = []
    ratios = []
    debug_crop = q_crop.copy()

    for i, cnt in enumerate(bubble_candidates):
        mask = np.zeros(thresh.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)

        # shrink the mask to account for the thickness of the bubble shape
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        shrunken_mask = cv2.erode(mask, kernel, iterations=2)

        total_pixels = cv2.countNonZero(shrunken_mask)
        filled_pixels = cv2.countNonZero(cv2.bitwise_and(shrunken_mask, thresh))
        fill_ratio = filled_pixels / total_pixels if total_pixels > 0 else 0
        ratios.append(fill_ratio)
        bubble_area = cv2.contourArea(cnt)
        bubble_perimeter = cv2.arcLength(cnt, True)
        bubble_circularity = 4 * math.pi * (bubble_area / (bubble_perimeter * bubble_perimeter))
        logger.info("Bubble %s: fill_ratio = %s, area = %s, circularity = %s.", chr(65+i), fill_ratio, bubble_area, bubble_circularity)

        color = (0, 255, 0)
        if fill_ratio > FILL_THRESHOLD:
            color = (255, 0, 0)
            filled_index.append(i)

        x, y, w, h = cv2.boundingRect(cnt)

        cv2.drawContours(debug_crop, [cnt], -1, color, 2)
        cv2.putText(debug_crop, f"{chr(65+i)}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # global debug image
        cnt_global = cnt + np.array([[[x1, y1]]])
        cv2.drawContours(debug_image, [cnt_global], -1, color, 2)
        cv2.putText(debug_image,
                    f"{chr(65+i)} {fill_ratio:.2f}",
                    (x1 + x, y1 + y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # cv2.imwrite('q_detected.jpg', debug_crop)

    if len(bubble_candidates) != 4:
        logger.debug("%s bubble candidates detected instead of 4.", len(bubble_candidates))

        # draw blue box in debug image
        cv2.rectangle(debug_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # draw blue box in checked image and write "+0" near top-right
        # cv2.rectangle(checked_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # cv2.putText(checked_image, "+0", (x1 + rw - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, cv2.LINE_AA)
        pil_draw.rectangle([(x1, y1), (x2, y2)], fill=None, outline=(0, 0, 200))
        pil_draw.text((x1 + rw - 5, y1 - 5), "?", fill=(0, 0, 200), font=font)

        return ''
    else:
        if not filled_index:
            logger.debug("No bubble detected as filled.")

            # draw red box in debug image
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (86, 86, 255), 2)

            # draw red box in checked image and write "+0" near top-right
            # cv2.rectangle(checked_image, (x1, y1), (x2, y2), (86, 86, 255), 2)
            # cv2.putText(checked_image, "+0", (x1 + rw - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, cv2.LINE_AA)
            pil_draw.rectangle([(x1, y1), (x2, y2)], fill=None, outline=(255, 86, 86))
            pil_draw.text((x1 + rw - 5, y1 - 5), "?", fill=(255, 86, 86), font=font)

            return ''
        elif len(filled_index) > 1:
            logger.debug("Multiple bubbles detected.")

            # draw red box in debug image
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (86, 86, 255), 2)

            # draw red box in checked image and write "+0" near top-right
            # cv2.rectangle(checked_image, (x1, y1), (x2, y2), (86, 86, 255), 2)
            # cv2.putText(checked_image, "+0", (x1 + rw - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, cv2.LINE_AA)
            pil_draw.rectangle([(x1, y1), (x2, y2)], fill=None, outline=(255, 86, 86))
            pil_draw.text((x1 + rw - 5, y1 - 5), "✘", fill=(255, 86, 86), font=font)

            return ''
        else:
            ans = chr(65+filled_index[0])
            logger.info("Detected bubble: %s, correct ans: %s.", ans, ans_key)
            if ans.lower() == ans_key.lower():
                logger.info("Correct ans.")
                # correct ans
                # draw green box in debug image
                cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # draw green box in checked image and write "+1" near top-right
                # cv2.rectangle(checked_image, (x1, y1), (x2, y2), (0, 127, 0), 2)
                # cv2.putText(checked_image, "+1", (x1 + rw - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 127, 0), 5, cv2.LINE_AA)
                pil_draw.rectangle([(x1, y1), (x2, y2)], fill=None, outline=(0, 127, 0))
                pil_draw.text((x1 + rw - 5, y1 - 5), "✔", fill=(0, 127, 0), font=font)

            else:
                # wrong ans
                # draw red box in debug image
                cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # draw red box in checked image
                # cv2.rectangle(checked_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # cv2.putText(checked_image, "+0", (x1 + rw - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, cv2.LINE_AA)
                pil_draw.rectangle([(x1, y1), (x2, y2)], fill=None, outline=(255, 86, 86))
                pil_draw.text((x1 + rw - 5, y1 - 5), "✘", fill=(255, 86, 86), font=font)
            return ans


def make_circle_mark(obtained, total, diameter=150):
    """Make a circle mark showing obtained/total marks.

    Args:
        obtained (int): Marks obtained
        total (int): Total marks
        diameter (int): Diameter of the mark

    Returns:
        PIL.Image: Circle mark image
    """
    # Canvas (RGBA so it supports transparency)
    img = Image.new("RGBA", (diameter, diameter), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    DARK_BLUE = (10, 20, 120, 255)

    # Circle border
    draw.ellipse(
        [(5, 5), (diameter-5, diameter-5)],
        outline=DARK_BLUE,
        width=7
    )

    # Horizontal line through center
    center_y = diameter // 2
    draw.line(
        [(20, center_y), (diameter-20, center_y)],
        fill=DARK_BLUE,
        width=7
    )

    # Load font
    try:
        font = ImageFont.truetype("NotoSans-Bold.ttf", 50)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # --- TOP TEXT (obtained marks) ---
    top_text = str(obtained)
    bbox = draw.textbbox((0, 0), top_text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    draw.text(
        ((diameter - tw) // 2, center_y - th - 30),
        top_text,
        fill=DARK_BLUE,
        font=font
    )

    # --- BOTTOM TEXT (total marks) ---
    bottom_text = str(total)
    bbox2 = draw.textbbox((0, 0), bottom_text, font=font)
    tw2 = bbox2[2] - bbox2[0]
    th2 = bbox2[3] - bbox2[1]

    draw.text(
        ((diameter - tw2) // 2, center_y),
        bottom_text,
        fill=DARK_BLUE,
        font=font
    )

    return img
