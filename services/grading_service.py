"""
Grading Service - Handles OMR processing, bubble detection, and answer grading.

This module contains functions for processing Optical Mark Recognition (OMR)
answers from worksheet images, detecting bubbles, and grading them against answer keys.
"""

import logging
from typing import Tuple, List
import cv2  # pylint: disable=no-member
import numpy as np
from PIL import ImageDraw, ImageFont
from tinydb import TinyDB
from services.image_service import detect_tags_25h9, get_roi_coordinates
from services.communication_service import send_message, send_image
from services.logging_service import log_to_sheet
from config import SETTINGS
from models import InputImageMeta, DetectionResult, WorksheetTemplate, ContourData, ROI

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


def check_worksheet(worksheet_meta: WorksheetTemplate) -> Tuple[List[str], List[int], bool]:
    """Process OMR answers using 25h9 tags. Also draw on debug, checked images.

    Args:
        dewarped_img: Processed image for OMR
        debug_img: Image for debug visualization
        checked_img: PIL image for marking answers
        worksheet_id: ID to lookup answer key

    Returns:
        tuple: (answers, score, success) where success indicates if processing completed
    """
    # Load answer key from database
    try:
        db = TinyDB('worksheets.json')
        ans_key = db.get(doc_id=worksheet_meta.worksheet_id).get('answerKey')
    except Exception as e:
        logger.error("Failed to load answer key for worksheet %s: %s", worksheet_meta.worksheet_id, e)
        return [], False

    worksheet_meta.answer_key = ans_key
    logger.info("Answer key for worksheet %s: %s", worksheet_meta.worksheet_id, ans_key)

    # Process answers for each tag
    answers = []
    score:List[int] = []

    roi_coordinates = get_roi_coordinates(worksheet_meta.row_detections_sorted)
    logging.debug("Extracted %s ROI images for OMR processing.", len(roi_coordinates))

    # debug image, PIL setup
    debug_image_array = worksheet_meta.debug_image.image_array
    font = ImageFont.truetype("NotoSansSymbols2-Regular.ttf", 60)
    pil_draw = ImageDraw.Draw(worksheet_meta.checked_image)

    for i, roi_coordinate in enumerate(roi_coordinates):
        # roi coordinates x1, y1, x2, y2
        x1, y1, x2, y2 = roi_coordinate.x1, roi_coordinate.y1, roi_coordinate.x2, roi_coordinate.y2

        q_no = i + 1
        q_ans = ans_key[i]
        logger.debug("Processing ROI %s", q_no)

        roi_image_array = worksheet_meta.preprocessed_image.image_array[
            y1:y2, 
            x1:x2
        ]

        ans, all_contours, bubble_candidates = detect_bubble(InputImageMeta(image_array=roi_image_array))

        answers.append(ans)
        logger.debug("Detected answer for question %s: %s (Correct answer: %s)", q_no, ans, q_ans)

        # draw all contours on debug image for visualization
        for idx, contour_meta in enumerate(all_contours):
            cnt = contour_meta.contour
            cnt_global = contour_meta.get_global_contour(roi_coordinate.x1, roi_coordinate.y1)
            cv2.drawContours(debug_image_array, [cnt_global], -1, (0, 0, 255), 1)

            # label the contour for debugging
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"]) + x1
                cY = int(M["m01"] / M["m00"]) + y1
            else:
                x, y, w, h = cv2.boundingRect(cnt)
                cX = (x + w // 2) + x1
                cY = (y + h // 2) + y1

            cv2.putText(
                debug_image_array,
                f"{idx+1}",
                (cX, cY),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2) # Red text
        
        # draw and label bubble candidates as A, B, C, and D
        for idx, contour_meta in enumerate(bubble_candidates):
            x, y, w, h = cv2.boundingRect(contour_meta.contour)

            color = (0, 255, 0) if chr(65+idx) == q_ans else (255, 86, 86)
            cnt_global = contour_meta.contour + np.array([[[x1, y1]]])

            cv2.drawContours(debug_image_array, [cnt_global], -1, color, 2)
            cv2.putText(debug_image_array,
                        f"{chr(65+idx)}",
                        (x1 + x, y1 + y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # draw box around ROI based on correct or not
        if ans == q_ans:
            score.append(1)
            logger.debug("Question %s correct.", q_no)

            # draw rectangle around ROI in debug image
            cv2.rectangle(debug_image_array, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # draw rectangle around ROI in checked image and write ✔ near top-right
            pil_draw.rectangle([(x1, y1), (x2, y2)], fill=None, outline=(0, 127, 0))
            pil_draw.text((x1 + roi_coordinate.width() - 5, y1 - 5), "✔", fill=(0, 127, 0), font=font)
        else:
            score.append(0)
            logger.debug("Question %s incorrect.", q_no)

            # draw rectangle around ROI in debug image
            cv2.rectangle(debug_image_array, (x1, y1), (x2, y2), (255, 86, 86), 2)

            # draw rectangle around ROI in checked image and write X near top-right
            pil_draw.rectangle([(x1, y1), (x2, y2)], fill=None, outline=(255, 86, 86))
            pil_draw.text((x1 + roi_coordinate.width() - 5, y1 - 5), "✘", fill=(255, 86, 86), font=font)

    # save score
    worksheet_meta.score = score
    worksheet_meta.marked_answers = answers

    logger.info("Finished checking answers.")
    return answers, score, True

# def handle_results(worksheet_meta: WorksheetTemplate) -> None:
#     """Handle grading results: save images, send messages, and log to sheets.

#     Args:
#         filepath: Original image path
#         answers: Detected answers list
#         ans_key: Correct answer key
#         debug_img: Debug visualization image
#         checked_img: PIL image with marked answers
#         from_no: Sender number
#         file_url: URL of original file
#         log_url: URL of log file
#     """
#     logger.info("Answers: %s", worksheet_meta.marked_answers)
#     score = worksheet_meta.score.count(1) if worksheet_meta.score else 0

#     # Save debug image
#     debug_filename = f'debug_{Path(filepath).stem}.jpg'
#     debug_filepath = os.path.join(DEBUG_PATH, debug_filename)
#     cv2.imwrite(debug_filepath, debug_img) # pylint: disable=no-member
#     logger.debug("Saved debug image at %s", debug_filepath)

#     # Save checked image with score
#     checked_filename = f'checked_{Path(filepath).stem}.jpg'
#     checked_filepath = os.path.join(CHECKED_PATH, checked_filename)
#     checked_url = f"http://{SERVER_IP}:3000/checked/{checked_filename}"

#     # Add marks circle to checked image
#     check_circle = make_circle_mark(score, len(ans_key))
#     checked_img.paste(check_circle, (100, 50), check_circle)
#     checked_img.save(checked_filepath)
#     logger.debug("Saved checked image at %s using PIL.", checked_filepath)

#     debug_url = f"http://{SERVER_IP}:3000/debug/{debug_filename}"

#     # Send results to user
#     send_message(
#         from_no,
#         f"Your marks: {score}/{len(ans_key)} \n" 
#         f"तुमचे मार्क: {score}/{len(ans_key)}")
#     logger.info("Sending checked image.")
#     send_image(from_no, checked_url)

#     # Log to Google Sheets
#     logsheet_args = (from_no, file_url, debug_url, checked_url, json.dumps(answers), score, log_url)
#     logger.debug("Logging %s", logsheet_args)
#     threading.Thread(target=log_to_sheet, args=logsheet_args).start()



# OMR Detection Functions
def show_roi_zones(points, debug_image):
    """Show ROI zones for debugging purposes.

    Args:
        points: Tag center points
        debug_image: Debug image to draw on
    """
    for point in points:
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


# def detect_bubble(worksheet_meta: WorksheetTemplate, roi_coordinates: ROI) -> str:
def detect_bubble(roi_image: InputImageMeta) -> tuple[str, list[ContourData], list[ContourData]]:
    """Detect filled bubble cropped ROI image. Also draw on debug and checked images.

    Args:
        roi_coordinates (ROI): Coordinates of the cropped ROI image for a single question

    Returns:
        str: Detected answer ('A', 'B', 'C', 'D') or '' if none/multiple detected,
        List of ContourData for all contours found,
        List of ContourData for contours that passed filtering criteria
    """

    # x1, y1, x2, y2 = roi_coordinates.x1, roi_coordinates.y1, roi_coordinates.x2, roi_coordinates.y2
    # rw = roi_coordinates.width()
    
    # debug_image = worksheet_meta.debug_image.image_array

    # logger.info("ROI coordinates: %s, %s to %s, %s.", x1, y1, x2, y2)

    # PIL setup for adding tick and cross marks
    # font = ImageFont.truetype("NotoSansSymbols2-Regular.ttf", 60)
    # pil_draw = ImageDraw.Draw(worksheet_meta.checked_image)

    # q_crop = worksheet_meta.preprocessed_image.image_array[y1:y2, x1:x2]
    q_crop = roi_image.image_array
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
    bubble_candidates: List[ContourData] = []
    all_contours: List[ContourData] = [ContourData(contour=cnt) for cnt in contours]

    for idx, contour_meta in enumerate(all_contours):
        # # draw every contour in red in debug image
        # cnt_global = cnt + np.array([[[x1, y1]]])
        # cv2.drawContours(debug_image, [cnt_global], -1, (0, 0, 255), 1)

        # # label the contour for debugging
        # M = cv2.moments(cnt)
        # if M["m00"] != 0:
        #     cX = int(M["m10"] / M["m00"]) + x1
        #     cY = int(M["m01"] / M["m00"]) + y1
        # else:
        #     x, y, w, h = cv2.boundingRect(cnt)
        #     cX = (x + w // 2) + x1
        #     cY = (y + h // 2) + y1

        # cv2.putText(
        #     debug_image,
        #     f"{idx+1}",
        #     (cX, cY),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     (0, 0, 255),
        #     2) # Red text

        area = contour_meta.area
        perimeter = contour_meta.perimeter
        circularity = contour_meta.circularity

        if perimeter == 0:
            continue
        
        logger.debug("Contour %s: area = %s, perimiter = %s, circularity = %s.",
            idx+1,
            area,
            perimeter,
            circularity)

        # contour checks: 1. area, 2. circularity
        # if there are still more than 4, check if they are
        # evenly spaced and horizontal, and remove the y outlier (todo)

        # 1. area condition
        if MIN_MARK_AREA < area < MAX_MARK_AREA:
            # 2. circularity condition
            if circularity > float(MIN_CIRCULARITY):
                bubble_candidates.append(contour_meta)

    bubble_candidates = sorted(bubble_candidates, key=lambda c: cv2.boundingRect(c.contour)[0])

    filled_index = []
    ratios = []
    # debug_crop = q_crop.copy()

    if len(bubble_candidates) == 4:
    # go through the bubble candidates to see which one is filled
        for i, contour_meta in enumerate(bubble_candidates):
            mask = np.zeros(thresh.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour_meta.contour], -1, 255, -1)

            # shrink the mask to account for the thickness of the bubble shape
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            shrunken_mask = cv2.erode(mask, kernel, iterations=2)

            total_pixels = cv2.countNonZero(shrunken_mask)
            filled_pixels = cv2.countNonZero(cv2.bitwise_and(shrunken_mask, thresh))
            fill_ratio = filled_pixels / total_pixels if total_pixels > 0 else 0
            ratios.append(fill_ratio)
            bubble_area = contour_meta.area
            bubble_circularity = contour_meta.circularity
            logger.info(
                "Bubble %s: fill_ratio = %s, area = %s, circularity = %s.",
                chr(65+i), fill_ratio, bubble_area, bubble_circularity)

            if fill_ratio > FILL_THRESHOLD:
                filled_index.append(i)

            # x, y, w, h = cv2.boundingRect(contour_meta.contour)

            # cv2.drawContours(debug_crop, [contour_meta.contour], -1, color, 2)
            # cv2.putText(debug_crop, f"{chr(65+i)}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # # global debug image
            # cnt_global = contour_meta.contour + np.array([[[x1, y1]]])
            # cv2.drawContours(debug_image, [cnt_global], -1, color, 2)
            # cv2.putText(debug_image,
            #             f"{chr(65+i)} {fill_ratio:.2f}",
            #             (x1 + x, y1 + y - 5),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # cv2.imwrite('q_detected.jpg', debug_crop)

        if not filled_index:
            logger.debug("No bubble detected as filled.")

            # # draw red box in debug image
            # cv2.rectangle(debug_image, (x1, y1), (x2, y2), (86, 86, 255), 2)

            # # draw red box in checked image and write "+0" near top-right
            # pil_draw.rectangle([(x1, y1), (x2, y2)], fill=None, outline=(255, 86, 86))
            # pil_draw.text((x1 + rw - 5, y1 - 5), "?", fill=(255, 86, 86), font=font)

            # return ''
            ans = ''
        elif len(filled_index) > 1:
            logger.debug("Multiple bubbles detected.")

            # # draw red box in debug image
            # cv2.rectangle(debug_image, (x1, y1), (x2, y2), (86, 86, 255), 2)

            # # draw red box in checked image and write "+0" near top-right
            # pil_draw.rectangle([(x1, y1), (x2, y2)], fill=None, outline=(255, 86, 86))
            # pil_draw.text((x1 + rw - 5, y1 - 5), "✘", fill=(255, 86, 86), font=font)

            # return ''
            ans = ''
        else:
            ans = chr(65+filled_index[0])
            # logger.info("Detected bubble: %s, correct ans: %s.", ans, ans_key)
            # if ans.lower() == ans_key.lower():
            #     logger.info("Correct ans.")
            #     # correct ans
            #     # draw green box in debug image
            #     cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            #     # draw green box in checked image and write "+1" near top-right
            #     # cv2.rectangle(checked_image, (x1, y1), (x2, y2), (0, 127, 0), 2)
            #     # cv2.putText(checked_image, "+1", (x1 + rw - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 127, 0), 5, cv2.LINE_AA)
            #     pil_draw.rectangle([(x1, y1), (x2, y2)], fill=None, outline=(0, 127, 0))
            #     pil_draw.text((x1 + rw - 5, y1 - 5), "✔", fill=(0, 127, 0), font=font)

            # else:
            #     # wrong ans
            #     # draw red box in debug image
            #     cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            #     # draw red box in checked image
            #     # cv2.rectangle(checked_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            #     # cv2.putText(checked_image, "+0", (x1 + rw - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, cv2.LINE_AA)
            #     pil_draw.rectangle([(x1, y1), (x2, y2)], fill=None, outline=(255, 86, 86))
            #     pil_draw.text((x1 + rw - 5, y1 - 5), "✘", fill=(255, 86, 86), font=font)

    else:
        # len(bubble_candidates) is not 4, cannot reliably detect answer
        logger.debug("%s bubble candidates detected instead of 4.", len(bubble_candidates))
        ans = ''

    return ans, all_contours, bubble_candidates