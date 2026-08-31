"""
Grading Service - Handles OMR processing, bubble detection, and answer grading.

This module contains functions for processing Optical Mark Recognition (OMR)
answers from worksheet images, detecting bubbles, and grading them against answer keys.
"""

import logging
from typing import Tuple, List
from pathlib import Path
import cv2  # pylint: disable=no-member
import numpy as np
from PIL import ImageDraw, ImageFont
from tinydb import TinyDB
from db import get_worksheet_json, get_answer_key, resolve_answer_key_for_template
from services.image_service import get_roi_coordinates, get_cropped_bubbles_roi
from services.communication_service import send_message, send_image
from services.logging_service import log_to_sheet
from services.inference import predict_bubble
from config import SETTINGS
from models import InputImageMeta, DetectionResult, WorksheetTemplate, ROI
from template_layouts import get_question_rois_for_template

logger = logging.getLogger(__name__)

DEBUG_PATH = SETTINGS.DEBUG_PATH
CHECKED_PATH = SETTINGS.CHECKED_PATH
SERVER_IP = SETTINGS.SERVER_IP

BUBBLES_FOLDER = Path(__file__).parent.parent / "bubbles"


def get_answer_key_for_question_slice(answer_key: List[str] | None, first_question_index: int | None, question_count: int) -> List[str]:
    """Return the relevant answer-key slice for a page.

    If the stored answer key contains fewer than the requested number of questions,
    the missing tail is simply ignored rather than causing an index error.
    """
    if not answer_key:
        return []

    start_index = max(0, (int(first_question_index or 1) - 1))
    end_index = start_index + max(0, int(question_count or 0))
    return answer_key[start_index:end_index]


def detect_bubble_inference(roi_image: InputImageMeta, debug=False, question_number=0) -> str:
    """
    Detect which bubble (A, B, C, D) is filled in a question region using inference.

    Args:
        roi_image (InputImageMeta): Metadata of the question region (ROI) image.
        debug (bool): Whether to save individual bubble images.
        question_number (int): Question number for naming saved images.
    
    Returns:
        str: The detected answer (e.g., 'A', 'B', 'C', 'D', or ''). (note: multiple filled bubbles will return '')
    """

    bubbles_marked = get_cropped_bubbles_roi(roi_image)
    option_labels = ['A', 'B', 'C', 'D']

    marked_options = []

    for b_idx, bubble in enumerate(bubbles_marked):

        bubble_prediction = predict_bubble(bubble)
        if bubble_prediction[1] == "Marked":
            marked_options.append(option_labels[b_idx])
        
        if debug:
            BUBBLES_FOLDER.mkdir(parents=True, exist_ok=True)
            bubble_image_path = BUBBLES_FOLDER / f"q{question_number:02d}_bubble_{option_labels[b_idx]}.png"
            bubble.save(bubble_image_path)
        
        logging.info("Bubble %s: %s with confidence %.2f%%. Probability: %.4f", option_labels[b_idx], bubble_prediction[1], bubble_prediction[2], bubble_prediction[3])

    if len(marked_options) == 1:
        return marked_options[0]
    else:
        return ""

def check_worksheet(worksheet_meta: WorksheetTemplate, debug=False) -> Tuple[List[str], List[int], bool]:
    """Process OMR answers using the classifier-based bubble detection flow.

    Args:
        worksheet_meta (WorksheetTemplate): Metadata of the worksheet including images and detections

    Returns:
        tuple: (answers, score, success) where success indicates if processing completed
    """
    # Load answer key from database (TinyDB)
    # try:
    #     db = TinyDB('worksheets.json')
    #     ans_key = db.get(doc_id=worksheet_meta.worksheet_id).get('answerKey')
    # except Exception as e:
    #     logger.error("Failed to load answer key for worksheet %s: %s", worksheet_meta.worksheet_id, e)
    #     return [], [], False
    
    # Load answer key from database (postgres). For basic_omr sheets, use the
    # scanned question-paper code to resolve the correct answer set instead of
    # a generic worksheet_id lookup.
    try:
        ans_key = resolve_answer_key_for_template(
            worksheet_meta.template_name,
            worksheet_meta.worksheet_id,
            getattr(worksheet_meta, "question_paper_code", None),
        )
        if ans_key is None:
            ans_key = get_answer_key(worksheet_meta.worksheet_id)
    except Exception as e:
        logger.error("Failed to load answer key from postgres for worksheet %s: %s", worksheet_meta.worksheet_id, e)
        return [], [], False

    if ans_key is None:
        logger.error("No answer key available for worksheet %s and template %s.", worksheet_meta.worksheet_id, worksheet_meta.template_name)
        return [], [], False

    first_question_index = getattr(worksheet_meta, "first_question_index", 1) or 1
    question_count = len(roi_coordinates) if 'roi_coordinates' in locals() else None
    if question_count is None:
        roi_coordinates = get_roi_coordinates(worksheet_meta.row_detections, worksheet_meta.template_name)
        question_count = len(roi_coordinates)

    ans_key = get_answer_key_for_question_slice(ans_key, first_question_index, question_count)
    worksheet_meta.answer_key = ans_key
    logger.info("Answer key for worksheet %s (template=%s, code=%s, first_question_index=%s): %s", worksheet_meta.worksheet_id, worksheet_meta.template_name, getattr(worksheet_meta, "question_paper_code", None), first_question_index, ans_key)

    # Process answers for each tag
    answers = []
    score:List[int] = []

    roi_coordinates = get_roi_coordinates(worksheet_meta.row_detections, worksheet_meta.template_name)
    logging.debug("Extracted %s ROI images for OMR processing.", len(roi_coordinates))

    # debug image, PIL setup
    debug_image_array = worksheet_meta.debug_image.image_array
    font = ImageFont.truetype("NotoSansSymbols2-Regular.ttf", 60)
    pil_draw = ImageDraw.Draw(worksheet_meta.checked_image)

    for i, roi_coordinate in enumerate(roi_coordinates):
        # roi coordinates x1, y1, x2, y2
        x1, y1, x2, y2 = roi_coordinate.x1, roi_coordinate.y1, roi_coordinate.x2, roi_coordinate.y2

        q_no = int(first_question_index) + i
        q_ans = ans_key[i] if i < len(ans_key) else ""
        logger.debug("Processing ROI %s", q_no)

        roi_image_array = worksheet_meta.blurred_image.image_array[
            y1:y2, 
            x1:x2
        ]
        roi_image = InputImageMeta(image_array=roi_image_array)

        # Save ROI image if debug enabled
        if debug:
            BUBBLES_FOLDER.mkdir(parents=True, exist_ok=True)
            roi_filename = BUBBLES_FOLDER / f"q{q_no:02d}_roi.jpg"
            cv2.imwrite(str(roi_filename), roi_image_array)

        ans = detect_bubble_inference(roi_image, debug=debug, question_number=q_no)
        answers.append(ans)
        logger.info("Detected answer for question %s: %s (Correct answer: %s)", q_no, ans, q_ans)

        # draw box around ROI based on correct or not
        if ans == q_ans:
            score.append(1)
            logger.debug("Question %s correct.", q_no)

            # draw rectangle around ROI in debug image
            cv2.rectangle(debug_image_array, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # draw rectangle around ROI in checked image and write ✔ near top-right
            pil_draw.rectangle([(x1, y1), (x2, y2)], fill=None, outline=(0, 127, 0))
            pil_draw.text((x1 + roi_coordinate.width() - 5, y1 - 5), "✔", fill=(0, 127, 0), font=font)
        elif ans == '':
            score.append(0)
            logger.debug("Question %s unanswered or multiple bubbles detected.", q_no)

            # draw rectangle around ROI in debug image
            cv2.rectangle(debug_image_array, (x1, y1), (x2, y2), (255, 86, 86), 2)

            # draw rectangle around ROI in checked image and write ? near top-right
            pil_draw.rectangle([(x1, y1), (x2, y2)], fill=None, outline=(255, 86, 86))
            pil_draw.text((x1 + roi_coordinate.width() - 5, y1 - 5), "?", fill=(255, 86, 86), font=font)
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
def show_roi_zones(points, debug_image, template_name: str | None = None):
    """Show ROI zones for debugging purposes.

    Args:
        points: Tag center points
        debug_image: Debug image to draw on
        template_name: Template whose ROI geometry is being displayed
    """
    question_rois = get_question_rois_for_template(template_name)
    for point in points:
        (point_x, point_y) = point

        for i, roi in enumerate(question_rois):
            (rx, ry, rw, rh) = roi
            x1 = point_x + rx
            y1 = point_y + ry
            x2 = x1 + rw
            y2 = y1 + rh

            color = (255, 0, 0)
            if i == 1:
                color = (0, 255, 0)
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, 2)

