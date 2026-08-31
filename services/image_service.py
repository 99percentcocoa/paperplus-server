"""
Image Service - Handles image download, processing, AprilTag detection, and tag utilities.

This module contains functions for downloading images from URLs,
detecting AprilTags for corner positioning, processing images for OMR,
and handling tag sorting and worksheet identification.
"""

import os
import logging
from pathlib import Path
import requests
from typing import List
import hashlib
import cv2  # pylint: disable=no-member
import numpy as np
from tinydb import TinyDB
from pupil_apriltags import Detector
from PIL import Image, ImageDraw, ImageFont
from config import SETTINGS
from models import DetectionResult, InputImageMeta, WorksheetTemplate, ROI, RollNumberError
from services.inference import predict_ocr
from db.worksheets import record_worksheet_page_metadata
from template_layouts import get_handwritten_field_roi, get_question_rois_for_template, get_template_layout

logger = logging.getLogger(__name__)

SAVE_DIR = SETTINGS.DOWNLOADS_PATH
DEWARPED_DIR = SETTINGS.DEWARPED_PATH
TARGET_WIDTH = SETTINGS.TARGET_WIDTH
TARGET_HEIGHT = SETTINGS.TARGET_HEIGHT

DEBUG_PATH = SETTINGS.DEBUG_PATH
CHECKED_PATH = SETTINGS.CHECKED_PATH
SERVER_IP = SETTINGS.SERVER_IP

legacy_template = get_template_layout("regular")
LEFT_QUESTION_ROI = legacy_template.left_question_roi
RIGHT_QUESTION_ROI = legacy_template.right_question_roi
QUESTION_ROI_COLUMNS = list(legacy_template.question_roi_columns)

# AprilTag detectors
at_detector_36h11 = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.2,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

at_detector_25h9 = Detector(
    families="tag25h9",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.2,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

# Tag configuration
# BASE = 586
ORIENTATION_ID = 0
db = TinyDB('worksheets.json')


def validate_question_paper_code(raw_value: str | None) -> str:
    """Validate a question paper code OCR result.

    The field should be a single uppercase letter in A-F.
    """
    if not isinstance(raw_value, str):
        return ""

    normalized = raw_value.strip().upper()
    if len(normalized) == 1 and normalized in {"A", "B", "C", "D", "E", "F"}:
        return normalized
    return ""


def detect_apriltags(input_image: InputImageMeta, tag_family: str) -> DetectionResult:
    """Detect AprilTags in the given input image.

    Args:
        input_image (InputImageMeta): Metadata of the input image.
        tag_family (str): The family of AprilTags to detect ("36h11" or "25h9").

    Returns:
        DetectionResult: The result of the detection containing detected tags.
    """
    if input_image.image_array is None:
        raise ValueError("Input image array is None. Please load the image before detection.")

    # convert to grayscale
    gray_image_array = cv2.cvtColor(input_image.image_array, cv2.COLOR_BGR2GRAY)

    if tag_family == "36h11":
        detector = at_detector_36h11
    elif tag_family == "25h9":
        detector = at_detector_25h9
    else:
        raise ValueError(f"Unsupported tag family: {tag_family}")

    h, w = input_image.image_array.shape[:2]
    fov = 60  # degrees, typical for smartphone cameras
    focal_length = (w / 2) / np.tan(np.radians(fov / 2))
    cx = w / 2
    cy = h / 2
    detections = detector.detect(
        gray_image_array,
        estimate_tag_pose=True,
        camera_params=[focal_length, focal_length, cx, cy],
        tag_size=0.01
    )

    return DetectionResult(
        input_image=input_image,
        detections=detections,
        tag_family=tag_family
    )


def download_image(url, session_id, sender_number):
    """Download image from URL and save to disk.

    Args:
        url (str): Image URL to download
        session_id (str): Session identifier
        sender_number (str): Sender's phone number

    Returns:
        tuple: (filepath, file_url) for the downloaded image
    """
    r = requests.get(url, stream=True, timeout=30)
    ext = r.headers.get("Content-Type", "image/jpeg").split("/")[-1]
    filename = f"{session_id}_{sender_number[1:]}.{ext}"
    file_url = f"http://{SETTINGS.SERVER_IP}:3000/files/{filename}"

    filepath = os.path.join(SAVE_DIR, filename)
    with open(filepath, "wb") as f:
        for chunk in r.iter_content(1024):
            f.write(chunk)

    logger.debug("Saved image: %s", filepath)
    return filepath, file_url


# def detect_and_validate_corner_tags(input_image_meta: InputImageMeta):
#     """Detect AprilTags for corner positioning and validate detection.

#     Args:
#         filepath (str): Path to the image file

#     Returns:
#         tuple: (corner_tags, success) where success indicates if exactly 4 tags found
#     """
#     # Detect corner tags (36h11)
#     detection_36h11 = detect_apriltags(input_image_meta, "36h11")

#     if len(corner_tags) < 4:
#         # Try processing again in case of faint printing
#         logger.info("Less than 4 corner tags detected. Reprocessing image for better detection.")
#         faint_preprocessed_img = faint_preprocess(filepath)
#         corner_tags = detect_tags_36h11(faint_preprocessed_img)

#     corner_tag_ids = [x.tag_id for x in corner_tags]
#     logger.debug("Detected corner tags: %s", corner_tag_ids)

    return corner_tags, len(corner_tags) == 4

def apply_median_blur(input_image: InputImageMeta, kernel_size: int = 31) -> InputImageMeta:
    """Apply median blur to the input image.

    Args:
        input_image (InputImageMeta): Metadata of the input image
        kernel_size (int): Size of the median filter kernel

    Returns:
        InputImageMeta: Metadata of the blurred image
    """
    blurred_image = cv2.medianBlur(input_image.image_array, kernel_size)
    return InputImageMeta(image_array=blurred_image)

def scan_image(input_image: InputImageMeta) -> WorksheetTemplate:
    """Process image: dewarp and prepare for OMR.

    Args:
        input_image (InputImageMeta): Metadata of the original image

    Returns:
        worksheet_template: (WorksheetTemplate) containing metadata of cropped image, debug image and detections.
        cropped_image: (InputImageMeta) Metadata of the cropped image.
        corner_detections: (DetectionResult) Result of AprilTag detections.
    """

    corner_detection_result = detect_apriltags(input_image, "36h11")
    cropped_image = crop_image(input_image, corner_detection_result)
    blurred_image = apply_median_blur(cropped_image)
    row_detection_result = detect_apriltags(cropped_image, "25h9")

    row_tags = [tag.tag_id for tag in row_detection_result.detections]
    row_metadata = decode_row_tag_metadata(row_tags)
    worksheet_id = row_metadata.get("worksheet_id")
    page_no = row_metadata.get("page_no")
    first_question_index = row_metadata.get("first_question_index")

    template_name = "regular"
    if worksheet_id is not None:
        try:
            from db.worksheets import get_worksheet_json
            worksheet_json = get_worksheet_json(worksheet_id)
            if isinstance(worksheet_json, dict):
                template_name = (
                    worksheet_json.get("template_name")
                    or worksheet_json.get("template_type")
                    or worksheet_json.get("worksheet_template")
                    or "regular"
                )
        except Exception:  # pragma: no cover - defensive fallback
            logger.debug("Could not resolve template name for worksheet %s from DB", worksheet_id, exc_info=True)

    layout = get_template_layout(template_name)
    roll_number_roi_spec = get_handwritten_field_roi(template_name, "roll_number")
    if roll_number_roi_spec is None:
        raise RollNumberError(f"No roll number ROI configured for template '{template_name}'.")

    x1, y1, x2, y2 = roll_number_roi_spec
    roll_number_roi = cropped_image.image_array[y1:y2, x1:x2]
    if roll_number_roi.size == 0 or roll_number_roi.shape[0] == 0 or roll_number_roi.shape[1] == 0:
        raise RollNumberError(f"Roll number ROI is empty (shape={roll_number_roi.shape}); cannot run OCR.")
    roll_number_roi_meta = InputImageMeta(image_array=roll_number_roi)
    roll_number = predict_ocr(roll_number_roi_meta)

    if not (isinstance(roll_number, str) and roll_number.isdigit() and len(roll_number) == 4):
        raise RollNumberError(f"Detected roll number '{roll_number}' is not a valid 4-digit number.")

    logger.debug("Roll number detected: %s", roll_number)

    paper_code = ""
    paper_code_roi_spec = get_handwritten_field_roi(template_name, "question_paper_code")
    if paper_code_roi_spec is not None:
        p_x1, p_y1, p_x2, p_y2 = paper_code_roi_spec
        paper_code_roi = cropped_image.image_array[p_y1:p_y2, p_x1:p_x2]
        if paper_code_roi.size > 0 and paper_code_roi.shape[0] > 0 and paper_code_roi.shape[1] > 0:
            raw_paper_code = predict_ocr(InputImageMeta(image_array=paper_code_roi)) or ""
            paper_code = validate_question_paper_code(raw_paper_code)
            logger.debug("Paper code OCR returned %r; validated value: %r", raw_paper_code, paper_code)

    debug_image = cropped_image

    checked_image = Image.fromarray(cv2.cvtColor(cropped_image.image_array, cv2.COLOR_BGR2RGB))  # pylint: disable=no-member


    worksheet_template = WorksheetTemplate(
        input_image=input_image,
        cropped_image=cropped_image,
        blurred_image=blurred_image,
        corner_detections=corner_detection_result,
        row_detections=row_detection_result,
        worksheet_id=worksheet_id,
        template_name=template_name,
        page_no=page_no,
        first_question_index=first_question_index,
        page_metadata=row_metadata,
        debug_image=debug_image,
        checked_image=checked_image,
        roll_number=roll_number,
        question_paper_code=paper_code
    )

    record_worksheet_page_metadata(
        worksheet_id=worksheet_id,
        page_no=page_no,
        first_question_index=first_question_index,
        page_metadata=row_metadata,
        expected_row_tag_count=13 if row_metadata.get("format") == "omr_v2" else 10,
    )

    return worksheet_template


# AprilTag Detection Functions
def detect_tags_36h11(image_input):
    """Detect AprilTags using 36h11 family.

    Args:
        image_input (str or np.ndarray): File path or image array

    Returns:
        list: List of detected tag objects
    """
    # case 1: image input is a file path
    if isinstance(image_input, str):
        img = cv2.imread(image_input)  # type: ignore[attr-defined]
    # case 2: image input is an opencv image array
    elif isinstance(image_input, np.ndarray):
        img = image_input
    else:
        raise ValueError("image_input must be a file path (str) or numpy array")
    
    # estimate camera params (fx, fy, cx, cy) based on image size and typical smartphone camera FOV
    h, w = img.shape[:2]
    fov = 60  # degrees, typical for smartphone cameras
    focal_length = (w / 2) / np.tan(np.radians(fov / 2))
    cx = w / 2
    cy = h / 2
    fx = fy = focal_length

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detection = at_detector_36h11.detect(img=gray_img, estimate_tag_pose=True, camera_params=[fx, fy, cx, cy], tag_size=0.01)
    return detection

def detect_tags_25h9(image_input):
    """Detect AprilTags using 25h9 family.

    Args:
        image_input (str or np.ndarray): File path or image array

    Returns:
        list: List of detected tag objects
    """
    # case 1: image input is a file path
    if isinstance(image_input, str):
        img = cv2.imread(image_input)  # type: ignore[attr-defined]
    # case 2: image input is an opencv image array
    elif isinstance(image_input, np.ndarray):
        img = image_input
    else:
        raise ValueError("image_input must be a file path (str) or numpy array")

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detection = at_detector_25h9.detect(gray_img)
    return detection

# crop image using corner tags
def crop_image(input_image: InputImageMeta, detections: DetectionResult) -> tuple[InputImageMeta, int]:
    """Crop the input image using the detected AprilTags.

    Args:
        input_image (InputImageMeta): Metadata of the input image.
        detections (DetectionResult): Detected AprilTags in the image.

    Returns:
        InputImageMeta: Metadata of the cropped image.
        Worksheet ID: Detected worksheet ID.
    """
    if input_image.image_array is None:
        raise ValueError("Input image is empty; cannot crop.")
    
    # worksheet_id, detections.sorted_corner_detections = detect_orientation_and_decode(detections)
    detections.sorted_corner_detections = detect_orientation_and_decode(detections)


    # Build source points (x, y) in float32 shape (4,2)
    src_pts = np.array([detections.sorted_corner_detections[i].center for i in range(4)], dtype=np.float32)
    dst_pts = np.array([[0, 0], [TARGET_WIDTH, 0], [TARGET_WIDTH, TARGET_HEIGHT], [0, TARGET_HEIGHT]], dtype="float32")

    # Compute perspective transform matrix
    t_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Perform the warp perspective to get the cropped image
    warped_image = cv2.warpPerspective(input_image.image_array, t_matrix, (TARGET_WIDTH, TARGET_HEIGHT))

    return InputImageMeta(image_array=warped_image)

# get cropped ROI images from worksheet
def get_roi_coordinates(row_detections: DetectionResult, template_name: str | None = None) -> list[ROI]:
    """Get ROI coordinates for each question, template-aware.

    Args:
        row_detections: Sorted detections of AprilTags representing rows
        template_name: Name of the template whose ROI geometry should be used

    Returns:
        List of ROI objects for each question's ROI
    """
    roi_coordinates = []
    question_rois = get_question_rois_for_template(template_name)

    logger.debug("Row detections for ROI cropping: %s", [d.tag_id for d in row_detections.detections])

    for i, detection in enumerate(row_detections.detections):
        logger.debug("In detection %d", i)
        anchor_x, anchor_y = detection.center

        for roi in question_rois:
            (rx, ry, rw, rh) = roi
            x1 = int(anchor_x + rx)
            y1 = int(anchor_y + ry)
            x2 = int(x1 + rw)
            y2 = int(y1 + rh)

            logger.info("ROI coordinates: %s, %s to %s, %s.", x1, y1, x2, y2)
            roi_coordinates.append(ROI(x1, y1, x2, y2))

    return roi_coordinates

def get_cropped_bubbles_roi(input_image: InputImageMeta) -> List[InputImageMeta]:
    """
    Get individual bubble images from an ROI image.

    Args:
        input_image (InputImageMeta): Metadata of the ROI input image.
    
    Returns:
        List[InputImageMeta]: List of bubble images extracted from the ROI.
    """

    roi_array = input_image.image_array
    roi_width = roi_array.shape[1]
    part_width = roi_width // 4
    roi_bubbles: List[InputImageMeta] = []
    for i in range(4):
        start_x = i * part_width
        end_x = start_x + part_width if i < 3 else roi_width
        roi_part = InputImageMeta(image_array=roi_array[:, start_x:end_x].copy())
        roi_bubbles.append(roi_part)

    return roi_bubbles


def faint_preprocess(fp):
    """Preprocessing for faint AprilTag detection.

    Args:
        fp (str): File path to image

    Returns:
        np.ndarray: Preprocessed image
    """
    img = cv2.imread(fp)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(gray)
    th = cv2.adaptiveThreshold(
        cl, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 81, 10
    )
    color_img = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    return color_img


def preprocess(img):
    """General preprocessing pipeline for scanned sheets.

    Args:
        img (np.ndarray): Input image

    Returns:
        np.ndarray: Preprocessed image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35,35))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    diff = cv2.absdiff(gray, background)
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    norm_inv = cv2.bitwise_not(norm)

    # convert back to BGR
    color_img = cv2.cvtColor(norm_inv, cv2.COLOR_GRAY2BGR)
    return color_img


# Tag Utility Functions
def sort_detections_clockwise(detections):
    """Arrange AprilTag detections in clockwise order.

    Args:
        detections (list): List of AprilTag detection objects

    Returns:
        list: Detections sorted clockwise starting from top-left
    """
    # Extract centers as (x, y)
    centers = np.array([d.center for d in detections])
    ids = [d.tag_id for d in detections]

    # Compute centroid of all tag centers
    cx, cy = np.mean(centers, axis=0)

    # Compute angles of each point relative to centroid
    # atan2(y - cy, x - cx) gives angle from x-axis
    angles = np.arctan2(centers[:,1] - cy, centers[:,0] - cx)

    # Sort by angle (clockwise)
    # Note: atan2 gives counterclockwise order by default,
    # so we sort descending for clockwise
    sorted_indices = np.argsort(angles)

    # Reorder detections and IDs
    detections_sorted = [detections[i] for i in sorted_indices]
    ids_sorted = [ids[i] for i in sorted_indices]
    logger.debug("Detected IDs: %s", ids_sorted)

    return detections_sorted

# encoding for row_ids
def encode_worksheet_id_rows(n: int):
    """Encode worksheet ID into three tag IDs for rows. Returns 5 tags (the first 5 row tags of the worksheet)"""
    digits = []

    for _ in range(5):
        digits.append(n % 35)
        n //= 35

    digits.reverse()
    return digits

def checksum(ids):
    """Calculate checksum for a list of 5 tag IDs."""
    digest = hashlib.sha256(bytes(ids)).digest()
    return [b % 35 for b in digest[:5]]

def worksheet_id_to_rows(n: int):
    data_tags = encode_worksheet_id_rows(n)
    check = checksum(data_tags)
    return data_tags + check

def decode_row_tag_metadata(tags):
    """Decode the worksheet ID and optional OMR page metadata from row tags.

    Legacy format: exactly 10 tags = 5 data + 5 checksum
    New format: exactly 13 tags = 10 legacy packet + 3 OMR metadata values
    """
    if len(tags) == 10:
        data = tags[:5]
        check = tags[5:]
        if checksum(data) != check:
            return {"worksheet_id": None, "page_no": None, "first_question_index": None, "format": "legacy"}
        value = 0
        for d in data:
            value = value * 35 + d
        return {"worksheet_id": value, "page_no": 1, "first_question_index": 1, "format": "legacy"}

    if len(tags) == 13:
        legacy_data = tags[:5]
        legacy_check = tags[5:10]
        if checksum(legacy_data) != legacy_check:
            return {"worksheet_id": None, "page_no": None, "first_question_index": None, "format": "omr_v2"}

        value = 0
        for d in legacy_data:
            value = value * 35 + d

        page_no = int(tags[10]) if 0 <= tags[10] <= 9 else 1
        first_question_index = int(tags[11]) if 0 <= tags[11] <= 999 else 1
        return {
            "worksheet_id": value,
            "page_no": page_no,
            "first_question_index": first_question_index,
            "format": "omr_v2",
        }

    raise ValueError("Expected 10 tags (legacy) or 13 tags (OMR v2 metadata format)")


def decode_row_tags(tags):
    """Decode worksheet ID from the legacy 10-tag layout or the newer 13-tag layout.

    Returns the worksheet_id for both formats. Invalid checksums return None.
    """
    metadata = decode_row_tag_metadata(tags)
    return metadata.get("worksheet_id")

# def encode_worksheet_id(n: int):
#     """Return tag IDs for TR, BR, BL given worksheet_id n.

#     Args:
#         n (int): Worksheet ID

#     Returns:
#         list: [TR, BR, BL] tag IDs

#     Raises:
#         ValueError: If worksheet_id is too large
#     """
#     if n >= BASE ** 3:
#         raise ValueError(f"Max worksheet_id is {BASE**3 - 1}")
#     ids = []
#     for _ in range(3):
#         ids.append(n % BASE)
#         n //= BASE
#     return ids  # [TR, BR, BL]


# def decode_from_tags(tr: int, br: int, bl: int):
#     """Return worksheet_id from three tag IDs.

#     Args:
#         tr (int): Top-right tag ID
#         br (int): Bottom-right tag ID
#         bl (int): Bottom-left tag ID

#     Returns:
#         int: Worksheet ID
#     """
#     return tr + br * BASE + bl * (BASE ** 2)


def rotate(lst, n):
    """Rotate list by n positions (clockwise).

    Args:
        lst (list): List to rotate
        n (int): Number of positions to rotate

    Returns:
        list: Rotated list
    """
    return lst[-n:] + lst[:-n]


def detect_orientation_and_decode(detection: DetectionResult):
    """Detect worksheet orientation and decode worksheet ID.

    Args:
        detection (list): List of 4 AprilTag detections in clockwise order

    Returns:
        tuple: (worksheet_id, rotated_detections) or (None, None) if not found
    """

    num_rotations = 0
    for rot in range(4):
        rotated = rotate(detection.sorted_corner_detections, rot)
        tag_ids = [d.tag_id for d in rotated]
        num_rotations += 1
        logger.debug(f"At rotation {num_rotations}, tag IDs: {tag_ids}")
        if tag_ids[0] == ORIENTATION_ID:        # TL found
            return rotated
        else:
            logger.debug(f"Orientation ID {ORIENTATION_ID} not found at rotation {num_rotations}.")

    # num_rotations = 0
    # for rot in range(4):
    #     # rot starts with 0
    #     rotated = rotate(detection.sorted_corner_detections, rot)
    #     tag_ids = [d.tag_id for d in rotated]
    #     num_rotations += 1
    #     print(f"At rotation {num_rotations}")
    #     if tag_ids[0] == ORIENTATION_ID:        # TL found

    #         worksheet_id = decode_from_tags(tag_ids[1], tag_ids[2], tag_ids[3])
    #         print(f"Scanned worksheet ID: {worksheet_id}")
    #         # return (worksheet_id, rotated)

    #         # check if worksheet id is in database
    #         if db.contains(doc_id=worksheet_id):
    #             print(
    #                 f"Found worksheet id {worksheet_id}: "
    #                 f"{db.get(doc_id=worksheet_id).get('name', '')}"
    #             )
    #             detection.sorted_corner_detections = rotated
    #             return (worksheet_id, rotated)
    #         else:
    #             print(f"Worksheet ID {worksheet_id} not found in database.")
    #             return None
    return None  # some error

def save_debug(worksheet_meta: WorksheetTemplate) -> None:
    """Save debug image (already contained in WorksheetTemplate) to DEWARPED_PATH with modified filename.

    Args:
        worksheet_meta (WorksheetTemplate): Metadata of the worksheet.
    
    Returns:
        None
    """
    original_path = worksheet_meta.input_image.image_path
    debug_filename = f"{Path(original_path).stem}_debug.jpg"
    debug_filepath = Path(SETTINGS.DEBUG_PATH) / debug_filename
    debug_url = f"http://{SERVER_IP}:3000/debug/{debug_filename}"

    # Draw the template-specific handwritten ROIs for debugging.
    roll_number_roi = get_handwritten_field_roi(worksheet_meta.template_name, "roll_number")
    debug_arr = worksheet_meta.debug_image.image_array.copy()
    if roll_number_roi is not None:
        x1, y1, x2, y2 = roll_number_roi
        cv2.rectangle(debug_arr, (x1, y1), (x2, y2), (0, 255, 0), 3)

    question_paper_code_roi = get_handwritten_field_roi(worksheet_meta.template_name, "question_paper_code")
    if question_paper_code_roi is not None:
        x1, y1, x2, y2 = question_paper_code_roi
        cv2.rectangle(debug_arr, (x1, y1), (x2, y2), (0, 0, 255), 3)

    worksheet_meta.debug_image = InputImageMeta(image_array=debug_arr)

    worksheet_meta.debug_image.save(debug_filepath)
    worksheet_meta.debug_image.image_url = debug_url
    logger.debug("Saved debug image to %s", debug_filepath)

def save_checked(worksheet_meta: WorksheetTemplate) -> None:
    """Save checked image (already contained in WorksheetTemplate) to DEWARPED_PATH with modified filename.

    Args:
        worksheet_meta (WorksheetTemplate): Metadata of the worksheet.
    
    Returns:
        None
    """
    
    # check whether score is already calculated
    if worksheet_meta.score is None or worksheet_meta.marked_answers is None or worksheet_meta.answer_key is None:
        raise ValueError("Score is not calculated yet. Cannot save checked image with marks.")
    
    score = sum(worksheet_meta.score)
    ans_key = worksheet_meta.answer_key
    checked_img = worksheet_meta.checked_image

    original_path = worksheet_meta.input_image.image_path
    checked_filename = f"{Path(original_path).stem}_checked.jpg"
    checked_filepath = Path(SETTINGS.CHECKED_PATH) / checked_filename

    # add marks circle to checked image
    check_circle = make_circle_mark(score, len(ans_key))
    checked_img.paste(check_circle, (100, 50), check_circle)
    checked_img.save(checked_filepath)

    checked_url = f"http://{SERVER_IP}:3000/checked/{checked_filename}"
    worksheet_meta.checked_image_url = checked_url
    # worksheet_meta.checked_image.save(checked_filepath)
    # worksheet_meta.checked_image.image_url = checked_url
    logger.debug("Saved checked image to %s", checked_filepath)


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
    # th2 = bbox2[3] - bbox2[1]

    draw.text(
        ((diameter - tw2) // 2, center_y),
        bottom_text,
        fill=DARK_BLUE,
        font=font
    )

    return img