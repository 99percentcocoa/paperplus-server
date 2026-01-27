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
import cv2  # pylint: disable=no-member
import numpy as np
from tinydb import TinyDB
from pupil_apriltags import Detector
from config import SETTINGS
from models import DetectionResult, InputImageMeta, WorksheetTemplate

logger = logging.getLogger(__name__)

SAVE_DIR = SETTINGS.DOWNLOADS_PATH
DEWARPED_DIR = SETTINGS.DEWARPED_PATH
TARGET_WIDTH = SETTINGS.TARGET_WIDTH
TARGET_HEIGHT = SETTINGS.TARGET_HEIGHT

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
BASE = 586
ORIENTATION_ID = 586
db = TinyDB('worksheets.json')

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

    detections = detector.detect(gray_image_array)

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


def scan_image(input_image: InputImageMeta):
    """Process image: dewarp, clean, and prepare for OMR.

    Args:
        input_image (InputImageMeta): Metadata of the original image

    Returns:
        cropped_image: (InputImageMeta) Metadata of the cropped image.
        preprocessed_image: (InputImageMeta) Metadata of the preprocessed image.
        corner_detections: (DetectionResult) Result of AprilTag detections.
    """

    corner_detections = detect_apriltags(input_image, "36h11")
    cropped_image, worksheet_id = crop_image(input_image, corner_detections)
    row_detections = detect_apriltags(cropped_image, "25h9")
    preprocessed_image = clean_document(cropped_image)

    return cropped_image, preprocessed_image, corner_detections, row_detections, worksheet_id


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

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detection = at_detector_36h11.detect(gray_img)
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


# Image Processing Functions
# def dewarp_omr(filepath, detection):
#     """Dewarp OMR image using AprilTag detections.

#     Args:
#         filepath (str): Path to the image file
#         detection (list): List of AprilTag detections in clockwise order

#     Returns:
#         np.ndarray: Dewarped image
#     """
#     image = cv2.imread(filepath)

#     # detections are already sorted in tl, tr, br, bl
#     corner_tags = [d.center for d in detection]
#     logger.debug("Pre arranging: %s", corner_tags)

#     tl = corner_tags[0]
#     tr = corner_tags[1]
#     br = corner_tags[2]
#     bl = corner_tags[3]
#     logger.debug("Final tags in order: %s, %s, %s, %s.", tl, tr, br, bl)

#     # Re-order the final source points: TL, TR, BR, BL
#     # This is the essential input for cv2.getPerspectiveTransform
#     src_pts_aligned = np.array([tl, tr, br, bl], dtype="float32")

#     dst_pts = np.array([
#         [0, 0],
#         [TARGET_WIDTH - 1, 0],
#         [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
#         [0, TARGET_HEIGHT - 1]], dtype="float32")

#     # Calculate the global perspective transform matrix (M)
#     t_matrix = cv2.getPerspectiveTransform(src_pts_aligned, dst_pts)

#     # Apply the dewarping
#     dewarped = cv2.warpPerspective(image, t_matrix, (TARGET_WIDTH, TARGET_HEIGHT))
#     return dewarped

# crop image using corner tags
def crop_image(input_image: InputImageMeta, detections: DetectionResult) -> InputImageMeta:
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

    # if len(detections.detections) < 4:
    #     raise ValueError("At least 4 AprilTags are required to crop the image.")

    # # Use sorted detections: top-left, top-right, bottom-right, bottom-left
    # ordered = detections.sorted_detections
    # if len(ordered) < 4:
    #     raise ValueError("Sorted detections did not yield 4 corners.")
    
    worksheet_id, detections.sorted_detections = detect_orientation_and_decode(detections)

    # validation and orientation

    # Build source points (x, y) in float32 shape (4,2)
    src_pts = np.array([detections.sorted_detections[i].center for i in range(4)], dtype=np.float32)
    dst_pts = np.array([[0, 0], [TARGET_WIDTH, 0], [TARGET_WIDTH, TARGET_HEIGHT], [0, TARGET_HEIGHT]], dtype="float32")

    # Compute perspective transform matrix
    t_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Perform the warp perspective to get the cropped image
    warped_image = cv2.warpPerspective(input_image.image_array, t_matrix, (TARGET_WIDTH, TARGET_HEIGHT))

    return InputImageMeta(image_array=warped_image), worksheet_id


# def clean_document(img):
#     """Clean and preprocess document image for OMR.

#     Args:
#         img (np.ndarray): Input image

#     Returns:
#         np.ndarray: Cleaned image
#     """
#     if img is None:
#         raise ValueError("Cannot load image.")

#     # 2. Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # 3. Remove noise (median filter works well for text)
#     denoised = cv2.medianBlur(gray, 3)

#     # 4. Shadow / illumination correction
#     #    We estimate the background by heavy blur
#     background = cv2.GaussianBlur(denoised, (99, 99), 0)

#     # Avoid divide-by-zero
#     background = background.astype(np.float32)
#     denoised = denoised.astype(np.float32)

#     # Normalize lighting
#     corrected = (denoised / (background + 1)) * 255
#     corrected = np.clip(corrected, 0, 255).astype(np.uint8)

#     # 5. Sharpen slightly (helps with blur)
#     kernel = np.array([
#         [0, -1, 0],
#         [-1,  5, -1],
#         [0, -1, 0]
#     ])
#     sharp = cv2.filter2D(corrected, -1, kernel)

#     # 6. Adaptive thresholding
#     #    Sauvola style (OpenCV uses a similar method)
#     binary = cv2.adaptiveThreshold(
#         sharp,
#         255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         101,   # block size
#         10    # constant subtracted
#     )

#     # 7. Small speckle removal
#     #    Remove small white or black dots
#     clean = cv2.medianBlur(binary, 3)
#     color_img = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)

#     return color_img

def clean_document(input_image: InputImageMeta) -> InputImageMeta:
    """Clean and preprocess document image for OMR.

    Args:
        input_image (InputImageMeta): Metadata of the input image.

    Returns:
        InputImageMeta: Metadata of the cleaned image.
    """
    if input_image.image_array is None:
        raise ValueError("Cannot load image.")

    img = input_image.image_array

    # 2. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Remove noise (median filter works well for text)
    denoised = cv2.medianBlur(gray, 3)

    # 4. Shadow / illumination correction
    #    We estimate the background by heavy blur
    background = cv2.GaussianBlur(denoised, (99, 99), 0)

    # Avoid divide-by-zero
    background = background.astype(np.float32)
    denoised = denoised.astype(np.float32)

    # Normalize lighting
    corrected = (denoised / (background + 1)) * 255
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    # 5. Sharpen slightly (helps with blur)
    kernel = np.array([
        [0, -1, 0],
        [-1,  5, -1],
        [0, -1, 0]
    ])
    sharp = cv2.filter2D(corrected, -1, kernel)

    # 6. Adaptive thresholding
    #    Sauvola style (OpenCV uses a similar method)
    binary = cv2.adaptiveThreshold(
        sharp,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        101,   # block size
        10    # constant subtracted
    )

    # 7. Small speckle removal
    #    Remove small white or black dots
    clean = cv2.medianBlur(binary, 3)
    color_img = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)

    return InputImageMeta(image_array=color_img)

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


def encode_worksheet_id(n: int):
    """Return tag IDs for TR, BR, BL given worksheet_id n.

    Args:
        n (int): Worksheet ID

    Returns:
        list: [TR, BR, BL] tag IDs

    Raises:
        ValueError: If worksheet_id is too large
    """
    if n >= BASE ** 3:
        raise ValueError(f"Max worksheet_id is {BASE**3 - 1}")
    ids = []
    for _ in range(3):
        ids.append(n % BASE)
        n //= BASE
    return ids  # [TR, BR, BL]


def decode_from_tags(tr: int, br: int, bl: int):
    """Return worksheet_id from three tag IDs.

    Args:
        tr (int): Top-right tag ID
        br (int): Bottom-right tag ID
        bl (int): Bottom-left tag ID

    Returns:
        int: Worksheet ID
    """
    return tr + br * BASE + bl * (BASE ** 2)


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
        # rot starts with 0
        rotated = rotate(detection.sorted_detections, rot)
        tag_ids = [d.tag_id for d in rotated]
        num_rotations += 1
        print(f"At rotation {num_rotations}")
        if tag_ids[0] == ORIENTATION_ID:        # TL found

            worksheet_id = decode_from_tags(tag_ids[1], tag_ids[2], tag_ids[3])
            print(f"Scanned worksheet ID: {worksheet_id}")
            # return (worksheet_id, rotated)

            # check if worksheet id is in database
            if db.contains(doc_id=worksheet_id):
                print(
                    f"Found worksheet id {worksheet_id}: "
                    f"{db.get(doc_id=worksheet_id).get('name', '')}"
                )
                detection.sorted_detections = rotated
                return (worksheet_id, rotated)
            else:
                print(f"Worksheet ID {worksheet_id} not found in database.")
                return None
    return None  # some error

def save_preprocessed(preprocessed_image: InputImageMeta):
    """Save preprocessed image to DEWARPED_PATH with modified filename.

    Args:
        preprocessed_image (InputImageMeta): Metadata of the preprocessed image.
    
    Returns:
        InputImageMeta: Updated metadata with new file path.
    """
    preprocessed_filename = f"{Path(preprocessed_image.image_path).stem}_preprocessed.jpg"
    preprocessed_filepath = Path(DEWARPED_DIR) / preprocessed_filename
    preprocessed_image.image_path = str(preprocessed_filepath)
    preprocessed_image.save()
    logger.debug("Saved preprocessed image to %s", preprocessed_filepath)

    return preprocessed_image