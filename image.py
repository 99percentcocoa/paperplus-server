# pylint: disable=no-member
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

TARGET_WIDTH = 1240
TARGET_HEIGHT = 1754

# dewarping using apriltags
# inputs:
#   1. filepath
#   2. detection (pre-arranged in clockwise order)
#   3. tag_ids (the correct order of tag_ids: tl, tr, br, bl)
def dewarp_omr(filepath, detection):
    image = cv2.imread(filepath)
    
    # detections are already sorted in tl, tr, br, bl
    corner_tags = [d.center for d in detection]
    logger.debug("Pre arranging: %s", corner_tags)

    tl = corner_tags[0]
    tr = corner_tags[1]
    br = corner_tags[2]
    bl = corner_tags[3]
    logger.debug("Final tags in order: %s, %s, %s, %s.", tl, tr, br, bl)
    
    # Re-order the final source points: TL, TR, BR, BL
    # This is the essential input for cv2.getPerspectiveTransform
    src_pts_aligned = np.array([tl, tr, br, bl], dtype="float32")

    dst_pts = np.array([
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1]], dtype="float32")

    # Calculate the global perspective transform matrix (M)
    t_matrix = cv2.getPerspectiveTransform(src_pts_aligned, dst_pts)

    # Apply the dewarping
    dewarped = cv2.warpPerspective(image, t_matrix, (TARGET_WIDTH, TARGET_HEIGHT))
    return dewarped

# split image into halves
# def split_img(image):
#     # image = cv2.imread(filepath)
#     height, width = image.shape[:2]

#     mid_x = width // 2

#     left_half = image[:, :mid_x]
#     right_half = image[:, mid_x:]

#     return [left_half, right_half]

    # cv2.imwrite(f"{filepath}_left.jpg", left_half)
    # cv2.imwrite(f"{filepath}_right.jpg", right_half)

# new preprocessing pipeline
def clean_document(img):
    if img is None:
        raise ValueError("Cannot load image.")

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

    return color_img

# preprocessing pipeline designed for a scanned sheet
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35,35))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    diff = cv2.absdiff(gray, background)
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    norm_inv = cv2.bitwise_not(norm)

    # convert back to BGR
    color_img = cv2.cvtColor(norm_inv, cv2.COLOR_GRAY2BGR)

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # enhanced = clahe.apply(norm_inv)
    # cv2.imwrite('clean_enhanced.jpg', enhanced)

    # median = cv2.medianBlur(enhanced, 3)
    # cv2.imwrite('clean_median.jpg', median)
    # bilateral = cv2.bilateralFilter(enhanced, 9, 75, 75)
    # cv2.imwrite('clean_bilateral.jpg', bilateral)

    # cv2.imwrite("clean_preprocessed.jpg", color_img)
    return color_img