"""Perspective dewarp + blur, ported from services/image_service.py crop_image/apply_median_blur."""

from typing import Any

from app.vision.apriltag_detector import DetectionResult, detect_orientation_and_decode
from app.vision.errors import CornerTagDetectionError

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover
    cv2 = None
    np = None


def crop_image(image_array, corner_detections: DetectionResult, target_width: int, target_height: int):
    """Perspective-warp the raw scan to a canonical (target_width x target_height) image."""
    if cv2 is None:
        raise RuntimeError("opencv-python is not installed; cannot dewarp image.")
    if image_array is None:
        raise ValueError("image_array is None; cannot crop.")

    oriented = detect_orientation_and_decode(corner_detections.sorted_corner_detections)
    if oriented is None:
        raise CornerTagDetectionError("Could not find the orientation (top-left) corner tag.")

    src_pts = np.array([oriented[i].center for i in range(4)], dtype=np.float32)
    dst_pts = np.array(
        [[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]],
        dtype=np.float32,
    )

    t_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image_array, t_matrix, (target_width, target_height))
    return warped


def apply_median_blur(image_array, kernel_size: int = 31):
    if cv2 is None:
        raise RuntimeError("opencv-python is not installed; cannot blur image.")
    return cv2.medianBlur(image_array, kernel_size)
