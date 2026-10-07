"""AprilTag detection — corner (36h11) and row (25h9) tags.

Real cv2/pupil_apriltags usage, guarded imports so pure-logic modules/tests don't need
these heavy deps installed.
"""

from dataclasses import dataclass, field
from typing import Any

from app.vision.errors import CornerTagDetectionError, RowTagDetectionError
from app.vision.geometry import sort_detections_clockwise
from app.vision.tags import ORIENTATION_ID, rotate
from app.vision.templates import get_template_row_tag_count

try:
    import cv2
    import numpy as np
    from pupil_apriltags import Detector
except ImportError:  # pragma: no cover - exercised only when deps are missing
    cv2 = None
    np = None
    Detector = None

FOV_DEGREES = 60  # typical smartphone camera field-of-view, used for AprilTag pose estimation
TAG_SIZE_METERS = 0.01

_DETECTOR_KWARGS = dict(
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.2,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0,
)

_detector_36h11 = None
_detector_25h9 = None


def _get_detector(tag_family: str):
    global _detector_36h11, _detector_25h9
    if Detector is None:
        raise RuntimeError("pupil_apriltags is not installed; cannot detect AprilTags.")
    if tag_family == "36h11":
        if _detector_36h11 is None:
            _detector_36h11 = Detector(families="tag36h11", **_DETECTOR_KWARGS)
        return _detector_36h11
    if tag_family == "25h9":
        if _detector_25h9 is None:
            _detector_25h9 = Detector(families="tag25h9", **_DETECTOR_KWARGS)
        return _detector_25h9
    raise ValueError(f"Unsupported tag family: {tag_family}")


@dataclass
class DetectionResult:
    """Results from AprilTag detection on an image; validates + sorts detections like the old models.py."""

    detections: list[Any]
    tag_family: str
    tag_ids: list[int] = field(init=False)
    sorted_corner_detections: list[Any] = field(default_factory=list, init=False)
    sorted_row_detections: list[Any] = field(default_factory=list, init=False)

    def __post_init__(self):
        num_detections = len(self.detections)

        if self.tag_family == "36h11":
            if num_detections < 4:
                raise CornerTagDetectionError(
                    f"Tag family '36h11' requires at least 4 detections, but got {num_detections}"
                )
            self.sorted_corner_detections = sort_detections_clockwise(self.detections)

        elif self.tag_family == "25h9":
            required_detections = get_template_row_tag_count("regular")
            if num_detections < required_detections:
                raise RowTagDetectionError(
                    f"Tag family '25h9' requires at least {required_detections} detections, but got {num_detections}"
                )
            valid_detections = [d for d in self.detections if 0 <= d.tag_id <= 34]
            self.detections = valid_detections
            self.sorted_row_detections = sorted(self.detections, key=lambda d: d.center[1])
            self.detections = self.sorted_row_detections

        self.tag_ids = [d.tag_id for d in self.detections]


def detect_apriltags(image_array, tag_family: str) -> DetectionResult:
    """Detect AprilTags in a BGR numpy image array (as produced by cv2.imread/cv2.imdecode)."""
    if cv2 is None:
        raise RuntimeError("opencv-python is not installed; cannot detect AprilTags.")
    if image_array is None:
        raise ValueError("image_array is None; cannot detect AprilTags.")

    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    detector = _get_detector(tag_family)

    h, w = image_array.shape[:2]
    focal_length = (w / 2) / np.tan(np.radians(FOV_DEGREES / 2))
    cx, cy = w / 2, h / 2

    detections = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=[focal_length, focal_length, cx, cy],
        tag_size=TAG_SIZE_METERS,
    )

    return DetectionResult(detections=detections, tag_family=tag_family)


def detect_orientation_and_decode(sorted_corner_detections: list[Any]) -> list[Any] | None:
    """Rotate the 4 sorted corner detections until the top-left tag has ORIENTATION_ID."""
    for rot in range(4):
        rotated = rotate(sorted_corner_detections, rot)
        tag_ids = [d.tag_id for d in rotated]
        if tag_ids[0] == ORIENTATION_ID:
            return rotated
    return None
