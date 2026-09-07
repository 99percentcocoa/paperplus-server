"""End-to-end scan pipeline: corner-tag detect -> dewarp -> row-tag detect -> OCR -> bubble marks.

Deliberately does NOT compute correctness/score — that needs the answer key, which lives in
api-service's DB. vision-service is stateless and only returns raw marks + confidences.
"""

from dataclasses import dataclass, field

from app.vision.apriltag_detector import detect_apriltags
from app.vision.bubble_inference import BubbleClassifier, detect_bubble_marks
from app.vision.dewarp import apply_median_blur, crop_image
from app.vision.errors import RollNumberError
from app.vision.ocr import OCRProvider, predict_ocr
from app.vision.tags import decode_row_tag_metadata, validate_question_paper_code
from app.vision.templates import get_handwritten_field_roi, get_roi_coordinates, infer_template_name_from_row_metadata


@dataclass
class QuestionMark:
    question_index: int
    marked_option: str | None
    confidence: float


@dataclass
class ScanResult:
    worksheet_id: int | None
    page_no: int | None
    first_question_index: int | None
    template_name: str
    roll_number: str | None
    roll_number_confidence: float | None
    question_paper_code: str | None
    question_marks: list[QuestionMark] = field(default_factory=list)
    dewarped_image_array: object = None
    debug_image_array: object = None


def _extract_roi(image_array, roi):
    x1, y1, x2, y2 = max(0, roi.x1), max(0, roi.y1), roi.x2, roi.y2
    return image_array[y1:y2, x1:x2]


def process_scan(
    image_array,
    *,
    target_width: int,
    target_height: int,
    bubble_classifier: BubbleClassifier,
    ocr_provider: OCRProvider,
    template_hint: str | None = None,
) -> ScanResult:
    corner_detections = detect_apriltags(image_array, "36h11")
    cropped = crop_image(image_array, corner_detections, target_width, target_height)
    blurred = apply_median_blur(cropped)

    row_detections = detect_apriltags(cropped, "25h9")
    row_metadata = decode_row_tag_metadata(row_detections.tag_ids)

    template_name = infer_template_name_from_row_metadata(row_metadata, template_hint)

    roll_number, roll_number_confidence = _read_roll_number(cropped, template_name, ocr_provider)
    question_paper_code = _read_question_paper_code(cropped, template_name, ocr_provider)

    row_centers = [d.center for d in row_detections.sorted_row_detections]
    question_rois = get_roi_coordinates(row_centers, template_name)

    first_question_index = row_metadata.get("first_question_index") or 1
    question_marks = []
    for offset, roi in enumerate(question_rois):
        roi_image = _extract_roi(blurred, roi)
        marked_option, confidence = detect_bubble_marks(roi_image, bubble_classifier)
        question_marks.append(
            QuestionMark(
                question_index=first_question_index + offset,
                marked_option=marked_option,
                confidence=confidence,
            )
        )

    return ScanResult(
        worksheet_id=row_metadata.get("worksheet_id"),
        page_no=row_metadata.get("page_no"),
        first_question_index=row_metadata.get("first_question_index"),
        template_name=template_name,
        roll_number=roll_number,
        roll_number_confidence=roll_number_confidence,
        question_paper_code=question_paper_code,
        question_marks=question_marks,
        dewarped_image_array=cropped,
        debug_image_array=cropped,
    )


def _read_roll_number(cropped_image_array, template_name: str, ocr_provider: OCRProvider) -> tuple[str, float | None]:
    roi_spec = get_handwritten_field_roi(template_name, "roll_number")
    if roi_spec is None:
        raise RollNumberError(f"No roll number ROI configured for template '{template_name}'.")

    x1, y1, x2, y2 = roi_spec
    roi_image = cropped_image_array[y1:y2, x1:x2]
    if roi_image.size == 0:
        raise RollNumberError(f"Roll number ROI is empty (shape={roi_image.shape}); cannot run OCR.")

    roll_number = predict_ocr(roi_image, ocr_provider)
    if not (isinstance(roll_number, str) and roll_number.isdigit() and len(roll_number) == 4):
        raise RollNumberError(f"Detected roll number '{roll_number}' is not a valid 4-digit number.")

    return roll_number, None  # TODO(Phase 3 follow-up): surface a real OCR confidence once PaddleOCRProvider is wired in.


def _read_question_paper_code(cropped_image_array, template_name: str, ocr_provider: OCRProvider) -> str:
    roi_spec = get_handwritten_field_roi(template_name, "question_paper_code")
    if roi_spec is None:
        return ""

    x1, y1, x2, y2 = roi_spec
    roi_image = cropped_image_array[y1:y2, x1:x2]
    if roi_image.size == 0:
        return ""

    raw_value = predict_ocr(roi_image, ocr_provider)
    return validate_question_paper_code(raw_value)
