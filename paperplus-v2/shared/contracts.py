"""Pydantic contracts shared between api-service and vision-service's /process endpoint.

vision-service infers the worksheet template itself from the scanned row tags (it has no DB
access), so callers only pass an optional `template_hint` override rather than a full spec.
"""

from pydantic import BaseModel


class ProcessRequest(BaseModel):
    correlation_id: str
    image_path: str
    template_hint: str | None = None


class QuestionMark(BaseModel):
    question_index: int
    marked_option: str | None
    confidence: float
    # Pixel ROI box of this question on the dewarped image (see dewarped_image_path below),
    # so api-service can draw the checked-image (correct/incorrect) annotation after grading
    # without re-deriving AprilTag/template geometry itself. Default 0 keeps this optional for
    # fixtures/tests that only exercise grading and don't care about pixel geometry.
    roi_x1: int = 0
    roi_y1: int = 0
    roi_x2: int = 0
    roi_y2: int = 0


class ProcessingResult(BaseModel):
    worksheet_id: int | None
    page_no: int | None
    first_question_index: int | None
    template_name: str
    roll_number: str | None
    roll_number_confidence: float | None
    question_paper_code: str | None
    question_marks: list[QuestionMark]
    dewarped_image_path: str | None = None
    debug_image_path: str | None = None

