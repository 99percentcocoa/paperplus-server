"""Pydantic contracts shared between api-service and vision-service's /process endpoint.

vision-service infers the worksheet template itself from the scanned row tags (it has no DB
access), so callers only pass an optional `template_hint` override rather than a full spec.

vision-service is fully stateless: the photo goes in as base64 bytes and the dewarped page comes
back as base64 bytes, so the two services never need to share a filesystem.
"""

from pydantic import BaseModel


class ProcessRequest(BaseModel):
    correlation_id: str
    image_b64: str  # the uploaded photo (JPEG/PNG bytes), base64-encoded
    template_hint: str | None = None
    # Admin-assisted retry: skip corner-tag (36h11) detection and perspective dewarp, and just
    # resize the photo to the canonical page size before row-tag decoding and everything after.
    skip_corner_tags: bool = False
    # Handwritten fields entered by an admin who can see the photo: when given, OCR for that field
    # is skipped and the value is used as-is.
    roll_number: str | None = None
    question_paper_code: str | None = None


class QuestionMark(BaseModel):
    question_index: int
    marked_option: str | None
    confidence: float
    # Pixel ROI box of this question on the dewarped image (see dewarped_image_b64 below),
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
    # Wire field: the dewarped page as a base64 JPEG, set by vision-service. The ROI boxes above are
    # in this image's coordinates, which is why api-service needs it (to draw the checked image).
    dewarped_image_b64: str | None = None
    # Local field, never sent by vision-service: api-service's HTTPVisionClient decodes
    # dewarped_image_b64 into its own storage, sets this path, and clears the base64 so the
    # blob is never persisted in scans.vision_result.
    dewarped_image_path: str | None = None

