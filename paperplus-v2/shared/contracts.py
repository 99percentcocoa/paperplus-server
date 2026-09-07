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

