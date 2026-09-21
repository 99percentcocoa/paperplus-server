from datetime import datetime
from enum import Enum

from sqlalchemy import CheckConstraint, Column, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel

from app.models.core import utcnow


class ProcessingState(str, Enum):
    UPLOADED = "uploaded"
    PREPROCESSING = "preprocessing"
    DEWARPED = "dewarped"
    REGISTERING = "registering"
    SCORING = "scoring"
    GRADED = "graded"
    FAILED = "failed"


class ScanReviewStatus(str, Enum):
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"
    CORRECTED = "corrected"
    APPROVED = "approved"


class Submission(SQLModel, table=True):
    __tablename__ = "submissions"
    __table_args__ = (
        CheckConstraint(
            "worksheet_category in ('practice','homework','test','omr')",
            name="ck_submissions_category",
        ),
    )

    submission_id: int | None = Field(default=None, primary_key=True)
    student_id: str = Field(foreign_key="students.student_id", index=True)
    worksheet_id: int = Field(foreign_key="worksheets.worksheet_id", index=True)
    worksheet_category: str = Field(default="practice")
    score: int | None = None
    from_number: str | None = None
    answers_json: dict | None = Field(default=None, sa_column=Column(JSONB))
    state: str = Field(default=ProcessingState.UPLOADED.value)
    ocr_confidence: float | None = None
    processing_started_at: datetime | None = None
    processing_completed_at: datetime | None = None
    submitted_at: datetime = Field(default_factory=utcnow)
    checked_image_path: str | None = None
    checked_image_url: str | None = None


class Attempt(SQLModel, table=True):
    __tablename__ = "attempts"

    attempt_id: int | None = Field(default=None, primary_key=True)
    student_id: str = Field(foreign_key="students.student_id", index=True)
    submission_id: int = Field(foreign_key="submissions.submission_id", index=True)
    question_id: int = Field(foreign_key="questions.question_id", index=True)
    worksheet_id: int = Field(foreign_key="worksheets.worksheet_id", index=True)
    is_correct: bool | None = None
    skill_code: str = Field(foreign_key="skills.skill_code")
    bubble_confidence: float | None = None
    attempted_at: datetime = Field(default_factory=utcnow)


class ScanOutcome(str, Enum):
    GRADED = "graded"
    FAILED = "failed"


class Scan(SQLModel, table=True):
    """One row per scanned page image, success or failure. Keeps the file locations and the full
    vision-service result (question_marks with ROI boxes, page_no, question_paper_code) that
    used to be logged and discarded, so the admin dashboard can show the scan and correct or
    re-grade it later without re-running vision.
    """

    __tablename__ = "scans"

    id: int | None = Field(default=None, primary_key=True)
    correlation_id: str = Field(index=True)
    from_number: str | None = None
    upload_path: str | None = None
    dewarped_path: str | None = None
    checked_image_path: str | None = None
    # Deliberately no FK: an unrecognized worksheet id is one of the failures this row records.
    worksheet_id: int | None = Field(default=None, index=True)
    page_no: int | None = None
    template_name: str | None = None
    roll_number: str | None = None
    question_paper_code: str | None = None
    vision_result: dict | None = Field(default=None, sa_column=Column(JSONB))
    outcome: str = Field(default=ScanOutcome.FAILED.value)
    submission_id: int | None = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("submissions.submission_id", ondelete="SET NULL"), index=True),
    )
    created_at: datetime = Field(default_factory=utcnow)


class ScanReview(SQLModel, table=True):
    __tablename__ = "scan_reviews"
    __table_args__ = (
        CheckConstraint(
            "status in ('failed','needs_review','corrected','approved')",
            name="ck_scan_reviews_status",
        ),
    )

    review_id: int | None = Field(default=None, primary_key=True)
    submission_id: int | None = Field(default=None, foreign_key="submissions.submission_id", index=True)
    student_id: str | None = None
    worksheet_id: int | None = Field(default=None, foreign_key="worksheets.worksheet_id", index=True)
    detected_roll_number: str | None = None
    correlation_id: str | None = Field(default=None, index=True)
    scan_id: int | None = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("scans.id", ondelete="SET NULL"), index=True),
    )
    status: str = Field(default=ScanReviewStatus.FAILED.value)
    error_reason: str | None = None
    original_answers: dict | None = Field(default=None, sa_column=Column(JSONB))
    corrected_answers: dict | None = Field(default=None, sa_column=Column(JSONB))
    original_score: int | None = None
    corrected_score: int | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    corrected_by: str | None = None
    corrected_at: datetime | None = None


class ProcessingEvent(SQLModel, table=True):
    """Durable audit trail for the state machine, replaces per-session log files."""

    __tablename__ = "processing_events"

    id: int | None = Field(default=None, primary_key=True)
    submission_id: int = Field(foreign_key="submissions.submission_id", index=True)
    state: str
    service_name: str
    correlation_id: str | None = Field(default=None, index=True)
    detail: dict = Field(default_factory=dict, sa_column=Column(JSONB, nullable=False))
    occurred_at: datetime = Field(default_factory=utcnow)
