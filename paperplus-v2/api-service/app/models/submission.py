from datetime import datetime
from enum import Enum

from sqlalchemy import CheckConstraint, Column
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
