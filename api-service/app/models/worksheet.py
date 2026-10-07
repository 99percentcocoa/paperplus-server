from datetime import datetime
from enum import Enum

from sqlalchemy import CheckConstraint, Column, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel

from app.models.core import utcnow


class WorksheetCategory(str, Enum):
    PRACTICE = "practice"
    HOMEWORK = "homework"
    TEST = "test"
    OMR = "omr"


class WorksheetTemplate(SQLModel, table=True):
    """Registry row per code-defined template (definition stays in code; this is just for lookup/audit)."""

    __tablename__ = "worksheet_templates"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)
    description: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class Worksheet(SQLModel, table=True):
    __tablename__ = "worksheets"
    __table_args__ = (
        CheckConstraint(
            "worksheet_category in ('practice','homework','test','omr')",
            name="ck_worksheets_category",
        ),
    )

    worksheet_id: int | None = Field(default=None, primary_key=True)
    worksheet_level: str | None = None
    max_score: int | None = None
    lang: str | None = None
    title: str | None = None
    worksheet_category: str = Field(default=WorksheetCategory.PRACTICE.value)
    sheet_version: str = Field(default="legacy")
    page_count: int = Field(default=1)
    total_question_count: int | None = None
    worksheet_metadata: dict = Field(default_factory=dict, sa_column=Column(JSONB, nullable=False))
    template_id: int | None = Field(default=None, foreign_key="worksheet_templates.id")
    worksheet_json: dict | None = Field(default=None, sa_column=Column(JSONB))
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class WorksheetPage(SQLModel, table=True):
    __tablename__ = "worksheet_pages"
    __table_args__ = (UniqueConstraint("worksheet_id", "page_no"),)

    worksheet_page_id: int | None = Field(default=None, primary_key=True)
    worksheet_id: int = Field(foreign_key="worksheets.worksheet_id")
    page_no: int
    first_question_index: int
    last_question_index: int
    expected_row_tag_count: int = Field(default=10)
    page_metadata: dict = Field(default_factory=dict, sa_column=Column(JSONB, nullable=False))


class Question(SQLModel, table=True):
    __tablename__ = "questions"

    question_id: int | None = Field(default=None, primary_key=True)
    worksheet_id: int | None = Field(default=None, foreign_key="worksheets.worksheet_id", index=True)
    skill_code: str = Field(foreign_key="skills.skill_code")
    question_json: dict | None = Field(default=None, sa_column=Column(JSONB))
    index: int | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class QuestionOption(SQLModel, table=True):
    """One row per bubble option; replaces parsing answers out of question_json/worksheet_json."""

    __tablename__ = "question_options"

    id: int | None = Field(default=None, primary_key=True)
    question_id: int = Field(foreign_key="questions.question_id", index=True)
    option_label: str
    option_value: str
    is_correct: bool = Field(default=False)


class QuestionPaperVariant(SQLModel, table=True):
    """Per-variant correct answer for a question (basic_omr prints fixed A-F answer keys,
    not per-copy shuffled bubbles — every printed copy of a variant shares one answer key).
    """

    __tablename__ = "question_paper_variants"
    __table_args__ = (
        CheckConstraint("question_paper_code ~ '^[A-F]$'", name="ck_qpv_code_valid"),
        UniqueConstraint("worksheet_id", "question_paper_code", "question_id"),
    )

    id: int | None = Field(default=None, primary_key=True)
    worksheet_id: int = Field(foreign_key="worksheets.worksheet_id", index=True)
    question_paper_code: str
    question_id: int = Field(foreign_key="questions.question_id", index=True)
    correct_option_label: str


class OMRAnswerSet(SQLModel, table=True):
    """Legacy/bridge table carried over as-is from the current system for migration compatibility."""

    __tablename__ = "omr_answer_sets"
    __table_args__ = (
        UniqueConstraint("template_name", "question_paper_code"),
        CheckConstraint("question_paper_code ~ '^[A-F]$'", name="ck_omr_answer_sets_code_valid"),
    )

    id: int | None = Field(default=None, primary_key=True)
    template_name: str = Field(index=True)
    question_paper_code: str
    worksheet_id: int | None = Field(default=None, foreign_key="worksheets.worksheet_id", index=True)
    answer_key_json: dict = Field(sa_column=Column(JSONB, nullable=False))
    created_at: datetime = Field(default_factory=utcnow)
