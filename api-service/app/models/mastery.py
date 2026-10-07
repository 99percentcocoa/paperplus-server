from datetime import datetime

from sqlmodel import Field, SQLModel

from app.models.core import utcnow


class StudentSkillMastery(SQLModel, table=True):
    """Current-value snapshot per student/skill (kept alongside MasteryHistory for fast lookups)."""

    __tablename__ = "student_skill_mastery"

    student_id: str = Field(foreign_key="students.student_id", primary_key=True)
    skill_code: str = Field(foreign_key="skills.skill_code", primary_key=True)
    mastery_score: float | None = None
    last_updated: datetime = Field(default_factory=utcnow)


class MasteryHistory(SQLModel, table=True):
    """Time-series record of mastery_score changes for trend analysis."""

    __tablename__ = "mastery_history"

    id: int | None = Field(default=None, primary_key=True)
    student_id: str = Field(foreign_key="students.student_id", index=True)
    skill_code: str = Field(foreign_key="skills.skill_code", index=True)
    mastery_score: float | None = None
    recorded_at: datetime = Field(default_factory=utcnow)
