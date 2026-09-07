from datetime import datetime, timezone

from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class School(SQLModel, table=True):
    __tablename__ = "schools"

    school_code: str = Field(primary_key=True)
    school_name: str
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class Student(SQLModel, table=True):
    __tablename__ = "students"

    # Kept as the natural text PK by decision (e.g. "PSV-2-1"); no surrogate key.
    student_id: str = Field(primary_key=True)
    student_name: str
    student_school_code: str | None = Field(default=None, foreign_key="schools.school_code")
    current_level: str | None = None
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class Skill(SQLModel, table=True):
    __tablename__ = "skills"

    skill_code: str = Field(primary_key=True)
    skill_name: str
    skill_level: str
    skill_weight: float = Field(default=1.0)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
