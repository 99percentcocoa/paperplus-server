from datetime import datetime, timezone

from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


DEFAULT_PROJECT_CODE = "paperplus"


class Project(SQLModel, table=True):
    """A program with its own students/schools and its own dashboard at /admin/{project_code}/
    (e.g. PaperPlus homework vs. Navodaya OMR tests). Student IDs stay globally unique across
    projects, so an incoming scan is routed to a project through its student."""

    __tablename__ = "projects"

    project_code: str = Field(primary_key=True)
    project_name: str
    created_at: datetime = Field(default_factory=utcnow)


class School(SQLModel, table=True):
    __tablename__ = "schools"

    school_code: str = Field(primary_key=True)
    school_name: str
    project_code: str = Field(default=DEFAULT_PROJECT_CODE, foreign_key="projects.project_code", index=True)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class Student(SQLModel, table=True):
    __tablename__ = "students"

    # Kept as the natural text PK by decision (e.g. "PSV-2-1"); no surrogate key.
    student_id: str = Field(primary_key=True)
    student_name: str
    student_school_code: str | None = Field(default=None, foreign_key="schools.school_code")
    # Carried on the student itself (not only via school): scans are routed by student, and a
    # student needn't have a school.
    project_code: str = Field(default=DEFAULT_PROJECT_CODE, foreign_key="projects.project_code", index=True)
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
