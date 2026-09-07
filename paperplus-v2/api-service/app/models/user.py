from datetime import datetime

from sqlmodel import Field, SQLModel

from app.models.core import utcnow


class User(SQLModel, table=True):
    __tablename__ = "users"

    user_id: int | None = Field(default=None, primary_key=True)
    user_name: str
    from_number: str = Field(unique=True, index=True)


class StudentGuardian(SQLModel, table=True):
    """Links a WhatsApp sender to a student; supports multiple children per phone number."""

    __tablename__ = "student_guardians"

    id: int | None = Field(default=None, primary_key=True)
    student_id: str = Field(foreign_key="students.student_id", index=True)
    from_number: str = Field(index=True)
    created_at: datetime = Field(default_factory=utcnow)
