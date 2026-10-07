from datetime import datetime
from enum import Enum

from sqlalchemy import CheckConstraint
from sqlmodel import Field, SQLModel

from app.models.core import utcnow


class MediaRole(str, Enum):
    UPLOADED = "uploaded"
    DEWARPED = "dewarped"
    DEBUG = "debug"
    CHECKED = "checked"


class Media(SQLModel, table=True):
    """One row per stored artifact; replaces the implicit *_PATH folder convention."""

    __tablename__ = "media"
    __table_args__ = (
        CheckConstraint(
            "owner_type in ('submission','worksheet')",
            name="ck_media_owner_type",
        ),
        CheckConstraint(
            "role in ('uploaded','dewarped','debug','checked')",
            name="ck_media_role",
        ),
    )

    media_id: int | None = Field(default=None, primary_key=True)
    owner_type: str
    owner_id: int
    role: str
    media_type: str = Field(default="image")
    storage_path: str
    created_at: datetime = Field(default_factory=utcnow)
