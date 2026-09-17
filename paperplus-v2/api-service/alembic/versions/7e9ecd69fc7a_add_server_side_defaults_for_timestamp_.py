"""add server-side defaults for timestamp and students columns

Every *_at/last_updated timestamp column across the schema is NOT NULL but only ever gets
populated via SQLModel's Python-side default_factory=utcnow -- there was never a database-level
DEFAULT, so any insert that bypasses the ORM (raw SQL, psql) hits a NOT NULL violation instead
of auto-populating, which is what a "created_at" column should do regardless of how the row got
inserted. Also adds real defaults for students.is_active (true) and students.current_level ('A')
so a raw SQL insert that omits them gets the same sensible defaults the app's own upsert_student()
helper already applies -- by explicit decision, not because the SQLModel class declares them
(Student.current_level's own Python default is None, not "A"; this migration makes "every new
student starts at level A, active" a real product decision enforced at the DB level).

Revision ID: 7e9ecd69fc7a
Revises: d299d5c1ae95
Create Date: 2026-09-17 20:28:24.975625

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision = '7e9ecd69fc7a'
down_revision = 'd299d5c1ae95'
branch_labels = None
depends_on = None


TIMESTAMP_COLUMNS = [
    ("media", "created_at"),
    ("schools", "created_at"),
    ("schools", "updated_at"),
    ("skills", "created_at"),
    ("skills", "updated_at"),
    ("worksheet_templates", "created_at"),
    ("students", "created_at"),
    ("students", "updated_at"),
    ("worksheets", "created_at"),
    ("worksheets", "updated_at"),
    ("mastery_history", "recorded_at"),
    ("omr_answer_sets", "created_at"),
    ("questions", "created_at"),
    ("questions", "updated_at"),
    ("student_guardians", "created_at"),
    ("student_skill_mastery", "last_updated"),
    ("submissions", "submitted_at"),
    ("attempts", "attempted_at"),
    ("processing_events", "occurred_at"),
    ("scan_reviews", "created_at"),
    ("scan_reviews", "updated_at"),
]


def upgrade() -> None:
    for table, column in TIMESTAMP_COLUMNS:
        op.alter_column(table, column, server_default=sa.text("now()"))

    op.alter_column("students", "is_active", server_default=sa.text("true"))
    op.alter_column("students", "current_level", server_default="A")


def downgrade() -> None:
    for table, column in TIMESTAMP_COLUMNS:
        op.alter_column(table, column, server_default=None)

    op.alter_column("students", "is_active", server_default=None)
    op.alter_column("students", "current_level", server_default=None)
