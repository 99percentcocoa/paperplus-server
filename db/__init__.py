"""Public DB package interface.

Re-exports from internal domain modules so callers can do:
    from db import process_submission, validate_sender
"""

# Composite workflows (most common public entry points)
from .flows import (
    add_worksheet_to_db,
    overwrite_submission_flow,
    process_submission,
    validate_sender,
)

# Infrastructure
from .connection import get_connection, get_cursor
from .migrations import run_migrations

# Domain CRUD
from .schools import upsert_school, get_school, list_schools
from .students import upsert_student, get_student, update_student_level, deactivate_student, list_students, normalize_student_id, next_student_id
from .skills import upsert_skill, get_skill, list_skills, import_skills_from_json
from .worksheets import (
    create_worksheet,
    check_test,
    get_worksheet,
    get_worksheet_json,
    get_worksheet_page,
    get_worksheet_pages,
    get_answer_key,
    list_worksheets,
    upsert_worksheet_page,
)
from .questions import insert_questions_for_worksheet, get_questions_for_worksheet, get_question
from .submissions import create_submission, get_submission, get_submissions_for_worksheet, get_latest_submission, get_latest_submission_by_worksheet, overwrite_submission
from .scan_reviews import create_scan_review, get_scan_review, list_scan_reviews, correct_scan_review
from .attempts import insert_attempts, delete_attempts_for_submission, get_attempts_for_submission
from .mastery import update_skill_mastery, get_skill_mastery, recalculate_skill_mastery, get_level_mastery_average, get_level_skill_coverage, determine_level_change, evaluate_and_update_level
from .media import create_media, get_media

__all__ = [
    # Flows
    "add_worksheet_to_db",
    "process_submission",
    "overwrite_submission_flow",
    "validate_sender",
    # Infrastructure
    "get_connection",
    "get_cursor",
    "run_migrations",
    # Schools
    "upsert_school",
    "get_school",
    "list_schools",
    # Students
    "upsert_student",
    "get_student",
    "update_student_level",
    "deactivate_student",
    "list_students",
    "normalize_student_id",
    "next_student_id",
    # Skills
    "upsert_skill",
    "get_skill",
    "list_skills",
    "import_skills_from_json",
    # Worksheets
    "create_worksheet",
    "check_test",
    "get_worksheet",
    "get_worksheet_json",
    "get_worksheet_page",
    "get_worksheet_pages",
    "get_answer_key",
    "list_worksheets",
    "upsert_worksheet_page",
    # Questions
    "insert_questions_for_worksheet",
    "get_questions_for_worksheet",
    "get_question",
    # Submissions
    "create_submission",
    "get_submission",
    "get_submissions_for_worksheet",
    "get_latest_submission",
    "get_latest_submission_by_worksheet",
    "overwrite_submission",
    # Scan reviews
    "create_scan_review",
    "get_scan_review",
    "list_scan_reviews",
    "correct_scan_review",
    # Attempts
    "insert_attempts",
    "delete_attempts_for_submission",
    "get_attempts_for_submission",
    # Mastery
    "update_skill_mastery",
    "get_skill_mastery",
    "recalculate_skill_mastery",
    "get_level_mastery_average",
    "get_level_skill_coverage",
    "determine_level_change",
    "evaluate_and_update_level",
    # Media
    "create_media",
    "get_media",
]
