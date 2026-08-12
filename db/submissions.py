"""Submission CRUD operations."""

import json

from .connection import get_connection


def create_submission(student_id: str, worksheet_id: int,
                      score: int, from_number: str, answers_json: dict | list,
                      worksheet_category: str = None) -> int:
    """Record a graded submission. Returns submission_id."""
    if worksheet_category is None:
        worksheet_category = "practice"
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO submissions
                       (student_id, worksheet_id, score, from_number, answers_json, worksheet_category)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   RETURNING submission_id""",
                (student_id, worksheet_id, score, from_number,
                 json.dumps(answers_json), worksheet_category),
            )
            return cur.fetchone()["submission_id"]


def get_submission(submission_id: int) -> dict | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM submissions WHERE submission_id = %s", (submission_id,))
            return cur.fetchone()


def get_submissions_for_worksheet(worksheet_id: int, student_id: str = None,
                                 worksheet_category: str = None) -> list[dict]:
    """Find submissions for a worksheet, optionally filtered by student or category."""
    clauses = ["worksheet_id = %s"]
    params = [worksheet_id]
    if student_id:
        clauses.append("student_id = %s")
        params.append(student_id)
    if worksheet_category:
        clauses.append("worksheet_category = %s")
        params.append(worksheet_category)
    where = "WHERE " + " AND ".join(clauses)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM submissions {where} ORDER BY submitted_at DESC", params
            )
            return cur.fetchall()


def get_latest_submission(student_id: str, worksheet_id: int) -> dict | None:
    """Return the most recent submission for a student + worksheet."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT * FROM submissions
                   WHERE student_id = %s AND worksheet_id = %s
                   ORDER BY submitted_at DESC LIMIT 1""",
                (student_id, worksheet_id),
            )
            return cur.fetchone()


def get_latest_submission_by_worksheet(worksheet_id: int) -> dict | None:
    """Return the most recent submission for a worksheet (any student)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT * FROM submissions
                   WHERE worksheet_id = %s
                   ORDER BY submitted_at DESC LIMIT 1""",
                (worksheet_id,),
            )
            return cur.fetchone()


def overwrite_submission(submission_id: int, score: int, answers_json: dict | list):
    """Overwrite an existing submission's score and answers."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE submissions
                   SET score = %s, answers_json = %s, submitted_at = now()
                   WHERE submission_id = %s""",
                (score, json.dumps(answers_json), submission_id),
            )
